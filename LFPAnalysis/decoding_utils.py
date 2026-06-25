"""Cross-validated decoding for LFP feature matrices.

Companion to :mod:`representational_utils`. Takes a per-trial feature matrix
``X`` ``(n_trials, n_features)`` (typically per-channel band power / HFA for one
region) and asks whether a task variable ``y`` can be decoded from the
*multivariate pattern* — the test that channel-collapsed univariate
regression (already content-null in this cohort) cannot rule out.

* Continuous ``y`` (affiliation, power, egocentric distance/angle): ridge
  regression, scored by cross-validated Pearson correlation between
  out-of-fold predictions and truth.
* Categorical ``y`` (character identity, choice, approach/avoid): LDA or
  logistic regression, scored by balanced accuracy (chance = 1/n_classes).

Significance is by **label permutation** within subject; group inference is a
signed-rank/t-test of per-subject scores against the chance level, with a
Benjamini–Hochberg helper for the pre-registered comparison family.

All randomness goes through an explicit ``rng``/``random_state`` — never the
implicit global RNG.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "infer_task",
    "residualize_columns",
    "decode_cv",
    "decode_with_permutation",
    "decode_crossgen",
    "decode_crossgen_grouped",
    "benjamini_hochberg",
    "group_test",
]


def residualize_columns(X, confound):
    """Linearly residualize each feature column on a confound (session-time control).

    Removes the linear trend of ``confound`` (e.g. ``decision_num``) from every column of
    ``X`` before decoding — the feature-side analogue of the SU pipeline's
    ``_residualize_fr_on_trial_num`` (firing rate residualized on ``trial_num_z``), and the
    decoder counterpart of the RSA drift-partial. Global (not within-fold), matching SU.

    Returns a residualized copy of ``X`` (unchanged if the confound has no variance).
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(confound, dtype=float)
    if X.size == 0 or not np.isfinite(z).all() or np.std(z) < 1e-9:
        return X.copy()
    zc = (z - z.mean()) / z.std()
    Z = np.column_stack([np.ones_like(zc), zc])
    beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return X - Z @ beta


def infer_task(y) -> str:
    """Heuristically classify ``y`` as ``"regression"`` or ``"classification"``.

    Float dtype with many distinct values -> regression; otherwise (integer
    labels / few uniques / non-numeric) -> classification.
    """
    y = np.asarray(y)
    if y.dtype.kind in "fc" and len(np.unique(y)) > 10:
        return "regression"
    return "classification"


def _make_estimator(task: str, *, classifier: str = "lda", alpha: float = 1.0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "regression":
        # ``ridgecv`` tunes the ridge penalty by internal (leave-one-out) CV on each training
        # fold -- the nested-CV upgrade needed when the feature space is high-dimensional
        # (whole-brain pooled decode, n_features >> n_trials), where a single fixed alpha is
        # arbitrary. Default (``Ridge(alpha=1.0)``) is byte-identical to the original path.
        if classifier == "ridgecv":
            from sklearn.linear_model import RidgeCV

            est = RidgeCV(alphas=np.logspace(-3, 3, 13))
        else:
            from sklearn.linear_model import Ridge

            est = Ridge(alpha=alpha)
    elif task == "classification":
        if classifier == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            est = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
        elif classifier == "logistic":
            from sklearn.linear_model import LogisticRegression

            est = LogisticRegression(max_iter=1000, C=1.0)
        elif classifier == "logl2cv":
            # L2-logistic with the penalty tuned by internal CV per training fold -- the
            # classification counterpart of ``ridgecv`` for the high-dimensional pooled decode.
            from sklearn.linear_model import LogisticRegressionCV

            est = LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), penalty="l2",
                                       solver="lbfgs", max_iter=2000)
        else:
            raise ValueError(f"unknown classifier={classifier!r}")
    else:
        raise ValueError(f"unknown task={task!r}")
    return Pipeline([("scale", StandardScaler()), ("est", est)])


def _score_regression(y_true, y_pred) -> float:
    """Cross-validated Pearson r between OOF predictions and truth."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def decode_cv(
    X,
    y,
    *,
    task: str | None = None,
    n_splits: int = 5,
    classifier: str = "lda",
    alpha: float = 1.0,
    random_state: int = 0,
) -> dict:
    """One cross-validated decoding score.

    Returns a dict with ``score``, ``score_name``, ``chance``, ``task``, ``n``,
    and ``n_classes`` (classification only). Regression score is OOF Pearson r
    (chance 0); classification score is balanced accuracy (chance 1/n_classes).

    Out-of-fold predictions are pooled across folds before scoring, which is
    more stable than averaging per-fold scores at small ``n``.
    """
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    finite = np.isfinite(X).all(axis=1)
    if y.dtype.kind in "fc":
        finite &= np.isfinite(y)
    X, y = X[finite], y[finite]
    n = len(y)

    task = task or infer_task(y)
    est = _make_estimator(task, classifier=classifier, alpha=alpha)

    if task == "regression":
        cv = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=random_state)
        y_pred = cross_val_predict(est, X, y.astype(float), cv=cv)
        return {
            "score": _score_regression(y, y_pred),
            "score_name": "oof_pearson_r",
            "chance": 0.0,
            "task": task,
            "n": int(n),
        }

    # classification
    from sklearn.metrics import balanced_accuracy_score

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    min_count = counts.min()
    k = int(min(n_splits, min_count)) if min_count >= 2 else 0
    if n_classes < 2 or k < 2:
        return {
            "score": np.nan, "score_name": "balanced_accuracy",
            "chance": np.nan if n_classes < 1 else 1.0 / n_classes,
            "task": task, "n": int(n), "n_classes": int(n_classes),
            "note": "too few classes/samples for CV",
        }
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(est, X, y, cv=cv)
    return {
        "score": float(balanced_accuracy_score(y, y_pred)),
        "score_name": "balanced_accuracy",
        "chance": 1.0 / n_classes,
        "task": task,
        "n": int(n),
        "n_classes": int(n_classes),
    }


def decode_with_permutation(
    X,
    y,
    *,
    task: str | None = None,
    n_perm: int = 200,
    n_splits: int = 5,
    classifier: str = "lda",
    alpha: float = 1.0,
    random_state: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """:func:`decode_cv` plus a label-permutation null.

    The null shuffles ``y`` and re-runs the *entire* CV (so any optimistic
    bias in the procedure is reflected in the null too). One-sided p-value
    (observed score greater than null), observed statistic included.

    Returns the :func:`decode_cv` dict augmented with ``p``, ``n_perm``, and
    ``null_mean``/``null_std``.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)

    obs = decode_cv(
        X, y, task=task, n_splits=n_splits, classifier=classifier,
        alpha=alpha, random_state=random_state,
    )
    if not np.isfinite(obs["score"]):
        return {**obs, "p": np.nan, "n_perm": 0, "null_mean": np.nan, "null_std": np.nan}

    y = np.asarray(y)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_sh = rng.permutation(y)
        null[i] = decode_cv(
            X, y_sh, task=obs["task"], n_splits=n_splits, classifier=classifier,
            alpha=alpha, random_state=random_state,
        )["score"]
    null = null[np.isfinite(null)]
    p = (1 + int(np.sum(null >= obs["score"]))) / (len(null) + 1)
    return {
        **obs,
        "p": float(p),
        "n_perm": int(len(null)),
        "null_mean": float(np.mean(null)) if len(null) else np.nan,
        "null_std": float(np.std(null)) if len(null) else np.nan,
    }


def _fit_residualizer(X, confound):
    """Fit a linear (intercept + standardised confound) residualizer on TRAIN rows.

    Companion to :func:`residualize_columns` for the cross-generalisation setting, where the
    residualization must be fit on the *training* group and applied to the held-out group with the
    SAME betas and the SAME confound standardisation — fitting on the test group would let its
    structure leak in. Label-blind (it regresses out a nuisance such as ``decision_num``, never the
    decoded label). Returns ``(beta, c_mean, c_std)`` or ``None`` (a no-op) when the confound is
    absent or constant.
    """
    X = np.asarray(X, dtype=float)
    c = np.asarray(confound, dtype=float)
    if X.size == 0 or c.size == 0 or not np.isfinite(c).all() or np.std(c) < 1e-9:
        return None
    cm, cs = float(c.mean()), float(c.std())
    zc = (c - cm) / cs
    Z = np.column_stack([np.ones_like(zc), zc])
    beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return (beta, cm, cs)


def _apply_residualizer(X, confound, fit):
    """Apply a :func:`_fit_residualizer` result (train betas) to any rows; no-op if ``fit`` is None."""
    X = np.asarray(X, dtype=float).copy()
    if fit is None:
        return X
    beta, cm, cs = fit
    c = np.asarray(confound, dtype=float)
    zc = (c - cm) / cs
    Z = np.column_stack([np.ones_like(zc), zc])
    return X - Z @ beta


def _crossgen_core(X_train, y_train, X_test, y_test, *, task, classifier, alpha,
                   n_perm, rng, C_train, C_test, shared_labels):
    """Train-on-A / test-on-B decode + label-permutation null. Internal worker for the public fns.

    Returns ``(obs_score, chance, score_name, null_array, n_train, n_test, task)``. The scaler +
    estimator are fit on TRAIN only and scored on TEST, so the score measures *transfer* across the
    split. The null re-runs the entire fit->predict with labels shuffled (train only; or, when
    ``shared_labels``, one shared shuffle applied to both — for the cross-window case where train and
    test are the same trials viewed in two windows). Residualisation, when requested, is fit on train
    and applied to test with train betas; it is label-blind so it is done once outside the perm loop.
    """
    from sklearn.metrics import balanced_accuracy_score

    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)

    if shared_labels:
        if Xtr.shape[0] != Xte.shape[0]:
            raise ValueError("shared_labels=True requires X_train and X_test to have identical rows")
        m = np.isfinite(Xtr).all(axis=1) & np.isfinite(Xte).all(axis=1)
        if ytr.dtype.kind in "fc":
            m &= np.isfinite(ytr)
        mtr = mte = m
    else:
        mtr = np.isfinite(Xtr).all(axis=1)
        mte = np.isfinite(Xte).all(axis=1)
        if ytr.dtype.kind in "fc":
            mtr &= np.isfinite(ytr)
        if yte.dtype.kind in "fc":
            mte &= np.isfinite(yte)

    Xtr, ytr = Xtr[mtr], ytr[mtr]
    Xte, yte = Xte[mte], yte[mte]
    ctr = None if C_train is None else np.asarray(C_train, float)[mtr]
    cte = None if C_test is None else np.asarray(C_test, float)[mte]
    n_train, n_test = len(ytr), len(yte)

    if n_train == 0 or n_test == 0:
        return (np.nan, np.nan, "balanced_accuracy", np.empty(0), n_train, n_test, task or "classification")

    task = task or infer_task(np.concatenate([ytr, yte]))

    # residualise (fit on train, applied to both) -- label-blind, so once and outside the perm loop
    fit = _fit_residualizer(Xtr, ctr) if ctr is not None else None
    if fit is not None:
        Xtr = _apply_residualizer(Xtr, ctr, fit)
        Xte = _apply_residualizer(Xte, cte, fit)

    if task == "classification":
        classes = np.unique(np.concatenate([ytr, yte]))
        chance = 1.0 / len(classes) if len(classes) else np.nan
        score_name = "balanced_accuracy"
    else:
        chance = 0.0
        score_name = "crossgen_pearson_r"

    def _fit_score(y_tr_use, y_te_use):
        if task == "classification" and (len(np.unique(y_tr_use)) < 2 or len(np.unique(y_te_use)) < 2):
            return np.nan
        if n_train < 2 or n_test < 1:
            return np.nan
        est = _make_estimator(task, classifier=classifier, alpha=alpha)
        try:
            est.fit(Xtr, y_tr_use)
            y_pred = est.predict(Xte)
        except Exception:
            return np.nan
        if task == "regression":
            return _score_regression(y_te_use, y_pred)
        return float(balanced_accuracy_score(y_te_use, y_pred))

    obs = _fit_score(ytr, yte)
    if not np.isfinite(obs):
        return (np.nan, chance, score_name, np.empty(0), n_train, n_test, task)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        if shared_labels:
            yp = rng.permutation(ytr)   # same trials in train & test -> one shuffle, both sides
            null[i] = _fit_score(yp, yp)
        else:
            null[i] = _fit_score(rng.permutation(ytr), yte)
    null = null[np.isfinite(null)]
    return (float(obs), chance, score_name, null, n_train, n_test, task)


def decode_crossgen(X_train, y_train, X_test, y_test, *, task: str | None = None,
                    classifier: str = "lda", alpha: float = 1.0, n_perm: int = 200,
                    random_state: int = 0, rng: np.random.Generator | None = None,
                    C_train=None, C_test=None, shared_labels: bool = False) -> dict:
    """Train-on-group-A / test-on-group-B decoding with a leakage-proof permutation null.

    The generalisation counterpart of :func:`decode_with_permutation` (which does within-matrix CV):
    the StandardScaler + estimator pipeline is fit on the TRAINING group only and scored on the
    disjoint TEST group, so an above-chance score means the code *transfers* across the split (across
    characters, or across task windows) — the test that distinguishes an abstract / shared
    representation from one that is present but group-specific (a per-group perceptual artefact).

    Parameters mirror :func:`decode_with_permutation`, plus:

    ``C_train`` / ``C_test``
        Optional confound (e.g. ``decision_num``) residualised out, **fit on train, applied to test
        with train betas** (pass both or neither). Label-blind drift control under the split.
    ``shared_labels``
        Set when train and test are the SAME trials viewed two ways (cross-window: decision vs
        narration). Train and test are masked to identical rows and the null permutes one shared label
        vector applied to both sides, preserving the train<->test trial correspondence.

    Returns ``{score, score_name, chance, task, n_train, n_test, p, n_perm, null_mean, null_std}``.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    if (C_train is None) != (C_test is None):
        raise ValueError("pass both C_train and C_test, or neither")
    obs, chance, score_name, null, n_train, n_test, task = _crossgen_core(
        X_train, y_train, X_test, y_test, task=task, classifier=classifier, alpha=alpha,
        n_perm=n_perm, rng=rng, C_train=C_train, C_test=C_test, shared_labels=shared_labels,
    )
    if not np.isfinite(obs):
        return {"score": np.nan, "score_name": score_name, "chance": chance, "task": task,
                "n_train": int(n_train), "n_test": int(n_test), "p": np.nan, "n_perm": 0,
                "null_mean": np.nan, "null_std": np.nan}
    p = (1 + int(np.sum(null >= obs))) / (len(null) + 1)
    return {"score": float(obs), "score_name": score_name, "chance": float(chance), "task": task,
            "n_train": int(n_train), "n_test": int(n_test), "p": float(p), "n_perm": int(len(null)),
            "null_mean": float(np.mean(null)) if len(null) else np.nan,
            "null_std": float(np.std(null)) if len(null) else np.nan}


def decode_crossgen_grouped(X, y, groups, *, fold: str = "leave_one_group_out",
                            task: str | None = None, classifier: str = "lda", alpha: float = 1.0,
                            n_perm: int = 200, random_state: int = 0,
                            rng: np.random.Generator | None = None, confound=None) -> dict:
    """Leave-one-group-out cross-generalisation, averaged over folds (CCGP-style abstraction test).

    For each unique level of ``groups`` (e.g. character identity), train on all other groups and test
    on the held-out one; average the held-out scores (trial-count weighted). Above-chance ⇒ the code
    *generalises across the grouping* (abstract / group-general), not bound to the trained groups'
    stimuli. A fold-averaged permutation null gives a per-subject p (each fold uses its own rng;
    per-permutation-index fold averages are independent Monte-Carlo draws of the averaged statistic;
    folds shorter than ``n_perm`` are chance-padded). The primary inference remains the across-subject
    test on the returned ``score``.

    Only ``fold="leave_one_group_out"`` is implemented. Returns the :func:`decode_crossgen` dict shape
    plus ``n_folds`` and ``fold_scores``.
    """
    if fold != "leave_one_group_out":
        raise ValueError(f"unsupported fold={fold!r}")
    if rng is None:
        rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    g = np.asarray(groups)
    conf = None if confound is None else np.asarray(confound, float)
    levels = list(np.unique(g))
    task = task or infer_task(y)

    fold_scores, fold_chances, fold_nulls, fold_ntest, fold_ntrain = [], [], [], [], []
    for j, lv in enumerate(levels):
        tr = g != lv
        te = g == lv
        ct = None if conf is None else conf[tr]
        ce = None if conf is None else conf[te]
        fr = np.random.default_rng(random_state + 1 + j)
        sc, ch, _sname, null, ntr, nte, task = _crossgen_core(
            X[tr], y[tr], X[te], y[te], task=task, classifier=classifier, alpha=alpha,
            n_perm=n_perm, rng=fr, C_train=ct, C_test=ce, shared_labels=False,
        )
        if not np.isfinite(sc):
            continue
        nz = np.full(n_perm, ch, dtype=float)      # chance-pad so folds align for element-wise mean
        nz[: min(len(null), n_perm)] = null[:n_perm]
        fold_scores.append(sc); fold_chances.append(ch); fold_nulls.append(nz)
        fold_ntest.append(nte); fold_ntrain.append(ntr)

    sname = "balanced_accuracy" if task == "classification" else "crossgen_pearson_r"
    n_folds = len(fold_scores)
    if n_folds < 2:
        return {"score": np.nan, "score_name": sname,
                "chance": float(fold_chances[0]) if fold_chances else np.nan, "task": task,
                "n_train": 0, "n_test": int(sum(fold_ntest)), "p": np.nan, "n_perm": 0,
                "null_mean": np.nan, "null_std": np.nan, "n_folds": int(n_folds),
                "fold_scores": [float(s) for s in fold_scores]}
    w = np.asarray(fold_ntest, dtype=float)
    w = w / w.sum()
    score_avg = float(np.average(fold_scores, weights=w))
    chance = float(np.average(fold_chances, weights=w))
    null_avg = np.average(np.vstack(fold_nulls), axis=0, weights=w)
    p = (1 + int(np.sum(null_avg >= score_avg))) / (len(null_avg) + 1)
    return {"score": score_avg, "score_name": sname, "chance": chance, "task": task,
            "n_train": int(np.mean(fold_ntrain)), "n_test": int(np.mean(fold_ntest)),
            "p": float(p), "n_perm": int(len(null_avg)),
            "null_mean": float(np.mean(null_avg)), "null_std": float(np.std(null_avg)),
            "n_folds": int(n_folds), "fold_scores": [float(s) for s in fold_scores]}


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Benjamini–Hochberg FDR. Returns ``(rejected_bool, qvalues)`` in input order.

    NaN p-values pass through as not-rejected with NaN q.
    """
    p = np.asarray(pvals, dtype=float)
    out_rej = np.zeros(p.shape, dtype=bool)
    out_q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out_rej, out_q
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.minimum(q, 1.0)
    q_orig = np.empty(m)
    q_orig[order] = q
    rej = q_orig <= alpha
    out_rej[valid] = rej
    out_q[valid] = q_orig
    return out_rej, out_q


def group_test(values, *, popmean: float = 0.0, test: str = "wilcoxon", alternative: str = "greater"):
    """Group inference on per-subject decoding scores against a chance level.

    Parameters
    ----------
    values
        Per-subject scores (e.g. OOF Pearson r, or balanced accuracy).
    popmean
        Null center: 0 for Pearson r; the chance accuracy (1/n_classes) for
        balanced accuracy (pass the per-variable chance).
    test
        ``"wilcoxon"`` (signed-rank, default) or ``"ttest"`` (one-sample t).
    alternative
        ``"greater"`` (decoding above chance), ``"two-sided"``, or ``"less"``.

    Returns
    -------
    dict
        ``{"stat", "p", "n", "median", "mean", "test", "popmean"}``.
    """
    from scipy import stats

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return {"stat": np.nan, "p": np.nan, "n": int(n),
                "median": float(np.median(v)) if n else np.nan,
                "mean": float(np.mean(v)) if n else np.nan,
                "test": test, "popmean": popmean}
    if test == "wilcoxon":
        d = v - popmean
        d = d[d != 0]
        if d.size < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = stats.wilcoxon(d, alternative=alternative)
    elif test == "ttest":
        res = stats.ttest_1samp(v, popmean, alternative=alternative)
        stat, p = res.statistic, res.pvalue
    else:
        raise ValueError(f"unknown test={test!r}")
    return {"stat": float(stat) if np.isfinite(stat) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
            "n": int(n), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "test": test, "popmean": float(popmean)}
