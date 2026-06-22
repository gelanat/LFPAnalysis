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
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=alpha)
    elif task == "classification":
        if classifier == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            est = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
        elif classifier == "logistic":
            from sklearn.linear_model import LogisticRegression

            est = LogisticRegression(max_iter=1000, C=1.0)
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
