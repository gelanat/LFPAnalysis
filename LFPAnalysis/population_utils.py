"""Unsupervised population-geometry analysis for LFP feature matrices.

Companion to :mod:`representational_utils` (the feature builders) and
:mod:`decoding_utils` (the supervised decoders). Where those ask "can task
variable *y* be read out of the pattern?", this module asks the complementary,
label-light question: **what structure is in the population activity, and how
much of it is task-tuned versus a condition-independent scaffold?**

Three tools, numpy/scipy/scikit-learn only:

* :func:`population_pca` — linear intrinsic dimensionality of a per-trial
  feature matrix (explained-variance spectrum + participation ratio).
* :func:`dpca_marginalize` — demixed variance decomposition (Kobak et al. 2016,
  *eLife*). Splits the trial-averaged tensor into orthogonal marginals — a
  **condition-independent** (time-only) component and one component per task
  factor (and its interactions) — and reports each as a fraction of the total
  population variance. This turns the project's "content-free scaffold" framing
  into a *measured ratio*: scaffold = condition-independent fraction, content =
  the task-factor fractions. Implemented as an exact ANOVA-style orthogonal
  decomposition (the non-empty marginals sum to the total variance), so no
  external dPCA package is needed (the PyPI ``dPCA`` uses deprecated NumPy APIs).
* :func:`manifold_embed` — linear (PCA) then optional nonlinear (Isomap /
  spectral / UMAP / t-SNE) embedding of single-trial states, for the geometry
  RSA in the manifold driver.

The inferential contract is unchanged from the rest of the project: everything
here is computed **per subject**; a per-subject summary statistic (a variance
fraction, an RSA r) is the thing carried to the across-subject second-level test
(``snt_lfp.reliability.reliability_report``). Nulls are built by the caller by
permuting only the *relation* to behaviour — the decomposition itself is
label-blind, so there is no double-dipping.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = [
    "population_pca",
    "dpca_marginalize",
    "manifold_embed",
]


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    """Standardise each column (feature) across rows (trials); constant columns pass through."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def population_pca(X, *, n_components=None, zscore: bool = True) -> dict:
    """Linear PCA of a per-trial feature matrix — the intrinsic-dimensionality baseline.

    Parameters
    ----------
    X
        ``(n_trials, n_features)`` feature matrix (e.g. from
        :func:`representational_utils.feature_matrix`). Rows with any non-finite
        value are dropped.
    n_components
        Number of PCs to keep (default ``min(n_trials, n_features)``).
    zscore
        Standardise each feature across trials first (recommended — otherwise a
        few high-variance channels dominate the spectrum).

    Returns
    -------
    dict
        ``{"scores", "components", "explained_variance_ratio", "cum_evr",
        "participation_ratio", "n_trials", "n_features"}``. ``participation_ratio``
        = ``(Σλ)² / Σλ²`` is a scalar effective dimensionality (1 = one dominant
        PC, up to ``min(n_trials, n_features)`` = isotropic), the compact summary
        for the across-subject test.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_trials, n_features), got shape {X.shape}")
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    n_trials, n_features = X.shape
    if n_trials < 2 or n_features < 1:
        return {"scores": np.empty((n_trials, 0)), "components": np.empty((0, n_features)),
                "explained_variance_ratio": np.empty(0), "cum_evr": np.empty(0),
                "participation_ratio": np.nan, "n_trials": int(n_trials),
                "n_features": int(n_features)}
    Xz = _zscore_cols(X) if zscore else X - X.mean(axis=0, keepdims=True)
    k = min(n_trials, n_features) if n_components is None else int(n_components)
    k = max(1, min(k, n_trials, n_features))
    pca = PCA(n_components=k, svd_solver="full")
    scores = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_
    lam = pca.explained_variance_
    pr = float((lam.sum() ** 2) / np.sum(lam ** 2)) if np.any(lam > 0) else np.nan
    return {"scores": scores, "components": pca.components_,
            "explained_variance_ratio": evr, "cum_evr": np.cumsum(evr),
            "participation_ratio": pr, "n_trials": int(n_trials),
            "n_features": int(n_features)}


# --------------------------------------------------------------------------- #
# Demixed variance decomposition (dPCA marginalisation)
# --------------------------------------------------------------------------- #
def _anova_marginals(A: np.ndarray, n_factors: int) -> dict:
    """Exact ANOVA-style orthogonal marginal decomposition of a tensor.

    ``A`` has shape ``(n_units, L_1, ..., L_k)`` (``k = n_factors``): axis 0 is
    the unit (contact) axis, axes ``1..k`` are categorical task factors. Returns
    ``{S: marginal}`` for every subset ``S`` of factor indices ``0..k-1``
    (including the empty set = per-unit grand mean), where each marginal is the
    part of ``A`` that depends on *exactly* the factors in ``S``:

        marg_S = mean_over(axes not in S) A  -  Σ_{U ⊊ S} marg_U

    computed with subsets in increasing-size order so every ``marg_U`` is ready.
    The marginals are mutually orthogonal, so (excluding the empty set) their
    Frobenius energies sum to ``||A - grand_mean||²`` — the property that makes
    the variance fractions in :func:`dpca_marginalize` well defined.
    """
    factors = list(range(n_factors))            # factor i -> array axis i+1
    subsets: list[tuple[int, ...]] = []
    for r in range(n_factors + 1):
        subsets.extend(combinations(factors, r))

    marg: dict[tuple[int, ...], np.ndarray] = {}
    for S in subsets:                            # increasing |S|: subsets of S already done
        complement = tuple(a + 1 for a in factors if a not in S)
        avg_S = A.mean(axis=complement, keepdims=True) if complement else A
        m = avg_S.astype(float, copy=True)
        S_set = set(S)
        for U in subsets:
            if U != S and set(U).issubset(S_set):
                m = m - marg[U]                  # marg[U] broadcasts over the S\U axes
        marg[S] = m
    return marg


def dpca_marginalize(A, factor_names, *, groups: dict | None = None) -> dict:
    """Demixed variance decomposition of a trial-averaged tensor (Kobak et al. 2016).

    Parameters
    ----------
    A
        Trial-averaged tensor, shape ``(n_units, L_1, ..., L_k)`` — axis 0 is the
        unit/contact axis, each remaining axis is a categorical task factor whose
        levels index that axis. Must be finite (the caller fills every cell; an
        empty ``(factor-level)`` cell is a design error, not a NaN to average
        over). One of the factors is conventionally **time** (the sliding-window
        axis); its lone marginal is the condition-independent "scaffold".
    factor_names
        Names for axes ``1..k`` in order, e.g. ``["dimension", "time"]`` or
        ``["dimension", "decision", "time"]``.
    groups
        Optional mapping ``{group_name: [subset_of_factor_names, ...]}`` whose
        per-subset fractions are summed into one reported number. Subsets are
        given as tuples/lists of factor names (e.g. the condition-independent
        group is ``[("time",)]``; a dimension group is
        ``[("dimension",), ("dimension", "time")]``). If ``None``, only the raw
        per-subset fractions are returned.

    Returns
    -------
    dict
        ``{"frac": {name_tuple: fraction}, "grouped": {group: fraction},
        "total_ss": float, "n_units": int, "shape": tuple}``. ``frac`` sums to
        1.0 over all non-empty subsets; ``grouped`` sums to 1.0 iff ``groups``
        partitions the non-empty subsets.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim < 2:
        raise ValueError(f"A must be at least 2D (n_units, ...factors), got shape {A.shape}")
    n_factors = A.ndim - 1
    if len(factor_names) != n_factors:
        raise ValueError(f"factor_names has {len(factor_names)} entries for {n_factors} factor axes")
    if not np.isfinite(A).all():
        raise ValueError("A has non-finite entries — fill every design cell before marginalising")

    marg = _anova_marginals(A, n_factors)
    full = A.shape
    name = {i: factor_names[i] for i in range(n_factors)}

    ss = {}
    for S, m in marg.items():
        ss[S] = float(np.sum(np.broadcast_to(m, full) ** 2))
    total = sum(v for S, v in ss.items() if S)            # exclude empty set (grand mean)

    frac = {}
    for S, v in ss.items():
        if not S:
            continue
        key = tuple(sorted(name[i] for i in S))
        frac[key] = (v / total) if total > 0 else np.nan

    out = {"frac": frac, "total_ss": total, "n_units": int(A.shape[0]), "shape": tuple(full)}
    if groups is not None:
        grouped = {}
        for gname, subsets in groups.items():
            s = 0.0
            for sub in subsets:
                key = tuple(sorted(sub))
                if key not in frac:
                    raise KeyError(f"group {gname!r} references unknown marginal {key}")
                s += frac[key]
            grouped[gname] = s
        out["grouped"] = grouped
    return out


# --------------------------------------------------------------------------- #
# Manifold embedding (linear PCA -> optional nonlinear)
# --------------------------------------------------------------------------- #
def manifold_embed(X, *, method: str = "isomap", n_pca: int = 10, n_components: int = 3,
                   n_neighbors: int = 8, zscore: bool = True, random_state: int = 0) -> np.ndarray:
    """Embed single-trial population states into a low-dimensional geometry.

    A PCA pre-projection to ``n_pca`` dims (denoise + speed) followed by the
    chosen embedding to ``n_components`` dims. ``method="pca"`` returns the linear
    projection itself — the reference against which a nonlinear embedding's RSA is
    compared (the ``Isomap_r − PCA_r`` contrast in the manifold driver).

    Parameters
    ----------
    X
        ``(n_trials, n_features)`` feature matrix. Non-finite rows are dropped;
        the returned embedding has one row per *retained* trial. Use the returned
        ``keep`` mask (second element) to align behavioural labels.
    method
        ``"pca"`` (linear), ``"isomap"``, ``"spectral"`` (Laplacian eigenmaps),
        ``"tsne"``, or ``"umap"`` (requires ``umap-learn``).
    n_pca
        PCA pre-projection width (capped at ``min(n_trials, n_features)``); set 0
        to skip and embed the raw (z-scored) features.
    n_components, n_neighbors
        Embedding dimensionality and neighbourhood size (ignored by ``"pca"``;
        t-SNE maps ``n_neighbors`` to a comparable perplexity).

    Returns
    -------
    (embedding, keep)
        ``embedding`` is ``(n_retained_trials, n_components)``; ``keep`` is the
        boolean row mask into the original ``X`` (finite rows).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_trials, n_features), got shape {X.shape}")
    keep = np.isfinite(X).all(axis=1)
    Xk = X[keep]
    n_trials, n_features = Xk.shape
    # effective embedding width: can't exceed the feature count or n_trials-1 (small regions, e.g.
    # bbhfa8 gives 1 feature/contact, so a low-contact ROI has n_features < n_components).
    nc = int(min(n_components, n_features, max(n_trials - 1, 1)))
    if n_trials < 3 or nc < 1:
        return np.full((n_trials, max(n_components, 1)), np.nan), keep

    Xz = _zscore_cols(Xk) if zscore else Xk - Xk.mean(axis=0, keepdims=True)

    if n_pca and n_pca > 0:
        from sklearn.decomposition import PCA
        k = int(min(max(nc, min(int(n_pca), n_trials, n_features)), n_trials, n_features))
        Xz = PCA(n_components=k, svd_solver="full", random_state=random_state).fit_transform(Xz)

    nn = int(min(n_neighbors, n_trials - 1))
    if method == "pca":
        from sklearn.decomposition import PCA
        emb = PCA(n_components=nc, svd_solver="full",
                  random_state=random_state).fit_transform(Xz)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        emb = Isomap(n_neighbors=nn, n_components=nc).fit_transform(Xz)
    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding
        emb = SpectralEmbedding(n_components=nc, n_neighbors=nn,
                                random_state=random_state).fit_transform(Xz)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = float(max(5, min(30, (n_trials - 1) / 3)))
        emb = TSNE(n_components=min(nc, 3), perplexity=perp, init="pca",
                   random_state=random_state).fit_transform(Xz)
    elif method == "umap":
        import umap  # umap-learn
        emb = umap.UMAP(n_neighbors=nn, n_components=nc,
                        random_state=random_state).fit_transform(Xz)
    else:
        raise ValueError(f"unknown method={method!r}")
    return np.asarray(emb, dtype=float), keep
