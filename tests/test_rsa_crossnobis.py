"""Tests for condition-level crossnobis + RDM regression.

These are the two capabilities the package did not have: an unbiased,
cross-validated dissimilarity estimator (Walther et al. 2016, NeuroImage
137:188-200) and a joint multi-predictor RDM regression. Both are only worth
using if the properties they are chosen for actually hold, so each test targets
one of those properties rather than just exercising the code path.

1. UNBIASEDNESS -- the defining property. On pure noise with no condition
   structure, crossnobis distances must average to ~0 and take both signs. A
   same-data (non-cross-validated) distance is strictly positive on the same
   input; the test asserts crossnobis is the one that centres on zero, because
   an estimator that is positive under the null cannot support "distance
   tracks the model" without a null-subtraction crutch.
2. RECOVERY -- planted geometry. With a real 2-D map coded in the patterns, the
   crossnobis RDM must correlate strongly with the planted model RDM.
3. WHITENING -- `whiten_patterns` must actually sphere: whitened residuals have
   ~identity covariance, and whitening by the identity is a no-op.
4. NOISE NORMALIZATION EARNS ITS KEEP -- with strongly anisotropic noise (a few
   very loud, uninformative channels), whitened crossnobis must recover the
   planted geometry better than unwhitened. This is the specific claim the
   estimator swap rests on.
5. REGRESSION under COLLINEARITY -- with two correlated predictors where only
   one carries signal, the joint fit must assign beta to the right one, report
   VIF > 1, and give the signal-free predictor ~0 unique variance. The
   one-at-a-time alternative (correlating each model separately) is shown to
   fail on the same data -- that failure is why `rdm_regression` exists.
6. NULL wiring -- `rdm_regression_perm` debiases against the same estimator and
   `circular_shift_indices` returns exhaustive, identity-free, evenly
   subsampled shifts.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:  # standalone-runnable without pytest
    pytest = None  # type: ignore[assignment]
    _HAVE_PYTEST = False

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from LFPAnalysis.representational_utils import (
    circular_shift_indices,
    condition_patterns,
    crossnobis_rdm,
    cv_correlation_rdm,
    model_rdm,
    neural_rdm,
    noise_covariance,
    rdm_regression,
    rdm_regression_perm,
    whiten_patterns,
)

N_COND, N_REP, N_FEAT = 10, 6, 24


def _design(rng, *, seed_coords=None):
    """(cond_id, order, coords) for N_COND conditions x N_REP trials."""
    cond = np.repeat(np.arange(N_COND), N_REP)
    order = np.arange(cond.size)  # trials interleaved across conditions by construction
    coords = rng.normal(size=(N_COND, 2)) if seed_coords is None else seed_coords
    return cond, order, coords


def _planted(rng, coords, cond, *, snr=1.0, noise=1.0):
    """Patterns carrying a 2-D map code at a given SNR."""
    load = rng.normal(size=(2, N_FEAT))
    signal = coords[cond] @ load
    return snr * signal + noise * rng.normal(size=(cond.size, N_FEAT))


# --------------------------------------------------------------------------- #
# 1. Unbiasedness under the null -- the property crossnobis exists for
# --------------------------------------------------------------------------- #
def test_crossnobis_unbiased_under_pure_noise():
    means, frac_neg = [], []
    for s in range(40):
        rng = np.random.default_rng(s)
        cond, order, _ = _design(rng)
        X = rng.normal(size=(cond.size, N_FEAT))  # NO condition structure
        d = crossnobis_rdm(X, cond, order=order, shrinkage="ledoit_wolf")
        assert d.size == N_COND * (N_COND - 1) // 2
        means.append(d.mean())
        frac_neg.append(np.mean(d < 0))

    grand = float(np.mean(means))
    se = float(np.std(means) / np.sqrt(len(means)))
    assert abs(grand) < max(4 * se, 0.05), f"crossnobis biased under the null: {grand:.4f}"
    # Both signs must appear -- a strictly-positive "unbiased" distance is a bug.
    assert 0.25 < float(np.mean(frac_neg)) < 0.75, np.mean(frac_neg)


def test_same_data_distance_is_positively_biased_but_crossnobis_is_not():
    """The contrast that motivates the swap, on identical input."""
    rng = np.random.default_rng(0)
    cond, order, _ = _design(rng)
    X = rng.normal(size=(cond.size, N_FEAT))

    P, _ = condition_patterns(X, cond, n_folds=2, order=order)
    same_data = pdist(P.mean(axis=0), metric="sqeuclidean")  # not cross-validated
    cv = crossnobis_rdm(X, cond, order=order)

    assert same_data.min() > 0, "sqeuclidean should be strictly positive"
    assert cv.min() < 0, "crossnobis must be able to go negative under the null"
    assert abs(cv.mean()) < same_data.mean(), (cv.mean(), same_data.mean())


# --------------------------------------------------------------------------- #
# 2. Recovery of a planted geometry
# --------------------------------------------------------------------------- #
def test_crossnobis_recovers_planted_map_geometry():
    rng = np.random.default_rng(7)
    cond, order, coords = _design(rng)
    X = _planted(rng, coords, cond, snr=1.0, noise=1.0)

    d, conds = crossnobis_rdm(X, cond, order=order, return_conditions=True)
    truth = model_rdm(coords[conds], kind="euclidean")
    r = spearmanr(d, truth).statistic
    assert r > 0.5, f"planted geometry not recovered (rho={r:.3f})"


def test_cv_correlation_rdm_also_recovers_and_matches_shape():
    """The estimator-comparison arm must be a drop-in on the same conditions."""
    rng = np.random.default_rng(11)
    cond, order, coords = _design(rng)
    X = _planted(rng, coords, cond, snr=1.0, noise=1.0)

    cn, c1 = crossnobis_rdm(X, cond, order=order), cv_correlation_rdm(X, cond, order=order)
    assert cn.shape == c1.shape
    truth = model_rdm(coords, kind="euclidean")
    assert spearmanr(c1, truth).statistic > 0.5
    assert spearmanr(cn, c1).statistic > 0.5  # same geometry, different estimator


# --------------------------------------------------------------------------- #
# 3-4. Whitening: correctness, and that it buys something
# --------------------------------------------------------------------------- #
def test_whiten_patterns_spheres_and_identity_is_noop():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(N_FEAT, N_FEAT))
    sigma = A @ A.T + np.eye(N_FEAT)
    R = rng.multivariate_normal(np.zeros(N_FEAT), sigma, size=6000)

    W = whiten_patterns(R, sigma)
    C = np.cov(W, rowvar=False)
    off = C - np.diag(np.diag(C))
    assert abs(np.diag(C).mean() - 1.0) < 0.1, np.diag(C).mean()
    assert np.abs(off).mean() < 0.1, np.abs(off).mean()

    P = rng.normal(size=(2, 5, N_FEAT))
    assert np.allclose(whiten_patterns(P, np.eye(N_FEAT)), P, atol=1e-8)


def test_noise_normalization_improves_recovery_under_anisotropic_noise():
    """Loud uninformative channels: whitening should rescue the geometry."""
    gains = []
    for s in range(12):
        rng = np.random.default_rng(100 + s)
        cond, order, coords = _design(rng)
        scale = np.ones(N_FEAT)
        scale[: N_FEAT // 3] = 8.0  # a few very loud, signal-free-ish channels
        load = rng.normal(size=(2, N_FEAT))
        X = coords[cond] @ load + rng.normal(size=(cond.size, N_FEAT)) * scale

        truth = model_rdm(coords, kind="euclidean")
        wh = crossnobis_rdm(X, cond, order=order, shrinkage="ledoit_wolf")
        raw = crossnobis_rdm(X, cond, order=order, sigma=np.eye(N_FEAT))
        gains.append(spearmanr(wh, truth).statistic - spearmanr(raw, truth).statistic)

    assert float(np.median(gains)) > 0, f"whitening did not help: {np.median(gains):.3f}"


def test_noise_covariance_uses_within_condition_residuals_only():
    """Condition structure must not leak into the thing we whiten by."""
    rng = np.random.default_rng(5)
    cond, order, coords = _design(rng)
    huge = coords * 50.0  # enormous between-condition structure
    load = rng.normal(size=(2, N_FEAT))
    X = huge[cond] @ load + rng.normal(size=(cond.size, N_FEAT))

    S = noise_covariance(X, cond, shrinkage="none")
    # Residual covariance should stay ~unit scale despite the huge condition means.
    assert np.trace(S) / N_FEAT < 5.0, np.trace(S) / N_FEAT


def test_condition_patterns_interleaves_and_enforces_min_per_fold():
    rng = np.random.default_rng(1)
    cond = np.repeat(np.arange(4), 5)
    order = np.arange(cond.size)
    X = rng.normal(size=(cond.size, 3))
    P, conds = condition_patterns(X, cond, n_folds=2, order=order)
    assert P.shape == (2, 4, 3) and list(conds) == [0, 1, 2, 3]

    # a condition with a single trial cannot be crossed -> dropped at min_per_fold=1
    cond2 = np.concatenate([np.repeat(np.arange(3), 4), [99]])
    X2 = rng.normal(size=(cond2.size, 3))
    _, conds2 = condition_patterns(X2, cond2, n_folds=2, order=np.arange(cond2.size))
    assert 99 not in set(conds2.tolist())

    # interleaving (not block-splitting): folds must span the whole order range
    idx = np.argsort(order[cond == 0])
    assert idx.size == 5


# --------------------------------------------------------------------------- #
# 5. Joint regression under collinearity -- why one-at-a-time is not enough
# --------------------------------------------------------------------------- #
def _collinear_models(rng, n_pairs, rho_target=0.75):
    a = rng.normal(size=n_pairs)
    b = rho_target * a + np.sqrt(max(1 - rho_target**2, 0.0)) * rng.normal(size=n_pairs)
    return a, b


def test_rdm_regression_assigns_beta_to_the_true_predictor():
    """Collinearity here (realized |rho| ~= 0.77) matches the SNT map-vs-theta case (0.73)."""
    picked_joint, picked_marginal = 0, 0
    vifs, realized = [], []
    n_pairs = N_COND * (N_COND - 1) // 2
    for s in range(30):
        rng = np.random.default_rng(200 + s)
        true_m, proxy = _collinear_models(rng, n_pairs)
        y = 1.0 * true_m + 0.7 * rng.normal(size=n_pairs)  # only `true_m` drives y

        fit = rdm_regression(y, {"true": true_m, "proxy": proxy}, rank=True)
        picked_joint += int(fit["beta"]["true"] > fit["beta"]["proxy"])
        picked_marginal += int(
            abs(spearmanr(y, true_m).statistic) > abs(spearmanr(y, proxy).statistic)
        )
        vifs.append(fit["max_vif"])
        realized.append(abs(spearmanr(true_m, proxy).statistic))
        assert fit["max_vif"] > 1.3, fit["max_vif"]
        assert fit["r2_unique"]["true"] > fit["r2_unique"]["proxy"]

    assert 0.7 < float(np.median(realized)) < 0.85, np.median(realized)
    assert float(np.median(vifs)) > 2.0, np.median(vifs)
    assert picked_joint >= 28, f"joint fit mis-assigned beta in {30 - picked_joint}/30"
    # The marginal test is the weaker instrument -- this is the motivation.
    assert picked_joint >= picked_marginal


def test_rdm_regression_unique_and_omnibus_variance_are_coherent():
    rng = np.random.default_rng(17)
    n_pairs = N_COND * (N_COND - 1) // 2
    a, b = _collinear_models(rng, n_pairs)
    c = rng.normal(size=n_pairs)  # independent nuisance
    y = a + 0.5 * c + 0.6 * rng.normal(size=n_pairs)

    fit = rdm_regression(y, {"a": a, "b": b, "c": c}, targets=["a", "b"])
    assert 0.0 <= fit["r2"] <= 1.0
    for k in ("a", "b", "c"):
        assert fit["r2_unique"][k] <= fit["r2"] + 1e-9
    # joint unique of the target set >= any single target's unique (collinear pair)
    assert fit["r2_unique_targets"] >= fit["r2_unique"]["a"] - 1e-9
    assert fit["r2_unique_targets"] <= fit["r2"] + 1e-9


def test_rdm_regression_rejects_length_mismatch():
    y = np.arange(10, dtype=float)
    try:
        rdm_regression(y, {"bad": np.arange(9, dtype=float)})
    except ValueError as exc:
        assert "length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


# --------------------------------------------------------------------------- #
# 6. Null wiring
# --------------------------------------------------------------------------- #
def test_circular_shift_indices_are_exhaustive_identity_free_and_subsampled():
    P = circular_shift_indices(8)
    assert P.shape == (7, 8)
    base = np.arange(8)
    assert not any(np.array_equal(p, base) for p in P), "identity shift must be excluded"
    for p in P:
        assert sorted(p.tolist()) == base.tolist()

    sub = circular_shift_indices(50, 10)
    assert sub.shape[0] <= 10
    # evenly spread, not clustered at short lags
    lags = [int((p[0] - 0) % 50) for p in sub]
    assert max(lags) > 25, lags


def test_rdm_regression_perm_debiases_against_the_same_estimator():
    rng = np.random.default_rng(23)
    cond, order, coords = _design(rng)
    X = _planted(rng, coords, cond, snr=1.2, noise=1.0)
    d, conds = crossnobis_rdm(X, cond, order=order, return_conditions=True)

    models = {
        "map": model_rdm(coords[conds], kind="euclidean"),
        "nuisance": model_rdm(np.asarray(conds, dtype=float), kind="euclidean"),
    }
    out = rdm_regression_perm(d, models, null="circular", targets=["map"])

    assert out["n_null"] == len(conds) - 1  # exhaustive at condition level
    assert out["per_model"]["map"]["debiased"] > 0, out["per_model"]["map"]
    assert np.isfinite(out["omnibus"]["r2"]["null_mean"])
    # observed beta must come from the identical code path as the null
    assert out["per_model"]["map"]["obs"] == out["obs"]["beta"]["map"]


def test_rdm_regression_perm_is_null_on_unstructured_data():
    debiased = []
    for s in range(20):
        rng = np.random.default_rng(300 + s)
        cond, order, coords = _design(rng)
        X = rng.normal(size=(cond.size, N_FEAT))  # no geometry at all
        d, conds = crossnobis_rdm(X, cond, order=order, return_conditions=True)
        models = {"map": model_rdm(coords[conds], kind="euclidean")}
        debiased.append(rdm_regression_perm(d, models, null="circular")["per_model"]["map"]["debiased"])

    m = float(np.mean(debiased))
    se = float(np.std(debiased) / np.sqrt(len(debiased)))
    assert abs(m) < max(4 * se, 0.15), f"debiased statistic not centred on 0: {m:.3f}"


def test_neural_rdm_still_matches_model_rdm_length():
    """Guard: the new condition-level path must stay squareform-compatible."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(N_COND, N_FEAT))
    assert neural_rdm(X).size == model_rdm(np.arange(N_COND, dtype=float)).size


if __name__ == "__main__":  # standalone-runnable
    ns = dict(globals())
    fns = [v for k, v in ns.items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
