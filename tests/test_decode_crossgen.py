"""Gate for decode_crossgen / decode_crossgen_grouped (train-on-A / test-on-B generalisation).

These back the two generalisation tests of the LFP dimension code: cross-character abstraction
(leave-one-character-out) and cross-window transfer (train decision, test narration). The gate checks
the property that makes a *generalisation* claim meaningful:

  1. a signal SHARED across the split is recovered (transfers);
  2. a signal that is strong WITHIN each group but does NOT transfer (sign flipped across the split)
     sits at/below chance with an honest one-sided null -- this is the discriminator between an
     abstract/shared code and a per-group (e.g. perceptual) artefact;
  3. label-free data gives an honest null (p > .05);
  4. the leave-one-group-out wrapper averages folds and recovers a shared signal;
  5. shared_labels (cross-window: same trials, two views) recovers and nulls honestly;
  6. the train-fit confound residualisation path does not destroy a real signal.
"""
import numpy as np

from LFPAnalysis import decoding_utils as du


def _planted_classification(rng, n=160, p=200, k=5, sep=1.2):
    """y in {0,1}; discriminative signal planted in the first k of p features (p can be >> n)."""
    X = rng.standard_normal((n, p))
    y = (rng.random(n) < 0.5).astype(int)
    X[:, :k] += sep * (y[:, None] - 0.5)
    return X, y


def test_crossgen_recovers_shared_signal():
    rng = np.random.default_rng(0)
    X, y = _planted_classification(rng, n=160)
    res = du.decode_crossgen(X[:80], y[:80], X[80:], y[80:], n_perm=200, random_state=0)
    assert res["score_name"] == "balanced_accuracy"
    assert res["chance"] == 0.5
    assert res["score"] > 0.6          # the shared signal transfers across the split
    assert res["p"] < 0.05


def test_crossgen_at_chance_when_signal_is_group_specific():
    """The crux: signal exists within each group but the discriminative axis flips across the split.

    A within-group decoder would succeed; a *generalisation* decoder must not. This is exactly the
    perceptual-confound threat (the two trial types differ per-group but share no transferable code).
    """
    rng = np.random.default_rng(1)
    n, p, k, sep = 80, 200, 5, 1.4
    Xtr = rng.standard_normal((n, p)); ytr = (rng.random(n) < 0.5).astype(int)
    Xtr[:, :k] += sep * (ytr[:, None] - 0.5)
    Xte = rng.standard_normal((n, p)); yte = (rng.random(n) < 0.5).astype(int)
    Xte[:, :k] -= sep * (yte[:, None] - 0.5)        # same features, OPPOSITE sign -> no transfer
    res = du.decode_crossgen(Xtr, ytr, Xte, yte, n_perm=200, random_state=0)
    assert res["score"] < 0.6                        # does not generalise (typically well below .5)
    assert res["p"] > 0.05                            # honest one-sided null


def test_crossgen_null_is_honest():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((160, 200))
    y = (rng.random(160) < 0.5).astype(int)          # label-free
    res = du.decode_crossgen(X[:80], y[:80], X[80:], y[80:], n_perm=200, random_state=0)
    assert res["p"] > 0.05


def test_crossgen_grouped_logo_recovers():
    rng = np.random.default_rng(3)
    n, p, k, sep = 200, 150, 5, 1.2
    X = rng.standard_normal((n, p)); y = (rng.random(n) < 0.5).astype(int)
    X[:, :k] += sep * (y[:, None] - 0.5)             # shared across all groups
    groups = rng.integers(0, 5, size=n)              # 5 character-like groups
    res = du.decode_crossgen_grouped(X, y, groups, n_perm=100, random_state=0)
    assert res["n_folds"] == 5
    assert res["score"] > 0.6
    assert res["p"] < 0.05


def test_crossgen_shared_labels_transfers_and_nulls():
    rng = np.random.default_rng(4)
    n, p, k, sep = 100, 120, 5, 1.3
    y = (rng.random(n) < 0.5).astype(int)
    Xa = rng.standard_normal((n, p)); Xa[:, :k] += sep * (y[:, None] - 0.5)   # window A
    Xb = rng.standard_normal((n, p)); Xb[:, :k] += sep * (y[:, None] - 0.5)   # window B, same trials
    res = du.decode_crossgen(Xa, y, Xb, y, shared_labels=True, n_perm=200, random_state=0)
    assert res["n_train"] == n and res["n_test"] == n
    assert res["score"] > 0.6
    assert res["p"] < 0.05
    Xn1 = rng.standard_normal((n, p)); Xn2 = rng.standard_normal((n, p))      # no shared signal
    nul = du.decode_crossgen(Xn1, y, Xn2, y, shared_labels=True, n_perm=200, random_state=0)
    assert nul["p"] > 0.05


def test_crossgen_confound_control_keeps_signal():
    """Train-fit confound residualisation (C_train/C_test) must not destroy a real, drift-orthogonal signal."""
    rng = np.random.default_rng(5)
    X, y = _planted_classification(rng, n=160)
    c = np.arange(160, dtype=float)                  # a decision_num-like drift confound
    res = du.decode_crossgen(X[:80], y[:80], X[80:], y[80:],
                             C_train=c[:80], C_test=c[80:], n_perm=100, random_state=0)
    assert res["score"] > 0.6


def test_crossgen_regression_path():
    rng = np.random.default_rng(7)
    n, p, k = 160, 50, 5
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:k] = rng.standard_normal(k) * 2.0
    yv = X @ beta + 0.5 * rng.standard_normal(n)
    res = du.decode_crossgen(X[:80], yv[:80], X[80:], yv[80:], task="regression",
                             classifier="ridgecv", n_perm=50, random_state=0)
    assert res["score_name"] == "crossgen_pearson_r"
    assert res["chance"] == 0.0
    assert res["score"] > 0.3


def test_crossgen_mismatched_confound_args_raise():
    rng = np.random.default_rng(8)
    X, y = _planted_classification(rng, n=80, p=20)
    c = np.arange(40, dtype=float)
    try:
        du.decode_crossgen(X[:40], y[:40], X[40:], y[40:], C_train=c, C_test=None, n_perm=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when only one of C_train/C_test is given")


# ---------------------------------------------------------------------------
# X_by_group -- per-fold feature matrices (inductive feature extraction)
#
# Exists so the feature EXTRACTION can be fit on training groups only (e.g. broadband
# sub-band normalization via representational_utils.feature_matrix(norm_fit_idx=...))
# without re-implementing the fold loop, the trial-count weighting, or the element-wise
# fold-null averaging in the analysis scripts.
# ---------------------------------------------------------------------------


def _grouped_fixture(seed=11, n=200, p=150, k=5, sep=1.2, n_groups=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (rng.random(n) < 0.5).astype(int)
    X[:, :k] += sep * (y[:, None] - 0.5)
    groups = rng.integers(0, n_groups, size=n)
    return X, y, groups


def test_crossgen_grouped_x_by_group_defaults_are_identical():
    """FROZEN-PIPELINE GUARD: None, {}, and an all-levels-mapped-to-X dict agree exactly."""
    X, y, g = _grouped_fixture()
    kw = dict(n_perm=100, random_state=0)
    base = du.decode_crossgen_grouped(X, y, g, **kw)
    empty = du.decode_crossgen_grouped(X, y, g, X_by_group={}, **kw)
    ident = du.decode_crossgen_grouped(
        X, y, g, X_by_group={lv: X for lv in np.unique(g)}, **kw)
    for other in (empty, ident):
        assert other["score"] == base["score"]
        assert other["p"] == base["p"]
        assert other["fold_scores"] == base["fold_scores"]


def test_crossgen_grouped_x_by_group_is_used_per_fold():
    """Destroying the signal in the matrix used for ONE held-out level must lower the
    average and that fold's score specifically -- proving the mapping is per-fold, not global."""
    X, y, g = _grouped_fixture(seed=12)
    levels = list(np.unique(g))
    victim = levels[0]
    rng = np.random.default_rng(0)
    X_dead = rng.standard_normal(X.shape)              # same shape, no signal at all
    kw = dict(n_perm=50, random_state=0)
    base = du.decode_crossgen_grouped(X, y, g, **kw)
    swapped = du.decode_crossgen_grouped(X, y, g, X_by_group={victim: X_dead}, **kw)

    assert swapped["n_folds"] == base["n_folds"]
    assert swapped["fold_scores"][0] < base["fold_scores"][0]     # the swapped fold degrades
    assert swapped["fold_scores"][1:] == base["fold_scores"][1:]  # the others are untouched
    assert swapped["score"] < base["score"]


def test_crossgen_grouped_x_by_group_shape_mismatch_raises():
    X, y, g = _grouped_fixture(seed=13)
    lv = np.unique(g)[0]
    try:
        du.decode_crossgen_grouped(X, y, g, X_by_group={lv: X[:, :3]}, n_perm=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on X_by_group shape mismatch")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok: {name}")
    print("all decode_crossgen tests passed")
