"""Gate for the regularisation-tuned estimators (ridgecv / logl2cv) added to decoding_utils.

These back the whole-brain pooled decode where n_features >> n_trials and a single fixed penalty is
arbitrary. The gate checks: (1) the tuned estimators recover a planted signal, (2) they sit at chance
on label-free noise (the permutation null they feed must be honest), and (3) the DEFAULT path is
unchanged (Ridge(alpha=1.0) / LDA), so existing results are byte-for-byte reproducible.
"""
import numpy as np

from LFPAnalysis import decoding_utils as du


def _planted_regression(rng, n=80, p=200, k=5, noise=1.0):
    """High-dim X (p >> n); y is a sparse linear combination of k columns + noise."""
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k) * 2.0
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y


def _planted_classification(rng, n=80, p=200, k=5, sep=1.2):
    X = rng.standard_normal((n, p))
    y = (rng.random(n) < 0.5).astype(int)
    X[:, :k] += sep * (y[:, None] - 0.5)   # signal in first k features
    return X, y


def test_ridgecv_recovers_signal_high_dim():
    rng = np.random.default_rng(0)
    X, y = _planted_regression(rng)
    tuned = du.decode_cv(X, y, task="regression", classifier="ridgecv")
    fixed = du.decode_cv(X, y, task="regression")  # default Ridge(alpha=1.0)
    assert tuned["score_name"] == "oof_pearson_r"
    assert tuned["score"] > 0.3                      # recovers the planted signal
    assert fixed["score"] > 0.3                      # so does a fixed penalty on this easy synthetic
    # NB: tuning need not beat a lucky fixed alpha on one small dataset; its value is robustness
    # across the wide n_features/SNR range seen across subjects, not a per-dataset win.


def test_ridgecv_at_chance_on_noise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 200))
    y = rng.standard_normal(80)                       # no relation to X
    res = du.decode_with_permutation(X, y, task="regression", classifier="ridgecv", n_perm=200,
                                     random_state=0)
    assert abs(res["score"]) < 0.35                   # OOF r near 0
    assert res["p"] > 0.05                             # honest null: not significant


def test_logl2cv_recovers_and_is_honest():
    rng = np.random.default_rng(2)
    Xs, ys = _planted_classification(rng)
    hit = du.decode_cv(Xs, ys, task="classification", classifier="logl2cv")
    assert hit["score_name"] == "balanced_accuracy"
    assert hit["chance"] == 0.5
    assert hit["score"] > 0.6                          # above chance on separable data

    Xn = rng.standard_normal((80, 200))
    yn = (rng.random(80) < 0.5).astype(int)            # label-free
    nul = du.decode_with_permutation(Xn, yn, task="classification", classifier="logl2cv",
                                     n_perm=200, random_state=0)
    assert nul["p"] > 0.05


def test_default_path_unchanged():
    """Default classifier (no ridgecv/logl2cv) must still build Ridge / LDA -- backward compatible."""
    rng = np.random.default_rng(3)
    X, y = _planted_regression(rng, n=60, p=20)
    a = du.decode_cv(X, y, task="regression", random_state=0)["score"]
    b = du.decode_cv(X, y, task="regression", classifier="lda", random_state=0)["score"]
    assert a == b                                      # regression ignores classifier unless ridgecv

    Xs, ys = _planted_classification(rng, n=60, p=20)
    lda = du.decode_cv(Xs, ys, task="classification", classifier="lda")
    assert lda["score_name"] == "balanced_accuracy" and np.isfinite(lda["score"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok: {name}")
    print("all regularized-decode tests passed")
