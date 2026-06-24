"""Tests for the aperiodic intrinsic-timescale estimators + the CTAD confound gate.

Grounding (Preston, Smith & Voytek 2026, Nat Hum Behav, "communication through aperiodic
dynamics"): two INDEPENDENT signals that each have autocorrelation structure become more
correlated as their (shared) intrinsic timescale increases (Yule's nonsense-correlation;
their ref 75). So shared aperiodic timescale can INFLATE inter-regional coupling with no
true communication -- a confound that must be partialled out before any "theta connectivity
encodes behaviour" claim. `test_ctad_inflation` is the gate: it must pass before any cohort
connectivity job runs.

Verifies:
1. knee_to_timescale_ms: the knee+exponent -> tau formula (analytic), and invalid-param NaN.
2. intrinsic_timescale_acw: recovers an AR(1) process's known timescale (1/e crossing) and
   orders monotonically with the AR coefficient; ACW-0 (zero crossing) > ACW-e (1/e crossing).
3. per_trial_timescale: tidy per-(channel,trial) tau with the optional knee-tau column.
4. CTAD: coupling between INDEPENDENT autocorrelated signals rises with shared timescale,
   and regressing the timescale out removes that timescale-driven inflation.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore[assignment]

import mne

from LFPAnalysis.analysis_utils import (
    knee_to_timescale_ms,
    intrinsic_timescale_acw,
    per_trial_timescale,
)

FS = 500.0


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a, float).ravel(), np.asarray(b, float).ravel())[0, 1])


def _ar1(phi, n, rng, burn=1000):
    """One AR(1) realization x[t] = phi*x[t-1] + e[t] (unit-innovation Gaussian)."""
    from scipy.signal import lfilter
    e = rng.standard_normal(n + burn)
    x = lfilter([1.0], [1.0, -float(phi)], e)
    return x[burn:]


def _ar1_tau_ms(phi):
    """Continuous-time timescale of a discrete AR(1): tau = -1/(fs*ln phi)."""
    return float(-1000.0 / (FS * np.log(phi)))


def test_knee_to_timescale_formula():
    # exponent=2 -> f_knee = sqrt(knee); tau = 1000/(2*pi*sqrt(knee))
    for knee in (4.0, 25.0, 100.0):
        expected = 1000.0 / (2.0 * np.pi * np.sqrt(knee))
        assert abs(knee_to_timescale_ms(knee, 2.0) - expected) < 1e-6
    # larger knee frequency -> shorter timescale
    assert knee_to_timescale_ms(100.0, 2.0) < knee_to_timescale_ms(4.0, 2.0)
    # invalid params -> NaN
    for bad in (knee_to_timescale_ms(0.0, 2.0), knee_to_timescale_ms(10.0, 0.0),
                knee_to_timescale_ms(np.nan, 2.0), knee_to_timescale_ms(-1.0, 2.0)):
        assert bad != bad   # NaN


def test_acw_recovers_ar1_timescale():
    rng = np.random.default_rng(0)
    phis = np.array([0.80, 0.90, 0.95, 0.975, 0.99])
    true_ms = np.array([_ar1_tau_ms(p) for p in phis])
    est_ms = np.array([intrinsic_timescale_acw(_ar1(p, 8000, rng), FS, kind="acwe",
                                               max_lag_ms=800.0) for p in phis])
    # monotone recovery + within tolerance per level
    assert _corr(true_ms, est_ms) > 0.95
    rel = np.abs(est_ms - true_ms) / true_ms
    assert (rel < 0.30).all(), f"rel err {rel}"
    # ACW-0 (zero crossing) occurs later than ACW-e (1/e crossing)
    x = _ar1(0.95, 8000, rng)
    assert (intrinsic_timescale_acw(x, FS, kind="acw0", max_lag_ms=800.0)
            > intrinsic_timescale_acw(x, FS, kind="acwe", max_lag_ms=800.0))
    # degenerate input -> NaN, no raise
    assert intrinsic_timescale_acw(np.zeros(2000), FS) != intrinsic_timescale_acw(np.zeros(2000), FS)


def test_per_trial_timescale_shape():
    rng = np.random.default_rng(1)
    n_ep, n_t = 12, 1500
    X = np.zeros((n_ep, 2, n_t))
    for e in range(n_ep):
        X[e, 0] = _ar1(0.90, n_t, rng)
        X[e, 1] = _ar1(0.95, n_t, rng)
    ep = mne.EpochsArray(X, mne.create_info(["c0", "c1"], FS, "seeg"), tmin=-1.5, verbose="ERROR")
    df = per_trial_timescale(ep, ["c0", "c1"], acw_kind="acwe", max_lag_ms=600.0)
    assert len(df) == n_ep * 2
    assert df.tau_acw_ms.notna().mean() > 0.9
    # the higher-phi channel has the longer median timescale
    med = df.groupby("channel").tau_acw_ms.median()
    assert med["c1"] > med["c0"]
    # knee-tau column appears when requested
    dfk = per_trial_timescale(ep, ["c0"], knee_tau=True)
    assert "tau_knee_ms" in dfk.columns


def test_ctad_inflation():
    """GATE: spurious coupling between independent signals grows with shared timescale,
    and a timescale covariate removes that timescale-driven inflation."""
    rng = np.random.default_rng(2026)
    n_t, n_pairs = 1500, 25
    phis = np.linspace(0.80, 0.985, 8)
    tau_est, absc = [], []
    for phi in phis:
        for _ in range(n_pairs):
            x = _ar1(phi, n_t, rng)
            y = _ar1(phi, n_t, rng)                       # INDEPENDENT of x
            absc.append(abs(_corr(x, y)))                 # spurious coupling (Yule)
            tau_est.append(0.5 * (intrinsic_timescale_acw(x, FS, max_lag_ms=600.0)
                                  + intrinsic_timescale_acw(y, FS, max_lag_ms=600.0)))
    tau_est = np.asarray(tau_est)
    absc = np.asarray(absc)

    # 1. the CTAD inflation: coupling between INDEPENDENT signals tracks shared timescale
    assert _corr(tau_est, absc) > 0.3, f"no timescale inflation (r={_corr(tau_est, absc):.3f})"

    # 2. raw coupling is clearly larger for long-timescale than short-timescale pairs
    order = np.argsort(tau_est)
    lo, hi = order[: len(order) // 3], order[-len(order) // 3:]
    raw_gap = absc[hi].mean() - absc[lo].mean()
    assert raw_gap > 0.05, f"raw long-vs-short gap too small ({raw_gap:.3f})"

    # 3. regress timescale out -> the long-vs-short inflation gap collapses
    b1, b0 = np.polyfit(tau_est, absc, 1)
    assert b1 > 0
    resid = absc - (b1 * tau_est + b0)
    resid_gap = resid[hi].mean() - resid[lo].mean()
    assert abs(resid_gap) < 0.02, f"tau-partial left residual gap {resid_gap:.3f}"


if __name__ == "__main__":
    test_knee_to_timescale_formula()
    test_acw_recovers_ar1_timescale()
    test_per_trial_timescale_shape()
    test_ctad_inflation()
    print("all aperiodic-timescale tests passed (incl. CTAD gate)")
