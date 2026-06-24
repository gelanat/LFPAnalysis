"""Tests for the verified 1/f-robust band-power estimator (`aperiodic_corrected_band_power`).

This is the gold-standard scalar band-power path for "real oscillation vs 1/f": multitaper
PSD -> FOOOF -> periodic band power (aperiodic-corrected) + aperiodic exponent/offset, with
fit QC (R^2/error) and *visible* attrition (failed fits become rows, never silently dropped).
Verifies:

1. RECOVERY: on a clean 1/f + 6 Hz theta channel the fit passes QC, recovers the aperiodic
   exponent, and finds a theta peak (peak_cf in band, periodic theta > a peakless band).
2. THE CORRECTION MATTERS: on a STEEP-1/f, no-oscillation channel the *raw* theta power is
   high (1/f piles power at low freq) but the *periodic* theta power is ~0 and
   aperiodic_fraction ~1 -- i.e. raw != periodic, which is the entire point.
3. EXPONENT ORDERING: the steep channel's exponent > the shallow channel's.
4. ATTRITION IS VISIBLE: a degenerate (all-NaN) channel yields fit_ok=False / qc_pass=False
   rows without raising.
5. PER-TRIAL: per_trial=True emits n_trials finite per-trial periodic-theta rows per channel.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:  # standalone-runnable without pytest
    pytest = None  # type: ignore[assignment]
    _HAVE_PYTEST = False

import mne

from LFPAnalysis.analysis_utils import (
    aperiodic_corrected_band_power,
    per_trial_fooof_band_power,
)

FS = 500.0
N_EP = 40
N_T = 1500          # 3.0 s at 500 Hz
BANDS = {"theta": (2.0, 8.0), "beta": (13.0, 30.0)}   # beta = peakless control band


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def _colored(exponent, *, theta_amp=0.0, theta_f=6.0, seed=0):
    """One channel of 1/f noise (power ∝ f^-exponent) + optional 6 Hz theta tone."""
    rng = np.random.default_rng(seed)
    rfreqs = np.fft.rfftfreq(N_T, 1.0 / FS)
    shape = np.ones_like(rfreqs)
    shape[1:] = rfreqs[1:] ** (-exponent / 2.0)        # amplitude ∝ f^(-exp/2) → power ∝ f^(-exp)
    t = np.arange(N_T) / FS
    X = np.zeros((N_EP, 1, N_T))
    for e in range(N_EP):
        x = np.fft.irfft(np.fft.rfft(rng.normal(size=N_T)) * shape, N_T)
        x = x / x.std()
        if theta_amp:
            x = x + theta_amp * np.sin(2 * np.pi * theta_f * t + rng.uniform(0, 2 * np.pi))
        X[e, 0] = x
    return X


def _epochs(*channels):
    """Stack single-channel arrays from `_colored` into one EpochsArray."""
    X = np.concatenate(channels, axis=1)
    info = mne.create_info([f"ch{i}" for i in range(X.shape[1])], FS, "seeg")
    return mne.EpochsArray(X, info, tmin=-1.5, verbose="ERROR")


def _row(df, ch, band, trial=-1):
    r = df[(df.channel == ch) & (df.band == band) & (df.trial == trial)]
    assert len(r) == 1, f"expected 1 row for {ch}/{band}/trial={trial}, got {len(r)}"
    return r.iloc[0]


def test_recovery_and_correction():
    """Clean theta channel recovers a peak; steep no-osc channel shows raw != periodic."""
    clean = _colored(1.0, theta_amp=0.7, seed=1)     # ch0: shallow 1/f + theta
    steep = _colored(2.2, theta_amp=0.0, seed=2)     # ch1: steep 1/f, NO oscillation
    ep = _epochs(clean, steep)
    df = aperiodic_corrected_band_power(
        ep, ["ch0", "ch1"], BANDS, fit_range=(2.0, 45.0), aperiodic_mode="fixed")

    c_theta = _row(df, "ch0", "theta")
    c_beta = _row(df, "ch0", "beta")
    s_theta = _row(df, "ch1", "theta")
    s_beta = _row(df, "ch1", "beta")

    # 1. clean channel: QC passes, exponent ~1, a strong theta peak, periodic theta > peakless beta
    assert bool(c_theta.qc_pass), f"clean fit failed QC (r2={c_theta.r_squared})"
    assert abs(c_theta.aperiodic_exponent - 1.0) < 0.6
    assert 4.0 <= c_theta.peak_cf <= 9.0 and c_theta.peak_pw > 0.4
    assert c_theta.periodic_band_power > c_beta.periodic_band_power   # theta peak vs peakless beta

    # 2. THE CORRECTION MATTERS: steep no-osc channel -> high raw theta but ~zero periodic theta
    assert s_theta.raw_band_power > s_beta.raw_band_power             # within-channel 1/f gradient (raw)
    assert abs(s_theta.periodic_band_power) < 0.15                    # no real oscillation left
    assert s_theta.aperiodic_fraction > 0.8                          # raw theta is mostly aperiodic
    # the correction separates a real oscillation (clean) from pure 1/f (steep)
    assert c_theta.periodic_band_power - s_theta.periodic_band_power > 0.15

    # 3. exponent ordering
    assert s_theta.aperiodic_exponent > c_theta.aperiodic_exponent + 0.4


def test_attrition_is_visible():
    """A degenerate (all-NaN) channel becomes a fit_ok=False / qc_pass=False row, no raise."""
    freqs = np.linspace(2.0, 45.0, 120)
    good = 10.0 ** (1.0 - 1.0 * np.log10(freqs))                      # clean 1/f
    good = good + 0.5 * np.exp(-0.5 * ((freqs - 6.0) / 1.0) ** 2)     # + theta bump
    psds = np.stack([np.tile(good, (8, 1)),
                     np.full((8, freqs.size), np.nan)], axis=1)       # (8 trials, 2 ch, n_freq)
    df = aperiodic_corrected_band_power(
        None, ["good", "bad"], BANDS, fit_range=(2.0, 45.0), psd=(psds, freqs))
    good_row = _row(df, "good", "theta")
    bad_row = _row(df, "bad", "theta")
    assert bool(good_row.fit_ok) and bool(good_row.qc_pass)
    assert (not bool(bad_row.fit_ok)) and (not bool(bad_row.qc_pass))
    assert bad_row.periodic_band_power != bad_row.periodic_band_power  # NaN, not crash


def test_per_trial_rows():
    """per_trial=True emits n_trials finite per-trial periodic-theta rows per channel."""
    ep = _epochs(_colored(1.2, theta_amp=0.7, seed=4))
    df = aperiodic_corrected_band_power(
        ep, ["ch0"], BANDS, fit_range=(2.0, 45.0), per_trial=True)
    pt = df[(df.channel == "ch0") & (df.band == "theta") & (df.trial >= 0)]
    assert len(pt) == N_EP
    assert np.isfinite(pt.periodic_band_power.to_numpy()).all()
    # per-trial periodic theta scatters around the summary value
    summ = _row(df, "ch0", "theta").periodic_band_power
    assert abs(pt.periodic_band_power.mean() - summ) < 0.3


def _colored_varying_exp(base=1.0, spread=0.8, seed=0):
    """N_EP trials, each 1/f noise with its OWN exponent (base + spread*z) + a fixed 6 Hz tone.
    Returns (X[N_EP,1,N_T], imposed_exponents[N_EP])."""
    rng = np.random.default_rng(seed)
    rfreqs = np.fft.rfftfreq(N_T, 1.0 / FS)
    t = np.arange(N_T) / FS
    exps = base + spread * rng.standard_normal(N_EP)
    X = np.zeros((N_EP, 1, N_T))
    for e in range(N_EP):
        shape = np.ones_like(rfreqs)
        shape[1:] = rfreqs[1:] ** (-exps[e] / 2.0)
        x = np.fft.irfft(np.fft.rfft(rng.normal(size=N_T)) * shape, N_T)
        X[e, 0] = x / x.std() + 0.5 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
    return X, exps


def test_per_trial_fooof_tracks_varying_exponent():
    """Per-trial FOOOF recovers a per-trial-VARYING aperiodic exponent (the aperiodic-vs-behaviour leg)."""
    X, exps = _colored_varying_exp(seed=7)
    ep = mne.EpochsArray(X, mne.create_info(["ch0"], FS, "seeg"), tmin=-1.5, verbose="ERROR")
    df = per_trial_fooof_band_power(ep, ["ch0"], BANDS, fit_range=(2.0, 45.0), aperiodic_mode="fixed")
    th = df[df.band == "theta"].sort_values("trial")
    assert len(th) == N_EP
    assert th.qc_pass.mean() > 0.6                          # most single-trial fits pass QC
    rec = th.aperiodic_exponent.to_numpy()
    m = np.isfinite(rec)
    assert _corr(rec[m], exps[m]) > 0.5                     # per-trial exponent tracks the imposed 1/f
    assert np.isfinite(th[th.fit_ok].periodic_band_power).all()  # periodic theta defined for good fits


if __name__ == "__main__":
    test_recovery_and_correction()
    test_attrition_is_visible()
    test_per_trial_rows()
    test_per_trial_fooof_tracks_varying_exponent()
    print("all aperiodic-band-power tests passed")
