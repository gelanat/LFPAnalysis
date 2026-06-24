"""Tests for subject-specific (individualized) theta-band detection.

Grounding (Preston, Smith & Voytek 2026, Nat Hum Behav): band power != oscillation
(Fourier fallacy) -> only treat a band as oscillatory where a real peak sits above the
1/f, and individualize the band to the detected peak (Suthana-style patient-specific theta).

Verifies:
1. RECOVERY: a region of channels carrying a 6 Hz tone gets theta_cf ~6 with band cf+/-2;
   a region carrying a 4 Hz (slow) tone gets a lower cf with the band clamped at >=2 Hz.
2. NO-PEAK FALLBACK: a region of steep, no-oscillation channels (no detectable peak) falls
   back to (2, 8) with fallback_used=True and theta_cf NaN.
3. min_with_peak: a region with only ONE peak channel (< the floor) also falls back.
4. KEEPS ALL CHANNELS: n_chan counts every channel in the region regardless of peak presence.
5. individualized_band / pair_band rule + fallback-ladder unit behaviour.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore[assignment]

import mne

from LFPAnalysis.analysis_utils import (
    detect_region_peak_band,
    individualized_band,
    pair_band,
)

FS = 500.0
N_EP = 40
N_T = 1500          # 3.0 s at 500 Hz


def _colored(exponent, *, theta_amp=0.0, theta_f=6.0, seed=0):
    """One channel of 1/f noise (power ∝ f^-exponent) + optional theta tone at theta_f."""
    rng = np.random.default_rng(seed)
    rfreqs = np.fft.rfftfreq(N_T, 1.0 / FS)
    shape = np.ones_like(rfreqs)
    shape[1:] = rfreqs[1:] ** (-exponent / 2.0)
    t = np.arange(N_T) / FS
    X = np.zeros((N_EP, 1, N_T))
    for e in range(N_EP):
        x = np.fft.irfft(np.fft.rfft(rng.normal(size=N_T)) * shape, N_T)
        x = x / x.std()
        if theta_amp:
            x = x + theta_amp * np.sin(2 * np.pi * theta_f * t + rng.uniform(0, 2 * np.pi))
        X[e, 0] = x
    return X


def _build():
    """Build a multi-region EpochsArray + reg_of; return (epochs, picks, reg_of)."""
    specs = [
        # (channel, region, exponent, theta_amp, theta_f)
        ("h0", "HPC", 1.0, 0.7, 6.0), ("h1", "HPC", 1.1, 0.7, 6.0), ("h2", "HPC", 0.9, 0.7, 6.0),
        ("a0", "AMY", 1.0, 0.7, 4.0), ("a1", "AMY", 1.1, 0.7, 4.0), ("a2", "AMY", 0.9, 0.7, 4.0),
        ("f0", "FLAT", 2.3, 0.0, 6.0), ("f1", "FLAT", 2.2, 0.0, 6.0), ("f2", "FLAT", 2.4, 0.0, 6.0),
        ("s0", "SOLO", 1.0, 0.7, 7.0), ("s1", "SOLO", 2.3, 0.0, 6.0),   # only 1 peak -> < floor
    ]
    chans = [_colored(exp, theta_amp=amp, theta_f=tf, seed=i)
             for i, (_, _, exp, amp, tf) in enumerate(specs)]
    X = np.concatenate(chans, axis=1)
    names = [c for c, *_ in specs]
    info = mne.create_info(names, FS, "seeg")
    ep = mne.EpochsArray(X, info, tmin=-1.5, verbose="ERROR")
    reg_of = {c: r for c, r, *_ in specs}
    return ep, names, reg_of


def test_detect_region_peak_band_recovers_and_falls_back():
    ep, picks, reg_of = _build()
    res = detect_region_peak_band(
        ep, picks, reg_of, ["HPC", "AMY", "FLAT", "SOLO"],
        fit_range=(2.0, 45.0), aperiodic_mode="fixed", search_band=(2.0, 12.0),
        min_with_peak=2,
    ).set_index("region")

    # 1. HPC: 6 Hz peak recovered, band = cf +/- 2, genuine (no fallback)
    hpc = res.loc["HPC"]
    assert 5.0 <= hpc.theta_cf <= 7.0, f"HPC cf={hpc.theta_cf}"
    assert not bool(hpc.fallback_used)
    assert abs(hpc.band_lo - (hpc.theta_cf - 2.0)) < 1e-6
    assert abs(hpc.band_hi - (hpc.theta_cf + 2.0)) < 1e-6
    assert int(hpc.n_chan) == 3 and int(hpc.n_with_peak) >= 2

    # 1b. AMY: slow (4 Hz) peak -> lower cf, band clamped at >=2 Hz
    amy = res.loc["AMY"]
    assert 3.0 <= amy.theta_cf <= 5.0, f"AMY cf={amy.theta_cf}"
    assert amy.band_lo >= 2.0 and not bool(amy.fallback_used)

    # 2. FLAT: no peak anywhere -> fallback (2, 8), theta_cf NaN, but all channels kept
    flat = res.loc["FLAT"]
    assert bool(flat.fallback_used)
    assert (flat.band_lo, flat.band_hi) == (2.0, 8.0)
    assert flat.theta_cf != flat.theta_cf            # NaN
    assert int(flat.n_chan) == 3                     # 4. keeps all channels

    # 3. SOLO: only one peak channel (< min_with_peak) -> fallback
    solo = res.loc["SOLO"]
    assert bool(solo.fallback_used) and (solo.band_lo, solo.band_hi) == (2.0, 8.0)
    assert int(solo.n_chan) == 2


def test_individualized_band_rules():
    assert individualized_band(6.0) == (4.0, 8.0)
    assert individualized_band(3.0) == (2.0, 5.0)              # clamped at min_lo=2
    assert individualized_band(11.5) == (9.5, 12.0)           # clamped at max_hi=12
    assert individualized_band(np.nan) == (2.0, 8.0)          # no-peak fallback
    assert individualized_band(None) == (2.0, 8.0)
    # half-bandwidth rule uses the supplied width
    lo, hi = individualized_band(6.0, rule="cf_pm_halfbw", half_width=1.0)
    assert (lo, hi) == (5.0, 7.0)


def test_pair_band_rules():
    s, t = (4.0, 8.0), (5.0, 9.0)
    # undirected union = span both
    assert pair_band(s, t)[:2] == (4.0, 9.0)
    assert pair_band(s, t)[2] == "union"
    # mean mode = band around mean centre with mean half-width
    lo, hi, rule = pair_band(s, t, mode="mean")
    assert rule == "mean" and abs(((lo + hi) / 2) - 6.5) < 1e-6
    # directed -> source band
    assert pair_band(s, t, directed=True)[:2] == (4.0, 8.0)
    # fallback ladder: source has no peak -> use target (undirected)
    assert pair_band(s, t, src_has_peak=False)[:2] == (5.0, 9.0)
    # directed but source peak missing -> target as fallback
    lo, hi, rule = pair_band(s, t, directed=True, src_has_peak=False)
    assert (lo, hi) == (5.0, 9.0) and rule == "tgt_peak_fallback"
    # neither has a peak -> default
    assert pair_band(s, t, src_has_peak=False, tgt_has_peak=False) == (2.0, 8.0, "fallback_default")


if __name__ == "__main__":
    test_detect_region_peak_band_recovers_and_falls_back()
    test_individualized_band_rules()
    test_pair_band_rules()
    print("all peak-band tests passed")
