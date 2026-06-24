"""Synthetic gate for the BOSC/eBOSC P-episode (rhythmic abundance) estimator.

Grounding (Preston, Smith & Voytek 2026, Nat Hum Behav): band power != oscillation
(the Fourier fallacy). A fixed-band filter returns power whether or not a genuine,
temporally sustained rhythm is present -- aperiodic 1/f activity passes a theta
filter just fine. Before claiming the low-theta directed flow rides a real
oscillation, we need a detector that fires ONLY for a sustained rhythm above the
aperiodic background. P-episode is that detector; this is its gate.

Verifies, on signals where the ground truth is controlled:
1. return contract (keys, per-freq shapes).
2. pure 1/f (pink) noise -> P-episode stays near the false-positive floor at every
   frequency (no sustained rhythm is hallucinated from aperiodic power).
3. a strong, full-duration 4 Hz rhythm in pink noise -> P-episode ~1 at 4 Hz and
   near-floor away from it (localized, abundant).
4. a 4 Hz rhythm present for ~half the time -> P-episode at 4 Hz is intermediate,
   clearly above the pure-noise floor and clearly below the full-duration case
   (abundance tracks how long the rhythm is actually present).
"""
from __future__ import annotations

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore[assignment]

from LFPAnalysis.oscillation_utils import pepisode_spectrum

FS = 500.0
DUR = 16.0                       # s per trial -- long so duration/edge never limits
N = int(FS * DUR)
N_TRIAL = 6
FREQS = np.arange(2.0, 20.01, 0.5)
PAD_S = 2.0                      # trim wavelet edge artifacts (>= 4 Hz wavelet half-width)
EXCLUDE = (2.0, 8.0)            # keep the (putative) theta rhythm out of its own background


def _pink(n, fs, rng, exponent=1.0):
    """Unit-variance 1/f^exponent noise via spectral shaping of white noise."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f[0] = f[1]
    x = np.fft.irfft(spec / (f ** (exponent / 2.0)), n)
    return x / np.std(x)


def _idx(f):
    return int(np.argmin(np.abs(FREQS - f)))


def _trials(rng, amp=0.0, frac=0.0):
    """N_TRIAL pink-noise trials, each with an optional 4 Hz sine over the first
    `frac` of the trial at amplitude `amp`."""
    t = np.arange(N) / FS
    rhythm_mask = t < (frac * DUR)
    out = np.empty((N_TRIAL, N))
    for k in range(N_TRIAL):
        x = _pink(N, FS, rng)
        if amp > 0 and frac > 0:
            x = x + amp * np.sin(2 * np.pi * 4.0 * t) * rhythm_mask
        out[k] = x
    return out


def test_return_contract():
    rng = np.random.default_rng(0)
    r = pepisode_spectrum(_trials(rng), FS, FREQS, exclude_peak=EXCLUDE, pad_s=PAD_S)
    for k in ("freqs", "pepisode", "bg_mp", "pt", "slope", "intercept",
              "n_trials", "n_clean_samples"):
        assert k in r, f"missing key {k}"
    assert r["pepisode"].shape == FREQS.shape
    assert r["pt"].shape == FREQS.shape
    assert r["n_trials"] == N_TRIAL
    assert np.all((r["pepisode"] >= 0) & (r["pepisode"] <= 1))


def test_pure_noise_low_pepisode():
    rng = np.random.default_rng(1)
    r = pepisode_spectrum(_trials(rng), FS, FREQS, exclude_peak=EXCLUDE, pad_s=PAD_S)
    # no sustained rhythm anywhere -> abundance stays near the false-positive floor
    assert r["pepisode"].max() < 0.25, f"pure 1/f hallucinated a rhythm (max={r['pepisode'].max():.3f})"
    assert r["pepisode"][_idx(4.0)] < 0.25


def test_sustained_rhythm_detected_and_localized():
    rng = np.random.default_rng(2)
    r = pepisode_spectrum(_trials(rng, amp=2.5, frac=1.0), FS, FREQS,
                          exclude_peak=EXCLUDE, pad_s=PAD_S)
    p4 = r["pepisode"][_idx(4.0)]
    p11 = r["pepisode"][_idx(11.0)]
    assert p4 > 0.6, f"sustained 4 Hz rhythm under-detected (P-episode={p4:.3f})"
    assert p4 - p11 > 0.4, f"rhythm not localized to 4 Hz (4Hz={p4:.3f}, 11Hz={p11:.3f})"


def test_abundance_tracks_presence():
    rng = np.random.default_rng(3)
    full = pepisode_spectrum(_trials(rng, amp=2.5, frac=1.0), FS, FREQS,
                             exclude_peak=EXCLUDE, pad_s=PAD_S)["pepisode"][_idx(4.0)]
    rng = np.random.default_rng(4)
    half = pepisode_spectrum(_trials(rng, amp=2.5, frac=0.5), FS, FREQS,
                             exclude_peak=EXCLUDE, pad_s=PAD_S)["pepisode"][_idx(4.0)]
    rng = np.random.default_rng(5)
    none = pepisode_spectrum(_trials(rng), FS, FREQS,
                             exclude_peak=EXCLUDE, pad_s=PAD_S)["pepisode"][_idx(4.0)]
    assert none < half < full, f"abundance should track presence (none={none:.3f}, half={half:.3f}, full={full:.3f})"
    assert 0.2 < half < 0.8, f"half-present rhythm abundance off ({half:.3f})"


def test_window_restriction_buffers_outside_counts_inside():
    # rhythm present only in the first half; detection uses the full timecourse as
    # buffer but abundance is read out only within `win`.
    rng = np.random.default_rng(6)
    data = _trials(rng, amp=2.5, frac=0.5)
    t = np.arange(N) / FS
    early = pepisode_spectrum(data, FS, FREQS, exclude_peak=EXCLUDE,
                              times=t, win=(1.0, 7.0))["pepisode"][_idx(4.0)]
    late = pepisode_spectrum(data, FS, FREQS, exclude_peak=EXCLUDE,
                             times=t, win=(9.0, 15.0))["pepisode"][_idx(4.0)]
    assert early > 0.6, f"rhythm window should be abundant ({early:.3f})"
    assert late < 0.25, f"post-rhythm window should be near floor ({late:.3f})"


if __name__ == "__main__":
    test_return_contract()
    test_pure_noise_low_pepisode()
    test_sustained_rhythm_detected_and_localized()
    test_abundance_tracks_presence()
    test_window_restriction_buffers_outside_counts_inside()
    print("all P-episode tests passed (synthetic oscillation-vs-1/f gate)")
