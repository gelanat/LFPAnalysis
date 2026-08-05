"""Tests for the per-trial amplitude-envelope correlation estimator.

``compute_aec_per_trial`` keeps the trial axis (returns ``(n_epochs, n_pairs)``) so per-trial
amplitude coupling can be regressed against trial-level behaviour, following the same contract as
``compute_psi_per_trial`` / ``compute_pte_per_trial``. Verifies:
1. Shape is ``(n_epochs, n_pairs)`` and finite on clean synthetic pairs; broadband (n_subbands=8)
   runs and stays finite.
2. Leakage rejection (the crux): two channels sharing a zero-lag common signal + envelope give a
   HIGH raw AEC but a NEAR-ZERO orthogonalised AEC (Hipp-2012 removes the volume-conduction term).
3. Recovery: two channels with independent carriers but a genuinely shared amplitude envelope give a
   POSITIVE orthogonalised AEC (the real amplitude coupling survives orthogonalisation).
4. tmin/tmax window slicing keeps the trial count; a zero-variance channel -> NaN, no crash.
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
from mne_connectivity import seed_target_indices

from LFPAnalysis import oscillation_utils as ou

FS = 500.0
N_EP = 40
N_T = 1500                     # 3 s at 500 Hz
BAND = (6.0, 14.0)             # narrowband math tests, carrier 10 Hz
CARRIER = 10.0
T = np.arange(N_T) / FS


def _trial_envelope(rng):
    """A slow (1 Hz) amplitude envelope with a random per-trial phase; strictly positive."""
    return 1.0 + 0.8 * np.sin(2 * np.pi * 1.0 * T + rng.uniform(0, 2 * np.pi))


def _shared_signal_epochs(phase_offset: bool, noise: float = 0.1, seed: int = 0):
    """Two channels sharing a per-trial amplitude envelope ``E(t)``.

    ``phase_offset=False`` -> identical carrier phase (zero-lag common signal = pure leakage).
    ``phase_offset=True``  -> independent random per-trial carrier phases (genuine envelope
    coupling with no consistent phase relationship).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N_EP, 2, N_T))
    for e in range(N_EP):
        env = _trial_envelope(rng)
        px = rng.uniform(0, 2 * np.pi)
        py = px if not phase_offset else rng.uniform(0, 2 * np.pi)
        X[e, 0] = env * np.cos(2 * np.pi * CARRIER * T + px) + rng.normal(0, noise, N_T)
        X[e, 1] = env * np.cos(2 * np.pi * CARRIER * T + py) + rng.normal(0, noise, N_T)
    info = mne.create_info(["s", "t"], FS, "seeg")
    return mne.EpochsArray(X, info, tmin=-1.0, verbose="ERROR")


def test_aec_shape_and_finite():
    ep = _shared_signal_epochs(phase_offset=True)
    idx = seed_target_indices([0], [1])
    r_orth = ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=True)
    r_raw = ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=False)
    assert r_orth.shape == r_raw.shape == (N_EP, 1)
    assert np.isfinite(r_orth).mean() > 0.9
    assert np.isfinite(r_raw).mean() > 0.9


def test_aec_leakage_rejection():
    """Zero-lag shared signal: raw AEC high, orthogonalised AEC near zero."""
    ep = _shared_signal_epochs(phase_offset=False)
    idx = seed_target_indices([0], [1])
    raw = np.nanmean(ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=False))
    orth = np.nanmean(ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=True))
    assert raw > 0.5                 # shared envelope -> strong raw amplitude correlation
    assert abs(orth) < 0.25          # orthogonalisation removes the zero-lag shared component
    assert orth < raw - 0.3          # decisively separated


def test_aec_recovery_of_genuine_coupling():
    """Independent carriers, shared envelope: orthogonalised AEC stays positive."""
    ep = _shared_signal_epochs(phase_offset=True)
    idx = seed_target_indices([0], [1])
    orth = np.nanmean(ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=True))
    orth_pos = np.nanmean(
        ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, orthogonalize=True) > 0)
    assert orth > 0.2                # genuine amplitude comodulation survives orthogonalisation
    assert orth_pos > 0.7            # and is sign-consistent across trials


def test_aec_broadband_multipair():
    """n_subbands=8 (broadband) runs over multiple pairs and stays finite."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (N_EP, 3, N_T))
    info = mne.create_info(["a", "b", "c"], FS, "seeg")
    ep = mne.EpochsArray(X, info, tmin=-1.0, verbose="ERROR")
    idx = seed_target_indices([0], [1, 2])   # cross-product -> 2 pairs (0-1, 0-2)
    assert len(idx[0]) == 2
    out = ou.compute_aec_per_trial(ep, idx, band=(70.0, 150.0), n_subbands=8, orthogonalize=True)
    assert out.shape == (N_EP, 2)
    assert np.isfinite(out).mean() > 0.9


def test_aec_window_and_nan_safety():
    """tmin/tmax keep the trial count; a zero-variance channel -> NaN for its pair, no crash."""
    ep = _shared_signal_epochs(phase_offset=True)
    idx = seed_target_indices([0], [1])
    full = ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1)
    win = ou.compute_aec_per_trial(ep, idx, band=BAND, n_subbands=1, tmin=-0.8, tmax=-0.3)
    assert full.shape == win.shape == (N_EP, 1)
    assert np.isfinite(win).mean() > 0.9

    # dead (all-zero) channel -> zero-variance envelope -> NaN, no exception
    X = ep.get_data(copy=True)
    X[:, 1, :] = 0.0
    ep2 = mne.EpochsArray(X, ep.info, tmin=-1.0, verbose="ERROR")
    out = ou.compute_aec_per_trial(ep2, idx, band=BAND, n_subbands=1)
    assert out.shape == (N_EP, 1)
    assert np.isnan(out).all()        # pair involves the dead channel -> all NaN, but no crash


if __name__ == "__main__":
    test_aec_shape_and_finite()
    test_aec_leakage_rejection()
    test_aec_recovery_of_genuine_coupling()
    test_aec_broadband_multipair()
    test_aec_window_and_nan_safety()
    print("all AEC per-trial estimator tests passed")
