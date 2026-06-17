"""Tests for the per-trial directed estimators added for trial-resolved connectivity.

`compute_psi_per_trial` and `compute_pte_per_trial` keep the trial axis (return
``(n_epochs, n_pairs)``) so per-trial connectivity can be regressed against trial-level
behavior. Verifies:
1. Shape is ``(n_epochs, n_pairs)`` and finite on a clean synthetic pair.
2. Direction: when the seed channel LEADS the target (target = delayed copy of seed),
   PSI is positive (mne convention: seed leads -> +) and net PTE is positive
   (PTE_{seed->target} - PTE_{target->seed} > 0), with the PTE delay matched to the lag.
3. Reversing seed/target flips the sign of both.
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
N_EP = 30
N_T = 1500
THETA = (2.0, 8.0)
CWT_FREQS = np.logspace(np.log10(2), np.log10(200), 30)
CWT_NCYC = np.floor(np.logspace(np.log10(3), np.log10(10), 30))


def _leader_follower(lag: int, seed: int = 1) -> mne.EpochsArray:
    """Two-channel epochs where ch0 (seed) leads ch1 (target) by `lag` samples (6 Hz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / FS
    X = np.zeros((N_EP, 2, N_T))
    for e in range(N_EP):
        ph = 2 * np.pi * 6.0 * t + 0.15 * rng.normal(0, 1, N_T).cumsum()
        base = np.sin(ph)
        X[e, 0] = base + rng.normal(0, 0.2, N_T)              # leader
        X[e, 1] = np.roll(base, lag) + rng.normal(0, 0.2, N_T)  # follower
    info = mne.create_info(["s", "t"], FS, "seeg")
    return mne.EpochsArray(X, info, tmin=-1.0, verbose="ERROR")


def test_psi_per_trial_shape_and_direction():
    ep = _leader_follower(lag=50)
    fwd = seed_target_indices([0], [1])   # seed=leader -> target=follower
    rev = seed_target_indices([1], [0])
    psi_fwd = ou.compute_psi_per_trial(ep, fwd, band=THETA, freqs=CWT_FREQS, n_cycles=CWT_NCYC)
    psi_rev = ou.compute_psi_per_trial(ep, rev, band=THETA, freqs=CWT_FREQS, n_cycles=CWT_NCYC)
    assert psi_fwd.shape == (N_EP, 1)
    assert np.isfinite(psi_fwd).all()
    assert np.nanmean(psi_fwd) > 0       # seed leads -> positive
    assert np.nanmean(psi_rev) < 0       # reversed -> negative
    assert np.nanmean(psi_fwd > 0) >= 0.7


def test_pte_per_trial_shape_and_direction():
    lag = 50
    ep = _leader_follower(lag=lag)
    fwd = seed_target_indices([0], [1])
    rev = seed_target_indices([1], [0])
    # PTE delay matched to the signal lag (the timescale of the directed coupling).
    pte_fwd = ou.compute_pte_per_trial(ep, fwd, band=THETA, delay=lag, net=True)
    pte_rev = ou.compute_pte_per_trial(ep, rev, band=THETA, delay=lag, net=True)
    assert pte_fwd.shape == (N_EP, 1)
    assert np.isfinite(pte_fwd).all()
    assert np.nanmean(pte_fwd) > 0       # net PTE positive when seed leads
    assert np.nanmean(pte_rev) < 0       # reversed -> negative
    assert np.nanmean(pte_fwd > 0) >= 0.7


def test_psi_per_trial_window_slicing():
    """tmin/tmax restrict the analysis window without changing the trial count."""
    ep = _leader_follower(lag=50)
    idx = seed_target_indices([0], [1])
    full = ou.compute_psi_per_trial(ep, idx, band=THETA, freqs=CWT_FREQS, n_cycles=CWT_NCYC)
    win = ou.compute_psi_per_trial(ep, idx, band=THETA, freqs=CWT_FREQS, n_cycles=CWT_NCYC,
                                   tmin=-0.8, tmax=-0.3)
    assert full.shape == win.shape == (N_EP, 1)
    assert np.isfinite(win).all()


if __name__ == "__main__":
    test_psi_per_trial_shape_and_direction()
    test_pte_per_trial_shape_and_direction()
    test_psi_per_trial_window_slicing()
    print("all per-trial estimator tests passed")
