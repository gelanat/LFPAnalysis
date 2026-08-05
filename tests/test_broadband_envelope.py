"""Tests for the normalized-broadband HFA substrate (`_broadband_envelope`).

The HFA reducers (`per_trial_band_power`, `per_trial_band_power_timeresolved`) gained
`n_subbands`/`subband_norm`. With `n_subbands=1` (default) they must reproduce the old
single-band path exactly; with `n_subbands>1` they split the band, normalize each
sub-band, and average -- the 1/f-robust broadband-HFA estimate. Verifies:

1. REGRESSION: `n_subbands=1` is byte-identical to the single-band `_band_envelope`
   formula, and the default equals an explicit `n_subbands=1` (frozen-pipeline guard).
2. SHAPE: broadband output matches single-band shape and is finite.
3. 1/f EQUALIZATION: with a dominant low-frequency tone (75 Hz, large amplitude) and a
   weak high-frequency tone (145 Hz), a single 70-150 Hz envelope is dominated by the
   low tone and barely tracks the high tone; the normalized broadband recovers the high
   tone far better -- the whole point of the method.
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

from LFPAnalysis.representational_utils import (
    _band_envelope,
    per_trial_band_power,
    per_trial_band_power_timeresolved,
)

FS = 500.0
N_EP = 60
N_T = 1500          # 3.0 s at 500 Hz
BAND = (70.0, 150.0)
TMIN, TMAX = -0.9, 1.9


def _white(seed: int = 0) -> mne.EpochsArray:
    """Two channels of band-limited-ish noise (for the regression/shape checks)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N_EP, 2, N_T))
    info = mne.create_info(["a", "b"], FS, "seeg")
    return mne.EpochsArray(X, info, tmin=-1.0, verbose="ERROR")


def _two_tone(seed: int = 0):
    """One channel with a DOMINANT 75 Hz tone (amp x5) + WEAK 145 Hz tone, both with
    independent per-trial amplitudes. Returns (epochs, a_low, a_high)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / FS
    a_low = rng.uniform(0.5, 1.5, N_EP)    # drives 75 Hz (low end of HFA band)
    a_high = rng.uniform(0.5, 1.5, N_EP)   # drives 145 Hz (high end) -- independent
    X = np.zeros((N_EP, 1, N_T))
    for e in range(N_EP):
        X[e, 0] = (5.0 * a_low[e] * np.sin(2 * np.pi * 75.0 * t)
                   + 1.0 * a_high[e] * np.sin(2 * np.pi * 145.0 * t)
                   + rng.normal(0, 0.3, N_T))
    info = mne.create_info(["ch0"], FS, "seeg")
    return mne.EpochsArray(X, info, tmin=-1.0, verbose="ERROR"), a_low, a_high


def _corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def test_single_band_is_byte_identical():
    """n_subbands=1 reproduces the single-band _band_envelope formula exactly."""
    ep = _white()
    picks = ["a", "b"]
    env, times, _ = _band_envelope(ep, picks, BAND, kind="power")
    mask = (times >= TMIN) & (times <= TMAX)
    ref = np.log10(env[:, :, mask].mean(axis=2) + np.finfo(float).tiny)

    default = per_trial_band_power(ep, picks, BAND, TMIN, TMAX)                 # default n_subbands=1
    explicit1 = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, n_subbands=1)
    assert np.array_equal(default, ref)
    assert np.array_equal(default, explicit1)


def test_timeresolved_default_unchanged():
    ep = _white(1)
    picks = ["a", "b"]
    d, c0 = per_trial_band_power_timeresolved(ep, picks, BAND, tmin=TMIN, tmax=TMAX)
    e1, c1 = per_trial_band_power_timeresolved(ep, picks, BAND, tmin=TMIN, tmax=TMAX, n_subbands=1)
    assert np.array_equal(d, e1) and np.array_equal(c0, c1)


def test_broadband_shape_and_finite():
    ep = _white(2)
    picks = ["a", "b"]
    sb = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, n_subbands=1)
    bb = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, n_subbands=8, subband_norm="zscore")
    assert bb.shape == sb.shape
    assert np.isfinite(bb).all()
    # time-resolved broadband too
    Xt, centers = per_trial_band_power_timeresolved(
        ep, picks, BAND, window_s=0.5, step_s=0.1, tmin=TMIN, tmax=TMAX, n_subbands=8)
    assert Xt.shape[0] == N_EP and Xt.shape[1] == 2 and Xt.shape[2] == centers.size
    assert np.isfinite(Xt).all()


def test_broadband_equalizes_1_over_f():
    """Single-band envelope is dominated by the loud low tone; normalized broadband
    recovers the weak high tone much better."""
    ep, a_low, a_high = _two_tone(seed=3)
    picks = ["ch0"]
    sb = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, n_subbands=1)[:, 0]
    bb = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, n_subbands=8, subband_norm="zscore")[:, 0]

    sb_low, sb_high = _corr(sb, a_low), _corr(sb, a_high)
    bb_high = _corr(bb, a_high)

    # single-band tracks the dominant low tone, largely misses the high tone
    assert sb_low > 0.6
    assert sb_high < 0.4
    # broadband recovers the high tone better than the single band does
    assert bb_high > sb_high
    assert bb_high > 0.2


# ---------------------------------------------------------------------------
# norm_fit_idx -- inductive (train-only) sub-band normalization
#
# The published behaviour pools the normalization statistics over ALL trials. That is
# label-blind, so it cannot manufacture a within-subject decoding effect, but it IS
# transductive across a train/test split: held-out trials contribute to the per-channel
# scale, and because sub-band averaging happens AFTER normalization the scaling is not
# undoable by a downstream StandardScaler. `norm_fit_idx` restricts the estimation trials
# so a cross-condition-generalization claim can be made inductively.
# ---------------------------------------------------------------------------


def test_norm_fit_idx_default_matches_all_trials():
    """FROZEN-PIPELINE GUARD: None (default) == explicitly passing every trial."""
    ep = _white(4)
    picks = ["a", "b"]
    kw = dict(n_subbands=8, subband_norm="zscore")
    default = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, **kw)
    all_idx = per_trial_band_power(
        ep, picks, BAND, TMIN, TMAX, norm_fit_idx=np.arange(N_EP), **kw)
    assert np.array_equal(default, all_idx)


def test_norm_fit_idx_is_inductive():
    """THE LOAD-BEARING TEST: fitting the norm on a train subset gives those trials
    exactly the features they would have had if the test trials had never been seen.

    Band-pass + Hilbert is per-trial (filtfilt runs along the time axis), so the only
    channel that could carry test-trial information into a train-trial feature is the
    normalization scalar. If train features computed from the FULL epoch set with
    norm_fit_idx=train equal train features computed from epochs[train] alone, that
    channel is closed.
    """
    ep = _white(5)
    picks = ["a", "b"]
    kw = dict(n_subbands=8, subband_norm="zscore")
    rng = np.random.default_rng(0)
    train = np.sort(rng.choice(N_EP, size=40, replace=False))

    full = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, norm_fit_idx=train, **kw)
    train_only = per_trial_band_power(ep[train], picks, BAND, TMIN, TMAX, **kw)
    assert np.allclose(full[train], train_only, rtol=1e-10, atol=1e-12)

    # and the transform is still APPLIED to the held-out trials (they are not dropped)
    assert full.shape == (N_EP, len(picks))
    assert np.isfinite(full).all()

    # ... while differing from the transductive (pooled-norm) version, i.e. it does something
    pooled = per_trial_band_power(ep, picks, BAND, TMIN, TMAX, **kw)
    assert not np.allclose(full, pooled)


def test_norm_fit_idx_bool_mask_equals_integer_index():
    ep = _white(6)
    picks = ["a", "b"]
    kw = dict(n_subbands=8, subband_norm="mean")
    train = np.arange(0, N_EP, 2)
    mask = np.zeros(N_EP, dtype=bool)
    mask[train] = True
    assert np.array_equal(
        per_trial_band_power(ep, picks, BAND, TMIN, TMAX, norm_fit_idx=train, **kw),
        per_trial_band_power(ep, picks, BAND, TMIN, TMAX, norm_fit_idx=mask, **kw),
    )


def test_norm_fit_idx_rejects_silent_noops_and_bad_input():
    """A no-op must raise, not be honoured silently -- the single-band path has no
    normalization step to restrict."""
    ep = _white(7)
    picks = ["a"]
    for bad, kw in [
        (np.arange(10), dict(n_subbands=1)),                        # nothing to restrict
        (np.array([], dtype=int), dict(n_subbands=8)),              # selects no trials
        (np.array([N_EP]), dict(n_subbands=8)),                     # out of range
        (np.zeros(3, dtype=bool), dict(n_subbands=8)),              # wrong-length bool mask
    ]:
        try:
            per_trial_band_power(ep, picks, BAND, TMIN, TMAX, norm_fit_idx=bad, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for norm_fit_idx={bad!r}, kw={kw}")


if __name__ == "__main__":
    test_single_band_is_byte_identical()
    test_timeresolved_default_unchanged()
    test_broadband_shape_and_finite()
    test_broadband_equalizes_1_over_f()
    test_norm_fit_idx_default_matches_all_trials()
    test_norm_fit_idx_is_inductive()
    test_norm_fit_idx_bool_mask_equals_integer_index()
    test_norm_fit_idx_rejects_silent_noops_and_bad_input()
    print("all broadband-envelope tests passed")
