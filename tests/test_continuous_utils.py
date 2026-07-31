"""Tests for `LFPAnalysis.continuous_utils`.

The load-bearing test is `test_matches_epoched_estimator_on_a_slice`: the continuous
envelope must equal the epoched one on the same samples, because the whole point of the
continuous-vs-trial-locked comparison it exists to serve is that the *estimator* is held
fixed and only the *time coverage* varies.
"""
from __future__ import annotations

import numpy as np
import pytest

from LFPAnalysis.continuous_utils import (
    block_cv_indices,
    circular_shift_offsets,
    continuous_band_envelope,
    decimate_envelope,
    iaaft_surrogate,
)

SFREQ = 500.0


def _pink(n_ch: int, n_t: int, seed: int = 0) -> np.ndarray:
    """1/f-ish multichannel noise -- realistic enough to exercise the sub-band norm."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_ch, n_t))
    f = np.fft.rfftfreq(n_t, d=1.0 / SFREQ)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])
    return np.fft.irfft(np.fft.rfft(w, axis=1) * scale[None, :], n=n_t, axis=1)


# ── continuous_band_envelope ──────────────────────────────────────────────────
def test_envelope_shape_and_finiteness():
    x = _pink(4, 5000)
    env = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=8)
    assert env.shape == x.shape
    assert np.all(np.isfinite(env))


def test_zscore_norm_is_centred_per_channel():
    """`subband_norm='zscore'` normalizes each channel x sub-band, so the pooled mean ~0."""
    x = _pink(3, 6000, seed=1)
    env = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=8,
                                   subband_norm="zscore")
    assert np.allclose(env.mean(axis=1), 0.0, atol=1e-8)


def test_mean_norm_stays_positive():
    x = _pink(3, 6000, seed=2)
    env = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=8,
                                   subband_norm="mean")
    assert np.all(env > 0)
    assert np.allclose(env.mean(axis=1), 1.0, atol=1e-8)


def test_valid_mask_excludes_artefact_from_the_scale():
    """A huge transient must not be allowed to set the scale every clean sample divides by."""
    x = _pink(2, 6000, seed=3)
    x[:, 3000:3050] += 200.0                        # a big artefact
    mask = np.ones(x.shape[1], dtype=bool)
    mask[2900:3150] = False

    env_masked = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=4,
                                          subband_norm="mean", valid_mask=mask)
    env_naive = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=4,
                                         subband_norm="mean")
    clean = np.ones(x.shape[1], dtype=bool)
    clean[2800:3250] = False
    # the naive scale is inflated by the artefact -> clean samples come out systematically
    # smaller than they should
    assert env_masked[:, clean].mean() > env_naive[:, clean].mean()


def test_matches_epoched_estimator_on_a_slice():
    """Continuous and epoched envelopes agree on the same samples (single 'trial').

    Same filter, same sub-band edges, same normalization -- and with one trial the epoched
    estimator's (trial, time) pooling reduces to the continuous estimator's time pooling.
    """
    mne = pytest.importorskip("mne")
    from LFPAnalysis.representational_utils import _broadband_envelope

    n_ch, n_t = 3, 4000
    x = _pink(n_ch, n_t, seed=4)
    names = [f"ch{i}" for i in range(n_ch)]
    info = mne.create_info(names, SFREQ, ch_types="seeg")
    ep = mne.EpochsArray(x[None, :, :], info, verbose="ERROR")

    env_ep, _, _ = _broadband_envelope(ep, names, (70.0, 150.0), n_subbands=8,
                                       subband_norm="zscore")
    env_co = continuous_band_envelope(x, SFREQ, (70.0, 150.0), n_subbands=8,
                                      subband_norm="zscore")
    assert np.allclose(env_ep[0], env_co, atol=1e-9)


def test_envelope_rejects_bad_input():
    with pytest.raises(ValueError):
        continuous_band_envelope(np.zeros(100), SFREQ, (70.0, 150.0))          # 1-D
    with pytest.raises(ValueError):
        continuous_band_envelope(np.zeros((2, 100)), SFREQ, (70.0, 150.0), n_subbands=0)
    with pytest.raises(ValueError):
        continuous_band_envelope(np.zeros((2, 100)), SFREQ, (70.0, 150.0),
                                 subband_norm="nope")


# ── decimate_envelope ─────────────────────────────────────────────────────────
def test_decimate_box_average_and_centres():
    env = np.tile(np.arange(1000, dtype=float), (2, 1))
    ds, t = decimate_envelope(env, SFREQ, 4.0)               # 125-sample bins
    assert ds.shape == (2, 8)
    assert np.allclose(ds[0, 0], np.arange(125).mean())
    assert np.allclose(t[0], 62.0 / SFREQ)


def test_decimate_nans_underpopulated_bins():
    env = np.ones((1, 1000))
    mask = np.ones(1000, dtype=bool)
    mask[:120] = False                                       # bin 0 keeps only 5/125
    ds, _ = decimate_envelope(env, SFREQ, 4.0, valid_mask=mask, min_valid_frac=0.5)
    assert np.isnan(ds[0, 0])
    assert np.all(np.isfinite(ds[0, 1:]))


# ── iaaft_surrogate ───────────────────────────────────────────────────────────
def test_iaaft_preserves_amplitude_distribution_exactly():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(2048))                 # random walk: skewed marginal
    y = iaaft_surrogate(x, rng=rng)
    assert np.allclose(np.sort(x), np.sort(y))               # exact, by construction


def test_iaaft_approximately_preserves_power_spectrum():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(2048))
    y = iaaft_surrogate(x, n_iter=200, rng=rng)
    px, py = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    # correlation of the log spectra: IAAFT trades exactness of one constraint for the
    # other, so this is close-but-not-equal by design
    r = np.corrcoef(np.log(px[1:] + 1e-12), np.log(py[1:] + 1e-12))[0, 1]
    assert r > 0.95


def test_iaaft_is_not_the_identity():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(1024))
    y = iaaft_surrogate(x, rng=rng)
    assert not np.allclose(x, y)


def test_iaaft_rejects_nan():
    x = np.arange(100, dtype=float)
    x[10] = np.nan
    with pytest.raises(ValueError):
        iaaft_surrogate(x)


# ── circular_shift_offsets ────────────────────────────────────────────────────
def test_shift_offsets_exclude_both_near_identity_tails():
    off = circular_shift_offsets(1000, min_shift=240)
    assert off.min() >= 240 and off.max() <= 760


def test_shift_offsets_subsample_spans_the_range():
    off = circular_shift_offsets(10000, n_shifts=50, min_shift=240)
    assert len(off) <= 50
    assert off.min() < 1000 and off.max() > 9000 - 240 - 1000


def test_shift_offsets_reject_impossible_min_shift():
    with pytest.raises(ValueError):
        circular_shift_offsets(100, min_shift=60)


# ── block_cv_indices ──────────────────────────────────────────────────────────
def test_block_cv_test_blocks_are_contiguous_and_partition():
    splits = block_cv_indices(1000, n_folds=5)
    seen = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(seen), np.arange(1000))
    for _, te in splits:
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))


def test_block_cv_gap_purges_training_neighbours():
    splits = block_cv_indices(1000, n_folds=5, gap=20)
    for tr, te in splits:
        assert np.min(np.abs(tr[:, None] - te[None, :])) > 20


def test_block_cv_train_and_test_never_overlap():
    for gap in (0, 10):
        for tr, te in block_cv_indices(500, n_folds=4, gap=gap):
            assert not np.intersect1d(tr, te).size


def test_block_cv_raises_when_gap_eats_the_training_set():
    with pytest.raises(ValueError):
        block_cv_indices(100, n_folds=2, gap=100)
