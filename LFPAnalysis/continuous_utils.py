"""Continuous (non-epoched) time-series utilities.

Everything else in this library takes an ``mne.Epochs`` and returns a per-trial feature.
That is the right contract for evoked designs, but it structurally cannot see a signal that
lives *between* trials, or one whose statistic is defined over the continuous session.

This module supplies the three primitives a continuous / naturalistic analysis needs and
that are otherwise absent from the library:

1. :func:`continuous_band_envelope` -- the same normalized-broadband envelope estimator as
   :func:`representational_utils._broadband_envelope`, on a ``(n_channels, n_times)`` array.
   Reuses the identical filter (:func:`pac_utils._filter_hilbert`) and sub-band edges
   (:func:`representational_utils.subband_edges`) so a continuous stream and an epoched one
   are the *same* measurement, which is what makes a continuous-vs-trial-locked comparison
   an apples-to-apples one.
2. :func:`iaaft_surrogate` -- an amplitude-adjusted Fourier-transform surrogate (Schreiber &
   Schmitz 1996, *PRL* 77:635). The autocorrelation-preserving null for a continuous
   regressor whose *marginal distribution* is far from Gaussian -- which a cumulative
   choice trajectory is. Complements, and does not replace, the circular shift.
3. :func:`block_cv_indices` / :func:`circular_shift_offsets` -- contiguous-block
   cross-validation and continuous circular-shift offsets. Random k-fold on an
   autocorrelated series leaks the test set into the training set through the
   autocorrelation and inflates every score; contiguous blocks with a purge gap do not.

References
----------
Schreiber & Schmitz (1996) *Phys Rev Lett* 77:635 -- iterative amplitude-adjusted surrogates.
Bergmeir & Benitez (2012) *Inf Sci* 191:192 -- blocked CV for autocorrelated series.
"""
from __future__ import annotations

import numpy as np

from LFPAnalysis.pac_utils import _filter_hilbert
from LFPAnalysis.representational_utils import subband_edges

__all__ = [
    "continuous_band_envelope",
    "decimate_envelope",
    "iaaft_surrogate",
    "block_cv_indices",
    "circular_shift_offsets",
]


# ── envelope ──────────────────────────────────────────────────────────────────
def continuous_band_envelope(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    n_subbands: int = 8,
    kind: str = "power",
    subband_norm: str = "zscore",
    filter_kind: str = "butter",
    order: int = 4,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized-broadband envelope of a continuous multichannel signal.

    Mirrors :func:`representational_utils._broadband_envelope` exactly -- split ``band``
    into ``n_subbands`` equal sub-bands, band-pass + Hilbert each, reduce to its envelope,
    normalize **per channel x sub-band**, then average -- with one difference forced by the
    absence of a trial axis: the normalization statistics are pooled over **time** only.

    Filtering is done **once over the whole session**, so there are no concatenation-edge
    transients. Restricting to a sub-interval afterwards (by slicing, or via ``valid_mask``)
    is therefore free of edge artifacts, which is what lets the same call serve both a
    continuous analysis and its trial-locked control.

    Parameters
    ----------
    data
        ``(n_channels, n_times)`` continuous signal.
    sfreq
        Sampling rate in Hz.
    band
        ``(lo, hi)`` in Hz. ``n_subbands=1`` gives a plain single-band envelope.
    kind
        ``"power"`` (envelope squared, default) or ``"amplitude"``.
    subband_norm
        ``"zscore"`` (default) or ``"mean"`` (fractional-power BHA). Matches the epoched
        estimator's options.
    valid_mask
        Optional ``(n_times,)`` boolean. When given, the per-channel x sub-band
        normalization statistics are computed over ``valid_mask`` samples **only**. Use it
        so that artefactual samples (IED transients) do not inflate the scale that every
        clean sample is then divided by. The filter still runs on the full signal --
        NaN-ing before the Hilbert would propagate across the whole series.

    Returns
    -------
    ``(n_channels, n_times)`` float array. Same units/conventions as the epoched estimator.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be (n_channels, n_times), got shape {data.shape}")
    if n_subbands < 1:
        raise ValueError(f"n_subbands must be >= 1, got {n_subbands}")
    if subband_norm not in ("zscore", "mean"):
        raise ValueError(f"subband_norm must be 'zscore' or 'mean', got {subband_norm!r}")
    if kind not in ("power", "amplitude"):
        raise ValueError(f"kind must be 'power' or 'amplitude', got {kind!r}")

    n_ch, n_t = data.shape
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (n_t,):
            raise ValueError(f"valid_mask must be ({n_t},), got {valid_mask.shape}")
        if not valid_mask.any():
            raise ValueError("valid_mask selects no samples")

    tiny = np.finfo(float).tiny
    acc = np.zeros((n_ch, n_t), dtype=float)
    for lo, hi in subband_edges(band, n_subbands):
        env = np.abs(_filter_hilbert(data, sfreq, (lo, hi), order=order,
                                     filter_kind=filter_kind))
        if kind == "power":
            env = env ** 2
        ref = env[:, valid_mask] if valid_mask is not None else env
        mean = ref.mean(axis=1, keepdims=True)
        if subband_norm == "zscore":
            std = ref.std(axis=1, keepdims=True)
            env = (env - mean) / (std + tiny)
        else:
            env = env / (mean + tiny)
        acc += env
    return acc / n_subbands


def decimate_envelope(
    env: np.ndarray,
    sfreq: float,
    target_hz: float,
    *,
    valid_mask: np.ndarray | None = None,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Box-average a continuous envelope onto a coarser regular grid.

    Averaging within non-overlapping windows **is** the anti-alias filter, so no separate
    low-pass is applied (and none should be: the envelope is already smooth, and an IIR
    pass here would smear the artefact exclusion).

    Parameters
    ----------
    env
        ``(n_channels, n_times)`` envelope.
    valid_mask
        Optional ``(n_times,)`` boolean of usable samples. Bins whose valid fraction falls
        below ``min_valid_frac`` are returned as ``NaN`` rather than silently averaged over
        a handful of samples -- downstream code must decide what to do with a hole, not
        inherit one disguised as data.

    Returns
    -------
    ``(env_ds, t_centers)`` -- ``(n_channels, n_bins)`` and ``(n_bins,)`` in seconds
    relative to sample 0.
    """
    env = np.asarray(env, dtype=float)
    if env.ndim != 2:
        raise ValueError(f"env must be (n_channels, n_times), got shape {env.shape}")
    if not (0 < target_hz <= sfreq):
        raise ValueError(f"target_hz must satisfy 0 < target_hz <= sfreq={sfreq}")

    n_ch, n_t = env.shape
    w = int(round(sfreq / target_hz))
    if w < 1:
        raise ValueError("target_hz too high for this sfreq")
    n_bins = n_t // w
    if n_bins < 1:
        raise ValueError("signal shorter than one output bin")

    trimmed = env[:, : n_bins * w].reshape(n_ch, n_bins, w)
    if valid_mask is None:
        out = trimmed.mean(axis=2)
    else:
        vm = np.asarray(valid_mask, dtype=bool)[: n_bins * w].reshape(n_bins, w)
        n_ok = vm.sum(axis=1)
        keep = n_ok >= max(1, int(np.ceil(min_valid_frac * w)))
        wm = vm[None, :, :].astype(float)
        num = (trimmed * wm).sum(axis=2)
        out = np.divide(num, n_ok[None, :], out=np.full((n_ch, n_bins), np.nan),
                        where=(n_ok[None, :] > 0))
        out[:, ~keep] = np.nan
    t = (np.arange(n_bins) * w + (w - 1) / 2.0) / sfreq
    return out, t


# ── surrogates ────────────────────────────────────────────────────────────────
def iaaft_surrogate(
    x: np.ndarray,
    *,
    n_iter: int = 100,
    rng: np.random.Generator | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996).

    Returns a series with (to convergence) the **same power spectrum** and **exactly the
    same amplitude distribution** as ``x``, but randomized phases. For a continuous
    behavioural regressor this is the complement to a circular shift: the circular shift
    reuses the *actual* series (so its autocorrelation is exact by construction, but only
    ``n-1`` distinct nulls exist and they are heavily dependent), whereas IAAFT generates
    genuinely independent draws that match the series' second-order structure and marginal.

    Neither is sufficient alone for a monotone trend: a strongly trending series has most
    of its power at the lowest frequency, and *any* spectrum-preserving surrogate will also
    trend. That is a property of the question, not a bug in the null -- it is precisely why
    a trend component must be separated out before this null is informative.

    Parameters
    ----------
    x
        1-D series. ``NaN`` is not permitted (there is no meaningful spectrum with holes).
    n_iter
        Maximum iterations. Convergence is usually reached in far fewer.
    tol
        Stop when the rank ordering stops changing (relative L2 change below ``tol``).

    Returns
    -------
    Surrogate of the same shape and dtype-family as ``x``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 4:
        raise ValueError(f"need at least 4 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("iaaft_surrogate requires finite input (no NaN/inf)")
    rng = np.random.default_rng() if rng is None else rng

    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    # start from a random shuffle -> correct marginal, destroyed spectrum
    y = rng.permutation(x)
    prev = None
    for _ in range(int(n_iter)):
        # (1) impose the target spectrum, keep the current phases
        phases = np.angle(np.fft.rfft(y))
        y = np.fft.irfft(target_amp * np.exp(1j * phases), n=x.size)
        # (2) impose the exact target marginal, keep the current rank ordering
        y = sorted_x[np.argsort(np.argsort(y))]
        if prev is not None:
            denom = np.linalg.norm(prev) or 1.0
            if np.linalg.norm(y - prev) / denom < tol:
                break
        prev = y.copy()
    return y


def circular_shift_offsets(
    n: int,
    *,
    n_shifts: int | None = None,
    min_shift: int = 1,
) -> np.ndarray:
    """Circular-shift amounts for a continuous series, excluding near-identity shifts.

    On a continuous, strongly autocorrelated series a shift of a few samples is
    effectively the identity, so the naive ``1..n-1`` set that
    :func:`representational_utils.circular_shift_indices` uses for trial-level labels is
    anti-conservative here. ``min_shift`` excludes both tails (a shift of ``n - k`` is a
    shift of ``-k``), and the retained range is sampled **evenly** so the null spans short
    and long lags rather than clustering.

    Parameters
    ----------
    n
        Series length in samples.
    n_shifts
        How many offsets to return. ``None`` -> all admissible offsets.
    min_shift
        Minimum absolute shift, in samples. For a 4 Hz envelope, 60 s is ``min_shift=240``.

    Returns
    -------
    ``(n_shifts,)`` integer offsets in ``[min_shift, n - min_shift]``.
    """
    if n < 4:
        raise ValueError(f"need at least 4 samples, got {n}")
    if not (1 <= min_shift <= n // 2):
        raise ValueError(f"min_shift must be in [1, n//2={n // 2}], got {min_shift}")
    admissible = np.arange(min_shift, n - min_shift + 1, dtype=int)
    if admissible.size == 0:
        raise ValueError(f"min_shift={min_shift} leaves no admissible shifts at n={n}")
    if n_shifts is None or n_shifts >= admissible.size:
        return admissible
    idx = np.linspace(0, admissible.size - 1, int(n_shifts)).round().astype(int)
    return admissible[np.unique(idx)]


# ── cross-validation ──────────────────────────────────────────────────────────
def block_cv_indices(
    n: int,
    *,
    n_folds: int = 5,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous-block CV splits with a purge gap, for autocorrelated series.

    Random k-fold on a continuous signal puts samples milliseconds apart into different
    folds; because they are near-identical, the model effectively sees its own test set and
    every score is inflated. Contiguous test blocks fix the gross leak; ``gap`` additionally
    **purges** the ``gap`` samples on each side of the test block from the training set, so
    the residual leak through the autocorrelation tail is bounded by the autocorrelation
    time rather than by the sampling rate.

    Parameters
    ----------
    n
        Number of samples.
    n_folds
        Number of contiguous test blocks.
    gap
        Samples purged from the training set on each side of the test block. Set it to at
        least the signal's autocorrelation time.

    Returns
    -------
    List of ``(train_idx, test_idx)``. Test blocks partition ``range(n)`` exactly; training
    sets are the complement minus the purge gap, so they do **not** partition it.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n < n_folds:
        raise ValueError(f"n={n} < n_folds={n_folds}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")

    bounds = np.linspace(0, n, n_folds + 1).round().astype(int)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        test = all_idx[lo:hi]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - gap):min(n, hi + gap)] = False
        train = all_idx[keep]
        if train.size == 0:
            raise ValueError(
                f"gap={gap} purges the entire training set for fold [{lo}, {hi}); "
                "reduce gap or n_folds")
        out.append((train, test))
    return out
