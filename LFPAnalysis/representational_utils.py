"""Representational analysis for LFP.

This module is the substrate for moving from *connectivity* (coupling
magnitude — which in the SNT cohort is content/behaviour-null) to
*representation* (what is encoded in the distributed pattern of activity).

The key object is a per-channel, per-trial **feature matrix** ``X`` of shape
``(n_trials, n_features)`` — band power / HFA envelope reduced over a window,
one column per channel (× band). Univariate, channel-collapsed versions of
these features were already shown to be content-null; the point here is the
*multivariate pattern across channels*, which a univariate test cannot rule
out. ``X`` feeds both decoding (``decoding_utils``) and the RSA helpers below.

Design choices made for parity with the rest of the project:

* Regions are selected by **exact ``SNT_region`` match** (``picks_in_region``),
  the same contact definition used by the entire connectivity/directionality
  battery — *not* ``analysis_utils.select_picks_rois`` (which matches the YBA
  atlas column and would select a different contact set).
* Filtering reuses ``pac_utils._filter_hilbert`` with the **Butterworth**
  default (the project's filter of record; long FIR is known to distort
  phase-domain directed measures here).

Dependencies are numpy/scipy only; decoders live in ``decoding_utils``.
"""
from __future__ import annotations

import numpy as np

from .pac_utils import _filter_hilbert

__all__ = [
    "canonical_egocentric",
    "reconstruct_cumulative_coord",
    "regime_labels",
    "serial_pairs",
    "picks_in_region",
    "per_trial_band_power",
    "feature_matrix",
    "per_trial_band_power_timeresolved",
    "feature_matrix_timeresolved",
    "per_trial_phase_binned_hfa",
    "feature_matrix_phase_binned",
    "feature_matrix_phase_binned_crossregion",
    "neural_rdm",
    "model_rdm",
    "rsa",
    "cross_region_rsa",
    "condition_patterns",
    "noise_covariance",
    "whiten_patterns",
    "crossnobis_rdm",
    "cv_correlation_rdm",
    "rdm_regression",
    "rdm_regression_perm",
    "circular_shift_indices",
]


# --------------------------------------------------------------------------- #
# Canonical SNT egocentric geometry
# --------------------------------------------------------------------------- #
def canonical_egocentric(x, y, affil=None, power=None, *, frame="pre", pov=(6.0, 0.0)):
    """Egocentric distance + SIGNED direction, matching the single-unit pipeline.

    Reproduces ``snt_su/scripts/snt_su_utils_canonical_circ.py::build_trial_table_for_unit``
    exactly (verified maxΔ=0 vs the stored ``r_current``/``cosine_current``). Use this
    rather than the stored angle columns, which are sign-blind:

    * ``raw_angle`` == ``|theta|`` (folded to [0, π]) — loses left/right.
    * ``cosine_current`` == ``cos(theta)`` — correct cosine, but no sine, so also
      sign-blind. The **signed** direction needs both ``cos`` and ``sin``.

    Parameters
    ----------
    x, y
        Allocentric POST-decision position (the stored ``x``, ``y`` columns).
    affil, power
        This trial's per-decision step on each axis (the stored ``affil``, ``power``).
        Required for ``frame="pre"``.
    frame
        ``"pre"`` → position during deliberation ``(x-affil, y-power)`` — the frame
        matching the pre-button-press decision window (and the stored ``r_current``).
        ``"post"`` → ``(x, y)`` (matches stored ``r_new``).
    pov
        Participant point-of-view anchor ``(POV_AFFILIATION, POV_POWER)`` = ``(6, 0)``
        (``snt_config``). RDM/Euclidean distances are POV-shift-invariant; the POV only
        affects the egocentric angle/distance.

    Returns
    -------
    dict with keys ``V`` (distance), ``theta`` (signed angle, rad ∈ [-π, π]),
    ``cos``, ``sin``, ``x_pos``, ``y_pos`` (allocentric position in this frame),
    ``x_rel``, ``y_rel`` (POV-relative).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if frame == "pre":
        if affil is None or power is None:
            raise ValueError("frame='pre' requires affil and power (the per-trial step)")
        x_pos = x - np.asarray(affil, dtype=float)
        y_pos = y - np.asarray(power, dtype=float)
    elif frame == "post":
        x_pos, y_pos = x, y
    else:
        raise ValueError(f"frame must be 'pre' or 'post', got {frame!r}")
    x_rel = x_pos - pov[0]
    y_rel = y_pos - pov[1]
    V = np.hypot(x_rel, y_rel)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(V > 0, y_rel / V, 0.0)
        sin = np.where(V > 0, -x_rel / V, 0.0)
    theta = np.arctan2(sin, cos)
    return dict(V=V, theta=theta, cos=cos, sin=sin,
                x_pos=x_pos, y_pos=y_pos, x_rel=x_rel, y_rel=y_rel)


# --------------------------------------------------------------------------- #
# Canonical SNT choice-sequence labels (regime / serial structure)
# --------------------------------------------------------------------------- #
def reconstruct_cumulative_coord(char_role_num, affil, power, decision_num):
    """Reconstruct POST-decision cumulative ``(x, y)`` position from per-decision steps.

    Each character starts at ``(0, 0)``; every decision moves ``±1`` on its axis (``affil``
    on affiliation trials, ``power`` on power trials, 0 otherwise). The cumulative position
    after trial *t* is the per-character running sum of signed steps up to *t*, ordered by
    ``decision_num``. Used as a fail-loud GATE: this reconstruction must equal the stored
    ``x``/``y`` metadata before any regime label derived from the same choice sequence can be
    trusted (the LFP cohort's trial-order/template equivalence is not otherwise verified).

    All inputs are 1-D arrays aligned to the trial order of ``epochs.metadata``. Returns
    ``(x_recon, y_recon)`` in the same order.
    """
    char = np.asarray(char_role_num)
    affil = np.asarray(affil, dtype=float)
    power = np.asarray(power, dtype=float)
    dnum = np.asarray(decision_num, dtype=float)
    n = len(char)
    x_recon = np.full(n, np.nan)
    y_recon = np.full(n, np.nan)
    for c in np.unique(char):
        idx = np.where(char == c)[0]
        order = idx[np.argsort(dnum[idx], kind="stable")]
        x_recon[order] = np.cumsum(np.nan_to_num(affil[order]))
        y_recon[order] = np.cumsum(np.nan_to_num(power[order]))
    return x_recon, y_recon


def regime_labels(char_role_num, dimension, decision, decision_num):
    """Per-trial canonical-regime label from each subject's own ``±1`` choice sequence.

    Regime is defined WITHIN each ``(character, axis)`` block, ordered by ``decision_num``,
    over the sequence of NON-zero signed steps (mirrors the behavioural ``_reversals``
    construct: no-response steps are skipped, not treated as sign 0). Each step from the
    second non-zero onward is compared to the previous non-zero step on the same axis:

    * power axis — on-regime = SAME sign as previous (perseveration; behavioural serial β>0)
    * affil axis — on-regime = OPPOSITE sign (alternation; behavioural serial β<0)

    First non-zero of a block, no-response (``decision==0``) and neutral trials
    (``dimension`` not in ``{affil, power}``) are left UNLABELED. ``decision`` is the signed
    step in coordinate convention (== ``affil`` on affil trials, == ``power`` on power trials).

    Returns a dict of input-aligned arrays: ``canonical_regime`` (1=on / 0=off / nan),
    ``regime_axis`` (``"affil"``/``"power"``/``""``), ``same_sign`` (1/0/nan) and ``labeled``
    (bool mask, ``np.isfinite(canonical_regime)``).
    """
    char = np.asarray(char_role_num)
    dim = np.asarray(dimension, dtype=object).astype(str)
    dec = np.asarray(decision, dtype=float)
    dnum = np.asarray(decision_num, dtype=float)
    n = len(char)
    regime = np.full(n, np.nan)
    same = np.full(n, np.nan)
    axis = np.full(n, "", dtype=object)
    for c in np.unique(char):
        for ax in ("affil", "power"):
            idx = np.where((char == c) & (dim == ax))[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(dnum[idx], kind="stable")]
            prev = None
            for i in order:
                d = dec[i]
                if not np.isfinite(d) or d == 0:
                    continue  # no-response: skipped, stays unlabeled, not a valid predecessor
                s = 1.0 if d > 0 else -1.0
                axis[i] = ax
                if prev is not None:
                    same_sign = (s == prev)
                    on = same_sign if ax == "power" else (not same_sign)
                    regime[i] = 1.0 if on else 0.0
                    same[i] = 1.0 if same_sign else 0.0
                prev = s
    return dict(canonical_regime=regime, regime_axis=axis, same_sign=same,
                labeled=np.isfinite(regime))


def serial_pairs(char_role_num, dimension, decision_num):
    """Consecutive same-``(character, axis)`` trial-index pairs for a lag-1 continuity metric.

    Within each ``(character, axis)`` block ordered by ``decision_num``, returns every
    adjacent pair of dimensional trials (``dimension`` in ``{affil, power}``; neutral
    excluded). A lag-1 neural continuity statistic (e.g. HFA-state cosine similarity or
    decoder-margin autocorrelation) is computed over these pairs and contrasted
    power-vs-affil, so axis-blind session drift cancels in the difference.

    Returns ``(idx_prev, idx_next, axis)`` — two int arrays and one str array, aligned
    pair-wise; indices are into the input (metadata) row order.
    """
    char = np.asarray(char_role_num)
    dim = np.asarray(dimension, dtype=object).astype(str)
    dnum = np.asarray(decision_num, dtype=float)
    ip, jn, axs = [], [], []
    for c in np.unique(char):
        for ax in ("affil", "power"):
            idx = np.where((char == c) & (dim == ax))[0]
            if idx.size < 2:
                continue
            order = idx[np.argsort(dnum[idx], kind="stable")]
            for a, b in zip(order[:-1], order[1:]):
                ip.append(int(a))
                jn.append(int(b))
                axs.append(ax)
    return np.asarray(ip, dtype=int), np.asarray(jn, dtype=int), np.asarray(axs, dtype=object)


# --------------------------------------------------------------------------- #
# Channel selection
# --------------------------------------------------------------------------- #
def picks_in_region(
    elec_df,
    roi: str,
    *,
    region_col: str = "SNT_region",
    hemi: str | None = None,
    ch_names: list[str] | None = None,
) -> list[str]:
    """Channel labels whose ``region_col`` exactly equals ``roi``.

    Parameters
    ----------
    elec_df
        Bipolar anatomy table (the extensionless ``labels_bp`` CSV), with a
        ``label`` column and ``region_col`` (default ``SNT_region``).
    roi
        Exact region string, e.g. ``"HPC"``, ``"OFC"``, ``"AMY"``, ``"ACC"``
        (case-sensitive — these match the canonical ``SNT_region`` values).
    hemi
        If given (``"l"``/``"r"``), restrict to that hemisphere. Compared
        case-insensitively against the ``hemisphere`` column.
    ch_names
        If given, intersect the result with these names (e.g.
        ``epochs.ch_names``) and preserve their order, so the returned picks
        are guaranteed to exist in the data.

    Returns
    -------
    list[str]
        Channel labels (possibly empty).
    """
    df = elec_df[elec_df[region_col] == roi]
    if hemi is not None and "hemisphere" in df.columns:
        df = df[df["hemisphere"].astype(str).str.lower() == hemi.lower()]
    picks = df["label"].astype(str).tolist()
    if ch_names is not None:
        keep = set(picks)
        picks = [c for c in ch_names if c in keep]
    return picks


# --------------------------------------------------------------------------- #
# Per-trial features
# --------------------------------------------------------------------------- #
def _band_envelope(
    epochs,
    picks: list[str],
    band: tuple[float, float],
    *,
    kind: str = "power",
    filter_kind: str = "butter",
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Analytic envelope of ``picks``, band-pass filtered over the *whole* epoch.

    Shared substrate for the three reducers (window-mean :func:`per_trial_band_power`,
    sliding-window :func:`per_trial_band_power_timeresolved`, theta-phase-binned
    :func:`per_trial_phase_binned_hfa`): the full epoch is band-pass + Hilbert
    filtered once (so any later analysis window is free of filter edge transients),
    and the analytic envelope is returned *without* time reduction.

    Returns
    -------
    env : np.ndarray
        ``(n_trials, len(picks), n_times)`` (``kind="power"`` -> squared envelope,
        ``"amplitude"`` -> envelope). Empty picks -> ``(n_trials, 0, n_times)``.
    times : np.ndarray
        ``epochs.times``.
    picks : list[str]
        The picks actually present in ``epochs`` (order preserved).
    """
    picks = [c for c in picks if c in epochs.ch_names]
    times = epochs.times
    n_trials = len(epochs)
    if not picks:
        return np.empty((n_trials, 0, times.size), dtype=float), times, picks

    fs = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks, copy=True)  # (n_trials, n_picks, n_times)
    analytic = _filter_hilbert(data, fs, band, order=order, filter_kind=filter_kind)
    env = np.abs(analytic)
    if kind == "power":
        env = env ** 2
    elif kind != "amplitude":
        raise ValueError(f"kind must be 'power' or 'amplitude', got {kind!r}")
    return env, times, picks


def _broadband_envelope(
    epochs,
    picks: list[str],
    band: tuple[float, float],
    *,
    n_subbands: int = 8,
    kind: str = "power",
    subband_norm: str = "zscore",
    filter_kind: str = "butter",
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalized-broadband HFA envelope: per-sub-band normalize, then average.

    The canonical broadband high-frequency-activity (BHA) estimate, which avoids the
    1/f dominance of a single wide band-pass: a 70-150 Hz envelope is weighted toward
    ~70-90 Hz, where aperiodic power is largest, so it under-represents the rest of the
    band. Here ``band`` is split into ``n_subbands`` equal sub-bands; each is band-pass
    + Hilbert filtered, reduced to its envelope (``kind``), and **normalized per channel
    x sub-band** so every band contributes comparably; the normalized envelopes are then
    averaged. Refs: Ray & Maunsell 2011 (PLoS Biol), Dubey & Ray 2019 (J Neurosci),
    Leszczynski et al. 2020 (Sci Adv).

    Normalization statistics are pooled over the subject's ``(trial, time)`` axes -- one
    scalar per channel x sub-band -- so trial-to-trial and temporal structure are fully
    preserved and the transform is **label-blind** (no decoding leakage), exactly as the
    ``log10`` in :func:`per_trial_band_power` is.

    Parameters
    ----------
    n_subbands
        Number of equal sub-bands spanning ``band`` (default 8 -> 10-Hz bands over 70-150).
    subband_norm
        ``"zscore"`` -> ``(env - mean) / std`` (equal-variance contribution per band;
        default); ``"mean"`` -> ``env / mean`` (classic fractional-power BHA, stays positive).

    Returns
    -------
    Same ``(env, times, picks)`` contract as :func:`_band_envelope`, so the window
    reducers are agnostic to which substrate produced the envelope.
    """
    picks = [c for c in picks if c in epochs.ch_names]
    times = epochs.times
    n_trials = len(epochs)
    if not picks:
        return np.empty((n_trials, 0, times.size), dtype=float), times, picks
    if n_subbands < 1:
        raise ValueError(f"n_subbands must be >= 1, got {n_subbands}")
    if subband_norm not in ("zscore", "mean"):
        raise ValueError(f"subband_norm must be 'zscore' or 'mean', got {subband_norm!r}")
    if kind not in ("power", "amplitude"):
        raise ValueError(f"kind must be 'power' or 'amplitude', got {kind!r}")

    fs = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks, copy=True)  # (n_trials, n_picks, n_times)
    edges = np.linspace(band[0], band[1], n_subbands + 1)
    tiny = np.finfo(float).tiny
    acc = np.zeros((n_trials, len(picks), times.size), dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        env = np.abs(_filter_hilbert(data, fs, (float(lo), float(hi)),
                                     order=order, filter_kind=filter_kind))
        if kind == "power":
            env = env ** 2
        # per channel x sub-band scalar(s), pooled over (trial, time) -> label-blind
        mean = env.mean(axis=(0, 2), keepdims=True)
        if subband_norm == "zscore":
            std = env.std(axis=(0, 2), keepdims=True)
            env = (env - mean) / (std + tiny)
        else:  # "mean"
            env = env / (mean + tiny)
        acc += env
    return acc / n_subbands, times, picks


def _window_centers(times, window_s, step_s, tmin, tmax) -> np.ndarray:
    """Sliding-window center grid, matching ``directionality.sliding_windows`` semantics.

    Reimplemented inline (this module is upstream and stays numpy/scipy-only, so it
    cannot import ``snt_lfp``) so the time axis is identical to the rest of the project.
    """
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    start_bound = float(times[0] if tmin is None else max(tmin, times[0]))
    stop_bound = float(times[-1] if tmax is None else min(tmax, times[-1]))
    first_center = start_bound + window_s / 2
    last_center = stop_bound - window_s / 2
    if first_center > last_center:
        return np.array([], dtype=float)
    return np.arange(first_center, last_center + step_s / 2, step_s)


def per_trial_band_power(
    epochs,
    picks: list[str],
    band: tuple[float, float],
    tmin: float,
    tmax: float,
    *,
    kind: str = "power",
    filter_kind: str = "butter",
    order: int = 4,
    log: bool = True,
    n_subbands: int = 1,
    subband_norm: str = "zscore",
) -> np.ndarray:
    """Per-channel, per-trial band power/amplitude over ``[tmin, tmax]``.

    The full epoch is band-pass + Hilbert filtered (so the analysis window is
    free of filter edge transients), then the analytic envelope is reduced
    over the window.

    Parameters
    ----------
    epochs
        MNE Epochs (preloaded).
    picks
        Channel labels to extract (order preserved). Empty -> ``(n_trials, 0)``.
    band
        ``(low, high)`` Hz, e.g. ``BANDS["theta"]`` or ``BANDS["high_gamma"]``
        for HFA.
    tmin, tmax
        Reduction window in seconds, relative to event (t=0).
    kind
        ``"power"`` -> mean squared envelope; ``"amplitude"`` -> mean envelope.
        (``"hfa"`` is just ``kind="power"`` on the high-gamma band.)
    filter_kind, order
        Passed to :func:`pac_utils._filter_hilbert` (Butterworth by default).
    log
        If True, return ``log10`` of the reduced value (standard for power,
        stabilises variance across channels). Forced off when ``n_subbands > 1``.
    n_subbands, subband_norm
        If ``n_subbands > 1``, use the normalized-broadband substrate
        (:func:`_broadband_envelope`, the 1/f-robust HFA estimate) instead of a single
        band-pass. Default ``n_subbands=1`` reproduces the single-band envelope exactly.

    Returns
    -------
    np.ndarray
        ``(n_trials, len(picks))``.
    """
    if n_subbands > 1:
        env, times, picks = _broadband_envelope(
            epochs, picks, band, n_subbands=n_subbands, kind=kind,
            subband_norm=subband_norm, filter_kind=filter_kind, order=order,
        )
        log = False  # per-sub-band normalization already stabilizes; z-score can be < 0
    else:
        env, times, picks = _band_envelope(
            epochs, picks, band, kind=kind, filter_kind=filter_kind, order=order
        )
    if not picks:
        return np.empty((len(epochs), 0), dtype=float)

    mask = (times >= tmin) & (times <= tmax)
    if not mask.any():
        raise ValueError(f"window [{tmin}, {tmax}] selects no samples in {times[0]}..{times[-1]}")
    reduced = env[:, :, mask].mean(axis=2)  # (n_trials, n_picks)

    if log:
        reduced = np.log10(reduced + np.finfo(float).tiny)
    return reduced


def feature_matrix(
    epochs,
    elec_df,
    roi: str,
    bands: dict[str, tuple[float, float]],
    tmin: float,
    tmax: float,
    *,
    hemi: str | None = None,
    kind: str = "power",
    region_col: str = "SNT_region",
    **power_kw,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stacked per-trial feature matrix for one region.

    Concatenates :func:`per_trial_band_power` across ``bands`` for every
    channel of ``roi``.

    Returns
    -------
    X : np.ndarray
        ``(n_trials, n_channels * n_bands)``.
    feature_names : list[str]
        ``"{channel}|{band}"`` for each column.
    picks : list[str]
        The channels used (in column-block order).
    """
    picks = picks_in_region(
        elec_df, roi, region_col=region_col, hemi=hemi, ch_names=epochs.ch_names
    )
    if not picks:
        return np.empty((len(epochs), 0), dtype=float), [], []

    blocks, names = [], []
    for bname, band in bands.items():
        blocks.append(
            per_trial_band_power(epochs, picks, band, tmin, tmax, kind=kind, **power_kw)
        )
        names.extend(f"{ch}|{bname}" for ch in picks)
    X = np.concatenate(blocks, axis=1)
    return X, names, picks


# --------------------------------------------------------------------------- #
# Time-resolved features (sliding window) -- a different reduction of the same
# envelope as per_trial_band_power. The window-averaged HFA decode was content-
# null; this exposes the temporal structure window-averaging discards.
# --------------------------------------------------------------------------- #
def per_trial_band_power_timeresolved(
    epochs,
    picks: list[str],
    band: tuple[float, float],
    *,
    window_s: float = 0.5,
    step_s: float = 0.1,
    tmin: float = -3.0,
    tmax: float = 0.0,
    kind: str = "power",
    filter_kind: str = "butter",
    order: int = 4,
    log: bool = True,
    n_subbands: int = 1,
    subband_norm: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel, per-trial band power in a SLIDING sub-window across the epoch.

    The analytic envelope is computed ONCE over the whole epoch (one filter pass)
    and then reduced in each sliding window -- the time-resolved generalisation of
    :func:`per_trial_band_power` (which reduces a single window). Window centers
    match ``snt_lfp.directionality.sliding_windows`` so the time axis is identical
    to the rest of the project. If ``n_subbands > 1``, the substrate is the normalized-
    broadband HFA envelope (:func:`_broadband_envelope`); ``n_subbands=1`` (default) is
    the single-band envelope, unchanged.

    Returns
    -------
    X_t : np.ndarray
        ``(n_trials, len(picks), n_windows)`` reduced power per sliding window.
    centers : np.ndarray
        ``(n_windows,)`` window-center times (s).
    """
    if n_subbands > 1:
        env, times, picks = _broadband_envelope(
            epochs, picks, band, n_subbands=n_subbands, kind=kind,
            subband_norm=subband_norm, filter_kind=filter_kind, order=order,
        )
        log = False  # per-sub-band normalization already stabilizes; z-score can be < 0
    else:
        env, times, picks = _band_envelope(
            epochs, picks, band, kind=kind, filter_kind=filter_kind, order=order
        )
    centers = _window_centers(times, window_s, step_s, tmin, tmax)
    if not picks or centers.size == 0:
        return np.empty((len(epochs), len(picks), centers.size), dtype=float), centers

    cols = []
    for c in centers:
        m = (times >= c - window_s / 2) & (times <= c + window_s / 2)
        cols.append(env[:, :, m].mean(axis=2))  # (n_trials, n_picks)
    X_t = np.stack(cols, axis=2)  # (n_trials, n_picks, n_windows)
    if log:
        X_t = np.log10(X_t + np.finfo(float).tiny)
    return X_t, centers


def feature_matrix_timeresolved(
    epochs,
    elec_df,
    roi: str,
    band: tuple[float, float],
    *,
    window_s: float = 0.5,
    step_s: float = 0.1,
    tmin: float = -3.0,
    tmax: float = 0.0,
    hemi: str | None = None,
    kind: str = "power",
    region_col: str = "SNT_region",
    **power_kw,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Time-resolved per-trial feature tensor for one region (single band).

    The sliding-window analogue of :func:`feature_matrix`. Single band (the headline
    is HFA-only); decode each window by slicing ``X[:, :, w]`` -> ``(n_trials, n_channels)``.

    Returns
    -------
    X : np.ndarray
        ``(n_trials, n_channels, n_windows)``.
    centers : np.ndarray
        ``(n_windows,)`` window-center times (s).
    picks : list[str]
        The channels used.
    """
    picks = picks_in_region(
        elec_df, roi, region_col=region_col, hemi=hemi, ch_names=epochs.ch_names
    )
    if not picks:
        return (np.empty((len(epochs), 0, 0), dtype=float), np.array([], dtype=float), [])
    X, centers = per_trial_band_power_timeresolved(
        epochs, picks, band, window_s=window_s, step_s=step_s,
        tmin=tmin, tmax=tmax, kind=kind, **power_kw,
    )
    return X, centers, picks


# --------------------------------------------------------------------------- #
# Event-locked features (per-trial onset). The sliding-window reducers above use
# window centers fixed relative to the epoch's t=0. When the event of interest
# occurs at a DIFFERENT, per-trial time (e.g. options/stimulus onset at t=-RT in a
# response-locked epoch), each trial must be reduced in windows centered on ITS OWN
# onset. The envelope is still computed once (label-blind), only the reduction is
# per-trial. Windows that fall outside the epoch are NaN (the decoder drops them).
# --------------------------------------------------------------------------- #
def per_trial_band_power_event_locked(
    epochs,
    picks: list[str],
    band: tuple[float, float],
    onset_s,
    offsets,
    *,
    window_s: float = 0.5,
    kind: str = "power",
    filter_kind: str = "butter",
    order: int = 4,
    log: bool = True,
    n_subbands: int = 1,
    subband_norm: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel, per-trial band power in windows locked to a PER-TRIAL event onset.

    The event-locked analogue of :func:`per_trial_band_power_timeresolved`. The analytic
    envelope is computed ONCE over the whole epoch (same substrate as the timeresolved
    reducer -- normalized-broadband if ``n_subbands > 1``, single band otherwise), then
    reduced in a sliding window centered on ``onset_s[trial] + offset`` for every offset.
    This re-locks a response-locked epoch to a per-trial stimulus onset (pass
    ``onset_s = -reaction_time``) without re-epoching the raw data.

    Parameters
    ----------
    onset_s : array-like, shape (n_trials,)
        Per-trial event time in seconds relative to the epoch's t=0 (the lock target).
        Non-finite entries yield an all-NaN trial.
    offsets : array-like, shape (n_offsets,)
        Window-center offsets (s) relative to each trial's onset (e.g. ``0 .. +2`` s
        after stimulus onset).
    window_s : float
        Reduction-window width (s), centered on ``onset + offset``.

    Returns
    -------
    X_t : np.ndarray
        ``(n_trials, len(picks), len(offsets))``. NaN wherever the window falls outside
        the epoch (partial-window trials are not silently truncated).
    offsets : np.ndarray
        ``(n_offsets,)`` echoed offsets (the event-locked time axis).
    """
    if n_subbands > 1:
        env, times, picks = _broadband_envelope(
            epochs, picks, band, n_subbands=n_subbands, kind=kind,
            subband_norm=subband_norm, filter_kind=filter_kind, order=order,
        )
        log = False  # per-sub-band normalization already stabilizes; z-score can be < 0
    else:
        env, times, picks = _band_envelope(
            epochs, picks, band, kind=kind, filter_kind=filter_kind, order=order
        )
    offsets = np.asarray(offsets, dtype=float)
    onset = np.asarray(onset_s, dtype=float)
    n_tr = len(epochs)
    if not picks or offsets.size == 0:
        return np.empty((n_tr, len(picks), offsets.size), dtype=float), offsets
    if onset.shape != (n_tr,):
        raise ValueError(f"onset_s must have shape ({n_tr},), got {onset.shape}")

    half = window_s / 2.0
    t0, t1 = float(times[0]), float(times[-1])
    X_t = np.full((n_tr, len(picks), offsets.size), np.nan, dtype=float)
    for j, off in enumerate(offsets):
        centers = onset + off  # (n_trials,)
        for i in range(n_tr):
            c = centers[i]
            if not np.isfinite(c) or (c - half) < t0 or (c + half) > t1:
                continue  # window outside the epoch -> leave NaN
            m = (times >= c - half) & (times <= c + half)
            if m.any():
                X_t[i, :, j] = env[i][:, m].mean(axis=1)  # (n_pick, n_win) -> (n_pick,)
    if log:
        X_t = np.log10(X_t + np.finfo(float).tiny)
    return X_t, offsets


def feature_matrix_event_locked(
    epochs,
    elec_df,
    roi: str,
    band: tuple[float, float],
    onset_s,
    offsets,
    *,
    window_s: float = 0.5,
    hemi: str | None = None,
    kind: str = "power",
    region_col: str = "SNT_region",
    **power_kw,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Event-locked (per-trial onset) per-trial feature tensor for one region (single band).

    The per-trial-onset analogue of :func:`feature_matrix_timeresolved`; decode each offset
    by slicing ``X[:, :, j]`` -> ``(n_trials, n_channels)``.

    Returns
    -------
    X : np.ndarray
        ``(n_trials, n_channels, n_offsets)``; NaN where a window leaves the epoch.
    offsets : np.ndarray
        ``(n_offsets,)`` event-locked time axis.
    picks : list[str]
        The channels used.
    """
    picks = picks_in_region(
        elec_df, roi, region_col=region_col, hemi=hemi, ch_names=epochs.ch_names
    )
    if not picks:
        return (np.empty((len(epochs), 0, 0), dtype=float), np.asarray(offsets, float), [])
    X, offs = per_trial_band_power_event_locked(
        epochs, picks, band, onset_s, offsets, window_s=window_s, kind=kind, **power_kw,
    )
    return X, offs, picks


# --------------------------------------------------------------------------- #
# Theta-phase-binned HFA features (Lisman/Jensen theta-gamma code). HFA power
# (amplitude) binned by concurrent theta PHASE -- HFA is broadband, never phase.
# --------------------------------------------------------------------------- #
def per_trial_phase_binned_hfa(
    epochs,
    picks: list[str],
    *,
    phase_band: tuple[float, float] = (4.0, 8.0),
    amp_band: tuple[float, float] = (70.0, 150.0),
    n_bins: int = 6,
    tmin: float = -3.0,
    tmax: float = 0.0,
    filter_kind: str = "butter",
    order: int = 4,
    log: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean HFA power within each theta-phase bin, per trial per channel (within-channel theta-gamma).

    For each trial x channel, the high-frequency-amplitude (``amp_band``) power is
    averaged within ``n_bins`` equal bins of the *same channel's* theta
    (``phase_band``) phase over ``[tmin, tmax]`` -- the Tort modulation-index
    construction kept per-trial (a single-trial-valid amplitude reduction, not a
    coupling magnitude). HFA contributes amplitude only; the only phase used is
    theta (HFA is broadband, not an oscillation; cf. Ray & Maunsell 2011).

    Returns
    -------
    X_phi : np.ndarray
        ``(n_trials, len(picks), n_bins)`` mean HFA power per theta-phase bin.
        Empty bins are NaN (the decoder's finite-row filter drops them).
    bin_centers : np.ndarray
        ``(n_bins,)`` phase-bin centers (rad, in [-pi, pi]).
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    env, times, picks = _band_envelope(
        epochs, picks, amp_band, kind="power", filter_kind=filter_kind, order=order
    )
    if not picks:
        return np.empty((len(epochs), 0, n_bins), dtype=float), bin_centers

    fs = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks, copy=True)
    phi = np.angle(_filter_hilbert(data, fs, phase_band, order=order, filter_kind=filter_kind))

    m = (times >= tmin) & (times <= tmax)
    if not m.any():
        raise ValueError(f"window [{tmin}, {tmax}] selects no samples in {times[0]}..{times[-1]}")
    env_w = env[:, :, m]  # (n_trials, n_picks, n_win)
    phi_w = phi[:, :, m]
    idx = np.clip(np.digitize(phi_w, edges) - 1, 0, n_bins - 1)  # bin index per sample

    n_tr, n_pk, _ = env_w.shape
    X_phi = np.full((n_tr, n_pk, n_bins), np.nan, dtype=float)
    for b in range(n_bins):
        sel = idx == b
        cnt = sel.sum(axis=2)
        s = np.where(sel, env_w, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore"):
            X_phi[:, :, b] = np.where(cnt > 0, s / cnt, np.nan)
    if log:
        X_phi = np.log10(X_phi + np.finfo(float).tiny)
    return X_phi, bin_centers


def feature_matrix_phase_binned(
    epochs,
    elec_df,
    roi: str,
    *,
    phase_band: tuple[float, float] = (4.0, 8.0),
    amp_band: tuple[float, float] = (70.0, 150.0),
    n_bins: int = 6,
    tmin: float = -3.0,
    tmax: float = 0.0,
    hemi: str | None = None,
    region_col: str = "SNT_region",
    **kw,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Theta-phase-binned HFA feature tensor for one region.

    Decode each phase bin by slicing ``X[:, :, b]`` -> ``(n_trials, n_channels)``.

    Returns
    -------
    X : np.ndarray
        ``(n_trials, n_channels, n_bins)``.
    bin_centers : np.ndarray
        ``(n_bins,)`` phase-bin centers (rad).
    picks : list[str]
        The channels used.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    picks = picks_in_region(
        elec_df, roi, region_col=region_col, hemi=hemi, ch_names=epochs.ch_names
    )
    if not picks:
        return np.empty((len(epochs), 0, n_bins), dtype=float), bin_centers, []
    X, bin_centers = per_trial_phase_binned_hfa(
        epochs, picks, phase_band=phase_band, amp_band=amp_band, n_bins=n_bins,
        tmin=tmin, tmax=tmax, **kw,
    )
    return X, bin_centers, picks


def feature_matrix_phase_binned_crossregion(
    epochs,
    elec_df,
    phase_roi: str,
    amp_roi: str,
    *,
    phase_band: tuple[float, float] = (4.0, 8.0),
    amp_band: tuple[float, float] = (30.0, 150.0),
    n_bins: int = 6,
    tmin: float = -3.0,
    tmax: float = 0.0,
    normalize: str = "sum",
    pair_reduce: str = "mean",
    hemis: tuple[str, ...] = ("l", "r"),
    region_col: str = "SNT_region",
    order: int = 4,
    filter_kind: str = "butter",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cross-region theta-phase-binned amplitude: TARGET amplitude binned by SOURCE theta phase.

    Mirrors the established OFC-theta -> HPC-HFA PAC-heading construction (Canolty/Tort per-trial
    reduction; docs/methods/pac_cross_region_20260603.md). For each within-hemisphere
    (source-channel, target-channel) pair: SOURCE (``phase_roi``) supplies theta phase
    (Butterworth+Hilbert angle), TARGET (``amp_roi``) the **linear** amplitude ``|analytic|`` (NOT
    power/log), binned over ``[tmin, tmax]`` into ``n_bins`` equal ``[-pi, pi]`` phase bins (mean per
    bin) -- a single-trial-valid amplitude reduction. Two choices carried from the heading result:
    (1) the amplitude band is the **broad/low gamma (30-150 / 30-70)** where heading coupling was
    confirmed -- 70-150 was NULL; (2) each trial's bin profile is **normalized across bins**
    (``normalize``) so the feature encodes coupling **depth/shape** (where heading lives), not the
    phase-averaged gamma **level** (null). Channels are combined to the subject downstream (subject =
    unit), never pooled across subjects.

    Parameters
    ----------
    phase_roi, amp_roi
        Source (theta phase) and target (amplitude) region strings, e.g. ``"OFC"`` -> ``"HPC"``.
    normalize
        Per-trial across-bin normalization of each pair profile: ``"sum"`` (fractional, mirrors the
        MVL denominator; default), ``"zscore"``, or ``"none"``.
    pair_reduce
        ``"mean"`` -> average the normalized profile across all (src, tgt) pairs -> one profile/trial
        (``n_features=1``; best-conditioned shape test); ``"none"`` -> every pair (``n_features=n_pairs``).

    Returns
    -------
    X : np.ndarray
        ``(n_trials, n_features, n_bins)``.
    bin_centers : np.ndarray
        ``(n_bins,)`` phase-bin centers (rad).
    pair_labels : list[str]
        ``"{src}|{tgt}|{hemi}"`` per feature (single ``"mean|{n}pairs"`` when ``pair_reduce="mean"``).
    """
    if normalize not in ("sum", "zscore", "none"):
        raise ValueError(f"normalize must be sum/zscore/none, got {normalize!r}")
    if pair_reduce not in ("mean", "none"):
        raise ValueError(f"pair_reduce must be mean/none, got {pair_reduce!r}")
    fs = float(epochs.info["sfreq"])
    times = epochs.times
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    n_trials = len(epochs)
    wmask = (times >= tmin) & (times <= tmax)
    if not wmask.any():
        raise ValueError(f"window [{tmin}, {tmax}] selects no samples")

    profiles, labels = [], []
    for h in hemis:
        src = picks_in_region(elec_df, phase_roi, region_col=region_col, hemi=h, ch_names=epochs.ch_names)
        tgt = picks_in_region(elec_df, amp_roi, region_col=region_col, hemi=h, ch_names=epochs.ch_names)
        if not src or not tgt:
            continue
        phi = np.angle(_filter_hilbert(epochs.get_data(picks=src, copy=True), fs, phase_band,
                                       order=order, filter_kind=filter_kind))[:, :, wmask]
        amp = np.abs(_filter_hilbert(epochs.get_data(picks=tgt, copy=True), fs, amp_band,
                                     order=order, filter_kind=filter_kind))[:, :, wmask]
        for si, s in enumerate(src):
            digit = np.clip(np.digitize(phi[:, si, :], edges) - 1, 0, n_bins - 1)  # (n_tr, n_win)
            for ti, t in enumerate(tgt):
                a = amp[:, ti, :]
                prof = np.full((n_trials, n_bins), np.nan)
                for b in range(n_bins):
                    sel = digit == b
                    cnt = sel.sum(axis=1)
                    prof[:, b] = np.where(cnt > 0, np.where(sel, a, 0.0).sum(axis=1) / np.maximum(cnt, 1),
                                          np.nan)
                profiles.append(prof)
                labels.append(f"{s}|{t}|{h}")
    if not profiles:
        return np.empty((n_trials, 0, n_bins), dtype=float), bin_centers, []

    X = np.stack(profiles, axis=1)  # (n_trials, n_pairs, n_bins)
    if normalize == "sum":
        ssum = np.nansum(X, axis=2, keepdims=True)
        X = np.where(ssum > 0, X / ssum, np.nan)
    elif normalize == "zscore":
        mu = np.nanmean(X, axis=2, keepdims=True)
        sd = np.nanstd(X, axis=2, keepdims=True)
        X = np.where(sd > 0, (X - mu) / sd, np.nan)
    if pair_reduce == "mean":
        X = np.nanmean(X, axis=1, keepdims=True)  # (n_trials, 1, n_bins)
        labels = [f"mean|{len(profiles)}pairs"]
    return X, bin_centers, labels


# --------------------------------------------------------------------------- #
# Representational (dis)similarity
# --------------------------------------------------------------------------- #
def neural_rdm(X: np.ndarray, *, metric: str = "correlation", zscore: bool = True) -> np.ndarray:
    """Condensed neural RDM (trial × trial dissimilarity) from a feature matrix.

    Parameters
    ----------
    X
        ``(n_trials, n_features)`` feature matrix.
    metric
        Any ``scipy.spatial.distance.pdist`` metric. ``"correlation"`` (1 −
        Pearson across features) and ``"euclidean"`` are the usual choices.
    zscore
        Standardise each feature (column) across trials before computing
        distances. Recommended for ``"euclidean"`` so high-variance channels
        don't dominate; harmless for ``"correlation"``.

    Returns
    -------
    np.ndarray
        Condensed upper-triangle vector, length ``n_trials*(n_trials-1)/2``
        (``scipy.spatial.distance.squareform`` compatible).
    """
    from scipy.spatial.distance import pdist

    Xz = np.asarray(X, dtype=float)
    if zscore:
        mu = Xz.mean(axis=0, keepdims=True)
        sd = Xz.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Xz = (Xz - mu) / sd
    return pdist(Xz, metric=metric)


def model_rdm(values, *, kind: str = "euclidean") -> np.ndarray:
    """Condensed model RDM from a per-trial variable.

    Parameters
    ----------
    values
        Per-trial values. For ``"euclidean"``: ``(n_trials,)`` or
        ``(n_trials, k)`` (e.g. social-space coordinates ``[affil, power]``).
        For ``"circular"``: ``(n_trials,)`` angles in radians (pairwise
        angular distance ∈ [0, π]). For ``"categorical"``: any labels
        (0 if same, 1 if different).

    Returns
    -------
    np.ndarray
        Condensed upper-triangle vector matching :func:`neural_rdm`.
    """
    from scipy.spatial.distance import pdist, squareform

    v = np.asarray(values)
    if kind == "euclidean":
        if v.ndim == 1:
            v = v[:, None]
        return pdist(v.astype(float), metric="euclidean")
    if kind == "circular":
        a = v.astype(float).ravel()
        diff = np.abs(a[:, None] - a[None, :])
        ang = np.minimum(diff, 2 * np.pi - diff)  # wrap to [0, pi]
        return squareform(ang, checks=False)
    if kind == "categorical":
        lab = v.ravel()
        same = (lab[:, None] == lab[None, :]).astype(float)
        return squareform(1.0 - same, checks=False)
    raise ValueError(f"unknown kind={kind!r}")


def rsa(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    *,
    method: str = "spearman",
    null: str = "random",
    n_perm: int = 1000,
    partial: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Second-order similarity between two condensed RDMs with a permutation test.

    Two null families (the choice matters when the conditions are ordered in time):

    * ``null="random"`` — relabel the conditions of ``rdm1`` by a *random*
      permutation. Standard, but anti-conservative when temporal autocorrelation
      (slow drift) makes nearby-in-time conditions spuriously similar.
    * ``null="circular"`` — relabel by every *contiguous circular shift*
      (``s = 1..n-1``). This **preserves the temporal autocorrelation** of the
      RDM in the null, so significance means "structure beyond what drift
      explains" without deleting variance from the data. Use this when the
      conditions are in trial order (the SU-consistent control). Exhaustive and
      deterministic (``n_perm``/``rng`` ignored).

    For the group-level drift-controlled statistic, center each subject's
    ``r`` by the returned ``null_mean`` (``r - null_mean``) and test across
    subjects — this is the non-over-residualizing alternative to ``partial``.

    Parameters
    ----------
    rdm1, rdm2
        Condensed RDMs (e.g. neural and model), same length. For ``null="circular"``
        the conditions of ``rdm1`` must be in trial order.
    method
        ``"spearman"`` (rank, default) or ``"pearson"``.
    null
        ``"random"`` or ``"circular"`` (see above).
    partial
        Optional condensed nuisance RDM linearly residualised out of *both* RDMs
        before correlating (e.g. a ``decision_num`` geometry). This is the
        *conservative* drift control; it can over-residualise variables that
        evolve over the session — prefer the ``"circular"`` null for those.
    n_perm
        Number of random permutations (``null="random"`` only).
    rng
        Optional numpy Generator (``null="random"`` only).

    Returns
    -------
    dict
        ``{"r", "p", "null_mean", "n_null", "null", "method"}``. ``p`` is the
        one-sided (r greater than null) p-value, observed included.
    """
    from scipy.spatial.distance import squareform
    from scipy.stats import rankdata

    a = np.asarray(rdm1, dtype=float)
    b = np.asarray(rdm2, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"RDMs differ in length: {a.shape} vs {b.shape}")

    def _resid(x, z):
        Z = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
        return x - Z @ beta

    def _corr(x, y):
        if method == "spearman":
            x, y = rankdata(x), rankdata(y)
        elif method != "pearson":
            raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
        xc, yc = x - x.mean(), y - y.mean()
        denom = np.sqrt((xc @ xc) * (yc @ yc))
        return float(xc @ yc / denom) if denom > 0 else 0.0

    if partial is not None:
        pz = np.asarray(partial, dtype=float)
        a = _resid(a, pz)
        b = _resid(b, pz)

    r_obs = _corr(a, b)
    sq = squareform(a, checks=False)
    n = sq.shape[0]

    null_vals = []
    if null == "circular":
        base = np.arange(n)
        for s in range(1, n):
            idx = (base + s) % n
            null_vals.append(_corr(squareform(sq[np.ix_(idx, idx)], checks=False), b))
    elif null == "random":
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(n_perm):
            perm = rng.permutation(n)
            null_vals.append(_corr(squareform(sq[np.ix_(perm, perm)], checks=False), b))
    else:
        raise ValueError(f"null must be 'random' or 'circular', got {null!r}")

    null_vals = np.asarray(null_vals, dtype=float)
    p_val = (1 + int(np.sum(null_vals >= r_obs))) / (len(null_vals) + 1)
    return {"r": r_obs, "p": p_val,
            "null_mean": float(np.mean(null_vals)) if null_vals.size else np.nan,
            "n_null": int(null_vals.size), "null": null, "method": method}


def cross_region_rsa(rdm_a: np.ndarray, rdm_b: np.ndarray, **kw) -> dict:
    """Mantel-style similarity between two *neural* RDMs (e.g. HPC vs OFC).

    Thin wrapper over :func:`rsa`; included for readability at call sites
    where the question is "do two regions share representational geometry".
    """
    return rsa(rdm_a, rdm_b, **kw)


# --------------------------------------------------------------------------- #
# Condition-level, cross-validated dissimilarity (crossnobis) + RDM regression
# --------------------------------------------------------------------------- #
# Why these exist, given `neural_rdm`/`rsa` above already work:
#
# * `neural_rdm(metric="correlation")` is the *least reliable* of the standard
#   dissimilarity estimators (Walther, Nili, Ejaz, Alink, Kriegeskorte &
#   Diedrichsen 2016, NeuroImage 137:188-200). That paper's recommendation is a
#   **continuous cross-validated distance with multivariate noise
#   normalization** — i.e. crossnobis. It is not implemented anywhere else in
#   this package.
# * Crossnobis is *undefined on singleton conditions*: it needs each condition
#   measured at least twice so the two independent measurements can be crossed.
#   That forces averaging trials into **conditions**, which is itself the main
#   sensitivity gain (averaging k trials cuts pattern noise by ~sqrt(k)).
# * `rsa()` accepts exactly ONE scalar nuisance RDM. Competing geometries that
#   are mutually collinear (in SNT: rho(|dpos|,|dtheta|) ~= 0.73) cannot be
#   adjudicated one-at-a-time — they need a joint design, unique/shared
#   variance partitioning, and VIFs. Hence `rdm_regression`.
#
# NOTE on nulls: `rdm_regression_perm` permutes the condensed neural RDM
# directly, which at condition level gives only `n_conditions - 1` exhaustive
# circular shifts (~9-11 in SNT) -- enough to *debias* a per-subject statistic,
# too few for a meaningful per-subject p. When the caller can rebuild the
# binning, prefer shifting the behavioural labels at the TRIAL level and
# recomputing conditions; that preserves drift autocorrelation and yields
# ~n_trials-1 nulls. `circular_shift_indices` is exposed for that purpose.


def circular_shift_indices(n: int, n_perm: int | None = None) -> np.ndarray:
    """Contiguous circular-shift index sets for an order-preserving null.

    Returns ``(n_shifts, n)`` integer index arrays, one per shift ``s`` in
    ``1..n-1`` (the identity ``s=0`` is excluded). If ``n_perm`` is given and
    smaller than ``n-1``, shifts are subsampled **evenly** across the range so
    the null still spans short and long lags rather than clustering at one end.

    A circular shift preserves the temporal autocorrelation of the shifted
    variable, so "significant vs this null" means *structure beyond what slow
    drift explains*. A random (iid) permutation destroys that autocorrelation
    and is anti-conservative for anything that evolves over a session.
    """
    n = int(n)
    if n < 3:
        return np.empty((0, n), dtype=int)
    shifts = np.arange(1, n)
    if n_perm is not None and 0 < int(n_perm) < shifts.size:
        sel = np.unique(np.linspace(0, shifts.size - 1, int(n_perm)).round().astype(int))
        shifts = shifts[sel]
    base = np.arange(n)
    return np.stack([(base + int(s)) % n for s in shifts])


def condition_patterns(
    X: np.ndarray,
    cond_id,
    *,
    n_folds: int = 2,
    order=None,
    min_per_fold: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-wise condition-mean activity patterns (the input to crossnobis).

    Trials of each condition are assigned to folds by **interleaving along
    ``order``** (default: row order). Interleaving rather than block-splitting
    matters when ``order`` is session time: it balances slow drift across folds,
    so a fold difference cannot be manufactured by drift. This is the same
    split rule the SNT ``dots_rsa`` split-half uses.

    Parameters
    ----------
    X
        ``(n_trials, n_features)``.
    cond_id
        ``(n_trials,)`` condition labels (any hashable dtype).
    n_folds
        Number of independent measurements per condition (2 = split-half).
    order
        ``(n_trials,)`` sort key used for interleaving (e.g. ``decision_num``).
        Defaults to row order.
    min_per_fold
        Conditions with fewer than this many trials in **any** fold are dropped.

    Returns
    -------
    (P, conds)
        ``P`` is ``(n_folds, n_conditions, n_features)``; ``conds`` is the
        ``(n_conditions,)`` array of surviving labels, in sorted order.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be (n_trials, n_features), got {X.shape}")
    cond = np.asarray(cond_id).ravel()
    if cond.size != X.shape[0]:
        raise ValueError(f"cond_id length {cond.size} != n_trials {X.shape[0]}")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for a cross-validated distance, got {n_folds}")
    key = np.arange(X.shape[0]) if order is None else np.asarray(order, dtype=float)

    conds_all = np.unique(cond)
    kept, blocks = [], []
    for c in conds_all:
        idx = np.flatnonzero(cond == c)
        idx = idx[np.argsort(key[idx], kind="stable")]
        folds = [idx[f::n_folds] for f in range(n_folds)]
        if min(len(f) for f in folds) < int(min_per_fold):
            continue
        kept.append(c)
        blocks.append(np.stack([X[f].mean(axis=0) for f in folds]))
    if not kept:
        return np.empty((n_folds, 0, X.shape[1])), np.asarray([])
    P = np.stack(blocks, axis=1)  # (n_folds, n_conditions, n_features)
    return P, np.asarray(kept)


def noise_covariance(X: np.ndarray, cond_id, *, shrinkage: str = "ledoit_wolf") -> np.ndarray:
    """Feature x feature noise covariance from **within-condition** residuals.

    Residuals are ``X`` minus its own condition mean, so only trial-to-trial
    noise contributes — the condition structure we want to measure is removed
    before estimating what to whiten by.

    ``shrinkage="ledoit_wolf"`` (default) is essentially mandatory in this
    regime: SNT gives ~36-48 features against ~58 trials, where the sample
    covariance is ill-conditioned and its inverse explodes. ``"none"`` returns
    the plain sample covariance (with condition-count dof correction) and
    ``"diagonal"`` returns only the variances (univariate normalization).
    """
    X = np.asarray(X, dtype=float)
    cond = np.asarray(cond_id).ravel()
    R = np.empty_like(X)
    for c in np.unique(cond):
        m = cond == c
        R[m] = X[m] - X[m].mean(axis=0, keepdims=True)

    if shrinkage == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        # residuals are already centered -> assume_centered avoids re-centering
        # across conditions, which would reintroduce condition structure.
        return np.asarray(LedoitWolf(assume_centered=True).fit(R).covariance_, dtype=float)
    dof = max(1, X.shape[0] - np.unique(cond).size)
    S = (R.T @ R) / dof
    if shrinkage == "diagonal":
        return np.diag(np.diag(S))
    if shrinkage == "none":
        return S
    raise ValueError(f"unknown shrinkage={shrinkage!r}")


def whiten_patterns(P: np.ndarray, sigma: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Multivariate noise normalization: right-multiply patterns by ``sigma^-1/2``.

    Uses a symmetric eigendecomposition with eigenvalues floored at ``eps *
    max(eigenvalue)``, so a rank-deficient covariance degrades gracefully
    instead of raising. Works on any array whose **last** axis is features.
    """
    S = np.asarray(sigma, dtype=float)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, float(eps) * max(w.max(), 1e-300))
    W = V @ np.diag(w ** -0.5) @ V.T
    return np.asarray(P, dtype=float) @ W


def crossnobis_rdm(
    X: np.ndarray,
    cond_id,
    *,
    n_folds: int = 2,
    order=None,
    sigma: np.ndarray | None = None,
    shrinkage: str = "ledoit_wolf",
    min_per_fold: int = 1,
    return_conditions: bool = False,
):
    """Cross-validated Mahalanobis (**crossnobis**) condition x condition RDM.

    ``d(i,j) = mean over ordered fold pairs (a,b), a != b, of
    ``(p_i^a - p_j^a) . Sigma^-1 (p_i^b - p_j^b) / n_features``.

    Because the two factors come from *independent* measurements, the noise
    contributions are uncorrelated and cancel in expectation: **``E[d] = 0``
    when conditions do not truly differ.** Distances are therefore continuous
    and **can be negative** — that is the estimator working, not a bug. Do not
    clip them; clipping reintroduces the positive bias crossnobis exists to
    remove. Reference: Walther et al. 2016, NeuroImage 137:188-200.

    Parameters
    ----------
    X, cond_id, n_folds, order, min_per_fold
        As in :func:`condition_patterns`.
    sigma
        Pre-computed noise covariance. If ``None`` it is estimated from ``X``
        via :func:`noise_covariance` with ``shrinkage``. Pass ``np.eye(n)`` for
        plain (unwhitened) cross-validated squared Euclidean distance.
    return_conditions
        If True, return ``(rdm, conds)`` instead of just ``rdm``.

    Returns
    -------
    np.ndarray
        Condensed upper-triangle vector, ``squareform``-compatible, matching
        :func:`model_rdm` built on the same condition order.
    """
    X = np.asarray(X, dtype=float)
    if sigma is None:
        sigma = noise_covariance(X, cond_id, shrinkage=shrinkage)
    P, conds = condition_patterns(X, cond_id, n_folds=n_folds, order=order,
                                  min_per_fold=min_per_fold)
    n_cond = P.shape[1]
    if n_cond < 2:
        out = np.empty(0)
        return (out, conds) if return_conditions else out

    Pw = whiten_patterns(P, sigma)  # (n_folds, n_cond, n_feat)
    n_feat = Pw.shape[2]
    iu = np.triu_indices(n_cond, k=1)

    acc, n_pairs = np.zeros(iu[0].size), 0
    for a in range(n_folds):
        for b in range(n_folds):
            if a == b:
                continue
            Da = Pw[a][iu[0]] - Pw[a][iu[1]]
            Db = Pw[b][iu[0]] - Pw[b][iu[1]]
            acc += np.einsum("ij,ij->i", Da, Db) / n_feat
            n_pairs += 1
    rdm = acc / max(n_pairs, 1)
    return (rdm, conds) if return_conditions else rdm


def cv_correlation_rdm(
    X: np.ndarray,
    cond_id,
    *,
    n_folds: int = 2,
    order=None,
    min_per_fold: int = 1,
    return_conditions: bool = False,
):
    """Cross-validated correlation distance (``1 - r``) between condition patterns.

    The estimator-comparison arm for :func:`crossnobis_rdm`: same conditions,
    same folds, same everything — only the distance differs. Running both makes
    a change in result attributable to the *estimator* rather than to the
    condition averaging, which a crossnobis-vs-trial-level comparison alone
    cannot separate.

    Unlike a same-data correlation RDM this crosses folds, so the diagonal is
    not trivially inflated; unlike crossnobis it is bounded and not unbiased.
    """
    P, conds = condition_patterns(X, cond_id, n_folds=n_folds, order=order,
                                  min_per_fold=min_per_fold)
    n_cond = P.shape[1]
    if n_cond < 2:
        out = np.empty(0)
        return (out, conds) if return_conditions else out

    Z = P - P.mean(axis=2, keepdims=True)
    nrm = np.linalg.norm(Z, axis=2, keepdims=True)
    nrm[nrm == 0] = 1.0
    Z = Z / nrm

    iu = np.triu_indices(n_cond, k=1)
    acc, n_pairs = np.zeros(iu[0].size), 0
    for a in range(n_folds):
        for b in range(n_folds):
            if a == b:
                continue
            acc += 1.0 - np.einsum("ij,ij->i", Z[a][iu[0]], Z[b][iu[1]])
            n_pairs += 1
    rdm = acc / max(n_pairs, 1)
    return (rdm, conds) if return_conditions else rdm


def _rank_z(v: np.ndarray, *, rank: bool) -> np.ndarray:
    from scipy.stats import rankdata

    x = np.asarray(v, dtype=float)
    if rank:
        x = rankdata(x)
    x = x - x.mean()
    sd = x.std()
    return x / sd if sd > 0 else x


def rdm_regression(
    neural: np.ndarray,
    model_rdms: dict,
    *,
    rank: bool = True,
    targets=None,
) -> dict:
    """Joint regression of a neural RDM on several model RDMs.

    This is the multi-predictor generalization of ``rsa(..., partial=...)``,
    which can only remove ONE nuisance RDM. When candidate geometries are
    mutually collinear, a one-at-a-time partial cannot say which one the data
    support — every model looks significant because they proxy each other. The
    joint fit, the VIFs, and the unique/shared partition are what make the
    question answerable (or show that it is not).

    ``rank=True`` (default) rank-transforms every RDM first, so betas are
    Spearman-style and robust to the monotone-but-nonlinear relation typical
    between neural dissimilarity and a model distance.

    Parameters
    ----------
    neural
        Condensed neural RDM.
    model_rdms
        ``{name: condensed RDM}``. All must match ``neural`` in length.
    targets
        Names treated as the geometries of interest (default: all). Their joint
        unique variance is reported as ``r2_unique_targets`` — the omnibus "is
        *any* of these geometries present beyond the rest" statistic, which
        stays interpretable even when individual betas are split by collinearity.

    Returns
    -------
    dict
        ``beta`` / ``vif`` / ``r2_unique`` (per predictor), plus ``r2``,
        ``r2_adj``, ``r2_unique_targets``, ``max_vif``, ``n_pairs``, ``names``.
    """
    y = np.asarray(neural, dtype=float).ravel()
    names = list(model_rdms.keys())
    if not names:
        raise ValueError("model_rdms is empty")
    for k in names:
        if np.asarray(model_rdms[k]).size != y.size:
            raise ValueError(
                f"model RDM {k!r} has length {np.asarray(model_rdms[k]).size}, neural has {y.size}"
            )
    tgt = list(names) if targets is None else [t for t in targets if t in names]

    yz = _rank_z(y, rank=rank)
    M = np.column_stack([_rank_z(model_rdms[k], rank=rank) for k in names])

    def _r2(design_cols):
        if not len(design_cols):
            return 0.0
        A = np.column_stack([np.ones(yz.size), M[:, design_cols]])
        beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
        resid = yz - A @ beta
        sst = float(yz @ yz)
        return 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0

    A_full = np.column_stack([np.ones(yz.size), M])
    beta_full, *_ = np.linalg.lstsq(A_full, yz, rcond=None)
    all_cols = list(range(len(names)))
    r2_full = _r2(all_cols)
    n, p = yz.size, len(names)
    r2_adj = 1.0 - (1.0 - r2_full) * (n - 1) / max(n - p - 1, 1)

    beta, vif, r2_unique = {}, {}, {}
    for i, k in enumerate(names):
        beta[k] = float(beta_full[i + 1])
        others = [j for j in all_cols if j != i]
        r2_unique[k] = float(max(0.0, r2_full - _r2(others)))
        if others:
            A_o = np.column_stack([np.ones(n), M[:, others]])
            b_o, *_ = np.linalg.lstsq(A_o, M[:, i], rcond=None)
            res = M[:, i] - A_o @ b_o
            sst_i = float(M[:, i] @ M[:, i])
            r2_i = 1.0 - float(res @ res) / sst_i if sst_i > 0 else 0.0
            vif[k] = float(1.0 / max(1.0 - r2_i, 1e-12))
        else:
            vif[k] = 1.0

    tgt_cols = [names.index(t) for t in tgt]
    non_tgt = [j for j in all_cols if j not in tgt_cols]
    return {
        "beta": beta,
        "vif": vif,
        "r2_unique": r2_unique,
        "r2": float(r2_full),
        "r2_adj": float(r2_adj),
        "r2_unique_targets": float(max(0.0, r2_full - _r2(non_tgt))),
        "max_vif": float(max(vif.values())) if vif else 1.0,
        "n_pairs": int(yz.size),
        "names": names,
    }


def rdm_regression_perm(
    neural: np.ndarray,
    model_rdms: dict,
    *,
    rank: bool = True,
    targets=None,
    null: str = "circular",
    n_perm: int = 300,
    rng: np.random.Generator | None = None,
) -> dict:
    """:func:`rdm_regression` with a condition-relabelling null.

    The null permutes the **conditions of the neural RDM** (via ``squareform``)
    and refits the *same* regression, so the observed statistic and its null
    share one code path — the project's same-estimator rule. Per-predictor
    ``debiased = beta - null_mean`` is the quantity to carry to a group test;
    the raw beta is not, because a finite-sample floor differs between designs.

    ``null="circular"`` is exhaustive over contiguous shifts and preserves
    drift autocorrelation. At condition level this yields only
    ``n_conditions - 1`` samples: adequate for debiasing, **too few for a
    per-subject p-value**. Shift trial-level labels and rebuild the conditions
    if a per-subject p is needed.

    Returns
    -------
    dict
        ``obs`` (the full :func:`rdm_regression` result), and per-predictor
        ``beta`` / ``null_mean`` / ``debiased`` / ``p``, plus the same four
        fields for ``r2`` and ``r2_unique_targets`` under key ``omnibus``.
    """
    from scipy.spatial.distance import squareform

    y = np.asarray(neural, dtype=float).ravel()
    obs = rdm_regression(y, model_rdms, rank=rank, targets=targets)
    names = obs["names"]

    sq = squareform(y, checks=False)
    n = sq.shape[0]
    if null == "circular":
        perms = circular_shift_indices(n, n_perm)
    elif null == "random":
        if rng is None:
            rng = np.random.default_rng()
        perms = np.stack([rng.permutation(n) for _ in range(int(n_perm))]) if n >= 3 \
            else np.empty((0, n), dtype=int)
    else:
        raise ValueError(f"null must be 'random' or 'circular', got {null!r}")

    nb = {k: [] for k in names}
    nr2, nrt = [], []
    for idx in perms:
        yp = squareform(sq[np.ix_(idx, idx)], checks=False)
        f = rdm_regression(yp, model_rdms, rank=rank, targets=targets)
        for k in names:
            nb[k].append(f["beta"][k])
        nr2.append(f["r2"])
        nrt.append(f["r2_unique_targets"])

    def _cell(o, arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return {"obs": float(o), "null_mean": np.nan, "debiased": np.nan,
                    "p": np.nan, "n_null": 0}
        return {"obs": float(o), "null_mean": float(a.mean()),
                "debiased": float(o - a.mean()),
                "p": (1 + int(np.sum(a >= o))) / (a.size + 1), "n_null": int(a.size)}

    return {
        "obs": obs,
        "per_model": {k: _cell(obs["beta"][k], nb[k]) for k in names},
        "omnibus": {"r2": _cell(obs["r2"], nr2),
                    "r2_unique_targets": _cell(obs["r2_unique_targets"], nrt)},
        "null": null,
        "n_null": int(len(perms)),
    }
