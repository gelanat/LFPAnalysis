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
    "picks_in_region",
    "per_trial_band_power",
    "feature_matrix",
    "per_trial_band_power_timeresolved",
    "feature_matrix_timeresolved",
    "per_trial_phase_binned_hfa",
    "feature_matrix_phase_binned",
    "neural_rdm",
    "model_rdm",
    "rsa",
    "cross_region_rsa",
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
