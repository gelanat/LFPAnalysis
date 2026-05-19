"""Phase-amplitude coupling (PAC) utilities — unified wrapper over multiple
backends for multi-method robustness.

Why two backends. Single-method PAC results are notoriously sensitive to
implementation choices (filter design, surrogate construction, modulation
index variant). The snt_lfp paper plan (Phase 3A) explicitly calls for
**convergence across `tensorpac` and `pactools`**: report PAC where both
agree, flag method-dependence where they don't.

What this module exposes:

    compute_pac(signal, fs, phase_band, amp_band, method=..., n_surr=200,
                seed=2025) -> dict

The signal is `(n_epochs, n_samples)` (or 1D for a single trial). Both
backends are configured for single-band MI rather than a comodulogram
sweep — most snt_lfp analyses target a specific (phase, amp) band pair
(e.g. theta 2-8 Hz × low-gamma 30-70 Hz).

Backends
--------
- `tensorpac_mi` — Tort (2010) Modulation Index via `tensorpac.Pac`.
  Uses `idpac=(2, 2, 1)`: Tort MI × swap-amp-time-blocks surrogate ×
  subtract-mean normalization. Reasonably fast; ships with several
  surrogate strategies.
- `pactools_mi` — Tort (2010) Modulation Index via
  `pactools.Comodulogram(method='tort')`. Built on top of
  `mne.filter` ; surrogate construction differs from tensorpac
  (time-shift by minimum_shift), making it a useful comparator.
- `pactools_glm` — GLM-PAC (Penny & Duzel 2008) via
  `pactools.Comodulogram(method='duprelatour')`. Model-based, robust
  to amplitude bias. Slower but conceptually independent of MI methods.

Returns
-------
A dict with the same keys regardless of backend:

    {
        "pac": float,                # observed MI
        "surrogate_mi": np.ndarray,  # (n_surr,) surrogate MIs
        "z_pac": float,              # z = (observed - mean(surr)) / std(surr)
        "p_value": float,            # empirical one-sided (right) p
        "method": str,               # the backend used
        "phase_band": (float, float),
        "amp_band": (float, float),
        "fs": float,
        "n_epochs": int,
        "n_samples": int,
        "backend_meta": dict,        # backend-specific extras
    }

Conventions
-----------
- All bands are passed as (lo, hi) tuples in Hz.
- Surrogate p-value is one-sided positive (PAC is non-negative by
  construction; sign of "directionality" comes from cross-region
  pairing, not from MI sign).
- Random seed is propagated where backends support it (tensorpac
  doesn't expose a seed; pactools does via `random_state`).

References
----------
- Tort et al. (2010) J. Neurophysiol. 104:1195-1210 (MI).
- Penny & Duzel (2008) Brain Topogr. 22:158-176 (GLM-PAC).
- Combrisson et al. (2020) PLOS Comput Biol 16:e1008302 (tensorpac).
- Dupré la Tour et al. (2017) PLOS Comput Biol 13:e1005893 (pactools).
"""

from __future__ import annotations

from typing import Any

import numpy as np

ArrayLike = np.ndarray


def _coerce_signal(signal: ArrayLike) -> tuple[ArrayLike, int, int]:
    """Return (signal_2d, n_epochs, n_samples) for `(n_epochs, n_samples)` or 1D."""
    arr = np.asarray(signal)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D signal, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal contains non-finite values; mask or interpolate before PAC")
    return arr, arr.shape[0], arr.shape[1]


def _compute_tensorpac(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surr: int = 200,
    seed: int | None = 2025,  # unused; tensorpac doesn't expose RNG, kept for interface parity
) -> dict[str, Any]:
    """Tensorpac Tort MI with swap-amp-time-blocks surrogate."""
    from tensorpac import Pac

    sig_2d, n_ep, n_samp = _coerce_signal(signal)
    # idpac = (PAC method, surrogate method, normalization method).
    #   PAC=2  → Tort Modulation Index
    #   SUR=2  → swap amplitude time blocks (most rigorous for stationarity)
    #   NORM=0 → no normalization (we'll z-score externally for parity with pactools)
    pac = Pac(
        idpac=(2, 2, 0),
        f_pha=[phase_band[0], phase_band[1]],
        f_amp=[amp_band[0], amp_band[1]],
        dcomplex="hilbert",
    )
    # filterfit returns shape (n_amp, n_pha, n_epochs) for the observed MI;
    # surrogates have shape (n_surr, n_amp, n_pha, n_epochs).
    observed = pac.filterfit(fs, sig_2d, n_perm=n_surr, random_state=seed)
    # Reduce: single-band → take element [0, 0, :]; mean across epochs gives
    # the pooled observed MI.
    mi_per_epoch = np.asarray(observed)[0, 0, :]  # (n_epochs,)
    observed_mi = float(np.mean(mi_per_epoch))
    surrogate = np.asarray(pac.surrogates)  # (n_surr, n_amp, n_pha, n_epochs)
    if surrogate.size == 0:
        surrogate_mi = np.full(n_surr, np.nan)
    else:
        # Pool over epochs within each surrogate → (n_surr,)
        surrogate_mi = surrogate[:, 0, 0, :].mean(axis=1)
    z_pac = (
        (observed_mi - surrogate_mi.mean()) / (surrogate_mi.std(ddof=1) + 1e-12)
        if np.all(np.isfinite(surrogate_mi))
        else np.nan
    )
    p_value = float((surrogate_mi >= observed_mi).mean()) if np.all(np.isfinite(surrogate_mi)) else np.nan
    return {
        "pac": observed_mi,
        "surrogate_mi": surrogate_mi,
        "z_pac": float(z_pac),
        "p_value": p_value,
        "method": "tensorpac_mi",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "idpac": (2, 2, 0),
            "dcomplex": "hilbert",
            "mi_per_epoch": mi_per_epoch,
        },
    }


def _compute_pactools(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    method: str = "tort",  # 'tort' or 'duprelatour'
    n_surr: int = 200,
    seed: int | None = 2025,
) -> dict[str, Any]:
    """Pactools Comodulogram restricted to a single (phase, amp) band pair."""
    from pactools import Comodulogram

    sig_2d, n_ep, n_samp = _coerce_signal(signal)
    phase_lo, phase_hi = phase_band
    amp_lo, amp_hi = amp_band
    # Comodulogram sweeps over `low_fq_range × high_fq_range`; we want a single
    # cell so we set narrow ranges at the band centers and matched widths.
    low_fq_range = [(phase_lo + phase_hi) / 2.0]
    high_fq_range = [(amp_lo + amp_hi) / 2.0]
    low_fq_width = max(phase_hi - phase_lo, 0.5)
    high_fq_width = max(amp_hi - amp_lo, 0.5)
    cmd = Comodulogram(
        fs=fs,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        method=method,
        n_surrogates=n_surr,
        random_state=seed,
        progress_bar=False,
        n_jobs=1,
    )
    # Pactools expects a 1D or (n_trials, n_samples) array; the latter is
    # treated as concatenated trials internally for Comodulogram fits.
    cmd.fit(sig_2d if n_ep > 1 else sig_2d[0])
    # comod_ shape: (n_low, n_high). We have one cell.
    observed_mi = float(cmd.comod_[0, 0])
    # surrogates_ shape: (n_surr, n_low, n_high)
    if cmd.surrogates_ is None or cmd.surrogates_.size == 0:
        surrogate_mi = np.full(n_surr, np.nan)
    else:
        surrogate_mi = np.asarray(cmd.surrogates_)[:, 0, 0]
    z_pac = (
        (observed_mi - surrogate_mi.mean()) / (surrogate_mi.std(ddof=1) + 1e-12)
        if np.all(np.isfinite(surrogate_mi))
        else np.nan
    )
    p_value = float((surrogate_mi >= observed_mi).mean()) if np.all(np.isfinite(surrogate_mi)) else np.nan
    return {
        "pac": observed_mi,
        "surrogate_mi": surrogate_mi,
        "z_pac": float(z_pac),
        "p_value": p_value,
        "method": f"pactools_{method}",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "method": method,
            "low_fq_range": low_fq_range,
            "high_fq_range": high_fq_range,
            "low_fq_width": low_fq_width,
            "high_fq_width": high_fq_width,
        },
    }


def compute_pac(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    method: str = "tensorpac_mi",
    n_surr: int = 200,
    seed: int | None = 2025,
) -> dict[str, Any]:
    """Compute phase-amplitude coupling with the requested backend.

    Parameters
    ----------
    signal : ndarray
        Shape `(n_epochs, n_samples)` or 1D `(n_samples,)`. Continuous LFP
        in microvolts (or whatever — units don't matter for MI, only ratios).
    fs : float
        Sample rate in Hz.
    phase_band : (lo, hi)
        Low-frequency band (the "phase" donor), Hz.
    amp_band : (lo, hi)
        High-frequency band (the "amplitude" recipient), Hz.
    method : str
        One of:
        - "tensorpac_mi" : Tort MI via tensorpac (fastest, swap-blocks surrogate)
        - "pactools_mi" : Tort MI via pactools (independent implementation)
        - "pactools_glm" : GLM-PAC via pactools (Penny & Duzel 2008)
    n_surr : int
        Number of surrogate permutations for the null distribution.
    seed : int | None
        Random seed (used where the backend supports it).
    """
    if method == "tensorpac_mi":
        return _compute_tensorpac(signal, fs, phase_band, amp_band, n_surr=n_surr, seed=seed)
    if method == "pactools_mi":
        return _compute_pactools(signal, fs, phase_band, amp_band, method="tort", n_surr=n_surr, seed=seed)
    if method == "pactools_glm":
        return _compute_pactools(signal, fs, phase_band, amp_band, method="duprelatour", n_surr=n_surr, seed=seed)
    raise ValueError(
        f"unknown method: {method!r} "
        "(expected 'tensorpac_mi' | 'pactools_mi' | 'pactools_glm')"
    )


def synthetic_pac_signal(
    fs: float = 500.0,
    duration_s: float = 4.0,
    phase_freq: float = 6.0,
    amp_freq: float = 60.0,
    coupling_strength: float = 0.5,
    noise_level: float = 0.2,
    seed: int | None = 2025,
) -> np.ndarray:
    """Build a synthetic phase-amplitude-coupled signal for smoke tests.

    The amplitude envelope of the high-frequency carrier is modulated by
    the trough of the low-frequency phase: amp = 1 + k * (1 - cos(phi)) / 2.
    `coupling_strength=0` → uncoupled; `coupling_strength=1` → maximal
    modulation. Useful for verifying that `compute_pac` returns a high MI
    for coupled signals and a low MI for uncoupled.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    phase = 2 * np.pi * phase_freq * t
    modulator = 1.0 + coupling_strength * (1.0 - np.cos(phase)) / 2.0
    carrier = np.sin(2 * np.pi * amp_freq * t)
    low = np.sin(phase)
    signal = low + modulator * carrier
    signal = signal + noise_level * rng.standard_normal(n)
    return signal
