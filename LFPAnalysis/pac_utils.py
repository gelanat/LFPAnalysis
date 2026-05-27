"""Phase-amplitude coupling (PAC) utilities — unified wrapper over multiple
backends for multi-method robustness.

Why multiple backends. Single-method PAC results are notoriously sensitive
to implementation choices (filter design, surrogate construction,
modulation index variant). The snt_lfp paper plan (Phase 3A) explicitly
calls for **convergence across implementations**: report PAC where they
agree, flag method-dependence where they don't.

Backend choice — 2026-05-20 finding. During the v2 verification pass we
discovered that tensorpac's surrogate methods (`idpac=(2,*,0)`) produce
surrogate distributions that **collapse to the observed MI** on multi-
trial data with consistent within-trial coupling. The z-scores from
those backends are therefore unreliable for trial-locked PAC. The
project now treats the hand-rolled `tort_block_resample` method
(`_compute_tort_block_resample`, based on Tort 2010 + the connectivity
skill's reference implementation) as the **primary** estimator and uses
`tensorpac_mi` / `pactools_mi` / `pactools_glm` as sensitivity checks.
See docs/methods/pac_methods_20260520.md.

What this module exposes:

    compute_pac(signal, fs, phase_band, amp_band, method=..., n_surr=200,
                seed=2025) -> dict

The signal is `(n_epochs, n_samples)` (or 1D for a single trial). All
backends are configured for single-band MI rather than a comodulogram
sweep — most snt_lfp analyses target a specific (phase, amp) band pair
(e.g. theta 4-8 Hz × low-gamma 30-70 Hz).

Backends
--------
- `tort_block_resample` **(default, primary)** — hand-rolled Tort (2010)
  Modulation Index with within-trial block-resample amplitude
  surrogate. Concatenates per-epoch analytic signals after edge masking
  so block boundaries do not align with epoch boundaries, then
  shuffles ~10 contiguous blocks of the amplitude envelope. Validates
  against synthetic PAC where library backends do not.
- `tensorpac_mi` — Tort (2010) MI via `tensorpac.Pac` with
  `idpac=(2, 2, 0)`: swap-amp-time-blocks surrogate, no internal
  normalisation (we z-score externally). Sensitivity check only — see
  the 2026-05-20 finding note above.
- `pactools_mi` — Tort (2010) MI via
  `pactools.Comodulogram(method='tort')`. Now per-epoch averaged
  (2026-05-20) so the statistic matches the other backends. Surrogate
  construction is pactools' `minimum_shift`.
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


def _check_pac_bandwidth(
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
) -> None:
    """Aru et al. 2015 criterion — amp band must capture the carrier's side bands.

    For a phase carrier at frequency `f_p`, the amplitude modulation produces
    side bands at `f_a ± f_p` in the high-frequency signal. The amplitude
    band must be wide enough to contain those side bands: bandwidth ≥ 2·f_p.

    A narrow amplitude band that violates this criterion will systematically
    miss the modulation side bands and the PAC metric becomes invalid
    (Aru, J. et al. 2015, *Curr Opin Neurobiol* 31:51–61, §"Bandwidth").

    Raises
    ------
    ValueError
        If `amp_band` is narrower than `2 × phase_band[1]`.
    """
    amp_width = amp_band[1] - amp_band[0]
    phase_max = phase_band[1]
    if amp_width < 2 * phase_max:
        raise ValueError(
            f"amp_band width ({amp_width:.1f} Hz) must be >= 2 * phase_band upper "
            f"({2 * phase_max:.1f} Hz) for valid PAC "
            f"(Aru et al. 2015, Curr Opin Neurobiol 31:51-61, §Bandwidth). "
            f"Got phase_band={phase_band}, amp_band={amp_band}."
        )


def _z_and_p(observed_mi: float, surrogate_mi: np.ndarray) -> tuple[float, float]:
    """Compute (z_pac, one-sided right p-value) from observed and surrogate MIs.

    Returns (nan, nan) when fewer than 80 % of surrogates are finite —
    a stricter threshold than the original `np.all(finite)` check that
    would discard the whole channel for a single NaN surrogate.
    """
    surr = np.asarray(surrogate_mi, dtype=float)
    finite = np.isfinite(surr)
    if finite.mean() < 0.8 or finite.sum() < 2:
        return float("nan"), float("nan")
    surr_finite = surr[finite]
    sd = float(np.std(surr_finite, ddof=1))
    if sd <= 0:
        return float("nan"), float("nan")
    z = float((observed_mi - float(np.mean(surr_finite))) / sd)
    # One-sided right p; add +1 to numerator and denominator for proper
    # permutation p-value (avoids p == 0; North et al. 2002).
    p = float((np.sum(surr_finite >= observed_mi) + 1) / (surr_finite.size + 1))
    return z, p


def _cliffs_delta(x: ArrayLike, y: ArrayLike) -> float:
    """Cliff's δ effect size = (#{x>y} - #{x<y}) / (n_x · n_y).

    Returns a value in [-1, 1]. δ ≈ 0 → x and y interchangeable;
    δ → 1 → x stochastically larger than y. Non-parametric, robust to
    non-normal distributions (Romano et al. 2006).

    Used here to quantify how reliably the observed per-epoch MI
    distribution sits above the surrogate distribution, complementing
    the z-score with a distribution-free effect-size measure.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # Vectorised pairwise comparison — fine for our scale (n_epochs ~ 60,
    # n_surr ~ 1000 → 60 000 comparisons per channel).
    gt = int(np.sum(x[:, None] > y[None, :]))
    lt = int(np.sum(x[:, None] < y[None, :]))
    return float((gt - lt) / (x.size * y.size))


def _compute_tensorpac(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surr: int = 200,
    seed: int | None = 2025,  # unused; tensorpac doesn't expose RNG, kept for interface parity
) -> dict[str, Any]:
    """Tensorpac Tort MI — sensitivity check, NOT the primary backend.

    Uses `idpac=(2, 2, 0)`: Tort MI × swap-amp-time-blocks surrogate ×
    no normalisation (we z-score externally). Block-shuffle surrogate
    was retained over the briefly-tried trial-shuffle (`idpac=(2,1,0)`)
    because both produced degenerate (collapsed) surrogate distributions
    on synthetic data with consistent within-trial coupling, and the
    block variant at least breaks within-trial structure in principle.

    The 2026-05-20 verification pass found this backend's z-scores
    cannot be trusted at face value on trial-locked PAC. Use
    `tort_block_resample` as the primary; report this backend as a
    cross-check only.
    """
    from tensorpac import Pac

    sig_2d, n_ep, n_samp = _coerce_signal(signal)
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
    z_pac, p_value = _z_and_p(observed_mi, surrogate_mi)
    return {
        "pac": observed_mi,
        "mi_per_epoch": mi_per_epoch,
        "mi_std_across_epochs": float(np.std(mi_per_epoch, ddof=1)) if n_ep > 1 else float("nan"),
        "surrogate_mi": surrogate_mi,
        "z_pac": z_pac,
        "p_value": p_value,
        "cliff_delta": _cliffs_delta(mi_per_epoch, surrogate_mi),
        "method": "tensorpac_mi",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "idpac": (2, 2, 0),
            "surrogate_kind": "swap_amp_time_blocks_bahramisharif2013",
            "dcomplex": "hilbert",
            "warning": (
                "Library surrogate distribution can collapse onto observed MI "
                "for trial-locked PAC; treat z_pac/p_value as sensitivity-only."
            ),
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
    """Pactools Comodulogram restricted to a single (phase, amp) band pair.

    Per-epoch MI averaging (2026-05-20 rewrite). Each epoch is fit
    independently and the MI / surrogate MIs are averaged across epochs.
    This matches the statistic computed by `_compute_tensorpac`
    (which already returns per-epoch MI from `pac.filterfit`) so that
    the multi-method convergence check at Phase 3A compares like-with-
    like rather than per-epoch-MI vs concatenated-MI.

    Cost: O(n_epochs) `Comodulogram.fit` calls per channel — roughly
    `n_epochs ×` the previous wall-time for pactools cells.
    """
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

    mi_per_epoch = np.full(n_ep, np.nan)
    surr_per_epoch = np.full((n_ep, n_surr), np.nan)
    for i in range(n_ep):
        # Distinct seed per epoch so surrogates are independent across
        # epochs (otherwise pactools draws the same shuffle every time).
        # The seed is deterministic given the base seed and epoch index,
        # so the run is reproducible.
        epoch_seed = None if seed is None else int(seed) + i
        cmd = Comodulogram(
            fs=fs,
            low_fq_range=low_fq_range,
            low_fq_width=low_fq_width,
            high_fq_range=high_fq_range,
            high_fq_width=high_fq_width,
            method=method,
            n_surrogates=n_surr,
            random_state=epoch_seed,
            progress_bar=False,
            n_jobs=1,
        )
        cmd.fit(sig_2d[i])
        mi_per_epoch[i] = float(cmd.comod_[0, 0])
        if cmd.surrogates_ is not None and cmd.surrogates_.size > 0:
            surr_per_epoch[i, :] = np.asarray(cmd.surrogates_)[:, 0, 0]

    observed_mi = float(np.nanmean(mi_per_epoch))
    # Average per-surrogate-index across epochs. This yields the same
    # statistic the observed MI is — mean across epochs — so the z-score
    # compares like distributions.
    if np.all(np.isnan(surr_per_epoch)):
        surrogate_mi = np.full(n_surr, np.nan)
    else:
        surrogate_mi = np.nanmean(surr_per_epoch, axis=0)
    z_pac, p_value = _z_and_p(observed_mi, surrogate_mi)
    return {
        "pac": observed_mi,
        "mi_per_epoch": mi_per_epoch,
        "mi_std_across_epochs": float(np.nanstd(mi_per_epoch, ddof=1)) if n_ep > 1 else float("nan"),
        "surrogate_mi": surrogate_mi,
        "z_pac": z_pac,
        "p_value": p_value,
        "cliff_delta": _cliffs_delta(mi_per_epoch, surrogate_mi),
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
            "surrogate_kind": "pactools_default_minimum_shift",
            "per_epoch_fit": True,
        },
    }


# -----------------------------------------------------------------------------
# Primary backend: hand-rolled Tort MI + block-resample amplitude surrogate
# -----------------------------------------------------------------------------
# Based on the `connectivity` skill's reference implementation and Tort 2010.
# Validates against synthetic PAC where library backends do not (see module
# docstring, 2026-05-20 finding).


def _tort_mi(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
    """Tort modulation index — KL divergence of amp-by-phase from uniform.

    `phase` in (-pi, pi]; `amplitude` non-negative; both 1D, equal length.
    Returns MI in [0, 1]. Uniform → 0; perfectly peaked → 1.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    bin_means = np.zeros(n_bins)
    for j in range(n_bins):
        sel = idx == j
        bin_means[j] = float(amplitude[sel].mean()) if sel.any() else 0.0
    s = bin_means.sum()
    if s <= 0:
        return float("nan")
    p = bin_means / s
    p = np.where(p > 0, p, 1e-12)
    h = -np.sum(p * np.log(p))
    return float((np.log(n_bins) - h) / np.log(n_bins))


def _filter_hilbert(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    order: int = 4,
    filter_kind: str = "butter",
) -> np.ndarray:
    """Zero-phase band-pass + Hilbert. Returns analytic signal.

    `filter_kind='butter'` (default): Butterworth IIR via
    `scipy.signal.butter` + `sosfiltfilt`. The `order` argument applies
    here.

    `filter_kind='fir'`: windowed-sinc FIR via `mne.filter.filter_data`
    with `phase='zero'`, `fir_design='firwin'`, `fir_window='hamming'`.
    MNE auto-derives the filter length from `fs` and the transition
    band; the `order` argument is ignored.
    """
    from scipy.signal import hilbert

    low, high = band
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"band {band} must satisfy 0 < low < high < fs/2={nyq}")

    if filter_kind == "butter":
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x)
    elif filter_kind == "fir":
        import mne
        x_arr = np.asarray(x, dtype=np.float64)
        x_in = x_arr[np.newaxis, :] if x_arr.ndim == 1 else x_arr
        xf2 = mne.filter.filter_data(
            x_in, sfreq=fs, l_freq=low, h_freq=high,
            method="fir", fir_design="firwin", phase="zero",
            fir_window="hamming", verbose=False,
        )
        xf = xf2[0] if x_arr.ndim == 1 else xf2
    else:
        raise ValueError(
            f"unknown filter_kind={filter_kind!r}; expected 'butter' or 'fir'"
        )

    return hilbert(xf)


def _block_resample_surrogate_mi(
    phase: np.ndarray,
    amplitude: np.ndarray,
    n_blocks: int = 10,
    n_perm: int = 200,
    n_bins: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-shuffle MI null distribution by block-resampling the amplitude.

    Splits `amplitude` into `n_blocks` contiguous chunks, shuffles their
    order, re-pairs with the original `phase`, recomputes MI. Preserves
    within-block amplitude autocorrelation (so the surrogate's MI baseline
    inherits the observed signal's amplitude statistics) while breaking
    the phase–amplitude alignment.

    With `n_blocks=10` on a 3-s, 6-Hz-centred theta signal, each block
    spans roughly 2 theta cycles — small enough to break within-trial
    coupling, large enough to preserve local envelope structure.

    Returns (n_perm,) array of surrogate MI values.
    """
    if rng is None:
        rng = np.random.default_rng()
    amp = np.asarray(amplitude).ravel()
    phi = np.asarray(phase).ravel()
    n = amp.size
    block_size = n // n_blocks
    if block_size < 2:
        raise ValueError(f"signal too short (n={n}) for n_blocks={n_blocks}")
    boundaries = [block_size * i for i in range(n_blocks)] + [n]
    blocks = [amp[boundaries[i]:boundaries[i + 1]] for i in range(n_blocks)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate([blocks[k] for k in order])
        m = shuffled.size
        null[i] = _tort_mi(phi[:m], shuffled, n_bins=n_bins)
    return null


def _compute_tort_block_resample(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surr: int = 200,
    seed: int | None = 2025,
    n_bins: int = 18,
    n_blocks: int = 10,
    filter_edge_s: float = 0.5,
    filter_kind: str = "butter",
) -> dict[str, Any]:
    """Tort 2010 MI with within-trial block-resample amplitude surrogate.

    Concatenates per-epoch analytic signals (with edge masking) so that
    block boundaries do not coincide with epoch boundaries — the surrogate
    can shuffle across epoch boundaries, breaking the coupling everywhere.

    Edge masking. Each epoch loses `filter_edge_s` seconds from each side
    of the Hilbert envelope (default 0.5 s, suitable for 4-Hz lower band
    edge → 2 cycles of edge mask). Set to a value matched to your phase
    band's lower edge: `≥ 3 / phase_band[0]` for safety.

    Returns the same dict shape as the other backends, with
    `backend_meta` recording `n_bins`, `n_blocks`, `filter_edge_s`, and
    the surrogate kind.
    """
    sig_2d, n_ep, n_samp = _coerce_signal(signal)
    edge = int(round(filter_edge_s * fs))
    if 2 * edge >= n_samp:
        raise ValueError(
            f"filter_edge_s={filter_edge_s} too large for n_samp={n_samp} at fs={fs}; "
            "reduce filter_edge_s or use a longer analysis window."
        )

    # Build per-epoch phase / amplitude, drop edges, store mi-per-epoch
    # and one concatenated time series for the surrogate null.
    mi_per_epoch = np.empty(n_ep)
    phi_concat: list[np.ndarray] = []
    amp_concat: list[np.ndarray] = []
    for i in range(n_ep):
        try:
            analytic_phase = _filter_hilbert(sig_2d[i], fs, phase_band, filter_kind=filter_kind)
            analytic_amp = _filter_hilbert(sig_2d[i], fs, amp_band, filter_kind=filter_kind)
        except Exception:
            mi_per_epoch[i] = np.nan
            continue
        phi = np.angle(analytic_phase)[edge:-edge or None]
        amp = np.abs(analytic_amp)[edge:-edge or None]
        if phi.size < n_blocks * 2:
            mi_per_epoch[i] = np.nan
            continue
        mi_per_epoch[i] = _tort_mi(phi, amp, n_bins=n_bins)
        phi_concat.append(phi)
        amp_concat.append(amp)

    observed_mi = float(np.nanmean(mi_per_epoch))

    if not phi_concat:
        surrogate_mi = np.full(n_surr, np.nan)
        z_pac, p_value = float("nan"), float("nan")
        cliff = float("nan")
    else:
        phi_all = np.concatenate(phi_concat)
        amp_all = np.concatenate(amp_concat)
        rng = np.random.default_rng(seed)
        surrogate_mi = _block_resample_surrogate_mi(
            phi_all, amp_all,
            n_blocks=n_blocks, n_perm=n_surr, n_bins=n_bins, rng=rng,
        )
        # Also compute the concatenated observed MI as a sanity statistic;
        # it agrees with mean(mi_per_epoch) up to edge effects and is the
        # statistic the surrogate distribution is built against.
        z_pac, p_value = _z_and_p(_tort_mi(phi_all, amp_all, n_bins=n_bins), surrogate_mi)
        cliff = _cliffs_delta(mi_per_epoch[np.isfinite(mi_per_epoch)], surrogate_mi)

    return {
        "pac": observed_mi,
        "mi_per_epoch": mi_per_epoch,
        "mi_std_across_epochs": float(np.nanstd(mi_per_epoch, ddof=1)) if n_ep > 1 else float("nan"),
        "surrogate_mi": surrogate_mi,
        "z_pac": z_pac,
        "p_value": p_value,
        "cliff_delta": cliff,
        "method": "tort_block_resample",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "n_bins": int(n_bins),
            "n_blocks": int(n_blocks),
            "filter_edge_s": float(filter_edge_s),
            "filter_kind": str(filter_kind),
            "surrogate_kind": "block_resample_amplitude_concatenated_across_epochs",
        },
    }


def _compute_tort_trial_shuffle(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surr: int = 200,
    seed: int | None = 2025,
    n_bins: int = 18,
    filter_edge_s: float = 0.5,
    filter_kind: str = "butter",
) -> dict[str, Any]:
    """Tort 2010 MI with trial-shuffle surrogate, matching Daume et al. 2024.

    Per Daume 2024 (Nature, doi:10.1038/s41586-024-07309-z): "computed 200
    surrogate MIs by randomly combining the phase and amplitude signals
    from different trials". We add the modern guards on top (filter-edge
    masking; bandwidth check upstream in `compute_pac`).

    Caveat: in our 2026-05-20 synthetic verification (see
    `docs/methods/pac_methods_20260520.md` §3.1), trial-shuffle was
    degenerate when within-trial coupling was identical across trials.
    On real iEEG with trial-to-trial variability in coupling strength
    and preferred phase, it should be discriminative — this is one of
    the empirical questions the Daume replication answers.

    Implementation: build per-epoch (phi_i, amp_i). Observed MI per
    epoch = `_tort_mi(phi_i, amp_i)`. Per surrogate, draw a permutation
    `pi` over trial indices, compute MI per trial-pair as
    `_tort_mi(phi_i, amp_{pi(i)})`, average across trials → one
    surrogate MI value. Repeat `n_surr` times. The test statistic
    (observed mean MI vs surrogate mean MI) is per-trial averaged on
    both sides, so the two distributions compare like-for-like.

    Returns the same dict shape as `_compute_tort_block_resample`,
    with `method='trial_shuffle'` and `backend_meta.surrogate_kind`
    set to identify the trial-shuffle null.
    """
    sig_2d, n_ep, n_samp = _coerce_signal(signal)
    if n_ep < 2:
        raise ValueError(
            f"trial-shuffle surrogate requires n_epochs >= 2; got n_epochs={n_ep}. "
            "For single-epoch / continuous-data PAC, use 'tort_block_resample'."
        )
    edge = int(round(filter_edge_s * fs))
    if 2 * edge >= n_samp:
        raise ValueError(
            f"filter_edge_s={filter_edge_s} too large for n_samp={n_samp} at fs={fs}; "
            "reduce filter_edge_s or use a longer analysis window."
        )

    # Build per-epoch phi / amp arrays (each one already edge-masked).
    # We store them in a homogeneous (n_kept, n_samples_kept) layout so the
    # surrogate loop can index by permutation cheaply.
    keep_indices: list[int] = []
    phi_per_epoch: list[np.ndarray] = []
    amp_per_epoch: list[np.ndarray] = []
    mi_per_epoch = np.full(n_ep, np.nan)
    for i in range(n_ep):
        try:
            analytic_phase = _filter_hilbert(sig_2d[i], fs, phase_band, filter_kind=filter_kind)
            analytic_amp = _filter_hilbert(sig_2d[i], fs, amp_band, filter_kind=filter_kind)
        except Exception:
            continue
        phi = np.angle(analytic_phase)[edge:-edge or None]
        amp = np.abs(analytic_amp)[edge:-edge or None]
        if phi.size < 2 * n_bins:
            continue
        mi_per_epoch[i] = _tort_mi(phi, amp, n_bins=n_bins)
        keep_indices.append(i)
        phi_per_epoch.append(phi)
        amp_per_epoch.append(amp)

    n_keep = len(keep_indices)
    observed_mi = float(np.nanmean(mi_per_epoch))

    if n_keep < 2:
        surrogate_mi = np.full(n_surr, np.nan)
        z_pac, p_value = float("nan"), float("nan")
        cliff = float("nan")
    else:
        rng = np.random.default_rng(seed)
        surrogate_mi = np.empty(n_surr)
        for s in range(n_surr):
            # Permutation with no fixed points (no trial paired with itself);
            # for very small n_keep, a derangement is hard, so we just resample
            # until at least 90 % of pairings are mismatched. For n_keep >= 5
            # any permutation will have ≤ 1/e ≈ 36 % fixed-point probability
            # so a simple loop with 5 tries gets a derangement reliably.
            for _ in range(5):
                perm = rng.permutation(n_keep)
                if np.mean(perm != np.arange(n_keep)) > 0.9 or n_keep < 5:
                    break
            mi_s = np.empty(n_keep)
            for k in range(n_keep):
                phi_k = phi_per_epoch[k]
                amp_pi = amp_per_epoch[perm[k]]
                # Lengths may differ by 1 sample due to integer edge truncation;
                # crop to common length.
                m = min(phi_k.size, amp_pi.size)
                mi_s[k] = _tort_mi(phi_k[:m], amp_pi[:m], n_bins=n_bins)
            surrogate_mi[s] = float(np.nanmean(mi_s))
        z_pac, p_value = _z_and_p(observed_mi, surrogate_mi)
        cliff = _cliffs_delta(mi_per_epoch[np.isfinite(mi_per_epoch)], surrogate_mi)

    return {
        "pac": observed_mi,
        "mi_per_epoch": mi_per_epoch,
        "mi_std_across_epochs": float(np.nanstd(mi_per_epoch, ddof=1)) if n_ep > 1 else float("nan"),
        "surrogate_mi": surrogate_mi,
        "z_pac": z_pac,
        "p_value": p_value,
        "cliff_delta": cliff,
        "method": "trial_shuffle",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "n_bins": int(n_bins),
            "filter_edge_s": float(filter_edge_s),
            "filter_kind": str(filter_kind),
            "surrogate_kind": "trial_shuffle_phase_vs_amp",
            "n_epochs_kept": int(n_keep),
            "match_to": "Daume 2024 Nature s41586-024-07309-z",
        },
    }


def _compute_tort_trial_shuffle_concatenated(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surr: int = 200,
    seed: int | None = 2025,
    n_bins: int = 18,
    filter_edge_s: float = 0.5,
    filter_kind: str = "butter",
) -> dict[str, Any]:
    """Tort 2010 MI with Daume-faithful trial-shuffle surrogate.

    Mirrors the MATLAB `cfc_tort_comodulogram.m` algorithm in
    `rutishauserlab/SBCAT-release-NWB`:

    ```matlab
    for s = 1:n_surrogates
        randind = randperm(n_trials);
        surrogate_amplitude = reshape(amplitude_trials(:,randind), numpoints, 1);
        mi_surr(s) = tort_mi(phase_concatenated, surrogate_amplitude);
    end
    ```

    The defining feature vs `_compute_tort_trial_shuffle`: shuffle the
    AMPLITUDE trial-order, **concatenate** the shuffled trials end-to-end,
    then compute Tort MI **once** on the concatenated `(phase, amp)`
    pair. Phase is not shuffled — only amplitude. The observed MI is
    likewise computed once on the (un-shuffled) concatenated signal.

    Distinct from `_compute_tort_trial_shuffle` (which computes MI per
    trial-pair and averages); empirically this gives a tighter null
    distribution and matches Daume Fig 2a's z-scoring exactly.
    """
    sig_2d, n_ep, n_samp = _coerce_signal(signal)
    if n_ep < 2:
        raise ValueError(
            "trial_shuffle_concatenated requires n_epochs >= 2; got "
            f"n_epochs={n_ep}. Use 'tort_block_resample' for single-trial / "
            "continuous data."
        )
    edge = int(round(filter_edge_s * fs))
    if 2 * edge >= n_samp:
        raise ValueError(
            f"filter_edge_s={filter_edge_s} too large for n_samp={n_samp} at fs={fs}; "
            "reduce filter_edge_s or use a longer analysis window."
        )

    # Build per-epoch phase / amplitude with edge masking (same as
    # _compute_tort_block_resample) — but keep them as a list of arrays
    # so we can shuffle by trial below.
    phi_per_epoch: list[np.ndarray] = []
    amp_per_epoch: list[np.ndarray] = []
    mi_per_epoch = np.full(n_ep, np.nan)
    for i in range(n_ep):
        try:
            analytic_phase = _filter_hilbert(sig_2d[i], fs, phase_band, filter_kind=filter_kind)
            analytic_amp = _filter_hilbert(sig_2d[i], fs, amp_band, filter_kind=filter_kind)
        except Exception:
            continue
        phi = np.angle(analytic_phase)[edge:-edge or None]
        amp = np.abs(analytic_amp)[edge:-edge or None]
        if phi.size < 2 * n_bins:
            continue
        mi_per_epoch[i] = _tort_mi(phi, amp, n_bins=n_bins)
        phi_per_epoch.append(phi)
        amp_per_epoch.append(amp)

    n_keep = len(phi_per_epoch)
    if n_keep < 2:
        return {
            "pac": float("nan"),
            "mi_per_epoch": mi_per_epoch,
            "mi_std_across_epochs": float("nan"),
            "surrogate_mi": np.full(n_surr, np.nan),
            "z_pac": float("nan"),
            "p_value": float("nan"),
            "cliff_delta": float("nan"),
            "method": "trial_shuffle_concatenated",
            "phase_band": phase_band,
            "amp_band": amp_band,
            "fs": float(fs),
            "n_epochs": int(n_ep),
            "n_samples": int(n_samp),
            "backend_meta": {
                "n_bins": int(n_bins),
                "filter_edge_s": float(filter_edge_s),
                "filter_kind": str(filter_kind),
                "surrogate_kind": "trial_shuffle_amp_concatenated",
                "n_epochs_kept": int(n_keep),
                "match_to": "Daume 2024 MATLAB cfc_tort_comodulogram.m",
            },
        }

    # Observed MI: Tort MI on the concatenated (unshuffled) signal.
    phi_all = np.concatenate(phi_per_epoch)
    amp_all = np.concatenate(amp_per_epoch)
    observed_mi = _tort_mi(phi_all, amp_all, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    surrogate_mi = np.empty(n_surr)
    for s in range(n_surr):
        perm = rng.permutation(n_keep)
        amp_shuf = np.concatenate([amp_per_epoch[k] for k in perm])
        # phi_all length might differ from amp_shuf if edge truncation
        # produced different lengths — crop to the smaller for safety.
        m = min(phi_all.size, amp_shuf.size)
        surrogate_mi[s] = _tort_mi(phi_all[:m], amp_shuf[:m], n_bins=n_bins)

    z_pac, p_value = _z_and_p(observed_mi, surrogate_mi)
    cliff = _cliffs_delta(mi_per_epoch[np.isfinite(mi_per_epoch)], surrogate_mi)

    return {
        "pac": float(observed_mi),
        "mi_per_epoch": mi_per_epoch,
        "mi_std_across_epochs": float(np.nanstd(mi_per_epoch, ddof=1)) if n_ep > 1 else float("nan"),
        "surrogate_mi": surrogate_mi,
        "z_pac": z_pac,
        "p_value": p_value,
        "cliff_delta": cliff,
        "method": "trial_shuffle_concatenated",
        "phase_band": phase_band,
        "amp_band": amp_band,
        "fs": float(fs),
        "n_epochs": int(n_ep),
        "n_samples": int(n_samp),
        "backend_meta": {
            "n_bins": int(n_bins),
            "filter_edge_s": float(filter_edge_s),
            "filter_kind": str(filter_kind),
            "surrogate_kind": "trial_shuffle_amp_concatenated",
            "n_epochs_kept": int(n_keep),
            "match_to": "Daume 2024 MATLAB cfc_tort_comodulogram.m",
        },
    }


def compute_pac(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    method: str = "tort_block_resample",
    n_surr: int = 200,
    seed: int | None = 2025,
    bypass_bandwidth_check: bool = False,
    filter_kind: str = "butter",
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
    bypass_bandwidth_check : bool
        If True, skip the Aru 2015 amp-vs-phase bandwidth assertion.
        Used internally by `compute_pac_grid(..., amp_bw='adaptive')`
        where Daume 2024 uses a slightly more permissive
        `amp_width = 2 × phase_center` instead of Aru's `2 × phase_max`.
        Default False — keep the strict guard for direct user calls.
    filter_kind : str
        Band-pass filter family used by the three tort backends:
        - "butter" (default): Butterworth IIR via `scipy.signal.butter` +
          `sosfiltfilt`.
        - "fir": windowed-sinc FIR via `mne.filter.filter_data`
          (phase='zero', fir_design='firwin', fir_window='hamming').
        No-op for tensorpac/pactools — those backends use their wrappers'
        internal filtering and ignore this argument.
    """
    if filter_kind not in {"butter", "fir"}:
        raise ValueError(
            f"unknown filter_kind={filter_kind!r}; expected 'butter' or 'fir'"
        )
    if not bypass_bandwidth_check:
        _check_pac_bandwidth(phase_band, amp_band)
    if method == "tort_block_resample":
        return _compute_tort_block_resample(signal, fs, phase_band, amp_band, n_surr=n_surr, seed=seed, filter_kind=filter_kind)
    if method == "trial_shuffle":
        return _compute_tort_trial_shuffle(signal, fs, phase_band, amp_band, n_surr=n_surr, seed=seed, filter_kind=filter_kind)
    if method == "trial_shuffle_concatenated":
        return _compute_tort_trial_shuffle_concatenated(signal, fs, phase_band, amp_band, n_surr=n_surr, seed=seed, filter_kind=filter_kind)
    # tensorpac / pactools backends use their wrappers' internal filtering;
    # filter_kind doesn't apply and is silently ignored here.
    if method == "tensorpac_mi":
        return _compute_tensorpac(signal, fs, phase_band, amp_band, n_surr=n_surr, seed=seed)
    if method == "pactools_mi":
        return _compute_pactools(signal, fs, phase_band, amp_band, method="tort", n_surr=n_surr, seed=seed)
    if method == "pactools_glm":
        return _compute_pactools(signal, fs, phase_band, amp_band, method="duprelatour", n_surr=n_surr, seed=seed)
    raise ValueError(
        f"unknown method: {method!r} "
        "(expected 'tort_block_resample' | 'trial_shuffle' | 'trial_shuffle_concatenated' | "
        "'tensorpac_mi' | 'pactools_mi' | 'pactools_glm')"
    )


# -----------------------------------------------------------------------------
# Robustness diagnostics — Aru et al. 2015 artifact checks
# -----------------------------------------------------------------------------


def bycycle_diagnostics(
    signal: ArrayLike,
    fs: float,
    phase_band: tuple[float, float],
    burst_thresh: dict | None = None,
) -> dict[str, Any]:
    """Per-channel waveform-shape diagnostics for PAC artifact screening.

    Non-sinusoidal carriers (asymmetric rise-decay, sharp peaks) generate
    artifactual PAC because the filter-Hilbert pipeline mixes harmonics
    into the high-frequency band (Aru et al. 2015, *Curr Opin Neurobiol*
    31:51-61; Cole & Voytek 2017, *Trends Cogn Sci* 21:137-149). Bycycle
    quantifies per-cycle waveform shape directly in the time domain.

    Returns
    -------
    dict with keys:
        rd_asym : float
            Median time-rise-decay-symmetry (Cole & Voytek 2017).
            0.5 = perfectly symmetric (sinusoidal-like);
            < 0.5 = decay shorter than rise (sharp down-stroke);
            > 0.5 = rise shorter than decay (sharp up-stroke).
            Project flag: |rd_asym - 0.5| > 0.1 → sensitivity check.
        pt_asym : float
            Median time-peak-trough-symmetry.
            0.5 = peak duration equals trough duration.
        period_stability : float
            CV of period across detected cycles. 0 = perfectly periodic;
            higher = more variable. Useful for distinguishing a stable
            theta rhythm from broadband 1/f activity in the theta range.
        n_cycles_detected : int
            Number of cycles found across all epochs.

    Returns NaN values when bycycle finds no cycles (signal lacks a
    detectable carrier in `phase_band`).

    Note
    ----
    Bycycle uses a burst-detection step to find oscillatory cycles vs.
    1/f activity. Default thresholds (`amp_fraction_threshold=0.3`,
    `amp_consistency_threshold=0.4`, `period_consistency_threshold=0.5`,
    `monotonicity_threshold=0.8`, `min_n_cycles=3`) are reasonable for
    LFP; override via `burst_thresh` if needed.
    """
    try:
        from bycycle.features import compute_features
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "bycycle is required for waveform-shape diagnostics. "
            "Install via `pip install bycycle` or rebuild the LFPAnalysis env."
        ) from e

    arr, n_ep, _ = _coerce_signal(signal)
    # bycycle 1.2.0 threshold_kwargs — these gate the `is_burst` column,
    # which we use to prefer bursting cycles when present. Default values
    # match `bycycle.burst.detect_bursts_cycles` defaults.
    if burst_thresh is None:
        burst_thresh = {
            "amp_fraction_threshold": 0.3,
            "amp_consistency_threshold": 0.4,
            "period_consistency_threshold": 0.5,
            "monotonicity_threshold": 0.8,
            "min_n_cycles": 3,
        }

    rd_list: list[float] = []
    pt_list: list[float] = []
    periods: list[float] = []
    n_cycles = 0
    n_cycles_burst = 0
    used_burst_filter = True
    for i in range(n_ep):
        try:
            df = compute_features(
                arr[i],
                fs,
                phase_band,
                center_extrema="trough",
                threshold_kwargs=burst_thresh,
            )
        except Exception:
            # bycycle can fail on flat / very short signals; skip the epoch
            continue
        if df is None or len(df) == 0:
            continue
        n_cycles += len(df)
        if "is_burst" in df.columns:
            burst_df = df[df["is_burst"].astype(bool)]
            n_cycles_burst += len(burst_df)
        else:
            burst_df = df

        # Prefer bursting cycles; fall back to all cycles per epoch if no
        # bursts were detected (e.g., on highly periodic / noise-free
        # synthetic data where the consistency thresholds reject every
        # cycle, or on broadband data where bycycle simply can't find a
        # sustained oscillation).
        use_df = burst_df if len(burst_df) > 0 else df
        rd_list.extend(use_df["time_rdsym"].dropna().tolist())
        pt_list.extend(use_df["time_ptsym"].dropna().tolist())
        periods.extend(use_df["period"].dropna().tolist())

    # If no epoch had a single bursting cycle, flag the diagnostic as
    # fall-back so downstream interpretation can discount it.
    if n_cycles_burst == 0 and n_cycles > 0:
        used_burst_filter = False

    if n_cycles == 0:
        return {
            "rd_asym": float("nan"),
            "pt_asym": float("nan"),
            "period_stability": float("nan"),
            "n_cycles_detected": 0,
            "n_cycles_burst": 0,
            "used_burst_filter": False,
        }

    rd_arr = np.asarray(rd_list, dtype=float)
    pt_arr = np.asarray(pt_list, dtype=float)
    per_arr = np.asarray(periods, dtype=float)
    return {
        "rd_asym": float(np.nanmedian(rd_arr)) if rd_arr.size else float("nan"),
        "pt_asym": float(np.nanmedian(pt_arr)) if pt_arr.size else float("nan"),
        "period_stability": (
            float(np.nanstd(per_arr, ddof=1) / np.nanmean(per_arr))
            if per_arr.size > 1 and np.nanmean(per_arr) > 0
            else float("nan")
        ),
        "n_cycles_detected": int(n_cycles),
        "n_cycles_burst": int(n_cycles_burst),
        "used_burst_filter": bool(used_burst_filter),
    }


# Default 3×3 comodulogram grid spanning delta–theta–alpha × low/mid/high gamma.
# The (8, 12) × (30, 50) cell will fail the bandwidth guard (amp_width=20 <
# 2 * 12 = 24) and is filled with NaN at compute time — interpretable, not a bug.
DEFAULT_PAC_GRID_3x3 = {
    "phase_bands": ((2.0, 4.0), (4.0, 8.0), (8.0, 12.0)),
    "amp_bands": ((30.0, 50.0), (50.0, 80.0), (80.0, 150.0)),
}


def compute_pac_grid_3x3(
    signal: ArrayLike,
    fs: float,
    phase_bands: tuple[tuple[float, float], ...] = DEFAULT_PAC_GRID_3x3["phase_bands"],
    amp_bands: tuple[tuple[float, float], ...] = DEFAULT_PAC_GRID_3x3["amp_bands"],
    method: str = "tensorpac_mi",
    n_surr: int = 200,
    seed: int | None = 2025,
) -> dict[str, Any]:
    """Coarse comodulogram on a `(n_phase × n_amp)` grid.

    A single-cell PAC value gives no information about whether the
    chosen band pair sits at a local maximum of the comodulogram.
    This 3×3 (or arbitrarily-shaped) sweep confirms or refutes the
    cell choice as the maximum across delta-theta-alpha × low/mid/high
    gamma. Per the connectivity skill, comodulogram pixels are NOT
    independent — cluster-permutation across the grid is required for
    a corrected p-value at the cohort level (see methods doc).

    Cells that violate the Aru 2015 bandwidth criterion
    (`amp_width < 2 × phase_max`) are filled with NaN rather than
    raising, so a single invalid corner doesn't blow up the grid.
    The result dict records which cells were skipped.

    Returns
    -------
    dict with:
        pac_grid : (n_phase, n_amp) np.ndarray of observed MI
        z_grid   : (n_phase, n_amp) np.ndarray of z_pac
        p_grid   : (n_phase, n_amp) np.ndarray of one-sided p-values
        cliff_grid : (n_phase, n_amp) np.ndarray of Cliff's δ
        valid_mask : (n_phase, n_amp) bool — True where bandwidth guard passed
        phase_bands, amp_bands, method, fs : as input
    """
    n_p = len(phase_bands)
    n_a = len(amp_bands)
    pac_grid = np.full((n_p, n_a), np.nan)
    z_grid = np.full((n_p, n_a), np.nan)
    p_grid = np.full((n_p, n_a), np.nan)
    cliff_grid = np.full((n_p, n_a), np.nan)
    valid_mask = np.zeros((n_p, n_a), dtype=bool)
    for i, pb in enumerate(phase_bands):
        for j, ab in enumerate(amp_bands):
            try:
                _check_pac_bandwidth(pb, ab)
            except ValueError:
                continue
            valid_mask[i, j] = True
            r = compute_pac(
                signal, fs=fs, phase_band=pb, amp_band=ab,
                method=method, n_surr=n_surr, seed=seed,
            )
            pac_grid[i, j] = r["pac"]
            z_grid[i, j] = r["z_pac"]
            p_grid[i, j] = r["p_value"]
            cliff_grid[i, j] = r.get("cliff_delta", float("nan"))
    return {
        "pac_grid": pac_grid,
        "z_grid": z_grid,
        "p_grid": p_grid,
        "cliff_grid": cliff_grid,
        "valid_mask": valid_mask,
        "phase_bands": phase_bands,
        "amp_bands": amp_bands,
        "method": method,
        "fs": float(fs),
    }


def compute_pac_grid(
    signal: ArrayLike,
    fs: float,
    phase_centers: ArrayLike,
    amp_centers: ArrayLike,
    phase_bw: float = 2.0,
    amp_bw: float | str = 20.0,
    method: str = "tort_block_resample",
    n_surr: int = 200,
    seed: int | None = 2025,
    filter_kind: str = "butter",
) -> dict[str, Any]:
    """Publication-quality comodulogram over arbitrary (phase, amp) grids.

    Generalises `compute_pac_grid_3x3`. Pass center frequencies and a
    constant bandwidth per axis; each grid cell uses the band
    `(center - bw/2, center + bw/2)`. Cells that violate the Aru 2015
    bandwidth criterion (`amp_bw < 2 × phase_center_max`) are filled with
    NaN and reported in `valid_mask`.

    Typical Daume-style settings for hippocampal theta-gamma PAC:
        phase_centers = np.arange(2, 15, 2)      # 2-14 Hz step 2
        amp_centers   = np.arange(30, 151, 5)    # 30-150 Hz step 5
        phase_bw      = 2.0
        amp_bw        = "adaptive"               # per-cell 2 · phase_center

    Parameters
    ----------
    phase_centers, amp_centers : array-like of float
        Center frequencies in Hz.
    phase_bw : float
        Phase-axis full bandwidth (Hz). Band for cell `i` is
        `(phase_centers[i] − phase_bw/2, phase_centers[i] + phase_bw/2)`.
    amp_bw : float | "adaptive"
        Amplitude-axis full bandwidth in Hz, OR the string `"adaptive"`.

        - **float**: fixed amp bandwidth across all cells. Band is
          `(amp_centers[j] − amp_bw/2, amp_centers[j] + amp_bw/2)`.
        - **"adaptive"** (Daume 2024 / SBCAT-release-NWB convention):
          per-cell amp bandwidth = 2 · phase_center. Band is
          `(amp_centers[j] − phase_centers[i], amp_centers[j] + phase_centers[i])`.
          Always satisfies the Aru 2015 criterion by construction.
    method, n_surr, seed, filter_kind
        Passed through to `compute_pac` per cell. See `compute_pac` for
        `filter_kind` semantics.

    Returns
    -------
    dict matching the schema of `compute_pac_grid_3x3` plus:
        phase_centers, amp_centers : ndarray
        phase_bw : float
        amp_bw : float | "adaptive"
    """
    phase_centers = np.asarray(phase_centers, dtype=float)
    amp_centers = np.asarray(amp_centers, dtype=float)
    n_p = phase_centers.size
    n_a = amp_centers.size
    adaptive_amp = isinstance(amp_bw, str) and amp_bw.lower() == "adaptive"
    pac_grid = np.full((n_p, n_a), np.nan)
    z_grid = np.full((n_p, n_a), np.nan)
    p_grid = np.full((n_p, n_a), np.nan)
    cliff_grid = np.full((n_p, n_a), np.nan)
    valid_mask = np.zeros((n_p, n_a), dtype=bool)
    for i, pc in enumerate(phase_centers):
        pb = (float(pc - phase_bw / 2.0), float(pc + phase_bw / 2.0))
        if pb[0] <= 0:
            # Skip — phase band must be > 0
            continue
        # Daume MATLAB: [HF − LF, HF + LF] where LF is the phase center.
        # Equivalent to amp_bw = 2 × phase_center for this cell.
        eff_amp_bw = (2.0 * float(pc)) if adaptive_amp else float(amp_bw)
        for j, ac in enumerate(amp_centers):
            ab = (float(ac - eff_amp_bw / 2.0), float(ac + eff_amp_bw / 2.0))
            if ab[0] <= 0:
                continue
            # Aru 2015 check uses phase_band_upper, which is stricter than
            # Daume's phase_center convention. In adaptive mode we bypass
            # the check — Daume's adaptive formula is itself the published
            # bandwidth rule from the SBCAT MATLAB code and is accepted
            # in the field for theta-gamma PAC at these grid resolutions.
            if not adaptive_amp:
                try:
                    _check_pac_bandwidth(pb, ab)
                except ValueError:
                    continue
            valid_mask[i, j] = True
            try:
                r = compute_pac(
                    signal, fs=fs, phase_band=pb, amp_band=ab,
                    method=method, n_surr=n_surr, seed=seed,
                    bypass_bandwidth_check=adaptive_amp,
                    filter_kind=filter_kind,
                )
            except Exception:
                continue
            pac_grid[i, j] = r["pac"]
            z_grid[i, j] = r["z_pac"]
            p_grid[i, j] = r["p_value"]
            cliff_grid[i, j] = r.get("cliff_delta", float("nan"))
    return {
        "pac_grid": pac_grid,
        "z_grid": z_grid,
        "p_grid": p_grid,
        "cliff_grid": cliff_grid,
        "valid_mask": valid_mask,
        "phase_centers": phase_centers,
        "amp_centers": amp_centers,
        "phase_bw": float(phase_bw),
        "amp_bw": amp_bw if adaptive_amp else float(amp_bw),
        "method": method,
        "filter_kind": str(filter_kind),
        "fs": float(fs),
    }


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
