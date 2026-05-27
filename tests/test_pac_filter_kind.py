"""Smoke test for the `filter_kind` parameter on `compute_pac`.

Verifies that:
1. Both `filter_kind='butter'` and `filter_kind='fir'` detect synthetic
   phase-amplitude coupling (PAC value substantially higher than on the
   same signal with `coupling_strength=0`).
2. The two filter families do not produce numerically identical PAC
   values — confirming the parameter actually changes the computation.
3. The `filter_kind` value flows through to `backend_meta` for
   traceability.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:  # standalone-runnable without pytest
    pytest = None  # type: ignore[assignment]
    _HAVE_PYTEST = False

from LFPAnalysis.pac_utils import compute_pac, synthetic_pac_signal


FS = 500.0
PHASE_BAND = (4.0, 8.0)
AMP_BAND = (50.0, 70.0)
N_SURR_FAST = 50  # smoke test — speed > null precision


def _build_signal(coupling_strength: float, seed: int = 2025) -> np.ndarray:
    """Three concatenated 4-s epochs of theta-gamma signal at the given coupling.

    Returns shape (3, n_samples). Multi-epoch so the test exercises the
    block-resample backend's epoch-concatenation path (analogous to
    real iEEG trial data).
    """
    return np.stack([
        synthetic_pac_signal(
            fs=FS, duration_s=4.0, phase_freq=6.0, amp_freq=60.0,
            coupling_strength=coupling_strength, noise_level=0.2, seed=seed + i,
        )
        for i in range(3)
    ])


def _maybe_parametrize(name, values):
    """Use pytest.parametrize if available, otherwise a no-op decorator.

    The standalone __main__ runner iterates over `values` explicitly.
    """
    if _HAVE_PYTEST:
        return pytest.mark.parametrize(name, values)
    return lambda f: f


@_maybe_parametrize("filter_kind", ["butter", "fir"])
def test_both_filter_kinds_detect_coupling(filter_kind):
    coupled = _build_signal(coupling_strength=0.7)
    uncoupled = _build_signal(coupling_strength=0.0)

    r_coupled = compute_pac(
        coupled, fs=FS, phase_band=PHASE_BAND, amp_band=AMP_BAND,
        method="tort_block_resample", n_surr=N_SURR_FAST,
        filter_kind=filter_kind,
    )
    r_uncoupled = compute_pac(
        uncoupled, fs=FS, phase_band=PHASE_BAND, amp_band=AMP_BAND,
        method="tort_block_resample", n_surr=N_SURR_FAST,
        filter_kind=filter_kind,
    )

    assert r_coupled["pac"] > r_uncoupled["pac"], (
        f"{filter_kind}: coupled MI={r_coupled['pac']:.4f} should exceed "
        f"uncoupled MI={r_uncoupled['pac']:.4f}"
    )
    assert r_coupled["backend_meta"]["filter_kind"] == filter_kind


def test_butter_and_fir_give_different_mi():
    sig = _build_signal(coupling_strength=0.7)
    r_butter = compute_pac(
        sig, fs=FS, phase_band=PHASE_BAND, amp_band=AMP_BAND,
        method="tort_block_resample", n_surr=N_SURR_FAST,
        filter_kind="butter",
    )
    r_fir = compute_pac(
        sig, fs=FS, phase_band=PHASE_BAND, amp_band=AMP_BAND,
        method="tort_block_resample", n_surr=N_SURR_FAST,
        filter_kind="fir",
    )
    assert r_butter["pac"] != r_fir["pac"], (
        "filter_kind='butter' and filter_kind='fir' produced identical MI — "
        "the parameter is not reaching the filter call."
    )


def test_unknown_filter_kind_raises():
    sig = _build_signal(coupling_strength=0.5)
    try:
        compute_pac(
            sig, fs=FS, phase_band=PHASE_BAND, amp_band=AMP_BAND,
            method="tort_block_resample", n_surr=N_SURR_FAST,
            filter_kind="bogus",
        )
    except ValueError as e:
        assert "filter_kind" in str(e), f"unexpected message: {e!r}"
        return
    raise AssertionError("expected ValueError for filter_kind='bogus'")


if __name__ == "__main__":
    # Standalone runner so the file is usable without pytest installed.
    test_both_filter_kinds_detect_coupling("butter")
    print("[ok] test_both_filter_kinds_detect_coupling[butter]")
    test_both_filter_kinds_detect_coupling("fir")
    print("[ok] test_both_filter_kinds_detect_coupling[fir]")
    test_butter_and_fir_give_different_mi()
    print("[ok] test_butter_and_fir_give_different_mi")
    test_unknown_filter_kind_raises()
    print("[ok] test_unknown_filter_kind_raises")
    print("\nAll filter_kind smoke tests passed.")
