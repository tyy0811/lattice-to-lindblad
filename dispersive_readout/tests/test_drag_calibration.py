"""DRAG β-calibration tests (combined max-ratio objective per spec §5.3)."""
from __future__ import annotations

import numpy as np
import pytest

from dispersive_readout.control.drag_calibration import calibrate_drag_beta
from dispersive_readout.physics.config import REFERENCE_DEVICE, DecoherenceParams


T_GATE = 10e-9
SIGMA = T_GATE / 4.0


def _zero_decoherence() -> DecoherenceParams:
    return DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)


def test_drag_calibration_returns_beta_in_range():
    """β_opt must lie inside the default search grid [0, 2]."""
    result = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        sigma=SIGMA,
        decoherence=_zero_decoherence(),
    )
    assert 0.0 <= result.beta_opt <= 2.0
    assert len(result.beta_grid) == len(result.p_final_curve)
    assert len(result.beta_grid) == len(result.p_peak_curve)
    assert len(result.beta_grid) == len(result.max_ratio_curve)


def test_drag_calibration_improves_or_matches_beta_one():
    """Soundness: the calibrated β_opt produces leakage ≤ leakage at β=1
    within numerical / β-grid resolution. Does NOT require β_opt ≈ 1."""
    result = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        sigma=SIGMA,
        decoherence=_zero_decoherence(),
    )
    # Find index of β closest to 1 in the grid
    idx_one = int(np.argmin(np.abs(result.beta_grid - 1.0)))
    idx_opt = int(np.argmin(np.abs(result.beta_grid - result.beta_opt)))
    # Max-ratio at β_opt must be ≤ max-ratio at β = 1 (β_opt is the argmin)
    assert result.max_ratio_curve[idx_opt] <= result.max_ratio_curve[idx_one] + 1e-12
