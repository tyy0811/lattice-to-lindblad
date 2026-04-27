"""DRAG β-calibration tests (gate-error objective per spec §5.3, post-N11)."""
from __future__ import annotations

import numpy as np
import pytest

from dispersive_readout.control.drag_calibration import calibrate_drag_beta
from dispersive_readout.physics.config import REFERENCE_DEVICE, DecoherenceParams


T_GATE = 20e-9
SIGMA = T_GATE / 4.0


def _zero_decoherence() -> DecoherenceParams:
    return DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)


def test_drag_calibration_returns_beta_in_perturbative_range():
    """β_opt must lie strictly inside the perturbative DRAG window [0, 1.2].

    The spec restricts the default search grid to [0, 1.2] (post-N11) to prevent
    the optimizer from selecting non-DRAG values that satisfy a leakage objective
    but break the gate. β_opt at the headline T_gate=20ns is empirically ≈ 0.5.
    """
    result = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        sigma=SIGMA,
        decoherence=_zero_decoherence(),
    )
    assert 0.0 <= result.beta_opt <= 1.2
    assert len(result.beta_grid) == len(result.gate_error_curve)
    assert len(result.beta_grid) == len(result.p_final_curve)
    assert len(result.beta_grid) == len(result.p_peak_curve)


def test_drag_calibration_minimizes_gate_error_on_grid():
    """Soundness: β_opt is the argmin of gate_error_curve (no other grid point
    has lower gate error)."""
    result = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        sigma=SIGMA,
        decoherence=_zero_decoherence(),
    )
    idx_opt = int(np.argmin(np.abs(result.beta_grid - result.beta_opt)))
    err_at_opt = result.gate_error_curve[idx_opt]
    assert err_at_opt == pytest.approx(np.min(result.gate_error_curve), abs=1e-15)


def test_drag_calibration_reports_leakage_minimizers_as_diagnostics():
    """V2b deliverable: the result exposes (β_min_fidelity, β_min_final_leak,
    β_min_peak_leak) triplet for the leakage-vs-fidelity trade-off curve."""
    result = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        sigma=SIGMA,
        decoherence=_zero_decoherence(),
    )
    # All three live on the same grid
    assert 0.0 <= result.beta_min_final_leak <= 1.2
    assert 0.0 <= result.beta_min_peak_leak <= 1.2
    # At T_gate = 20ns, the trade-off is real: the leakage minimizers should not
    # coincide with β_opt (they sit at higher β, where the gate is borderline).
    # This documents the trade-off as data, not just prose.
    assert result.beta_min_final_leak != result.beta_opt or result.beta_min_peak_leak != result.beta_opt
