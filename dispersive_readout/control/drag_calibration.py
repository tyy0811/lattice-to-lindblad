"""DRAG β-calibration via gate-error minimization (spec §5.3, post-N11 amendment).

Calibration objective: argmin_β (1 − F_transfer(β)).

The β grid is restricted to the perturbative DRAG range [0, 1.2] (default, 25 points)
to prevent the optimizer from selecting non-DRAG values that produce broken gates
under a leakage-only objective. See spec §12.1 (N11) for the methodology lesson.

Final-leakage and peak-leakage curves are computed and returned as **diagnostic
arrays** alongside the calibration result — they are reported in panel (b) and the
YAML cache, but they are NOT calibration targets. The leakage-vs-fidelity
trade-off is itself a Module 5a finding (V2b).

Pre-N11 history: an earlier version used a combined-max-ratio leakage objective.
That objective minimized leakage ratios but at long T_gate where leakage is
~1e-7 at all β, the ratio was noise-dominated and selected β values from the
non-perturbative regime that produced gates with 1−F up to 0.4. The fidelity
objective with the [0, 1.2] grid yields β_opt ≈ 0.5 across the panel-(b) sweep,
with 1−F < 1e-4 at T_gate ≥ 15 ns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..analysis.gate_metrics import (
    leakage_peak,
    leakage_population,
    transfer_fidelity_0_to_1,
)
from ..physics.config import DecoherenceParams, DeviceConfig
from .gate_simulator import simulate_x_gate


@dataclass(frozen=True)
class DragCalibrationResult:
    """Output of `calibrate_drag_beta`.

    beta_opt              : argmin_β (1 − F_transfer(β)) over `beta_grid`.
    beta_grid             : np.ndarray, the β values searched.
    gate_error_curve      : np.ndarray, 1 − F_transfer(β).
    p_final_curve         : np.ndarray, final leakage at each β (diagnostic).
    p_peak_curve          : np.ndarray, peak leakage at each β (diagnostic).
    p_final_no_drag       : float, baseline final leakage (β = 0).
    p_peak_no_drag        : float, baseline peak leakage (β = 0).
    beta_min_final_leak   : float, β minimizing final leakage on the grid (diagnostic).
    beta_min_peak_leak    : float, β minimizing peak leakage on the grid (diagnostic).

    The (beta_opt, beta_min_final_leak, beta_min_peak_leak) triplet is V2b's
    leakage-vs-fidelity trade-off characterization (spec §6 V2b, §12.1 N11).
    """
    beta_opt: float
    beta_grid: np.ndarray
    gate_error_curve: np.ndarray
    p_final_curve: np.ndarray
    p_peak_curve: np.ndarray
    p_final_no_drag: float
    p_peak_no_drag: float
    beta_min_final_leak: float
    beta_min_peak_leak: float


def calibrate_drag_beta(
    device: DeviceConfig,
    T_gate: float,
    sigma: float,
    beta_grid: Optional[np.ndarray] = None,
    n_levels: int = 4,
    decoherence: Optional[DecoherenceParams] = None,
) -> DragCalibrationResult:
    """Calibrate DRAG β by gate-error minimization on a perturbative β grid.

    Parameters
    ----------
    device : DeviceConfig
    T_gate, sigma : float — pulse duration and width (s).
    beta_grid : np.ndarray | None — defaults to np.linspace(0.0, 1.2, 25).
        The [0, 1.2] range is the perturbative DRAG-1 window for sin²-windowed
        envelopes at REFERENCE_DEVICE α; β > 1.2 enters the non-perturbative
        regime where the gate breaks down (see §12.1 N11).
    n_levels : int — Hilbert-space truncation. Default 4.
    decoherence : DecoherenceParams | None — defaults to zeroed Lindblad
        (calibration is a coherent-leakage / coherent-error minimization).
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 1.2, 25)
    if decoherence is None:
        decoherence = DecoherenceParams(
            gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False
        )

    # Baseline: no DRAG (β = 0)
    baseline = simulate_x_gate(
        device=device,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=False,
        beta=0.0,
        decoherence=decoherence,
        sigma=sigma,
    )
    p_final_no_drag = leakage_population(baseline.rho_final, n_levels)
    p_peak_no_drag = leakage_peak(baseline.rho_t, n_levels)
    err_baseline = 1.0 - transfer_fidelity_0_to_1(baseline.rho_final)

    gate_error_curve = np.empty(len(beta_grid))
    p_final_curve = np.empty(len(beta_grid))
    p_peak_curve = np.empty(len(beta_grid))

    for i, beta in enumerate(beta_grid):
        if beta == 0.0:
            gate_error_curve[i] = err_baseline
            p_final_curve[i] = p_final_no_drag
            p_peak_curve[i] = p_peak_no_drag
            continue
        result = simulate_x_gate(
            device=device,
            T_gate=T_gate,
            n_levels=n_levels,
            drag=True,
            beta=float(beta),
            decoherence=decoherence,
            sigma=sigma,
        )
        gate_error_curve[i] = 1.0 - transfer_fidelity_0_to_1(result.rho_final)
        p_final_curve[i] = leakage_population(result.rho_final, n_levels)
        p_peak_curve[i] = leakage_peak(result.rho_t, n_levels)

    idx_opt = int(np.argmin(gate_error_curve))
    beta_opt = float(beta_grid[idx_opt])
    beta_min_final_leak = float(beta_grid[int(np.argmin(p_final_curve))])
    beta_min_peak_leak = float(beta_grid[int(np.argmin(p_peak_curve))])

    return DragCalibrationResult(
        beta_opt=beta_opt,
        beta_grid=np.asarray(beta_grid),
        gate_error_curve=gate_error_curve,
        p_final_curve=p_final_curve,
        p_peak_curve=p_peak_curve,
        p_final_no_drag=float(p_final_no_drag),
        p_peak_no_drag=float(p_peak_no_drag),
        beta_min_final_leak=beta_min_final_leak,
        beta_min_peak_leak=beta_min_peak_leak,
    )
