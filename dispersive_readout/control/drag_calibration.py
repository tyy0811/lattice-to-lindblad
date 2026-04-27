"""DRAG β-calibration via the combined max-ratio objective.

For each β in the grid, run `simulate_x_gate(decoherence=zeroed)` to extract
both final and peak leakage. Compute ratios against the no-DRAG baseline
(β = 0 at the same T_gate, σ). β_opt minimizes the max of these two ratios:

    β_opt = argmin_β max(P_final(β)/P_final_no_DRAG, P_peak(β)/P_peak_no_DRAG).

This makes calibration consistent with V2's both-final-and-peak suppression
criterion (spec §6, §12 (53)). A pure-final-leakage argmin can pick a β that
suppresses endpoint population but leaves transient |2⟩ excursions; a pure-peak
argmin can do the reverse. Combined max-ratio avoids both failure modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..analysis.gate_metrics import leakage_peak, leakage_population
from ..physics.config import DecoherenceParams, DeviceConfig
from .gate_simulator import simulate_x_gate


@dataclass(frozen=True)
class DragCalibrationResult:
    """Output of `calibrate_drag_beta`.

    beta_opt        : argmin of max(P_final/P_final_no_DRAG, P_peak/P_peak_no_DRAG).
    beta_grid       : np.ndarray, the β values searched.
    p_final_curve   : np.ndarray, final leakage at each β.
    p_peak_curve    : np.ndarray, peak leakage at each β.
    max_ratio_curve : np.ndarray, max(final-ratio, peak-ratio) at each β.
    p_final_no_drag : float, baseline final leakage (β = 0).
    p_peak_no_drag  : float, baseline peak leakage (β = 0).
    """
    beta_opt: float
    beta_grid: np.ndarray
    p_final_curve: np.ndarray
    p_peak_curve: np.ndarray
    max_ratio_curve: np.ndarray
    p_final_no_drag: float
    p_peak_no_drag: float


def calibrate_drag_beta(
    device: DeviceConfig,
    T_gate: float,
    sigma: float,
    beta_grid: Optional[np.ndarray] = None,
    n_levels: int = 4,
    decoherence: Optional[DecoherenceParams] = None,
) -> DragCalibrationResult:
    """Calibrate DRAG β by combined max-ratio minimization (spec §5.3).

    decoherence defaults to a zeroed DecoherenceParams (calibration is a
    coherent-leakage minimization; including Lindblad would convolve T₁
    decay with the DRAG suppression we're trying to measure).
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 2.0, 21)
    if decoherence is None:
        decoherence = DecoherenceParams(
            gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False
        )

    # Baseline: no DRAG (β = 0), same T_gate, σ, decoherence
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

    # Sweep β
    p_final_curve = np.empty(len(beta_grid))
    p_peak_curve = np.empty(len(beta_grid))
    for i, beta in enumerate(beta_grid):
        if beta == 0.0:
            # Reuse baseline
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
        p_final_curve[i] = leakage_population(result.rho_final, n_levels)
        p_peak_curve[i] = leakage_peak(result.rho_t, n_levels)

    # Combined max-ratio
    eps = 1e-30  # guard against zero baseline
    ratio_final = p_final_curve / (p_final_no_drag + eps)
    ratio_peak = p_peak_curve / (p_peak_no_drag + eps)
    max_ratio_curve = np.maximum(ratio_final, ratio_peak)
    idx_opt = int(np.argmin(max_ratio_curve))
    beta_opt = float(beta_grid[idx_opt])

    return DragCalibrationResult(
        beta_opt=beta_opt,
        beta_grid=np.asarray(beta_grid),
        p_final_curve=p_final_curve,
        p_peak_curve=p_peak_curve,
        max_ratio_curve=max_ratio_curve,
        p_final_no_drag=float(p_final_no_drag),
        p_peak_no_drag=float(p_peak_no_drag),
    )
