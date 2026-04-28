"""DRAG β-calibration via average X-gate fidelity (spec §5.3, post-N11 + N12).

Calibration objective: argmin_β (1 − F_avg(β)) where F_avg is the average
X-gate fidelity over the Pauli set {|0⟩, |1⟩, |+⟩, |+i⟩}. Using the average
(rather than the one-way `|0⟩ → |1⟩` transfer) catches asymmetric forward/
reverse action and coherent superposition-state phase errors that a one-way
metric would silently miss (post-N12, see spec §12.1).

The β grid is restricted to the perturbative DRAG-1 window `[0, 1.2]` by
default. Custom grids are accepted, but values outside `[0, 1.2]` require
explicit `allow_nonperturbative=True` opt-in (post-N12 hardening) — this
prevents downstream callers from re-opening the round-9 failure mode where
the optimizer drifts to non-perturbative β producing broken gates.

Final-leakage and peak-leakage curves are computed and returned as
**diagnostic arrays** alongside the calibration result — they are V2b
deliverables (panel (b) inset 2 + YAML schema), not calibration targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..analysis.gate_metrics import (
    average_gate_fidelity_x,
    leakage_peak,
    leakage_population,
)
from ..physics.config import DecoherenceParams, DeviceConfig
from .gate_simulator import simulate_x_gate


# Perturbative DRAG-1 β range (post-N11). Values outside this range are
# non-perturbative for sin²-windowed envelopes at REFERENCE_DEVICE α and
# require explicit opt-in.
PERTURBATIVE_BETA_MIN = 0.0
PERTURBATIVE_BETA_MAX = 1.2


@dataclass(frozen=True)
class DragCalibrationResult:
    """Output of `calibrate_drag_beta`.

    beta_opt              : argmin_β (1 − F_avg(β)) over `beta_grid`.
    beta_grid             : np.ndarray, the β values searched.
    gate_error_curve      : np.ndarray, 1 − F_avg(β) — shipped objective curve.
    f_avg_curve           : np.ndarray, F_avg(β).
    per_state_fidelity_curve : np.ndarray of shape (n_betas, 4),
        per-input-state fidelities [|0⟩→|1⟩, |1⟩→|0⟩, |+⟩→|+⟩, |+i⟩→|-i⟩].
    p_final_curve         : np.ndarray, final leakage at each β (diagnostic).
    p_peak_curve          : np.ndarray, peak leakage at each β (diagnostic).
    p_final_no_drag       : float, baseline final leakage (β = 0).
    p_peak_no_drag        : float, baseline peak leakage (β = 0).
    beta_min_final_leak   : float, β minimizing final leakage on the grid.
    beta_min_peak_leak    : float, β minimizing peak leakage on the grid.
    perturbative_safe     : bool, True iff the searched β grid lies entirely
        within `[PERTURBATIVE_BETA_MIN, PERTURBATIVE_BETA_MAX]`. Downstream
        callers publishing as "calibrated DRAG-1" output should require True.

    The `(beta_opt, beta_min_final_leak, beta_min_peak_leak)` triplet exposes
    the leakage-vs-fidelity trade-off as data (V2b).
    """
    beta_opt: float
    beta_grid: np.ndarray
    gate_error_curve: np.ndarray
    f_avg_curve: np.ndarray
    per_state_fidelity_curve: np.ndarray
    p_final_curve: np.ndarray
    p_peak_curve: np.ndarray
    p_final_no_drag: float
    p_peak_no_drag: float
    beta_min_final_leak: float
    beta_min_peak_leak: float
    perturbative_safe: bool


def _validate_beta_grid(beta_grid: np.ndarray, allow_nonperturbative: bool) -> bool:
    """Validate finiteness and (default) perturbative-range safety of the β grid.

    Returns
    -------
    perturbative_safe : bool
        True iff every β in the grid is within `[PERTURBATIVE_BETA_MIN,
        PERTURBATIVE_BETA_MAX]`. The caller stores this on the result so
        downstream code can refuse to publish non-perturbative output.

    Raises
    ------
    ValueError
        If any β is non-finite, or if any β is outside `[PERTURBATIVE_BETA_MIN,
        PERTURBATIVE_BETA_MAX]` and `allow_nonperturbative` is False.
    """
    arr = np.asarray(beta_grid, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"beta_grid must contain only finite values; got {arr.tolist()}."
        )
    in_range = (arr >= PERTURBATIVE_BETA_MIN) & (arr <= PERTURBATIVE_BETA_MAX)
    perturbative_safe = bool(np.all(in_range))
    if not perturbative_safe and not allow_nonperturbative:
        bad = arr[~in_range].tolist()
        raise ValueError(
            f"beta_grid contains values outside the perturbative range "
            f"[{PERTURBATIVE_BETA_MIN}, {PERTURBATIVE_BETA_MAX}]: {bad}. "
            f"Pass allow_nonperturbative=True to opt in to a non-perturbative "
            f"sweep (the result will have perturbative_safe=False and must "
            f"not be published as calibrated DRAG-1 output). See spec §12.1 (N11)."
        )
    return perturbative_safe


def calibrate_drag_beta(
    device: DeviceConfig,
    T_gate: float,
    sigma: float,
    beta_grid: Optional[np.ndarray] = None,
    n_levels: int = 4,
    decoherence: Optional[DecoherenceParams] = None,
    allow_nonperturbative: bool = False,
) -> DragCalibrationResult:
    """Calibrate DRAG β by average-gate-fidelity minimization on a perturbative grid.

    Parameters
    ----------
    device : DeviceConfig
    T_gate, sigma : float — pulse duration and width (s).
    beta_grid : np.ndarray | None — defaults to `np.linspace(0.0, 1.2, 25)`.
        Values outside `[0, 1.2]` require `allow_nonperturbative=True`.
    n_levels : int — Hilbert-space truncation. Default 4.
    decoherence : DecoherenceParams | None — defaults to zeroed Lindblad
        (calibration is a coherent-error minimization).
    allow_nonperturbative : bool — opt-in to β values outside `[0, 1.2]`.
        When True the result has `perturbative_safe=False`; callers
        publishing as calibrated DRAG-1 output must refuse such results.

    Raises
    ------
    ValueError
        If `beta_grid` contains non-finite values, or if it contains values
        outside `[0, 1.2]` without `allow_nonperturbative=True`.
    """
    if beta_grid is None:
        beta_grid = np.linspace(PERTURBATIVE_BETA_MIN, PERTURBATIVE_BETA_MAX, 25)
    perturbative_safe = _validate_beta_grid(beta_grid, allow_nonperturbative)

    if decoherence is None:
        decoherence = DecoherenceParams(
            gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False
        )

    # Baseline leakage at β = 0 (no DRAG)
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

    n_betas = len(beta_grid)
    f_avg_curve = np.empty(n_betas)
    per_state_curve = np.empty((n_betas, 4))
    p_final_curve = np.empty(n_betas)
    p_peak_curve = np.empty(n_betas)

    for i, beta in enumerate(beta_grid):
        beta_f = float(beta)
        # Average-gate fidelity (4-input Pauli set)
        f_avg, per_state = average_gate_fidelity_x(
            device=device,
            T_gate=T_gate,
            n_levels=n_levels,
            drag=(beta_f != 0.0),
            beta=beta_f,
            decoherence=decoherence,
            sigma=sigma,
        )
        f_avg_curve[i] = f_avg
        per_state_curve[i, :] = np.asarray(per_state)

        # Diagnostic leakage curves (from |0⟩ start — the "natural" leakage source)
        if beta_f == 0.0:
            p_final_curve[i] = p_final_no_drag
            p_peak_curve[i] = p_peak_no_drag
        else:
            r = simulate_x_gate(
                device=device,
                T_gate=T_gate,
                n_levels=n_levels,
                drag=True,
                beta=beta_f,
                decoherence=decoherence,
                sigma=sigma,
            )
            p_final_curve[i] = leakage_population(r.rho_final, n_levels)
            p_peak_curve[i] = leakage_peak(r.rho_t, n_levels)

    gate_error_curve = 1.0 - f_avg_curve
    idx_opt = int(np.argmin(gate_error_curve))
    beta_opt = float(beta_grid[idx_opt])
    beta_min_final_leak = float(beta_grid[int(np.argmin(p_final_curve))])
    beta_min_peak_leak = float(beta_grid[int(np.argmin(p_peak_curve))])

    return DragCalibrationResult(
        beta_opt=beta_opt,
        beta_grid=np.asarray(beta_grid, dtype=float),
        gate_error_curve=gate_error_curve,
        f_avg_curve=f_avg_curve,
        per_state_fidelity_curve=per_state_curve,
        p_final_curve=p_final_curve,
        p_peak_curve=p_peak_curve,
        p_final_no_drag=float(p_final_no_drag),
        p_peak_no_drag=float(p_peak_no_drag),
        beta_min_final_leak=beta_min_final_leak,
        beta_min_peak_leak=beta_min_peak_leak,
        perturbative_safe=perturbative_safe,
    )
