"""Analytic per-segment cavity-EOM integration for direct-jump active reset.

Solves dα/dt = -(κ/2 + i·δ_s)·α - i·ε in closed form on each segment of
a piecewise-constant qubit-state history. Returns both the time-resolved
α trajectory (for diagnostics) and the closed-form integrated IQ (the
quantity consumed by extract_joint_matrix's threshold step).

Convention: integrated IQ uses Module 1's `∫ α(t) dt` convention (no 1/τ
averaging factor), so Module 1's σ_per_quadrature = √(τ/(4κ)) noise scale
applies directly without rescaling. See compute_assignment_fidelity in
physics/readout_model.py for the canonical noise formula.
"""
from __future__ import annotations

import numpy as np


def _segment_integral_factor(rate: complex, dt: float) -> complex:
    """Returns (1 - exp(-rate·dt)) / rate, stably for all dt.

    For |rate·dt| < 1e-8, evaluates the Taylor expansion
        dt · (1 - rate·dt/2 + (rate·dt)²/6)
    to avoid catastrophic cancellation in `1 - exp(...)`. Otherwise uses
    np.expm1 for numerical accuracy.

    This matters at very short final segments (e.g., a qubit jump occurring
    at t_jump = τ_meas - ε with ε picosecond-scale), which are plausible
    across the V7 sweep range.
    """
    x = rate * dt
    if abs(x) < 1e-8:
        # Taylor: (1 - exp(-x)) / rate = dt · (1 - x/2 + x²/6 - ...)
        return dt * (1.0 - x / 2.0 + (x * x) / 6.0)
    return -np.expm1(-x) / rate


from dispersive_readout.physics.config import DeviceConfig, DriveParams
from dispersive_readout.physics.dispersive import dispersive_shift_full
from dispersive_readout.physics.transmon import (
    charge_operator_matrix_elements,
    diagonalize_transmon,
)


def _chi_per_level(device: DeviceConfig) -> np.ndarray:
    """Per-level dispersive shifts χ_j (rad/s), sourced from
    physics.dispersive.dispersive_shift_full. Single source of truth for
    the χ convention; never re-derived locally.
    """
    energies, eigenstates = diagonalize_transmon(device.transmon, device.truncation)
    n_mat = charge_operator_matrix_elements(eigenstates, device.truncation)
    return dispersive_shift_full(
        energies, n_mat, device.coupling.g, device.resonator.omega_r,
    )


def pointer_steady_state(
    device: DeviceConfig,
    drive_params: DriveParams,
    qubit_state: int,
) -> complex:
    """α_∞(s) = -i·ε_drive / (κ/2 + i·δ_s) for qubit state s ∈ {0, 1}.

    δ_s = (ω_r − ω_d) + χ_s where ω_d = ω_r + drive_params.detuning, so
    (ω_r − ω_d) = -drive_params.detuning. χ_s comes from the per-level
    dispersive-shift array (single source of truth: physics.dispersive).

    Convention check: this matches the cavity drift in
    physics.lindblad.build_hamiltonian's H_r = (omega_r - omega_d) * a†a
    plus the χ_j a†a term from H_chi when projected onto qubit eigenstate
    |s⟩. See V4a (test_pointer_response_matches_simulate_readout_in_no_
    jump_limit) for the integrated-IQ-level cross-check.
    """
    if qubit_state not in (0, 1):
        raise ValueError(
            f"qubit_state ∈ {{0, 1}} required in v0 (got {qubit_state}); "
            f"thermal/leakage-state pointer responses are v1.5 territory."
        )
    chi_per_level = _chi_per_level(device)
    chi_s = float(chi_per_level[qubit_state])

    kappa = device.resonator.kappa
    eps = drive_params.amplitude
    delta_s = -drive_params.detuning + chi_s

    return -1j * eps / (kappa / 2.0 + 1j * delta_s)
