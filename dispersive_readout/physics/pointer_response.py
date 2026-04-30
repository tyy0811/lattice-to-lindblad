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


def compute_alpha_trajectory(
    device: DeviceConfig,
    drive_params: DriveParams,
    history: "QubitStateHistory",
    t_grid: np.ndarray,
) -> tuple[np.ndarray, complex]:
    """Closed-form per-segment integration of dα/dt = -(κ/2 + i·δ_s)·α - i·ε.

    Returns (alpha_trajectory, integrated_iq) where:
    - alpha_trajectory: complex ndarray shape (len(t_grid),). For each
      t in t_grid, the value of α(t) given the piecewise-constant qubit
      state history. For diagnostics/plotting; NOT consumed by
      extract_joint_matrix.
    - integrated_iq: complex scalar = ∫_0^t_total α(t) dt, computed
      exactly per segment (sum of segment-wise closed-form integrals).
      NOT via numerical quadrature on alpha_trajectory. This is the
      V4a-contract-relevant quantity.

    Plotting note: when called with a 2-point t_grid (as inside
    extract_joint_matrix's inner loop), the returned trajectory has only
    endpoint values [α(0), α(t_total)] and is NOT useful for plotting α(t)
    — it would render as a straight line and miss the κ/2-scale relaxation
    dynamics. For diagnostic visualization, call separately with a denser
    grid (e.g., np.linspace(0, t_total, 200)).

    v0 assumes a square pulse: ε(t) = drive_params.amplitude for t ∈
    [0, t_total]. v1.5 may parameterize the envelope.
    """
    # Avoid circular import — QubitStateHistory lives in control/, but we
    # can't import it at module top because reset_protocol.py imports from
    # this file. Defer to runtime.
    from dispersive_readout.control.reset_protocol import QubitStateHistory  # noqa: F401

    chi_per_level = _chi_per_level(device)
    kappa = device.resonator.kappa
    eps = drive_params.amplitude
    omega_r_minus_omega_d = -drive_params.detuning  # ω_r − (ω_r + detuning) = −detuning

    def _delta_s(s: int) -> float:
        return omega_r_minus_omega_d + float(chi_per_level[s])

    def _alpha_inf(s: int) -> complex:
        return -1j * eps / (kappa / 2.0 + 1j * _delta_s(s))

    # Build segment list with explicit (t_start, t_end, qubit_state)
    segments = []
    for i, (t_start, q) in enumerate(history.segments):
        t_end = (
            history.segments[i + 1][0] if i + 1 < len(history.segments)
            else history.t_total
        )
        segments.append((t_start, t_end, q))

    # Step through segments analytically. α(t) is continuous across jumps;
    # only δ_s changes, after which α relaxes toward the new α_∞ at rate κ/2.
    integrated_iq: complex = 0.0 + 0.0j
    alpha_at_segment_start: complex = 0.0 + 0.0j  # cavity starts in vacuum
    alpha_grid = np.empty(len(t_grid), dtype=complex)

    for seg_start, seg_end, q in segments:
        a_inf = _alpha_inf(q)
        rate = kappa / 2.0 + 1j * _delta_s(q)
        delta_t_seg = seg_end - seg_start

        # Closed-form integral over this segment:
        # ∫_seg α(t) dt = a_inf · Δt + (α(seg_start) − a_inf) · _segment_integral_factor(rate, Δt)
        integrated_iq += (
            a_inf * delta_t_seg
            + (alpha_at_segment_start - a_inf)
            * _segment_integral_factor(rate, delta_t_seg)
        )

        # Populate alpha_grid for any t in this segment
        in_seg = (t_grid >= seg_start) & (t_grid <= seg_end)
        for j in np.where(in_seg)[0]:
            tau = t_grid[j] - seg_start
            alpha_grid[j] = a_inf + (alpha_at_segment_start - a_inf) * np.exp(-rate * tau)

        # Update alpha at end of this segment for the next one
        alpha_at_segment_start = (
            a_inf + (alpha_at_segment_start - a_inf) * np.exp(-rate * delta_t_seg)
        )

    return alpha_grid, integrated_iq
