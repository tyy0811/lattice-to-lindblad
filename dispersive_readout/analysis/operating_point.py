"""Operating-point dataclass and analytic drive-amplitude calibration.

See MODULE_2_SPEC.md §2.3 for the closed-form calibration derivation
and §5.1 for the API contract.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..physics.config import DeviceConfig, DriveParams, REFERENCE_DEVICE
from ..physics.dispersive import dispersive_shift_full
from ..physics.transmon import charge_operator_matrix_elements, diagonalize_transmon
from ..physics.readout_model import (
    simulate_readout,
    compute_assignment_fidelity,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Fixed operating point for error-budget analysis.

    Attributes
    ----------
    device : DeviceConfig
    drive : DriveParams
        Readout drive with amplitude calibrated per §2.3.
    integration_window : tuple[float, float]
        (t0, t1) for IQ integration, seconds.
    n_shots : int
        Shots per fidelity evaluation.
    """
    device: DeviceConfig
    drive: DriveParams
    integration_window: tuple[float, float]
    n_shots: int


def _response_factor_M(device: DeviceConfig) -> complex:
    """Steady-state separation-per-unit-drive factor M for on-resonance drive.

    M = 1/(κ/2 − iχ_0) − 1/(κ/2 − iχ_1). Uses the per-level χ_j from
    dispersive_shift_full (non-RWA 2nd-order PT including Bloch-Siegert).
    |M| has units of s/rad.
    """
    tr = device.truncation
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    chi = dispersive_shift_full(energies, n_mat, device.coupling.g,
                                 device.resonator.omega_r)
    kappa = device.resonator.kappa
    M = 1.0 / (0.5 * kappa - 1j * chi[0]) - 1.0 / (0.5 * kappa - 1j * chi[1])
    return M


def _analytic_epsilon_0(
    device: DeviceConfig, target_fidelity: float, t_int: float
) -> float:
    """Solve ε₀ from SNR_target = 2 × |M| × sqrt(κ T_int) × ε₀.

    SNR_target = 2 × Φ⁻¹(F_target) from F = 1 − Q(SNR/2) and Q(x) = 1 − Φ(x).
    """
    snr_target = 2.0 * norm.ppf(target_fidelity)
    M = _response_factor_M(device)
    kappa = device.resonator.kappa
    epsilon_0 = snr_target / (2.0 * abs(M) * math.sqrt(kappa * t_int))
    return float(epsilon_0)


def _grid_search_epsilon_0(
    device: DeviceConfig,
    duration: float,
    integration_window: tuple[float, float],
    target_fidelity: float,
    n_shots: int,
    n_grid: int = 15,
) -> float:
    """Fallback: grid-scan the low-ε branch, return lowest ε with F ≥ target.

    Bracket: ε_min (where F ≈ 0.5, chosen at 0.1× analytic) to ε_max
    (where n̄_peak ≈ 0.5 × N_resonator).
    """
    epsilon_analytic = _analytic_epsilon_0(
        device, target_fidelity, integration_window[1] - integration_window[0]
    )
    eps_min = 0.1 * epsilon_analytic
    eps_max = 3.0 * epsilon_analytic
    grid = np.linspace(eps_min, eps_max, n_grid)

    for eps in grid:
        drv = DriveParams(amplitude=float(eps), duration=duration, detuning=0.0)
        r0 = simulate_readout(device, drv, initial_qubit_state=0)
        r1 = simulate_readout(device, drv, initial_qubit_state=1)
        f = compute_assignment_fidelity(
            r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian"
        )
        if f.F_assign >= target_fidelity:
            return float(eps)

    raise RuntimeError(
        f"Grid search did not find ε₀ achieving F ≥ {target_fidelity} on "
        f"low-ε branch [{eps_min:.2e}, {eps_max:.2e}] rad/s. Target unreachable."
    )


def calibrate_drive_amplitude(
    device: DeviceConfig,
    duration: float,
    integration_window: tuple[float, float],
    target_fidelity: float = 0.99,
    n_shots: int = 10_000,
    sigma_tolerance_factor: float = 3.0,
) -> float:
    """Analytic drive-amplitude calibration with simulation-verified fallback.

    Computes ε₀ from the dispersive-regime steady-state SNR formula
    (§2.3). Verifies against a simulation; if the measured F deviates
    from target by more than sigma_tolerance_factor × σ_shot, falls back
    to grid search on the low-ε branch and emits a warning.

    Parameters
    ----------
    device : DeviceConfig
    duration : float
        Pulse duration in seconds.
    integration_window : tuple[float, float]
        (t0, t1) for IQ integration.
    target_fidelity : float
        F target for calibration; default 0.99.
    n_shots : int
        Shots for the verification measurement.
    sigma_tolerance_factor : float
        Fallback trigger band in units of σ_shot.

    Returns
    -------
    epsilon_0 : float
        Drive amplitude in rad/s.

    Raises
    ------
    RuntimeError
        If both analytic and grid search fail to achieve target.
    """
    t_int = integration_window[1] - integration_window[0]
    eps_analytic = _analytic_epsilon_0(device, target_fidelity, t_int)

    # Verification sim at eps_analytic
    drv = DriveParams(amplitude=eps_analytic, duration=duration, detuning=0.0)
    r0 = simulate_readout(device, drv, initial_qubit_state=0)
    r1 = simulate_readout(device, drv, initial_qubit_state=1)
    f_verified = compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian",
        rng=np.random.default_rng(seed=42),  # deterministic verification
    )

    sigma_shot = math.sqrt(
        target_fidelity * (1.0 - target_fidelity) / n_shots
    )
    tolerance = sigma_tolerance_factor * sigma_shot

    if abs(f_verified.F_assign - target_fidelity) <= tolerance:
        return eps_analytic

    warnings.warn(
        f"Analytic calibration gave F_verified={f_verified.F_assign:.4f}, "
        f"expected {target_fidelity}±{tolerance:.4f}. Falling back to grid "
        f"search on low-ε branch.",
        RuntimeWarning,
    )
    return _grid_search_epsilon_0(
        device, duration, integration_window, target_fidelity, n_shots
    )


def get_reference_operating_point(n_shots: int = 10_000) -> OperatingPoint:
    """Return the canonical operating point for Figure 2.

    Calibration runs on first call (< 3 s total: analytic solve + one
    verification sim × two qubit states). No persistent cache — fast
    enough to compute on demand.

    Parameters
    ----------
    n_shots : int
        Shots per fidelity evaluation. Default 10_000 for CI speed; Figure 2
        script calls with n_shots=100_000 to recover the physics-dominated
        regime for the waterfall (amendment 8: at 10_000 shots, (F_ideal −
        F_full) ≈ σ_shot, so B2's per-channel bars and residual-vs-denom
        ratio are noise-smeared). The calibration itself always uses 10_000
        shots — σ_shot tolerance is fine for single-point ε₀ recovery.
    """
    integration_window = (50e-9, 500e-9)
    epsilon_0 = calibrate_drive_amplitude(
        device=REFERENCE_DEVICE,
        duration=500e-9,
        integration_window=integration_window,
        target_fidelity=0.99,
        n_shots=10_000,
    )
    return OperatingPoint(
        device=REFERENCE_DEVICE,
        drive=DriveParams(
            amplitude=epsilon_0,
            duration=500e-9,
            detuning=0.0,
            edge_sigma=2e-9,
        ),
        integration_window=integration_window,
        n_shots=n_shots,
    )
