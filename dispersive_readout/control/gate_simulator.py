"""Single-qubit X-gate simulator on a qubit-only Duffing-oscillator Hilbert space.

Frame: rotating frame at drive frequency ω_d = ω_q (the |0⟩↔|1⟩ transition).
Drive: in-phase Ω_x(t) and DRAG quadrature Ω_y(t) (both rad/s).
Anharmonicity: α extracted from `transmon_summary` — single source of truth.
Lindblad channels (when enabled): qubit T₁ relaxation with Duffing-ladder
n-scaling (b → √n |n−1⟩⟨n|), per-level pure dephasing matching Module 2's
convention (lindblad.py:92-105), Purcell-as-effective-qubit-channel with
γ_P = (g/Δ)²·κ at the |0⟩↔|1⟩ transition, and thermal heating.

For n_levels = 2, the Duffing anharmonicity term α/2 · b†b · (b†b−1) vanishes
(b†b ∈ {0, 1}), so α is not invoked and `transmon_summary` is not called.
This is the V1 path. n_levels ≥ 3 invokes `transmon_summary`.

If drag=True with n_levels=2, ValueError — DRAG requires α to be defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import qutip as qt

from ..physics.config import DecoherenceParams, DeviceConfig
from ..physics.transmon import transmon_summary
from .pulses import (
    calibrate_pi_pulse_amplitude,
    drag_correction,
    sin2_windowed_gaussian,
    sin2_windowed_gaussian_derivative,
)


@dataclass(frozen=True)
class GateResult:
    """Output of a single X-gate simulation.

    rho_final     : Qobj (n_levels × n_levels) at t = T_gate.
    rho_t         : list of Qobj at each t_array sample.
    t_array       : np.ndarray, simulation time grid.
    alpha_used    : float, anharmonicity used in the Hamiltonian (rad/s);
                    NaN if n_levels = 2 (anharmonicity term vanishes).
    beta_used     : float.
    A_calibrated  : float, π-pulse amplitude.
    pulse_area    : float, ∫_0^T Ω_x dt (should ≈ π).
    """
    rho_final: qt.Qobj
    rho_t: list
    t_array: np.ndarray
    alpha_used: float
    beta_used: float
    A_calibrated: float
    pulse_area: float


def _build_qubit_collapse_operators(
    device: DeviceConfig,
    n_levels: int,
    decoherence: DecoherenceParams,
) -> List[qt.Qobj]:
    """Qubit-only Duffing-basis Lindblad operators.

    Conventions:
    - Relaxation:    √(γ_1 (1 + n_th)) · b   (n-scaled by √n via destroy ladder)
    - Pure dephasing: √γ_φ · |j⟩⟨j| for j = 0..n_levels-1 (per-level, matches
                      lindblad.py:92-105).
    - Purcell:       √(γ_P (1 + n_th)) · b   with γ_P = (g/Δ)² · κ where
                     Δ = ω_q − ω_r (computed from transmon_summary['omega_01']
                     and device.resonator.omega_r).
    - Thermal:       √(γ_1 n_th) · b†
    """
    c_ops: list[qt.Qobj] = []
    b = qt.destroy(n_levels)
    bdag = b.dag()

    g1 = decoherence.gamma_1
    gphi = decoherence.gamma_phi
    n_th = decoherence.n_th

    # 1. Qubit T₁ relaxation (Duffing-ladder n-scaling via destroy operator)
    if g1 > 0:
        c_ops.append(np.sqrt(g1 * (1.0 + n_th)) * b)

    # 2. Per-level pure dephasing (matches Module 2 convention, lindblad.py:92-105)
    if gphi > 0:
        for j in range(n_levels):
            proj = qt.basis(n_levels, j) * qt.basis(n_levels, j).dag()
            c_ops.append(np.sqrt(gphi) * proj)

    # 3. Thermal heating
    if g1 > 0 and n_th > 0:
        c_ops.append(np.sqrt(g1 * n_th) * bdag)

    # 4. Purcell decay, leading-order dispersive limit (spec §3.6)
    if decoherence.purcell_enabled and n_levels >= 2:
        summary = transmon_summary(device.transmon, device.truncation)
        omega_q = summary["omega_01"]
        delta = omega_q - device.resonator.omega_r  # signed; squared in formula
        g = device.coupling.g
        kappa = device.resonator.kappa
        gamma_purcell = (g / delta) ** 2 * kappa
        if gamma_purcell > 0:
            c_ops.append(np.sqrt(gamma_purcell * (1.0 + n_th)) * b)

    return c_ops


def simulate_x_gate(
    device: DeviceConfig,
    T_gate: float,
    n_levels: int = 4,
    drag: bool = True,
    beta: float = 1.0,
    decoherence: DecoherenceParams | None = None,
    sigma: float | None = None,
    n_time_points: int = 401,
) -> GateResult:
    """Simulate a single X-gate cycle on the qubit (Duffing approximation).

    Parameters
    ----------
    device : DeviceConfig
        Provides transmon parameters (for α via transmon_summary), resonator
        parameters (for Δ in Purcell rate), coupling g, and decoherence.
    T_gate : float
        Total pulse length in seconds.
    n_levels : int
        Duffing-Hilbert-space dimension. Must be 2, 3, 4, or 5. At n_levels = 2,
        the anharmonicity term vanishes and α is not invoked (V1 path).
    drag : bool
        If True, add the DRAG quadrature drive Ω_y(t).
    beta : float
        DRAG coefficient. Only used when drag=True.
    decoherence : DecoherenceParams | None
        Override for device.decoherence. Pass a zeroed-out DecoherenceParams
        to disable Lindblad (V1, V2, V4, V5a). None means use device.decoherence.
    sigma : float | None
        Pulse-width parameter. Defaults to T_gate / 4.
    n_time_points : int
        Number of mesolve output samples.

    Raises
    ------
    ValueError
        If drag=True with n_levels=2 (α undefined in two-level Duffing).
        If n_levels not in {2, 3, 4, 5}.
    """
    if n_levels not in (2, 3, 4, 5):
        raise ValueError(f"n_levels must be 2, 3, 4, or 5; got {n_levels}.")
    if drag and n_levels == 2:
        raise ValueError(
            "drag=True is incompatible with n_levels=2 — the Duffing anharmonicity "
            "term vanishes for a two-level system, so α is undefined and DRAG "
            "cannot be computed. Use n_levels >= 3 for DRAG validation."
        )
    if decoherence is None:
        decoherence = device.decoherence
    if sigma is None:
        sigma = T_gate / 4.0

    # Calibrate π-pulse amplitude
    A = calibrate_pi_pulse_amplitude(T_gate, sigma)

    # Anharmonicity (only invoked at n_levels >= 3; vanishes at n_levels = 2)
    if n_levels >= 3:
        summary = transmon_summary(device.transmon, device.truncation)
        alpha = float(summary["alpha"])
    else:
        alpha = float("nan")

    # Hilbert-space operators
    b = qt.destroy(n_levels)
    bdag = b.dag()

    # Static anharmonicity term (zero for n_levels=2)
    if n_levels >= 3:
        H_anharm = 0.5 * alpha * bdag * b * (bdag * b - qt.qeye(n_levels))
    else:
        H_anharm = 0 * qt.qeye(n_levels)

    # Drive operators
    H_drive_x = 0.5 * (b + bdag)
    H_drive_y = 0.5 * (-1j * b + 1j * bdag)

    # Time-dependent drive callables (QuTiP 5.x signature: (t, args) -> complex)
    def omega_x(t, args):
        return float(sin2_windowed_gaussian(t, A, T_gate, sigma))

    def omega_y(t, args):
        if not drag:
            return 0.0
        return float(drag_correction(t, A, T_gate, sigma, alpha, beta=beta))

    H_total = [H_anharm, [H_drive_x, omega_x]]
    if drag:
        H_total.append([H_drive_y, omega_y])

    # Initial state |0⟩⟨0|
    psi0 = qt.basis(n_levels, 0)
    rho0 = psi0 * psi0.dag()

    # Time grid
    t_array = np.linspace(0.0, T_gate, n_time_points)

    # Collapse operators
    c_ops = _build_qubit_collapse_operators(device, n_levels, decoherence)

    # Solver options (matches Module 1's pattern in readout_model.py)
    opts = {"nsteps": 10_000, "atol": 1e-10, "rtol": 1e-8, "store_states": True}

    result = qt.mesolve(
        H=H_total,
        rho0=rho0,
        tlist=t_array,
        c_ops=c_ops,
        e_ops=[],
        options=opts,
    )

    rho_t = list(result.states)
    rho_final = rho_t[-1]

    # Verify pulse area as a soft sanity check (not an assertion — info field)
    pulse_area = float(np.trapezoid(
        sin2_windowed_gaussian(t_array, A, T_gate, sigma),
        t_array,
    ))

    return GateResult(
        rho_final=rho_final,
        rho_t=rho_t,
        t_array=t_array,
        alpha_used=alpha,
        beta_used=beta if drag else 0.0,
        A_calibrated=A,
        pulse_area=pulse_area,
    )
