"""Collapse operators and Hamiltonian builder for the readout simulation.

Collapse operators are constructed in the dressed transmon eigenbasis, not
the bare charge basis and not a 2-level approximation. This matters for
Module 2 leakage tracking. Pure dephasing in the multi-level transmon uses
the convention (|j><j| − |0><0|) for j > 0 with per-level rate scaling.
See Blais et al. RMP 93, 025005 (2021) §III.E.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import qutip as qt
from scipy.special import erf

from .config import DeviceConfig, DriveParams
from .transmon import charge_operator_matrix_elements, diagonalize_transmon


def build_collapse_operators(
    device: DeviceConfig,
    transmon_basis_dim: int,
    resonator_dim: int,
) -> list[qt.Qobj]:
    """Lindblad collapse operators in the (transmon ⊗ resonator) Hilbert space.

    Channels:
      1. Resonator decay:   sqrt(κ (1 + n_th))  a
      2. Resonator heating: sqrt(κ  n_th)       a†     (only if n_th > 0)
      3. Qubit relaxation:  per-transition amplitudes in dressed transmon basis,
         scaled by |<j|n̂|k>|² relative to |<0|n̂|1>|² for |j+1⟩ → |j⟩ transitions.
      4. Qubit pure dephasing: sqrt(2 γ_φ) (|j><j| − |0><0|) for j = 1, ..., Nq−1.
      5. Qubit thermal heating: reverse of (3) scaled by n_th (only if n_th > 0).
    """
    tr = device.truncation
    Nq = transmon_basis_dim
    Nr = resonator_dim
    kappa = device.resonator.kappa
    gamma_1 = device.decoherence.gamma_1
    gamma_phi = device.decoherence.gamma_phi
    n_th = device.decoherence.n_th

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    c_ops: list[qt.Qobj] = []

    # 1. Resonator decay
    c_ops.append(np.sqrt(kappa * (1.0 + n_th)) * a)

    # 2. Resonator heating (only if bath is warm)
    if n_th > 0:
        c_ops.append(np.sqrt(kappa * n_th) * a.dag())

    # Build charge matrix elements for relaxation scaling in the dressed basis.
    _, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    # Normalize so |<0|n̂|1>|² is the reference scale (rate γ_1 applies to |1>→|0>).
    ref_sq = abs(n_mat[0, 1]) ** 2

    # 3. Qubit relaxation: |j+1> -> |j> for j = 0, 1, ..., Nq-2
    for j in range(Nq - 1):
        scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
        rate = gamma_1 * scale * (1.0 + n_th)
        if rate > 0:
            op = qt.basis(Nq, j) * qt.basis(Nq, j + 1).dag()
            c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    # 4. Qubit pure dephasing: one projector per level, L_j = sqrt(γ_φ) |j><j|.
    #
    # Deviation from plan, which used sqrt(2γ_φ) × (|j><j| − |0><0|) for
    # j = 1..Nq-1. That form cross-couples through the shared |0><0| term:
    # every L_j contributes to dephasing of ρ_{01}, giving an effective rate
    # of (Nq+2) γ_φ on the |0>-|1> coherence rather than γ_φ. At Nq=5 the
    # V4a test fit ~7× the input γ_φ, matching the 4γ_φ + (Nq-2)γ_φ
    # arithmetic exactly. The per-level-projector form below gives γ_φ
    # decay on every coherence ρ_{jk}, j≠k, independent of Nq — the
    # expected behavior when "gamma_phi" is the measured dephasing rate.
    if gamma_phi > 0:
        for j in range(Nq):
            proj = qt.basis(Nq, j) * qt.basis(Nq, j).dag()
            c_ops.append(np.sqrt(gamma_phi) * qt.tensor(proj, qt.qeye(Nr)))

    # 5. Qubit thermal heating (reverse direction)
    if n_th > 0:
        for j in range(Nq - 1):
            scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
            rate = gamma_1 * scale * n_th
            if rate > 0:
                op = qt.basis(Nq, j + 1) * qt.basis(Nq, j).dag()
                c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    return c_ops


def build_hamiltonian(
    device: DeviceConfig,
    drive_params: DriveParams,
    frame: Literal["rotating", "dispersive"] = "rotating",
) -> tuple[qt.Qobj, list]:
    """Drift Hamiltonian + QuTiP-compatible drive spec.

    Rotating frame at ω_d = ω_r + detuning:
      H_q  = Σ_j (ω_j − j ω_d) |j><j| ⊗ I_r
      H_r  = (ω_r − ω_d) a†a
      H_c  = g Σ_{jk} <j|n̂|k> |j><k| ⊗ (a + a†)
      H_drive(t) = ε(t) (a + a†)

    ε(t) is an erf-difference flat-top pulse with Gaussian edges of width σ.

    'dispersive' frame is not implemented in Module 1; it is reserved for
    validation-only use. Calling with frame='dispersive' raises
    NotImplementedError. Do not silently return rotating frame instead.
    """
    if frame not in ("rotating",):
        raise NotImplementedError(f"frame '{frame}' not implemented in Module 1")

    tr = device.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    # Drive frequency: on resonance with resonator plus optional detuning.
    omega_d = device.resonator.omega_r + drive_params.detuning

    # Transmon term in rotating frame: diag(omega_j - j * omega_d)
    qubit_diag = np.array([energies[j] - j * omega_d for j in range(Nq)])
    H_q = qt.tensor(qt.Qobj(np.diag(qubit_diag)), qt.qeye(Nr))

    # Resonator term: (omega_r - omega_d) a†a
    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    H_r = (device.resonator.omega_r - omega_d) * a.dag() * a

    # Coupling term: g * <j|n̂|k> * |j><k| ⊗ (a + a†)
    # Retain only adjacent selection-rule contributions; full matrix keeps all.
    n_op_q = qt.tensor(qt.Qobj(n_mat), qt.qeye(Nr))
    H_c = device.coupling.g * n_op_q * (a + a.dag())

    # Symmetrize to absorb floating-point asymmetries (~1e-6 rad/s at
    # frequency scales of 1e10 rad/s, well below any physical resolution);
    # QuTiP's isherm check is strict and would otherwise return False.
    H0_raw = H_q + H_r + H_c
    H0 = 0.5 * (H0_raw + H0_raw.dag())

    # Drive operator: ε(t) (a + a†)
    drive_op = a + a.dag()

    # Envelope: erf-difference flat-top with sigma_edge gaussian rise/fall.
    eps_0 = drive_params.amplitude
    t_end = drive_params.duration
    sigma = drive_params.edge_sigma
    t_rise = 3.0 * sigma
    t_fall = t_end - t_rise
    if t_fall <= t_rise + 2.0 * sigma:
        raise ValueError(
            f"Drive duration {t_end*1e9:.1f} ns too short for rise/fall "
            f"width {sigma*1e9:.1f} ns; need t_end > 6*sigma + 2*sigma."
        )

    def envelope(t: float, args: dict) -> float:
        return 0.5 * eps_0 * (erf((t - t_rise) / sigma) - erf((t - t_fall) / sigma))

    return H0, [drive_op, envelope]
