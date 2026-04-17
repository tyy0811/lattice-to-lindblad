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

    # 4. Qubit pure dephasing: rate sqrt(2 γ_φ) for each upper level
    for j in range(1, Nq):
        if gamma_phi > 0:
            proj = (
                qt.basis(Nq, j) * qt.basis(Nq, j).dag()
                - qt.basis(Nq, 0) * qt.basis(Nq, 0).dag()
            )
            c_ops.append(np.sqrt(2.0 * gamma_phi) * qt.tensor(proj, qt.qeye(Nr)))

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
    """Rotating-frame drift + time-dependent drive spec. Implemented in Task 11."""
    raise NotImplementedError  # Task 11
