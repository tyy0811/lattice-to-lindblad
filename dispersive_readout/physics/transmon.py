"""Charge-basis transmon: Hamiltonian, diagonalization, matrix elements, summary.

Convention: ground-state energy is shifted to 0 after diagonalization.
All energies in rad/s. The transmon eigenbasis ("dressed transmon basis")
is the reference basis used by lindblad.py and readout_model.py.
"""
from __future__ import annotations

import numpy as np

from .config import TransmonParams, TruncationParams


def charge_basis_hamiltonian(
    params: TransmonParams,
    trunc: TruncationParams,
) -> np.ndarray:
    """Transmon Hamiltonian in the charge basis.

    H = 4 E_C (n - n_g)^2 - (E_J / 2) (|n><n+1| + |n+1><n|)

    The charge ladder runs over n = -N//2, ..., +N//2 and must be odd-sized
    so it is symmetric about n = 0. Returns a real symmetric matrix in rad/s.
    """
    N = trunc.N_charge
    if N % 2 == 0:
        raise ValueError(f"N_charge must be odd (got {N}) so the ladder is symmetric about zero.")
    n_values = np.arange(-(N // 2), N // 2 + 1, dtype=float)
    H = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(H, 4.0 * params.E_C * (n_values - params.n_g) ** 2)
    off = -0.5 * params.E_J
    idx = np.arange(N - 1)
    H[idx, idx + 1] = off
    H[idx + 1, idx] = off
    return H


def diagonalize_transmon(
    params: TransmonParams,
    trunc: TruncationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize and return (energies, eigenstates) for the lowest N_transmon levels.

    Ground-state energy is shifted to 0. Eigenstates are returned as a
    (N_charge, N_transmon) array whose columns are eigenvectors in the charge
    basis.
    """
    H = charge_basis_hamiltonian(params, trunc)
    # np.linalg.eigh returns ascending eigenvalues for Hermitian input.
    eigvals_all, eigvecs_all = np.linalg.eigh(H)
    energies = eigvals_all[: trunc.N_transmon].copy()
    eigenstates = eigvecs_all[:, : trunc.N_transmon].copy()
    energies -= energies[0]  # shift ground state to zero by convention
    return energies, eigenstates


def charge_operator_matrix_elements(
    eigenstates: np.ndarray,
    trunc: TruncationParams,
) -> np.ndarray:
    """<j|n_hat|k> in the truncated transmon basis."""
    raise NotImplementedError  # Task 4


def transmon_summary(params: TransmonParams, trunc: TruncationParams) -> dict:
    """Diagnostic dict: omega_01, omega_12, alpha, E_J_over_E_C, charge dispersion, n-matrix elements."""
    raise NotImplementedError  # Task 4
