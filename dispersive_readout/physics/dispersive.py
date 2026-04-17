"""Analytic and numerical dispersive-shift formulas.

χ convention: χ ≡ (χ_1 − χ_0)/2, the half-splitting observable in readout.
dispersive_shift_full returns per-level χ_j; the caller computes the
half-splitting from those as needed.
"""
from __future__ import annotations

import numpy as np

from .config import DeviceConfig


def dispersive_shift_two_level(g: float, Delta: float) -> float:
    """Two-level-limit dispersive shift: χ = g² / Δ.

    Inputs are in rad/s; output in rad/s. For Δ < 0 (qubit below resonator,
    the reference device's regime) this is negative.
    """
    return (g ** 2) / Delta


def dispersive_shift_full(
    energies: np.ndarray,
    n_matrix: np.ndarray,
    g: float,
    omega_r: float,
) -> np.ndarray:
    """Multi-level per-level dispersive shifts χ_j.

    χ_j = sum_{k != j} |g <j|n̂|k>|² [ 1/(ω_j - ω_k - ω_r) - 1/(ω_j - ω_k + ω_r) ]

    The observable readout shift is (χ_1 − χ_0)/2.
    """
    N = len(energies)
    chi = np.zeros(N, dtype=float)
    for j in range(N):
        total = 0.0
        for k in range(N):
            if k == j:
                continue
            coupling_sq = (g * abs(n_matrix[j, k])) ** 2
            delta_jk = energies[j] - energies[k]
            denom_minus = delta_jk - omega_r
            denom_plus = delta_jk + omega_r
            if denom_minus == 0.0 or denom_plus == 0.0:
                raise ValueError(
                    f"Degeneracy in denominators at j={j}, k={k}: "
                    f"delta={delta_jk}, omega_r={omega_r}"
                )
            total += coupling_sq * (1.0 / denom_minus - 1.0 / denom_plus)
        chi[j] = total
    return chi


def dispersive_shift_from_simulation(device: DeviceConfig) -> float:
    """Extract χ ≡ (χ₁ − χ₀)/2 from the dressed Jaynes-Cummings spectrum."""
    raise NotImplementedError  # Task 8
