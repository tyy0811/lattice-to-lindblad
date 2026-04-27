"""Smoke tests for gate metric helpers (transfer fidelity, leakage, ε_X)."""
from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from dispersive_readout.analysis.gate_metrics import (
    epsilon_x_from_transfer,
    leakage_peak,
    leakage_population,
    transfer_fidelity_0_to_1,
)


def _basis_proj(n_levels: int, j: int) -> qt.Qobj:
    return qt.basis(n_levels, j) * qt.basis(n_levels, j).dag()


def test_transfer_fidelity_orthogonal_states():
    """⟨1|ρ|1⟩ = 1 for ρ = |1⟩⟨1|, 0 for ρ = |0⟩⟨0|."""
    rho_1 = _basis_proj(4, 1)
    rho_0 = _basis_proj(4, 0)
    assert transfer_fidelity_0_to_1(rho_1) == pytest.approx(1.0)
    assert transfer_fidelity_0_to_1(rho_0) == pytest.approx(0.0)


def test_leakage_population_sums_levels_two_and_above():
    """ρ = 0.5|0⟩⟨0| + 0.2|1⟩⟨1| + 0.3|2⟩⟨2| → leakage = 0.3."""
    n_levels = 4
    rho = 0.5 * _basis_proj(n_levels, 0) + 0.2 * _basis_proj(n_levels, 1) + 0.3 * _basis_proj(n_levels, 2)
    assert leakage_population(rho, n_levels) == pytest.approx(0.3)


def test_leakage_peak_max_over_trajectory():
    """Synthetic rho_t with known peak leakage at midpoint."""
    n_levels = 3
    populations = [0.0, 0.05, 0.4, 0.05, 0.0]
    rho_t = [
        (1.0 - p) * _basis_proj(n_levels, 1) + p * _basis_proj(n_levels, 2)
        for p in populations
    ]
    assert leakage_peak(rho_t, n_levels) == pytest.approx(0.4)


def test_epsilon_x_complement_to_transfer():
    """epsilon_x_from_transfer(ρ) + transfer_fidelity_0_to_1(ρ) == 1."""
    rho = 0.7 * _basis_proj(4, 1) + 0.2 * _basis_proj(4, 0) + 0.1 * _basis_proj(4, 2)
    assert (
        epsilon_x_from_transfer(rho) + transfer_fidelity_0_to_1(rho)
    ) == pytest.approx(1.0, abs=1e-12)
