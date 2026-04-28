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


def test_average_gate_fidelity_x_returns_value_and_per_state():
    """Post-N12: average_gate_fidelity_x runs simulate_x_gate from each of the
    four Pauli-set input states and returns (F_avg, per_state). At T=20ns with
    fidelity-optimal β=0.5 (decoherence zeroed), F_avg ≈ 0.9999 — comfortably
    above the one-way transfer fidelity at the same β.
    """
    from dispersive_readout.analysis.gate_metrics import average_gate_fidelity_x
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DecoherenceParams

    decoh_zero = DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)
    F_avg, per_state = average_gate_fidelity_x(
        device=REFERENCE_DEVICE,
        T_gate=20e-9,
        n_levels=4,
        drag=True,
        beta=0.5,
        decoherence=decoh_zero,
        sigma=20e-9 / 4.0,
    )
    assert 0.0 <= F_avg <= 1.0
    assert len(per_state) == 4
    # Empirical: at β=0.5, T=20ns, REFERENCE α, decoherence zeroed, F_avg > 0.999.
    assert F_avg > 0.999, f"Expected F_avg > 0.999 at β=0.5/T=20ns; got {F_avg:.6f}"


def test_average_gate_fidelity_x_catches_phase_error_one_way_misses():
    """Methodology test (post-N12 / Codex finding): a pulse implementing iX
    (with global phase) preserves transfer fidelity 0→1 but rotates |+i⟩
    differently. average_gate_fidelity_x catches this; transfer alone doesn't.

    Practical check: for the actual sin²-windowed gate at β=0 (no DRAG, severe
    coherent error), the per-state |+i⟩→|-i⟩ fidelity is materially below the
    |0⟩→|1⟩ transfer fidelity. The two metrics diverge whenever DRAG is sub-
    optimal — so a non-trivial relationship between them exists, which is
    exactly what motivated the metric upgrade.
    """
    from dispersive_readout.analysis.gate_metrics import average_gate_fidelity_x
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DecoherenceParams

    decoh_zero = DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)
    # No DRAG at T=20ns: significant coherent error
    F_avg, per_state = average_gate_fidelity_x(
        device=REFERENCE_DEVICE,
        T_gate=20e-9,
        n_levels=4,
        drag=False,
        beta=0.0,
        decoherence=decoh_zero,
        sigma=20e-9 / 4.0,
    )
    transfer_0to1 = per_state[0]   # |0⟩→|1⟩
    transfer_1to0 = per_state[1]   # |1⟩→|0⟩
    # Both directions should give similar fidelity (X is symmetric in the
    # qubit subspace; Duffing breaks this slightly but symmetrically).
    assert abs(transfer_0to1 - transfer_1to0) < 0.05, (
        f"Forward/reverse asymmetry suggests a metric bug: |0⟩→|1⟩={transfer_0to1:.4f}, "
        f"|1⟩→|0⟩={transfer_1to0:.4f}."
    )
    # F_avg lies between max and min of per_state values
    assert min(per_state) <= F_avg <= max(per_state) + 1e-12
