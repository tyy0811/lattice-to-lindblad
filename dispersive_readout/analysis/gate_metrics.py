"""Gate-cycle metrics: transfer fidelity, leakage (final + peak), ε_X.

These operate on qubit-only density matrices in the Duffing-oscillator basis
returned by `dispersive_readout.control.gate_simulator.simulate_x_gate`.

ε_X = 1 − F_transfer is the downstream-consumable bit-flip-error number that
the eventual Module 5b spec inherits as data (see MODULE_5a_SPEC.md §10).
"""
from __future__ import annotations

from typing import Iterable

import qutip as qt


def transfer_fidelity_0_to_1(rho_final: qt.Qobj) -> float:
    """Population in |1⟩ after the gate: ⟨1|ρ|1⟩."""
    n_levels = rho_final.shape[0]
    proj_1 = qt.basis(n_levels, 1) * qt.basis(n_levels, 1).dag()
    return float((proj_1 * rho_final).tr().real)


def leakage_population(rho_final: qt.Qobj, n_levels: int) -> float:
    """Σ_{k≥2} ⟨k|ρ|k⟩ — total population that escaped the qubit subspace."""
    total = 0.0
    for k in range(2, n_levels):
        proj_k = qt.basis(n_levels, k) * qt.basis(n_levels, k).dag()
        total += float((proj_k * rho_final).tr().real)
    return total


def leakage_peak(rho_t: Iterable[qt.Qobj], n_levels: int) -> float:
    """Maximum of Σ_{k≥2} ⟨k|ρ(t)|k⟩ over a time-resolved density-matrix list."""
    return max(leakage_population(rho, n_levels) for rho in rho_t)


def epsilon_x_from_transfer(rho_final: qt.Qobj) -> float:
    """Classical bit-flip error of the conditional X-gate: 1 − ⟨1|ρ|1⟩.

    See MODULE_5a_SPEC.md §10 for the headline `ε_X^ref` reporting contract.
    """
    return 1.0 - transfer_fidelity_0_to_1(rho_final)
