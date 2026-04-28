"""Gate-cycle metrics: transfer fidelity (one-way diagnostic), average X-gate
fidelity over the Pauli set (shipped headline metric, post-N12), leakage
(final + peak), and ε_X.

These operate on qubit-only density matrices in the Duffing-oscillator basis
returned by `dispersive_readout.control.gate_simulator.simulate_x_gate`.

ε_X = 1 − F_avg (computed via `average_gate_fidelity_x`) is the downstream-
consumable bit-flip-error number that Module 5b inherits as data — averaged
over the four Pauli eigenstates {|0⟩, |1⟩, |+⟩, |+i⟩}, which catches
asymmetric forward/reverse action and coherent superposition-state phase
errors that a one-way population transfer would silently miss
(see MODULE_5a_SPEC.md §10 and §12.1 (N12)).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import qutip as qt

if TYPE_CHECKING:
    from ..physics.config import DecoherenceParams, DeviceConfig


def transfer_fidelity_0_to_1(rho_final: qt.Qobj) -> float:
    """One-way population transfer ⟨1|ρ|1⟩ for a gate initialized from |0⟩.

    DIAGNOSTIC ONLY — this measures only `|0⟩ → |1⟩` and is **not** a complete
    X-gate metric. A pulse can satisfy this scalar while having a bad
    `|1⟩ → |0⟩` action or a coherent superposition-state phase error. For the
    shipped X-gate-error number, use `average_gate_fidelity_x` (post-N12).
    """
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
    """One-way bit-flip-error diagnostic: 1 − ⟨1|ρ|1⟩ from a |0⟩ start.

    DIAGNOSTIC ONLY (post-N12) — see `transfer_fidelity_0_to_1` for the
    failure modes this metric misses. Use `epsilon_x_average` (1 − F_avg)
    for the shipped headline number.
    """
    return 1.0 - transfer_fidelity_0_to_1(rho_final)


def _pauli_x_input_target_pairs(n_levels: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Four input/target ket pairs for X-gate fidelity averaging:
    |0⟩→|1⟩, |1⟩→|0⟩, |+⟩→|+⟩, |+i⟩→|-i⟩ (X eigenstates / Y eigenstates with
    global phase absorbed)."""
    e0 = qt.basis(n_levels, 0)
    e1 = qt.basis(n_levels, 1)
    plus = (e0 + e1).unit()
    plus_i = (e0 + 1j * e1).unit()
    minus_i = (e0 - 1j * e1).unit()
    return [
        (e0, e1),
        (e1, e0),
        (plus, plus),
        (plus_i, minus_i),
    ]


def average_gate_fidelity_x(
    device: "DeviceConfig",
    T_gate: float,
    n_levels: int = 4,
    drag: bool = True,
    beta: float = 1.0,
    decoherence: "DecoherenceParams | None" = None,
    sigma: float | None = None,
) -> tuple[float, list[float]]:
    """Average X-gate fidelity over the Pauli set {|0⟩, |1⟩, |+⟩, |+i⟩}.

    Runs `simulate_x_gate` from each of the four input states and computes
    `⟨target | ρ_out | target⟩` against the corresponding X-target state. The
    average is the shipped headline X-gate fidelity (post-N12); the per-state
    list is returned alongside as a diagnostic.

    Failure modes caught (vs one-way `transfer_fidelity_0_to_1`):
    - Asymmetric forward/reverse action (`|0⟩→|1⟩` works but `|1⟩→|0⟩` doesn't).
    - Coherent phase errors visible only on superposition inputs (e.g.,
      a pulse implementing `iX` instead of `X` would still pass transfer
      fidelity but fail on |+i⟩).

    Returns
    -------
    (F_avg, per_state_fidelities)
        F_avg : float in [0, 1]; mean of the four per-input fidelities.
        per_state_fidelities : list of floats in input order
                               [|0⟩→|1⟩, |1⟩→|0⟩, |+⟩→|+⟩, |+i⟩→|-i⟩].
    """
    # Deferred import to avoid a circular dependency
    # (gate_simulator imports control.pulses, which doesn't touch this module).
    from ..control.gate_simulator import simulate_x_gate

    pairs = _pauli_x_input_target_pairs(n_levels)
    fids: list[float] = []
    for psi_in, psi_target in pairs:
        result = simulate_x_gate(
            device=device,
            T_gate=T_gate,
            n_levels=n_levels,
            drag=drag,
            beta=beta,
            decoherence=decoherence,
            sigma=sigma,
            init_state=psi_in,
        )
        proj_target = psi_target * psi_target.dag()
        fids.append(float((proj_target * result.rho_final).tr().real))
    f_avg = sum(fids) / len(fids)
    return f_avg, fids


def epsilon_x_average(F_avg: float) -> float:
    """Shipped headline X-gate error: 1 − F_avg over the Pauli set (post-N12)."""
    return 1.0 - F_avg
