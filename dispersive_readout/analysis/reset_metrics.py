"""Module 5b — derived metrics composing JointMatrix outputs.

active_advantage at matched duration: composes physics-tier JointMatrix
data and control-tier reset formulas into a single scalar comparison.
Lives in analysis/ because it's a derived comparison metric, not a
primitive — physics/ types flow downward to control + analysis; neither
of those imports from each other.
"""
from __future__ import annotations

from dispersive_readout.control.reset_protocol import (
    passive_reset_residual,
    reset_residual_single_cycle,
)
from dispersive_readout.physics.joint_matrix import JointMatrix


def active_advantage(
    p_e: float,
    joint: JointMatrix,
    gate_error: float,
    T1: float,
    tau_meas: float,
    tau_gate: float,
) -> float:
    """p_e · passive_reset_residual(T1, tau_meas + tau_gate)
       − reset_residual_single_cycle(p_e, joint, gate_error).

    Positive value: active reset beats passive at this matched duration.
    Negative value: passive baseline dominates.

    The matched duration includes the gate time τ_gate (5a's headline
    20 ns) so the comparison is fair: both protocols consume the same
    wall-clock time before the next operation.

    Prior-aware passive baseline (fix from adversarial review): the v0
    passive residual at prior p_e is

        p_e · exp(-(τ_meas + τ_gate) / T₁) + (1 − p_e) · 0
      = p_e · passive_reset_residual(T₁, τ_total)

    The leading p_e factor is required because passive_reset_residual
    itself is the conditional residual GIVEN the qubit started in |e⟩.
    The (1 − p_e) ground-state weight contributes 0 to passive residual
    in v0's strict zero-temperature regime (extract_joint_matrix and
    device_idx18 both enforce n_th = 0). Without the leading p_e the
    comparison can publish a phantom "active advantage" at p_e < 1 that
    is purely an artifact of an unweighted baseline.

    For p_e = 1 the formula reduces to its previous form (the leading
    factor is unity), so existing worst-case-prior tests remain unchanged.
    """
    return (
        p_e * passive_reset_residual(T1, tau_meas + tau_gate)
        - reset_residual_single_cycle(p_e, joint, gate_error)
    )
