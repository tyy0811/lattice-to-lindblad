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
    """passive_reset_residual(T1, tau_meas + tau_gate)
       − reset_residual_single_cycle(p_e, joint, gate_error).

    Positive value: active reset beats passive at this matched duration.
    Negative value: passive baseline dominates.

    The matched duration includes the gate time τ_gate (5a's headline
    20 ns) so the comparison is fair: both protocols consume the same
    wall-clock time before the next operation.
    """
    return (
        passive_reset_residual(T1, tau_meas + tau_gate)
        - reset_residual_single_cycle(p_e, joint, gate_error)
    )
