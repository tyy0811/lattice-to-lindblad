"""Module 5b — reset_metrics tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.analysis.reset_metrics import active_advantage
from dispersive_readout.physics.joint_matrix import JointMatrix


def _synthetic_J():
    return JointMatrix(
        probabilities={
            (1, 1, 0): 0.10, (1, 1, 1): 0.85, (1, 0, 0): 0.04, (1, 0, 1): 0.01,
            (0, 0, 0): 0.97, (0, 0, 1): 0.02, (0, 1, 0): 0.005, (0, 1, 1): 0.005,
        },
        binomial_se={k: 0.0 for k in [
            (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1),
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        ]},
        n_trajectories=1000,
        operating_point={'tau_meas': 1e-6},
    )


def test_active_advantage_at_matched_duration():
    """active_advantage = passive(τ_meas + τ_gate) − active(p_e, J, ε_X).
    Positive: active beats passive. Negative: passive dominates.
    """
    J = _synthetic_J()
    T1 = 5e-6
    tau_meas = 1e-6
    tau_gate = 20e-9
    eps_x = 0.0

    expected_passive = math.exp(-(tau_meas + tau_gate) / T1)
    # active = P(1,1,0) + P(1,0,1) at ε_X=0, p_e=1
    expected_active = 0.10 + 0.01
    expected_advantage = expected_passive - expected_active

    actual = active_advantage(
        p_e=1.0, joint=J, gate_error=eps_x,
        T1=T1, tau_meas=tau_meas, tau_gate=tau_gate,
    )
    assert actual == pytest.approx(expected_advantage, rel=1e-12)
