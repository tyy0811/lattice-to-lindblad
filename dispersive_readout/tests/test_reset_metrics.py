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
    """active_advantage = p_e · passive(τ_meas + τ_gate) − active(p_e, J, ε_X).
    Positive: active beats passive. Negative: passive dominates.

    Worst-case prior (p_e = 1): the leading p_e factor is unity and the
    formula reduces to the v0 baseline.
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


def test_active_advantage_scales_passive_baseline_by_prior():
    """Adversarial-review fix: at p_e < 1, the passive baseline is the
    prior-weighted residual p_e · exp(-τ/T₁) (since (1-p_e) ground-state
    weight contributes 0 in v0's zero-temperature regime). Without this
    leading p_e factor, an unweighted baseline could publish a phantom
    "active advantage" at p_e < 1 that is purely an artifact.

    Worst-case sentinel: at p_e = 0 the prior-aware advantage must be
    NON-positive (passive baseline = 0; active residual = (1-p_e)·branch_g
    ≥ 0 from the ground-prep false-positive entries).
    """
    J = _synthetic_J()
    T1 = 5e-6
    tau_meas = 1e-6
    tau_gate = 20e-9
    eps_x = 0.0

    # p_e = 0.5: passive should be 0.5 · exp(-τ/T₁), not exp(-τ/T₁).
    p_e = 0.5
    expected_passive = p_e * math.exp(-(tau_meas + tau_gate) / T1)
    # reset_residual_single_cycle at p_e=0.5, ε_X=0:
    # 0.5 · (P(1,1,0) + P(1,0,1)) + 0.5 · (P(0,1,0) + P(0,0,1))
    # = 0.5 · (0.10 + 0.01) + 0.5 · (0.005 + 0.02)
    expected_active = 0.5 * (0.10 + 0.01) + 0.5 * (0.005 + 0.02)
    expected_advantage = expected_passive - expected_active

    actual = active_advantage(
        p_e=p_e, joint=J, gate_error=eps_x,
        T1=T1, tau_meas=tau_meas, tau_gate=tau_gate,
    )
    assert actual == pytest.approx(expected_advantage, rel=1e-12)

    # p_e = 0: passive baseline is exactly 0; active is (1-p_e)·branch_g
    # = branch_g ≥ 0; so advantage ≤ 0. The test catches the bug from the
    # adversarial review where unweighted passive would have given a
    # spuriously large positive advantage even with p_e = 0.
    advantage_at_zero_prior = active_advantage(
        p_e=0.0, joint=J, gate_error=eps_x,
        T1=T1, tau_meas=tau_meas, tau_gate=tau_gate,
    )
    assert advantage_at_zero_prior <= 0.0
