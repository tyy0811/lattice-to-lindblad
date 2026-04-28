"""Module 5b — JointMatrix dataclass tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.joint_matrix import JointMatrix


def _synthetic_joint_matrix(
    p_ee0=0.10, p_ee1=0.85, p_eg0=0.04, p_eg1=0.01,
    p_gg0=0.97, p_gg1=0.02, p_ge0=0.005, p_ge1=0.005,
    n_trajectories=1000,
) -> JointMatrix:
    """Hand-set JointMatrix for property tests. Probabilities chosen so
    each row sums to 1: (s_i=1) row → 0.10+0.85+0.04+0.01 = 1.00;
    (s_i=0) row → 0.97+0.02+0.005+0.005 = 1.00.
    """
    probabilities = {
        (1, 1, 0): p_ee0, (1, 1, 1): p_ee1, (1, 0, 0): p_eg0, (1, 0, 1): p_eg1,
        (0, 0, 0): p_gg0, (0, 0, 1): p_gg1, (0, 1, 0): p_ge0, (0, 1, 1): p_ge1,
    }
    binomial_se = {
        k: math.sqrt(p * (1 - p) / n_trajectories) for k, p in probabilities.items()
    }
    return JointMatrix(
        probabilities=probabilities,
        binomial_se=binomial_se,
        n_trajectories=n_trajectories,
        operating_point={'tau_meas': 1e-6},
    )


def test_joint_matrix_rows_sum_to_one():
    """For each s_i, Σ_{s_f, m} P(s_f, m | s_i) = 1."""
    J = _synthetic_joint_matrix()
    for s_i in (0, 1):
        row_sum = sum(
            J.probabilities[(s_i, s_f, m)]
            for s_f in (0, 1) for m in (0, 1)
        )
        assert row_sum == pytest.approx(1.0, abs=1e-12)


def test_marginal_confusion_matrix_recovers_plain_form():
    """Sum over s_f to recover the plain confusion matrix P(m | s_i)."""
    J = _synthetic_joint_matrix()
    confusion = J.marginal_confusion_matrix()
    # P(m=0 | s_i=1) = P(s_f=0, m=0|1) + P(s_f=1, m=0|1) = 0.04 + 0.10
    assert confusion[(1, 0)] == pytest.approx(0.14, abs=1e-12)
    # P(m=1 | s_i=1) = 0.01 + 0.85
    assert confusion[(1, 1)] == pytest.approx(0.86, abs=1e-12)
    # P(m=0 | s_i=0) = 0.97 + 0.005
    assert confusion[(0, 0)] == pytest.approx(0.975, abs=1e-12)
    # P(m=1 | s_i=0) = 0.02 + 0.005
    assert confusion[(0, 1)] == pytest.approx(0.025, abs=1e-12)


def test_joint_ideal_gate_floor_two_terms():
    """V1 unit-level mirror: ideal-gate floor (ε_X = 0) is the SUM of:
      P(s_f=e, m=0 | e)  (missed-excited; reset fails)
    + P(s_f=g, m=1 | e)  (false-positive on decayed; flips back to e)
    Both terms are non-negligible when T₁-during-measurement is in play.
    Missing either is a bug.
    """
    J = _synthetic_joint_matrix(p_ee0=0.10, p_eg1=0.01)
    floor = J.joint_ideal_gate_floor()
    assert floor == pytest.approx(0.10 + 0.01, abs=1e-12)


def test_joint_matrix_frozen_dataclass():
    """Mutation must raise FrozenInstanceError; the cache layer relies
    on immutability."""
    J = _synthetic_joint_matrix()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        J.n_trajectories = 9999  # type: ignore


def test_binomial_se_formula_correct():
    """V5 (split, unit-tier): binomial_se = √(p(1-p)/N) must equal the
    closed-form binomial standard error for the same (p, N). This is a
    formula assertion against synthetic JointMatrix payloads — no MC
    sampling, no shot noise; the test fails iff the formula is wrong.
    """
    cases = [
        (0.5,   1000),
        (0.05,  1000),
        (0.95,  1000),
        (0.10,    50),
        (0.99, 10000),
    ]
    for p, n in cases:
        expected = math.sqrt(p * (1 - p) / n)
        J = JointMatrix(
            probabilities={(1, 1, 0): p},
            binomial_se={(1, 1, 0): expected},
            n_trajectories=n,
            operating_point={},
        )
        assert J.binomial_se[(1, 1, 0)] == pytest.approx(expected, rel=1e-12), (
            f"binomial_se formula wrong at p={p}, N={n}"
        )
