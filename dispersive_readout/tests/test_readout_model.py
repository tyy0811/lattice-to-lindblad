"""Readout-model integration tests (dynamics + IQ separation + assignment fidelity)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.readout_model import (
    AssignmentFidelityResult,
    ReadoutResult,
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)

_TWO_PI = 2.0 * math.pi


def _default_drive() -> DriveParams:
    return DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)


def test_simulate_readout_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    t_list = np.linspace(0.0, 500e-9, 101)
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0, t_list=t_list)
    assert isinstance(res, ReadoutResult)
    assert res.t.shape == (101,)
    assert res.a_expectation.shape == (101,)
    assert res.photon_number.shape == (101,)
    assert res.qubit_populations.shape == (101, d.truncation.N_transmon)
    assert res.drive_envelope.shape == (101,)


def test_simulate_readout_photon_number_is_nonnegative():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0)
    assert np.all(res.photon_number >= -1e-10)


def test_simulate_readout_populations_sum_to_unity():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=1)
    totals = res.qubit_populations.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-3)


def test_simulate_readout_iq_trajectories_separate_for_0_and_1():
    """The steady-state ⟨a⟩ for |0> and |1> must differ by a measurable amount."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    res0 = simulate_readout(d, drv, initial_qubit_state=0)
    res1 = simulate_readout(d, drv, initial_qubit_state=1)
    # Compare mean ⟨a⟩ over the last 20% of the window (after the rise transient)
    tail0 = res0.a_expectation[int(0.8 * len(res0.a_expectation)):]
    tail1 = res1.a_expectation[int(0.8 * len(res1.a_expectation)):]
    sep = abs(tail0.mean() - tail1.mean())
    assert sep > 0.05, f"IQ separation {sep:.4f} too small — dispersive regime lost?"
