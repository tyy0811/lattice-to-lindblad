"""Readout-model integration tests (dynamics + IQ separation + assignment fidelity).

All tests here drive a full Lindblad master-equation integration via QuTiP
mesolve. Marked @pytest.mark.slow so the fast TDD suite can skip them via
`pytest -m "not slow"`. Run on-demand with `pytest -m slow` or the default
`pytest`.
"""
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

pytestmark = pytest.mark.slow


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


def test_assignment_fidelity_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    assert isinstance(f, AssignmentFidelityResult)
    assert 0.0 <= f.F_assign <= 1.0
    assert f.separation_distance > 0.0
    assert f.snr > 0.0


def test_assignment_fidelity_ideal_is_at_least_as_large_as_gaussian():
    """With no shot noise, fidelity is bounded above by the 'gaussian' noise case."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f_g = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    f_i = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="ideal")
    assert f_i.F_assign >= f_g.F_assign - 1e-9


def test_assignment_fidelity_sanity_on_reference_device():
    """Reference device should hit ≥ 95% assignment fidelity at a drive
    amplitude that stays within the dispersive regime.

    Uses ε/2π = 10 MHz and a 450 ns integration window (50–500 ns). The
    SNR scales as ε × √T; at these parameters the physical SNR formula
    SNR² = 4κ |Δα|² T gives ≈ 3.9, which yields F ≳ 0.97 via the
    perpendicular-bisector discriminator. Mean intracavity photon number
    stays ~6 — below the N_resonator=15 warning threshold and well below
    the critical n_crit = (Δ/2g)² ≈ 100 for the dispersive approximation.

    The _default_drive (ε/2π = 2 MHz) is kept low for other tests so
    photons stay well under the Fock cutoff and those tests run fast.
    """
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 10e6, duration=500e-9, detuning=0.0)
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (50e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=20000, noise_model="gaussian")
    assert f.F_assign >= 0.95, f"Reference device fidelity {f.F_assign:.4f} below 0.95 — flag to human."


def test_snr_vs_integration_time_shape_and_monotone_rise():
    """SNR should rise roughly as sqrt(t) over short integrations and plateau."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    t_int = np.linspace(50e-9, 450e-9, 9)
    snr = snr_vs_integration_time(d, drv, t_int)
    assert snr.shape == (9,)
    # Monotone rise before plateau: first half must be non-decreasing (tolerating noise)
    early = snr[: len(snr) // 2]
    assert np.all(np.diff(early) >= -0.05), f"SNR not rising: {early}"
    # Final SNR should exceed the first SNR
    assert snr[-1] > snr[0]
