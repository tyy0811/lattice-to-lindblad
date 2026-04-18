"""Lindblad collapse-operator and Hamiltonian-builder tests."""
from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.lindblad import (
    build_collapse_operators,
    build_hamiltonian,
)

_TWO_PI = 2.0 * math.pi


def test_collapse_operators_returned_as_qobj_list():
    d = REFERENCE_DEVICE
    c_ops = build_collapse_operators(d, d.truncation.N_transmon, d.truncation.N_resonator)
    assert isinstance(c_ops, list)
    for op in c_ops:
        assert isinstance(op, qt.Qobj)


def test_collapse_operator_shapes_match_full_hilbert_space():
    d = REFERENCE_DEVICE
    total_dim = d.truncation.N_transmon * d.truncation.N_resonator
    c_ops = build_collapse_operators(d, d.truncation.N_transmon, d.truncation.N_resonator)
    for op in c_ops:
        assert op.shape == (total_dim, total_dim)


def test_collapse_list_has_expected_channel_count():
    """Reference device (n_th > 0) builds:
       2 resonator ops (decay + heating)
       + 2*(Nq-1) qubit transitions (relaxation + thermal heating)
       + Nq dephasing ops (one per level, L_j = sqrt(γ_φ) |j><j|)
       + (Nq-1) Purcell decay ops (added in Task 15 refactor since the
         dispersive-frame Hamiltonian has transverse coupling transformed
         out, so Purcell must be wired in explicitly).

    Catches accidental omission of any channel. Quantitative rate correctness
    is validated end-to-end by the V3 / V4 physics tests.
    """
    d = REFERENCE_DEVICE
    Nq = d.truncation.N_transmon
    c_ops = build_collapse_operators(d, Nq, d.truncation.N_resonator)
    expected = 2 + 2 * (Nq - 1) + Nq + (Nq - 1)
    assert len(c_ops) == expected, f"expected {expected} collapse ops, got {len(c_ops)}"


def test_collapse_list_reduces_when_thermal_zero():
    """When n_th = 0 and n_th_r = 0, thermal-excitation operators must be omitted."""
    from dataclasses import replace
    d = REFERENCE_DEVICE
    d_cold = replace(d, decoherence=replace(d.decoherence, n_th=0.0))
    c_cold = build_collapse_operators(
        d_cold, d_cold.truncation.N_transmon, d_cold.truncation.N_resonator
    )
    c_warm = build_collapse_operators(
        d, d.truncation.N_transmon, d.truncation.N_resonator
    )
    assert len(c_cold) < len(c_warm)


def test_build_hamiltonian_returns_drift_and_drive_spec():
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 5e6, duration=500e-9, detuning=0.0)
    H0, drive_spec = build_hamiltonian(d, drv, frame="rotating")
    assert isinstance(H0, qt.Qobj)
    assert H0.isherm
    # QuTiP-compatible H(t) form: [op, callable]
    assert isinstance(drive_spec, list) and len(drive_spec) == 2
    op, func = drive_spec
    assert isinstance(op, qt.Qobj)
    assert callable(func)
    # Drive envelope at t=0 should be ~0 (rising edge not yet reached)
    eps0 = func(0.0, {})
    assert abs(eps0) < drv.amplitude * 0.1


def test_drive_envelope_peaks_near_midpulse():
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 5e6, duration=500e-9, detuning=0.0)
    _, drive_spec = build_hamiltonian(d, drv, frame="rotating")
    _, func = drive_spec
    mid = drv.duration / 2.0
    # Midpulse should be within 1% of full amplitude
    assert abs(func(mid, {}) - drv.amplitude) < 0.01 * drv.amplitude


def test_purcell_disabled_removes_purcell_collapse_operators():
    """Setting purcell_enabled=False must omit the Purcell channel operators."""
    from dispersive_readout.physics.config import (
        DecoherenceParams, DeviceConfig, REFERENCE_DEVICE
    )
    from dispersive_readout.physics.lindblad import build_collapse_operators

    tr = REFERENCE_DEVICE.truncation
    Nq, Nr = tr.N_transmon, tr.N_resonator

    device_on = REFERENCE_DEVICE  # purcell_enabled=True by default
    device_off = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=DecoherenceParams(
            gamma_1=REFERENCE_DEVICE.decoherence.gamma_1,
            gamma_phi=REFERENCE_DEVICE.decoherence.gamma_phi,
            n_th=REFERENCE_DEVICE.decoherence.n_th,
            purcell_enabled=False,
        ),
        truncation=tr,
    )

    c_ops_on = build_collapse_operators(device_on, Nq, Nr)
    c_ops_off = build_collapse_operators(device_off, Nq, Nr)

    # Purcell adds Nq-1 operators (|j> -> |j-1> for j=1..Nq-1)
    assert len(c_ops_on) - len(c_ops_off) == Nq - 1
