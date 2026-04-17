"""Config dataclass and REFERENCE_DEVICE tests."""
from __future__ import annotations

import dataclasses
import math

import pytest

from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    CouplingParams,
    DecoherenceParams,
    DeviceConfig,
    DriveParams,
    ResonatorParams,
    TransmonParams,
    TruncationParams,
)

_TWO_PI = 2.0 * math.pi


def test_all_config_dataclasses_are_frozen():
    """Accidental mutation of config inside simulation code must be blocked."""
    for cls in (
        TransmonParams,
        ResonatorParams,
        CouplingParams,
        DecoherenceParams,
        DriveParams,
        TruncationParams,
        DeviceConfig,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} is not frozen"


def test_transmon_unit_conversion():
    """E_C_Hz and E_J_Hz are the rad/s values divided by 2π."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9)
    assert p.E_C_Hz == pytest.approx(210e6, rel=1e-12)
    assert p.E_J_Hz == pytest.approx(15.5e9, rel=1e-12)


def test_reference_device_matches_spec():
    """Reference device encodes Marxer 2508.16437 anchor values.

    Tolerance 1% allows the implementer to derive kappa/gamma from T-times etc.,
    but the top-level numbers must match the spec.
    """
    d = REFERENCE_DEVICE
    assert d.transmon.E_C_Hz == pytest.approx(210e6, rel=0.01)
    assert d.transmon.E_J_Hz == pytest.approx(15.5e9, rel=0.01)
    assert (d.resonator.omega_r / _TWO_PI) == pytest.approx(7.3e9, rel=0.01)
    assert (d.resonator.kappa / _TWO_PI) == pytest.approx(5e6, rel=0.01)
    assert (d.coupling.g / _TWO_PI) == pytest.approx(120e6, rel=0.01)
    # T1 = 30 us → γ1 = 1/T1
    assert d.decoherence.gamma_1 == pytest.approx(1.0 / 30e-6, rel=0.01)
    assert d.decoherence.n_th == pytest.approx(0.01, rel=0.01)


def test_truncation_defaults_match_spec():
    t = TruncationParams()
    assert t.N_charge == 13
    assert t.N_transmon == 5
    assert t.N_resonator == 15
    # N_charge must be odd so the charge ladder is symmetric about zero
    assert t.N_charge % 2 == 1
