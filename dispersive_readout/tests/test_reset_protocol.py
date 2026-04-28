"""Module 5b — direct-jump joint transition-readout active reset tests."""
from __future__ import annotations

import numpy as np
import pytest

from dispersive_readout.control.reset_protocol import QubitStateHistory


def test_qubit_state_history_validates_monotonicity():
    """t_start values must be strictly monotonically increasing."""
    # Valid: monotonic
    QubitStateHistory(segments=((0.0, 1), (5e-7, 0)), t_total=1e-6)

    # Invalid: t_start values equal
    with pytest.raises(ValueError, match="monotonic"):
        QubitStateHistory(segments=((0.0, 1), (0.0, 0)), t_total=1e-6)

    # Invalid: t_start values decreasing
    with pytest.raises(ValueError, match="monotonic"):
        QubitStateHistory(segments=((0.0, 1), (5e-7, 0), (3e-7, 1)), t_total=1e-6)


def test_qubit_state_history_rejects_nonzero_start():
    """First segment must start at t=0.0."""
    with pytest.raises(ValueError, match="must start at 0"):
        QubitStateHistory(segments=((1e-9, 0),), t_total=1e-6)


def test_qubit_state_history_rejects_segment_past_t_total():
    """All t_start values must be < t_total."""
    with pytest.raises(ValueError, match="t_total"):
        QubitStateHistory(segments=((0.0, 1), (2e-6, 0)), t_total=1e-6)


def test_qubit_state_history_rejects_invalid_qubit_state():
    """qubit_state ∈ {0, 1} (v0 has no thermal; no |2⟩+ states)."""
    with pytest.raises(ValueError, match="qubit_state"):
        QubitStateHistory(segments=((0.0, 2),), t_total=1e-6)


def test_qubit_state_history_frozen():
    """Mutation should raise FrozenInstanceError."""
    h = QubitStateHistory(segments=((0.0, 1),), t_total=1e-6)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        h.t_total = 2e-6  # type: ignore


# ---------------------------------------------------------------------------
# Day 2.1 — closed_loop_demo_drive_params + device_idx18
# ---------------------------------------------------------------------------

import textwrap
from pathlib import Path

from dispersive_readout.control.reset_protocol import (
    closed_loop_demo_drive_params,
    device_idx18,
)
from dispersive_readout.physics.config import REFERENCE_DEVICE


SYNTHETIC_CLOSED_LOOP_YAML = textwrap.dedent("""
chosen: &id001
  index: 18
  T_1_us: 5.352036822392957
  T_2_echo_us: 6.548245787821845
  omega_q_GHz: 4.7223593505964505
  epsilon_0_opt: 140000000.0
  tau_opt_ns: 500.0
  F_assign_opt: 0.9923550115430083
  dominant_loss_channel: T1_intrinsic
  selection_criterion: min_F_shared_argmax_regime
""")


def test_closed_loop_demo_drive_params_eps_drive_fixed():
    """ε_drive = 140 MHz from the closed-loop YAML, regardless of the
    duration argument. Sanity guard against a sloppy refactor that updates
    eps_drive when it shouldn't.
    """
    drive_a = closed_loop_demo_drive_params(duration=500e-9)
    drive_b = closed_loop_demo_drive_params(duration=2e-6)
    assert drive_a.amplitude == 140e6
    assert drive_b.amplitude == drive_a.amplitude
    assert drive_a.duration == 500e-9
    assert drive_b.duration == 2e-6


def test_device_idx18_matches_yaml(tmp_path):
    """device_idx18 with synthetic YAML returns a DeviceConfig with
    idx=18's overrides applied to REFERENCE_DEVICE base."""
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)

    device = device_idx18(yaml_path=yaml_file)

    expected_gamma_1 = 1.0 / 5.352036822392957e-6
    assert device.decoherence.gamma_1 == pytest.approx(expected_gamma_1, rel=1e-9)

    assert device.resonator.kappa == REFERENCE_DEVICE.resonator.kappa
    assert device.decoherence.n_th == REFERENCE_DEVICE.decoherence.n_th


def test_device_idx18_raises_on_missing_yaml(tmp_path):
    """Clear FileNotFoundError if the YAML doesn't exist."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        device_idx18(yaml_path=missing)


def test_device_idx18_raises_on_high_thermal(tmp_path, monkeypatch):
    """v0 enforces n̄_q < 0.05 in device_idx18; raise NotImplementedError
    otherwise (v1.5 thermal-excitation territory)."""
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)

    from dispersive_readout.physics.config import (
        DecoherenceParams,
        DeviceConfig,
    )
    high_thermal = DecoherenceParams(
        gamma_1=REFERENCE_DEVICE.decoherence.gamma_1,
        gamma_phi=REFERENCE_DEVICE.decoherence.gamma_phi,
        n_th=0.10,  # above 0.05 threshold
    )
    high_thermal_ref = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=high_thermal,
        truncation=REFERENCE_DEVICE.truncation,
    )
    monkeypatch.setattr(
        "dispersive_readout.control.reset_protocol.REFERENCE_DEVICE",
        high_thermal_ref,
    )
    with pytest.raises(NotImplementedError, match="thermal"):
        device_idx18(yaml_path=yaml_file)
