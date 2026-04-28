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


# ---------------------------------------------------------------------------
# Day 2.2 — load_eps_x_5a with provenance + mtime
# ---------------------------------------------------------------------------

from dispersive_readout.control.reset_protocol import load_eps_x_5a


SYNTHETIC_FIG5A_YAML = textwrap.dedent("""
device: REFERENCE_DEVICE
calibration_objective: argmin_β (1 − F_avg)
shipped_metric: epsilon_x = 1 - F_avg
alpha_2pi_Hz: -234199032.4133016
sweep_T_gate_ns: [5.0, 10.0, 15.0, 20.0, 25.0]
beta_opt_fidelity: [0.65, 0.55, 0.50, 0.50, 0.50]
epsilon_x_drag_opt: [3.5e-3, 1.8e-3, 1.0e-3, 8.12e-4, 7.5e-4]
F_avg_drag_opt: [0.9965, 0.9982, 0.999, 0.999188, 0.99925]
""")


def test_load_eps_x_5a_provenance_capture(tmp_path):
    """Returns (eps_x, provenance) with the four+ required keys."""
    yaml_file = tmp_path / "fig5a_drag_leakage_data.yaml"
    yaml_file.write_text(SYNTHETIC_FIG5A_YAML)

    eps_x, provenance = load_eps_x_5a(t_gate=20e-9, yaml_path=yaml_file)

    assert eps_x == pytest.approx(8.12e-4)
    assert 'source_yaml' in provenance
    assert 'source_mtime' in provenance
    assert 'T_gate_ns' in provenance
    assert provenance['T_gate_ns'] == 20.0
    assert 'beta_opt' in provenance
    assert provenance['beta_opt'] == 0.50
    assert 'F_avg_drag_opt' in provenance
    assert provenance['F_avg_drag_opt'] == pytest.approx(1 - 8.12e-4)


def test_load_eps_x_5a_provenance_mtime_matches_yaml_file(tmp_path):
    """Guards the staleness-detection capability: provenance mtime must
    equal yaml_path.stat().st_mtime at load time, so any future re-render
    of fig5a's YAML is detectable as a mtime advance in fig5b's data YAML.
    """
    yaml_file = tmp_path / "fig5a_drag_leakage_data.yaml"
    yaml_file.write_text(SYNTHETIC_FIG5A_YAML)
    expected_mtime = yaml_file.stat().st_mtime

    _, provenance = load_eps_x_5a(t_gate=20e-9, yaml_path=yaml_file)
    assert provenance['source_mtime'] == expected_mtime


def test_load_eps_x_5a_raises_on_t_gate_not_in_sweep(tmp_path):
    """If T_gate isn't in the YAML's sweep grid, raise ValueError with
    the available grid in the message."""
    yaml_file = tmp_path / "fig5a_drag_leakage_data.yaml"
    yaml_file.write_text(SYNTHETIC_FIG5A_YAML)
    with pytest.raises(ValueError, match="not in 5a's sweep"):
        load_eps_x_5a(t_gate=12e-9, yaml_path=yaml_file)


def test_load_eps_x_5a_raises_on_missing_yaml(tmp_path):
    """Clear FileNotFoundError if the YAML doesn't exist."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="5a data YAML not found"):
        load_eps_x_5a(yaml_path=missing)
