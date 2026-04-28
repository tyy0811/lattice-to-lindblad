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


# ---------------------------------------------------------------------------
# Day 2.3 — purcell_rate_1_to_0
# ---------------------------------------------------------------------------

from dispersive_readout.control.reset_protocol import purcell_rate_1_to_0
from dispersive_readout.physics.transmon import (
    charge_operator_matrix_elements,
    diagonalize_transmon,
)


def test_purcell_rate_1_to_0_matches_lindblad_formula():
    """Single source of truth: purcell_rate_1_to_0 must produce the same
    γ_P_{1→0} value that physics.lindblad.build_collapse_operators uses
    when constructing the |0⟩⟨1| collapse operator. If these diverge,
    5b and Module 1 are simulating different physics.
    """
    device = REFERENCE_DEVICE
    rate_helper = purcell_rate_1_to_0(device)

    # Recompute the formula from lindblad.py's build_collapse_operators body
    # at j=1: γ_P = (g·|n_{0,1}| / Δ_{1,0})²·κ·(1 + n_th)
    energies, eigenstates = diagonalize_transmon(
        device.transmon, device.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, device.truncation)
    delta_10 = energies[1] - energies[0] - device.resonator.omega_r
    n_elem_01 = abs(n_mat[0, 1])
    rate_reference = (
        (device.coupling.g * n_elem_01 / delta_10) ** 2
        * device.resonator.kappa
        * (1.0 + device.decoherence.n_th)
    )
    assert rate_helper == pytest.approx(rate_reference, rel=1e-12)


# ---------------------------------------------------------------------------
# Day 2.4 — extract_joint_matrix direct-jump sampler
# ---------------------------------------------------------------------------

from dataclasses import replace as dc_replace

from dispersive_readout.control.reset_protocol import extract_joint_matrix
from dispersive_readout.physics.config import DecoherenceParams


class _ZeroNoiseRNG:
    """RNG wrapper that returns 0.0 from standard_normal but otherwise
    delegates to the wrapped numpy rng. Used to deterministically zero
    out shot noise in extract_joint_matrix's state-label bookkeeping
    test. Implements `spawn` and `exponential` (the only other rng
    methods extract_joint_matrix calls).
    """

    def __init__(self, base_rng: np.random.Generator) -> None:
        self._base = base_rng

    def standard_normal(self, *args, **kwargs):
        if args:
            shape = args[0]
            return np.zeros(shape if isinstance(shape, tuple) else (shape,))
        return 0.0

    def exponential(self, *args, **kwargs):
        return self._base.exponential(*args, **kwargs)

    def spawn(self, n_children):
        return [_ZeroNoiseRNG(child) for child in self._base.spawn(n_children)]


def test_extract_joint_matrix_state_label_bookkeeping(tmp_path):
    """State-label bookkeeping in direct-jump.

    Recipe for deterministic verification:
      - γ_eff = 0 (no jumps possible) → s_f always == s_i
      - σ_iq · standard_normal() = 0 (zero-noise rng) → m classified
        deterministically: noisy_iq == centroid_g for s_i=0,
        noisy_iq == centroid_e for s_i=1.

    Joint matrix should have only TWO non-zero entries:
      P(s_f=0, m=0 | s_i=0) = 1   (g-prep stays in g; iq lands at α_∞_g)
      P(s_f=1, m=1 | s_i=1) = 1   (e-prep stays in e; iq lands at α_∞_e)

    All other 6 entries must be exactly 0.

    Note: the more direct strategy of "make κ huge so σ → 0" doesn't
    work — |α_∞| ∝ 1/κ collapses faster than σ shrinks (σ scales as
    1/√κ), so the SNR actually decreases. Zeroing the rng's
    standard_normal output is the principled fix.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)

    # Force γ_eff = 0 by zeroing γ_1 and disabling Purcell
    no_decay = DecoherenceParams(
        gamma_1=0.0, gamma_phi=device.decoherence.gamma_phi,
        n_th=device.decoherence.n_th, purcell_enabled=False,
    )
    device_no_decay = dc_replace(device, decoherence=no_decay)

    drive = closed_loop_demo_drive_params(duration=500e-9)
    rng = _ZeroNoiseRNG(np.random.default_rng(seed=42))
    J = extract_joint_matrix(device_no_decay, drive, n_trajectories=100, rng=rng)

    # Two non-zero entries expected
    assert J.probabilities[(0, 0, 0)] == pytest.approx(1.0, abs=1e-12)
    assert J.probabilities[(1, 1, 1)] == pytest.approx(1.0, abs=1e-12)

    # All others zero
    for s_i, s_f, m in [
        (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0),
    ]:
        assert J.probabilities[(s_i, s_f, m)] == pytest.approx(0.0, abs=1e-12)


def test_decay_during_measurement_creates_P_ge0_in_sweep(tmp_path):
    """V7 (blocking, integration-tier): sweep τ_meas/T₁ ∈ [0.1, 2.0] and
    require P(s_f=g, m=0 | e) > 0.05 + 2·SE at SOME point in the sweep.

    This demonstrates the conceptual finding: in a regime where T₁
    relaxation during measurement is non-negligible, the joint matrix
    distinguishes 'qubit decayed AND measurement correctly read ground'
    (reset succeeds) from 'qubit stayed excited AND measurement missed
    it' (reset fails). The plain confusion matrix conflates these.

    SE-aware threshold: at N=1000 and p≈0.05, SE ≈ 0.007, so the robust
    bound is > 0.05 + 2·0.007 ≈ 0.064.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    T1 = 1.0 / device.decoherence.gamma_1

    tau_meas_grid = T1 * np.linspace(0.1, 2.0, 8)
    rng = np.random.default_rng(seed=20260428)
    rng_subs = rng.spawn(len(tau_meas_grid))

    p_decayed_missed_curve = []
    p_decayed_missed_se_curve = []
    for tau, sub_rng in zip(tau_meas_grid, rng_subs):
        drive = closed_loop_demo_drive_params(duration=tau)
        J = extract_joint_matrix(device, drive, n_trajectories=1000, rng=sub_rng)
        p_decayed_missed_curve.append(J.probabilities[(1, 0, 0)])
        p_decayed_missed_se_curve.append(J.binomial_se[(1, 0, 0)])

    p_decayed_missed = np.array(p_decayed_missed_curve)
    p_decayed_missed_se = np.array(p_decayed_missed_se_curve)
    se_aware_threshold = 0.05 + 2.0 * p_decayed_missed_se
    margin_above_threshold = p_decayed_missed - se_aware_threshold

    assert margin_above_threshold.max() > 0, (
        f"V7 failed: P(s_f=g, m=0 | e) never exceeded 0.05 + 2·SE in the sweep. "
        f"Max value: {p_decayed_missed.max():.4f}, "
        f"SE at that point: {p_decayed_missed_se[np.argmax(p_decayed_missed)]:.4f}. "
        f"This is the regime characterization that V7 demonstrates."
    )


def test_no_mcsolve_in_reset_protocol():
    """Lint-grade enforcement: v0 has no mcsolve import / call.

    Looks at code lines only (strips comments + docstrings). Mentions of
    mcsolve in module docstrings (which document why v0 excludes it) are
    intentional and must not trigger the lint.
    """
    import ast
    from dispersive_readout.control import reset_protocol
    src = open(reset_protocol.__file__).read()
    tree = ast.parse(src)
    # Strip top-level + nested docstrings: collect them and check the rest
    docstrings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.append(doc)
    src_no_docstrings = src
    for doc in docstrings:
        src_no_docstrings = src_no_docstrings.replace(doc, '')
    # Strip comments
    src_no_comments = '\n'.join(
        line for line in src_no_docstrings.splitlines()
        if not line.lstrip().startswith('#')
    )
    assert 'mcsolve' not in src_no_comments, (
        "v0 reset_protocol must not import or call mcsolve "
        "(found mcsolve outside docstrings/comments)"
    )
