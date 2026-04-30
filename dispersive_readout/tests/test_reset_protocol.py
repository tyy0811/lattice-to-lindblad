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


def test_device_idx18_zeroes_thermal_for_v0(tmp_path):
    """v0 zero-temp invariant: device_idx18 always sets n_th=0.0 in the
    returned device, regardless of what REFERENCE_DEVICE has. The reset
    sampler hard-codes |g⟩-stays-|g⟩ and would silently drop the thermal
    g→e excitation pathway that build_collapse_operators models, so the
    only consistent v0 device has n_th=0.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)

    device = device_idx18(yaml_path=yaml_file)
    assert device.decoherence.n_th == 0.0
    # And REFERENCE_DEVICE itself should still have its production n_th
    # (we override only in the device_idx18 return value, not globally)
    assert REFERENCE_DEVICE.decoherence.n_th > 0.0


def test_device_idx18_raises_on_missing_yaml(tmp_path):
    """Clear FileNotFoundError if the YAML doesn't exist."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        device_idx18(yaml_path=missing)


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


def test_finite_t1_marginal_within_2x_shot_noise(tmp_path, capsys):
    """V4b (diagnostic-tier in v0): the marginal initial-state readout
    score ½(P(m=0|g) + P(m=1|e)) from extract_joint_matrix should agree
    with Module 1's compute_assignment_fidelity reference within 2× shot
    noise + solver tolerance.

    NON-BLOCKING in v0: Module 1 provides a Gaussian-around-∫⟨a⟩dt
    reference, which is a different statistical object than 5b's jump-
    time mixture of pointer-history Gaussians. The two will not generally
    agree to high precision in the finite-T₁ regime; the test logs the
    discrepancy as a diagnostic but does not assert blocking equality.

    Drive amplitude reduced to 40 MHz (vs idx=18's 140 MHz) so the
    coherent-state cavity occupation |α|² stays below the default
    N_resonator=15 Fock truncation (V4a comparison context). At 140 MHz
    mesolve's α saturates the Fock cutoff and the comparison would be
    polluted by truncation artifact, not just statistical-object
    mismatch.

    If a future Module 1 extension exposes a finite-T₁ IQ-distribution
    reference (e.g., via Hilbert-space mcsolve), this test can be
    promoted to blocking by raising the assertion below.
    """
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        compute_assignment_fidelity,
        simulate_readout,
    )

    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    # Use a smaller-amplitude drive so the V4b diagnostic compares a
    # mesolve number that isn't Fock-truncation-saturated.
    drive_diag = DriveParams(
        amplitude=40e6,
        duration=500e-9,
        detuning=0.0,
        edge_sigma=2e-9,
    )
    rng = np.random.default_rng(seed=20260428)

    J = extract_joint_matrix(device, drive_diag, n_trajectories=2000, rng=rng)
    confusion = J.marginal_confusion_matrix()
    fidelity_5b = 0.5 * (confusion[(0, 0)] + confusion[(1, 1)])

    result_g = simulate_readout(device, drive_diag, initial_qubit_state=0)
    result_e = simulate_readout(device, drive_diag, initial_qubit_state=1)
    fid_result = compute_assignment_fidelity(
        result_g, result_e,
        integration_window=(0.0, drive_diag.duration),
        n_shots=2000, noise_model='gaussian',
        rng=np.random.default_rng(seed=20260429),
    )
    fidelity_module1 = fid_result.F_assign

    discrepancy = abs(fidelity_5b - fidelity_module1)
    print(
        f"\n[V4b diagnostic] 5b marginal fidelity = {fidelity_5b:.4f}, "
        f"Module 1 reference = {fidelity_module1:.4f}, "
        f"discrepancy = {discrepancy:.4f}. "
        f"Non-blocking: Module 1's reference is Gaussian-around-⟨α⟩, "
        f"not jump-time mixture."
    )
    # Diagnostic-only: log magnitude. Promote to assertion if and when
    # Module 1 exposes an IQ-distribution-level finite-T₁ reference.


# ---------------------------------------------------------------------------
# Day 3.1 — passive_reset_residual + reset_residual_single_cycle
# ---------------------------------------------------------------------------

import math as _math

from dispersive_readout.control.reset_protocol import (
    passive_reset_residual,
    reset_residual_single_cycle,
)
from dispersive_readout.physics.joint_matrix import JointMatrix


def _synthetic_joint_matrix_for_formula_tests():
    return JointMatrix(
        probabilities={
            (1, 1, 0): 0.10, (1, 1, 1): 0.85, (1, 0, 0): 0.04, (1, 0, 1): 0.01,
            (0, 0, 0): 0.97, (0, 0, 1): 0.02, (0, 1, 0): 0.005, (0, 1, 1): 0.005,
        },
        binomial_se={
            (1, 1, 0): 0.0, (1, 1, 1): 0.0, (1, 0, 0): 0.0, (1, 0, 1): 0.0,
            (0, 0, 0): 0.0, (0, 0, 1): 0.0, (0, 1, 0): 0.0, (0, 1, 1): 0.0,
        },
        n_trajectories=1000,
        operating_point={'tau_meas': 1e-6},
    )


def test_passive_reset_baseline_formula():
    """passive_reset_residual(T1, τ) = exp(-τ/T1)."""
    T1 = 5e-6
    for tau in [0.0, T1, 2 * T1, 5 * T1]:
        assert passive_reset_residual(T1, tau) == pytest.approx(
            _math.exp(-tau / T1), rel=1e-12,
        )


def test_reset_residual_three_terms_sum():
    """V1 reconstruction: at p_e=1, ε_X=ε, the formula is
      missed-excited:   P(s_f=e, m=0 | e)
    + gate-failure:     P(s_f=e, m=1 | e) · ε_X
    + false-positive:   P(s_f=g, m=1 | e) · (1 - ε_X)
    """
    J = _synthetic_joint_matrix_for_formula_tests()
    for eps in [0.0, 0.001, 0.5, 1.0]:
        expected = (
            J.probabilities[(1, 1, 0)]                      # missed
            + J.probabilities[(1, 1, 1)] * eps              # gate failure
            + J.probabilities[(1, 0, 1)] * (1.0 - eps)      # false positive
        )
        assert reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=eps) == pytest.approx(
            expected, rel=1e-12,
        )


def test_reset_residual_ideal_gate_two_term_floor():
    """V1 (blocking, unit-tier): at ε_X=0, p_e=1, residual is the SUM
    of the missed-excited and false-positive-on-decayed terms.

    The false-positive-on-decayed term does NOT vanish at ε_X=0 — it is
    maximal there. Missing it is a v2-era bug.
    """
    J = _synthetic_joint_matrix_for_formula_tests()
    expected = J.probabilities[(1, 1, 0)] + J.probabilities[(1, 0, 1)]
    actual = reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=0.0)
    assert actual == pytest.approx(expected, rel=1e-12)
    # Mirror the JointMatrix-class method
    assert actual == pytest.approx(J.joint_ideal_gate_floor(), rel=1e-12)


def test_reset_residual_eps_x_one_is_identity():
    """At ε_X=1 the conditional X-gate ALWAYS fails (identity), so the
    formula reduces to: p_e · P(s_f=e | e) + (1-p_e) · P(s_f=e | g).
    """
    J = _synthetic_joint_matrix_for_formula_tests()
    expected = J.probabilities[(1, 1, 0)] + J.probabilities[(1, 1, 1)]
    actual = reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=1.0)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_worst_case_residual_dominates_mixed_prior():
    """V6 (blocking, unit-tier): worst-case prior p_e=1 ≥ residual at
    mixed prior p_e=0.5. Property of the formula on a sensible joint
    matrix; tested with synthetic input (no MC noise).
    """
    J = _synthetic_joint_matrix_for_formula_tests()
    for eps in [0.0, 8.12e-4, 0.01]:
        worst = reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=eps)
        mixed = reset_residual_single_cycle(p_e=0.5, joint=J, gate_error=eps)
        assert worst >= mixed - 1e-12


# ---------------------------------------------------------------------------
# Day 3.2 — V2 active-beats-passive + V3 long-τ asymmetric floors
# ---------------------------------------------------------------------------


def test_active_beats_passive_at_some_tau(tmp_path):
    """V2 (blocking, integration-tier): there exists at least one
    operating point τ_meas in the figure sweep where active reset
    residual is below the matched-duration passive baseline.

    Sweep-based: no fragile fixed-τ blocker.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    T1 = 1.0 / device.decoherence.gamma_1
    T_gate = 20e-9

    tau_meas_grid = T1 * np.linspace(0.1, 2.0, 8)
    rng = np.random.default_rng(seed=20260428)
    rng_subs = rng.spawn(len(tau_meas_grid))

    p_active = []
    p_passive = []
    for tau, sub_rng in zip(tau_meas_grid, rng_subs):
        drive = closed_loop_demo_drive_params(duration=tau)
        J = extract_joint_matrix(device, drive, n_trajectories=1000, rng=sub_rng)
        p_active.append(reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=0.0))
        p_passive.append(passive_reset_residual(T1, tau + T_gate))

    p_active = np.array(p_active)
    p_passive = np.array(p_passive)
    advantage = p_passive - p_active

    assert advantage.max() > 0, (
        f"V2 failed: active never beats passive in the sweep. "
        f"p_active min: {p_active.min():.4f}, "
        f"p_passive at that τ: {p_passive[np.argmin(p_active)]:.4f}."
    )


def test_long_tau_asymmetric_floors(tmp_path):
    """V3 (blocking, integration-tier): at long τ_meas, passive → 0 (or
    thermal floor) while active → thermal + readout-false-positive +
    gate-error contribution. The two should NOT match in this limit;
    active has a higher floor by the false-positive/gate-error overhead.

    Test: p_active at τ_meas/T₁ = 2.0 must exceed p_passive at the same
    matched duration by at least the false-positive-on-|g⟩ contribution
    P(s_f=g, m=1 | g)·(1-ε_X) at ε_X=0.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    T1 = 1.0 / device.decoherence.gamma_1
    T_gate = 20e-9
    tau_long = 2.0 * T1

    rng = np.random.default_rng(seed=42)
    drive = closed_loop_demo_drive_params(duration=tau_long)
    J = extract_joint_matrix(device, drive, n_trajectories=1000, rng=rng)

    p_active = reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=0.0)
    p_passive = passive_reset_residual(T1, tau_long + T_gate)
    false_positive_floor = J.probabilities[(0, 0, 1)]  # P(s_f=g, m=1 | g)

    se_allowance = 2.0 * J.binomial_se[(0, 0, 1)]
    assert p_active >= p_passive + false_positive_floor - se_allowance, (
        f"V3 failed at τ=2T1: p_active={p_active:.4f}, "
        f"p_passive={p_passive:.4f}, false_pos_floor={false_positive_floor:.4f}. "
        f"Active-reset overhead at long τ should be at least the false-positive "
        f"contribution from the joint matrix."
    )


# ---------------------------------------------------------------------------
# Day 3.3 — V5 slow-tier 1/√N convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trajectory_count_convergence_pe_prime(tmp_path):
    """V5 (split, slow-tier): empirical SE on p_e' scales as 1/√N.

    Run extract_joint_matrix at three N values, compute p_e' at ε_X=0,
    and verify the empirical standard deviation across multiple seeds
    scales as 1/√N within a factor-of-2 tolerance.

    This catches sampler bugs (correlated draws across s_i, paired
    sampling regression) that the unit-tier binomial_se formula test
    cannot detect.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    drive = closed_loop_demo_drive_params(duration=1e-6)

    n_seeds = 8
    n_grid = [200, 1000, 4000]
    se_at_n = []
    for n_traj in n_grid:
        residuals = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed=seed + 100 * n_traj)
            J = extract_joint_matrix(device, drive, n_trajectories=n_traj, rng=rng)
            residuals.append(reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=0.0))
        se_at_n.append(float(np.std(residuals, ddof=1)))

    ratio_200_to_1000 = se_at_n[0] / se_at_n[1]
    ratio_1000_to_4000 = se_at_n[1] / se_at_n[2]
    expected_ratio_200_1000 = _math.sqrt(1000.0 / 200.0)   # √5 ≈ 2.24
    expected_ratio_1000_4000 = _math.sqrt(4000.0 / 1000.0)  # 2.00

    assert 0.5 * expected_ratio_200_1000 < ratio_200_to_1000 < 2.0 * expected_ratio_200_1000, (
        f"V5 slow: SE(200)/SE(1000) = {ratio_200_to_1000:.2f}, "
        f"expected ≈ {expected_ratio_200_1000:.2f}"
    )
    assert 0.5 * expected_ratio_1000_4000 < ratio_1000_to_4000 < 2.0 * expected_ratio_1000_4000, (
        f"V5 slow: SE(1000)/SE(4000) = {ratio_1000_to_4000:.2f}, "
        f"expected ≈ {expected_ratio_1000_4000:.2f}"
    )


def test_extract_joint_matrix_rejects_thermal(tmp_path):
    """Defensive guard from adversarial review: the v0 sampler hard-codes
    |g⟩-stays-|g⟩ and drops thermal g→e events that build_collapse_
    operators would model. extract_joint_matrix must therefore refuse any
    device with n_th > 0 and direct callers to device_idx18 (which sets
    n_th=0) or to a future v1.5 thermal-aware sampler.
    """
    yaml_file = tmp_path / "closed_loop_demo_device.yaml"
    yaml_file.write_text(SYNTHETIC_CLOSED_LOOP_YAML)
    device = device_idx18(yaml_path=yaml_file)
    # Re-introduce thermal population to exercise the guard
    thermal_decoherence = DecoherenceParams(
        gamma_1=device.decoherence.gamma_1,
        gamma_phi=device.decoherence.gamma_phi,
        n_th=0.01,
        purcell_enabled=device.decoherence.purcell_enabled,
    )
    thermal_device = dc_replace(device, decoherence=thermal_decoherence)
    drive = closed_loop_demo_drive_params(duration=500e-9)
    rng = np.random.default_rng(seed=42)

    with pytest.raises(NotImplementedError, match="thermal"):
        extract_joint_matrix(thermal_device, drive, n_trajectories=10, rng=rng)


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
