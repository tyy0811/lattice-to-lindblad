"""Stage 06 Module 4 — optimization layer tests (O1–O24).

Test catalog per MODULE_4_SPEC.md §6.1. Convention: each test function's
docstring cites the spec test ID it implements.
"""
from __future__ import annotations


def test_policy_constants_present_and_frozen():
    """Policy constants must live in source with locked values (Q6 lock)."""
    from dispersive_readout.optimization.sensitivity import (
        SENSITIVITY_FD_STEP,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
        SENSITIVITY_WARNING_THRESHOLD,
    )
    assert SENSITIVITY_FD_STEP == 0.05, (
        f"SENSITIVITY_FD_STEP changed from spec-locked 0.05 to {SENSITIVITY_FD_STEP}; "
        "requires spec amendment"
    )
    assert SENSITIVITY_RENDER_BAR_THRESHOLD == 0.03, (
        f"SENSITIVITY_RENDER_BAR_THRESHOLD changed from spec-locked 0.03 "
        f"to {SENSITIVITY_RENDER_BAR_THRESHOLD}; requires spec amendment"
    )
    assert SENSITIVITY_WARNING_THRESHOLD == 0.3, (
        f"SENSITIVITY_WARNING_THRESHOLD changed from amended value 0.3 "
        f"to {SENSITIVITY_WARNING_THRESHOLD}; requires spec amendment. "
        "See docs/module4_diagnostics/sensitivity_ceiling_characterization.md "
        "for the three-check verification that justified the 2.0 → 0.3 amendment."
    )


import re
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
# O6.1 — SensitivityResult schema validation (spec §6 test catalog)
# ────────────────────────────────────────────────────────────────────

def test_O6_1_sensitivity_result_accepts_valid():
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    r = SensitivityResult(
        parameter="chi_scale",
        reference_value=1.0,
        reference_unit="dimensionless",
        sensitivity=0.42,
        sensitivity_uncertainty=0.01,
        F_reference=0.99,
    )
    assert r.parameter == "chi_scale"
    assert r.step_size_used == 0.05  # default = SENSITIVITY_FD_STEP


def test_O6_1_sensitivity_result_rejects_negative_uncertainty():
    from pydantic import ValidationError
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    with pytest.raises(ValidationError, match="sensitivity_uncertainty"):
        SensitivityResult(
            parameter="kappa",
            reference_value=1e7,
            reference_unit="rad/s",
            sensitivity=-0.2,
            sensitivity_uncertainty=-0.001,  # invalid
            F_reference=0.99,
        )


def test_O6_1_sensitivity_result_rejects_unknown_parameter_name():
    from pydantic import ValidationError
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    with pytest.raises(ValidationError):
        SensitivityResult(
            parameter="not_a_real_parameter",  # not in ParameterName Literal
            reference_value=1.0,
            reference_unit="",
            sensitivity=0.1,
            sensitivity_uncertainty=0.01,
            F_reference=0.99,
        )


def test_O6_1_sensitivity_result_noise_consistent_flag_matches_threshold():
    from dispersive_readout.optimization.sensitivity import (
        SensitivityResult,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
    )
    just_below = SENSITIVITY_RENDER_BAR_THRESHOLD * 0.9
    r = SensitivityResult(
        parameter="n_th",
        reference_value=0.01,
        reference_unit="",
        sensitivity=just_below,
        sensitivity_uncertainty=1e-4,
        F_reference=0.99,
    )
    # Schema should auto-compute or the computed-flag helper should match threshold
    assert r.noise_consistent_with_zero is True, (
        f"|S|={just_below} < {SENSITIVITY_RENDER_BAR_THRESHOLD} should flag "
        "noise_consistent_with_zero=True"
    )


# ────────────────────────────────────────────────────────────────────
# O8 — analytic-objective-contract enforcement (Q8 lock)
# ────────────────────────────────────────────────────────────────────
#
# Strengthened per Module-4 execution finding: Module 1 ships three
# noise_model values: 'ideal' (F=1, zero-noise limit), 'analytic'
# (F = Φ(SNR/2), ensemble-mean F under the gaussian noise model in the
# continuous-shot limit), and 'gaussian' (empirical shot-sampled F).
# Sensitivity and Pareto inner loops must use 'analytic' exclusively:
#   - 'gaussian' pollutes FD gradients and SLSQP f-evals with shot noise.
#   - 'ideal' saturates to F=1.0, so log F = 0 and sensitivity is 0.
# The test asserts both negatives (forbidden modes absent) and positive
# ('analytic' present at least once).

_OPTIMIZATION_DIR = Path("dispersive_readout") / "optimization"
_FORBIDDEN_GAUSSIAN = re.compile(r"""noise_model\s*=\s*["']gaussian["']""")
_FORBIDDEN_IDEAL = re.compile(r"""noise_model\s*=\s*["']ideal["']""")
_REQUIRED_ANALYTIC = re.compile(r"""noise_model\s*=\s*["']analytic["']""")


def test_O8_no_gaussian_noise_inside_sensitivity_module():
    """Q8 lock: sensitivity.py must never use noise_model='gaussian' inside
    its inner loops — FD gradients become unreliable under shot noise."""
    src = (_OPTIMIZATION_DIR / "sensitivity.py").read_text()
    matches = _FORBIDDEN_GAUSSIAN.findall(src)
    assert matches == [], (
        f"Q8 contract violated: sensitivity.py contains "
        f"noise_model='gaussian' at {len(matches)} call site(s). Inner-loop "
        "F-evaluations must use noise_model='analytic'. See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_no_ideal_noise_inside_sensitivity_module():
    """Q8 lock (strengthened): sensitivity.py must never use noise_model='ideal'
    either — it returns F=1.0 whenever centroids differ, so log F = 0 and the
    finite-difference sensitivity is identically zero. Use 'analytic'."""
    src = (_OPTIMIZATION_DIR / "sensitivity.py").read_text()
    matches = _FORBIDDEN_IDEAL.findall(src)
    assert matches == [], (
        f"Q8 contract violated: sensitivity.py contains "
        f"noise_model='ideal' at {len(matches)} call site(s). Ideal mode "
        "is the zero-noise (F=1) limit — inner loops need 'analytic' (F=Φ(SNR/2)). "
        "See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_analytic_mode_present_in_sensitivity_module():
    """Q8 lock (positive assertion): sensitivity.py must invoke Module 1's
    noise_model='analytic' at least once. Catches the regression where
    someone removes the kwarg entirely and picks up Module 1's default
    ('gaussian')."""
    src = (_OPTIMIZATION_DIR / "sensitivity.py").read_text()
    matches = _REQUIRED_ANALYTIC.findall(src)
    assert len(matches) >= 1, (
        "Q8 contract violated: sensitivity.py must call "
        "compute_assignment_fidelity(..., noise_model='analytic') at least once. "
        "A missing kwarg picks up Module 1's default ('gaussian'), which pollutes "
        "FD gradients with shot noise. See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_no_gaussian_noise_inside_pareto_module():
    """Q8 lock: pareto.py must never use noise_model='gaussian' inside
    SLSQP function evaluations — optimizer noise pollutes FD gradients."""
    pareto_path = _OPTIMIZATION_DIR / "pareto.py"
    if not pareto_path.exists():
        pytest.skip("pareto.py not yet created — Task 12")
    src = pareto_path.read_text()
    matches = _FORBIDDEN_GAUSSIAN.findall(src)
    assert matches == [], (
        f"Q8 contract violated: pareto.py contains "
        f"noise_model='gaussian' at {len(matches)} call site(s). Inner-loop "
        "SLSQP evaluations must use noise_model='analytic'. See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_no_ideal_noise_inside_pareto_module():
    """Q8 lock (strengthened): pareto.py must never use noise_model='ideal'."""
    pareto_path = _OPTIMIZATION_DIR / "pareto.py"
    if not pareto_path.exists():
        pytest.skip("pareto.py not yet created — Task 12")
    src = pareto_path.read_text()
    matches = _FORBIDDEN_IDEAL.findall(src)
    assert matches == [], (
        f"Q8 contract violated: pareto.py contains "
        f"noise_model='ideal' at {len(matches)} call site(s). "
        "Ideal mode is the zero-noise (F=1) limit — SLSQP needs 'analytic'. "
        "See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_analytic_mode_present_in_pareto_module():
    """Q8 lock (positive assertion): pareto.py must invoke noise_model='analytic'."""
    pareto_path = _OPTIMIZATION_DIR / "pareto.py"
    if not pareto_path.exists():
        pytest.skip("pareto.py not yet created — Task 12")
    src = pareto_path.read_text()
    if "Placeholder implementation so O10 smoke succeeds" in src:
        pytest.skip(
            "pareto.py is the Task-11 stub for Modal smoke; positive "
            "noise_model='analytic' assertion activates with Task-13 "
            "find_pareto_point SLSQP implementation."
        )
    matches = _REQUIRED_ANALYTIC.findall(src)
    assert len(matches) >= 1, (
        "Q8 contract violated: pareto.py must call "
        "compute_assignment_fidelity(..., noise_model='analytic') at least once."
    )


# Item-11 amendment scope-extension: regime_map.py's validate_analytic_vs_lindblad
# also runs inner-loop simulate_readout + compute_assignment_fidelity calls that
# qualify under the same Q8 contract. Task-10 execution caught the regression at
# the Lindblad-validation step (F_sim=1.0 exactly → 23–27% deviation vs analytic).
# The contract now scans regime_map.py too so the next regression fires statically.

def test_O8_no_gaussian_noise_inside_regime_map_module():
    """Q8 lock: regime_map.py inner-loop sims must not use noise_model='gaussian'."""
    src = (_OPTIMIZATION_DIR / "regime_map.py").read_text()
    matches = _FORBIDDEN_GAUSSIAN.findall(src)
    assert matches == [], (
        f"Q8 contract violated: regime_map.py contains "
        f"noise_model='gaussian' at {len(matches)} call site(s). Inner-loop "
        "Lindblad-validation evaluations must use noise_model='analytic'. "
        "See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_no_ideal_noise_inside_regime_map_module():
    """Q8 lock (item-11 scope-extension): regime_map.py must never use
    noise_model='ideal'. Caught at Task-10 execution: ideal mode returns F=1.0
    and the analytic-vs-Lindblad comparison sees 23–27% spurious deviation."""
    src = (_OPTIMIZATION_DIR / "regime_map.py").read_text()
    matches = _FORBIDDEN_IDEAL.findall(src)
    assert matches == [], (
        f"Q8 contract violated: regime_map.py contains "
        f"noise_model='ideal' at {len(matches)} call site(s). Ideal mode is the "
        "zero-noise (F=1) limit — Lindblad-validation needs 'analytic'. "
        "See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_analytic_mode_present_in_regime_map_module():
    """Q8 lock (positive assertion): regime_map.py must invoke noise_model='analytic'."""
    src = (_OPTIMIZATION_DIR / "regime_map.py").read_text()
    matches = _REQUIRED_ANALYTIC.findall(src)
    assert len(matches) >= 1, (
        "Q8 contract violated: regime_map.py must call "
        "compute_assignment_fidelity(..., noise_model='analytic') at least once."
    )


# ────────────────────────────────────────────────────────────────────
# O1a / O1b — sensitivity sign sanity (physics-falsifiable invariant)
#
# Split per Module 4 execution finding: under the SW-2 simulator, REFERENCE
# sits ~18% past the F_assign peak in χ-space, so S_χ at REFERENCE is
# slightly negative (-0.03 ± 0.014, noise-consistent at the 0.03 rendering
# threshold). Similarly, |S_{γ_1}| ≈ 5e-4 is below threshold because T_1 is
# ~3% of REFERENCE's error budget (Purcell dominates).
#
# O1a asserts signs for parameters whose |S| rises above
# SENSITIVITY_RENDER_BAR_THRESHOLD — i.e., parameters that would render as
# filled bars on the tornado. These are the parameters where the sign is
# visible in the publication figure and a wrong sign would be a
# publication-grade bug. Must-fire assertion, no relaxation.
#
# O1b logs near-zero parameters with their measured values to an artifact
# in test_output/ for manual review. No assertion; documents that χ, γ_1,
# n_th sit at the tornado-noise floor under this REFERENCE.
# ────────────────────────────────────────────────────────────────────

_EXPECTED_SIGNS_AT_REFERENCE: dict[str, int] = {
    # sign: +1 / -1. Read from physics:
    #   χ/ε₀/τ/κ ↑ → F ↑ (in the SNR-monotone regime)
    #   γ_1 / γ_φ / n̄_th ↑ → F ↓
    # Parameters at REFERENCE sitting near the F-peak (|S| below threshold)
    # are logged by O1b with no sign assertion.
    "chi_scale":  +1,
    "kappa":      +1,
    "epsilon_0":  +1,
    "tau":        +1,
    "gamma_1":    -1,
    "gamma_phi":  -1,
    "n_th":       -1,
}


@pytest.fixture(scope="module")
def _reference_all_sensitivities():
    """Shared-module fixture: compute all 7 sensitivities once so O1a and
    O1b don't each pay 7×3 = 21 Lindblad sims."""
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_log_sensitivity

    op = get_reference_operating_point(n_shots=10_000)
    results = [
        compute_log_sensitivity(op, p)
        for p in ("chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau")
    ]
    return results


def test_O1a_sensitivity_signs_for_bar_rendered_parameters(_reference_all_sensitivities):
    """Physics-falsifiable: for each parameter whose |S| ≥ rendering threshold
    (would show as a filled bar on the tornado), the sign must match the
    physics prediction. This is the assertion you cannot relax without
    papering over a publication-grade bug — a wrong sign here propagates
    directly into Figure 4 Panel (a).
    """
    from dispersive_readout.optimization.sensitivity import SENSITIVITY_RENDER_BAR_THRESHOLD

    results = _reference_all_sensitivities
    bar_rendered = [r for r in results if abs(r.sensitivity) >= SENSITIVITY_RENDER_BAR_THRESHOLD]
    assert len(bar_rendered) >= 1, (
        "Expected at least one parameter to render as a filled bar "
        f"(|S| ≥ {SENSITIVITY_RENDER_BAR_THRESHOLD}); got none. "
        "Tornado plot will be empty — check simulator state."
    )

    for r in bar_rendered:
        expected_sign = _EXPECTED_SIGNS_AT_REFERENCE[r.parameter]
        observed_sign = 1 if r.sensitivity > 0 else -1
        assert observed_sign == expected_sign, (
            f"S_{r.parameter} = {r.sensitivity:+.4f} has wrong sign "
            f"(expected {'>0' if expected_sign > 0 else '<0'}) and |S| is "
            f"above the rendering threshold {SENSITIVITY_RENDER_BAR_THRESHOLD}. "
            "Wrong sign → simulator or sensitivity code has a bug. "
            "DO NOT fix by flipping signs in the figure."
        )


def test_O1b_log_near_zero_sensitivities(_reference_all_sensitivities):
    """Near-zero parameters at REFERENCE (|S| < rendering threshold). No
    sign assertion — their sign is noise-consistent with zero by definition.
    Output is written to test_output/o1b_near_zero_sensitivities.txt for
    manual review alongside Figure 4 Panel (a) preparation.

    Purpose: at this REFERENCE, χ/γ_1/n̄_th are expected near-zero because
    REFERENCE sits at the F-peak in χ and T_1/thermal are ~3% of the error
    budget. If a FUTURE bug flips a near-zero sign into bar-rendered
    territory with the WRONG sign, O1a will fire — this test is purely for
    documentation of the measured landscape, not CI gating.
    """
    from pathlib import Path
    from dispersive_readout.optimization.sensitivity import SENSITIVITY_RENDER_BAR_THRESHOLD

    results = _reference_all_sensitivities
    near_zero = [r for r in results if abs(r.sensitivity) < SENSITIVITY_RENDER_BAR_THRESHOLD]

    out = Path("test_output") / "o1b_near_zero_sensitivities.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "O1b — near-zero sensitivities at REFERENCE (noise-consistent with zero).",
        f"Rendering threshold = {SENSITIVITY_RENDER_BAR_THRESHOLD}.",
        f"F_reference = {results[0].F_reference:.6f}",
        "",
        f"{'parameter':<12}  {'S':>10}  {'sigma(S)':>10}  {'|S|/sigma':>9}  note",
    ]
    for r in near_zero:
        ratio = abs(r.sensitivity) / r.sensitivity_uncertainty if r.sensitivity_uncertainty > 0 else float("inf")
        # S = 0.0 exactly (not "measured zero") indicates float-precision
        # underflow in (log F_plus - log F_minus) — the ΔF from the ±5%
        # perturbation fell below np.log precision at F ≈ 0.99. Annotate so
        # readers don't interpret 0.0 as "exactly measured zero."
        note = "(float underflow, see spec §3.1 note)" if r.sensitivity == 0.0 else ""
        lines.append(
            f"{r.parameter:<12}  {r.sensitivity:+10.5f}  "
            f"{r.sensitivity_uncertainty:10.5f}  {ratio:9.2f}  {note}"
        )
    lines.append("")
    lines.append(
        "Interpretation (spec §0 Q1-amended): at this REFERENCE, F_assign "
        "peaks at chi_scale ~ 0.85 under the SW-2 simulator; REFERENCE sits "
        "~18% past the peak so |S_chi| is near-zero. S_gamma_1 and S_n_th are "
        "near-zero because T_1 and thermal contribute <= 3% of REFERENCE's "
        "error budget (Purcell dominates, per Module 2 fig2_data.yaml). "
        "S_gamma_phi = 0.0 exactly is a float-precision underflow (not "
        "physics zero) — T_2_echo ~ 2*T_1 at REFERENCE puts gamma_phi at "
        "the pure-dephasing-free limit, and a 5% perturbation produces |ΔF| "
        "below np.log precision at F ~ 0.99. Spec-amended: pure dephasing "
        "is not a meaningful error-budget contributor at this REFERENCE."
    )
    out.write_text("\n".join(lines) + "\n")
    # No sign assertion — log-only.
    assert True


# ────────────────────────────────────────────────────────────────────
# O12–O18 — per-parameter unit checks (all 7 parameters)
# ────────────────────────────────────────────────────────────────────

_ALL_PARAMETERS = [
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
]


# ────────────────────────────────────────────────────────────────────
# Codex adversarial-review regression tests (Day-10 fixes)
# Finding #1: τ-window mismatch (high severity)
# Finding #2: zero-reference collapse (medium severity)
# ────────────────────────────────────────────────────────────────────

def test_tau_probe_integration_window_tracks_perturbed_tau():
    """Regression: for parameter='tau', the per-probe integration window
    must have its right edge equal to the perturbed drive.duration. The
    50 ns window start (κ-ramp-up exclusion) stays fixed. Closes the
    Codex high-severity finding where ±5% τ probes compared one pulse
    shorter than the window and one pulse longer than the window."""
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import (
        _perturbed_device_drive_window_scale,
    )

    op = get_reference_operating_point(n_shots=10_000)
    for delta in (-0.05, +0.05):
        _dev, drive, window, _chi = _perturbed_device_drive_window_scale(op, "tau", delta)
        expected_duration = op.drive.duration * (1.0 + delta)
        assert drive.duration == pytest.approx(expected_duration, rel=1e-12)
        assert window[1] == pytest.approx(drive.duration, rel=1e-12), (
            f"τ probe at δ={delta}: integration_window[1]={window[1]} does "
            f"not track perturbed drive.duration={drive.duration}. Window "
            "must co-perturb with τ per spec §0.1 amendment."
        )
        assert window[0] == op.integration_window[0], (
            f"τ probe at δ={delta}: window start shifted from "
            f"{op.integration_window[0]} to {window[0]}. κ-ramp-up "
            "exclusion should stay fixed."
        )


def test_fd_dispatcher_probe_configuration_self_consistent():
    """Regression: for every parameter, the dispatcher's returned triple
    (device, drive, integration_window) must be self-consistent —
    non-perturbed fields unchanged, perturbed fields aligned. Closes
    the class of bugs where an FD dispatcher silently couples a probed
    parameter to an un-probed field (Codex adversarial finding class)."""
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import (
        _perturbed_device_drive_window_scale,
    )

    op = get_reference_operating_point(n_shots=10_000)
    params = ("chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau")
    for p in params:
        for delta in (-0.05, +0.05):
            _dev, drive, window, _chi = _perturbed_device_drive_window_scale(op, p, delta)
            if p == "tau":
                # τ probe: duration and window end must match.
                assert drive.duration == pytest.approx(window[1], rel=1e-12), (
                    f"{p} probe at δ={delta}: drive.duration={drive.duration} "
                    f"!= window[1]={window[1]}"
                )
            else:
                # Non-τ probes must leave drive.duration AND window unchanged
                # from the reference operating point.
                assert drive.duration == op.drive.duration, (
                    f"{p} probe at δ={delta} inadvertently perturbed drive.duration"
                )
                assert window == op.integration_window, (
                    f"{p} probe at δ={delta} inadvertently perturbed integration_window"
                )


def test_compute_log_sensitivity_raises_at_zero_reference_value():
    """Regression: multiplicative perturbation θ·(1±h) collapses to 0 at
    both branches when θ=0, so central FD returns silent S=0.
    compute_log_sensitivity must raise ValueError with Koch-back-solve
    guidance. Closes Codex medium-severity finding #2."""
    from dataclasses import replace
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.analysis.operating_point import (
        get_reference_operating_point, OperatingPoint,
    )
    from dispersive_readout.optimization.sensitivity import compute_log_sensitivity

    ref_op = get_reference_operating_point(n_shots=10_000)
    zero_gamma_phi = replace(REFERENCE_DEVICE.decoherence, gamma_phi=0.0)
    dev_zero = replace(REFERENCE_DEVICE, decoherence=zero_gamma_phi)
    op_zero = OperatingPoint(
        device=dev_zero,
        drive=ref_op.drive,
        integration_window=ref_op.integration_window,
        n_shots=ref_op.n_shots,
    )

    with pytest.raises(ValueError, match=r"reference_value is exactly 0\.0"):
        compute_log_sensitivity(op_zero, "gamma_phi")


@pytest.mark.parametrize("parameter", _ALL_PARAMETERS)
def test_O12_O18_per_parameter_sensitivity_finite_and_typed(parameter):
    """Each of the 7 parameters returns a finite SensitivityResult at REFERENCE."""
    import math
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import (
        compute_log_sensitivity,
        SensitivityResult,
    )

    op = get_reference_operating_point(n_shots=10_000)
    r = compute_log_sensitivity(op, parameter)

    assert isinstance(r, SensitivityResult)
    assert r.parameter == parameter
    assert math.isfinite(r.sensitivity), f"S_{parameter} is not finite: {r.sensitivity}"
    assert math.isfinite(r.sensitivity_uncertainty)
    assert r.sensitivity_uncertainty > 0.0
    assert 0.0 < r.F_reference <= 1.0


# ────────────────────────────────────────────────────────────────────
# O2 — step-independence: S at h=0.05 vs h=0.025 within 10%
# ────────────────────────────────────────────────────────────────────

def test_O2_step_independence_epsilon_0():
    """S_epsilon_0 at h=0.05 and h=0.025 must agree to within 10%.

    Per spec §6.1 O2. Original plan targeted chi_scale but S_chi sits at
    the noise floor at REFERENCE (see O1b log + spec §0 Q1-amended). Use
    epsilon_0 as the step-independence anchor: it renders as a filled bar
    (|S| ≈ 0.05, ~3.5σ above zero) so the comparison tests FD-truncation
    error rather than shot-noise floor artifacts.
    """
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_log_sensitivity

    op = get_reference_operating_point(n_shots=10_000)
    s_coarse = compute_log_sensitivity(op, "epsilon_0", step_size=0.05)
    s_fine = compute_log_sensitivity(op, "epsilon_0", step_size=0.025)
    rel_diff = abs(s_fine.sensitivity - s_coarse.sensitivity) / abs(s_coarse.sensitivity)
    assert rel_diff < 0.10, (
        f"S_epsilon_0 at h=0.025 ({s_fine.sensitivity:.4f}) differs from h=0.05 "
        f"({s_coarse.sensitivity:.4f}) by {rel_diff*100:.1f}% (> 10%). "
        "Reduce Lindblad solver rtol, or investigate FD-truncation error."
    )


def test_compute_all_sensitivities_returns_seven():
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_all_sensitivities

    op = get_reference_operating_point(n_shots=10_000)
    results = compute_all_sensitivities(op)
    assert len(results) == 7
    params = {r.parameter for r in results}
    assert params == {
        "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
    }


def test_rank_sensitivities_sorts_by_absolute_magnitude_desc():
    from dispersive_readout.optimization.sensitivity import (
        SensitivityResult, rank_sensitivities,
    )
    inputs = [
        SensitivityResult(
            parameter="chi_scale", reference_value=1.0, reference_unit="",
            sensitivity=0.1, sensitivity_uncertainty=0.01, F_reference=0.99,
        ),
        SensitivityResult(
            parameter="gamma_1", reference_value=1e4, reference_unit="1/s",
            sensitivity=-0.5, sensitivity_uncertainty=0.02, F_reference=0.99,
        ),
        SensitivityResult(
            parameter="kappa", reference_value=3e7, reference_unit="rad/s",
            sensitivity=0.3, sensitivity_uncertainty=0.01, F_reference=0.99,
        ),
    ]
    ranked = rank_sensitivities(inputs)
    assert [r.parameter for r in ranked] == ["gamma_1", "kappa", "chi_scale"]


# ────────────────────────────────────────────────────────────────────
# O24 — Day-10 cross-check: S_g vs 2·S_chi (Q1 caption artifact)
# ────────────────────────────────────────────────────────────────────

def test_O24_day_10_cross_check_logged_and_within_threshold():
    """Compute S_chi via chi_scale and S_g via ±5% on coupling.g; write
    |S_g − 2·S_chi| / (2·|S_chi|) to day10_cross_check.txt for the
    Figure 4 caption. Test computes and logs; it does not gate on
    agreement (spec §9 item 2 — decision goes in caption, not fix)."""
    import math
    from pathlib import Path
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import day_10_cross_check_s_g_vs_s_chi

    op = get_reference_operating_point(n_shots=10_000)
    result = day_10_cross_check_s_g_vs_s_chi(op)

    # Assert structure
    for key in ("S_chi", "S_g", "predicted_S_g", "residual", "residual_fractional"):
        assert key in result, f"Missing key: {key}"
        assert math.isfinite(result[key])

    # Write artifact for Figure 4 caption
    artifact_path = Path("06_Dispersive_Readout/figures/day10_cross_check.txt")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        f"Day-10 Q1 cross-check at REFERENCE_DEVICE:\n"
        f"  S_chi (via chi_scale ± 0.05)   = {result['S_chi']:+.4f}\n"
        f"  S_g   (via coupling.g ± 0.05)  = {result['S_g']:+.4f}\n"
        f"  Predicted S_g = 2 · S_chi       = {result['predicted_S_g']:+.4f}\n"
        f"  Residual |S_g - 2*S_chi|        = {abs(result['residual']):.4f}\n"
        f"  Fractional |residual|/|2*S_chi| = {result['residual_fractional']*100:.2f}%\n"
        f"\n"
        f"Note (spec §0 Q1-amended): at this REFERENCE, S_chi sits at the\n"
        f"tornado noise floor (|S_chi| < 0.03), so the fractional residual\n"
        f"|S_g - 2*S_chi|/|2*S_chi| can be large without indicating Q1\n"
        f"orthogonality failure — the denominator is small. The caption\n"
        f"cites the residual verbatim alongside both raw numbers.\n"
    )

    # Structure-only assertion — test computes and logs, does not gate on
    # the residual magnitude per spec §9 item 2.
    assert 0.0 <= result["residual_fractional"]


# ────────────────────────────────────────────────────────────────────
# O11 — sensitivity_warnings fires on boundary-proximate device (Q4 lock)
# ────────────────────────────────────────────────────────────────────

def test_O11_sensitivity_warning_fires_at_high_drive_regime():
    """Device driven at ε/2π = 15 MHz (7.5× REFERENCE's ~2 MHz) should force
    |S_epsilon_0| > SENSITIVITY_WARNING_THRESHOLD and trigger the warning
    policy. Tests the threshold via a direct sensitivity computation; the
    full `sensitivity_warnings` assembly lands in Task 15's
    RecommendationReport.

    Amended from T_1-stress probe to drive-stress probe after Module 4
    execution finding: empirical |S| ceiling under the Lindblad simulator
    caps at ~0.4 across realistic parameter space. A drive-stress regime
    is a realistic operating-point choice a user might make (high drive
    to reduce τ_readout); the T_1-stress probe required hardware-extreme
    T_1 ≲ 0.2 µs devices to approach the amended threshold. See
    docs/module4_diagnostics/sensitivity_ceiling_characterization.md.
    """
    import math
    from dataclasses import replace
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.analysis.operating_point import (
        get_reference_operating_point, OperatingPoint,
    )
    from dispersive_readout.optimization.sensitivity import (
        compute_log_sensitivity,
        SENSITIVITY_WARNING_THRESHOLD,
    )

    ref_op = get_reference_operating_point(n_shots=10_000)
    high_drive = DriveParams(
        amplitude=2.0 * math.pi * 15.0e6,
        duration=ref_op.drive.duration,
        detuning=0.0,
    )
    op_high_drive = OperatingPoint(
        device=ref_op.device,
        drive=high_drive,
        integration_window=ref_op.integration_window,
        n_shots=ref_op.n_shots,
    )

    s_eps = compute_log_sensitivity(op_high_drive, "epsilon_0")
    assert abs(s_eps.sensitivity) > SENSITIVITY_WARNING_THRESHOLD, (
        f"High-drive regime (ε/2π=15 MHz) gave |S_epsilon_0|={abs(s_eps.sensitivity):.3f}, "
        f"expected > {SENSITIVITY_WARNING_THRESHOLD}. Either the threshold "
        f"drifted above the characterized ~0.4 empirical ceiling, or the "
        f"drive-stress regime isn't hitting the dominance level. Re-run "
        "docs/module4_diagnostics/check_*.py to re-verify the ceiling."
    )


# ────────────────────────────────────────────────────────────────────
# Decoherence-envelope linearization: (1 − γτ/2)^½ vs exp(−γτ/4)
# ────────────────────────────────────────────────────────────────────

def test_decoherence_envelope_linear_agrees_with_exp_within_1pct():
    """Linearized envelope (1 − γτ/2)^½ must agree with exp(−γτ/4) within
    1% over the regime map's y-axis range [1e-4, 1e-1]. Caption claims
    this explicitly — if it fails, add the correction term or re-linearize."""
    import numpy as np
    gamma_tau = np.logspace(-4, -1, 40)
    linear = np.sqrt(1.0 - gamma_tau / 2.0)
    expon = np.exp(-gamma_tau / 4.0)
    rel_dev = np.abs(linear - expon) / expon
    assert rel_dev.max() < 0.01, (
        f"Max relative deviation {rel_dev.max()*100:.2f}% > 1% at gamma_tau="
        f"{gamma_tau[rel_dev.argmax()]:.3e}. Caption claim 'deviation from "
        "exp form < 1% over y-axis range' is false — add correction term."
    )


# NOTE: Task-8's n_phot-monotone and chi-over-kappa-half-peak tests were
# removed in the item-15 amendment (Day 11 PM). They tested the textbook
# 2-level antisymmetric formula's parametrization (n_phot as a free input,
# universal peak at χ/κ=0.5). Under the per-level formula those properties
# don't hold: n_phot is not a free input (derived from ε, κ, χ_j); the peak
# location depends on REFERENCE per-level chi structure (device-specific).
# See docs/module4_diagnostics/per_level_analytic_derivation.md §6.


def test_f_analytic_dispersive_at_REFERENCE_anchor_matches_F_sim_within_1pct():
    """At REFERENCE chart coordinates, the per-level analytic F must match
    REFERENCE's F_sim ≈ 0.989 to within 1% absolute. Tight bound: REFERENCE
    is the natural anchor for the per-level formula, so agreement here should
    be ~0.02% (numerically observed). Using 1% gives safety margin for
    Module-1-side small numerical drift across versions."""
    from dispersive_readout.optimization.regime_map import (
        f_analytic_dispersive, _reference_chi_magnitude,
    )
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    chi_diff = _reference_chi_magnitude()
    chi_over_kappa = chi_diff / REFERENCE_DEVICE.resonator.kappa
    gamma_1_tau = REFERENCE_DEVICE.decoherence.gamma_1 * 5e-7  # REF drive duration
    F = float(f_analytic_dispersive(chi_over_kappa, gamma_1_tau))
    F_REF_sim = 0.9899  # session-frozen REFERENCE F_sim per Day-10 closeout
    assert abs(F - F_REF_sim) < 0.01, (
        f"F_analytic at REFERENCE anchor = {F:.4f}, expected ~{F_REF_sim:.4f} ± 0.01. "
        "Per-level formula should match REFERENCE F_sim tightly."
    )


def test_f_analytic_dispersive_per_level_chart_form_consistency():
    """The chart wrapper f_analytic_dispersive(χ/κ, γτ) must equal the
    workhorse f_analytic_dispersive_per_level when given REFERENCE-anchored
    inputs at the same chart coordinates."""
    from dispersive_readout.optimization.regime_map import (
        f_analytic_dispersive, f_analytic_dispersive_per_level,
        _reference_per_level_chi, _reference_drive_and_window, _reference_chi_magnitude,
    )
    chi_over_kappa = 0.5
    gamma_1_tau = 1e-3
    chi_0_ref, chi_1_ref = _reference_per_level_chi()
    chi_diff_ref = _reference_chi_magnitude()
    epsilon, T_window = _reference_drive_and_window()
    target_kappa = chi_diff_ref / chi_over_kappa

    F_chart = float(f_analytic_dispersive(chi_over_kappa, gamma_1_tau))
    F_per_level = float(f_analytic_dispersive_per_level(
        chi_0=chi_0_ref, chi_1=chi_1_ref, kappa=target_kappa,
        epsilon=epsilon, T_window=T_window, gamma_1_tau=gamma_1_tau,
    ))
    assert abs(F_chart - F_per_level) < 1e-12, (
        f"Chart wrapper {F_chart} ≠ per-level workhorse {F_per_level} "
        "at the same coordinates (REFERENCE-anchored)."
    )


# ────────────────────────────────────────────────────────────────────
# Published-device-points data validation (Q5 lock)
# ────────────────────────────────────────────────────────────────────

def test_PUBLISHED_DEVICE_POINTS_has_four_entries_labeled_correctly():
    """The 4 markers of the regime map — Marxer Q1, Marxer Q2, Bengtsson,
    Garnet — with Hazra OMITTED per Q5 lock."""
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    labels = [p.label for p in PUBLISHED_DEVICE_POINTS]
    assert len(PUBLISHED_DEVICE_POINTS) == 4, (
        f"Expected exactly 4 device points, got {len(PUBLISHED_DEVICE_POINTS)}. "
        "Hazra must be OMITTED per Q5 (dimon, non-standard transmon)."
    )
    assert all("Hazra" not in lab for lab in labels), (
        f"Hazra must not appear in plotted device points. Labels: {labels}"
    )
    expected_substrings = ["Marxer Q1", "Marxer Q2", "Bengtsson", "Garnet"]
    for expected in expected_substrings:
        assert any(expected in lab for lab in labels), (
            f"Missing expected device '{expected}' from PUBLISHED_DEVICE_POINTS. "
            f"Actual labels: {labels}"
        )


def test_PUBLISHED_DEVICE_POINTS_coordinates_are_physical():
    """chi/kappa and gamma_1*tau must be finite, positive, and within the
    regime map's x-axis [0.1, 10] and y-axis [1e-4, 1e-1] ranges."""
    import math
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    for p in PUBLISHED_DEVICE_POINTS:
        assert math.isfinite(p.chi_over_kappa) and p.chi_over_kappa > 0
        assert math.isfinite(p.gamma_1_tau) and p.gamma_1_tau > 0
        assert 0.1 <= p.chi_over_kappa <= 10.0, (
            f"{p.label}: chi_over_kappa={p.chi_over_kappa} outside [0.1, 10]"
        )
        assert 1e-4 <= p.gamma_1_tau <= 1e-1, (
            f"{p.label}: gamma_1_tau={p.gamma_1_tau} outside [1e-4, 1e-1]"
        )
        if p.reported_F_assign is not None:
            assert 0.5 <= p.reported_F_assign <= 1.0


def test_marxer_q1_is_primary_anchor_with_F_reported():
    """Marxer Q1 must have reported_F_assign set — it's the F_sim annotation
    anchor for Panel (b) per Q3 Refinement 1."""
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
    assert q1.reported_F_assign is not None
    assert q1.reported_F_assign > 0.99


# ────────────────────────────────────────────────────────────────────
# Analytic-boundary monotonicity tests
# ────────────────────────────────────────────────────────────────────

def test_purcell_boundary_decreases_with_chi_over_kappa():
    """Under γ_Purcell = κ · (g/Δ)² with (g, Δ) at REFERENCE and
    κ(x) = χ_REF / x, γ_Purcell ∝ 1/x, so τ_readout(x) at γ_P·τ=0.1 grows
    with x, and y_Purcell(x) = γ_1·0.1/γ_P(x) also grows with x.
    Boundary is monotone non-decreasing in x."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import purcell_boundary
    x = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    y = purcell_boundary(x)
    assert np.all(np.diff(y) >= 0), (
        f"Purcell boundary not monotone in x: y = {y}"
    )


def test_resonator_too_slow_is_constant_in_x():
    """kappa·tau_readout = 1 at fixed REFERENCE κ is a horizontal line."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import resonator_too_slow_boundary
    x = np.array([0.3, 1.0, 3.0])
    y = resonator_too_slow_boundary(x)
    assert np.allclose(y, y[0]), (
        f"Resonator-too-slow line not constant: {y}"
    )


# ────────────────────────────────────────────────────────────────────
# O3a / O3b / O3c — per-level analytic vs Lindblad at 3 points
# Per item-15 amendment (Day 11 PM): the closed-form analytic surface
# uses per-level dispersive shifts (χ_0, χ_1) — not the textbook
# antisymmetric ±χ/2 — and the integrated SNR carries the 2√(κ·T_window)
# factor that the old formula omitted. Validates against Module 1's
# Lindblad simulator at REFERENCE per-level anchor.
# Derivation: docs/module4_diagnostics/per_level_analytic_derivation.md.
# ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _validation_report():
    """Cache the 3-point Lindblad validation across O3a/O3b/O3c (~1 min total)."""
    from dispersive_readout.optimization.regime_map import validate_analytic_vs_lindblad
    return validate_analytic_vs_lindblad()


def test_O3a_per_level_analytic_vs_lindblad_at_marxer_q1(_validation_report):
    """Per-level analytic F at Marxer Q1's coordinates must agree with
    Module 1's Lindblad F_sim within 5% (REFERENCE is the Marxer Q1 anchor;
    expected agreement is much tighter, ~0.1%)."""
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    marxer_q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
    pt = next(
        p for p in _validation_report["per_point"]
        if abs(p["chi_over_kappa"] - marxer_q1.chi_over_kappa) < 1e-6
    )
    assert pt["deviation_fractional"] < 0.05, (
        f"O3a deviation {pt['deviation_fractional']*100:.2f}% > 5% at Marxer Q1 "
        f"({marxer_q1.chi_over_kappa:.2f}, {marxer_q1.gamma_1_tau:.2e}). "
        f"F_an={pt['F_analytic']:.4f}, F_sim={pt['F_lindblad']:.4f}. "
        "Per-level formula should match REFERENCE-anchor Marxer Q1 to <1% — "
        "if this fails, the per-level chi computation may have regressed."
    )


def test_O3b_per_level_analytic_vs_lindblad_at_midrange_point(_validation_report):
    """At (χ/κ=1.0, γ_1·τ=0.01), per-level analytic must agree within 5%.
    This is the most stringent of the three points (largest deviation under
    the linearized weak-drive approximation)."""
    pt = next(
        p for p in _validation_report["per_point"]
        if abs(p["chi_over_kappa"] - 1.0) < 1e-6
    )
    assert pt["deviation_fractional"] < 0.05, (
        f"O3b deviation {pt['deviation_fractional']*100:.2f}% > 5% at (1.0, 0.01). "
        f"F_an={pt['F_analytic']:.4f}, F_sim={pt['F_lindblad']:.4f}. "
        "Caption claim 'Lindblad-validated to <5%' fails."
    )


def test_O3c_per_level_analytic_vs_lindblad_at_weak_decoherence(_validation_report):
    """At (χ/κ=0.5, γ_1·τ=10⁻³), weak-decoherence near dispersive optimum,
    per-level analytic must agree within 5%."""
    pt = next(
        p for p in _validation_report["per_point"]
        if abs(p["chi_over_kappa"] - 0.5) < 1e-6
    )
    assert pt["deviation_fractional"] < 0.05, (
        f"O3c deviation {pt['deviation_fractional']*100:.2f}% > 5% at (0.5, 1e-3). "
        f"F_an={pt['F_analytic']:.4f}, F_sim={pt['F_lindblad']:.4f}."
    )


def test_O3_max_deviation_under_5pct(_validation_report):
    """Aggregate gate: max deviation across O3a/O3b/O3c must be <5% per the
    item-15 amendment caption claim. If this fires but individual O3a/b/c
    pass, the per-point gate has drifted."""
    max_dev = _validation_report["max_deviation_fractional"]
    assert max_dev < 0.05, (
        f"Max O3 deviation {max_dev*100:.2f}% > 5%. "
        "Per-level analytic surface no longer matches Lindblad to spec; "
        "re-derivation needed."
    )


# ────────────────────────────────────────────────────────────────────
# O10 — Modal image smoke test (Q2 pre-warm task)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O10_modal_pareto_one_tuple_smoke():
    """Pre-warm the Module 4 Modal image and dispatch one trivial
    pareto_one_tuple call via .map([one_tuple]). Confirms credentials,
    image build, and serialization work before Day 12's Pareto run."""
    import os

    if os.environ.get("SKIP_MODAL_TESTS") == "1":
        pytest.skip("SKIP_MODAL_TESTS=1 set — skip Modal smoke in CI")

    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.modal_pareto import (
        app, pareto_one_tuple,
    )
    from dispersive_readout.optimization.pareto import ParetoPoint

    # Modal's .map takes iterables; dispatch exactly one tuple.
    with app.run():
        results = list(pareto_one_tuple.map([REFERENCE_DEVICE], [500e-9]))

    assert len(results) == 1
    assert isinstance(results[0], ParetoPoint)


# ────────────────────────────────────────────────────────────────────
# O6.2 — ParetoPoint schema validation
# ────────────────────────────────────────────────────────────────────

def test_O6_2_pareto_point_accepts_valid():
    from dispersive_readout.optimization.pareto import ParetoPoint
    p = ParetoPoint(
        device_id="deadbeef",
        device_label="REFERENCE (≈ Marxer Q1)",
        tau_max=500e-9,
        epsilon_0_opt=5e7,
        tau_opt=480e-9,
        F_assign_opt=0.9984,
        F_assign_uncertainty=1.2e-3,
        dominant_loss_channel="T1_intrinsic",
        solver_converged=True,
    )
    assert p.tau_opt <= p.tau_max


def test_O6_2_pareto_point_rejects_tau_opt_exceeding_tau_max():
    from pydantic import ValidationError
    from dispersive_readout.optimization.pareto import ParetoPoint
    with pytest.raises(ValidationError):
        ParetoPoint(
            device_id="deadbeef",
            device_label="REFERENCE",
            tau_max=500e-9,
            epsilon_0_opt=5e7,
            tau_opt=520e-9,  # > tau_max, must reject
            F_assign_opt=0.99,
            F_assign_uncertainty=1e-3,
            dominant_loss_channel="T1_intrinsic",
            solver_converged=True,
        )


# ────────────────────────────────────────────────────────────────────
# O22 / O23 — bridge round-trip for V2 (T1=40us) and V3 (T1=20us, kappa=6MHz)
# ────────────────────────────────────────────────────────────────────

def test_O22_build_variant_v2_garnet_like():
    """V2 swaps decoherence.gamma_1 = 1/40us, leaves resonator and coupling
    at REFERENCE. gamma_phi recomputed via Koch back-solve."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import build_variant, PARETO_DEVICE_VARIANTS

    spec = next(v for v in PARETO_DEVICE_VARIANTS if v["T1_us"] == 40.0)
    variant = build_variant(spec)

    assert variant.decoherence.gamma_1 == pytest.approx(1.0 / 40e-6, rel=1e-9)
    assert variant.resonator.kappa == REFERENCE_DEVICE.resonator.kappa
    assert variant.coupling.g == REFERENCE_DEVICE.coupling.g
    # Koch back-solve for gamma_phi: gamma_phi = 1/T2_echo - gamma_1/2
    # T2_echo preserved at REFERENCE's value
    T2_echo_REF = 2.0 / (REFERENCE_DEVICE.decoherence.gamma_1 +
                         2.0 * REFERENCE_DEVICE.decoherence.gamma_phi)
    expected_gamma_phi = max(1.0 / T2_echo_REF - 0.5 * (1.0 / 40e-6), 0.0)
    assert variant.decoherence.gamma_phi == pytest.approx(expected_gamma_phi, rel=1e-9)


def test_O23_build_variant_v3_bengtsson_like():
    """V3 swaps T1=20us AND kappa/2pi=6MHz."""
    import math
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import build_variant, PARETO_DEVICE_VARIANTS

    spec = next(
        v for v in PARETO_DEVICE_VARIANTS
        if v["T1_us"] == 20.0 and v["kappa_MHz"] == 6.0
    )
    variant = build_variant(spec)

    assert variant.decoherence.gamma_1 == pytest.approx(1.0 / 20e-6, rel=1e-9)
    assert variant.resonator.kappa == pytest.approx(2.0 * math.pi * 6e6, rel=1e-9)
    assert variant.coupling.g == REFERENCE_DEVICE.coupling.g


# ────────────────────────────────────────────────────────────────────
# O19–O21 — Pareto edge cases
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O19_pareto_at_lower_tau_max_boundary_feasible():
    """τ_max = 100 ns must return a feasible ParetoPoint, possibly with
    lower F_opt than at larger τ_max but still > 0.5."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=100e-9)
    assert p.solver_converged
    assert p.F_assign_opt > 0.5
    assert p.tau_opt <= p.tau_max * 1.001


@pytest.mark.slow
def test_O20_pareto_at_upper_tau_max_boundary_feasible():
    """τ_max = 2 µs must return a feasible ParetoPoint. At this budget
    REFERENCE achieves F >> 0.99."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=2000e-9)
    assert p.solver_converged
    assert p.F_assign_opt > 0.99


@pytest.mark.slow
def test_O21_pareto_infeasibility_at_extreme_drive_bounds():
    """If ε_0 bounds exclude all F > 0.5, find_pareto_point returns
    solver_converged=False (or raises). Either signal is acceptable;
    test that failure is surfaced, not silent."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    # Extremely low drive amplitude bounds — F cannot exceed 0.5
    try:
        p = find_pareto_point(
            REFERENCE_DEVICE,
            tau_max=500e-9,
            epsilon_0_bounds=(1.0, 1e3),  # 1–1000 rad/s is absurdly low
        )
    except RuntimeError:
        return  # raised; also acceptable
    # Otherwise: solver must flag non-convergence OR low F
    assert (not p.solver_converged) or p.F_assign_opt < 0.6, (
        f"Infeasible regime produced converged={p.solver_converged} "
        f"F={p.F_assign_opt:.3f} — failure was not surfaced."
    )


# ────────────────────────────────────────────────────────────────────
# O4 — Pareto monotonicity in τ_max
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O4_pareto_monotonic_in_tau_max_for_reference():
    """F_opt non-decreasing along REFERENCE's Pareto curve.

    Relaxing τ_max cannot make F_opt worse; if it does, SLSQP is stuck
    at a local minimum — spec §9 item 4 says increase warm_start grid
    density before changing solvers."""
    import numpy as np
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import compute_pareto_frontier

    curve = compute_pareto_frontier(
        REFERENCE_DEVICE,
        tau_max_values=np.array([200e-9, 500e-9, 1000e-9, 2000e-9]),
        device_label="REFERENCE (test)",
    )
    F_opts = [p.F_assign_opt for p in curve]
    # Non-decreasing within 5σ_shot slack (shot-noise σ ~ 1e-3, 5σ ≈ 5e-3)
    for a, b in zip(F_opts, F_opts[1:]):
        assert b >= a - 5e-3, (
            f"F_opt decreased from {a:.4f} -> {b:.4f} across adjacent τ_max. "
            "Increase n_warm_start_grid_side from 10 to 20 and retry."
        )


# ────────────────────────────────────────────────────────────────────
# Amendment #10 regression guard — Day-13 warm-start grid bug
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_warm_start_resolves_basin_at_reference():
    """Pareto optimum at REFERENCE, tau_max=500ns must clear F > 0.99.

    Under the pre-fix 5-point linear warm-start, F_opt returned 0.929
    (stuck at eps_opt=2.5e8 warm-start winner; SLSQP could not cross
    the inter-grid valley to the true basin at eps ~ 1.59e8).
    Log-spaced 10-point warm-start resolves the basin: F_opt ~ 0.993.
    See docs/module4_diagnostics/warm_start_grid_bug.md."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=500e-9)
    assert p.F_assign_opt >= 0.99, (
        f"REFERENCE at tau_max=500ns returned F_opt={p.F_assign_opt:.4f} "
        f"(eps_opt={p.epsilon_0_opt:.3e}, tau_opt={p.tau_opt*1e9:.1f}ns). "
        "Expected F_opt >= 0.99. Under the pre-fix warm-start grid "
        "(5-point linear), F_opt was 0.929 — SLSQP trapped outside "
        "the true basin at eps ~ 1.59e8. If this regresses, the "
        "warm-start topology may have reverted to linear on eps."
    )
    # Sanity: solver landed at tau boundary (physically correct at 500ns
    # with REFERENCE decoherence — F monotone in tau up to ~T_phi).
    assert p.tau_opt >= 500e-9 * 0.999, (
        f"tau_opt={p.tau_opt*1e9:.1f} ns < tau_max boundary; "
        "either decoherence dominates earlier than expected or "
        "SLSQP is stuck."
    )


# ────────────────────────────────────────────────────────────────────
# O6.3 — RecommendationReport schema
# ────────────────────────────────────────────────────────────────────

def test_O6_3_recommendation_report_accepts_valid():
    from dispersive_readout.optimization.recommend import RecommendationReport
    from dispersive_readout.optimization.sensitivity import SensitivityResult

    s = SensitivityResult(
        parameter="chi_scale", reference_value=1.0, reference_unit="",
        sensitivity=0.4, sensitivity_uncertainty=0.01, F_reference=0.99,
    )
    r = RecommendationReport(
        device_parameters_fitted={
            "T_1": 86e-6, "T_2_echo": 40e-6,
            "omega_q": 4.9e9 * 2 * 3.14159,
        },
        optimal_drive={
            "amplitude": 5e7, "duration": 480e-9,
            "detuning": 0.0, "edge_sigma": 2e-9,
        },
        predicted_F_assign=0.9984,
        predicted_F_uncertainty=1e-3,
        top_3_sensitivities=[s, s, s],
        all_sensitivities=[s, s, s, s, s, s, s],
        dominant_loss_channel="T1_intrinsic",
        sensitivity_warnings=[],
        recommendation_narrative="...",
    )
    assert r.predicted_F_assign == 0.9984
    assert len(r.top_3_sensitivities) == 3


def test_O6_3_recommendation_report_rejects_empty_all_sensitivities():
    from pydantic import ValidationError
    from dispersive_readout.optimization.recommend import RecommendationReport
    with pytest.raises(ValidationError):
        RecommendationReport(
            device_parameters_fitted={},
            optimal_drive={},
            predicted_F_assign=0.99,
            predicted_F_uncertainty=1e-3,
            top_3_sensitivities=[],
            all_sensitivities=[],              # empty → reject
            dominant_loss_channel="T1_intrinsic",
            sensitivity_warnings=[],
            recommendation_narrative="",
        )


# ────────────────────────────────────────────────────────────────────
# _format_value_with_sigma — metrology σ convention (Q9b + Nit 1)
# ────────────────────────────────────────────────────────────────────

def test_format_value_with_sigma_rounds_up_to_one_sig_fig():
    """σ=0.00022 rounds UP to 0.0003 at 1 sig fig (metrology standard).
    Value matches σ's last-decimal position."""
    from dispersive_readout.optimization.recommend import _format_value_with_sigma
    val_s, sig_s = _format_value_with_sigma(value=0.99943, sigma=0.00022)
    # 0.00022 at 1 sig fig, rounded up → 0.0003; value to 4 decimals matching
    assert sig_s == "0.0003", f"Expected '0.0003', got {sig_s!r}"
    assert val_s == "0.9994", f"Expected '0.9994', got {val_s!r}"


def test_format_value_with_sigma_handles_asymmetric():
    from dispersive_readout.optimization.recommend import _format_value_with_sigma
    val_s, sig_s = _format_value_with_sigma(
        value=86.0, sigma=0.0, sigma_lo=3.0, sigma_hi=5.0,
    )
    # Asymmetric: value +σ_hi / −σ_lo; both σ rounded up to 1 sig fig.
    # Accept ASCII hyphen or Unicode minus (U+2212), with or without space.
    assert "+5" in sig_s
    assert any(tok in sig_s for tok in ("−3", "-3", "− 3", "- 3"))


# ────────────────────────────────────────────────────────────────────
# Narrative round-trip: no raw format tokens leak into the output
# ────────────────────────────────────────────────────────────────────

def test_generate_narrative_contains_no_raw_format_tokens():
    """If the template f-string is mis-populated, raw {placeholder}
    tokens will appear. Spec §9 item 8 — fix the formatting, not the text."""
    from dispersive_readout.optimization.recommend import (
        RecommendationReport, generate_narrative,
    )
    from dispersive_readout.optimization.sensitivity import SensitivityResult

    s = SensitivityResult(
        parameter="chi_scale", reference_value=1.0, reference_unit="",
        sensitivity=0.42, sensitivity_uncertainty=0.02, F_reference=0.99,
    )
    import math as _math
    r = RecommendationReport(
        device_parameters_fitted={
            "T_1": {"value": 86e-6, "uncertainty": 2e-6},
            "T_2_echo": {"value": 40e-6, "uncertainty": 1.5e-6},
            "omega_q": {"value": 4.89e9 * 2 * _math.pi, "uncertainty": 5e6 * 2 * _math.pi},
        },
        optimal_drive={
            "amplitude": 5e7, "duration": 480e-9, "detuning": 0.0, "edge_sigma": 2e-9,
        },
        predicted_F_assign=0.9984,
        predicted_F_uncertainty=1.2e-3,
        top_3_sensitivities=[s, s, s],
        all_sensitivities=[s, s, s, s, s, s, s],
        dominant_loss_channel="T1_intrinsic",
        sensitivity_warnings=[],
        recommendation_narrative="",
    )
    narrative = generate_narrative(r)
    # No raw {...} tokens should remain
    assert "{" not in narrative and "}" not in narrative, (
        f"Narrative has unsubstituted format tokens: {narrative}"
    )
    # Dominant channel name should appear
    assert "T1_intrinsic" in narrative


# ────────────────────────────────────────────────────────────────────
# O5 — closed-loop Pareto demo tests (Day-13 Amendment #11 bifurcation)
#
# Splits each of O5a/O5b into -shift and -confirm variants gated on the
# demo device's drive relative to REFERENCE's drive. Exactly one variant
# per test (a/b) fires per run; the other skips with reason.
#
# Rationale (Amendment #11 shared-argmax finding): for the SEED=42
# recovery harness, all 50 devices inherit REFERENCE's (kappa, g,
# omega_r), so the Pareto argmax location is decoherence-invariant;
# the pipeline's recommended drive converges bit-identically to
# REFERENCE's across the harness. O5a's original "F_opt > F_default + 0.005"
# assertion is not applicable here because F_opt = F_default by
# construction. The -confirm variant asserts that the pipeline
# correctly reports no shift, while preserving -shift for the
# generic-harness case (different kappa/g across devices). See
# docs/module4_diagnostics/warm_start_grid_bug.md.
# ────────────────────────────────────────────────────────────────────

_DEMO_YAML_PATH = Path("06_Dispersive_Readout/figures/closed_loop_demo_device.yaml")
_DRIVE_SHARED_TOL = 0.05     # 5% in both eps and tau
_N_SHOTS_O5 = 10_000


def _drive_within_tolerance(
    eps_demo: float, tau_demo: float,
    eps_ref: float, tau_ref: float,
    eps_tol: float = _DRIVE_SHARED_TOL, tau_tol: float = _DRIVE_SHARED_TOL,
) -> bool:
    """Return True iff demo drive is within (eps_tol, tau_tol) of REFERENCE."""
    eps_ratio = max(eps_demo / eps_ref, eps_ref / eps_demo)
    tau_ratio = max(tau_demo / tau_ref, tau_ref / tau_demo)
    return (eps_ratio - 1.0) < eps_tol and (tau_ratio - 1.0) < tau_tol


def _shot_noise_sigma_F(F: float, n_shots: int = _N_SHOTS_O5) -> float:
    """Binomial-proportion SE of F_assign at n_shots (independent Bernoulli)."""
    import math as _math
    return _math.sqrt(F * (1.0 - F) / n_shots)


def _compute_demo_device_gate() -> tuple[bool | None, str]:
    """Return (is_shared_argmax, reason_string) evaluated at test-collection
    time from the committed closed_loop_demo_device.yaml.

    None means the yaml is missing or malformed; both shift and confirm
    variants skip. True means the demo drive is within 5% of REFERENCE
    on both axes (shared-argmax regime); shift variant skips, confirm runs.
    False means the demo drive is outside 5% on either axis (distinct-argmax
    regime); confirm skips, shift runs.
    """
    import yaml
    if not _DEMO_YAML_PATH.exists():
        return None, f"{_DEMO_YAML_PATH} missing - run pick_closed_loop_demo_device.py"
    try:
        payload = yaml.safe_load(_DEMO_YAML_PATH.read_text())
        c = payload["chosen"]
        r = payload["reference_optimum"]
        eps_demo = float(c["epsilon_0_opt"])
        tau_demo = float(c["tau_opt_ns"]) * 1e-9
        eps_ref = float(r["epsilon_0_opt"])
        tau_ref = float(r["tau_opt_ns"]) * 1e-9
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"malformed {_DEMO_YAML_PATH}: {exc!r}"
    shared = _drive_within_tolerance(eps_demo, tau_demo, eps_ref, tau_ref)
    eps_ratio = max(eps_demo / eps_ref, eps_ref / eps_demo)
    tau_ratio = max(tau_demo / tau_ref, tau_ref / tau_demo)
    return shared, (
        f"eps ratio={eps_ratio:.4f}, tau ratio={tau_ratio:.4f} "
        f"(shared={'yes' if shared else 'no'}; tol={_DRIVE_SHARED_TOL})"
    )


_IS_SHARED_ARGMAX, _GATE_REASON = _compute_demo_device_gate()


def _load_demo_device():
    """Load the Day-13 picked demo device; build its DeviceConfig + payload."""
    from dataclasses import replace
    import yaml
    from dispersive_readout.physics.config import REFERENCE_DEVICE

    if not _DEMO_YAML_PATH.exists():
        pytest.skip(
            "closed_loop_demo_device.yaml missing - run "
            "scripts/pick_closed_loop_demo_device.py first (Task 17 Step 2)."
        )
    payload = yaml.safe_load(_DEMO_YAML_PATH.read_text())
    c = payload["chosen"]
    new_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=1.0 / (c["T_1_us"] * 1e-6),
        gamma_phi=max(
            1.0 / (c["T_2_echo_us"] * 1e-6) - 0.5 / (c["T_1_us"] * 1e-6), 0.0,
        ),
    )
    return replace(REFERENCE_DEVICE, decoherence=new_dec), payload


# ── O5a-shift ────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    _IS_SHARED_ARGMAX is None or _IS_SHARED_ARGMAX is True,
    reason=f"Shared-argmax regime (or yaml missing); O5a-shift N/A. {_GATE_REASON}",
)
def test_O5a_shift_closed_loop_modeled_improvement():
    """F_opt_analytic - F_default_analytic > 0.005 on the fitted demo device.

    Fires only when demo drive differs from REFERENCE by > 5% on at
    least one axis. Threshold 0.005 exceeds SLSQP ftol (1e-6) and
    Lindblad rtol (~1e-5) - asserts genuine modeled improvement.
    """
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, payload = _load_demo_device()
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    r0 = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1 = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default = compute_assignment_fidelity(
        r0, r1, (50e-9, ref_drive.duration),
        n_shots=_N_SHOTS_O5, noise_model="analytic",
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    F_opt = p_opt.F_assign_opt

    delta = F_opt - F_default
    assert delta > 0.005, (
        f"F_opt - F_default = {delta:.4f} <= 0.005 in the -shift regime "
        f"(eps/tau drift > 5%). Either the recommendation bridge is "
        "miscalibrated or the solver converged to a spurious optimum."
    )


# ── O5a-confirm ──────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    _IS_SHARED_ARGMAX is None or _IS_SHARED_ARGMAX is False,
    reason=f"Non-shared-argmax regime (or yaml missing); O5a-confirm N/A. {_GATE_REASON}",
)
def test_O5a_confirm_closed_loop_statistical_consistency():
    """Shared-argmax regime: verify pipeline correctly confirms the shared
    optimum without spurious drift, AND shows the decoherence penalty.

    Three assertions (Day-13 Amendment #11):
      1. Consistency: |F_opt - F_default| < 2*sigma_shot (zero by construction
         since demo drive = REFERENCE drive; guards against future regressions).
      2. Identity: demo drive is within 5% of REFERENCE on both axes
         (re-asserts the gate's precondition for future auditability).
      3. Decoherence visibility: F_opt(demo) < F_opt(REFERENCE) by more than
         shot noise (the pipeline correctly reports the harder device's
         lower F despite using the shared optimal drive).
    """
    import yaml
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, payload = _load_demo_device()
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    r0 = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1 = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default = compute_assignment_fidelity(
        r0, r1, (50e-9, ref_drive.duration),
        n_shots=_N_SHOTS_O5, noise_model="analytic",
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    F_opt = p_opt.F_assign_opt

    # (1) Consistency: F_opt agrees with F_default within shot noise
    sigma_F = _shot_noise_sigma_F(F_opt, _N_SHOTS_O5)
    assert abs(F_opt - F_default) < 2.0 * sigma_F, (
        f"Shared-argmax regime but F_opt={F_opt:.5f} differs from "
        f"F_default={F_default:.5f} by {abs(F_opt - F_default):.5f} "
        f"(> 2 sigma_shot = {2*sigma_F:.5f}). Indicates the pipeline "
        "is recommending a subtly different drive than REFERENCE's."
    )

    # (2) Identity: demo drive within 5% of REFERENCE on both axes
    c = payload["chosen"]
    r = payload["reference_optimum"]
    eps_ratio = max(c["epsilon_0_opt"] / r["epsilon_0_opt"], r["epsilon_0_opt"] / c["epsilon_0_opt"])
    tau_ratio = max(c["tau_opt_ns"] / r["tau_opt_ns"], r["tau_opt_ns"] / c["tau_opt_ns"])
    assert (eps_ratio - 1.0) < _DRIVE_SHARED_TOL, (
        f"Demo drive eps ratio {eps_ratio:.4f} exceeds 5% tolerance; "
        "gate selection was wrong."
    )
    assert (tau_ratio - 1.0) < _DRIVE_SHARED_TOL, (
        f"Demo drive tau ratio {tau_ratio:.4f} exceeds 5% tolerance; "
        "gate selection was wrong."
    )

    # (3) Decoherence visibility: F_opt(demo) < F_opt(REFERENCE) - sigma_F
    ref_F_opt = float(r["F_assign_opt"])
    assert F_opt < ref_F_opt - sigma_F, (
        f"Decoherence penalty not visible: demo F_opt={F_opt:.5f} is not "
        f"measurably less than REFERENCE F_opt={ref_F_opt:.5f} "
        f"(delta={F_opt - ref_F_opt:+.5f}, sigma_shot={sigma_F:.5f}). "
        "Demo device should exhibit lower F due to faster decoherence."
    )


# ── O5b-shift ────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    _IS_SHARED_ARGMAX is None or _IS_SHARED_ARGMAX is True,
    reason=f"Shared-argmax regime (or yaml missing); O5b-shift N/A. {_GATE_REASON}",
)
def test_O5b_shift_closed_loop_shot_noise_detectability():
    """Welch-t-style test at n=10^4: p < 0.05 on modeled F_default vs F_opt.

    Fires only when demo drive differs from REFERENCE by > 5% on at
    least one axis. Asserts the modeled improvement is measurable at
    the spec's shot budget.
    """
    import math
    from scipy import stats as sp_stats
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, payload = _load_demo_device()
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    r0_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default = compute_assignment_fidelity(
        r0_d, r1_d, (50e-9, ref_drive.duration),
        n_shots=_N_SHOTS_O5, noise_model="analytic",
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    opt_drive = DriveParams(
        amplitude=p_opt.epsilon_0_opt, duration=p_opt.tau_opt, detuning=0.0,
    )
    r0_o = simulate_readout(demo_device, opt_drive, initial_qubit_state=0)
    r1_o = simulate_readout(demo_device, opt_drive, initial_qubit_state=1)
    F_opt = compute_assignment_fidelity(
        r0_o, r1_o, (50e-9, opt_drive.duration),
        n_shots=_N_SHOTS_O5, noise_model="analytic",
    ).F_assign

    sigma_d = _shot_noise_sigma_F(F_default, _N_SHOTS_O5)
    sigma_o = _shot_noise_sigma_F(F_opt, _N_SHOTS_O5)
    z = (F_opt - F_default) / math.sqrt(sigma_d ** 2 + sigma_o ** 2)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    assert p_value < 0.05, (
        f"Welch-t p = {p_value:.4f} >= 0.05 in the -shift regime: "
        "shot-noise detectability fails at n=10^4. Modeled improvement "
        f"(F_opt={F_opt:.5f}, F_default={F_default:.5f}) is smaller "
        "than shot noise."
    )


# ── O5b-confirm ──────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    _IS_SHARED_ARGMAX is None or _IS_SHARED_ARGMAX is False,
    reason=f"Non-shared-argmax regime (or yaml missing); O5b-confirm N/A. {_GATE_REASON}",
)
def test_O5b_confirm_closed_loop_shot_noise_indistinguishable():
    """Shared-argmax regime: assert |F_opt - F_default| is NOT detectable
    at the spec's shot budget (p > 0.05). Confirms the pipeline does not
    invent a spurious shot-noise-detectable shift when none exists."""
    import math
    from scipy import stats as sp_stats
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, payload = _load_demo_device()
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    r0_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default = compute_assignment_fidelity(
        r0_d, r1_d, (50e-9, ref_drive.duration),
        n_shots=_N_SHOTS_O5, noise_model="analytic",
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    F_opt = p_opt.F_assign_opt

    sigma_d = _shot_noise_sigma_F(F_default, _N_SHOTS_O5)
    sigma_o = _shot_noise_sigma_F(F_opt, _N_SHOTS_O5)
    z = (F_opt - F_default) / math.sqrt(sigma_d ** 2 + sigma_o ** 2)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    assert p_value > 0.05, (
        f"Welch-t p = {p_value:.4f} <= 0.05 in the -confirm regime: "
        f"F_opt={F_opt:.5f} and F_default={F_default:.5f} are "
        f"shot-noise-distinguishable (delta={F_opt - F_default:+.5f}), "
        "but the picker's gate reported shared-argmax regime. Either "
        "the gate tolerance is wrong or the pipeline is applying a "
        "spurious drive shift."
    )


# ────────────────────────────────────────────────────────────────────
# O9 - regression gate: regenerate sensitivities and compare against
# committed artifact (SEED=42 stable, ±2% per value)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O9_regression_gate_against_committed_yaml():
    """Regenerate per-parameter S_theta at SEED=42; compare to committed
    fig4_data.yaml. Module 3 C3 tolerance convention, with the point-vs-
    bar split from Amendment #13:

      - Bar-rendered parameters (|ref_S| >= SENSITIVITY_RENDER_BAR_THRESHOLD):
        strict ±2% relative drift (meaningful central value).
      - Point-with-errorbar parameters (|ref_S| < threshold): shot-noise-
        consistent regime; assert absolute drift <= sigma_S from the pinned
        YAML (no well-defined central value to assert 2% of).

    If the fitter legitimately improves: regenerate the artifact via
    scripts/regenerate_fig4_data.py and re-commit."""
    from pathlib import Path
    import yaml
    from dispersive_readout.optimization.sensitivity import (
        SENSITIVITY_RENDER_BAR_THRESHOLD,
    )

    committed_path = Path("06_Dispersive_Readout/figures/fig4_data.yaml")
    if not committed_path.exists():
        pytest.skip(
            "fig4_data.yaml missing - run "
            "scripts/regenerate_fig4_data.py first (Task 20 Step 1)."
        )
    committed = yaml.safe_load(committed_path.read_text())

    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_all_sensitivities

    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)

    TOL_REL = 0.02
    for observed, pinned in zip(sens, committed["sensitivities"]):
        assert observed.parameter == pinned["parameter"], (
            f"Parameter ordering drift: observed[{observed.parameter!r}] "
            f"vs pinned[{pinned['parameter']!r}]"
        )
        ref_S = pinned["S"]
        obs_S = observed.sensitivity
        sigma_S = pinned["sigma_S"]

        if abs(ref_S) >= SENSITIVITY_RENDER_BAR_THRESHOLD:
            # Bar-rendered: assert 2% relative drift
            rel = abs(obs_S - ref_S) / abs(ref_S)
            assert rel < TOL_REL, (
                f"Sensitivity S_{observed.parameter} drifted from pinned "
                f"{ref_S:.4f} to {obs_S:.4f} ({rel*100:.2f}% > 2% for "
                "bar-rendered parameter). If intentional, regenerate "
                "fig4_data.yaml."
            )
        else:
            # Point-with-errorbar: shot-noise-consistent with zero.
            # Assert absolute drift stays within the pinned sigma_S (1 sigma
            # shot-noise band); anything larger would indicate a real drift.
            abs_drift = abs(obs_S - ref_S)
            assert abs_drift <= sigma_S, (
                f"Sensitivity S_{observed.parameter} drifted from pinned "
                f"{ref_S:.5f} to {obs_S:.5f} (absolute {abs_drift:.5f} > "
                f"sigma_S={sigma_S:.5f}) for point-with-errorbar parameter. "
                "If intentional, regenerate fig4_data.yaml."
            )
