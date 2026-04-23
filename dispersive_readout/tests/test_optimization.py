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
    matches = _REQUIRED_ANALYTIC.findall(src)
    assert len(matches) >= 1, (
        "Q8 contract violated: pareto.py must call "
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
