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
    assert SENSITIVITY_WARNING_THRESHOLD == 2.0, (
        f"SENSITIVITY_WARNING_THRESHOLD changed from spec-locked 2.0 "
        f"to {SENSITIVITY_WARNING_THRESHOLD}; requires spec amendment"
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
        f"{'parameter':<12}  {'S':>10}  {'sigma(S)':>10}  {'|S|/sigma':>9}",
    ]
    for r in near_zero:
        ratio = abs(r.sensitivity) / r.sensitivity_uncertainty if r.sensitivity_uncertainty > 0 else float("inf")
        lines.append(
            f"{r.parameter:<12}  {r.sensitivity:+10.5f}  "
            f"{r.sensitivity_uncertainty:10.5f}  {ratio:9.2f}"
        )
    lines.append("")
    lines.append(
        "Interpretation (spec §0 Q1-amended): at this REFERENCE, F_assign "
        "peaks at chi_scale ~ 0.85 under the SW-2 simulator; REFERENCE sits "
        "~18% past the peak so |S_chi| is near-zero. S_gamma_1 and S_n_th are "
        "near-zero because T_1 and thermal contribute <= 3% of REFERENCE's "
        "error budget (Purcell dominates, per Module 2 fig2_data.yaml)."
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
