"""Closed-loop recommendation pipeline.

See MODULE_4_SPEC.md §3.4, §5.5. Narrow-scope closed loop (fitted T_1,
T_2, ω_q over REFERENCE resonator) per Q4 lock. Template-rendered
narrative with IQM-table rounding + metrology σ convention per Q9b.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, field_validator

from .sensitivity import SensitivityResult


class RecommendationReport(BaseModel):
    """Closed-loop output: fit → recommend → report."""
    device_parameters_fitted: dict[str, Any]
    optimal_drive: dict[str, Any]
    predicted_F_assign: float
    predicted_F_uncertainty: float
    top_3_sensitivities: list[SensitivityResult]
    all_sensitivities: list[SensitivityResult]
    dominant_loss_channel: str
    sensitivity_warnings: list[str]
    recommendation_narrative: str
    scope_caveat: str = (
        "Closed-loop scope: fitted (T_1, T_2, ω_q) over fixed REFERENCE "
        "resonator and coupling. Full closed-loop including resonator "
        "spectroscopy and AC-Stark characterization is post-submission roadmap."
    )

    @field_validator("all_sensitivities")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("all_sensitivities must be non-empty")
        return v


# ────────────────────────────────────────────────────────────────────
# Metrology helper (Q9b + post-commit Nit 1)
# ────────────────────────────────────────────────────────────────────

def _round_up_to_n_sig_figs(x: float, n: int) -> tuple[float, int]:
    """Round x UP to n significant figures; return (rounded, shift).

    shift is the number of decimals needed to display x at n sig figs.
    """
    if x == 0.0:
        return 0.0, 0
    magnitude = math.floor(math.log10(abs(x)))
    shift = n - 1 - magnitude
    factor = 10 ** shift
    rounded = math.ceil(abs(x) * factor) / factor
    rounded = math.copysign(rounded, x)
    return rounded, shift


def _format_value_with_sigma(
    value: float,
    sigma: float,
    sigma_lo: float | None = None,
    sigma_hi: float | None = None,
) -> tuple[str, str]:
    """Return (value_str, sigma_str) per metrology-σ convention.

    σ is rounded UP to 1 significant figure; value's display decimal
    position matches σ's last-decimal position.
    """
    # Asymmetric case
    if sigma_lo is not None or sigma_hi is not None:
        s_lo, shift_lo = _round_up_to_n_sig_figs(sigma_lo or 0.0, 1)
        s_hi, shift_hi = _round_up_to_n_sig_figs(sigma_hi or 0.0, 1)
        shift = max(shift_hi, shift_lo, 0)
        val_fmt = f"{{:.{shift}f}}".format(value)
        sig_str = f"+{s_hi:.{shift}f} / −{s_lo:.{shift}f}"
        return val_fmt, sig_str

    # Symmetric case
    sigma_rounded, shift = _round_up_to_n_sig_figs(sigma, 1)
    shift = max(shift, 0)
    val_fmt = f"{{:.{shift}f}}".format(value)
    sig_fmt = f"{{:.{shift}f}}".format(sigma_rounded)
    return val_fmt, sig_fmt


# ────────────────────────────────────────────────────────────────────
# Narrative rendering + closed-loop recommendation pipeline
# ────────────────────────────────────────────────────────────────────

from pathlib import Path

import yaml

from ..physics.config import DriveParams
from .pareto import find_pareto_point
from .sensitivity import (
    compute_all_sensitivities,
    rank_sensitivities,
    SENSITIVITY_WARNING_THRESHOLD,
)


# Escalation policy: 'solver_failed' means no valid Pareto point. 'unknown'
# means the error-budget attribution raised but the Pareto point is valid
# (F computed, solver converged); surface it as a narrative caveat rather
# than raising, since Task 14's cached data shows ~10% of Pareto points hit
# 'unknown' under normal operation.
_ESCALATE_LOSS_CHANNELS = frozenset({"solver_failed"})


def generate_narrative(report: RecommendationReport) -> str:
    """IQM-table rounding + metrology σ convention (Q9b + Nit 1).

    Delegates per-value formatting to _format_value_with_sigma so the
    metrology σ convention is applied consistently.
    """
    fitted = report.device_parameters_fitted

    T1_val, T1_sig = _format_value_with_sigma(
        fitted["T_1"]["value"] * 1e6,         # → µs
        fitted["T_1"]["uncertainty"] * 1e6,
    )
    T2_val, T2_sig = _format_value_with_sigma(
        fitted["T_2_echo"]["value"] * 1e6,
        fitted["T_2_echo"]["uncertainty"] * 1e6,
    )
    omega_val, omega_sig = _format_value_with_sigma(
        fitted["omega_q"]["value"] / (2.0 * math.pi * 1e9),   # → GHz / 2π
        fitted["omega_q"]["uncertainty"] / (2.0 * math.pi * 1e9),
    )

    drive = report.optimal_drive
    eps_MHz_2pi = drive["amplitude"] / (2.0 * math.pi * 1e6)
    tau_ns = int(round(drive["duration"] * 1e9))

    F_val, F_sig = _format_value_with_sigma(
        report.predicted_F_assign, report.predicted_F_uncertainty,
    )

    top3_fmt = ", ".join(
        f"{s.parameter} (S={s.sensitivity:+.3f})"
        for s in report.top_3_sensitivities
    )

    warning_block = ""
    if report.sensitivity_warnings:
        warning_block = (
            "\n[WARNING: "
            + "; ".join(report.sensitivity_warnings)
            + "]"
        )

    return (
        f"For the fitted device (T_1 = {T1_val} ± {T1_sig} µs, "
        f"T_2_echo = {T2_val} ± {T2_sig} µs, "
        f"ω_q/2π = {omega_val} ± {omega_sig} GHz), the recommended "
        f"readout configuration is ε_0/2π = {eps_MHz_2pi:.2g} MHz at "
        f"τ = {tau_ns} ns; predicted F_assign = {F_val} ± {F_sig}. "
        f"The dominant remaining loss channel at this optimum is "
        f"{report.dominant_loss_channel}; the top-3 parameters by |S_θ| "
        f"are {top3_fmt}.{warning_block}"
    )


def recommend_from_fitted_parameters(
    fitted,                           # dispersive_readout.characterization.ExtractedParameterPack
    tau_max: float = 500e-9,
) -> RecommendationReport:
    """Narrow closed-loop recommendation.

    1. Bridge fitted parameters to DeviceConfig via to_device_config()
       (Module 3 — inherits REFERENCE resonator/coupling).
    2. Find Pareto point at tau_max.
    3. Compute sensitivities at the per-device optimum (not REFERENCE).
    4. Emit warnings for |S_θ| > SENSITIVITY_WARNING_THRESHOLD.
    5. Render narrative template.

    Escalation policy (user directive, Day-13): if find_pareto_point
    returns an opaque dominant_loss_channel ("solver_failed" or "unknown"),
    raise RuntimeError. A failure on the recommend path is diagnostic —
    silently delegating to the narrative hides it.
    """
    from ..analysis.operating_point import OperatingPoint
    device = fitted.to_device_config()

    pareto = find_pareto_point(device, tau_max=tau_max)

    if pareto.dominant_loss_channel in _ESCALATE_LOSS_CHANNELS:
        raise RuntimeError(
            f"find_pareto_point returned dominant_loss_channel="
            f"{pareto.dominant_loss_channel!r} on the fitted device "
            f"(tau_max={tau_max*1e9:.0f}ns, F_opt={pareto.F_assign_opt:.4f}, "
            f"solver_converged={pareto.solver_converged}). "
            "Diagnose before continuing: either the fitted device is "
            "pathological (e.g. near a regime-change boundary) or the "
            "Pareto solver hit an edge case. Do not silently fall back "
            "to the narrative on solver failures."
        )

    # Sensitivities at the PER-DEVICE optimum
    drive_opt = DriveParams(
        amplitude=pareto.epsilon_0_opt,
        duration=pareto.tau_opt,
        detuning=0.0,
    )
    op_at_opt = OperatingPoint(
        device=device,
        drive=drive_opt,
        integration_window=(50e-9, pareto.tau_opt),
        n_shots=10_000,
    )
    all_sens = compute_all_sensitivities(op_at_opt)
    ranked = rank_sensitivities(all_sens)

    warnings_ = [
        f"|S_{s.parameter}| = {abs(s.sensitivity):.2f} at fitted-device optimum: "
        f"device sits near regime-change boundary; linearized sensitivity "
        f"ranking is locally unreliable."
        for s in ranked
        if abs(s.sensitivity) > SENSITIVITY_WARNING_THRESHOLD
    ]
    # Surface 'unknown' dominant_loss_channel as a caveat rather than raising
    # (error-budget attribution failed but Pareto point is valid).
    if pareto.dominant_loss_channel == "unknown":
        warnings_.append(
            f"dominant_loss_channel='unknown' at the fitted-device optimum: "
            f"error-budget attribution raised while the Pareto point itself "
            f"converged (F_opt={pareto.F_assign_opt:.4f}, "
            f"eps_0_opt={pareto.epsilon_0_opt:.3e}, tau_opt={pareto.tau_opt*1e9:.1f} ns). "
            "Narrative reports 'unknown'; diagnose error_budget for this "
            "device before attributing a physical loss channel."
        )

    # Extract fitted parameter values for the narrative.
    fitted_as_dict = {
        p.name: {"value": p.value, "uncertainty": p.uncertainty}
        for p in fitted.fitted_parameters
        if p.name in {"T_1", "T_2_echo", "omega_q"}
    }

    report = RecommendationReport(
        device_parameters_fitted=fitted_as_dict,
        optimal_drive={
            "amplitude": pareto.epsilon_0_opt,
            "duration": pareto.tau_opt,
            "detuning": 0.0,
            "edge_sigma": 2e-9,
        },
        predicted_F_assign=pareto.F_assign_opt,
        predicted_F_uncertainty=pareto.F_assign_uncertainty,
        top_3_sensitivities=ranked[:3],
        all_sensitivities=ranked,
        dominant_loss_channel=pareto.dominant_loss_channel,
        sensitivity_warnings=warnings_,
        recommendation_narrative="",          # filled below
    )
    # Render and re-inject narrative (Pydantic immutable → new instance)
    return report.model_copy(
        update={"recommendation_narrative": generate_narrative(report)}
    )


def export_recommendation_to_yaml(
    report: RecommendationReport, path: str | Path,
) -> None:
    """Serialize RecommendationReport to YAML (closed-loop artifact)."""
    data = report.model_dump()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
