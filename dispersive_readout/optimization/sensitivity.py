"""Sensitivity-analysis policy constants and compute functions.

Policy constants (Q1, Q4, Q6 locks) are defined here — not in figure scripts —
so they are auditable, test-targeted, and version-controlled alongside the
numbers they gate.

Q8 contract (MODULE_4_SPEC.md §3.6): all F-evaluations in this module's
inner loops use noise_model='analytic'. Forbidden: 'gaussian' (shot noise
pollutes FD gradients) and 'ideal' (zero-SNR limit returns F=1, useless
for sensitivity analysis). Shot-noise sampling appears only in O5b's
Welch-t detectability check. Enforced by test O8.
"""
from __future__ import annotations


# Central finite-difference fractional perturbation.
# Rationale: large enough to beat simulator numerical noise; small enough
# that higher-order FD error remains <1% (confirmed by O2 step-independence).
SENSITIVITY_FD_STEP: float = 0.05

# Below this, render sensitivity as point-with-errorbar (not filled bar).
# Rationale (Q6/β): 10× below the spec's 0.3 dominance threshold; deterministic
# across runs (avoids filled-bar flicker between 0.025 and 0.035 replicates).
SENSITIVITY_RENDER_BAR_THRESHOLD: float = 0.03

# Above this, emit a boundary-proximity warning in RecommendationReport.
# Rationale (Q4, amended during Module 4 execution): signals devices where
# one parameter reaches dominance-level sensitivity, flagging operating
# points where a single parameter controls F_assign and the linearized
# ranking is less informative.
#
# Amended 2.0 → 0.3 after three-check verification (docs/module4_diagnostics/):
#   - Empirical |S| ceiling under the Lindblad simulator caps at ~0.4 across
#     the realistic parameter space (|S_ε0|_max = 0.39 at ε/2π = 15 MHz;
#     |S_γ1|_max = 0.25 at T_1 = 0.22 µs).
#   - Ceiling verified as genuine Lindblad physics (not solver / truncation /
#     Purcell-leak artifact) via three independent reproducibility checks.
#   - 0.3 aligns with spec §2.1's "dominance" level.
#   - Threshold 2.0 was unreachable under the simulator (dead code).
SENSITIVITY_WARNING_THRESHOLD: float = 0.3


from typing import Literal
from pydantic import BaseModel, field_validator, model_validator


ParameterName = Literal[
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
]


class SensitivityResult(BaseModel):
    """Normalized log-sensitivity of F_assign to one parameter.

    See MODULE_4_SPEC.md §5.1 for the schema contract.
    """
    parameter: ParameterName
    reference_value: float
    reference_unit: str
    sensitivity: float                      # S_θ = ∂ ln F / ∂ ln θ
    sensitivity_uncertainty: float          # σ(S_θ) from analytic SE propagation
    F_reference: float                      # F at θ_ref
    step_size_used: float = SENSITIVITY_FD_STEP
    method: Literal["finite_diff", "autodiff"] = "finite_diff"
    noise_consistent_with_zero: bool = False  # auto-populated (|S| < threshold)

    @field_validator("sensitivity_uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"sensitivity_uncertainty must be >= 0 (got {v})"
            )
        return v

    @field_validator("F_reference")
    @classmethod
    def _valid_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"F_reference must be in [0, 1] (got {v})")
        return v

    @model_validator(mode="after")
    def _auto_flag_noise_consistent(self):
        """Auto-populate noise_consistent_with_zero from |sensitivity|."""
        flag = abs(self.sensitivity) < SENSITIVITY_RENDER_BAR_THRESHOLD
        # Pydantic v2 model_validator 'after' allows field reassignment.
        object.__setattr__(self, "noise_consistent_with_zero", flag)
        return self


import math
from dataclasses import replace
import numpy as np

from ..physics.config import DeviceConfig, DriveParams
from ..physics.readout_model import simulate_readout, compute_assignment_fidelity
from ..analysis.operating_point import OperatingPoint


def _evaluate_F_analytic(
    device: DeviceConfig,
    drive: DriveParams,
    integration_window: tuple[float, float],
    n_shots: int,
    chi_scale: float = 1.0,
) -> float:
    """Single-point F_assign evaluation — Module 1's noise_model='analytic'.

    F = Φ(SNR/2), the ensemble-mean F under the gaussian noise model in the
    continuous-shot limit. Module 1's 'analytic' mode is the canonical path
    (pinned by the gaussian → analytic invariant test). Per the Q8 contract
    above, this module must not use 'gaussian' (shot noise) or 'ideal'
    (F=1.0 zero-noise limit) in sensitivity inner loops.
    """
    r0 = simulate_readout(
        device, drive, initial_qubit_state=0, chi_scale=chi_scale,
    )
    r1 = simulate_readout(
        device, drive, initial_qubit_state=1, chi_scale=chi_scale,
    )
    return compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="analytic",
    ).F_assign


def _perturbed_device_drive_scale(
    op: OperatingPoint,
    parameter: ParameterName,
    fractional_delta: float,
) -> tuple[DeviceConfig, DriveParams, float]:
    """Return (perturbed_device, perturbed_drive, chi_scale) for one perturbation.

    Returns the trio that `_evaluate_F_analytic` needs; all non-perturbed fields
    are copied unchanged.
    """
    device, drive = op.device, op.drive
    chi_scale = 1.0  # baseline; only chi_scale-parameter path overrides

    if parameter == "chi_scale":
        chi_scale = 1.0 + fractional_delta
    elif parameter == "kappa":
        new_res = replace(device.resonator, kappa=device.resonator.kappa * (1.0 + fractional_delta))
        device = replace(device, resonator=new_res)
    elif parameter == "gamma_1":
        new_dec = replace(
            device.decoherence,
            gamma_1=device.decoherence.gamma_1 * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "gamma_phi":
        new_dec = replace(
            device.decoherence,
            gamma_phi=device.decoherence.gamma_phi * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "n_th":
        new_dec = replace(
            device.decoherence,
            n_th=device.decoherence.n_th * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "epsilon_0":
        drive = replace(drive, amplitude=drive.amplitude * (1.0 + fractional_delta))
    elif parameter == "tau":
        drive = replace(drive, duration=drive.duration * (1.0 + fractional_delta))
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

    return device, drive, chi_scale


def _reference_value_and_unit(op: OperatingPoint, parameter: ParameterName) -> tuple[float, str]:
    """Return (θ_ref, unit_str) for the parameter at the operating point."""
    mapping = {
        "chi_scale":  (1.0, "dimensionless (multiplicative)"),
        "kappa":      (op.device.resonator.kappa, "rad/s"),
        "gamma_1":    (op.device.decoherence.gamma_1, "1/s"),
        "gamma_phi":  (op.device.decoherence.gamma_phi, "1/s"),
        "n_th":       (op.device.decoherence.n_th, "dimensionless"),
        "epsilon_0":  (op.drive.amplitude, "rad/s"),
        "tau":        (op.drive.duration, "s"),
    }
    return mapping[parameter]


def compute_log_sensitivity(
    operating_point: OperatingPoint,
    parameter: ParameterName,
    step_size: float = SENSITIVITY_FD_STEP,
) -> SensitivityResult:
    """Compute S_θ = ∂ ln F / ∂ ln θ via central finite differences.

    Uses Module 1's analytic F pathway (Φ(SNR/2)) at both probe points;
    σ(S_θ) is propagated from the analytic binomial SE on F_ref.
    """
    op = operating_point
    integration_window = op.integration_window
    n_shots = op.n_shots

    # Reference F (unperturbed)
    F_ref = _evaluate_F_analytic(
        op.device, op.drive, integration_window, n_shots, chi_scale=1.0,
    )

    # Plus perturbation
    dev_p, drv_p, chi_p = _perturbed_device_drive_scale(op, parameter, +step_size)
    F_plus = _evaluate_F_analytic(dev_p, drv_p, integration_window, n_shots, chi_scale=chi_p)

    # Minus perturbation
    dev_m, drv_m, chi_m = _perturbed_device_drive_scale(op, parameter, -step_size)
    F_minus = _evaluate_F_analytic(dev_m, drv_m, integration_window, n_shots, chi_scale=chi_m)

    # Central finite difference in log-log space
    S = (math.log(F_plus) - math.log(F_minus)) / (2.0 * step_size)

    # Uncertainty propagation from analytic binomial SE on F_ref.
    # σ(F) = sqrt(F(1-F)/n); propagate to σ(ln F) = σ(F)/F;
    # central-diff uncertainty: sqrt(2) * σ(ln F) / (2h) = σ(F) / (sqrt(2) * h * F).
    sigma_F_ref = math.sqrt(F_ref * (1.0 - F_ref) / n_shots)
    sigma_S = sigma_F_ref / (math.sqrt(2.0) * step_size * F_ref)

    theta_ref, unit = _reference_value_and_unit(op, parameter)

    return SensitivityResult(
        parameter=parameter,
        reference_value=theta_ref,
        reference_unit=unit,
        sensitivity=float(S),
        sensitivity_uncertainty=float(sigma_S),
        F_reference=float(F_ref),
        step_size_used=step_size,
        method="finite_diff",
    )


_ALL_PARAMETER_NAMES: tuple[ParameterName, ...] = (
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
)


def compute_all_sensitivities(
    operating_point: OperatingPoint,
    parameters: list[ParameterName] | None = None,
    step_size: float = SENSITIVITY_FD_STEP,
) -> list[SensitivityResult]:
    """Compute sensitivities for all 7 parameters (default) at the operating point.

    Parameters
    ----------
    operating_point : OperatingPoint
        Device + drive + integration window + n_shots.
    parameters : list[ParameterName] | None
        Subset to compute. Default (None) → all 7.
    step_size : float
        Fractional perturbation per parameter; defaults to SENSITIVITY_FD_STEP.
    """
    params = parameters if parameters is not None else list(_ALL_PARAMETER_NAMES)
    return [compute_log_sensitivity(operating_point, p, step_size=step_size) for p in params]


def rank_sensitivities(results: list[SensitivityResult]) -> list[SensitivityResult]:
    """Sort by |sensitivity|, descending. Stable sort (ties preserve input order).

    Used for tornado-plot ordering per spec §7 Panel (a).
    """
    return sorted(results, key=lambda r: abs(r.sensitivity), reverse=True)


def day_10_cross_check_s_g_vs_s_chi(
    operating_point: OperatingPoint,
    step_size: float = SENSITIVITY_FD_STEP,
) -> dict:
    """Day-10 Q1 cross-check: compute S_g via ±5% on coupling.g and compare
    to 2·S_chi. Returns dict with keys S_chi, S_g, predicted_S_g, residual,
    residual_fractional. The values feed the Figure 4 caption verbatim.

    Under the transmon χ ≈ 2 g² α / (Δ(Δ+α)) at fixed (κ, α), S_g ≈ 2·S_chi
    holds at leading order. Any deviation quantifies Purcell-coupling
    contamination in an (A)-style χ-sensitivity (which is exactly why Q1
    chose chi_scale as an orthogonal lever).
    """
    op = operating_point

    # S_chi via chi_scale (Q1 locked path)
    s_chi = compute_log_sensitivity(op, "chi_scale", step_size=step_size).sensitivity

    # S_g via direct perturbation of coupling.g — re-derives γ_Purcell,
    # so this deliberately carries the Purcell-coupling overlap Q1 is
    # measuring against.
    def _F_at_g(g_value: float) -> float:
        new_coupling = replace(op.device.coupling, g=g_value)
        new_device = replace(op.device, coupling=new_coupling)
        return _evaluate_F_analytic(
            new_device, op.drive, op.integration_window, op.n_shots, chi_scale=1.0,
        )

    g_ref = op.device.coupling.g
    F_plus = _F_at_g(g_ref * (1.0 + step_size))
    F_minus = _F_at_g(g_ref * (1.0 - step_size))
    s_g = (math.log(F_plus) - math.log(F_minus)) / (2.0 * step_size)

    predicted_s_g = 2.0 * s_chi
    residual = s_g - predicted_s_g
    residual_fractional = abs(residual) / max(abs(predicted_s_g), 1e-12)

    return {
        "S_chi": float(s_chi),
        "S_g": float(s_g),
        "predicted_S_g": float(predicted_s_g),
        "residual": float(residual),
        "residual_fractional": float(residual_fractional),
    }
