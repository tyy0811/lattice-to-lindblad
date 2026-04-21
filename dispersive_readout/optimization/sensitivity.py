"""Sensitivity-analysis policy constants and (later) compute functions.

Policy constants (Q1, Q4, Q6 locks) are defined here — not in figure scripts —
so they are auditable, test-targeted, and version-controlled alongside the
numbers they gate.
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
# Rationale (Q4): signals devices where linearized sensitivity is locally
# unreliable — regime-change boundary (Purcell, dispersive breakdown) is near.
SENSITIVITY_WARNING_THRESHOLD: float = 2.0


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
