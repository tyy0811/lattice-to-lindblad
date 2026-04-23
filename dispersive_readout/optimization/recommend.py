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
