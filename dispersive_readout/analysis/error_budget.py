"""Coherent/incoherent error-budget decomposition data model and computation.

See MODULE_2_SPEC.md §2 (methodology), §5.3 (schemas), §6 (tests).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


ChannelName = Literal[
    "T1_intrinsic",
    "pure_dephasing",
    "thermal",
    "purcell",
    "drive_amplitude",
    "drive_detuning",
]

ChannelGroup = Literal["active_loss", "calibration_sensitivity"]


class ChannelContribution(BaseModel):
    """Single channel's contribution to the error budget.

    For active_loss channels: delta_F = F_c_off - F_full (non-negative modulo
    shot noise); uncertainty is analytic binomial SE propagated in quadrature.
    For calibration_sensitivity channels: delta_F = mean(|F_full - F_±|)
    (non-negative by construction); uncertainty is the ± asymmetry |F_+ - F_-|/2.
    """
    name: ChannelName
    group: ChannelGroup
    delta_F: float
    delta_F_uncertainty: float
    description: str
    perturbation_description: str | None = None

    @field_validator("delta_F")
    @classmethod
    def nonnegative(cls, v: float) -> float:
        if v < -0.005:
            raise ValueError(
                f"Channel contribution unexpectedly negative: {v}. "
                f"Small negatives from shot noise are floored to zero; "
                f"< -0.005 indicates a bug in the turn-off logic."
            )
        return max(v, 0.0)


class ErrorBudget(BaseModel):
    """Complete error budget at a single operating point.

    The additivity identity (F_ideal − F_full) = Σ_active ΔF_c + R_active
    holds only for the active-loss group (§2.1). Calibration-sensitivity
    channels do not enter this identity.
    """
    operating_point_id: str
    F_full: float
    F_ideal: float
    channels: list[ChannelContribution]
    residual_active: float
    residual_active_uncertainty: float

    @property
    def active_loss_channels(self) -> list[ChannelContribution]:
        return [c for c in self.channels if c.group == "active_loss"]

    @property
    def calibration_channels(self) -> list[ChannelContribution]:
        return [c for c in self.channels if c.group == "calibration_sensitivity"]

    @property
    def total_infidelity(self) -> float:
        return 1.0 - self.F_full

    @property
    def explained_active_loss(self) -> float:
        return sum(c.delta_F for c in self.active_loss_channels)


def export_budget_to_yaml(budget: ErrorBudget, path) -> None:
    """Serialize an ErrorBudget to YAML at `path` (str or Path).

    Preserves all fields and the channel list in order. Used by
    scripts/fig2_error_budget.py and test B5 round-trip.
    """
    import yaml
    from pathlib import Path

    payload = {
        "operating_point_id": budget.operating_point_id,
        "F_full": budget.F_full,
        "F_ideal": budget.F_ideal,
        "residual_active": budget.residual_active,
        "residual_active_uncertainty": budget.residual_active_uncertainty,
        "channels": [
            {
                "name": c.name,
                "group": c.group,
                "delta_F": c.delta_F,
                "delta_F_uncertainty": c.delta_F_uncertainty,
                "description": c.description,
                "perturbation_description": c.perturbation_description,
            }
            for c in budget.channels
        ],
    }
    Path(path).write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    )
