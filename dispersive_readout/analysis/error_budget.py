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


import math
from dataclasses import replace

import numpy as np

from ..physics.config import DecoherenceParams, DeviceConfig, DriveParams
from ..physics.readout_model import (
    simulate_readout,
    compute_assignment_fidelity,
)


def _F_at(
    device: DeviceConfig,
    drive: DriveParams,
    integration_window: tuple[float, float],
    n_shots: int,
) -> tuple[float, float]:
    """Simulate |0>, |1>, return (F_assign, σ_F) using independent shot draws."""
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    f = compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian",
        rng=None,  # ephemeral RNG → independent draws
    )
    return float(f.F_assign), float(f.F_assign_uncertainty)


def _device_with_decoherence(device: DeviceConfig, **overrides) -> DeviceConfig:
    """Return a copy of device with the given DecoherenceParams field overrides."""
    new_dec = replace(device.decoherence, **overrides)
    return DeviceConfig(
        transmon=device.transmon,
        resonator=device.resonator,
        coupling=device.coupling,
        decoherence=new_dec,
        truncation=device.truncation,
    )


def compute_channel_contribution(
    operating_point,
    channel: ChannelName,
) -> ChannelContribution:
    """Compute the marginal fidelity loss attributable to a single channel.

    Active-loss channels (T1, dephasing, thermal, Purcell) zero their
    respective field and compute ΔF = F_off − F_full. Calibration-sensitivity
    channels (drive_amplitude, drive_detuning) perturb DriveParams and
    compute mean-of-absolute losses.

    See MODULE_2_SPEC.md §2.1 and §2.3 for details.
    """
    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    # Baseline F (all channels on)
    F_full, sigma_full = _F_at(device, drive, window, n_shots)

    if channel == "T1_intrinsic":
        dev_off = _device_with_decoherence(device, gamma_1=0.0)
        F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
        delta_F = F_off - F_full
        sigma_delta = math.sqrt(sigma_off**2 + sigma_full**2)
        return ChannelContribution(
            name="T1_intrinsic",
            group="active_loss",
            delta_F=delta_F,
            delta_F_uncertainty=sigma_delta,
            description="Fidelity loss from intrinsic T1 relaxation (γ_1).",
        )

    raise NotImplementedError(f"Channel {channel!r} not yet implemented.")
