"""Coherent/incoherent error-budget decomposition data model and computation.

See MODULE_2_SPEC.md §2 (methodology), §5.3 (schemas), §6 (tests).
Amendment 9 changes the shared-baseline budget computation, the signed-ΔF
validator, and the calibration-sensitivity uncertainty propagation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator

from ..physics.config import DecoherenceParams, DeviceConfig, DriveParams
from ..physics.readout_model import (
    compute_assignment_fidelity,
    simulate_readout,
)


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

    For active_loss channels: delta_F = F_c_off - F_full (may be slightly
    negative from shot noise; signed value is preserved). Uncertainty is
    analytic binomial SE propagated in quadrature.

    For calibration_sensitivity channels: delta_F = mean(|F_full - F_±|)
    (non-negative by construction). Uncertainty combines the ± asymmetry
    and the shot-noise SE of the mean-of-abs estimator in quadrature
    (amendment 9c).
    """
    name: ChannelName
    group: ChannelGroup
    delta_F: float
    delta_F_uncertainty: float
    description: str
    perturbation_description: str | None = None

    @field_validator("delta_F")
    @classmethod
    def not_significantly_negative(cls, v: float) -> float:
        # Amendment 9b: preserve signed values. The -0.005 hard gate still
        # catches turn-off-logic bugs that would push ΔF strongly negative,
        # but small shot-noise negatives are stored as-is so the residual
        # and the YAML do not get a one-sided bias.
        if v < -0.005:
            raise ValueError(
                f"Channel contribution unexpectedly negative: {v}. "
                f"< -0.005 indicates a bug in the turn-off logic; small "
                f"shot-noise-range negatives are preserved as signed."
            )
        return v


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


# Active-loss channel → DecoherenceParams override to turn that channel off.
_ACTIVE_LOSS_OVERRIDES: dict[ChannelName, dict] = {
    "T1_intrinsic":   {"gamma_1": 0.0},
    "pure_dephasing": {"gamma_phi": 0.0},
    "thermal":        {"n_th": 0.0},
    "purcell":        {"purcell_enabled": False},
}

_ACTIVE_LOSS_DESCRIPTIONS: dict[ChannelName, str] = {
    "T1_intrinsic":   "Fidelity loss from intrinsic T1 relaxation (γ_1).",
    "pure_dephasing": "Fidelity loss from pure dephasing (γ_φ).",
    "thermal":        "Fidelity loss from thermal bath occupation (n_th).",
    "purcell":        "Fidelity loss from Purcell-enhanced decay (g²κ/Δ²).",
}


def _active_loss_contribution(
    operating_point,
    channel: ChannelName,
    F_full: float,
    sigma_full: float,
) -> ChannelContribution:
    """Active-loss ΔF using a caller-provided shared baseline (amendment 9a)."""
    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    dev_off = _device_with_decoherence(device, **_ACTIVE_LOSS_OVERRIDES[channel])
    F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
    delta_F = F_off - F_full
    sigma_delta = math.sqrt(sigma_off ** 2 + sigma_full ** 2)
    return ChannelContribution(
        name=channel,
        group="active_loss",
        delta_F=delta_F,
        delta_F_uncertainty=sigma_delta,
        description=_ACTIVE_LOSS_DESCRIPTIONS[channel],
    )


def _calibration_contribution(
    operating_point,
    channel: ChannelName,
    F_full: float,
    sigma_full: float,
) -> ChannelContribution:
    """Calibration-sensitivity ΔF with shot-noise-propagated uncertainty
    (amendment 9c)."""
    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    if channel == "drive_amplitude":
        perturbation = 0.05
        drive_plus = replace(drive, amplitude=drive.amplitude * (1.0 + perturbation))
        drive_minus = replace(drive, amplitude=drive.amplitude * (1.0 - perturbation))
        description = "Fidelity loss under ±5% drive amplitude miscalibration."
        perturbation_description = "amplitude ±5% of nominal ε₀"
    elif channel == "drive_detuning":
        kappa = device.resonator.kappa
        perturbation = kappa / 4.0
        drive_plus = replace(drive, detuning=drive.detuning + perturbation)
        drive_minus = replace(drive, detuning=drive.detuning - perturbation)
        description = "Fidelity loss under ±κ/4 drive detuning error."
        perturbation_description = "detuning ±κ/4 about nominal ω_d = ω_r"
    else:
        raise ValueError(f"Not a calibration-sensitivity channel: {channel!r}")

    F_plus, sigma_plus = _F_at(device, drive_plus, window, n_shots)
    F_minus, sigma_minus = _F_at(device, drive_minus, window, n_shots)

    delta_F = 0.5 * (abs(F_full - F_plus) + abs(F_full - F_minus))
    # Amendment 9c: combine asymmetry and shot-noise SE of mean-of-abs.
    # Var(0.5(F_full − F_+ + F_full − F_−)) = σ_F_full² + 0.25(σ_+² + σ_−²)
    # (assuming independent shot draws; correct when each _F_at call uses its
    # own ephemeral RNG, which it does).
    err_asymmetry = 0.5 * abs(F_plus - F_minus)
    sigma_shot_sq = sigma_full ** 2 + 0.25 * (sigma_plus ** 2 + sigma_minus ** 2)
    err_total = math.sqrt(err_asymmetry ** 2 + sigma_shot_sq)

    return ChannelContribution(
        name=channel,
        group="calibration_sensitivity",
        delta_F=delta_F,
        delta_F_uncertainty=err_total,
        description=description,
        perturbation_description=perturbation_description,
    )


def compute_channel_contribution(
    operating_point,
    channel: ChannelName,
) -> ChannelContribution:
    """Public per-channel API: computes a fresh baseline and returns one
    ChannelContribution. For standalone use (single-channel sanity
    checks, tests). For the full budget, `compute_full_error_budget`
    reuses a single baseline across channels (amendment 9a).
    """
    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    F_full, sigma_full = _F_at(device, drive, window, n_shots)

    if channel in _ACTIVE_LOSS_OVERRIDES:
        return _active_loss_contribution(operating_point, channel, F_full, sigma_full)
    if channel in ("drive_amplitude", "drive_detuning"):
        return _calibration_contribution(operating_point, channel, F_full, sigma_full)
    raise NotImplementedError(f"Channel {channel!r} not yet implemented.")


def _operating_point_id(operating_point) -> str:
    """Deterministic hash of OperatingPoint fields for traceability in YAML."""
    device = operating_point.device
    payload = {
        "omega_r": device.resonator.omega_r,
        "kappa": device.resonator.kappa,
        "g": device.coupling.g,
        "E_C": device.transmon.E_C,
        "E_J": device.transmon.E_J,
        "gamma_1": device.decoherence.gamma_1,
        "gamma_phi": device.decoherence.gamma_phi,
        "n_th": device.decoherence.n_th,
        "amplitude": operating_point.drive.amplitude,
        "duration": operating_point.drive.duration,
        "detuning": operating_point.drive.detuning,
        "window": list(operating_point.integration_window),
        "n_shots": operating_point.n_shots,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


_DEFAULT_CHANNELS: list[ChannelName] = [
    "T1_intrinsic",
    "pure_dephasing",
    "thermal",
    "purcell",
    "drive_amplitude",
    "drive_detuning",
]


def compute_full_error_budget(
    operating_point,
    channels: list[ChannelName] | None = None,
) -> ErrorBudget:
    """Compute the complete error budget at the given operating point.

    Per amendment 9a, all six channel contributions share a single F_full
    baseline sampled once at the budget level. This variance-reduces the
    per-channel ΔF estimates and keeps the residual math consistent with
    the channel math (both reference the same F_full sample).

    Returns an ErrorBudget with:
    - F_full: shared baseline fidelity (all channels on)
    - F_ideal: ceiling with all 4 active-loss channels disabled
    - channels: list of 6 ChannelContribution
    - residual_active: R_active = (F_ideal - F_full) - Σ_active ΔF_c
    - residual_active_uncertainty: quadrature-propagated σ_R
    """
    if channels is None:
        channels = _DEFAULT_CHANNELS

    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    # Single shared baseline, reused by every channel (amendment 9a).
    F_full, sigma_full = _F_at(device, drive, window, n_shots)

    # F_ideal: all active-loss channels disabled simultaneously
    dev_ideal = _device_with_decoherence(
        device,
        gamma_1=0.0,
        gamma_phi=0.0,
        n_th=0.0,
        purcell_enabled=False,
    )
    F_ideal, sigma_ideal = _F_at(dev_ideal, drive, window, n_shots)

    contributions: list[ChannelContribution] = []
    for ch in channels:
        if ch in _ACTIVE_LOSS_OVERRIDES:
            contributions.append(
                _active_loss_contribution(operating_point, ch, F_full, sigma_full)
            )
        elif ch in ("drive_amplitude", "drive_detuning"):
            contributions.append(
                _calibration_contribution(operating_point, ch, F_full, sigma_full)
            )
        else:
            raise ValueError(f"Unknown channel: {ch!r}")

    active = [c for c in contributions if c.group == "active_loss"]
    N = len(active)
    active_sum = sum(c.delta_F for c in active)
    residual_active = (F_ideal - F_full) - active_sum
    # Shared-baseline σ_R propagation. With F_full shared across all channel
    # ΔFs, R = F_ideal − F_full − Σ(F_off_c − F_full) = F_ideal + (N−1)·F_full
    # − Σ F_off_c, so
    #   σ_R² = σ_ideal² + (N−1)²·σ_full² + Σ σ_off_c²
    # Individual σ_ΔF_c² = σ_off_c² + σ_full², so
    #   Σ σ_off_c² = Σ σ_ΔF_c² − N·σ_full²
    # → σ_R² = σ_ideal² + ((N−1)² − N)·σ_full² + Σ σ_ΔF_c²
    # For N=4 the coefficient on σ_full² is 5 (vs the naive 1 that treats
    # ΔFs as independent). Amendment 9a.
    sum_sigma_delta_sq = sum(c.delta_F_uncertainty ** 2 for c in active)
    coeff = (N - 1) ** 2 - N
    sigma_residual_sq = (
        sigma_ideal ** 2
        + coeff * sigma_full ** 2
        + sum_sigma_delta_sq
    )
    sigma_residual = math.sqrt(sigma_residual_sq)

    return ErrorBudget(
        operating_point_id=_operating_point_id(operating_point),
        F_full=F_full,
        F_ideal=F_ideal,
        channels=contributions,
        residual_active=residual_active,
        residual_active_uncertainty=sigma_residual,
    )
