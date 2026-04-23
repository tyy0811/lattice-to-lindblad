"""Pareto-frontier computation for Module 4.

See MODULE_4_SPEC.md §3.3, §5.3. SLSQP + 5×5 warm-start over (ε_0, τ)
against a noise-free analytic objective (Q8 contract). Uncertainty is
analytic binomial SE on reported F_opt.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

from ..physics.config import DeviceConfig, DriveParams, REFERENCE_DEVICE


# ────────────────────────────────────────────────────────────────────
# Spec §3.3 — locked data
# ────────────────────────────────────────────────────────────────────

PARETO_DEVICE_VARIANTS: list[dict[str, Any]] = [
    {
        "label": "REFERENCE (≈ Marxer Q1)",
        "T1_us": None,
        "kappa_MHz": None,
    },
    {
        "label": "T_1 = 40 µs (Garnet-like)",
        "T1_us": 40.0,
        "kappa_MHz": None,
    },
    {
        "label": "T_1 = 20 µs, κ/2π = 6 MHz (Bengtsson-like)",
        "T1_us": 20.0,
        "kappa_MHz": 6.0,
    },
]


# 10 log-spaced points from 100 ns to 2 µs per spec §3.3
TAU_MAX_GRID_NS: np.ndarray = np.logspace(np.log10(100.0), np.log10(2000.0), 10)


# ────────────────────────────────────────────────────────────────────
# Spec §5.3 — ParetoPoint schema
# ────────────────────────────────────────────────────────────────────

class ParetoPoint(BaseModel):
    """Optimal (ε_0, τ) at one τ_max constraint, for one device."""
    device_id: str                        # hash of DeviceConfig (audit trail)
    device_label: str
    tau_max: float
    epsilon_0_opt: float
    tau_opt: float
    F_assign_opt: float                   # analytic Gaussian-overlap F at optimum
    F_assign_uncertainty: float           # analytic binomial SE at n_shots
    dominant_loss_channel: str
    solver_converged: bool

    @field_validator("F_assign_opt")
    @classmethod
    def _valid_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"F_assign_opt must be in [0, 1] (got {v})")
        return v

    @model_validator(mode="after")
    def _tau_opt_within_tau_max(self):
        # 0.1% tolerance for solver slop
        if self.tau_opt > self.tau_max * 1.001:
            raise ValueError(
                f"tau_opt ({self.tau_opt}) exceeds tau_max ({self.tau_max}) "
                "beyond 0.1% solver tolerance"
            )
        return self


# ────────────────────────────────────────────────────────────────────
# build_variant — Koch back-solve for γ_φ preserves T2_echo at REFERENCE
# ────────────────────────────────────────────────────────────────────

def _device_id(device: DeviceConfig) -> str:
    """Deterministic short hash of the DeviceConfig for audit trail."""
    summary = {
        "T1_us": 1e6 / device.decoherence.gamma_1,
        "T2_rate": device.decoherence.gamma_phi,
        "n_th": device.decoherence.n_th,
        "kappa": device.resonator.kappa,
        "g": device.coupling.g,
        "omega_r": device.resonator.omega_r,
    }
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:12]


def build_variant(variant_spec: dict[str, Any]) -> DeviceConfig:
    """Construct a PARETO_DEVICE_VARIANTS entry from REFERENCE_DEVICE.

    Koch back-solve convention (Module 3 compatibility):
        T_2_echo is held at REFERENCE's value;
        gamma_phi is recomputed as max(1/T_2_echo - gamma_1/2, 0.0).
    This matches ExtractedParameterPack.to_device_config() so V2/V3
    construction is bridge-consistent with the closed-loop demo device.
    """
    dec_ref = REFERENCE_DEVICE.decoherence
    res_ref = REFERENCE_DEVICE.resonator

    T2_echo_REF = 2.0 / (dec_ref.gamma_1 + 2.0 * dec_ref.gamma_phi)

    # Decoherence substitution
    if variant_spec["T1_us"] is None:
        new_gamma_1 = dec_ref.gamma_1
    else:
        new_gamma_1 = 1.0 / (variant_spec["T1_us"] * 1e-6)
    new_gamma_phi = max(1.0 / T2_echo_REF - 0.5 * new_gamma_1, 0.0)
    new_dec = replace(dec_ref, gamma_1=new_gamma_1, gamma_phi=new_gamma_phi)

    # Resonator substitution
    if variant_spec["kappa_MHz"] is None:
        new_res = res_ref
    else:
        new_kappa = 2.0 * math.pi * variant_spec["kappa_MHz"] * 1e6
        new_res = replace(res_ref, kappa=new_kappa)

    return replace(REFERENCE_DEVICE, decoherence=new_dec, resonator=new_res)


# ────────────────────────────────────────────────────────────────────
# Task 13 will replace this stub with the real SLSQP + 5×5 warm-start
# implementation. Kept here so optimization/__init__.py remains importable
# between Task-12 and Task-13 commits (and so O10 Modal smoke still runs).
# ────────────────────────────────────────────────────────────────────

def find_pareto_point(device: DeviceConfig, tau_max: float) -> ParetoPoint:
    """Placeholder implementation so O10 smoke succeeds. Task 13 replaces
    the body with SLSQP + 5×5 warm-start over (epsilon_0, tau)."""
    return ParetoPoint(
        device_id=_device_id(device),
        device_label="<placeholder>",
        tau_max=float(tau_max),
        epsilon_0_opt=0.0,
        tau_opt=float(tau_max),
        F_assign_opt=0.5,
        F_assign_uncertainty=0.01,
        dominant_loss_channel="placeholder",
        solver_converged=False,
    )
