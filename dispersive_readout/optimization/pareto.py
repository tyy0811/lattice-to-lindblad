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
# Spec §5.3 — find_pareto_point: SLSQP + 5×5 warm-start
# All F evaluations use noise_model='analytic' per Q8 contract (amended
# spec §0.1 item 11: Q8 forbids 'gaussian' AND 'ideal', requires
# 'analytic' at least once; the finite-SNR analytic pathway F=Φ(SNR/2)).
# ────────────────────────────────────────────────────────────────────

from scipy.optimize import minimize

from ..physics.readout_model import simulate_readout, compute_assignment_fidelity


def _F_analytic_at(
    device: DeviceConfig, eps_0: float, tau: float,
    integration_window: tuple[float, float | None] = (50e-9, None),
) -> float:
    """Finite-SNR analytic F_assign at (eps_0, tau). Uses
    noise_model='analytic' per Q8 contract — F = Φ(SNR/2), the ensemble-
    mean F under the Gaussian noise model in the continuous-shot limit."""
    drive = DriveParams(amplitude=float(eps_0), duration=float(tau), detuning=0.0)
    t_win = (integration_window[0], tau) if integration_window[1] is None else integration_window
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    return compute_assignment_fidelity(
        r0, r1, t_win, n_shots=10_000, noise_model="analytic",
    ).F_assign


def _warm_start_grid_best(
    device: DeviceConfig,
    eps_0_bounds: tuple[float, float],
    tau_bounds: tuple[float, float],
    n_side: int = 5,
) -> tuple[float | None, float | None, float]:
    """Scan a 5×5 (ε_0, τ) grid and return (eps_star, tau_star, F_star)."""
    eps_grid = np.linspace(eps_0_bounds[0], eps_0_bounds[1], n_side)
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], n_side)

    best_eps, best_tau, best_F = None, None, -1.0
    for e in eps_grid:
        for t in tau_grid:
            try:
                F = _F_analytic_at(device, e, t)
            except Exception:
                continue
            if F > best_F:
                best_eps, best_tau, best_F = float(e), float(t), float(F)
    return best_eps, best_tau, best_F


def find_pareto_point(
    device: DeviceConfig,
    tau_max: float,
    epsilon_0_bounds: tuple[float, float] = (1e6, 1e9),
    tau_bounds: tuple[float, float] | None = None,
    n_warm_start_grid_side: int = 5,
) -> ParetoPoint:
    """Find (ε_0, τ) that maximize F_assign subject to τ ≤ tau_max.

    1. Coarse 5×5 grid warm-start.
    2. SLSQP local refinement against -F (minimize).
    3. Analytic binomial SE on the converged F_opt.
    All F evaluations use noise_model='analytic' (amended Q8 contract).
    """
    if tau_bounds is None:
        tau_bounds = (50e-9, tau_max)

    e_star, t_star, F_warm = _warm_start_grid_best(
        device, epsilon_0_bounds, tau_bounds, n_side=n_warm_start_grid_side,
    )
    if e_star is None:
        # All grid evaluations failed — solver cannot proceed
        return ParetoPoint(
            device_id=_device_id(device),
            device_label="<unknown>",
            tau_max=float(tau_max),
            epsilon_0_opt=float(epsilon_0_bounds[0]),
            tau_opt=float(tau_bounds[0]),
            F_assign_opt=0.5,
            F_assign_uncertainty=1e-3,
            dominant_loss_channel="solver_failed",
            solver_converged=False,
        )

    def neg_F(x: np.ndarray) -> float:
        return -_F_analytic_at(device, x[0], x[1])

    res = minimize(
        neg_F,
        x0=np.array([e_star, t_star]),
        method="SLSQP",
        bounds=[epsilon_0_bounds, tau_bounds],
        options={"ftol": 1e-6, "maxiter": 80},
    )

    eps_opt = float(np.clip(res.x[0], *epsilon_0_bounds))
    tau_opt = float(np.clip(res.x[1], *tau_bounds))
    F_opt = float(-res.fun)

    sigma_F = math.sqrt(F_opt * (1.0 - F_opt) / 10_000.0)

    # Dominant loss channel: query Module 2's error-budget at this operating point.
    try:
        from ..analysis.operating_point import OperatingPoint
        from ..analysis.error_budget import compute_full_error_budget
        op = OperatingPoint(
            device=device,
            drive=DriveParams(amplitude=eps_opt, duration=tau_opt, detuning=0.0),
            integration_window=(50e-9, tau_opt),
            n_shots=10_000,
        )
        budget = compute_full_error_budget(op)
        # Dominant active-loss channel = max delta_F among active_loss
        active = budget.active_loss_channels
        if active:
            dominant = max(active, key=lambda c: c.delta_F).name
        else:
            dominant = "none"
    except Exception:
        # If error-budget query fails, don't fail the Pareto point — label unknown
        dominant = "unknown"

    return ParetoPoint(
        device_id=_device_id(device),
        device_label="<set-by-caller>",
        tau_max=float(tau_max),
        epsilon_0_opt=eps_opt,
        tau_opt=tau_opt,
        F_assign_opt=F_opt,
        F_assign_uncertainty=float(sigma_F),
        dominant_loss_channel=str(dominant),
        solver_converged=bool(res.success),
    )
