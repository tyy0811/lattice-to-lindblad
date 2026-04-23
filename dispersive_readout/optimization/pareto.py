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
    mean F under the Gaussian noise model in the continuous-shot limit.

    Zero-width integration window guard (Day-13 finding): when callers
    (SLSQP finite-difference probes, warm-start grid at tau lower bound)
    evaluate at tau <= integration_window[0], `integrated_iq` raises
    "Window contains fewer than 2 samples". Return F=0.5 (chance level)
    so the solver sees a bad region instead of propagating an exception.
    The warm-start grid's try/except swallowed this previously; SLSQP
    FD probes cannot tolerate the exception, hence the explicit guard."""
    t_win = (integration_window[0], tau) if integration_window[1] is None else integration_window
    if t_win[1] <= t_win[0] + 1e-12:
        return 0.5
    drive = DriveParams(amplitude=float(eps_0), duration=float(tau), detuning=0.0)
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    return compute_assignment_fidelity(
        r0, r1, t_win, n_shots=10_000, noise_model="analytic",
    ).F_assign


def _warm_start_grid_all(
    device: DeviceConfig,
    eps_0_bounds: tuple[float, float],
    tau_bounds: tuple[float, float],
    n_side: int = 10,
) -> list[tuple[float, float, float]]:
    """Scan an n_side x n_side (eps_0, tau) grid; return all (eps, tau, F) tuples.

    Grid topology (Day-13 solver-bug fix — Amendment #10):
      - eps_0: log-spaced over the bounds. The eps domain spans 3 decades
        (1e6 to 1e9 by default); linear spacing structurally missed the
        basin at eps ~ 1.59e8 in the n=5 regime. See
        docs/module4_diagnostics/warm_start_grid_bug.md.
      - tau: linear-spaced over the bounds. The tau domain spans ~1 decade
        and F is monotone in tau up to boundary saturation; log-spacing
        is overkill.

    Returns all grid evaluations rather than the best one so the caller
    can pick top-K diverse starting points for multi-start SLSQP —
    necessary because the F(eps) surface at fixed tau is multimodal
    in this decoherence regime (two peaks, ~0.08 F-deep valley between
    them at REFERENCE; verified across the full T_1 harness range).
    Skips points whose F evaluation raises (e.g., zero-width integration
    windows at tau=50ns when tau_bounds[0] equals the grid's tau edge).
    """
    eps_grid = np.logspace(
        np.log10(eps_0_bounds[0]), np.log10(eps_0_bounds[1]), n_side,
    )
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], n_side)

    points: list[tuple[float, float, float]] = []
    for e in eps_grid:
        for t in tau_grid:
            try:
                F = _F_analytic_at(device, e, t)
            except Exception:
                continue
            points.append((float(e), float(t), float(F)))
    return points


def _select_top_k_diverse(
    points: list[tuple[float, float, float]],
    k: int = 5,
    min_eps_ratio: float = 1.2,
) -> list[tuple[float, float, float]]:
    """Select top-k (eps, tau, F) points by F with eps-only separation.

    Multi-start SLSQP needs starts in DIFFERENT eps basins because the
    F(eps, tau) surface is multimodal in eps (two peaks at REFERENCE,
    verified via Day-13 scans) but effectively unimodal in tau (monotone
    up to boundary saturation). Enforcing separation in tau is
    counterproductive: two starts at the same eps but different tau
    converge to the same eps basin; that's not diversity.

    Two points (eps1, _, _) and (eps2, _, _) are "too close" if
        max(eps1/eps2, eps2/eps1) < min_eps_ratio
    i.e. within the given factor in eps (symmetric ratio).
    """
    sorted_desc = sorted(points, key=lambda p: p[2], reverse=True)
    selected: list[tuple[float, float, float]] = []
    for candidate in sorted_desc:
        eps_c, _, _ = candidate
        too_close = False
        for sel_eps, _, _ in selected:
            eps_ratio = max(eps_c / sel_eps, sel_eps / eps_c)
            if eps_ratio < min_eps_ratio:
                too_close = True
                break
        if not too_close:
            selected.append(candidate)
            if len(selected) >= k:
                break
    return selected


def _refine_eps_sub_grid(
    device: DeviceConfig,
    eps_start: float,
    tau_start: float,
    eps_bounds: tuple[float, float],
    n_points: int = 5,
    half_width_ratio: float = 0.40,
) -> tuple[float, float, float]:
    """5-point linear sub-grid in eps centered on eps_start, width
    +/- half_width_ratio * eps_start. tau held fixed at tau_start.

    Purpose (Day-13 fix): multi-start SLSQP from raw coarse-grid winners
    failed to resolve peak #2 at REFERENCE because no n=10 log grid point
    sits inside peak #2's 0.15-decade basin. A +/-40% linear sub-grid
    around a coarse-grid winner at eps=2.15e8 covers [1.29e8, 3.01e8],
    which includes eps=1.72e8 (within peak #2 basin, F ~ 0.988). SLSQP
    from the sub-grid best then refines into peak #2 at 1.51e8 cleanly.

    Returns (eps, tau, F) of the sub-grid best.
    """
    eps_lo = max(eps_start * (1.0 - half_width_ratio), eps_bounds[0])
    eps_hi = min(eps_start * (1.0 + half_width_ratio), eps_bounds[1])
    eps_sub = np.linspace(eps_lo, eps_hi, n_points)
    best_eps, best_tau, best_F = eps_start, tau_start, -1.0
    for e in eps_sub:
        try:
            F = _F_analytic_at(device, float(e), float(tau_start))
        except Exception:
            continue
        if F > best_F:
            best_eps, best_tau, best_F = float(e), float(tau_start), float(F)
    return best_eps, best_tau, best_F


def find_pareto_point(
    device: DeviceConfig,
    tau_max: float,
    epsilon_0_bounds: tuple[float, float] = (1e6, 1e9),
    tau_bounds: tuple[float, float] | None = None,
    n_warm_start_grid_side: int = 10,
    k_multi_start: int = 5,
) -> ParetoPoint:
    """Find (eps_0, tau) that maximize F_assign subject to tau <= tau_max.

    1. n_warm_start_grid_side x n_warm_start_grid_side warm-start grid
       (log-spaced on eps, linear on tau).
    2. Select top-k_multi_start diverse starting points by F (symmetric-
       ratio min-separation filter: 1.2x eps, 1.1x tau).
    3. Run SLSQP local refinement against -F from each top-K start; take
       the argmax F across the K refinements.
    4. Analytic binomial SE on the converged F_opt.
    All F evaluations use noise_model='analytic' (amended Q8 contract).

    Solver topology (Day-13 Amendment #10):
      - 5-point linear warm-start structurally missed the basin at
        eps ~ 1.59e8 (SLSQP trapped on the descending tail past the
        global argmax). n=10 log-spacing alone did not resolve it: a
        1D F(eps) scan at REFERENCE, tau=500ns exposed a two-peak
        structure (global peak ~1.51e8, secondary ~7.76e7, separated
        by a sharp valley) — no grid density alone crosses a local
        maximum from a gradient method.
      - Multi-start SLSQP with K=5 diverse starts resolves this.
        K=5 (not K=3) is the risk-averse choice against unknown
        additional local maxima in the 2D (eps, tau) surface.
      - Verified across 3 devices (REFERENCE + T_1 extremes [5.4, 91.9]
        us from the 50-device recovery harness at SEED=42): peak #2
        at eps ~ 1.51e8 is the global argmax for all three, with
        peak #1 at eps ~ 7.76e7 the secondary.
      - Full diagnostic: docs/module4_diagnostics/warm_start_grid_bug.md.
    """
    if tau_bounds is None:
        tau_bounds = (50e-9, tau_max)

    all_points = _warm_start_grid_all(
        device, epsilon_0_bounds, tau_bounds, n_side=n_warm_start_grid_side,
    )
    if not all_points:
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

    top_k_starts = _select_top_k_diverse(
        all_points, k=k_multi_start, min_eps_ratio=1.2,
    )

    def neg_F(x: np.ndarray) -> float:
        return -_F_analytic_at(device, x[0], x[1])

    best_res = None
    best_x: tuple[float, float] | None = None
    for eps_start, tau_start, _ in top_k_starts:
        # Sub-grid refinement around the coarse-grid start: 5-point linear
        # eps sweep +/- 40%, tau held fixed. This pulls SLSQP into the
        # nearest basin's interior even when the coarse grid didn't
        # sample it directly (Day-13 fix: n=10 log grid misses peak #2's
        # 0.15-decade basin; sub-grid lands inside).
        refined_eps, refined_tau, _ = _refine_eps_sub_grid(
            device, eps_start, tau_start, epsilon_0_bounds,
            n_points=5, half_width_ratio=0.40,
        )
        res = minimize(
            neg_F,
            x0=np.array([refined_eps, refined_tau]),
            method="SLSQP",
            bounds=[epsilon_0_bounds, tau_bounds],
            options={"ftol": 1e-6, "maxiter": 80},
        )
        # Pick argmax F across K refinements (F = -res.fun since we minimize -F).
        if best_res is None or (-res.fun) > (-best_res.fun):
            best_res = res
            best_x = (float(res.x[0]), float(res.x[1]))

    # best_res cannot be None here because all_points was non-empty and
    # every top-K start produced a minimize() result (SLSQP always returns
    # something, even if unconverged).
    assert best_res is not None and best_x is not None
    eps_opt = float(np.clip(best_x[0], *epsilon_0_bounds))
    tau_opt = float(np.clip(best_x[1], *tau_bounds))
    F_opt = float(-best_res.fun)
    res = best_res   # name re-bound so the rest of the function reads unchanged

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


# ────────────────────────────────────────────────────────────────────
# Spec §3.3 — batch frontier computation with Modal dispatch
# ────────────────────────────────────────────────────────────────────

def compute_pareto_frontier(
    device: DeviceConfig,
    tau_max_values: np.ndarray | None = None,
    device_label: str = "<unnamed>",
    use_modal: bool = False,
) -> list[ParetoPoint]:
    """Trace one device's Pareto frontier across tau_max values.

    Parameters
    ----------
    tau_max_values : np.ndarray, optional
        Defaults to TAU_MAX_GRID_NS * 1e-9 (10 log-spaced points, 100 ns - 2 µs).
    device_label : str
        Human-readable label; stamped onto each ParetoPoint.device_label.
    use_modal : bool
        If True, dispatch via modal_pareto.pareto_one_tuple.map(...).
        If False (default), run serial list(map(...)).

    Returns
    -------
    list[ParetoPoint], ordered by tau_max ascending.
    """
    if tau_max_values is None:
        tau_max_values = TAU_MAX_GRID_NS * 1e-9
    tau_max_list = [float(t) for t in tau_max_values]

    if use_modal:
        from .modal_pareto import app, pareto_one_tuple
        with app.run():
            results = list(pareto_one_tuple.map(
                [device] * len(tau_max_list), tau_max_list,
            ))
    else:
        results = [find_pareto_point(device, t) for t in tau_max_list]

    # Stamp the human-readable label
    labeled = []
    for p in results:
        labeled.append(ParetoPoint(
            device_id=p.device_id,
            device_label=device_label,
            tau_max=p.tau_max,
            epsilon_0_opt=p.epsilon_0_opt,
            tau_opt=p.tau_opt,
            F_assign_opt=p.F_assign_opt,
            F_assign_uncertainty=p.F_assign_uncertainty,
            dominant_loss_channel=p.dominant_loss_channel,
            solver_converged=p.solver_converged,
        ))
    return labeled
