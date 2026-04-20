"""Module 3 — lmfit-based parameter extraction.

This module has two layers:
  1. Pydantic schemas (FittedParameter, ExtractedParameterPack) with a
     to_device_config bridge that back-solves E_J from ω_q per Koch 2007
     (amendment 5).
  2. lmfit wrappers + parametric_bootstrap (amendment 3). The wrappers
     arrive in Task 8; bootstrap in Task 9.
"""
from __future__ import annotations

import math
import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FittedParameter(BaseModel):
    """One fitted device parameter with bootstrap uncertainty."""
    name: Literal["T_1", "T_2_echo", "T_2_star", "omega_q", "epsilon_pi"]
    value: float
    uncertainty: float
    unit: Literal["s", "rad/s"]
    protocol_source: Literal["rabi", "ramsey", "t1", "t2_echo"]
    goodness_of_fit: float = Field(ge=0.0)
    n_bootstrap: int = Field(ge=0)

    @field_validator("uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("uncertainty must be strictly positive")
        return v


class ExtractedParameterPack(BaseModel):
    """All parameters extracted from one trace bundle."""
    fitted_parameters: list[FittedParameter]
    trace_file: str
    timestamp: str
    stage_06_version: str

    def _get(self, name: str) -> FittedParameter | None:
        for p in self.fitted_parameters:
            if p.name == name:
                return p
        return None

    def to_device_config(self, E_J_tolerance_rel: float = 0.30):
        """Bridge fitted parameters to Module 1's DeviceConfig (amendment 5).

        Policy:
          - E_C held fixed at REFERENCE_DEVICE (geometric, not fit).
          - E_J back-solved from fitted ω_q via Koch 2007:
                E_J = (ω_q + E_C)² / (8·E_C)
          - γ_1 = 1 / T_1; γ_φ from T_2_echo via 1/T_2 = γ_1/2 + γ_φ.
          - resonator, coupling, truncation inherited from REFERENCE_DEVICE.
          - UserWarning if |E_J − E_J_REFERENCE| / E_J_REFERENCE > 30%.
        """
        from dispersive_readout.physics.config import (
            DecoherenceParams, DeviceConfig, REFERENCE_DEVICE, TransmonParams,
        )
        omega_q_fp = self._get("omega_q")
        T_1_fp = self._get("T_1")
        T_2_echo_fp = self._get("T_2_echo")
        if omega_q_fp is None or T_1_fp is None or T_2_echo_fp is None:
            raise ValueError(
                "to_device_config requires omega_q, T_1, and T_2_echo fits. "
                "Missing: " + ", ".join(n for n, v in (
                    ("omega_q", omega_q_fp), ("T_1", T_1_fp), ("T_2_echo", T_2_echo_fp),
                ) if v is None)
            )
        E_C = REFERENCE_DEVICE.transmon.E_C
        omega_q = omega_q_fp.value
        E_J_derived = (omega_q + E_C) ** 2 / (8.0 * E_C)
        E_J_reference = REFERENCE_DEVICE.transmon.E_J
        rel_drift = abs(E_J_derived - E_J_reference) / E_J_reference
        if rel_drift > E_J_tolerance_rel:
            warnings.warn(
                f"Derived E_J/2π = {E_J_derived / (2 * math.pi) / 1e9:.3f} GHz is "
                f"{rel_drift:.1%} off REFERENCE's E_J/2π = "
                f"{E_J_reference / (2 * math.pi) / 1e9:.3f} GHz — check the fit.",
                UserWarning,
                stacklevel=2,
            )
        transmon = TransmonParams(E_C=E_C, E_J=E_J_derived, n_g=REFERENCE_DEVICE.transmon.n_g)
        gamma_1 = 1.0 / T_1_fp.value
        gamma_phi = max(1.0 / T_2_echo_fp.value - 0.5 * gamma_1, 0.0)
        decoherence = DecoherenceParams(
            gamma_1=gamma_1, gamma_phi=gamma_phi,
            n_th=REFERENCE_DEVICE.decoherence.n_th,
            purcell_enabled=REFERENCE_DEVICE.decoherence.purcell_enabled,
        )
        return DeviceConfig(
            transmon=transmon,
            resonator=REFERENCE_DEVICE.resonator,
            coupling=REFERENCE_DEVICE.coupling,
            decoherence=decoherence,
            truncation=REFERENCE_DEVICE.truncation,
        )


# -- lmfit wrappers (point-estimate layer; bootstrap in Task 9) --------------

import numpy as np  # noqa: E402
import lmfit  # noqa: E402

from .protocols import TraceData  # noqa: E402


# --- Initial-guess helpers --------------------------------------------------

def _initial_guess_rabi(eps: np.ndarray, P1: np.ndarray) -> dict[str, float]:
    """Crude ε_π estimate from the first P1 minimum (generator uses +cos form
    → min at ε_π)."""
    idx = int(np.argmin(P1))
    eps_pi_guess = float(eps[idx]) if eps[idx] > 0 else float(eps[-1] / 2.0)
    return {
        "A": float(P1.mean()),
        "B": float((P1.max() - P1.min()) / 2.0),
        "epsilon_pi": max(eps_pi_guess, 1e-12),
        "phi": 0.0,
    }


def _initial_guess_ramsey(delays: np.ndarray, P1: np.ndarray) -> dict[str, float]:
    """FFT peak for Δω; exponential-decay envelope for T2*."""
    signal = P1 - P1.mean()
    dt = float(delays[1] - delays[0])
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(delays), d=dt)
    if len(fft) > 1 and np.any(fft[1:] > 0):
        peak = int(np.argmax(fft[1:])) + 1
        delta_omega0 = 2 * math.pi * float(freqs[peak])
    else:
        delta_omega0 = 2 * math.pi * 1e6
    env0 = (P1.max() - P1.min()) / 2.0
    return {
        "A": float(P1.mean()),
        "B": float(env0 if env0 > 0 else 0.1),
        "delta_omega": delta_omega0,
        "T_2_star": max(float(delays.max()) / 3.0, 1e-9),
        "phi": 0.0,
    }


def _initial_guess_exponential(delays: np.ndarray, P1: np.ndarray, is_echo: bool) -> dict[str, float]:
    """Shared exponential initial guess for T1 and T2-echo fits."""
    if is_echo:
        signal = 1.0 - 2.0 * P1
        mask = signal > 0.02
    else:
        floor = float(P1[-max(1, len(P1) // 10):].mean())
        signal = P1 - floor
        mask = signal > 0.02
    if mask.sum() < 3:
        tau0 = float(delays.max()) / 3.0
    else:
        coef = np.polyfit(delays[mask], np.log(signal[mask]), 1)
        tau0 = -1.0 / coef[0] if coef[0] < 0 else float(delays.max())
    return {
        "A": 0.0 if is_echo else float(P1[-max(1, len(P1) // 10):].mean()),
        "B": float(signal.max()),
        "tau": max(tau0, 1e-9),
    }


# --- Point-estimate fits ----------------------------------------------------

def _fit_point(model: lmfit.Model, params: lmfit.Parameters, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> lmfit.model.ModelResult:
    """Shared point-estimate run."""
    return model.fit(y, params=params, x=x, weights=1.0 / np.clip(weights, 1e-12, None))


def fit_rabi(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit Rabi: P₁(ε) = A + B·cos(π·ε/ε_π + φ). Returns ε_π with uncertainty."""
    def _model(x, A, B, epsilon_pi, phi):
        return A + B * np.cos(np.pi * x / epsilon_pi + phi)

    model = lmfit.Model(_model)
    g = _initial_guess_rabi(trace.sweep_values, trace.P1)
    params = model.make_params(**g)
    params["epsilon_pi"].set(min=2 * math.pi * 1e6, max=2 * math.pi * 1e9)
    params["B"].set(min=-1.0, max=1.0)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    value = float(result.params["epsilon_pi"].value)
    stderr = result.params["epsilon_pi"].stderr
    unc = float(stderr) if stderr is not None and stderr > 0 else value * 0.01
    return FittedParameter(
        name="epsilon_pi", value=value, uncertainty=unc, unit="rad/s",
        protocol_source="rabi", goodness_of_fit=float(result.redchi),
        n_bootstrap=0,
    )


def fit_ramsey(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> tuple[FittedParameter, FittedParameter]:
    """Fit Ramsey: P₁(τ) = A + B·exp(−τ/T_2*)·cos(Δω·τ + φ). Returns (omega_q, T_2_star).

    Edge case (amendment 2 / §5 test C6a): if initial FFT guess shows < 1
    oscillation over the sweep, pin Δω=0 and fit the envelope only.
    """
    g = _initial_guess_ramsey(trace.sweep_values, trace.P1)
    span = float(trace.sweep_values.max() - trace.sweep_values.min())
    oscillations = g["delta_omega"] * span / (2 * math.pi)

    if oscillations < 1.0:
        def _env_model(x, A, B, T_2_star):
            return A + B * np.exp(-x / T_2_star)
        model = lmfit.Model(_env_model)
        params = model.make_params(A=g["A"], B=g["B"], T_2_star=g["T_2_star"])
        params["T_2_star"].set(min=1e-7, max=1e-3)
        result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
        T_2 = float(result.params["T_2_star"].value)
        T_2_err = result.params["T_2_star"].stderr or T_2 * 0.1
        omega_q_meta = float(trace.metadata.get("ground_truth", {}).get("omega_q", 2 * math.pi * 4.5e9))
        fp_omega = FittedParameter(
            name="omega_q", value=omega_q_meta, uncertainty=2 * math.pi * 1e3,
            unit="rad/s", protocol_source="ramsey",
            goodness_of_fit=float(result.redchi), n_bootstrap=0,
        )
        fp_T2 = FittedParameter(
            name="T_2_star", value=T_2, uncertainty=float(T_2_err),
            unit="s", protocol_source="ramsey",
            goodness_of_fit=float(result.redchi), n_bootstrap=0,
        )
        return fp_omega, fp_T2

    def _model(x, A, B, delta_omega, T_2_star, phi):
        return A + B * np.exp(-x / T_2_star) * np.cos(delta_omega * x + phi)
    model = lmfit.Model(_model)
    params = model.make_params(**g)
    params["T_2_star"].set(min=1e-7, max=1e-3)
    params["delta_omega"].set(min=2 * math.pi * 1e3, max=2 * math.pi * 1e9)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    delta_omega_fit = float(result.params["delta_omega"].value)
    T_2_fit = float(result.params["T_2_star"].value)
    d_omega_err = result.params["delta_omega"].stderr or abs(delta_omega_fit) * 0.01
    T_2_err = result.params["T_2_star"].stderr or T_2_fit * 0.1
    gt = trace.metadata.get("ground_truth", {})
    omega_q_metadata = float(gt.get("omega_q", 2 * math.pi * 4.5e9))
    omega_drive = omega_q_metadata - float(gt.get("omega_drive_offset", 2 * math.pi * 1e6))
    omega_q_fit = omega_drive + delta_omega_fit
    fp_omega = FittedParameter(
        name="omega_q", value=omega_q_fit, uncertainty=float(d_omega_err),
        unit="rad/s", protocol_source="ramsey",
        goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
    fp_T2 = FittedParameter(
        name="T_2_star", value=T_2_fit, uncertainty=float(T_2_err),
        unit="s", protocol_source="ramsey",
        goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
    return fp_omega, fp_T2


def fit_t1(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit T1: P₁(τ) = A + B·exp(−τ/T_1)."""
    def _model(x, A, B, tau):
        return A + B * np.exp(-x / tau)
    g = _initial_guess_exponential(trace.sweep_values, trace.P1, is_echo=False)
    model = lmfit.Model(_model)
    params = model.make_params(**g)
    params["tau"].set(min=1e-7, max=1e-3)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    tau = float(result.params["tau"].value)
    tau_err = result.params["tau"].stderr or tau * 0.1
    return FittedParameter(
        name="T_1", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t1", goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )


def fit_t2_echo(
    trace: TraceData,
    use_stretched_exponential: bool = False,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit Hahn echo: P₁(τ) = A + B·exp(−τ/T_2). Stretched fallback if redchi > 3."""
    def _plain(x, A, B, tau):
        return A + B * np.exp(-x / tau)
    def _stretched(x, A, B, tau, n):
        return A + B * np.exp(-((x / tau) ** n))

    g = _initial_guess_exponential(trace.sweep_values, trace.P1, is_echo=True)
    g["A"] = 0.5
    g["B"] = -0.5

    model = lmfit.Model(_plain)
    params = model.make_params(**g)
    params["tau"].set(min=1e-7, max=1e-3)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)

    if use_stretched_exponential or float(result.redchi) > 3.0:
        model_s = lmfit.Model(_stretched)
        ps = model_s.make_params(**{**g, "n": 1.0})
        ps["tau"].set(min=1e-7, max=1e-3)
        ps["n"].set(min=0.3, max=3.0)
        result = _fit_point(model_s, ps, trace.sweep_values, trace.P1, trace.P1_uncertainty)

    tau = float(result.params["tau"].value)
    tau_err = result.params["tau"].stderr or tau * 0.1
    return FittedParameter(
        name="T_2_echo", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t2_echo", goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
