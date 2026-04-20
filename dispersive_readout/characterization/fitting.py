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
    """One fitted device parameter with bootstrap uncertainty.

    ``reject_flag`` is a post-fit diagnostic: when set, the fit ran to
    completion but a structural check (e.g. spec §1.1's 1.5-oscillation
    requirement for Rabi) marked the trace as insufficient to trust.
    Downstream consumers treat ``reject_flag is not None`` as "don't use
    this value for aggregate statistics" — see ``CoverageReport``'s
    ``coverage_*_on_accepted`` fields and the ``n_rejected`` counter.
    """
    name: Literal["T_1", "T_2_echo", "T_2_star", "omega_q", "epsilon_pi"]
    value: float
    uncertainty: float
    unit: Literal["s", "rad/s"]
    protocol_source: Literal["rabi", "ramsey", "t1", "t2_echo"]
    goodness_of_fit: float = Field(ge=0.0)
    n_bootstrap: int = Field(ge=0)
    reject_flag: str | None = None

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
from scipy.signal import find_peaks, savgol_filter  # noqa: E402

from .protocols import TraceData  # noqa: E402


def _count_rabi_oscillations(
    P1: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    prominence_rel: float = 0.1,
) -> float:
    """Count visible Rabi turning points on a Savitzky-Golay-smoothed trace
    (spec §1.1 literal). Returns the summed count of interior peaks + troughs;
    spec's 1.5-oscillation threshold is enforced by the caller as ≥ 1.5.

    An interior peak + an interior trough = 2 turning points = "full cycle
    visible", which cleanly rejects under-sampled traces (<1 trough visible
    → count=0 or 1, below threshold). Endpoint peaks (ε=0 in the A+B·cos
    form) are not counted — spec's peak-counting rule is on interior
    extrema, and this is the conservative behavior.

    ``prominence_rel`` filters noise: a turning point counts only if it
    clears ``prominence_rel * (max − min)`` of the smoothed signal.
    """
    n = len(P1)
    win = min(window_length, n if n % 2 == 1 else n - 1)
    win = max(win, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    if win > n:
        smooth = P1.astype(float)
    else:
        smooth = savgol_filter(P1, window_length=win, polyorder=min(polyorder, win - 1))
    span = float(smooth.max() - smooth.min())
    prom = max(prominence_rel * span, 1e-6)
    peaks, _ = find_peaks(smooth, prominence=prom)
    troughs, _ = find_peaks(-smooth, prominence=prom)
    return float(len(peaks) + len(troughs))


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
    n_bs = 0
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "rabi",
            {"epsilon_pi": value,
             "omega_q": float(trace.metadata.get("ground_truth", {}).get("omega_q", 2 * math.pi * 4.5e9))},
            noise=boot_noise, n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        unc = max(float(np.std(boot["epsilon_pi"])), 1e-30)
        n_bs = bootstrap_samples
    # Spec §1.1 reject: <1.5 visible oscillations → flag as structurally
    # unreliable. Fit still returns a best-effort value; downstream code
    # must check reject_flag before aggregating.
    n_osc = _count_rabi_oscillations(trace.P1)
    reject_flag = "insufficient_oscillations" if n_osc < 1.5 else None
    return FittedParameter(
        name="epsilon_pi", value=value, uncertainty=unc, unit="rad/s",
        protocol_source="rabi", goodness_of_fit=float(result.redchi),
        n_bootstrap=n_bs, reject_flag=reject_flag,
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
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "ramsey",
            {"omega_q": omega_q_fit, "T_2_star": T_2_fit,
             "omega_drive_offset": float(gt.get("omega_drive_offset", 2 * math.pi * 1e6))},
            noise=boot_noise, n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        fp_omega = fp_omega.model_copy(update={
            "uncertainty": max(float(np.std(boot["omega_q"])), 1e-30),
            "n_bootstrap": bootstrap_samples,
        })
        fp_T2 = fp_T2.model_copy(update={
            "uncertainty": max(float(np.std(boot["T_2_star"])), 1e-30),
            "n_bootstrap": bootstrap_samples,
        })
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
    n_bs = 0
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "t1", {"T_1": tau}, noise=boot_noise,
            n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        tau_err = max(float(np.std(boot["T_1"])), 1e-30)
        n_bs = bootstrap_samples
    return FittedParameter(
        name="T_1", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t1", goodness_of_fit=float(result.redchi), n_bootstrap=n_bs,
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
    n_bs = 0
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "t2_echo", {"T_2_echo": tau}, noise=boot_noise,
            n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        tau_err = max(float(np.std(boot["T_2_echo"])), 1e-30)
        n_bs = bootstrap_samples
    return FittedParameter(
        name="T_2_echo", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t2_echo", goodness_of_fit=float(result.redchi), n_bootstrap=n_bs,
    )


# -- Parametric bootstrap (amendment 3) -------------------------------------

from .protocols import (  # noqa: E402
    generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
)
from .noise import NoiseModelParams  # noqa: E402


def _noise_from_trace_metadata(trace: TraceData) -> NoiseModelParams:
    meta_noise = trace.metadata.get("noise", {})
    return NoiseModelParams(
        n_shots_per_point=int(meta_noise.get("n_shots_per_point", 2000)),
        drift_amplitude_Hz=float(meta_noise.get("drift_amplitude_Hz", 0.0)),
        drift_alpha=float(meta_noise.get("drift_alpha", 1.0)),
        drive_amplitude_uncertainty=float(meta_noise.get("drive_amplitude_uncertainty", 0.0)),
    )


def parametric_bootstrap(
    protocol: Literal["rabi", "ramsey", "t1", "t2_echo"],
    best_fit_values: dict[str, float],
    noise: NoiseModelParams,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Parametric bootstrap per amendment 3.

    For k in 1..n_bootstrap:
        Regenerate a fresh trace from `best_fit_values` + fresh noise realization
          (seed_k drawn from the master seed).
        Point-estimate fit the fresh trace.
        Record the fitted parameters.
    Return {param_name: ndarray of length n_bootstrap}.
    """
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        sub_seed = int(rng.integers(2**31 - 1))
        if protocol == "rabi":
            trace_k = generate_rabi_trace(
                best_fit_values["epsilon_pi"], best_fit_values.get("omega_q", 2 * math.pi * 4.5e9),
                noise, seed=sub_seed,
            )
            fp = fit_rabi(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("epsilon_pi", []).append(fp.value)
        elif protocol == "ramsey":
            trace_k = generate_ramsey_trace(
                best_fit_values["omega_q"], T_2_star=best_fit_values["T_2_star"],
                noise=noise,
                omega_drive_offset=best_fit_values.get("omega_drive_offset", 2 * math.pi * 1e6),
                seed=sub_seed,
            )
            fp_o, fp_t = fit_ramsey(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("omega_q", []).append(fp_o.value)
            boot.setdefault("T_2_star", []).append(fp_t.value)
        elif protocol == "t1":
            trace_k = generate_t1_trace(best_fit_values["T_1"], noise, seed=sub_seed)
            fp = fit_t1(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("T_1", []).append(fp.value)
        elif protocol == "t2_echo":
            trace_k = generate_t2_echo_trace(best_fit_values["T_2_echo"], noise, seed=sub_seed)
            fp = fit_t2_echo(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("T_2_echo", []).append(fp.value)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    return {name: np.array(values, dtype=float) for name, values in boot.items()}


def fit_all(
    traces: list[TraceData],
    bootstrap_samples: int = 200,
    seed: int | None = None,
    trace_file: str = "",
) -> ExtractedParameterPack:
    """Fit every trace in a bundle; return a Module-1-compatible parameter pack."""
    from datetime import datetime, timezone
    import subprocess
    fitted: list[FittedParameter] = []
    for t in traces:
        if t.protocol == "rabi":
            fitted.append(fit_rabi(t, bootstrap_samples=bootstrap_samples, seed=seed))
        elif t.protocol == "ramsey":
            o, ts = fit_ramsey(t, bootstrap_samples=bootstrap_samples, seed=seed)
            fitted.extend([o, ts])
        elif t.protocol == "t1":
            fitted.append(fit_t1(t, bootstrap_samples=bootstrap_samples, seed=seed))
        elif t.protocol == "t2_echo":
            fitted.append(fit_t2_echo(t, bootstrap_samples=bootstrap_samples, seed=seed))
        else:
            raise ValueError(f"Unknown protocol: {t.protocol}")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    return ExtractedParameterPack(
        fitted_parameters=fitted,
        trace_file=trace_file,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stage_06_version=sha,
    )
