"""Module 3 — parameter recovery harness (G2).

Built around the pure function `fit_one_device(device, noise, seed) → list[RecoveryResult]`
per amendment 8; serial fallback is list(map(...)), Modal mode would be
fit_one_device.map(...). The harness aggregates a CoverageReport per
parameter, with the 2σ binomial CI required by amendment 4.

Amendment 9: the committed recovery_coverage_report.yaml pins the
device list alongside the coverage statistics, so the artifact is
self-describing under numpy default_rng changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .fitting import FittedParameter, fit_rabi, fit_ramsey, fit_t1, fit_t2_echo
from .noise import NoiseModelParams
from .protocols import (
    generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
)


@dataclass(frozen=True)
class DeviceGroundTruth:
    """One synthetic device's ground truth."""
    T_1: float
    T_2_echo: float
    omega_q: float
    epsilon_pi: float
    thermal_offset: float = 0.0
    ramsey_detuning: float = 2.0 * math.pi * 1e6


@dataclass(frozen=True)
class RecoveryResult:
    parameter_name: str
    ground_truth: float
    fitted_value: float
    fitted_uncertainty: float
    z_score: float
    within_1_sigma: bool
    within_2_sigma: bool


@dataclass(frozen=True)
class CoverageReport:
    parameter_name: str
    n_devices: int
    coverage_1_sigma: float
    coverage_2_sigma: float
    coverage_1_sigma_ci_low: float
    coverage_1_sigma_ci_high: float
    coverage_2_sigma_ci_low: float
    coverage_2_sigma_ci_high: float
    bias: float
    bias_uncertainty: float


def _binomial_2sigma_ci(p: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return max(0.0, p - 2.0 * se), min(1.0, p + 2.0 * se)


def _make_recovery_result(param_name: str, truth: float, fp: FittedParameter) -> RecoveryResult:
    unc = max(fp.uncertainty, 1e-30)
    z = (fp.value - truth) / unc
    return RecoveryResult(
        parameter_name=param_name,
        ground_truth=float(truth),
        fitted_value=float(fp.value),
        fitted_uncertainty=float(unc),
        z_score=float(z),
        within_1_sigma=abs(z) <= 1.0,
        within_2_sigma=abs(z) <= 2.0,
    )


def fit_one_device(
    device: DeviceGroundTruth,
    noise: NoiseModelParams,
    seed: int,
) -> list[RecoveryResult]:
    """Pure function: generate 4 traces, fit, compare to truth (amendment 8)."""
    rng = np.random.default_rng(seed)
    rabi_seed = int(rng.integers(2**31 - 1))
    ramsey_seed = int(rng.integers(2**31 - 1))
    t1_seed = int(rng.integers(2**31 - 1))
    t2_seed = int(rng.integers(2**31 - 1))
    fit_seeds = [int(rng.integers(2**31 - 1)) for _ in range(4)]

    rabi_trace = generate_rabi_trace(device.epsilon_pi, device.omega_q, noise, seed=rabi_seed)
    ramsey_trace = generate_ramsey_trace(
        device.omega_q, T_2_star=device.T_2_echo, noise=noise,
        omega_drive_offset=device.ramsey_detuning, seed=ramsey_seed,
    )
    t1_trace = generate_t1_trace(device.T_1, noise, thermal_offset=device.thermal_offset, seed=t1_seed)
    t2_trace = generate_t2_echo_trace(device.T_2_echo, noise, seed=t2_seed)

    fp_eps = fit_rabi(rabi_trace, bootstrap_samples=200, seed=fit_seeds[0])
    fp_omega, _fp_T2star = fit_ramsey(ramsey_trace, bootstrap_samples=200, seed=fit_seeds[1])
    fp_T1 = fit_t1(t1_trace, bootstrap_samples=200, seed=fit_seeds[2])
    fp_T2 = fit_t2_echo(t2_trace, bootstrap_samples=200, seed=fit_seeds[3])

    return [
        _make_recovery_result("T_1", device.T_1, fp_T1),
        _make_recovery_result("T_2_echo", device.T_2_echo, fp_T2),
        _make_recovery_result("omega_q", device.omega_q, fp_omega),
        _make_recovery_result("epsilon_pi", device.epsilon_pi, fp_eps),
    ]


def generate_synthetic_device_family(n_devices: int, seed: int) -> list[DeviceGroundTruth]:
    """Log-uniform(T_1, T_2_echo) in [5 µs, 100 µs]; uniform(ω_q/2π) in [4 GHz, 5 GHz].

    Physical constraint: T_2_echo ≤ 2·T_1·0.95 (Hahn echo bounded above by 2T_1,
    with 0.95 margin for bootstrap fluctuations).

    Deterministic overrides (not subject to sampling):
      device[0]: ramsey_detuning = 0 (zero-detuning edge case, C6a)
      device[1]: thermal_offset = 0.08 (elevated-thermal edge case, C6b)
    """
    rng = np.random.default_rng(seed)
    out: list[DeviceGroundTruth] = []

    out.append(DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6,
        omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
        thermal_offset=0.0,
        ramsey_detuning=0.0,
    ))
    out.append(DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6,
        omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
        thermal_offset=0.08,
        ramsey_detuning=2 * math.pi * 1e6,
    ))

    log_lo = math.log(5e-6)
    log_hi = math.log(100e-6)
    while len(out) < n_devices:
        T_1 = math.exp(rng.uniform(log_lo, log_hi))
        T_2 = math.exp(rng.uniform(log_lo, log_hi))
        if T_2 > 2.0 * T_1 * 0.95:
            continue
        omega_q = 2 * math.pi * rng.uniform(4e9, 5e9)
        epsilon_pi = 2 * math.pi * 50e6 * (1.0 + 0.2 * rng.standard_normal())
        out.append(DeviceGroundTruth(
            T_1=T_1, T_2_echo=T_2, omega_q=omega_q,
            epsilon_pi=epsilon_pi,
            thermal_offset=0.0,
            ramsey_detuning=2 * math.pi * 1e6,
        ))
    return out


import yaml  # noqa: E402


def run_recovery_harness(
    n_devices: int = 50,
    noise: NoiseModelParams | None = None,
    seed: int = 42,
) -> tuple[dict[str, CoverageReport], list[DeviceGroundTruth]]:
    """Run the full harness at the given seed; return (reports, devices)."""
    if noise is None:
        noise = NoiseModelParams()
    devices = generate_synthetic_device_family(n_devices=n_devices, seed=seed)
    rng = np.random.default_rng(seed)
    results_by_param: dict[str, list[RecoveryResult]] = {
        "T_1": [], "T_2_echo": [], "omega_q": [], "epsilon_pi": [],
    }
    for d in devices:
        sub_seed = int(rng.integers(2**31 - 1))
        for r in fit_one_device(d, noise, seed=sub_seed):
            results_by_param[r.parameter_name].append(r)

    reports: dict[str, CoverageReport] = {}
    for name, records in results_by_param.items():
        n = len(records)
        cov1 = sum(r.within_1_sigma for r in records) / n
        cov2 = sum(r.within_2_sigma for r in records) / n
        c1_lo, c1_hi = _binomial_2sigma_ci(cov1, n)
        c2_lo, c2_hi = _binomial_2sigma_ci(cov2, n)
        diffs = np.array([r.fitted_value - r.ground_truth for r in records])
        bias = float(diffs.mean())
        bias_unc = float(diffs.std(ddof=1) / math.sqrt(n))
        reports[name] = CoverageReport(
            parameter_name=name,
            n_devices=n,
            coverage_1_sigma=cov1,
            coverage_2_sigma=cov2,
            coverage_1_sigma_ci_low=c1_lo,
            coverage_1_sigma_ci_high=c1_hi,
            coverage_2_sigma_ci_low=c2_lo,
            coverage_2_sigma_ci_high=c2_hi,
            bias=bias,
            bias_uncertainty=bias_unc,
        )
    return reports, devices


def save_coverage_report(
    reports: dict[str, CoverageReport],
    devices: list[DeviceGroundTruth],
    path: str | Path,
    seed: int,
) -> None:
    """Serialize the coverage report + device list (for RNG stability)."""
    payload = {
        "seed": seed,
        "n_devices": len(devices),
        "coverage": {name: asdict(rep) for name, rep in reports.items()},
        "devices": [asdict(d) for d in devices],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_committed_coverage_report(path: str | Path) -> dict[str, CoverageReport]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        name: CoverageReport(**rec)
        for name, rec in data["coverage"].items()
    }


def format_recovery_table(reports: dict[str, CoverageReport]) -> str:
    lines = [
        "| Parameter | Cov 1σ (target 68%) | 2σ CI | Cov 2σ (target 95%) | 2σ CI | Bias |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        lines.append(
            f"| `{name}` | {r.coverage_1_sigma:.1%} | "
            f"[{r.coverage_1_sigma_ci_low:.1%}, {r.coverage_1_sigma_ci_high:.1%}] | "
            f"{r.coverage_2_sigma:.1%} | "
            f"[{r.coverage_2_sigma_ci_low:.1%}, {r.coverage_2_sigma_ci_high:.1%}] | "
            f"{r.bias:+.3e} ± {r.bias_uncertainty:.1e} |"
        )
    return "\n".join(lines)
