"""Stage 06 Module 3 Figure 3 — characterization pipeline + parameter recovery.

Layout (2×2):
  (a) Rabi fit + residuals
  (b) Ramsey fit + residuals
  (c) T1 decay + residuals
  (d) Parameter-recovery parity plot (2×2 of sub-panels: T1, T2, ω_q, ε_π),
      fitted vs ground truth with y=x line, colored by |z| ≤ 1, annotated
      with observed 2σ coverage + 2σ binomial CI.

Style-matched to Figures 1 and 2: 150 DPI, same palette, point-with-errorbar
convention on near-identity values.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.characterization.fitting import (
    fit_rabi, fit_ramsey, fit_t1,
)
from dispersive_readout.characterization.noise import NoiseModelParams
from dispersive_readout.characterization.protocols import (
    generate_rabi_trace, generate_ramsey_trace, generate_t1_trace,
)
from dispersive_readout.characterization.recovery import (
    load_committed_coverage_report, run_recovery_harness, fit_one_device,
)


_OUT = Path("06_Dispersive_Readout/figures/fig3_characterization.png")
_COMMITTED_REPORT = Path("06_Dispersive_Readout/figures/recovery_coverage_report.yaml")


def _panel_rabi(ax_fit, ax_res):
    noise = NoiseModelParams()
    eps_pi = 2 * math.pi * 50e6
    trace = generate_rabi_trace(eps_pi, 2 * math.pi * 4.5e9, noise, seed=42)
    fp = fit_rabi(trace, bootstrap_samples=50, seed=42)
    eps = trace.sweep_values
    ax_fit.errorbar(eps / (2 * math.pi * 1e6), trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    model_P = 0.5 + 0.5 * np.cos(np.pi * eps / fp.value)
    ax_fit.plot(eps / (2 * math.pi * 1e6), model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(a) Rabi — $\varepsilon_\pi/2\pi$ = {fp.value/(2*math.pi*1e6):.2f} MHz $\pm$ {fp.uncertainty/(2*math.pi*1e6):.2f} MHz, $\chi^2_\nu$={fp.goodness_of_fit:.2f}")
    ax_res.errorbar(eps / (2 * math.pi * 1e6), trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\varepsilon / 2\pi$ (MHz)")
    ax_res.set_ylabel("residual")


def _panel_ramsey(ax_fit, ax_res):
    noise = NoiseModelParams()
    omega_q = 2 * math.pi * 4.5e9
    T_2_star = 20e-6
    trace = generate_ramsey_trace(omega_q, T_2_star=T_2_star, noise=noise, seed=42)
    fp_o, fp_t = fit_ramsey(trace, bootstrap_samples=50, seed=42)
    delays = trace.sweep_values
    ax_fit.errorbar(delays * 1e6, trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    omega_drive = omega_q - trace.metadata["ground_truth"]["omega_drive_offset"]
    delta_omega = fp_o.value - omega_drive
    model_P = 0.5 - 0.5 * np.exp(-delays / fp_t.value) * np.cos(delta_omega * delays)
    ax_fit.plot(delays * 1e6, model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(b) Ramsey — $\omega_q/2\pi$={fp_o.value/(2*math.pi*1e9):.4f} GHz, $T_2^*$={fp_t.value*1e6:.1f} $\pm$ {fp_t.uncertainty*1e6:.1f} µs")
    ax_res.errorbar(delays * 1e6, trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\tau$ (µs)")
    ax_res.set_ylabel("residual")


def _panel_t1(ax_fit, ax_res):
    noise = NoiseModelParams()
    T_1 = 30e-6
    trace = generate_t1_trace(T_1, noise, seed=42)
    fp = fit_t1(trace, bootstrap_samples=50, seed=42)
    delays = trace.sweep_values
    ax_fit.errorbar(delays * 1e6, trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    model_P = np.exp(-delays / fp.value)
    ax_fit.plot(delays * 1e6, model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(c) T1 — T$_1$ = {fp.value*1e6:.2f} $\pm$ {fp.uncertainty*1e6:.2f} µs, $\chi^2_\nu$={fp.goodness_of_fit:.2f}")
    ax_res.errorbar(delays * 1e6, trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\tau$ (µs)")
    ax_res.set_ylabel("residual")


def _panel_recovery(fig, gs_slot):
    """Build a 2×2 of parity sub-panels inside the outer gridspec slot."""
    sub = gs_slot.subgridspec(2, 2, hspace=0.45, wspace=0.40)
    noise = NoiseModelParams()
    obs_reports, devices = run_recovery_harness(n_devices=50, noise=noise, seed=42)
    pairs = {"T_1": [], "T_2_echo": [], "omega_q": [], "epsilon_pi": []}
    rng = np.random.default_rng(42)
    for d in devices:
        sub_seed = int(rng.integers(2**31 - 1))
        for r in fit_one_device(d, noise, seed=sub_seed):
            pairs[r.parameter_name].append((r.ground_truth, r.fitted_value, r.fitted_uncertainty, r.within_1_sigma))
    param_order = ["T_1", "T_2_echo", "omega_q", "epsilon_pi"]
    units = {"T_1": ("µs", 1e6), "T_2_echo": ("µs", 1e6),
             "omega_q": ("GHz", 1.0 / (2 * math.pi * 1e9)),
             "epsilon_pi": ("MHz", 1.0 / (2 * math.pi * 1e6))}
    for i, name in enumerate(param_order):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        lab, scale = units[name]
        x = np.array([p[0] * scale for p in pairs[name]])
        y = np.array([p[1] * scale for p in pairs[name]])
        yerr = np.array([p[2] * scale for p in pairs[name]])
        cov1 = np.array([p[3] for p in pairs[name]])
        ax.errorbar(x[cov1], y[cov1], yerr=yerr[cov1], fmt="o", ms=3, color="tab:blue", label="|z|≤1", capsize=0, alpha=0.6)
        ax.errorbar(x[~cov1], y[~cov1], yerr=yerr[~cov1], fmt="x", ms=4, color="tab:orange", label="|z|>1", capsize=0, alpha=0.7)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
        cov2 = obs_reports[name].coverage_2_sigma
        ci = (obs_reports[name].coverage_2_sigma_ci_low, obs_reports[name].coverage_2_sigma_ci_high)
        ax.set_title(rf"{name}: 2$\sigma$={cov2:.0%} [{ci[0]:.0%},{ci[1]:.0%}]", fontsize=8)
        ax.set_xlabel(f"truth ({lab})", fontsize=8)
        ax.set_ylabel(f"fit ({lab})", fontsize=8)
        ax.tick_params(labelsize=7)


def main() -> None:
    fig = plt.figure(figsize=(12, 9), dpi=150)
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    ga = outer[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_rabi(fig.add_subplot(ga[0]), fig.add_subplot(ga[1]))
    gb = outer[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_ramsey(fig.add_subplot(gb[0]), fig.add_subplot(gb[1]))
    gc = outer[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_t1(fig.add_subplot(gc[0]), fig.add_subplot(gc[1]))
    _panel_recovery(fig, outer[1, 1])
    fig.suptitle("Figure 3 — Characterization pipeline + 50-device parameter recovery (SEED=42)", fontsize=11)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, bbox_inches="tight", dpi=150)
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
