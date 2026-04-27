"""Module 5a driver — fig5a: DRAG leakage suppression on REFERENCE_DEVICE.

Panel (a): trajectories P_0(t), P_1(t), P_{≥2}(t) at T_gate=20ns, σ=5ns
           for no-DRAG, β=1, β=β_opt (fidelity-optimal per spec §5.3 post-N11).
Panel (b): speed-leakage tradeoff — final + peak leakage vs T_gate.
           Inset 1: ε_X(T_gate) under full Lindblad.
           Inset 2: V2b leakage-vs-fidelity trade-off — β_opt_fidelity,
                    β_min_final_leak, β_min_peak_leak vs T_gate.

Output:
    06_Dispersive_Readout/figures/fig5a_drag_leakage.png  (2-panel + 2 insets)
    06_Dispersive_Readout/figures/fig5a_drag_leakage_data.yaml
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import yaml

from dispersive_readout.analysis.gate_metrics import (
    epsilon_x_from_transfer,
    leakage_peak,
    leakage_population,
    transfer_fidelity_0_to_1,
)
from dispersive_readout.control.drag_calibration import calibrate_drag_beta
from dispersive_readout.control.gate_simulator import simulate_x_gate
from dispersive_readout.physics.config import REFERENCE_DEVICE, DecoherenceParams
from dispersive_readout.physics.transmon import transmon_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "06_Dispersive_Readout" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

T_GATE_SWEEP_NS = np.array([5, 7, 10, 15, 20, 25, 30, 40, 50])


def _zero_decoherence() -> DecoherenceParams:
    return DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)


def _populations_from_rho_t(rho_t, n_levels):
    """Return (P_0, P_1, P_{≥2}) trajectories as np.ndarrays of shape (T,)."""
    P0 = np.array([float((qt.basis(n_levels, 0).proj() * r).tr().real) for r in rho_t])
    P1 = np.array([float((qt.basis(n_levels, 1).proj() * r).tr().real) for r in rho_t])
    P_leak = np.array([leakage_population(r, n_levels) for r in rho_t])
    return P0, P1, P_leak


def _panel_a(ax_top, ax_mid, ax_bot, T_gate=20e-9, n_levels=4):
    """Panel (a): trajectories for no-DRAG, β=1, β=β_opt at T_gate=20ns."""
    sigma = T_gate / 4.0
    decoh_zero = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE, T_gate=T_gate, sigma=sigma,
        n_levels=n_levels, decoherence=decoh_zero,
    )

    cases = [
        ("no DRAG", False, 0.0, "C0"),
        ("β = 1", True, 1.0, "C1"),
        (f"β = β_opt = {cal.beta_opt:.2f}", True, cal.beta_opt, "C2"),
    ]
    for label, drag, beta, color in cases:
        r = simulate_x_gate(
            device=REFERENCE_DEVICE, T_gate=T_gate, n_levels=n_levels,
            drag=drag, beta=beta, decoherence=decoh_zero, sigma=sigma,
        )
        P0, P1, P_leak = _populations_from_rho_t(r.rho_t, n_levels)
        t_ns = r.t_array * 1e9
        ax_top.plot(t_ns, P0, color=color, label=label)
        ax_mid.plot(t_ns, P1, color=color, label=label)
        ax_bot.plot(t_ns, P_leak, color=color, label=label)

    ax_top.set_ylabel("$P_0(t)$")
    ax_mid.set_ylabel("$P_1(t)$")
    ax_bot.set_ylabel(r"$P_{\geq 2}(t)$")
    ax_bot.set_xlabel("$t$ (ns)")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.set_title(f"Panel (a) — $T_{{\\rm gate}}$ = {T_gate*1e9:.0f} ns, REFERENCE_DEVICE")


def _panel_b_sweep(n_levels=4):
    """Run the T_gate sweep producing leakage curves, ε_X(T_gate), and V2b
    leakage-vs-fidelity trade-off triplet."""
    decoh_zero = _zero_decoherence()
    decoh_full = REFERENCE_DEVICE.decoherence

    rows = []
    for tg_ns in T_GATE_SWEEP_NS:
        T_gate = float(tg_ns) * 1e-9
        sigma = T_gate / 4.0

        cal = calibrate_drag_beta(
            device=REFERENCE_DEVICE, T_gate=T_gate, sigma=sigma,
            n_levels=n_levels, decoherence=decoh_zero,
        )

        # Leakage curves at no-DRAG and at fidelity-optimal β_opt
        r_no_drag = simulate_x_gate(
            device=REFERENCE_DEVICE, T_gate=T_gate, n_levels=n_levels,
            drag=False, beta=0.0, decoherence=decoh_zero, sigma=sigma,
        )
        r_opt_coh = simulate_x_gate(
            device=REFERENCE_DEVICE, T_gate=T_gate, n_levels=n_levels,
            drag=True, beta=cal.beta_opt, decoherence=decoh_zero, sigma=sigma,
        )
        leak_final_no_drag = leakage_population(r_no_drag.rho_final, n_levels)
        leak_peak_no_drag = leakage_peak(r_no_drag.rho_t, n_levels)
        leak_final_opt = leakage_population(r_opt_coh.rho_final, n_levels)
        leak_peak_opt = leakage_peak(r_opt_coh.rho_t, n_levels)

        # ε_X with full Lindblad at β_opt (headline / curve data per §10)
        r_opt_full = simulate_x_gate(
            device=REFERENCE_DEVICE, T_gate=T_gate, n_levels=n_levels,
            drag=True, beta=cal.beta_opt, decoherence=decoh_full, sigma=sigma,
        )
        F_opt_full = transfer_fidelity_0_to_1(r_opt_full.rho_final)
        eps_x = epsilon_x_from_transfer(r_opt_full.rho_final)

        # V2b trade-off triplet (already computed during calibration)
        rows.append({
            "T_gate_ns": float(tg_ns),
            "beta_opt": float(cal.beta_opt),
            "leakage_final_no_drag": float(leak_final_no_drag),
            "leakage_final_drag_opt": float(leak_final_opt),
            "leakage_peak_no_drag": float(leak_peak_no_drag),
            "leakage_peak_drag_opt": float(leak_peak_opt),
            "epsilon_x_drag_opt": float(eps_x),
            "F_transfer_drag_opt": float(F_opt_full),
            # V2b trade-off (post-N11)
            "beta_min_final_leak": float(cal.beta_min_final_leak),
            "beta_min_peak_leak": float(cal.beta_min_peak_leak),
            "final_leak_supp_at_fidelity_opt": float(leak_final_no_drag / max(leak_final_opt, 1e-30)),
            "peak_leak_supp_at_fidelity_opt": float(leak_peak_no_drag / max(leak_peak_opt, 1e-30)),
        })

    return rows


def _save_yaml(rows, alpha_2pi_hz):
    payload = {
        "device": "REFERENCE_DEVICE",
        "device_provenance": "Marxer arXiv:2508.16437 + Bengtsson PRL 132 100603 (2024)",
        "calibration_objective": "argmin_β (1 − F_transfer); β grid [0, 1.2] (post-N11)",
        "alpha_2pi_Hz": float(alpha_2pi_hz),
        "sweep_T_gate_ns": [r["T_gate_ns"] for r in rows],
        "beta_opt_fidelity": [r["beta_opt"] for r in rows],
        "beta_opt_final_leak": [r["beta_min_final_leak"] for r in rows],
        "beta_opt_peak_leak": [r["beta_min_peak_leak"] for r in rows],
        "leakage_final_no_drag": [r["leakage_final_no_drag"] for r in rows],
        "leakage_final_drag_opt": [r["leakage_final_drag_opt"] for r in rows],
        "leakage_peak_no_drag": [r["leakage_peak_no_drag"] for r in rows],
        "leakage_peak_drag_opt": [r["leakage_peak_drag_opt"] for r in rows],
        "final_leak_supp_at_fidelity_opt": [r["final_leak_supp_at_fidelity_opt"] for r in rows],
        "peak_leak_supp_at_fidelity_opt": [r["peak_leak_supp_at_fidelity_opt"] for r in rows],
        "epsilon_x_drag_opt": [r["epsilon_x_drag_opt"] for r in rows],
        "F_transfer_drag_opt": [r["F_transfer_drag_opt"] for r in rows],
        "notes": (
            "Headline value: epsilon_x_drag_opt at T_gate = 20 ns. "
            "Module 5b spec consumes this YAML as data. "
            "V2b trade-off: beta_opt_fidelity vs beta_opt_final_leak vs "
            "beta_opt_peak_leak diverge across the perturbative β grid — "
            "characterizes leakage-vs-fidelity trade-off as a Module 5a finding."
        ),
    }
    out = FIGURES_DIR / "fig5a_drag_leakage_data.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Wrote {out}")


def _panel_b(ax_main, ax_inset_eps, ax_inset_v2b, rows):
    """Panel (b): leakage final + peak vs T_gate, with two insets."""
    tg = np.array([r["T_gate_ns"] for r in rows])
    ax_main.semilogy(tg, [r["leakage_final_no_drag"] for r in rows], "o-", color="C0",
                     label=r"$P_{\geq 2}(T)$ no DRAG")
    ax_main.semilogy(tg, [r["leakage_peak_no_drag"] for r in rows], "s--", color="C0",
                     alpha=0.6, label=r"$\max_t P_{\geq 2}(t)$ no DRAG")
    ax_main.semilogy(tg, [r["leakage_final_drag_opt"] for r in rows], "o-", color="C2",
                     label=r"$P_{\geq 2}(T)$ $\beta_{\rm opt}$")
    ax_main.semilogy(tg, [r["leakage_peak_drag_opt"] for r in rows], "s--", color="C2",
                     alpha=0.6, label=r"$\max_t P_{\geq 2}(t)$ $\beta_{\rm opt}$")
    ax_main.set_xlabel("$T_{\\rm gate}$ (ns)")
    ax_main.set_ylabel("Leakage population")
    ax_main.set_title("Panel (b) — speed-leakage tradeoff (REFERENCE α; decoherence zeroed)")
    ax_main.legend(loc="lower left", fontsize=7)
    ax_main.grid(True, which="both", alpha=0.3)

    # Inset 1: ε_X(T_gate) under full Lindblad
    ax_inset_eps.semilogy(tg, [r["epsilon_x_drag_opt"] for r in rows], "x-", color="C3")
    ax_inset_eps.set_xlabel("$T_{\\rm gate}$ (ns)", fontsize=7)
    ax_inset_eps.set_ylabel(r"$\varepsilon_X$ (full Lindblad)", fontsize=7)
    ax_inset_eps.tick_params(labelsize=6)
    ax_inset_eps.set_title(r"$\varepsilon_X(T_{\rm gate})$", fontsize=7)
    ax_inset_eps.grid(True, alpha=0.3)

    # Inset 2: V2b leakage-vs-fidelity trade-off — three β minimizers
    ax_inset_v2b.plot(tg, [r["beta_opt"] for r in rows], "o-", label=r"$\beta_{\rm fid}$", color="C2", markersize=4)
    ax_inset_v2b.plot(tg, [r["beta_min_final_leak"] for r in rows], "s--", label=r"$\beta_{\rm fin}$", color="C4", markersize=4)
    ax_inset_v2b.plot(tg, [r["beta_min_peak_leak"] for r in rows], "^:", label=r"$\beta_{\rm peak}$", color="C5", markersize=4)
    ax_inset_v2b.set_xlabel("$T_{\\rm gate}$ (ns)", fontsize=7)
    ax_inset_v2b.set_ylabel(r"$\beta$", fontsize=7)
    ax_inset_v2b.tick_params(labelsize=6)
    ax_inset_v2b.set_title("V2b: leakage-vs-fidelity trade-off", fontsize=7)
    ax_inset_v2b.legend(loc="best", fontsize=6)
    ax_inset_v2b.grid(True, alpha=0.3)


def main():
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.1])
    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_mid = fig.add_subplot(gs[1, 0], sharex=ax_a_top)
    ax_a_bot = fig.add_subplot(gs[2, 0], sharex=ax_a_top)
    ax_b = fig.add_subplot(gs[:, 1])

    _panel_a(ax_a_top, ax_a_mid, ax_a_bot)

    print("Running Panel (b) T_gate sweep — this may take a few minutes...")
    rows = _panel_b_sweep()

    ax_b_inset_eps = ax_b.inset_axes([0.08, 0.08, 0.40, 0.30])
    ax_b_inset_v2b = ax_b.inset_axes([0.55, 0.08, 0.40, 0.30])
    _panel_b(ax_b, ax_b_inset_eps, ax_b_inset_v2b, rows)

    fig.tight_layout()

    alpha = transmon_summary(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)["alpha"]
    alpha_2pi_hz = alpha / (2 * math.pi)
    _save_yaml(rows, alpha_2pi_hz)

    out = FIGURES_DIR / "fig5a_drag_leakage.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    headline = next(r for r in rows if r["T_gate_ns"] == 20.0)
    print(f"Headline: ε_X^ref(T_gate=20ns) = {headline['epsilon_x_drag_opt']:.3e} "
          f"at β_opt={headline['beta_opt']:.2f}")


if __name__ == "__main__":
    main()
