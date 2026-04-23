"""Standalone Figure 4 Panel (c) — 3 parameter-anchored Pareto frontiers
plus the closed-loop recommendation arrow.

Produces figures/fig4_panel_c_pareto.png and fig4_panel_c_data.yaml.
Day-13's fig4_optimization.py imports render_pareto() for the composite.

Modal dispatch: set USE_MODAL=1 to dispatch all 3×10=30 (device, τ_max)
tuples in a single `pareto_one_tuple.map(...)` call (Q2 lock); otherwise
runs serial per variant. `return_exceptions=True` on Modal so one slow
worker hitting the per-input timeout doesn't abort the whole batch.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dispersive_readout.optimization.pareto import (
    PARETO_DEVICE_VARIANTS, TAU_MAX_GRID_NS,
    ParetoPoint, build_variant, compute_pareto_frontier,
)


_VARIANT_STYLES = {
    "REFERENCE (≈ Marxer Q1)":                  {"color": "#2C3E50", "marker": "o"},
    "T_1 = 40 µs (Garnet-like)":                 {"color": "#7F8C8D", "marker": "s"},
    "T_1 = 20 µs, κ/2π = 6 MHz (Bengtsson-like)": {"color": "#566573", "marker": "^"},
}


def render_pareto(ax: plt.Axes, frontiers: dict[str, list]) -> None:
    for label, points in frontiers.items():
        style = _VARIANT_STYLES.get(label, {"color": "black", "marker": "o"})
        tau_ns = [p.tau_opt * 1e9 for p in points]
        F = np.array([p.F_assign_opt for p in points])
        sigma = np.array([p.F_assign_uncertainty for p in points])

        ax.fill_between(tau_ns, F - sigma, F + sigma, color=style["color"], alpha=0.15)
        ax.plot(tau_ns, F, color=style["color"], linestyle="-", linewidth=1.2)
        ax.scatter(tau_ns, F, marker=style["marker"], s=36, color=style["color"],
                   edgecolors="white", linewidths=0.8, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Readout duration $\tau_{\rm opt}$ (ns, log)")
    ax.set_ylabel(r"$F_{\rm assign}$ at optimum")
    ax.set_title("Speed–fidelity Pareto frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _relabel(p: ParetoPoint, label: str) -> ParetoPoint:
    """Stamp the variant label onto the ParetoPoint returned by a Modal worker."""
    return ParetoPoint(
        device_id=p.device_id,
        device_label=label,
        tau_max=p.tau_max,
        epsilon_0_opt=p.epsilon_0_opt,
        tau_opt=p.tau_opt,
        F_assign_opt=p.F_assign_opt,
        F_assign_uncertainty=p.F_assign_uncertainty,
        dominant_loss_channel=p.dominant_loss_channel,
        solver_converged=p.solver_converged,
    )


def _collect_frontiers_modal() -> dict[str, list[ParetoPoint]]:
    """Q2-locked single-session .map() over all 3×10=30 tuples.

    `return_exceptions=True` so one slow worker (per-input timeout) leaves
    the rest of the batch intact — any successful tuples are retained with
    a `[MISSING]` placeholder logged for the failed ones.
    """
    from dispersive_readout.optimization.modal_pareto import app, pareto_one_tuple

    # Build the 30-tuple batch with variant→label bookkeeping
    variants = [(spec["label"], build_variant(spec)) for spec in PARETO_DEVICE_VARIANTS]
    tau_max_grid = [float(t) for t in TAU_MAX_GRID_NS * 1e-9]
    labels: list[str] = []
    devices = []
    tau_maxes: list[float] = []
    for label, device in variants:
        for tau_max in tau_max_grid:
            labels.append(label)
            devices.append(device)
            tau_maxes.append(tau_max)

    print(f"Modal: dispatching {len(labels)} tuples in one .map() session "
          f"(3 variants × 10 τ_max)")

    with app.run():
        results = list(pareto_one_tuple.map(
            devices, tau_maxes, return_exceptions=True,
        ))

    # Group by label, stamping and logging failures
    frontiers: dict[str, list[ParetoPoint]] = {label: [] for label, _ in variants}
    n_ok = n_fail = 0
    for label, tau_max, res in zip(labels, tau_maxes, results):
        if isinstance(res, ParetoPoint):
            frontiers[label].append(_relabel(res, label))
            n_ok += 1
        else:
            print(f"  [MISSING] {label!r} τ_max={tau_max*1e9:.0f} ns → "
                  f"{type(res).__name__}: {res}")
            n_fail += 1

    print(f"Modal: {n_ok}/{len(labels)} succeeded, {n_fail} failed")
    return frontiers


def _collect_frontiers_serial() -> dict[str, list[ParetoPoint]]:
    frontiers: dict[str, list[ParetoPoint]] = {}
    for spec in PARETO_DEVICE_VARIANTS:
        device = build_variant(spec)
        frontiers[spec["label"]] = compute_pareto_frontier(
            device, tau_max_values=TAU_MAX_GRID_NS * 1e-9,
            device_label=spec["label"], use_modal=False,
        )
    return frontiers


def main() -> None:
    use_modal = os.environ.get("USE_MODAL", "0") == "1"
    if use_modal:
        frontiers = _collect_frontiers_modal()
    else:
        frontiers = _collect_frontiers_serial()

    # Persist YAML (skip empty variants — would produce malformed plot)
    out_dir = Path("06_Dispersive_Readout/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        label: [p.model_dump() for p in points]
        for label, points in frontiers.items() if points
    }
    with open(out_dir / "fig4_panel_c_data.yaml", "w") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    render_pareto(ax, {k: v for k, v in frontiers.items() if v})
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_panel_c_pareto.png", dpi=150)
    print(f"Wrote {out_dir / 'fig4_panel_c_pareto.png'}")


if __name__ == "__main__":
    main()
