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


# Three distinct styles (color + linestyle + marker) so variants are
# legible at a glance — prior 3-greys palette blurred them together per
# external reviewer feedback.
_VARIANT_STYLES = {
    "REFERENCE (≈ Marxer Q1)":                   {"color": "#1F3A5F", "marker": "o", "ls": "-"},
    "T_1 = 40 µs (Garnet-like)":                  {"color": "#E8801A", "marker": "s", "ls": "--"},
    "T_1 = 20 µs, κ/2π = 6 MHz (Bengtsson-like)": {"color": "#2E7D32", "marker": "^", "ls": "-."},
}


def _find_reference_knee(points: list) -> tuple[float, float] | None:
    """Anchor annotation for the REFERENCE curve: the last τ at which F gains
    ≥ 0.03 over the previous τ_max grid point. Past this knee, each further
    doubling of τ buys only diminishing fidelity, which is the practical
    "done" point for an operator budgeting readout time."""
    if len(points) < 2:
        return None
    # Sort by tau_opt just in case input ordering isn't guaranteed.
    pts = sorted(points, key=lambda p: p.tau_opt)
    best_idx = 0
    for i in range(1, len(pts)):
        if pts[i].F_assign_opt - pts[i - 1].F_assign_opt >= 0.03:
            best_idx = i
    return (pts[best_idx].tau_opt * 1e9, pts[best_idx].F_assign_opt)


def render_pareto(ax: plt.Axes, frontiers: dict[str, list]) -> None:
    for label, points in frontiers.items():
        style = _VARIANT_STYLES.get(label, {"color": "black", "marker": "o", "ls": "-"})
        tau_ns = [p.tau_opt * 1e9 for p in points]
        F = np.array([p.F_assign_opt for p in points])
        sigma = np.array([p.F_assign_uncertainty for p in points])

        ax.fill_between(tau_ns, F - sigma, F + sigma, color=style["color"], alpha=0.15)
        ax.plot(tau_ns, F, color=style["color"], linestyle=style["ls"], linewidth=1.4)
        ax.scatter(tau_ns, F, marker=style["marker"], s=42, color=style["color"],
                   edgecolors="white", linewidths=0.8, label=label, zorder=5)

    # Anchor annotation on the REFERENCE curve: show the practical knee
    # point (last big-gain step). Makes one concrete operating point visible
    # on the plot itself — pairs this panel to Module 2 / Module 4's
    # REFERENCE working point without forcing the reader to scan the legend.
    ref_points = frontiers.get("REFERENCE (≈ Marxer Q1)")
    if ref_points:
        knee = _find_reference_knee(ref_points)
        if knee is not None:
            tau_knee, F_knee = knee
            ax.annotate(
                rf"REFERENCE knee: $\tau={tau_knee:.0f}$ ns, $F={F_knee:.3f}$",
                xy=(tau_knee, F_knee), xycoords="data",
                xytext=(12, -28), textcoords="offset points",
                fontsize=8.5, color="#1F3A5F",
                arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#1F3A5F", alpha=0.85),
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Readout duration $\tau_{\rm opt}$ (ns, log)")
    ax.set_ylabel(r"$F_{\rm assign}$ at optimum")
    # Subtitle names the objective so an external reader knows what the
    # frontier is a frontier OF.
    ax.set_title(
        "Speed–fidelity Pareto frontier\n"
        r"maximize $F_{\rm assign}$ over $(\varepsilon_0, \tau)$ subject to "
        r"$\tau \leq \tau_{\rm max}$; shaded = analytic binomial SE "
        r"at $n_{\rm shots}=10^4$",
        fontsize=10,
    )
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


def _load_frontiers_from_yaml(path: Path) -> dict[str, list[ParetoPoint]]:
    """Re-hydrate ParetoPoint records from a previously-persisted yaml.
    Skips the expensive Pareto re-computation when only the rendering
    layer (styles, annotations, title) changed."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    out: dict[str, list[ParetoPoint]] = {}
    for label, records in raw.items():
        out[label] = [ParetoPoint(**r) for r in records]
    return out


def main() -> None:
    use_modal  = os.environ.get("USE_MODAL",  "0") == "1"
    use_cached = os.environ.get("USE_CACHED", "0") == "1"

    out_dir = Path("06_Dispersive_Readout/figures")
    yaml_path = out_dir / "fig4_panel_c_data.yaml"

    if use_cached:
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"USE_CACHED=1 but {yaml_path} does not exist; "
                "run once without USE_CACHED to generate the Pareto data."
            )
        frontiers = _load_frontiers_from_yaml(yaml_path)
    elif use_modal:
        frontiers = _collect_frontiers_modal()
    else:
        frontiers = _collect_frontiers_serial()

    # Persist YAML on re-compute paths (skip on cache-read since we just
    # loaded from it; rewriting with identical content is noise)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not use_cached:
        serializable = {
            label: [p.model_dump() for p in points]
            for label, points in frontiers.items() if points
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    render_pareto(ax, {k: v for k, v in frontiers.items() if v})
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_panel_c_pareto.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {out_dir / 'fig4_panel_c_pareto.png'}")


if __name__ == "__main__":
    main()
