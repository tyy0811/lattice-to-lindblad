"""Render Figure 2 (error budget waterfall) + export YAML data.

Two candidate layouts are rendered side-by-side as separate PNGs during
Task 11; this Task 10 version produces Candidate B (classic waterfall) as
the default and the YAML data export. Uses n_shots=100_000 per spec
amendment 8 to recover the physics-dominated regime. Run from repo root:

    python 06_Dispersive_Readout/scripts/fig2_error_budget.py

Outputs:
  06_Dispersive_Readout/figures/fig2_error_budget.png (150 DPI, ~1200 px)
  06_Dispersive_Readout/figures/fig2_data.yaml (ErrorBudget serialized)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.analysis import (
    ErrorBudget,
    compute_full_error_budget,
    export_budget_to_yaml,
    get_reference_operating_point,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "06_Dispersive_Readout" / "figures"


def _render_candidate_B(budget: ErrorBudget, path: Path) -> None:
    """Classic waterfall: Ideal floor | active loss stack | R_active | === | cal sens."""
    active = budget.active_loss_channels
    calib = budget.calibration_channels
    ideal_floor = 1.0 - budget.F_ideal

    # Bars left-to-right
    labels = (
        ["Ideal\nfloor"]
        + [c.name.replace("_", "\n") for c in active]
        + ["R_active"]
        + [""]  # separator
        + [c.name.replace("_", "\n") for c in calib]
    )
    values = (
        [ideal_floor]
        + [c.delta_F for c in active]
        + [budget.residual_active]
        + [0.0]  # separator (invisible)
        + [c.delta_F for c in calib]
    )
    errors = (
        [0.0]
        + [c.delta_F_uncertainty for c in active]
        + [budget.residual_active_uncertainty]
        + [0.0]
        + [c.delta_F_uncertainty for c in calib]
    )
    # Scale to 10^-3 units for readability
    values_milli = [v * 1e3 for v in values]
    errors_milli = [e * 1e3 for e in errors]

    # Color palette
    warm = plt.cm.OrRd(np.linspace(0.4, 0.85, len(active)))
    cool = plt.cm.Blues(np.linspace(0.5, 0.85, len(calib)))
    colors = (
        ["#888888"]                   # ideal floor grey
        + list(warm)                   # active loss warm
        + ["#555555"]                  # residual dark grey
        + ["none"]                     # separator
        + list(cool)                   # cal sens cool
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, values_milli, color=colors, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, values_milli, yerr=errors_milli, fmt="none",
                ecolor="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Contribution to 1 − F (× 10⁻³)", fontsize=10)
    ax.set_title(
        f"Assignment Infidelity Decomposition — REFERENCE_DEVICE\n"
        f"F_full = {budget.F_full:.4f}, F_ideal = {budget.F_ideal:.4f}, "
        f"n_shots = 10⁵",
        fontsize=10,
    )
    # Group separator
    ax.axvline(x=len(active) + 1.5, color="gray", linestyle="--", linewidth=0.6)
    # Group labels
    ax.text(1 + len(active) / 2 - 0.5, ax.get_ylim()[1] * 0.92, "Active loss",
            ha="center", fontsize=9, style="italic")
    ax.text(len(active) + 3 + len(calib) / 2 - 0.5, ax.get_ylim()[1] * 0.92,
            "Calibration sensitivity", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Computing reference operating point (calibration + verification)...")
    # n_shots=100_000 per amendment 8: recovers physics-dominated regime so
    # Figure 2 bars are visually resolvable above shot noise.
    op = get_reference_operating_point(n_shots=100_000)
    print(f"  ε₀ = {op.drive.amplitude:.3e} rad/s "
          f"(= {op.drive.amplitude / (2 * np.pi):.3e} Hz)")

    print("Computing full error budget (14 sims at n_shots=1e5, ~15 min)...")
    budget = compute_full_error_budget(op)
    print(f"  F_full = {budget.F_full:.5f}")
    print(f"  F_ideal = {budget.F_ideal:.5f}")
    print(f"  R_active = {budget.residual_active:.5f} "
          f"± {budget.residual_active_uncertainty:.5f}")
    for c in budget.channels:
        print(f"  {c.name:20s}  ΔF = {c.delta_F:.5f} ± {c.delta_F_uncertainty:.5f}")

    png_path = FIG_DIR / "fig2_error_budget.png"
    yaml_path = FIG_DIR / "fig2_data.yaml"
    _render_candidate_B(budget, png_path)
    export_budget_to_yaml(budget, yaml_path)
    print(f"Wrote {png_path} and {yaml_path}")


if __name__ == "__main__":
    main()
