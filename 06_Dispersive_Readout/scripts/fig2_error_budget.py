"""Render Figure 2 (two-panel error budget + calibration sensitivity).

Panel A: active-loss decomposition at the reference operating point (ideal
floor, four decoherence channels sorted by magnitude, residual). Bars for
channels that are 1σ-resolvable above zero; point-with-errorbar for
near-zero channels so the figure does not imply a negative observed loss.

Panel B: calibration sensitivity under named perturbations (±5% drive
amplitude, ±κ/4 drive detuning). Independent y-axis because these bars
are not additive with Panel A — they answer a different question
(robustness derivative, not budget component).

Uses n_shots=100_000 per amendment 8 to recover the physics-dominated
regime. Amendment 9 (shared baseline + signed ΔF + shot-noise err)
applied. Run from repo root:

    python 06_Dispersive_Readout/scripts/fig2_error_budget.py

Outputs:
  06_Dispersive_Readout/figures/fig2_error_budget.png (150 DPI)
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


# Labels for calibration-sensitivity bars include the perturbation size
# so the reader does not have to read the caption to interpret them.
_CALIB_PERTURBATION_LABEL: dict[str, str] = {
    "drive_amplitude": "drive amplitude\n(±5%)",
    "drive_detuning":  "drive detuning\n(±κ/4)",
}

_ACTIVE_CHANNEL_LABEL: dict[str, str] = {
    "T1_intrinsic":   "T1",
    "pure_dephasing": "dephasing",
    "thermal":        "thermal",
    "purcell":        "Purcell",
}


def _bar_or_point(
    ax, x, value_milli, err_milli, color,
    bar_width=0.7, force_point=False, annotate=True,
):
    """Filled bar if value > 1σ above zero; else point-with-errorbar.

    Prevents the final figure from showing filled negative bars — those
    confuse a reader into treating them as real negative losses, when
    they are shot-noise excursions around a channel ΔF ≈ 0. Signed
    values remain in the YAML and the test harness.

    force_point=True renders as point-with-errorbar regardless of the
    value/uncertainty ratio. Used for the residual bar, where a filled
    rectangle invites over-interpretation ("model is missing something")
    even when |R| sits within the shot-noise propagation of the identity.
    """
    is_bar = (not force_point) and (value_milli > err_milli) and (value_milli > 0)
    if is_bar:
        ax.bar(
            x, value_milli, width=bar_width,
            color=color, edgecolor="black", linewidth=0.6,
        )
        if err_milli > 0:
            ax.errorbar(
                x, value_milli, yerr=err_milli, fmt="none",
                ecolor="black", capsize=2, linewidth=0.8,
            )
    else:
        # Near-zero, negative, or force_point: point-with-errorbar, no fill.
        ax.errorbar(
            x, value_milli, yerr=err_milli if err_milli > 0 else None,
            fmt="o", color=color, markersize=5,
            ecolor="black", capsize=3, linewidth=0.8,
        )

    if annotate:
        # Place the numeric annotation just above the upper error-bar cap so
        # even negative values get a readable label. Format at 0.1 resolution
        # which is appropriate for ×10⁻³ scale where channels sit at 0.1–30.
        top = value_milli + err_milli if err_milli > 0 else value_milli
        text = f"{value_milli:+.1f}" if value_milli < 0 else f"{value_milli:.1f}"
        ax.text(
            x, top, text,
            ha="center", va="bottom", fontsize=7,
            clip_on=False,
        )


def _render_two_panel(budget: ErrorBudget, path: Path) -> None:
    """Two-panel figure: Panel A active loss, Panel B calibration sensitivity.

    Panel A is a classic-waterfall ordering: ideal floor (leftmost) →
    active channels sorted by |ΔF| descending → residual (rightmost).
    Panel B lists calibration sensitivities with explicit perturbation
    sizes on the x-axis labels.
    """
    # ---- Panel A data ----
    active = budget.active_loss_channels
    # Sort by |ΔF| descending so the dominant loss sits next to the ideal floor.
    active_sorted = sorted(active, key=lambda c: abs(c.delta_F), reverse=True)
    ideal_floor = 1.0 - budget.F_ideal
    warm = plt.cm.OrRd(np.linspace(0.4, 0.85, len(active_sorted)))

    # ---- Panel B data ----
    calib = budget.calibration_channels
    cool = plt.cm.Blues(np.linspace(0.5, 0.85, max(len(calib), 1)))

    # ---- Layout ----
    fig, (ax_A, ax_B) = plt.subplots(
        1, 2,
        figsize=(10.0, 4.5), dpi=150,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_A, ax_B):
        ax.set_facecolor("white")

    # ---- Panel A render ----
    panel_A_labels = (
        ["Ideal\nreadout floor"]
        + [_ACTIVE_CHANNEL_LABEL[c.name] for c in active_sorted]
        + ["residual"]
    )
    panel_A_values = (
        [ideal_floor]
        + [c.delta_F for c in active_sorted]
        + [budget.residual_active]
    )
    panel_A_errors = (
        [0.0]
        + [c.delta_F_uncertainty for c in active_sorted]
        + [budget.residual_active_uncertainty]
    )
    panel_A_colors = (
        ["#888888"]              # ideal floor — grey
        + list(warm)             # active-loss channels — warm
        + ["#555555"]            # residual — darker grey
    )

    n_entries = len(panel_A_values)
    for i, (val, err, col) in enumerate(zip(panel_A_values, panel_A_errors, panel_A_colors)):
        # Last entry is the residual: always render as point-with-errorbar,
        # regardless of whether |R| happens to exceed σ_R on a given run.
        is_residual = (i == n_entries - 1)
        _bar_or_point(
            ax_A, i, val * 1e3, err * 1e3, col,
            force_point=is_residual,
        )

    ax_A.axhline(0.0, color="gray", linewidth=0.5)
    ax_A.set_xticks(np.arange(len(panel_A_labels)))
    ax_A.set_xticklabels(panel_A_labels, fontsize=9)
    ax_A.set_ylabel("Contribution to 1 − F  (× 10⁻³)", fontsize=10)
    ax_A.set_title("A. Active-loss decomposition", fontsize=10, loc="left")
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)

    # ---- Panel B render ----
    panel_B_labels = [_CALIB_PERTURBATION_LABEL[c.name] for c in calib]
    panel_B_values = [c.delta_F for c in calib]
    panel_B_errors = [c.delta_F_uncertainty for c in calib]

    for i, (val, err, col) in enumerate(zip(panel_B_values, panel_B_errors, cool)):
        _bar_or_point(ax_B, i, val * 1e3, err * 1e3, col)

    ax_B.axhline(0.0, color="gray", linewidth=0.5)
    ax_B.set_xticks(np.arange(len(panel_B_labels)))
    ax_B.set_xticklabels(panel_B_labels, fontsize=9)
    ax_B.set_ylabel("ΔF under perturbation  (× 10⁻³)", fontsize=10)
    ax_B.set_title("B. Calibration sensitivity", fontsize=10, loc="left")
    ax_B.spines["top"].set_visible(False)
    ax_B.spines["right"].set_visible(False)

    fig.suptitle(
        "Assignment infidelity at REFERENCE_DEVICE",
        fontsize=11, y=1.02,
    )
    # Small subtitle under the main title with the anchoring numbers.
    fig.text(
        0.5, 0.955,
        f"F_full = {budget.F_full:.4f}  ·  F_ideal = {budget.F_ideal:.4f}  ·  "
        f"n_shots = 10⁵",
        ha="center", va="top", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _compute_n_bar_over_n_crit(op) -> tuple[float, float, float]:
    """Return (n̄_peak, n_crit, ratio) at the operating point.

    n_crit = (Δ_10 / (2g))² per Shillito 2022. n̄_peak is the maximum
    mean-photon-number measured in a baseline simulation starting in |1⟩
    (the worse case; drive populates the resonator more there).
    """
    from dispersive_readout.physics import simulate_readout
    from dispersive_readout.physics.transmon import (
        diagonalize_transmon,
    )

    device = op.device
    tr = device.truncation
    energies, _ = diagonalize_transmon(device.transmon, tr)
    g = device.coupling.g
    omega_r = device.resonator.omega_r
    delta_10 = energies[1] - energies[0] - omega_r
    n_crit = (delta_10 / (2.0 * g)) ** 2

    r = simulate_readout(device, op.drive, initial_qubit_state=1)
    n_bar_peak = float(r.photon_number.max())

    return n_bar_peak, float(n_crit), n_bar_peak / float(n_crit)


def main() -> None:
    print("Computing reference operating point (calibration + verification)...")
    # n_shots=100_000 per amendment 8: recovers physics-dominated regime so
    # Figure 2 bars are visually resolvable above shot noise.
    op = get_reference_operating_point(n_shots=100_000)
    print(f"  ε₀ = {op.drive.amplitude:.3e} rad/s "
          f"(= {op.drive.amplitude / (2 * np.pi):.3e} Hz)")

    print("Computing full error budget (14 sims at n_shots=1e5, ~3 min)...")
    budget = compute_full_error_budget(op)
    print(f"  F_full  = {budget.F_full:.5f}")
    print(f"  F_ideal = {budget.F_ideal:.5f}")
    print(f"  R_active = {budget.residual_active:.5f} "
          f"± {budget.residual_active_uncertainty:.5f}")
    for c in budget.channels:
        print(f"  {c.name:20s}  ΔF = {c.delta_F:+.5f} ± {c.delta_F_uncertainty:.5f}")

    n_bar, n_crit, ratio = _compute_n_bar_over_n_crit(op)
    print(f"  n̄_peak = {n_bar:.2f}, n_crit = {n_crit:.1f}, n̄/n_crit = {ratio:.3f}")

    png_path = FIG_DIR / "fig2_error_budget.png"
    yaml_path = FIG_DIR / "fig2_data.yaml"
    _render_two_panel(budget, png_path)
    export_budget_to_yaml(budget, yaml_path)
    print(f"Wrote {png_path} and {yaml_path}")


if __name__ == "__main__":
    main()
