"""Standalone Figure 4 Panel (a) — sensitivity tornado at REFERENCE.

Produces figures/fig4_panel_a_tornado.png. Day-13's fig4_optimization.py
will import render_tornado() for the composite without duplicating the
rendering logic.

Memory rules applied (feedback_figure_presentation):
    - Cool palette for sensitivities (two shades for positive/negative)
    - Horizontal bar chart sorted by |S| descending
    - Numeric annotation adjacent to each bar
    - y-axis labels include (±5%) perturbation scale
    - Point-with-errorbar when |S| < SENSITIVITY_RENDER_BAR_THRESHOLD
    - Anchoring subtitle with F_ref, τ_int, n̄_phot, n_shots

Spec amendments applied (see docs/working/module4_amendment_draft.md):
    - O1a/O1b split: bar-rendered parameters get signs asserted; near-zero
      parameters render as point-with-errorbar and are annotated with
      measured value alongside.
    - χ sits at noise-floor at REFERENCE (|S_χ|=0.029 < 0.03 threshold):
      renders as point-with-errorbar per Q1 amendment.
    - γ_φ at REFERENCE: |S_γφ| = 0.0 exactly (float underflow from
      T_2_echo ≈ 2·T_1); renders as point at x=0 with symmetric errorbar.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.physics.readout_model import simulate_readout
from dispersive_readout.optimization.sensitivity import (
    SensitivityResult,
    compute_all_sensitivities,
    rank_sensitivities,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
)


_DISPLAY_LABELS: dict[str, str] = {
    "chi_scale": r"$\chi$ scale ($\pm 5\%$)",
    "kappa":     r"$\kappa$ ($\pm 5\%$)",
    "gamma_1":   r"$T_1$ (via $\gamma_1$, $\pm 5\%$)",
    "gamma_phi": r"$T_\varphi$ (via $\gamma_\varphi$, $\pm 5\%$)",
    "n_th":      r"$\bar n_{\rm th}$ ($\pm 5\%$)",
    "epsilon_0": r"$\varepsilon_0$ ($\pm 5\%$)",
    "tau":       r"$\tau$ ($\pm 5\%$)",
}


def _infer_steady_state_photon_number() -> float:
    """Average photon number over last 20% of the integration window.

    Same heuristic that Task 10's regime-map computation will use; factored
    here so the Panel (a) subtitle's n̄_phot matches Panel (b)'s subtitle.
    """
    op = get_reference_operating_point(n_shots=10_000)
    r0 = simulate_readout(op.device, op.drive, initial_qubit_state=0)
    t = r0.t
    t0, t1 = op.integration_window
    mask = (t >= t0 + 0.8 * (t1 - t0)) & (t <= t1)
    return float(np.mean(r0.photon_number[mask]))


def render_tornado(
    ax: plt.Axes,
    sensitivities: list[SensitivityResult],
    anchoring: str,
) -> None:
    """Render the tornado panel on the provided axis.

    Reusable by Day-13's fig4_optimization.py composite — same signature
    and visual conventions.
    """
    ranked = rank_sensitivities(sensitivities)
    # Plot order: largest |S| at top
    ys = np.arange(len(ranked))[::-1]
    labels = [_DISPLAY_LABELS[r.parameter] for r in ranked]

    # Sign-contrast palette: cool blue for positive (helps F), warm orange-
    # red for negative (hurts F). Two distinct hues make sign legible at a
    # glance — prior two-shade cool palette read as one color to readers.
    color_pos = "#4A90E2"
    color_neg = "#D35400"

    for y, r in zip(ys, ranked):
        S = r.sensitivity
        sigma = r.sensitivity_uncertainty
        is_noise_like = r.noise_consistent_with_zero
        color = color_pos if S >= 0 else color_neg
        if is_noise_like:
            # Point-with-errorbar rendering for near-zero parameters.
            # Day-14 round-2 polish: lighter label color (#999) and
            # central-value only (no ± σ) so the bar-rendered ε_0 label
            # retains visual weight and the annotation layer doesn't
            # compete with the plotted data.
            ax.errorbar(
                [S], [y], xerr=[sigma], fmt="o", color=color,
                capsize=3, markersize=5, zorder=5,
            )
            label_x = S + sigma + 0.004
            ax.text(
                label_x, y, f"{S:+.3f}",
                va="center", ha="left", fontsize=9, color="#999999",
            )
        else:
            # Filled bar for bar-rendered parameters (|S| >= threshold).
            # Keep dimgray and full weight — this is the lever the figure
            # is actually pointing at.
            ax.barh(
                [y], [S], color=color, edgecolor="black", linewidth=0.5,
                alpha=0.9, zorder=3,
            )
            offset = 0.008 * (1 if S >= 0 else -1)
            ha = "left" if S >= 0 else "right"
            ax.text(
                S + offset, y, f"{S:+.3f}", va="center", ha=ha,
                fontsize=9.5, fontweight="semibold", color="dimgray",
            )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlabel(
        r"Normalized log-sensitivity $S_\theta = \partial \ln F / \partial \ln \theta$",
        fontsize=10,
    )
    ax.set_title(
        "Parameter sensitivity of $F_{\\rm assign}$ at REFERENCE (Marxer arXiv:2508.16437)\n"
        + anchoring, fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade the noise-consistent band symmetrically around x=0 for visual
    # context — bar-rendered parameters sit outside the band, noise-
    # consistent parameters sit inside it and render as point-with-errorbar
    # per the Q1 amendment. Band's role is explained in the figure caption
    # to keep the plot itself uncluttered.
    ax.axvspan(
        -SENSITIVITY_RENDER_BAR_THRESHOLD,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
        color="lightgray", alpha=0.2, zorder=0,
    )


def main() -> None:
    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)
    n_phot = _infer_steady_state_photon_number()

    tau_int_ns = (op.integration_window[1] - op.integration_window[0]) * 1e9
    anchoring = (
        f"$F_{{\\rm ref}}={sens[0].F_reference:.4f}$, "
        f"$\\tau_{{\\rm int}} = {tau_int_ns:.0f}$ ns, "
        f"$\\bar n_{{\\rm phot}} = {n_phot:.2f}$, "
        f"$n_{{\\rm shots}}=10^4$"
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    render_tornado(ax, sens, anchoring)
    fig.tight_layout()

    out = Path("06_Dispersive_Readout/figures/fig4_panel_a_tornado.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    # Print subtitle for caption-language verification
    print(f"  F_ref  = {sens[0].F_reference:.4f}")
    print(f"  τ_int  = {tau_int_ns:.0f} ns")
    print(f"  n̄_phot = {n_phot:.3f}")
    print("  Top-3 by |S|:")
    for r in rank_sensitivities(sens)[:3]:
        print(f"    {r.parameter:<12}  S={r.sensitivity:+.4f} ± {r.sensitivity_uncertainty:.4f}")


if __name__ == "__main__":
    main()
