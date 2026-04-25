"""Figure 4 composite: sensitivity tornado + analytic regime map + Pareto frontier.

Imports the three panel renderers from their standalone scripts and
assembles a 3-panel horizontal layout with a figure-wide caption
containing the three locked caveats (Q1 orthogonality, Q3 analytic
regime, Q4 closed-loop scope).

Data sources (Day-13 user directive: reuse cached artifacts where
deterministic output is already committed):
  Panel (a): compute_all_sensitivities(op) + day_10_cross_check (fresh; ~6 min)
  Panel (b): compute_analytic_regime_map() (instant) + cached
            fig4_panel_b_validation.yaml (validation from Task 11)
  Panel (c): cached fig4_panel_c_data.yaml (30-tuple Pareto from Task 14)

See MODULE_4_SPEC.md section 7 for the locked design contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import yaml

# Reuse standalone-panel renderers (no rendering duplication)
from fig4_panel_a_tornado import render_tornado
from fig4_panel_b_regime import render_regime_map
from fig4_panel_c_pareto import render_pareto, _load_frontiers_from_yaml, load_closed_loop_demo

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.optimization.sensitivity import (
    compute_all_sensitivities, day_10_cross_check_s_g_vs_s_chi,
)
from dispersive_readout.optimization.regime_map import compute_analytic_regime_map


_FIGURES_DIR = Path("06_Dispersive_Readout/figures")


def main() -> None:
    # Panel (a) data: sensitivities + Day-10 cross-check (fresh)
    op = get_reference_operating_point(n_shots=10_000)
    print("Panel (a): computing 7-parameter sensitivity tornado + Day-10 cross-check")
    sens = compute_all_sensitivities(op)
    cross = day_10_cross_check_s_g_vs_s_chi(op)
    print(
        f"  Day-10: S_chi={cross['S_chi']:+.4f}, S_g={cross['S_g']:+.4f}, "
        f"residual_fractional={cross['residual_fractional']:.4f}"
    )

    # Panel (b) data: analytic grid (instant) + cached validation
    print("Panel (b): computing analytic regime grid + loading cached validation")
    grid = compute_analytic_regime_map()
    validation_path = _FIGURES_DIR / "fig4_panel_b_validation.yaml"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"{validation_path} missing; run scripts/fig4_panel_b_regime.py "
            "first to regenerate the Lindblad-validation artifact."
        )
    validation = yaml.safe_load(validation_path.read_text())

    # Panel (c) data: cached Pareto frontiers
    print("Panel (c): loading cached Pareto frontiers from fig4_panel_c_data.yaml")
    cache_path = _FIGURES_DIR / "fig4_panel_c_data.yaml"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing; run scripts/fig4_panel_c_pareto.py "
            "(USE_MODAL=1) first to regenerate the Pareto frontier cache."
        )
    frontiers = _load_frontiers_from_yaml(cache_path)

    # Composite figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    tau_int_ns = (op.integration_window[1] - op.integration_window[0]) * 1e9
    anchoring_a = (
        f"$F_{{\\rm ref}}={sens[0].F_reference:.4f}$, "
        f"$\\tau_{{\\rm int}}$ = {tau_int_ns:.0f} ns"
    )
    render_tornado(axes[0], sens, anchoring_a)
    render_regime_map(axes[1], validation, grid)
    render_pareto(axes[2], frontiers, closed_loop=load_closed_loop_demo())

    # Bold (a)(b)(c) labels inside the upper-left of each subplot — kept
    # compact (fontsize 13) per Day-14 round-2 review so they don't
    # compete with the per-panel titles.
    for ax, letter in zip(axes, ("a", "b", "c")):
        ax.text(
            0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )

    # Figure-wide caption with the 3 locked caveats + anchor numbers.
    # Explicit line breaks (\n) per paragraph — matplotlib's wrap=True is
    # unreliable for fig.text at composite-figure widths; pre-wrapped text
    # renders predictably.
    max_dev_pct = validation["max_deviation_fractional"] * 100.0
    residual_pct = cross["residual_fractional"] * 100.0
    n_validation_points = len(validation.get("per_point", []))
    # Caption per Day-14 review round 2: recruiter-facing hero-figure
    # brevity. Day-10 residual, S_χ orthogonality detail, Hazra citation,
    # and per-regime physics narrative moved to the report body. Caption
    # keeps only: one-sentence per panel + synthetic/honesty caveat.
    # Scope caveat rendered as a separate smaller paragraph below.
    # (Unused helpers preserved for possible report-text re-use.)
    _ = residual_pct, max_dev_pct, n_validation_points  # kept computed for report
    caption_main = (
        r"$\bf{Figure\ 4.}$ Optimization layer for dispersive transmon readout."
        "\n"
        r"$\bf{(a)}$ Local normalized sensitivities $S_\theta = \partial\ln F_{\rm assign}/\partial\ln\theta$ "
        r"at the REFERENCE operating point."
        "\n"
        r"Near $F_{\rm assign} \approx 0.99$, $F$-space sensitivities are compressed; the local optimum "
        r"is control-dominated, with $\varepsilon_0$ the only clearly bar-rendered lever."
        "\n"
        r"$\bf{(b)}$ Analytic regime map over $\chi/\kappa$ and $\gamma_1 \tau_{\rm readout}$, with "
        r"REFERENCE-family device anchors and Lindblad spot-checks."
        "\n"
        r"Dashed lines mark Purcell, $\chi$-phase accumulation, and resonator-response boundaries."
        "\n"
        r"$\bf{(c)}$ Speed–fidelity Pareto frontiers for three REFERENCE-family parameter variants."
        "\n"
        r"The open marker shows a fitted synthetic demo device mapped by the closed-loop "
        r"recommendation pipeline to its Pareto operating point; curves are simulator-predicted "
        r"under parameter substitution,"
        "\n"
        r"$\bf{not}$ claims about the cited devices' native hardware."
    )
    caption_scope = (
        r"Closed-loop scope: fitted ($T_1$, $T_2$, $\omega_q$) over fixed REFERENCE "
        r"resonator/coupling; full resonator spectroscopy and AC-Stark calibration are "
        r"post-submission extensions."
    )
    fig.text(0.01, -0.03, caption_main, fontsize=11, ha="left", va="top")
    fig.text(0.01, -0.33, caption_scope, fontsize=9.5, ha="left", va="top",
             color="#444444", style="italic")

    out = _FIGURES_DIR / "fig4_optimization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
