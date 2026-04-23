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
from fig4_panel_c_pareto import render_pareto, _load_frontiers_from_yaml

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
    render_pareto(axes[2], frontiers)

    # Figure-wide caption with the 3 locked caveats + anchor numbers
    max_dev_pct = validation["max_deviation_fractional"] * 100.0
    residual_pct = cross["residual_fractional"] * 100.0
    n_validation_points = len(validation.get("per_point", []))
    caption = (
        r"$\bf{Figure\ 4.}$ Optimization layer for dispersive transmon readout. "
        r"$\bf{(a)}$ Normalized log-sensitivities of $F_{\rm assign}$ to 7 parameters "
        r"at REFERENCE (Marxer arXiv:2508.16437); sensitivities computed with parameters "
        r"treated as independent axes via chi_scale. Day-10 cross-check "
        rf"$|S_g - 2 S_\chi| / |2 S_\chi|$ = {residual_pct:.2f}% (raw: "
        rf"$S_\chi$={cross['S_chi']:+.3f}, $S_g$={cross['S_g']:+.3f}; "
        r"denominator is small because $|S_\chi|$ sits at the tornado noise floor, "
        r"so a few-percent residual does not indicate Q1 orthogonality failure). "
        r"$\bf{(b)}$ Analytic regime map (Bengtsson 2024 PRL section II + Blais RMP 2021 section V.B); "
        rf"Lindblad-validated at {n_validation_points} points, max deviation {max_dev_pct:.2f}%. "
        r"Hazra 2407.10934 (dimon, non-standard $\chi$-mediation) cited in reference list "
        r"but not plotted. "
        r"$\bf{(c)}$ Pareto frontiers for 3 parameter-anchored variants of REFERENCE "
        r"(V1=REFERENCE, V2=$T_1$=40 $\mu$s, V3=$T_1$=20 $\mu$s + $\kappa/2\pi$=6 MHz). "
        r"Curves represent the Pareto frontier predicted by this work's simulator under "
        r"parameter substitution - NOT the frontier achievable on the cited devices' native "
        r"hardware. Frontiers transition between two dispersive-readout operating regimes "
        r"around $\tau_{\rm max} \approx$ 450 ns: the low-photon regime "
        r"($\varepsilon_0 \approx 8 \times 10^7$ rad/s, peak \#1 - dispersive shift clean, "
        r"shot-noise-limited contrast) optimal at short integration, and the "
        r"dispersive-saturation regime ($\varepsilon_0 \approx 1.4 \times 10^8$ rad/s, "
        r"peak \#2 - $\bar{n}_{\rm phot}$ near bifurcation threshold, decoherence-limited "
        r"integration) optimal at long integration. The specific transition $\tau_{\rm max}$ "
        r"is set by the trade-off between the two regimes. "
        r"Closed-loop scope: fitted ($T_1$, $T_2$, $\omega_q$) over fixed REFERENCE "
        r"resonator and coupling; full closed-loop including resonator spectroscopy is "
        r"post-submission roadmap. $n_{\rm shots} = 10^4$ throughout."
    )
    fig.text(0.01, -0.02, caption, wrap=True, fontsize=9, ha="left")

    out = _FIGURES_DIR / "fig4_optimization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
