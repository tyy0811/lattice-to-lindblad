"""Standalone Figure 4 Panel (b) — REFERENCE-family-anchored regime map.

Item-15 amendment (MODULE_4_SPEC.md §0.3): the surface is the per-level
analytic F_assign computed under Option-C κ-sweep at fixed REFERENCE
per-level dispersive shifts (χ_0, χ_1) and drive (ε, T_window).
**Not a universal F(χ/κ, γ_1τ) law** — see derivation document and
caption blockquote in regime_map.py.

Outputs:
  - figures/fig4_panel_b_regime.png  (150 dpi)
  - figures/fig4_panel_b_validation.yaml  (regression-gate artifact;
    cited in Figure 4 caption max-deviation claim)

Run from repo root:
  PYTHONPATH=. python 06_Dispersive_Readout/scripts/fig4_panel_b_regime.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dispersive_readout.optimization.regime_map import (
    compute_analytic_regime_map,
    validate_analytic_vs_lindblad,
    purcell_boundary,
    dispersive_breakdown_boundary,
    resonator_too_slow_boundary,
    PUBLISHED_DEVICE_POINTS,
)


_MARKER_COLOR_MAP = {"warm_orange": "#E8801A", "red": "#C0392B"}

# Boundary-line labels for the legend. Each function is called once inside
# render_regime_map; the label string matches the caption phrasing.
_BOUNDARY_LABELS = {
    "purcell":   "Purcell boundary",
    "breakdown": "Dispersive breakdown (χ/κ ≲ 1)",
    "slow":      "Resonator-too-slow (κ·τ_readout ≲ 1)",
}


def render_regime_map(ax: plt.Axes, validation: dict, grid: dict) -> None:
    x_axis = grid["chi_over_kappa_axis"]
    y_axis = grid["gamma_1_tau_axis"]
    F = grid["F_grid"]

    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    pcm = ax.pcolormesh(X, Y, F, cmap="viridis", shading="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(pcm, ax=ax, label=r"$F_{\rm assign}$")

    # Denser contour set reveals more structure in the 0.8-1.0 regime where
    # most of the interesting physics sits; 0.5/0.7 anchor the low-F corners.
    cs = ax.contour(
        X, Y, F, levels=[0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
        colors="white", linestyles="dashed", linewidths=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=8)

    # Analytic boundaries — each labeled so the legend maps line to meaning.
    x_fine = np.logspace(np.log10(x_axis[0]), np.log10(x_axis[-1]), 200)
    for bkey, yfn in (
        ("purcell",   purcell_boundary),
        ("breakdown", dispersive_breakdown_boundary),
        ("slow",      resonator_too_slow_boundary),
    ):
        y_boundary = yfn(x_fine)
        mask = (y_boundary >= y_axis[0]) & (y_boundary <= y_axis[-1])
        ax.plot(
            x_fine[mask], y_boundary[mask],
            color="grey", linestyle="--", linewidth=1.0,
            label=_BOUNDARY_LABELS[bkey],
        )

    # 4 device markers, each labeled by its publication anchor so the legend
    # pairs marker → device. Estimated points (Bengtsson, Garnet) keep their
    # red-family color per Q6 encoding; primary published points (Marxer
    # Q1/Q2) stay in warm_orange.
    for p in PUBLISHED_DEVICE_POINTS:
        color = _MARKER_COLOR_MAP[p.marker_color]
        ax.scatter(
            [p.chi_over_kappa], [p.gamma_1_tau],
            marker=p.marker, s=180, c=color, edgecolors="white", linewidths=1.5,
            zorder=10, label=p.label,
        )
        if "Marxer Q1" in p.label and p.reported_F_assign is not None:
            q1_sim = next(
                pt["F_lindblad"] for pt in validation["per_point"]
                if abs(pt["chi_over_kappa"] - p.chi_over_kappa) < 1e-6
            )
            ax.annotate(
                f"$F_{{\\rm sim}} = {q1_sim:.4f}$",
                xy=(p.chi_over_kappa, p.gamma_1_tau),
                xytext=(8, 8), textcoords="offset points", fontsize=9,
                color="white",
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\chi/\kappa \equiv |\chi_0 - \chi_1|/\kappa$")
    ax.set_ylabel(r"$\gamma_1 \cdot \tau_{\rm readout}$")

    # Legend: markers (devices) on upper-right, lines (boundaries) on lower-
    # left to avoid overlap with the high-F ridge and the low-F corners.
    handles, labels = ax.get_legend_handles_labels()
    device_idx = [i for i, lbl in enumerate(labels) if lbl not in _BOUNDARY_LABELS.values()]
    boundary_idx = [i for i, lbl in enumerate(labels) if lbl in _BOUNDARY_LABELS.values()]
    if device_idx:
        leg_dev = ax.legend(
            [handles[i] for i in device_idx], [labels[i] for i in device_idx],
            loc="upper right", fontsize=7.5, framealpha=0.9,
            title="Published devices", title_fontsize=8,
        )
        ax.add_artist(leg_dev)
    if boundary_idx:
        ax.legend(
            [handles[i] for i in boundary_idx], [labels[i] for i in boundary_idx],
            loc="lower left", fontsize=7.5, framealpha=0.9,
            title="Analytic boundaries", title_fontsize=8,
        )

    chi_0, chi_1 = grid["chi_per_level_anchor"]
    eps = grid["epsilon"]
    Tw = grid["T_window"]
    max_dev_pct = validation["max_deviation_fractional"] * 100.0
    # Short title + two compact subtitle lines. Detail (validation summary)
    # keeps a visible anchor on the figure itself; anchoring parameters on
    # line 2 stay readable within the axis width.
    ax.set_title(
        "REFERENCE-family-anchored regime map\n"
        rf"$\chi_0/(2\pi)=${chi_0/(2*np.pi)/1e6:.2f} MHz, "
        rf"$\chi_1/(2\pi)=${chi_1/(2*np.pi)/1e6:.2f} MHz, "
        rf"$\varepsilon/(2\pi)=${eps/(2*np.pi)/1e6:.2f} MHz, "
        rf"$T_{{\rm window}}=${Tw*1e9:.0f} ns" "\n"
        rf"Lindblad-validated max deviation $\leq{max_dev_pct:.2f}\%$",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main() -> None:
    grid = compute_analytic_regime_map()
    validation = validate_analytic_vs_lindblad()

    out_dir = Path("06_Dispersive_Readout/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert tuple anchor to list for YAML round-trip readability
    validation_yaml = {
        **validation,
        "chi_per_level_anchor": list(validation["chi_per_level_anchor"]),
    }
    with open(out_dir / "fig4_panel_b_validation.yaml", "w") as f:
        yaml.safe_dump(validation_yaml, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    render_regime_map(ax, validation, grid)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_panel_b_regime.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {out_dir / 'fig4_panel_b_regime.png'}")
    print(f"Wrote {out_dir / 'fig4_panel_b_validation.yaml'}")


if __name__ == "__main__":
    main()
