"""Regenerate fig4_data.yaml - the O9 regression-gate artifact.

Pins per-sensitivity S_theta values, per-Pareto-point (F_opt, epsilon_0, tau),
regime-grid F-values hash, regime-grid chi_per_level anchor, and Day-10
cross-check residual at SEED=42.

Data-source strategy (Day-13):
  - Sensitivities + Day-10 cross-check: computed fresh (~6 min of Lindblad).
  - Regime grid: computed fresh (instant, pure analytic).
  - Pareto frontiers: loaded from committed fig4_panel_c_data.yaml cache by
    default; --use-modal forces fresh Modal dispatch (~45 min). The cached
    artifact is already the deterministic output of Task 14's Modal run
    at SEED=42, so re-computing it produces identical numbers.

Output: 06_Dispersive_Readout/figures/fig4_data.yaml
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.optimization.sensitivity import (
    compute_all_sensitivities, day_10_cross_check_s_g_vs_s_chi,
)
from dispersive_readout.optimization.regime_map import compute_analytic_regime_map
from dispersive_readout.optimization.pareto import (
    PARETO_DEVICE_VARIANTS, TAU_MAX_GRID_NS,
    ParetoPoint, build_variant, compute_pareto_frontier,
)


_FIGURES_DIR = Path("06_Dispersive_Readout/figures")


def _load_frontiers_from_cache() -> dict[str, list[ParetoPoint]]:
    cache_path = _FIGURES_DIR / "fig4_panel_c_data.yaml"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing; run scripts/fig4_panel_c_pareto.py "
            "(USE_MODAL=1) first, or invoke this script with --use-modal."
        )
    with open(cache_path) as f:
        raw = yaml.safe_load(f)
    return {label: [ParetoPoint(**r) for r in records] for label, records in raw.items()}


def _compute_frontiers_fresh(use_modal: bool) -> dict[str, list[ParetoPoint]]:
    frontiers: dict[str, list[ParetoPoint]] = {}
    for spec in PARETO_DEVICE_VARIANTS:
        device = build_variant(spec)
        frontiers[spec["label"]] = compute_pareto_frontier(
            device, tau_max_values=TAU_MAX_GRID_NS * 1e-9,
            device_label=spec["label"], use_modal=use_modal,
        )
    return frontiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pareto-source",
        choices=("cache", "modal", "serial"),
        default="cache",
        help=(
            "cache: read committed fig4_panel_c_data.yaml (instant; default; "
            "matches Task 14's Modal output bit-for-bit). "
            "modal: re-dispatch via Modal (~45 min). "
            "serial: re-compute serial (~4.5 hr; only if Modal unavailable)."
        ),
    )
    args = parser.parse_args()

    print("Computing sensitivities + Day-10 cross-check (fresh; ~6 min)")
    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)
    cross = day_10_cross_check_s_g_vs_s_chi(op)

    print("Computing analytic regime grid (instant)")
    grid = compute_analytic_regime_map()

    print(f"Loading Pareto frontiers: source={args.pareto_source}")
    if args.pareto_source == "cache":
        frontiers = _load_frontiers_from_cache()
    elif args.pareto_source == "modal":
        frontiers = _compute_frontiers_fresh(use_modal=True)
    else:
        frontiers = _compute_frontiers_fresh(use_modal=False)

    frontiers_payload = {
        label: [
            {
                "tau_max_ns": round(p.tau_max * 1e9, 3),
                "epsilon_0_opt": float(p.epsilon_0_opt),
                "tau_opt_ns": round(p.tau_opt * 1e9, 3),
                "F_assign_opt": round(p.F_assign_opt, 6),
            }
            for p in pts
        ]
        for label, pts in frontiers.items()
    }

    chi_0, chi_1 = grid["chi_per_level_anchor"]
    payload = {
        "seed": 42,
        "sensitivities": [
            {
                "parameter": s.parameter,
                "S": round(s.sensitivity, 4),
                "sigma_S": round(s.sensitivity_uncertainty, 5),
                "F_reference": round(s.F_reference, 5),
            }
            for s in sens
        ],
        "day_10_cross_check": {
            "S_chi": round(cross["S_chi"], 4),
            "S_g": round(cross["S_g"], 4),
            "residual_fractional": round(cross["residual_fractional"], 4),
        },
        "regime_grid_hash": hashlib.sha256(
            np.ascontiguousarray(grid["F_grid"]).tobytes()
        ).hexdigest(),
        "regime_chi_per_level_anchor": [float(chi_0), float(chi_1)],
        "regime_epsilon": float(grid["epsilon"]),
        "regime_T_window": float(grid["T_window"]),
        "pareto_frontiers": frontiers_payload,
    }

    out = _FIGURES_DIR / "fig4_data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
