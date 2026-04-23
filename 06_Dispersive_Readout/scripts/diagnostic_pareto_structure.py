"""Day-13 expanded diagnostic B: verify Pareto optimum structure.

Per user directive (Day-13 sharpening), run 3 diagnostics on 3 devices
(REFERENCE + low-T_1 extreme + high-T_1 extreme from the 50-device
recovery harness):

  1. Does eps_opt shift under finer warm-start grid?
     Compare find_pareto_point(n_warm_start_grid_side=5) vs =20 on each
     of 3 devices at tau_max=500 ns.
  2. Does tau_opt stay at boundary under all grid densities?
     Report tau_opt at both grid densities.
  3. Is F(eps, tau) flat or sharply peaked near the claimed optimum?
     Scan F_analytic on a 21x21 grid around (eps_star, tau_star) for
     each device: eps in [0.2x, 4x] log-spaced, tau in [0.5x, 1.5x]
     linear-spaced around REFERENCE's (2.5075e8, 500e-9).

Output: prints to stdout + writes figures/diagnostic_pareto_structure.yaml
(diagnostic only; not a regression artifact).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from dispersive_readout.optimization.modal_pareto import (
    app, pareto_one_tuple_with_grid_density, F_analytic_at_point,
)
from dispersive_readout.physics.config import REFERENCE_DEVICE


_TAU_MAX = 500e-9
_EPS_REF = 2.5075e8        # observed warm-start winner across all devices
_TAU_REF = _TAU_MAX


def _build_synthetic(entry: dict):
    new_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=1.0 / entry["T_1"],
        gamma_phi=max(
            1.0 / entry["T_2_echo"] - 0.5 / entry["T_1"], 0.0,
        ),
        n_th=max(
            float(entry.get("thermal_offset", 0.0)),
            REFERENCE_DEVICE.decoherence.n_th,
        ),
    )
    return replace(REFERENCE_DEVICE, decoherence=new_dec)


def main() -> None:
    payload = yaml.safe_load(
        Path("06_Dispersive_Readout/figures/recovery_coverage_report.yaml").read_text()
    )
    devs_raw = payload["devices"]

    # Pick 3 representative devices: REFERENCE + T_1 extremes from harness.
    by_T1 = sorted(range(len(devs_raw)), key=lambda i: devs_raw[i]["T_1"])
    idx_low, idx_high = by_T1[0], by_T1[-1]
    dev_low, dev_high = _build_synthetic(devs_raw[idx_low]), _build_synthetic(devs_raw[idx_high])
    dev_labels = [
        ("REFERENCE (T_1=30us)", REFERENCE_DEVICE),
        (f"harness_low_T1 (idx={idx_low}, T_1={devs_raw[idx_low]['T_1']*1e6:.1f}us)", dev_low),
        (f"harness_high_T1 (idx={idx_high}, T_1={devs_raw[idx_high]['T_1']*1e6:.1f}us)", dev_high),
    ]

    # ── Diagnostic 1+2: Pareto at 5-point vs 20-point warm-start ──────
    print("=== Diagnostic 1+2: warm-start grid density ===")
    devices_for_pareto = [d for _, d in dev_labels] * 2
    tau_maxes = [_TAU_MAX] * 6
    n_warms = [5, 5, 5, 20, 20, 20]
    print("Dispatching 6 Pareto calls (3 devices x {5, 20}-point warm-start)")
    with app.run():
        pareto_results = list(pareto_one_tuple_with_grid_density.map(
            devices_for_pareto, tau_maxes, n_warms,
        ))
    grid_comparison = []
    for i, (label, _) in enumerate(dev_labels):
        p5, p20 = pareto_results[i], pareto_results[i + 3]
        deps_rel = abs(p20.epsilon_0_opt - p5.epsilon_0_opt) / p5.epsilon_0_opt
        dtau_rel = abs(p20.tau_opt - p5.tau_opt) / p5.tau_opt
        dF_abs = p20.F_assign_opt - p5.F_assign_opt
        tau_at_boundary_5 = p5.tau_opt >= _TAU_MAX * 0.999
        tau_at_boundary_20 = p20.tau_opt >= _TAU_MAX * 0.999
        print(f"\n  {label}")
        print(f"    5-pt:  eps={p5.epsilon_0_opt:.4e}  tau={p5.tau_opt*1e9:6.2f}ns  F={p5.F_assign_opt:.6f}  "
              f"boundary={tau_at_boundary_5}  loss={p5.dominant_loss_channel}")
        print(f"    20-pt: eps={p20.epsilon_0_opt:.4e}  tau={p20.tau_opt*1e9:6.2f}ns  F={p20.F_assign_opt:.6f}  "
              f"boundary={tau_at_boundary_20}  loss={p20.dominant_loss_channel}")
        print(f"    drift: deps={deps_rel*100:.3f}%  dtau={dtau_rel*100:.3f}%  dF={dF_abs:+.6f}")
        grid_comparison.append({
            "label": label,
            "pareto_5pt": {"eps": float(p5.epsilon_0_opt), "tau_ns": float(p5.tau_opt*1e9),
                           "F": float(p5.F_assign_opt), "boundary": bool(tau_at_boundary_5)},
            "pareto_20pt": {"eps": float(p20.epsilon_0_opt), "tau_ns": float(p20.tau_opt*1e9),
                            "F": float(p20.F_assign_opt), "boundary": bool(tau_at_boundary_20)},
            "delta_eps_rel": float(deps_rel),
            "delta_tau_rel": float(dtau_rel),
            "delta_F_abs": float(dF_abs),
        })

    # ── Diagnostic 3: F-surface 21x21 grid around the claimed optimum ─
    print("\n=== Diagnostic 3: F(eps, tau) surface near claimed optimum ===")
    # Log in eps, linear in tau (since tau is constrained range 250-750 ns)
    eps_axis = np.logspace(np.log10(5e7), np.log10(1e9), 21)       # 0.2x .. 4x around 2.5e8
    tau_axis = np.linspace(250e-9, 750e-9, 21)                      # 0.5x .. 1.5x around 500 ns
    surface_points = [(float(e), float(t)) for e in eps_axis for t in tau_axis]   # 441
    n_points_per_device = len(surface_points)

    all_devices = []
    all_eps = []
    all_tau = []
    for _, d in dev_labels:
        all_devices.extend([d] * n_points_per_device)
        all_eps.extend([pt[0] for pt in surface_points])
        all_tau.extend([pt[1] for pt in surface_points])
    n_total = len(all_eps)
    print(f"Dispatching {n_total} F-surface evals (3 devices x 21x21 grid)")

    with app.run():
        F_values = list(F_analytic_at_point.map(all_devices, all_eps, all_tau))

    # Group by device and analyze
    surface_summary = []
    for i, (label, _) in enumerate(dev_labels):
        dev_Fs = np.array(F_values[i*n_points_per_device:(i+1)*n_points_per_device])
        F_grid = dev_Fs.reshape(21, 21)   # [eps_idx, tau_idx]

        F_max = np.nanmax(F_grid)
        F_min = np.nanmax([np.nanmin(F_grid), 0.0])   # guard nan
        max_idx_flat = np.nanargmax(F_grid)
        eps_star_idx, tau_star_idx = np.unravel_index(max_idx_flat, F_grid.shape)
        eps_star = float(eps_axis[eps_star_idx])
        tau_star = float(tau_axis[tau_star_idx])

        # Flatness probe: range of F within ±20% in eps and ±10% in tau of claimed optimum
        eps_within = np.abs(np.log10(eps_axis / _EPS_REF)) <= np.log10(1.2)
        tau_within = np.abs(tau_axis - _TAU_REF) / _TAU_REF <= 0.10
        F_local = F_grid[np.ix_(eps_within, tau_within)]
        F_local_range = float(np.nanmax(F_local) - np.nanmin(F_local))

        print(f"\n  {label}")
        print(f"    Grid argmax:  eps={eps_star:.4e} (expected {_EPS_REF:.4e})")
        print(f"                  tau={tau_star*1e9:.1f}ns (expected {_TAU_REF*1e9:.0f}ns)")
        print(f"    F_max on grid  = {F_max:.6f}")
        print(f"    F_min on grid  = {F_min:.6f}")
        print(f"    F_max - F_min over full grid = {F_max - F_min:.6f}")
        print(f"    Flatness near optimum (eps ±20%, tau ±10%): "
              f"F range = {F_local_range:.6f}")
        surface_summary.append({
            "label": label,
            "grid_argmax_eps": eps_star,
            "grid_argmax_tau_ns": tau_star * 1e9,
            "F_max": float(F_max),
            "F_min": float(F_min),
            "F_range_full_grid": float(F_max - F_min),
            "F_range_near_optimum_eps20pct_tau10pct": float(F_local_range),
        })

    # ── Persist for audit trail ───────────────────────────────────────
    out = Path("06_Dispersive_Readout/figures/diagnostic_pareto_structure.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(
            {
                "grid_density_comparison": grid_comparison,
                "F_surface_summary": surface_summary,
                "eps_axis": [float(x) for x in eps_axis],
                "tau_axis_ns": [float(x * 1e9) for x in tau_axis],
                "tau_max_ns": _TAU_MAX * 1e9,
                "device_indices_low_high": [int(idx_low), int(idx_high)],
            },
            f,
            sort_keys=False,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
