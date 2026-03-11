#!/usr/bin/env python3
"""Schwinger symmetry-resolved entanglement driver.

Scientific responsibility:
    Compute sector weights for bond-charge decomposition at a bipartition.
Main inputs:
    Schwinger parameters (N, mass, coupling, chi), cut, output options.
Main outputs:
    symmetry_resolved_entanglement*.csv, *.png, and metadata JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "l2l").exists():
        sys.path.insert(0, str(_p))
        break

import matplotlib.pyplot as plt
import numpy as np

from l2l.entanglement import extract_schmidt_values_by_sector, compute_sector_weights
from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter


def sanitize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_HERE.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_total_entropy_from_sector_lambdas(sector_to_lambdas: dict[str, list[float]]) -> float:
    """Compute total von Neumann entropy from all Schmidt values at the cut."""
    all_vals = []
    for lambdas in sector_to_lambdas.values():
        arr = np.asarray(lambdas, dtype=float)
        if arr.size:
            all_vals.append(arr * arr)

    if not all_vals:
        return 0.0

    probs = np.concatenate(all_vals)
    positive = probs > 0.0
    return float(-np.sum(probs[positive] * np.log(probs[positive])))


def schmidt_to_entanglement_levels(sorted_schmidt_values: np.ndarray) -> np.ndarray:
    """Compute xi = -log(lambda^2) from Schmidt values sorted descending."""
    arr = np.asarray(sorted_schmidt_values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    arr = np.sort(arr)[::-1]
    pos = arr > 0.0
    return -np.log(arr[pos] * arr[pos])


def parse_mass_list(mass_list: str) -> list[float]:
    """Parse comma-separated masses for the fixed 4-point trend benchmark."""
    try:
        masses = [float(x.strip()) for x in mass_list.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --mass-list '{mass_list}'. Use comma-separated floats.") from exc

    required = [0.05, 0.08, 0.125, 0.20]
    if len(masses) != 4:
        raise ValueError(
            "--mass-list mode is restricted to exactly 4 masses for the benchmark: "
            "0.05,0.08,0.125,0.20"
        )

    sorted_masses = sorted(masses)
    sorted_required = sorted(required)
    if not all(np.isclose(a, b, rtol=0.0, atol=1e-12) for a, b in zip(sorted_masses, sorted_required)):
        raise ValueError(
            "--mass-list currently supports only the fixed benchmark masses: 0.05,0.08,0.125,0.20"
        )
    return masses


def compute_sector_statistics(
    sector_to_lambdas: dict[str, list[float]],
    *,
    model: str,
    N: int,
    mass: float,
    coupling: float,
    chi: int,
    cut: int,
    mass_label: str,
) -> dict[str, object]:
    """Compute sector-wise weights and entropy decomposition terms."""
    sector_to_weight = compute_sector_weights(sector_to_lambdas)
    total_weight = float(sum(sector_to_weight.values()))
    if not np.isclose(total_weight, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Sector weights do not sum to 1 (mass={mass:g}): {total_weight:.8f}")

    sector_to_count = {label: len(lambdas) for label, lambdas in sector_to_lambdas.items()}
    sorted_by_weight = sorted(sector_to_weight.keys(), key=lambda s: sector_to_weight[s], reverse=True)

    sector_to_cum_weight: dict[str, float] = {}
    cum = 0.0
    for sector in sorted_by_weight:
        cum += float(sector_to_weight[sector])
        sector_to_cum_weight[sector] = cum

    sector_to_shannon: dict[str, float] = {}
    sector_to_intrasector_entropy: dict[str, float] = {}
    sector_to_weighted_intra: dict[str, float] = {}

    for sector, lambdas in sector_to_lambdas.items():
        p_q = float(sector_to_weight[sector])
        sector_to_shannon[sector] = float(-p_q * np.log(p_q)) if p_q > 0.0 else 0.0

        vals_q = np.asarray(lambdas, dtype=float)
        vals_q = vals_q * vals_q
        if p_q > 0.0 and vals_q.size:
            r_q = vals_q / p_q
            pos = r_q > 0.0
            s_q = float(-np.sum(r_q[pos] * np.log(r_q[pos])))
        else:
            s_q = 0.0
        sector_to_intrasector_entropy[sector] = s_q
        sector_to_weighted_intra[sector] = p_q * s_q

    shannon_sector_entropy = float(sum(sector_to_shannon.values()))
    weighted_intrasector_entropy_sum = float(sum(sector_to_weighted_intra.values()))
    reconstructed_total_entropy = shannon_sector_entropy + weighted_intrasector_entropy_sum
    total_entropy_direct = compute_total_entropy_from_sector_lambdas(sector_to_lambdas)
    reconstruction_error = reconstructed_total_entropy - total_entropy_direct
    if not np.isclose(reconstructed_total_entropy, total_entropy_direct, rtol=1e-8, atol=1e-10):
        raise ValueError(
            f"Entropy decomposition inconsistent (mass={mass:g}): "
            f"reconstructed={reconstructed_total_entropy:.12e}, direct={total_entropy_direct:.12e}, "
            f"error={reconstruction_error:.3e}"
        )

    schmidt_blocks = []
    for lambdas in sector_to_lambdas.values():
        block = np.asarray(lambdas, dtype=float).ravel()
        if block.size:
            schmidt_blocks.append(block)
    if schmidt_blocks:
        sorted_schmidt_values = np.sort(np.concatenate(schmidt_blocks))[::-1]
    else:
        sorted_schmidt_values = np.array([], dtype=float)
    xi_levels = schmidt_to_entanglement_levels(sorted_schmidt_values)

    rows = []
    for sector in sorted_by_weight:
        rows.append(
            {
                "sector": sector,
                "weight": float(sector_to_weight[sector]),
                "cum_weight": float(sector_to_cum_weight[sector]),
                "shannon_contribution": float(sector_to_shannon[sector]),
                "n_schmidt": int(sector_to_count[sector]),
                "cut": int(cut),
                "model": model,
                "N": int(N),
                "m_over_g": float(mass),
                "x": float(coupling),
                "chi": int(chi),
                "mass_label": mass_label,
                "intrasector_entropy": float(sector_to_intrasector_entropy[sector]),
                "weighted_intrasector_entropy": float(sector_to_weighted_intra[sector]),
            }
        )

    return {
        "rows": rows,
        "sorted_by_weight": sorted_by_weight,
        "sector_to_weight": sector_to_weight,
        "sector_to_shannon": sector_to_shannon,
        "total_weight": total_weight,
        "sorted_schmidt_values": sorted_schmidt_values,
        "xi_levels": xi_levels,
        "decomposition_available": True,
        "sector_entropy_reconstruction": {
            "shannon_sector_entropy": shannon_sector_entropy,
            "weighted_intrasector_entropy_sum": weighted_intrasector_entropy_sum,
            "reconstructed_total_entropy": reconstructed_total_entropy,
            "direct_total_entropy": total_entropy_direct,
            "reconstruction_error": reconstruction_error,
        },
    }


def extract_named_sector_weight(stats: dict[str, Any], sector_label: str) -> float:
    """Safely extract one sector weight from the per-mass statistics."""
    return float(stats["sector_to_weight"].get(sector_label, 0.0))


def run_mass_point(args: argparse.Namespace, mass_value: float, mass_label: str) -> dict[str, object]:
    """Compute Schwinger state and sector statistics for one mass point."""
    print(f"Computing Schwinger ground state for m/g={mass_value}...")
    adapter = SchwingerMassGapAdapter(m_over_g=mass_value, E0=0.0)
    try:
        result = adapter.dmrg_solve_point(args.N, {"x": args.coupling}, chi=args.chi, return_mps=True)
    except ModuleNotFoundError as exc:
        if exc.name == "tenpy":
            raise RuntimeError(
                "Missing dependency 'tenpy'. Use the project interpreter "
                "'.venv/bin/python' from the repository root."
            ) from exc
        raise
    psi = result["psi0"]
    print(f"E0 = {result['E0']:.10f}")

    print(f"Extracting Schmidt values by sector at cut {args.cut}...")
    sector_to_lambdas = extract_schmidt_values_by_sector(psi, args.cut)
    stats = compute_sector_statistics(
        sector_to_lambdas,
        model="schwinger",
        N=args.N,
        mass=mass_value,
        coupling=args.coupling,
        chi=args.chi,
        cut=args.cut,
        mass_label=mass_label,
    )
    print()
    return {
        "mass": mass_value,
        "mass_label": mass_label,
        "stats": stats,
    }


def build_mass_trend_summary(mass_results: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    """Build per-mass benchmark summary rows and dominant sectors used for trend plots."""
    summary_rows: list[dict[str, object]] = []

    all_labels = set()
    for result in mass_results:
        all_labels.update(result["stats"]["sector_to_weight"].keys())

    if all_labels:
        ranked = sorted(
            all_labels,
            key=lambda label: max(r["stats"]["sector_to_weight"].get(label, 0.0) for r in mass_results),
            reverse=True,
        )
    else:
        ranked = []

    required_labels = [label for label in ("q0", "q-2") if label in all_labels]
    non_required_ranked = [label for label in ranked if label not in required_labels]
    dominant = required_labels + non_required_ranked
    dominant = dominant[:4]

    for result in sorted(mass_results, key=lambda x: float(x["mass"])):
        stats = result["stats"]
        sorted_by_weight = stats["sorted_by_weight"]
        rec = stats["sector_entropy_reconstruction"]
        row = {
            "N": int(stats["rows"][0]["N"]) if stats["rows"] else None,
            "mass": float(result["mass"]),
            "coupling": float(stats["rows"][0]["x"]) if stats["rows"] else None,
            "chi": int(stats["rows"][0]["chi"]) if stats["rows"] else None,
            "cut": int(stats["rows"][0]["cut"]) if stats["rows"] else None,
            "n_sectors": len(sorted_by_weight),
            "top_2_cum_weight": float(sum(stats["sector_to_weight"][s] for s in sorted_by_weight[:2])),
            "top_3_cum_weight": float(sum(stats["sector_to_weight"][s] for s in sorted_by_weight[:3])),
            "q0_weight": extract_named_sector_weight(stats, "q0"),
            "q_minus2_weight": extract_named_sector_weight(stats, "q-2"),
            "shannon_sector_entropy": float(rec["shannon_sector_entropy"]),
            "weighted_intrasector_entropy_sum": float(rec["weighted_intrasector_entropy_sum"]),
            "reconstructed_total_entropy": float(rec["reconstructed_total_entropy"]),
            "reconstruction_error": float(rec["reconstruction_error"]),
        }
        summary_rows.append(row)

    return summary_rows, dominant


def save_mass_trend_summary_csv(outdir: Path, summary_rows: list[dict[str, object]], tag_suffix: str) -> Path:
    """Save compact mass-trend benchmark table."""
    filepath = outdir / f"symmetry_resolved_mass_trend_summary{tag_suffix}.csv"
    if not summary_rows:
        raise ValueError("No mass-trend summary rows were produced.")

    columns = [
        "N",
        "mass",
        "coupling",
        "chi",
        "cut",
        "n_sectors",
        "top_2_cum_weight",
        "top_3_cum_weight",
        "q0_weight",
        "q_minus2_weight",
        "shannon_sector_entropy",
        "weighted_intrasector_entropy_sum",
        "reconstructed_total_entropy",
        "reconstruction_error",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return filepath


def build_bridge_summary_rows(mass_results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build bridge rows connecting symmetry-resolved and ordinary entanglement observables."""
    rows: list[dict[str, object]] = []
    for result in sorted(mass_results, key=lambda x: float(x["mass"])):
        stats = result["stats"]
        rec = stats["sector_entropy_reconstruction"]
        total_entropy = float(rec["direct_total_entropy"])
        reconstructed = float(rec["reconstructed_total_entropy"])
        if not np.isclose(total_entropy, reconstructed, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"Bridge consistency failure at mass={result['mass']}: "
                f"direct={total_entropy:.12e}, reconstructed={reconstructed:.12e}"
            )

        sorted_by_weight = stats["sorted_by_weight"]
        top2 = float(sum(stats["sector_to_weight"][s] for s in sorted_by_weight[:2]))
        q0_weight = extract_named_sector_weight(stats, "q0")
        q_minus2_weight = extract_named_sector_weight(stats, "q-2")
        xi_levels = np.asarray(stats["xi_levels"], dtype=float)
        xi_0 = float(xi_levels[0]) if xi_levels.size > 0 else float("nan")
        xi_1 = float(xi_levels[1]) if xi_levels.size > 1 else float("nan")
        xi_2 = float(xi_levels[2]) if xi_levels.size > 2 else float("nan")
        schmidt = np.asarray(stats["sorted_schmidt_values"], dtype=float)
        schmidt_sq = schmidt * schmidt
        retained_weight_rank_2 = float(np.sum(schmidt_sq[:2])) if schmidt_sq.size >= 2 else float("nan")
        retained_weight_rank_3 = float(np.sum(schmidt_sq[:3])) if schmidt_sq.size >= 3 else float("nan")

        first_row = stats["rows"][0] if stats["rows"] else {}
        rows.append(
            {
                "mass": float(result["mass"]),
                "N": int(first_row.get("N", -1)),
                "coupling": float(first_row.get("x", np.nan)),
                "chi": int(first_row.get("chi", -1)),
                "cut": int(first_row.get("cut", -1)),
                "total_entropy": total_entropy,
                "q0_weight": q0_weight,
                "q_minus2_weight": q_minus2_weight,
                "top_2_cum_weight": top2,
                "shannon_sector_entropy": float(rec["shannon_sector_entropy"]),
                "weighted_intrasector_entropy_sum": float(rec["weighted_intrasector_entropy_sum"]),
                "xi_0": xi_0,
                "xi_1": xi_1,
                "xi_2": xi_2,
                "retained_weight_rank_2": retained_weight_rank_2,
                "retained_weight_rank_3": retained_weight_rank_3,
            }
        )
    return rows


def save_bridge_summary_csv(outdir: Path, bridge_rows: list[dict[str, object]], tag_suffix: str) -> Path:
    """Save entropy/spectrum bridge table for mass-trend mode."""
    filepath = outdir / f"symmetry_resolved_entropy_spectrum_bridge{tag_suffix}.csv"
    if not bridge_rows:
        raise ValueError("No bridge summary rows were produced.")

    columns = [
        "mass",
        "N",
        "coupling",
        "chi",
        "cut",
        "total_entropy",
        "q0_weight",
        "q_minus2_weight",
        "top_2_cum_weight",
        "shannon_sector_entropy",
        "weighted_intrasector_entropy_sum",
        "xi_0",
        "xi_1",
        "xi_2",
        "retained_weight_rank_2",
        "retained_weight_rank_3",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in bridge_rows:
            writer.writerow(row)
    return filepath


def plot_bridge_summary_two_panel(outdir: Path, bridge_rows: list[dict[str, object]], tag_suffix: str) -> Path:
    """Plot compact bridge figure linking decomposition terms and ordinary xi levels."""
    filepath = outdir / f"symmetry_resolved_entropy_spectrum_bridge{tag_suffix}.png"
    masses = [row["mass"] for row in bridge_rows]

    total_entropy = [row["total_entropy"] for row in bridge_rows]
    shannon = [row["shannon_sector_entropy"] for row in bridge_rows]
    intra = [row["weighted_intrasector_entropy_sum"] for row in bridge_rows]
    xi_0 = [row["xi_0"] for row in bridge_rows]
    xi_1 = [row["xi_1"] for row in bridge_rows]
    xi_2 = [row["xi_2"] for row in bridge_rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=150)
    ax0.plot(masses, total_entropy, marker="o", linewidth=2.0, label=r"$S_{\mathrm{total}}$")
    ax0.plot(masses, shannon, marker="s", linewidth=2.0, label=r"$H(\{p_q\})$")
    ax0.plot(masses, intra, marker="^", linewidth=2.0, label=r"$\sum_q p_q S_q$")
    ax0.set_xlabel("Mass $m/g$", fontsize=11)
    ax0.set_ylabel("Entropy", fontsize=11)
    ax0.set_title("A) Entropy decomposition vs mass", fontsize=11)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, frameon=False)

    ax1.plot(masses, xi_0, marker="o", linewidth=2.0, label=r"$\xi_0$")
    ax1.plot(masses, xi_1, marker="s", linewidth=2.0, label=r"$\xi_1$")
    ax1.plot(masses, xi_2, marker="^", linewidth=2.0, label=r"$\xi_2$")
    ax1.set_xlabel("Mass $m/g$", fontsize=11)
    ax1.set_ylabel(r"Entanglement levels $\xi_i$", fontsize=11)
    ax1.set_title("B) Leading ordinary entanglement levels", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=220, bbox_inches="tight")
    plt.close()
    return filepath


def save_sector_csv(
    outdir: Path,
    rows: list[dict[str, object]],
    tag_suffix: str,
) -> Path:
    """Save sector data to CSV."""
    filepath = outdir / f"symmetry_resolved_entanglement{tag_suffix}.csv"

    columns = [
        "sector",
        "weight",
        "cum_weight",
        "shannon_contribution",
        "n_schmidt",
        "cut",
        "model",
        "N",
        "m_over_g",
        "x",
        "chi",
        "mass_label",
        "intrasector_entropy",
        "weighted_intrasector_entropy",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return filepath


def plot_sector_weights_or_comparison(
    outdir: Path,
    mass_results: list[dict[str, object]],
    args: argparse.Namespace,
    tag_suffix: str,
) -> Path:
    """Generate single-mass or two-mass comparison figure."""
    filepath = outdir / f"symmetry_resolved_entanglement{tag_suffix}.png"

    if len(mass_results) == 1:
        stats = mass_results[0]["stats"]
        sector_to_weight = stats["sector_to_weight"]
        if args.sort_by == "sector":
            sorted_sectors = sorted(sector_to_weight.keys())
        else:
            sorted_sectors = stats["sorted_by_weight"]

        display_sectors = sorted_sectors[:args.max_sectors]
        weights = [sector_to_weight[s] for s in display_sectors]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        x = np.arange(len(display_sectors))
        bars = ax.bar(x, weights, color="C0", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(display_sectors, rotation=45, ha="right")
        ax.set_xlabel("Bond-charge sector", fontsize=12)
        ax.set_ylabel("Sector weight $p_q$", fontsize=12)
        ax.set_title(
            f"Symmetry-resolved entanglement: N={args.N}, m/g={args.mass}, x={args.coupling}, cut={args.cut}",
            fontsize=11,
        )
        ax.set_ylim(0, max(weights) * 1.1 if weights else 1.0)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, w in zip(bars, weights):
            if w > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{w:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    else:
        stats_a = mass_results[0]["stats"]
        stats_b = mass_results[1]["stats"]
        label_a = mass_results[0]["mass_label"]
        label_b = mass_results[1]["mass_label"]

        union_sectors = set(stats_a["sector_to_weight"]).union(set(stats_b["sector_to_weight"]))
        if args.sort_by == "sector":
            ranked = sorted(union_sectors)
        else:
            ranked = sorted(
                union_sectors,
                key=lambda s: max(stats_a["sector_to_weight"].get(s, 0.0), stats_b["sector_to_weight"].get(s, 0.0)),
                reverse=True,
            )
        display_sectors = ranked[:args.max_sectors]
        x = np.arange(len(display_sectors))
        width = 0.38

        weights_a = [stats_a["sector_to_weight"].get(s, 0.0) for s in display_sectors]
        weights_b = [stats_b["sector_to_weight"].get(s, 0.0) for s in display_sectors]
        shannon_a = [stats_a["sector_to_shannon"].get(s, 0.0) for s in display_sectors]
        shannon_b = [stats_b["sector_to_shannon"].get(s, 0.0) for s in display_sectors]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharex=True)
        ax0, ax1 = axes

        ax0.bar(x - width / 2, weights_a, width, label=label_a, color="C0", edgecolor="black", linewidth=0.5)
        ax0.bar(x + width / 2, weights_b, width, label=label_b, color="C1", edgecolor="black", linewidth=0.5)
        ax0.set_ylabel("Sector weight $p_q$", fontsize=11)
        ax0.set_xlabel("Bond-charge sector", fontsize=11)
        ax0.set_title("Sector weights", fontsize=11)
        ax0.grid(True, alpha=0.3, axis="y")
        ax0.legend(fontsize=9, frameon=False)

        ax1.bar(
            x - width / 2,
            shannon_a,
            width,
            label=label_a,
            color="C0",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax1.bar(
            x + width / 2,
            shannon_b,
            width,
            label=label_b,
            color="C1",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax1.set_ylabel("Shannon contribution $-p_q \\log p_q$", fontsize=11)
        ax1.set_xlabel("Bond-charge sector", fontsize=11)
        ax1.set_title("Sector Shannon contributions", fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(display_sectors, rotation=45, ha="right")

        fig.suptitle(
            f"Symmetry-resolved comparison: N={args.N}, x={args.coupling}, cut={args.cut}, chi={args.chi}",
            fontsize=11,
            y=1.02,
        )

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()

    return filepath


def plot_mass_trend_three_panel(
    outdir: Path,
    mass_results: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    dominant_labels: list[str],
    args: argparse.Namespace,
    tag_suffix: str,
) -> Path:
    """Generate compact 3-panel mass-trend figure for the fixed 4-mass benchmark."""
    filepath = outdir / f"symmetry_resolved_mass_trend{tag_suffix}.png"
    if not summary_rows:
        raise ValueError("Cannot plot mass trend without summary rows.")

    masses_sorted = [row["mass"] for row in summary_rows]
    mass_to_stats = {float(r["mass"]): r["stats"] for r in mass_results}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150)
    ax_a, ax_b, ax_c = axes

    sectors = dominant_labels
    x = np.arange(len(sectors))
    n_masses = len(masses_sorted)
    width = 0.18 if n_masses >= 4 else max(0.25, 0.8 / max(n_masses, 1))
    offsets = (np.arange(n_masses) - (n_masses - 1) / 2.0) * width

    for i, mass in enumerate(masses_sorted):
        weights = [mass_to_stats[mass]["sector_to_weight"].get(sector, 0.0) for sector in sectors]
        ax_a.bar(
            x + offsets[i],
            weights,
            width=width,
            label=f"m/g={mass:g}",
            edgecolor="black",
            linewidth=0.4,
        )
    ax_a.set_xlabel("Bond-charge sector label $q$", fontsize=11)
    ax_a.set_ylabel("Sector weight $p_q$", fontsize=11)
    ax_a.set_title("A) Dominant sector weights", fontsize=11)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sectors, rotation=35, ha="right")
    if sectors:
        ax_a.legend(fontsize=7.5, frameon=False, ncol=2)

    for i, mass in enumerate(masses_sorted):
        shannon = [mass_to_stats[mass]["sector_to_shannon"].get(sector, 0.0) for sector in sectors]
        ax_b.bar(
            x + offsets[i],
            shannon,
            width=width,
            label=f"m/g={mass:g}",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
    ax_b.set_xlabel("Bond-charge sector label $q$", fontsize=11)
    ax_b.set_ylabel(r"Shannon contribution $-p_q \log p_q$", fontsize=11)
    ax_b.set_title("B) Dominant sector Shannon terms", fontsize=11)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(sectors, rotation=35, ha="right")

    shannon = [row["shannon_sector_entropy"] for row in summary_rows]
    intra = [row["weighted_intrasector_entropy_sum"] for row in summary_rows]
    ax_c.plot(masses_sorted, shannon, marker="o", linewidth=2.0, label=r"$H(\{p_q\})$")
    ax_c.plot(masses_sorted, intra, marker="s", linewidth=2.0, label=r"$\sum_q p_q S_q$")
    ax_c.set_xlabel("Mass $m/g$", fontsize=11)
    ax_c.set_ylabel("Entropy contribution", fontsize=11)
    ax_c.set_title("C) Inter- vs intra-sector entropy", fontsize=11)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Symmetry-resolved mass trend: N={args.N}, x={args.coupling}, chi={args.chi}, cut={args.cut}",
        fontsize=11,
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=220, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    return filepath


def save_metadata(
    outdir: Path,
    args: argparse.Namespace,
    tag_suffix: str,
    *,
    mass_results: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    bridge_rows: list[dict[str, object]],
    bridge_summary_csv_path: Path | None,
    bridge_summary_figure_path: Path | None,
    dominant_sector_labels_used_in_figure: list[str],
    output_files: dict[str, Path],
) -> Path:
    """Save run metadata to JSON."""
    filepath = outdir / f"symmetry_resolved_metadata{tag_suffix}.json"

    serialized_outputs = {k: str(v) for k, v in output_files.items()}
    serialized_outputs["metadata"] = str(filepath)

    sector_definition = (
        "bond-charge-like sector label q taken from conserved-block Schmidt decomposition "
        "on the bipartition bond"
    )

    per_mass = []
    for result in mass_results:
        stats = result["stats"]
        sorted_by_weight = stats["sorted_by_weight"]
        sector_to_weight = stats["sector_to_weight"]
        top_2_cum_weight = float(sum(sector_to_weight[s] for s in sorted_by_weight[:2]))
        top_3_cum_weight = float(sum(sector_to_weight[s] for s in sorted_by_weight[:3]))
        top_sectors = [{"sector": s, "weight": float(sector_to_weight[s])} for s in sorted_by_weight[:3]]

        per_mass.append(
            {
                "mass_label": result["mass_label"],
                "m_over_g": float(result["mass"]),
                "n_sectors": len(sorted_by_weight),
                "total_weight": float(stats["total_weight"]),
                "top_2_cum_weight": top_2_cum_weight,
                "top_3_cum_weight": top_3_cum_weight,
                "top_sectors": top_sectors,
                "sector_entropy_reconstruction": stats["sector_entropy_reconstruction"],
            }
        )

    primary = per_mass[0]
    mass_trend_enabled = len(mass_results) == 4 and args.mass_list is not None

    def by_mass(field: str) -> dict[str, float]:
        return {f"{row['mass']:.6g}": float(row[field]) for row in summary_rows}

    metadata = {
        "script": "schwinger_symmetry_resolved_entanglement.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "output_directory": str(outdir.resolve()),
        "tag": args.tag or "",
        "sector_definition": sector_definition,
        "sector_note": "q indexes the bond sector labels used to group Schmidt values at the chosen cut.",
        "mass_trend_enabled": mass_trend_enabled,
        "masses_run": [float(r["mass"]) for r in mass_results],
        "bridge_summary_enabled": bool(bridge_rows),
        "bridge_summary_csv_path": str(bridge_summary_csv_path) if bridge_summary_csv_path is not None else None,
        "bridge_summary_figure_path": (
            str(bridge_summary_figure_path) if bridge_summary_figure_path is not None else None
        ),
        "bridge_quantities": [
            "total_entropy",
            "q0_weight",
            "q_minus2_weight",
            "top_2_cum_weight",
            "shannon_sector_entropy",
            "weighted_intrasector_entropy_sum",
            "xi_0",
            "xi_1",
            "xi_2",
            "retained_weight_rank_2",
            "retained_weight_rank_3",
        ],
        "comparison_enabled": bool(args.compare_mass is not None),
        "masses_compared": [float(r["mass"]) for r in mass_results],
        "decomposition_available": True,
        "n_sectors": primary["n_sectors"],
        "total_weight": primary["total_weight"],
        "top_2_cum_weight": primary["top_2_cum_weight"],
        "top_3_cum_weight": primary["top_3_cum_weight"],
        "top_sectors": primary["top_sectors"],
        "sector_entropy_reconstruction": primary["sector_entropy_reconstruction"],
        "top_2_cum_weight_by_mass": by_mass("top_2_cum_weight") if summary_rows else {},
        "top_3_cum_weight_by_mass": by_mass("top_3_cum_weight") if summary_rows else {},
        "q0_weight_by_mass": by_mass("q0_weight") if summary_rows else {},
        "q_minus2_weight_by_mass": by_mass("q_minus2_weight") if summary_rows else {},
        "shannon_sector_entropy_by_mass": by_mass("shannon_sector_entropy") if summary_rows else {},
        "weighted_intrasector_entropy_sum_by_mass": by_mass("weighted_intrasector_entropy_sum") if summary_rows else {},
        "reconstructed_total_entropy_by_mass": by_mass("reconstructed_total_entropy") if summary_rows else {},
        "reconstruction_error_by_mass": by_mass("reconstruction_error") if summary_rows else {},
        "dominant_sector_labels_used_in_figure": dominant_sector_labels_used_in_figure,
        "per_mass": per_mass,
        "outputs": serialized_outputs,
        "args": {
            "N": args.N,
            "m_over_g": args.mass,
            "compare_mass": args.compare_mass,
            "mass_list": args.mass_list,
            "x": args.coupling,
            "chi": args.chi,
            "cut": args.cut,
            "max_sectors": args.max_sectors,
            "sort_by": args.sort_by,
            "force": args.force,
            "show": args.show,
        },
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute symmetry-resolved entanglement by bond sector for Schwinger model."
    )

    model_args = parser.add_argument_group("Model Parameters")
    model_args.add_argument("--N", type=int, required=True, help="System size")
    model_args.add_argument("--mass", type=float, default=None, help="m/g ratio")
    model_args.add_argument(
        "--compare-mass",
        type=float,
        default=None,
        help="Optional second mass for matched two-point comparison at fixed N, x, chi, and cut",
    )
    model_args.add_argument(
        "--mass-list",
        type=str,
        default=None,
        help="Fixed 4-mass benchmark list: 0.05,0.08,0.125,0.20",
    )
    model_args.add_argument("--coupling", type=float, required=True, help="x = 1/(ag)^2")
    model_args.add_argument("--chi", type=int, required=True, help="Max bond dimension")

    analysis_args = parser.add_argument_group("Analysis Options")
    analysis_args.add_argument("--cut", type=int, default=None, help="Cut index (default: center)")
    analysis_args.add_argument("--max-sectors", type=int, default=12, help="Max sectors in plot")
    analysis_args.add_argument("--sort-by", type=str, default="weight", choices=["weight", "sector"])

    io_args = parser.add_argument_group("Output")
    io_args.add_argument("--outdir", type=str, required=True, help="Output directory")
    io_args.add_argument("--tag", type=str, default=None, help="Output tag suffix")
    io_args.add_argument("--force", action="store_true", help="Overwrite existing")
    io_args.add_argument("--show", action="store_true", help="Display figure")

    args = parser.parse_args()

    if args.cut is None:
        args.cut = (args.N - 1) // 2
    if args.mass_list is not None and args.compare_mass is not None:
        raise ValueError("Use either --compare-mass or --mass-list, not both.")
    if args.mass_list is None and args.mass is None:
        raise ValueError("--mass is required unless --mass-list is provided.")
    if args.mass_list is not None and args.mass is not None and args.compare_mass is None:
        # keep explicit but non-fatal; --mass is ignored in trend mode
        print("Note: --mass is ignored because --mass-list is provided.")
    if args.mass_list is not None:
        masses = parse_mass_list(args.mass_list)
    elif args.compare_mass is not None:
        masses = [args.mass, args.compare_mass]
    else:
        masses = [args.mass]

    if args.compare_mass is not None and np.isclose(args.compare_mass, args.mass):
        raise ValueError("--compare-mass must differ from --mass for a meaningful comparison run.")

    print(f"Symmetry-resolved entanglement analysis")
    if args.mass_list is not None:
        print(f"N={args.N}, mass-list={masses}, x={args.coupling}, chi={args.chi}")
    elif args.compare_mass is None:
        print(f"N={args.N}, m/g={args.mass}, x={args.coupling}, chi={args.chi}")
    else:
        print(
            f"N={args.N}, masses=({args.mass}, {args.compare_mass}), "
            f"x={args.coupling}, chi={args.chi}"
        )
    print(f"Cut: {args.cut}")
    print()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_tag(args.tag)}" if args.tag else ""

    # Check for existing outputs
    expected_files = [outdir / f"symmetry_resolved_entanglement{tag_suffix}.csv"]
    if len(masses) == 4:
        expected_files.extend(
            [
                outdir / f"symmetry_resolved_mass_trend_summary{tag_suffix}.csv",
                outdir / f"symmetry_resolved_mass_trend{tag_suffix}.png",
                outdir / f"symmetry_resolved_entropy_spectrum_bridge{tag_suffix}.csv",
                outdir / f"symmetry_resolved_entropy_spectrum_bridge{tag_suffix}.png",
                outdir / f"symmetry_resolved_metadata{tag_suffix}.json",
            ]
        )
    else:
        expected_files.extend(
            [
                outdir / f"symmetry_resolved_entanglement{tag_suffix}.png",
                outdir / f"symmetry_resolved_metadata{tag_suffix}.json",
            ]
        )
    existing = [f for f in expected_files if f.exists()]
    if not args.force and existing:
        print(f"Error: Output files already exist: {[str(p) for p in existing]}")
        print("Use --force to overwrite.")
        sys.exit(1)

    mass_results = [run_mass_point(args, masses[0], f"m/g={masses[0]:g}")]
    for mass_value in masses[1:]:
        mass_results.append(run_mass_point(args, mass_value, f"m/g={mass_value:g}"))

    for result in mass_results:
        stats = result["stats"]
        sorted_sectors = stats["sorted_by_weight"]
        sector_to_weight = stats["sector_to_weight"]
        top_3 = sorted_sectors[:3]
        top_3_str = ", ".join(f"{s}={sector_to_weight[s]:.3f}" for s in top_3)
        top_2_cum_weight = sum(sector_to_weight[s] for s in sorted_sectors[:2])
        top_3_cum_weight = sum(sector_to_weight[s] for s in sorted_sectors[:3])
        rec = stats["sector_entropy_reconstruction"]

        print(f"Summary for {result['mass_label']}:")
        print(f"  Number of sectors: {len(sorted_sectors)}")
        print(f"  Top 3 sector weights: {top_3_str}")
        print(f"  Cumulative weight (top 2): {top_2_cum_weight:.6f}")
        print(f"  Cumulative weight (top 3): {top_3_cum_weight:.6f}")
        print(f"  Total sector weight: {stats['total_weight']:.6f} (normalized)")
        print(f"  Shannon sector entropy: {rec['shannon_sector_entropy']:.6f}")
        print(f"  Weighted intra-sector entropy: {rec['weighted_intrasector_entropy_sum']:.6f}")
        print(f"  Reconstructed total entropy: {rec['reconstructed_total_entropy']:.6f}")
        print(f"  Reconstruction error: {rec['reconstruction_error']:.3e}")
        print()

    csv_rows = [row for result in mass_results for row in result["stats"]["rows"]]
    if args.sort_by == "sector":
        csv_rows = sorted(csv_rows, key=lambda r: (r["mass_label"], r["sector"]))

    summary_rows, dominant_labels = build_mass_trend_summary(mass_results)
    bridge_rows = build_bridge_summary_rows(mass_results)

    # Save CSV
    csv_file = save_sector_csv(outdir, csv_rows, tag_suffix)
    print(f"Saved CSV: {csv_file}")

    output_files: dict[str, Path] = {"csv": csv_file}
    bridge_csv_path: Path | None = None
    bridge_fig_path: Path | None = None
    if len(masses) == 4:
        summary_file = save_mass_trend_summary_csv(outdir, summary_rows, tag_suffix)
        print(f"Saved mass-trend summary CSV: {summary_file}")
        png_file = plot_mass_trend_three_panel(outdir, mass_results, summary_rows, dominant_labels, args, tag_suffix)
        print(f"Saved mass-trend figure: {png_file}")
        bridge_csv_path = save_bridge_summary_csv(outdir, bridge_rows, tag_suffix)
        print(f"Saved bridge summary CSV: {bridge_csv_path}")
        bridge_fig_path = plot_bridge_summary_two_panel(outdir, bridge_rows, tag_suffix)
        print(f"Saved bridge summary figure: {bridge_fig_path}")
        output_files["mass_trend_summary_csv"] = summary_file
        output_files["mass_trend_figure"] = png_file
        output_files["bridge_summary_csv"] = bridge_csv_path
        output_files["bridge_summary_figure"] = bridge_fig_path
    else:
        png_file = plot_sector_weights_or_comparison(outdir, mass_results, args, tag_suffix)
        print(f"Saved figure: {png_file}")
        output_files["figure"] = png_file

    # Save metadata
    metadata_file = save_metadata(
        outdir, args, tag_suffix,
        mass_results=mass_results,
        summary_rows=summary_rows,
        bridge_rows=bridge_rows if len(masses) == 4 else [],
        bridge_summary_csv_path=bridge_csv_path,
        bridge_summary_figure_path=bridge_fig_path,
        dominant_sector_labels_used_in_figure=dominant_labels,
        output_files=output_files,
    )
    print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    main()
