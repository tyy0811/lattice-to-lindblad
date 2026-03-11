#!/usr/bin/env python3
"""Run publication-grade validation bundles for Schwinger entanglement outputs.

This runner keeps existing physics drivers intact and shells out to:
- schwinger_entanglement_entropy.py
- schwinger_entanglement_spectrum.py
- schmidt_decay_analysis.py (not required for core validation metrics here)

Bundles:
1) truncation_study      (chi convergence with deltas/fit/uncertainty)
2) finite_size_scaling   (1/N fits with high-chi spread uncertainty)
3) cleanup               (manifest-first duplicate archival)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "05_Entanglement_Structure_QI"
ENTROPY = SCRIPT_DIR / "schwinger_entanglement_entropy.py"
SPECTRUM = SCRIPT_DIR / "schwinger_entanglement_spectrum.py"
DECAY = SCRIPT_DIR / "schmidt_decay_analysis.py"
PYTHON = ROOT / ".venv" / "bin" / "python"

OUT_BASE = SCRIPT_DIR / "publication_validation"
TRUNC_BASE = OUT_BASE / "truncation_study"
SIZE_BASE = OUT_BASE / "finite_size_scaling"
CLEANUP_BASE = OUT_BASE / "cleanup"

REPRESENTATIVE_NOTE = (
    "Representative interior cut chosen for spectrum/Schmidt comparison at a near-central bond; "
    "entropy maximum is tracked separately in the full entropy profile."
)
PEAK_TIE_POLICY = "mirrored_boundary_right"


@dataclass(frozen=True)
class SchwingerPoint:
    N: int
    mass: float
    coupling: float
    chi: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def ensure_dirs() -> None:
    for d in [OUT_BASE, TRUNC_BASE, SIZE_BASE, CLEANUP_BASE]:
        d.mkdir(parents=True, exist_ok=True)


def cmd_to_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def init_provenance() -> dict[str, Any]:
    return {
        "commands_requested": [],
        "commands_executed": [],
        "commands_reused_cached_outputs": [],
        "artifacts": {},
    }


def _append_cmd_event(provenance: dict[str, Any], key: str, bundle: str, step: str, cmd_str: str) -> None:
    provenance[key].append(
        {
            "bundle": bundle,
            "step": step,
            "command": cmd_str,
            "timestamp_observed": now_iso(),
        }
    )


def record_artifact(
    provenance: dict[str, Any],
    *,
    bundle: str,
    artifact_id: str,
    status: str,
    output_paths: dict[str, Path | str],
) -> None:
    bundle_artifacts = provenance["artifacts"].setdefault(bundle, {})
    bundle_artifacts[artifact_id] = {
        "status": status,
        "timestamp_observed": now_iso(),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }


def bundle_provenance(provenance: dict[str, Any], bundle: str) -> dict[str, Any]:
    return {
        "commands_requested": [x for x in provenance["commands_requested"] if x["bundle"] == bundle],
        "commands_executed": [x for x in provenance["commands_executed"] if x["bundle"] == bundle],
        "commands_reused_cached_outputs": [
            x for x in provenance["commands_reused_cached_outputs"] if x["bundle"] == bundle
        ],
        "artifacts": provenance["artifacts"].get(bundle, {}),
    }


def run_cmd(cmd: list[str], *, provenance: dict[str, Any], bundle: str, step: str) -> None:
    cmd_str = cmd_to_str(cmd)
    _append_cmd_event(provenance, "commands_executed", bundle, step, cmd_str)
    print(f"\n$ {cmd_str}")
    env = {**os.environ, "MPLCONFIGDIR": "/tmp/mpl_stage3"}
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}")


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_entropy_csv(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["cut"]), float(row["entropy"])))
    return rows


def read_spectrum_csv(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["level_index"]), float(row["lambda"]), float(row["xi"])))
    return sorted(rows, key=lambda t: t[0])


def extract_optional_truncation_metric(metadata: dict[str, Any]) -> float | None:
    candidates = [
        metadata.get("discarded_weight"),
        metadata.get("truncation_error"),
        metadata.get("first_neglected_schmidt_weight"),
        metadata.get("first_neglected_weight"),
    ]
    args = metadata.get("args", {})
    candidates.extend(
        [
            args.get("discarded_weight"),
            args.get("truncation_error"),
            args.get("first_neglected_schmidt_weight"),
            args.get("first_neglected_weight"),
        ]
    )
    for c in candidates:
        if c is None:
            continue
        try:
            return float(c)
        except Exception:
            continue
    return None


def entropy_run(
    point: SchwingerPoint,
    outdir: Path,
    *,
    tag: str,
    role: str,
    provenance: dict[str, Any],
    bundle: str,
    step: str,
) -> dict[str, Any]:
    tag_s = sanitize_tag(tag)
    outputs = {
        "metadata": outdir / f"entropy_profile_metadata_{tag_s}.json",
        "csv": outdir / f"entropy_profile_{tag_s}.csv",
        "png": outdir / f"entropy_profile_{tag_s}.png",
    }
    cmd = [
        str(PYTHON),
        str(ENTROPY),
        "--N",
        str(point.N),
        "--mass",
        str(point.mass),
        "--coupling",
        str(point.coupling),
        "--chi",
        str(point.chi),
        "--bc",
        "open",
        "--state-source",
        "compute",
        "--application-bundle-role",
        role,
        "--outdir",
        str(outdir),
        "--tag",
        tag,
        "--force",
    ]
    cmd_str = cmd_to_str(cmd)
    _append_cmd_event(provenance, "commands_requested", bundle, step, cmd_str)

    if all(p.exists() for p in outputs.values()):
        _append_cmd_event(provenance, "commands_reused_cached_outputs", bundle, step, cmd_str)
        record_artifact(provenance, bundle=bundle, artifact_id=step, status="reused", output_paths=outputs)
        return outputs

    run_cmd(cmd, provenance=provenance, bundle=bundle, step=step)
    record_artifact(provenance, bundle=bundle, artifact_id=step, status="executed", output_paths=outputs)
    return outputs


def spectrum_run(
    point: SchwingerPoint,
    outdir: Path,
    *,
    tag: str,
    role: str,
    cut: int,
    entropy_max_cut: int,
    provenance: dict[str, Any],
    bundle: str,
    step: str,
    nlevels: int = 8,
) -> dict[str, Any]:
    tag_s = sanitize_tag(tag)
    outputs = {
        "metadata": outdir / f"entanglement_spectrum_metadata_{tag_s}.json",
        "csv": outdir / f"entanglement_spectrum_schwinger_{tag_s}.csv",
        "png": outdir / f"entanglement_spectrum_{tag_s}.png",
    }
    cmd = [
        str(PYTHON),
        str(SPECTRUM),
        "--N",
        str(point.N),
        "--mass",
        str(point.mass),
        "--coupling",
        str(point.coupling),
        "--chi",
        str(point.chi),
        "--bc",
        "open",
        "--state-source",
        "compute",
        "--application-bundle-role",
        role,
        "--cut",
        str(cut),
        "--nlevels",
        str(nlevels),
        "--entropy-max-cut",
        str(entropy_max_cut),
        "--representative-cut-note",
        REPRESENTATIVE_NOTE,
        "--outdir",
        str(outdir),
        "--tag",
        tag,
        "--force",
    ]
    cmd_str = cmd_to_str(cmd)
    _append_cmd_event(provenance, "commands_requested", bundle, step, cmd_str)

    if all(p.exists() for p in outputs.values()):
        _append_cmd_event(provenance, "commands_reused_cached_outputs", bundle, step, cmd_str)
        record_artifact(provenance, bundle=bundle, artifact_id=step, status="reused", output_paths=outputs)
        return outputs

    run_cmd(cmd, provenance=provenance, bundle=bundle, step=step)
    record_artifact(provenance, bundle=bundle, artifact_id=step, status="executed", output_paths=outputs)
    return outputs


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "intercept_std": float("nan"),
            "rmse": float("nan"),
        }

    if len(x) >= 3:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = coeffs
        intercept_std = float(np.sqrt(max(cov[1, 1], 0.0)))
    else:
        slope, intercept = np.polyfit(x, y, 1)
        intercept_std = float("nan")

    y_hat = slope * x + intercept
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "intercept_std": intercept_std,
        "rmse": rmse,
    }


def _infer_system_size_from_cuts(cuts: np.ndarray) -> int:
    cuts = np.asarray(cuts, dtype=int)
    cmin = int(np.min(cuts))
    cmax = int(np.max(cuts))
    return cmax + 2 if cmin == 0 else cmax + 1


def find_entropy_peak_info(
    cuts,
    entropies,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    tie_policy: str = PEAK_TIE_POLICY,
    N: int | None = None,
) -> dict[str, Any]:
    cuts = np.asarray(cuts, dtype=int)
    entropies = np.asarray(entropies, dtype=float)
    if cuts.size == 0 or entropies.size == 0:
        raise RuntimeError("Cannot select entropy peak: empty cuts/entropies arrays.")
    if cuts.size != entropies.size:
        raise RuntimeError(
            f"Cannot select entropy peak: cuts/entropies length mismatch ({cuts.size} vs {entropies.size})."
        )
    max_entropy = float(np.max(entropies))
    tied_mask = np.isclose(entropies, max_entropy, atol=atol, rtol=rtol)
    tied_max_cuts = sorted(int(c) for c in cuts[tied_mask])
    if not tied_max_cuts:
        raise RuntimeError(
            "Entropy-peak tie resolution failed: no cuts matched the global maximum within tolerance."
        )

    if tie_policy not in {"mirrored_boundary_right", "edge_nearest_right"}:
        raise ValueError(f"Unsupported tie policy '{tie_policy}'.")

    cmin = int(np.min(cuts))
    cmax = int(np.max(cuts))

    def edge_dist(cut: int) -> int:
        return int(min(cut - cmin, cmax - cut))

    min_edge_dist = min(edge_dist(c) for c in tied_max_cuts)
    edge_nearest = [c for c in tied_max_cuts if edge_dist(c) == min_edge_dist]
    chosen_i_max = max(edge_nearest)
    d_edge = int(min(edge_dist(c) for c in tied_max_cuts))

    return {
        "max_entropy": max_entropy,
        "tied_max_cuts": tied_max_cuts,
        "chosen_i_max": int(chosen_i_max),
        "d_edge": d_edge,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_truncation_study(provenance: dict[str, Any]) -> dict[str, Any]:
    print("Running truncation/convergence study...")

    chis = [16, 24, 32, 48, 64, 96, 128]
    point_base = SchwingerPoint(N=20, mass=0.125, coupling=4.0, chi=64)
    representative_cut = point_base.N // 2
    tag = "publication"
    role = "comparison"

    rows: list[dict[str, Any]] = []
    entropy_profiles: dict[int, list[tuple[int, float]]] = {}

    for chi in chis:
        point = SchwingerPoint(point_base.N, point_base.mass, point_base.coupling, chi)
        outdir = TRUNC_BASE / f"chi{chi}"
        outdir.mkdir(parents=True, exist_ok=True)

        ent = entropy_run(
            point,
            outdir,
            tag=tag,
            role=role,
            provenance=provenance,
            bundle="truncation_study",
            step=f"chi{chi}_entropy",
        )
        ent_meta = load_json(ent["metadata"])
        ent_rows = read_entropy_csv(ent["csv"])
        entropy_profiles[chi] = ent_rows
        ent_by_cut = {cut: val for cut, val in ent_rows}
        if representative_cut not in ent_by_cut:
            available = np.array(sorted(ent_by_cut.keys()))
            nearest = int(available[np.argmin(np.abs(available - representative_cut))])
            rep_cut_used = nearest
        else:
            rep_cut_used = representative_cut

        spec = spectrum_run(
            point,
            outdir,
            tag=tag,
            role=role,
            cut=rep_cut_used,
            entropy_max_cut=int(ent_meta["max_entropy_cut"]),
            provenance=provenance,
            bundle="truncation_study",
            step=f"chi{chi}_spectrum",
            nlevels=8,
        )
        spec_meta = load_json(spec["metadata"])
        spec_rows = read_spectrum_csv(spec["csv"])
        xis = [r[2] for r in spec_rows]

        trunc_metric = extract_optional_truncation_metric(spec_meta)

        row = {
            "chi": chi,
            "representative_cut": rep_cut_used,
            "S_center": float(ent_by_cut[rep_cut_used]),
            "S_max": float(ent_meta["max_entropy"]),
            "xi_0": float(xis[0]) if len(xis) > 0 else float("nan"),
            "xi_1": float(xis[1]) if len(xis) > 1 else float("nan"),
            "xi_2": float(xis[2]) if len(xis) > 2 else float("nan"),
            "xi_3": float(xis[3]) if len(xis) > 3 else float("nan"),
            "discarded_weight_or_trunc_error": trunc_metric,
            "entropy_max_cut": int(ent_meta["max_entropy_cut"]),
            "tied_max_cuts": json.dumps([int(c) for c in ent_meta.get("tied_max_cuts", [])]),
            "n_tied_max_cuts": int(ent_meta.get("n_tied_max_cuts", 0)),
            "tie_policy": ent_meta.get("tie_policy", PEAK_TIE_POLICY),
            "chosen_i_max": int(ent_meta.get("chosen_i_max", ent_meta["max_entropy_cut"])),
            "d_edge": int(ent_meta.get("d_edge", 0)),
            "entropy_csv": str(ent["csv"]),
            "spectrum_csv": str(spec["csv"]),
            "entropy_metadata": str(ent["metadata"]),
            "spectrum_metadata": str(spec["metadata"]),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["chi"]))
    chi_max = max(chis)
    ref = next(r for r in rows if int(r["chi"]) == chi_max)

    for r in rows:
        r["delta_S_center"] = float(r["S_center"] - ref["S_center"])
        r["delta_S_max"] = float(r["S_max"] - ref["S_max"])
        for k in range(4):
            rk = f"xi_{k}"
            r[f"delta_xi_{k}"] = float(r[rk] - ref[rk])

    trunc_values = [r["discarded_weight_or_trunc_error"] for r in rows]
    use_trunc_metric = all(v is not None and np.isfinite(v) and v >= 0.0 for v in trunc_values)
    if use_trunc_metric:
        x_all = np.array([float(v) for v in trunc_values], dtype=float)
        fit_x_label = "discarded_weight_or_trunc_error"
    else:
        x_all = np.array([1.0 / float(r["chi"]) for r in rows], dtype=float)
        fit_x_label = "1/chi"

    fit_observables = ["S_center", "S_max", "xi_0", "xi_1", "xi_2", "xi_3"]
    fit_results: dict[str, dict[str, float]] = {}
    uncertainty: dict[str, dict[str, float]] = {}

    n_fit = min(4, len(rows))
    fit_slice = slice(len(rows) - n_fit, len(rows))
    x_fit = x_all[fit_slice]

    for obs in fit_observables:
        y_all = np.array([float(r[obs]) for r in rows], dtype=float)
        y_fit = y_all[fit_slice]
        fit = fit_linear(x_fit, y_fit)
        spread_last3 = float(np.max(y_all[-3:]) - np.min(y_all[-3:])) if len(y_all) >= 3 else float("nan")
        fit_rmse = float(fit["rmse"])
        if np.isfinite(spread_last3) and np.isfinite(fit_rmse):
            combined = max(spread_last3, fit_rmse)
        elif np.isfinite(spread_last3):
            combined = spread_last3
        elif np.isfinite(fit_rmse):
            combined = fit_rmse
        else:
            combined = float("nan")

        fit_results[obs] = fit
        uncertainty[obs] = {
            "spread_last3": spread_last3,
            "fit_rmse": fit_rmse,
            "combined": combined,
        }

    chi64 = next((r for r in rows if int(r["chi"]) == 64), None)

    table_csv = TRUNC_BASE / "chi_convergence_table.csv"
    table_json = TRUNC_BASE / "chi_convergence_table.json"
    table_fields = [
        "chi",
        "representative_cut",
        "S_center",
        "S_max",
        "xi_0",
        "xi_1",
        "xi_2",
        "xi_3",
        "discarded_weight_or_trunc_error",
        "delta_S_center",
        "delta_S_max",
        "delta_xi_0",
        "delta_xi_1",
        "delta_xi_2",
        "delta_xi_3",
        "entropy_max_cut",
        "tied_max_cuts",
        "n_tied_max_cuts",
        "tie_policy",
        "chosen_i_max",
        "d_edge",
        "entropy_csv",
        "spectrum_csv",
        "entropy_metadata",
        "spectrum_metadata",
    ]
    write_csv(table_csv, rows, table_fields)
    record_artifact(
        provenance,
        bundle="truncation_study",
        artifact_id="chi_convergence_table_csv",
        status="executed",
        output_paths={"table_csv": table_csv},
    )
    save_json(
        table_json,
        {
            "timestamp": now_iso(),
            "fit_variable": fit_x_label,
            "rows": rows,
            "fit_results": fit_results,
            "uncertainty": uncertainty,
            "representative_cut": representative_cut,
            "chi_values": chis,
        },
    )
    record_artifact(
        provenance,
        bundle="truncation_study",
        artifact_id="chi_convergence_table_json",
        status="executed",
        output_paths={"table_json": table_json},
    )

    fig_path = TRUNC_BASE / "truncation_convergence_figure.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)

    chi_arr = np.array([int(r["chi"]) for r in rows], dtype=float)

    ax = axes[0, 0]
    ax.plot(chi_arr, [r["S_center"] for r in rows], "o-", label=f"S_center (cut={representative_cut})")
    ax.plot(chi_arr, [r["S_max"] for r in rows], "s--", label="S_max")
    ax.set_xlabel("chi")
    ax.set_ylabel(r"Entropy $S_{\rm vN}$")
    ax.set_title("Entropy vs chi")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for k in range(4):
        ax.plot(chi_arr, [r[f"xi_{k}"] for r in rows], "o-", label=f"$\\xi_{k}$")
    ax.set_xlabel("chi")
    ax.set_ylabel(r"Entanglement levels $\xi_k$")
    ax.set_title(f"First four levels at cut {representative_cut}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    abs_delta_center = np.abs([r["delta_S_center"] for r in rows])
    abs_delta_max = np.abs([r["delta_S_max"] for r in rows])
    ax.plot(chi_arr, abs_delta_center, "o-", label=r"$|\Delta S_{center}|$")
    ax.plot(chi_arr, abs_delta_max, "s--", label=r"$|\Delta S_{max}|$")
    for k in range(4):
        ax.plot(chi_arr, np.abs([r[f"delta_xi_{k}"] for r in rows]), "-", alpha=0.7, label=f"$|\\Delta \\xi_{k}|$")
    ax.set_yscale("log")
    ax.set_xlabel("chi")
    ax.set_ylabel(r"Absolute delta to $\chi_{max}$")
    ax.set_title(f"Deltas to chi={chi_max}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.axis("off")
    txt = [
        f"Fit variable: {fit_x_label}",
        "Uncertainty method:",
        "combined = max(spread(last 3), fit RMSE)",
    ]
    if chi64 is not None:
        txt.extend(
            [
                "",
                "At chi=64 vs chi_max:",
                f"|ΔS_center| = {abs(chi64['delta_S_center']):.3e}",
                f"|ΔS_max| = {abs(chi64['delta_S_max']):.3e}",
                f"|Δxi_0| = {abs(chi64['delta_xi_0']):.3e}",
                f"|Δxi_1| = {abs(chi64['delta_xi_1']):.3e}",
                f"|Δxi_2| = {abs(chi64['delta_xi_2']):.3e}",
                f"|Δxi_3| = {abs(chi64['delta_xi_3']):.3e}",
            ]
        )
    ax.text(0.01, 0.99, "\n".join(txt), va="top", ha="left", fontsize=9)

    fig.suptitle("Truncation/Convergence Study (N=20, m/g=0.125, x=4.0)", fontsize=12)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    record_artifact(
        provenance,
        bundle="truncation_study",
        artifact_id="truncation_convergence_figure",
        status="executed",
        output_paths={"figure": fig_path},
    )

    summary_md = ROOT / "APPLICATION_TRUNCATION_STUDY_SUMMARY.md"
    with open(summary_md, "w") as f:
        f.write("# Application Truncation Study Summary\n\n")
        f.write("## What Was Run\n")
        f.write("- Fixed point: `N=20`, `m/g=0.125`, `x=4.0`\n")
        f.write("- Bond-dimension ladder: `chi = 16, 24, 32, 48, 64, 96, 128`\n")
        f.write(f"- Representative cut: `{representative_cut}`\n")
        f.write("- Observables: `S_center`, `S_max`, and `xi_0..xi_3`\n\n")

        f.write("## Quantified Convergence to chi_max\n")
        if chi64 is not None:
            f.write(f"- `|S_center(64)-S_center(128)| = {abs(chi64['delta_S_center']):.6e}`\n")
            f.write(f"- `|S_max(64)-S_max(128)| = {abs(chi64['delta_S_max']):.6e}`\n")
            for k in range(4):
                f.write(
                    f"- `|xi_{k}(64)-xi_{k}(128)| = {abs(chi64[f'delta_xi_{k}']):.6e}`\n"
                )
        else:
            f.write("- `chi=64` was unavailable; tolerances versus `chi_max` not reported.\n")

        f.write("\n## Extrapolation and Uncertainty\n")
        f.write(f"- Fit variable: `{fit_x_label}` (fallback to `1/chi` used when truncation metric unavailable).\n")
        f.write("- Fit form: linear model over highest 4 chi points.\n")
        f.write("- Numerical uncertainty per observable: `max(spread over last 3 points, fit RMSE)`.\n")
        for obs in fit_observables:
            fit = fit_results[obs]
            unc = uncertainty[obs]
            f.write(
                f"- `{obs}` extrapolated = `{fit['intercept']:.8f}`; "
                f"fit RMSE = `{unc['fit_rmse']:.3e}`; spread(last3) = `{unc['spread_last3']:.3e}`; "
                f"combined uncertainty = `{unc['combined']:.3e}`\n"
            )

        f.write("\n## Limitations\n")
        f.write("- Driver metadata currently do not expose discarded-weight/truncation-error fields; those entries are null.\n")
        f.write("- Linear extrapolation is deliberately lightweight and used as a consistency check, not a high-order model.\n")
    record_artifact(
        provenance,
        bundle="truncation_study",
        artifact_id="truncation_summary_markdown",
        status="executed",
        output_paths={"summary_markdown": summary_md},
    )

    record_artifact(
        provenance,
        bundle="truncation_study",
        artifact_id="truncation_bundle_metadata",
        status="executed",
        output_paths={"metadata": TRUNC_BASE / "truncation_study_metadata.json"},
    )
    metadata = {
        "bundle": "truncation_study",
        "timestamp": now_iso(),
        "git_commit": get_git_commit(),
        "parameter_grid": {
            "N": point_base.N,
            "mass": point_base.mass,
            "coupling": point_base.coupling,
            "chi_values": chis,
        },
        "representative_cut": representative_cut,
        "entropy_max_cut_values": {str(r["chi"]): int(r["entropy_max_cut"]) for r in rows},
        "tied_max_cuts": {str(r["chi"]): r["tied_max_cuts"] for r in rows},
        "chosen_i_max": {str(r["chi"]): int(r["chosen_i_max"]) for r in rows},
        "d_edge": {str(r["chi"]): int(r["d_edge"]) for r in rows},
        "tie_policy": PEAK_TIE_POLICY,
        "tied_max_cuts_by_chi": {str(r["chi"]): r["tied_max_cuts"] for r in rows},
        "chosen_i_max_by_chi": {str(r["chi"]): int(r["chosen_i_max"]) for r in rows},
        "d_edge_by_chi": {str(r["chi"]): int(r["d_edge"]) for r in rows},
        "d_edge_definition": "minimum nearest-edge distance over tied maximum cuts",
        "chosen_i_max_definition": (
            "canonical representative of tied entropy maxima under mirrored_boundary_right policy"
        ),
        "representative_cut_note": REPRESENTATIVE_NOTE,
        "uncertainty_method": "max(spread over last 3 chi points, linear-fit RMSE)",
        "fit_method": {
            "form": "linear",
            "independent_variable": fit_x_label,
            "fit_points": n_fit,
            "observables": fit_observables,
        },
        "outputs": {
            "base_directory": str(TRUNC_BASE),
            "table_csv": str(table_csv),
            "table_json": str(table_json),
            "figure": str(fig_path),
            "summary_markdown": str(summary_md),
        },
        "commands_requested": bundle_provenance(provenance, "truncation_study")["commands_requested"],
        "commands_executed": bundle_provenance(provenance, "truncation_study")["commands_executed"],
        "commands_reused_cached_outputs": bundle_provenance(provenance, "truncation_study")[
            "commands_reused_cached_outputs"
        ],
        "artifact_provenance": bundle_provenance(provenance, "truncation_study")["artifacts"],
    }
    save_json(TRUNC_BASE / "truncation_study_metadata.json", metadata)

    return {
        "rows": rows,
        "fit_results": fit_results,
        "uncertainty": uncertainty,
        "metadata_path": TRUNC_BASE / "truncation_study_metadata.json",
        "table_csv": table_csv,
        "table_json": table_json,
        "figure": fig_path,
        "summary": summary_md,
        "representative_cut": representative_cut,
    }


def make_finite_size_scaling(provenance: dict[str, Any]) -> dict[str, Any]:
    print("Running finite-size scaling study...")

    sizes = [12, 16, 20, 24, 28, 32]
    chi_ladder = [64, 96, 128]
    mass = 0.125
    coupling = 4.0
    tag = "publication"
    role = "comparison"

    raw_rows: list[dict[str, Any]] = []
    profile_cache: dict[tuple[int, int], list[tuple[int, float]]] = {}

    for N in sizes:
        for chi in chi_ladder:
            outdir = SIZE_BASE / f"N{N}" / f"chi{chi}"
            outdir.mkdir(parents=True, exist_ok=True)
            point = SchwingerPoint(N=N, mass=mass, coupling=coupling, chi=chi)

            ent = entropy_run(
                point,
                outdir,
                tag=tag,
                role=role,
                provenance=provenance,
                bundle="finite_size_scaling",
                step=f"N{N}_chi{chi}_entropy",
            )
            ent_meta = load_json(ent["metadata"])
            profile = read_entropy_csv(ent["csv"])
            profile_cache[(N, chi)] = profile

            cuts = np.array([c for c, _ in profile], dtype=int)
            vals = np.array([v for _, v in profile], dtype=float)
            peak_info = find_entropy_peak_info(cuts, vals, tie_policy=PEAK_TIE_POLICY, N=N)
            S_peak = float(peak_info["max_entropy"])
            tied_max_cuts = [int(c) for c in peak_info["tied_max_cuts"]]
            i_max = int(peak_info["chosen_i_max"])

            mid_cut = N // 2
            if mid_cut in cuts:
                S_mid = float(vals[np.where(cuts == mid_cut)[0][0]])
                mid_cut_used = mid_cut
            else:
                nearest_idx = int(np.argmin(np.abs(cuts - mid_cut)))
                mid_cut_used = int(cuts[nearest_idx])
                S_mid = float(vals[nearest_idx])

            A = float(S_peak - S_mid)
            d_edge = int(peak_info["d_edge"])

            raw_rows.append(
                {
                    "N": N,
                    "chi": chi,
                    "mid_cut": mid_cut_used,
                    "S_peak": S_peak,
                    "S_mid": S_mid,
                    "A": A,
                    "i_max": i_max,
                    "chosen_i_max": i_max,
                    "d_edge": d_edge,
                    "tied_max_cuts": json.dumps(tied_max_cuts),
                    "n_tied_max_cuts": len(tied_max_cuts),
                    "tie_policy": PEAK_TIE_POLICY,
                    "entropy_max_cut": int(ent_meta["max_entropy_cut"]),
                    "entropy_csv": str(ent["csv"]),
                    "entropy_metadata": str(ent["metadata"]),
                }
            )

    feasible_chis = [c for c in chi_ladder if all(any(r["N"] == N and r["chi"] == c for r in raw_rows) for N in sizes)]
    if 128 in feasible_chis:
        chi_choice = 128
        chi_choice_reason = "All requested sizes completed at chi=128; selected highest chi for scaling observables."
    elif feasible_chis:
        chi_choice = max(feasible_chis)
        chi_choice_reason = (
            f"chi=128 not feasible for all sizes in this run; selected highest common chi={chi_choice}."
        )
    else:
        chi_choice = max(chi_ladder)
        chi_choice_reason = "No common chi across all sizes; fallback to nominal highest chi value."

    selected_rows: list[dict[str, Any]] = []
    for N in sizes:
        candidates = [r for r in raw_rows if r["N"] == N and r["chi"] == chi_choice]
        if not candidates:
            candidates = sorted([r for r in raw_rows if r["N"] == N], key=lambda x: x["chi"], reverse=True)
        if not candidates:
            continue
        base = dict(candidates[0])

        spread_pool = [r for r in raw_rows if r["N"] == N and r["chi"] in chi_ladder]
        spread_pool = sorted(spread_pool, key=lambda r: r["chi"])

        def spread(obs: str) -> float | None:
            vals = [float(r[obs]) for r in spread_pool]
            if len(vals) < 2:
                return None
            return float(max(vals) - min(vals))

        base["unc_S_peak"] = spread("S_peak")
        base["unc_S_mid"] = spread("S_mid")
        base["unc_A"] = spread("A")
        base["available_chi_for_uncertainty"] = [int(r["chi"]) for r in spread_pool]
        selected_rows.append(base)

    selected_rows = sorted(selected_rows, key=lambda r: r["N"])
    if not selected_rows:
        raise RuntimeError("Finite-size scaling failed: no selected rows were produced.")

    x = np.array([1.0 / float(r["N"]) for r in selected_rows], dtype=float)
    y_peak = np.array([float(r["S_peak"]) for r in selected_rows], dtype=float)
    y_mid = np.array([float(r["S_mid"]) for r in selected_rows], dtype=float)
    y_A = np.array([float(r["A"]) for r in selected_rows], dtype=float)

    fit_peak = fit_linear(x, y_peak)
    fit_mid = fit_linear(x, y_mid)
    fit_A = fit_linear(x, y_A)

    table_csv = SIZE_BASE / "finite_size_scaling_table.csv"
    table_json = SIZE_BASE / "finite_size_scaling_table.json"
    raw_csv = SIZE_BASE / "finite_size_raw_table.csv"
    raw_json = SIZE_BASE / "finite_size_raw_table.json"

    selected_fields = [
        "N",
        "chi",
        "mid_cut",
        "S_peak",
        "S_mid",
        "A",
        "i_max",
        "chosen_i_max",
        "d_edge",
        "tied_max_cuts",
        "n_tied_max_cuts",
        "tie_policy",
        "unc_S_peak",
        "unc_S_mid",
        "unc_A",
        "available_chi_for_uncertainty",
        "entropy_csv",
        "entropy_metadata",
    ]
    raw_fields = [
        "N",
        "chi",
        "mid_cut",
        "S_peak",
        "S_mid",
        "A",
        "i_max",
        "chosen_i_max",
        "d_edge",
        "tied_max_cuts",
        "n_tied_max_cuts",
        "tie_policy",
        "entropy_max_cut",
        "entropy_csv",
        "entropy_metadata",
    ]

    write_csv(table_csv, selected_rows, selected_fields)
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_scaling_table_csv",
        status="executed",
        output_paths={"selected_table_csv": table_csv},
    )
    write_csv(raw_csv, raw_rows, raw_fields)
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_raw_table_csv",
        status="executed",
        output_paths={"raw_table_csv": raw_csv},
    )

    save_json(
        table_json,
        {
            "selected_rows": selected_rows,
            "chi_choice": chi_choice,
            "chi_choice_reason": chi_choice_reason,
            "fits": {
                "S_peak": fit_peak,
                "S_mid": fit_mid,
                "A": fit_A,
            },
        },
    )
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_scaling_table_json",
        status="executed",
        output_paths={"selected_table_json": table_json},
    )
    save_json(raw_json, {"rows": raw_rows})
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_raw_table_json",
        status="executed",
        output_paths={"raw_table_json": raw_json},
    )

    fig_scaling = SIZE_BASE / "finite_size_scaling_figure.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=160)

    x_plot = np.array([1.0 / r["N"] for r in selected_rows], dtype=float)

    def plot_obs(ax, y_vals, y_errs, fit, label, color):
        y = np.array(y_vals, dtype=float)
        err = np.array([np.nan if v is None else float(v) for v in y_errs], dtype=float)
        if np.all(np.isnan(err)):
            ax.plot(x_plot, y, "o", color=color, label=label)
        else:
            err_clean = np.where(np.isnan(err), 0.0, err)
            ax.errorbar(x_plot, y, yerr=err_clean, fmt="o", color=color, label=label, capsize=3)
        x_line = np.linspace(min(x_plot) * 0.9, max(x_plot) * 1.05, 200)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, "--", color=color, alpha=0.9)
        ax.set_xlabel("1/N")
        ax.grid(True, alpha=0.3)

    plot_obs(
        axes[0],
        [r["S_peak"] for r in selected_rows],
        [r["unc_S_peak"] for r in selected_rows],
        fit_peak,
        "S_peak",
        "C0",
    )
    axes[0].set_ylabel(r"Entropy $S_{\rm vN}$")
    axes[0].set_title("S_peak vs 1/N")

    plot_obs(
        axes[1],
        [r["S_mid"] for r in selected_rows],
        [r["unc_S_mid"] for r in selected_rows],
        fit_mid,
        "S_mid",
        "C1",
    )
    axes[1].set_title("S_mid vs 1/N")

    plot_obs(
        axes[2],
        [r["A"] for r in selected_rows],
        [r["unc_A"] for r in selected_rows],
        fit_A,
        "A = S_peak - S_mid",
        "C2",
    )
    axes[2].set_title("A vs 1/N")

    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle(f"Finite-size scaling (m/g={mass}, x={coupling}, chi={chi_choice})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_scaling, dpi=220, bbox_inches="tight")
    plt.close(fig)
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_scaling_figure",
        status="executed",
        output_paths={"scaling_figure": fig_scaling},
    )

    fig_profile = SIZE_BASE / "finite_size_structural_profile_collapse.png"
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    for r in selected_rows:
        N = int(r["N"])
        chi = int(r["chi"])
        profile = profile_cache[(N, chi)]
        cuts = np.array([c for c, _ in profile], dtype=float)
        vals = np.array([v for _, v in profile], dtype=float)
        d_edge = np.minimum(cuts, N - cuts)

        d_unique = np.unique(d_edge)
        means = []
        for d in d_unique:
            means.append(float(np.mean(vals[d_edge == d])))
        ax.plot(d_unique, means, "o-", linewidth=1.5, markersize=4, label=f"N={N}")

    ax.set_xlabel("Edge-distance cut coordinate d = min(i, N-i)")
    ax.set_ylabel(r"Entropy $S_{\rm vN}$")
    ax.set_title(f"Structural profile collapse (chi={chi_choice}, m/g={mass}, x={coupling})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_profile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_structural_profile_figure",
        status="executed",
        output_paths={"structural_profile_figure": fig_profile},
    )

    nmax_row = max(selected_rows, key=lambda r: r["N"])

    def saturation_statement(obs: str, fit: dict[str, float], unc_key: str) -> str:
        val_nmax = float(nmax_row[obs])
        y_inf = float(fit["intercept"])
        unc = nmax_row.get(unc_key)
        if unc is None or not np.isfinite(unc):
            unc = fit["rmse"]
        diff = abs(val_nmax - y_inf)
        if np.isfinite(unc) and diff <= max(unc, 1e-8):
            return f"Yes, within estimated numerical uncertainty (|Nmax - intercept|={diff:.3e})."
        return f"Partially; residual finite-size drift remains (|Nmax - intercept|={diff:.3e})."

    A_intercept = float(fit_A["intercept"])
    A_unc = float(fit_A["intercept_std"]) if np.isfinite(fit_A["intercept_std"]) else float(fit_A["rmse"])
    if np.isfinite(A_unc) and A_intercept - A_unc > 0:
        A_statement = "A(N) remains finite within current fit uncertainty."
    else:
        A_statement = "A(N) may decay; current fit uncertainty still overlaps smaller asymptotic values."

    d_edges = [int(r["d_edge"]) for r in selected_rows]
    if max(d_edges) <= 3:
        edge_statement = "Entropy maximum remains near the boundary in edge-distance coordinates for all tested sizes."
    else:
        edge_statement = "Peak position drifts away from strict boundary adjacency for part of the tested size range."

    summary_md = ROOT / "APPLICATION_FINITE_SIZE_SCALING_SUMMARY.md"
    with open(summary_md, "w") as f:
        f.write("# Application Finite-Size Scaling Summary\n\n")
        f.write("## What Was Run\n")
        f.write("- Regime: `m/g=0.125`, `x=4.0`\n")
        f.write("- Sizes: `N = 12, 16, 20, 24, 28, 32`\n")
        f.write("- High-chi uncertainty set per N: attempted `chi = 64, 96, 128`\n")
        f.write(f"- Selected scaling chi: `{chi_choice}`\n")
        f.write(f"- Chi choice reason: {chi_choice_reason}\n")
        f.write("- Mid-cut definition: `i = floor(N/2)` (nearest available cut if needed)\n")
        f.write(
            "- Edge-distance formula: `d_edge = min(min(cut-cut_min, cut_max-cut) for cut in tied_max_cuts)`\n"
        )
        f.write(f"- Tie policy for i_max: `{PEAK_TIE_POLICY}`\n\n")

        f.write("## Fit Summary (linear in 1/N)\n")
        f.write(
            f"- `S_peak`: intercept={fit_peak['intercept']:.8f}, slope={fit_peak['slope']:.5f}, "
            f"intercept_std={fit_peak['intercept_std']:.3e}, rmse={fit_peak['rmse']:.3e}\n"
        )
        f.write(
            f"- `S_mid`: intercept={fit_mid['intercept']:.8f}, slope={fit_mid['slope']:.5f}, "
            f"intercept_std={fit_mid['intercept_std']:.3e}, rmse={fit_mid['rmse']:.3e}\n"
        )
        f.write(
            f"- `A`: intercept={fit_A['intercept']:.8f}, slope={fit_A['slope']:.5f}, "
            f"intercept_std={fit_A['intercept_std']:.3e}, rmse={fit_A['rmse']:.3e}\n\n"
        )

        f.write("## Required Claims\n")
        f.write(f"- Does `S_peak` appear to saturate with size? {saturation_statement('S_peak', fit_peak, 'unc_S_peak')}\n")
        f.write(f"- Does `S_mid` appear to saturate with size? {saturation_statement('S_mid', fit_mid, 'unc_S_mid')}\n")
        f.write(f"- Does `A(N)` remain finite or decay? {A_statement}\n")
        f.write(f"- Is entropy maximum pinned near the boundary? {edge_statement}\n\n")

        f.write("## Limitations\n")
        f.write("- Fits are intentionally simple (linear in 1/N) and should be interpreted as controlled trend estimates.\n")
        f.write("- Uncertainty bars come from high-chi spread and do not include systematic model-selection uncertainty.\n")
    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_summary_markdown",
        status="executed",
        output_paths={"summary_markdown": summary_md},
    )

    record_artifact(
        provenance,
        bundle="finite_size_scaling",
        artifact_id="finite_size_bundle_metadata",
        status="executed",
        output_paths={"metadata": SIZE_BASE / "finite_size_scaling_metadata.json"},
    )
    metadata = {
        "bundle": "finite_size_scaling",
        "timestamp": now_iso(),
        "git_commit": get_git_commit(),
        "parameter_grid": {
            "mass": mass,
            "coupling": coupling,
            "N_values": sizes,
            "chi_ladder_for_uncertainty": chi_ladder,
        },
        "size_study_chi_choice": chi_choice,
        "size_study_chi_choice_reason": chi_choice_reason,
        "tied_max_cuts": {str(r["N"]): r["tied_max_cuts"] for r in selected_rows},
        "chosen_i_max": {str(r["N"]): int(r["chosen_i_max"]) for r in selected_rows},
        "d_edge": {str(r["N"]): int(r["d_edge"]) for r in selected_rows},
        "mid_cut_definition": "i=floor(N/2), nearest available cut if absent",
        "edge_distance_formula": "d_edge = min(min(cut-cut_min, cut_max-cut) for cut in tied_max_cuts)",
        "d_edge_definition": "minimum nearest-edge distance over tied maximum cuts",
        "chosen_i_max_definition": (
            "canonical representative of tied entropy maxima under mirrored_boundary_right policy"
        ),
        "tie_policy": PEAK_TIE_POLICY,
        "uncertainty_method": "high-chi spread across available chi in {64,96,128}",
        "fit_method": {
            "form": "linear",
            "independent_variable": "1/N",
            "observables": ["S_peak", "S_mid", "A"],
        },
        "outputs": {
            "base_directory": str(SIZE_BASE),
            "selected_table_csv": str(table_csv),
            "selected_table_json": str(table_json),
            "raw_table_csv": str(raw_csv),
            "raw_table_json": str(raw_json),
            "scaling_figure": str(fig_scaling),
            "structural_profile_figure": str(fig_profile),
            "summary_markdown": str(summary_md),
        },
        "commands_requested": bundle_provenance(provenance, "finite_size_scaling")["commands_requested"],
        "commands_executed": bundle_provenance(provenance, "finite_size_scaling")["commands_executed"],
        "commands_reused_cached_outputs": bundle_provenance(provenance, "finite_size_scaling")[
            "commands_reused_cached_outputs"
        ],
        "artifact_provenance": bundle_provenance(provenance, "finite_size_scaling")["artifacts"],
    }
    save_json(SIZE_BASE / "finite_size_scaling_metadata.json", metadata)

    return {
        "selected_rows": selected_rows,
        "raw_rows": raw_rows,
        "fits": {"S_peak": fit_peak, "S_mid": fit_mid, "A": fit_A},
        "metadata_path": SIZE_BASE / "finite_size_scaling_metadata.json",
        "table_csv": table_csv,
        "table_json": table_json,
        "raw_csv": raw_csv,
        "raw_json": raw_json,
        "figure_scaling": fig_scaling,
        "figure_profile": fig_profile,
        "summary": summary_md,
        "chi_choice": chi_choice,
        "chi_choice_reason": chi_choice_reason,
    }


def collect_output_paths(obj: Any) -> list[Path]:
    found: list[Path] = []
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(collect_output_paths(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(collect_output_paths(v))
    elif isinstance(obj, str):
        p = Path(obj)
        if p.suffix.lower() in {".png", ".csv"}:
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            found.append(p)
    return found


def newest_bundle_referenced_files() -> tuple[Path | None, set[Path]]:
    metadata_candidates: list[tuple[Path, str]] = []
    for p in SCRIPT_DIR.rglob("*metadata.json"):
        try:
            md = load_json(p)
        except Exception:
            continue
        if "bundle" in md and "timestamp" in md:
            metadata_candidates.append((p, str(md["timestamp"])))
    if not metadata_candidates:
        return None, set()
    newest_path, _ = max(metadata_candidates, key=lambda t: t[1])
    newest = load_json(newest_path)
    refs = set(collect_output_paths(newest.get("outputs", {})))
    return newest_path, refs


def build_duplicate_groups() -> list[dict[str, Any]]:
    per_run_metadata: list[Path] = []
    for p in SCRIPT_DIR.rglob("*.json"):
        if p.name in {
            "truncation_study_metadata.json",
            "finite_size_scaling_metadata.json",
            "application_breadth_metadata.json",
            "mass_sweep_metadata.json",
            "chi_convergence_metadata.json",
            "size_check_metadata.json",
        }:
            continue
        if p.name.endswith("metadata.json") or "_metadata" in p.name or p.name == "run_metadata.json":
            per_run_metadata.append(p)

    records: list[dict[str, Any]] = []
    for p in per_run_metadata:
        try:
            md = load_json(p)
        except Exception:
            continue

        outputs = md.get("outputs")
        if not isinstance(outputs, dict):
            continue

        args = md.get("args", {})
        script = str(md.get("script", "unknown"))
        bundle_role = str(md.get("application_bundle_role", args.get("application_bundle_role", "unknown")))
        output_dir_raw = md.get("output_directory")
        if isinstance(output_dir_raw, str) and output_dir_raw:
            output_dir = Path(output_dir_raw)
            if not output_dir.is_absolute():
                output_dir = (ROOT / output_dir).resolve()
        else:
            output_dir = p.parent.resolve()
        try:
            rel = output_dir.resolve().relative_to(SCRIPT_DIR.resolve())
            if len(rel.parts) >= 2:
                bundle_context = f"{rel.parts[0]}/{rel.parts[1]}"
            elif len(rel.parts) == 1:
                bundle_context = rel.parts[0]
            else:
                bundle_context = "."
        except Exception:
            bundle_context = "external"

        params = {
            "N": args.get("N"),
            "mass": args.get("m_over_g", args.get("mass")),
            "coupling": args.get("x", args.get("coupling")),
            "chi": args.get("chi"),
            "bc": args.get("bc", md.get("bc")),
            "bundle_role": bundle_role,
            "bundle_context": bundle_context,
        }

        for out_key, out_path in outputs.items():
            if not isinstance(out_path, str):
                continue
            pp = Path(out_path)
            if pp.suffix.lower() not in {".png", ".csv"}:
                continue
            if not pp.is_absolute():
                pp = (ROOT / pp).resolve()
            if not pp.exists():
                continue

            records.append(
                {
                    "path": pp,
                    "script": script,
                    "artifact_type": out_key,
                    "params": params,
                    "metadata_path": p.resolve(),
                    "timestamp": md.get("timestamp", ""),
                }
            )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        prm = r["params"]
        key = (
            r["script"],
            r["artifact_type"],
            prm.get("N"),
            prm.get("mass"),
            prm.get("coupling"),
            prm.get("chi"),
            prm.get("bc"),
            prm.get("bundle_role"),
            prm.get("bundle_context"),
        )
        grouped[key].append(r)

    dup_groups: list[dict[str, Any]] = []
    for key, recs in grouped.items():
        uniq = {}
        for r in recs:
            uniq[str(r["path"])] = r
        unique_recs = list(uniq.values())
        if len(unique_recs) <= 1:
            continue
        dup_groups.append({"key": key, "records": unique_recs})

    return dup_groups


def cleanup_duplicates() -> dict[str, Any]:
    print("Building duplicate manifest and archiving duplicates...")

    CLEANUP_BASE.mkdir(parents=True, exist_ok=True)
    archive_base = CLEANUP_BASE / "archive_duplicates"
    archive_base.mkdir(parents=True, exist_ok=True)

    newest_bundle_md, newest_refs = newest_bundle_referenced_files()
    dup_groups = build_duplicate_groups()

    manifest_groups: list[dict[str, Any]] = []
    planned_moves: list[dict[str, str]] = []

    for g in dup_groups:
        recs = g["records"]

        newest_ref_recs = [r for r in recs if r["path"] in newest_refs]
        if newest_ref_recs:
            canonical = max(newest_ref_recs, key=lambda r: r["path"].stat().st_mtime)
            reason = "Kept artifact referenced by newest bundle-level metadata; newest mtime within that subset."
        else:
            recs_sorted = sorted(recs, key=lambda r: r["path"].stat().st_mtime, reverse=True)
            canonical = recs_sorted[0]
            if len(recs_sorted) > 1 and recs_sorted[0]["path"].stat().st_mtime == recs_sorted[1]["path"].stat().st_mtime:
                in_canonical_dir = [r for r in recs_sorted if str(OUT_BASE.resolve()) in str(r["path"]) ]
                if in_canonical_dir:
                    canonical = in_canonical_dir[0]
                    reason = "Metadata tie and mtime tie; kept file in canonical application output directory."
                else:
                    reason = "No newest-bundle reference; kept newest file by mtime."
            else:
                reason = "No newest-bundle reference; kept newest file by mtime."

        files = [str(r["path"]) for r in recs]
        to_archive = [r for r in recs if r is not canonical]
        for r in to_archive:
            rel = r["path"].resolve().relative_to(ROOT.resolve())
            dst = archive_base / rel
            planned_moves.append({"src": str(r["path"]), "dst": str(dst)})

        manifest_groups.append(
            {
                "parameter_set": {
                    "N": g["key"][2],
                    "mass": g["key"][3],
                    "coupling": g["key"][4],
                    "chi": g["key"][5],
                    "bc": g["key"][6],
                    "bundle_role": g["key"][7],
                    "bundle_context": g["key"][8],
                },
                "script": g["key"][0],
                "artifact_type": g["key"][1],
                "files_considered": files,
                "canonical_file": str(canonical["path"]),
                "reason": reason,
                "planned_archive_files": [str(r["path"]) for r in to_archive],
            }
        )

    manifest = {
        "timestamp": now_iso(),
        "mode": "archive_duplicates",
        "newest_bundle_metadata_considered": str(newest_bundle_md) if newest_bundle_md else None,
        "duplicate_groups": manifest_groups,
        "planned_archive_moves": planned_moves,
    }

    manifest_path = CLEANUP_BASE / "duplicate_results_manifest.json"
    save_json(manifest_path, manifest)

    moved = 0
    touched_dirs: set[str] = set()
    retained: set[str] = set()

    for g in manifest_groups:
        retained.add(g["canonical_file"])

    for mv in planned_moves:
        src = Path(mv["src"])
        dst = Path(mv["dst"])
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1
        touched_dirs.add(str(src.parent))

    summary_path = CLEANUP_BASE / "duplicate_results_cleanup_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Duplicate Results Cleanup Summary\n\n")
        f.write("- Cleanup mode: archive duplicates (no permanent deletion)\n")
        f.write(f"- Manifest: `{manifest_path}`\n")
        f.write(f"- Files archived: `{moved}`\n")
        f.write(f"- Duplicate groups considered: `{len(manifest_groups)}`\n")
        f.write("\n## Directories Touched\n")
        if touched_dirs:
            for d in sorted(touched_dirs):
                f.write(f"- `{d}`\n")
        else:
            f.write("- None\n")
        f.write("\n## Canonical Outputs Retained\n")
        if retained:
            for p in sorted(retained):
                f.write(f"- `{p}`\n")
        else:
            f.write("- None\n")

    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "archived_count": moved,
        "groups": len(manifest_groups),
        "touched_dirs": sorted(touched_dirs),
    }


def make_fit_summary_table(
    trunc_data: dict[str, Any],
    size_data: dict[str, Any],
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []

    for obs, fit in trunc_data["fit_results"].items():
        unc = trunc_data["uncertainty"][obs]
        rows.append(
            {
                "observable": obs,
                "sizes_used": "N=20",
                "chi_used": "16,24,32,48,64,96,128",
                "fit_form": "linear in 1/chi (fallback) or truncation metric",
                "extrapolated_value": fit["intercept"],
                "fit_uncertainty": unc["combined"],
                "verdict": "stable" if abs(unc["combined"]) < 1e-2 else "needs caution",
            }
        )

    for obs, fit in size_data["fits"].items():
        rows.append(
            {
                "observable": obs,
                "sizes_used": "12,16,20,24,28,32",
                "chi_used": str(size_data["chi_choice"]),
                "fit_form": "linear in 1/N",
                "extrapolated_value": fit["intercept"],
                "fit_uncertainty": fit["intercept_std"] if np.isfinite(fit["intercept_std"]) else fit["rmse"],
                "verdict": "controlled trend",
            }
        )

    csv_path = OUT_BASE / "observable_fit_summary.csv"
    json_path = OUT_BASE / "observable_fit_summary.json"
    md_path = ROOT / "APPLICATION_PUBLICATION_VALIDATION_TABLE.md"

    fields = [
        "observable",
        "sizes_used",
        "chi_used",
        "fit_form",
        "extrapolated_value",
        "fit_uncertainty",
        "verdict",
    ]
    write_csv(csv_path, rows, fields)
    save_json(json_path, {"rows": rows})

    with open(md_path, "w") as f:
        f.write("# Application Publication Validation Table\n\n")
        f.write("| observable | sizes used | chi used | fit form | extrapolated value | fit uncertainty | verdict |\n")
        f.write("|---|---|---|---|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['observable']} | {r['sizes_used']} | {r['chi_used']} | {r['fit_form']} "
                f"| {float(r['extrapolated_value']):.8f} | {float(r['fit_uncertainty']):.3e} | {r['verdict']} |\n"
            )

    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run publication-grade validation bundles (truncation, finite-size scaling, duplicate cleanup)."
        )
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _ = parse_args(argv)
    ensure_dirs()
    provenance = init_provenance()
    prior_top_meta = (OUT_BASE / "publication_validation_metadata.json").exists()

    trunc_data = make_truncation_study(provenance)
    size_data = make_finite_size_scaling(provenance)
    fit_table = make_fit_summary_table(trunc_data, size_data)
    record_artifact(
        provenance,
        bundle="publication_grade_validation",
        artifact_id="fit_summary_table",
        status="executed",
        output_paths={
            "csv": fit_table["csv"],
            "json": fit_table["json"],
            "markdown": fit_table["markdown"],
        },
    )
    cleanup_data = cleanup_duplicates()
    record_artifact(
        provenance,
        bundle="publication_grade_validation",
        artifact_id="cleanup_bundle",
        status="executed",
        output_paths={
            "manifest": cleanup_data["manifest"],
            "summary_markdown": cleanup_data["summary"],
        },
    )

    commands_requested = provenance["commands_requested"]
    commands_executed = provenance["commands_executed"]
    commands_reused = provenance["commands_reused_cached_outputs"]
    full_run_performed = (
        len(commands_requested) > 0
        and len(commands_executed) == len(commands_requested)
        and len(commands_reused) == 0
    )
    rerun_after_cleanup = bool(prior_top_meta and len(commands_reused) > 0)
    provenance_note = (
        "commands_requested lists full intended CLI invocations; commands_executed are actually run; "
        "commands_reused_cached_outputs were skipped due to pre-existing complete outputs."
    )

    record_artifact(
        provenance,
        bundle="publication_grade_validation",
        artifact_id="publication_validation_metadata",
        status="executed",
        output_paths={"metadata": OUT_BASE / "publication_validation_metadata.json"},
    )
    top_meta = {
        "bundle": "publication_grade_validation",
        "timestamp": now_iso(),
        "git_commit": get_git_commit(),
        "full_run_performed": full_run_performed,
        "rerun_after_cleanup": rerun_after_cleanup,
        "provenance_note": provenance_note,
        "commands_requested": commands_requested,
        "commands_executed": commands_executed,
        "commands_reused_cached_outputs": commands_reused,
        "commands_run_total": len(commands_executed),
        "bundles": {
            "truncation_study_metadata": str(TRUNC_BASE / "truncation_study_metadata.json"),
            "finite_size_scaling_metadata": str(SIZE_BASE / "finite_size_scaling_metadata.json"),
            "duplicate_results_manifest": str(cleanup_data["manifest"]),
        },
        "outputs": {
            "truncation": {
                "table_csv": str(trunc_data["table_csv"]),
                "table_json": str(trunc_data["table_json"]),
                "figure": str(trunc_data["figure"]),
                "summary_markdown": str(trunc_data["summary"]),
            },
            "finite_size": {
                "table_csv": str(size_data["table_csv"]),
                "table_json": str(size_data["table_json"]),
                "raw_csv": str(size_data["raw_csv"]),
                "raw_json": str(size_data["raw_json"]),
                "scaling_figure": str(size_data["figure_scaling"]),
                "structural_profile_figure": str(size_data["figure_profile"]),
                "summary_markdown": str(size_data["summary"]),
            },
            "fit_summary_table": {
                "csv": str(fit_table["csv"]),
                "json": str(fit_table["json"]),
                "markdown": str(fit_table["markdown"]),
            },
            "cleanup": {
                "manifest": str(cleanup_data["manifest"]),
                "summary_markdown": str(cleanup_data["summary"]),
                "archived_count": cleanup_data["archived_count"],
            },
        },
        "artifact_provenance": bundle_provenance(provenance, "publication_grade_validation")["artifacts"],
    }
    save_json(OUT_BASE / "publication_validation_metadata.json", top_meta)

    print("\nPublication-grade validation completed.")
    print(f"Truncation metadata: {TRUNC_BASE / 'truncation_study_metadata.json'}")
    print(f"Finite-size metadata: {SIZE_BASE / 'finite_size_scaling_metadata.json'}")
    print(f"Cleanup manifest: {cleanup_data['manifest']}")
    print(f"Top metadata: {OUT_BASE / 'publication_validation_metadata.json'}")
    print(f"Commands executed: {len(commands_executed)}")


if __name__ == "__main__":
    main()
