#!/usr/bin/env python3
"""Schwinger Schmidt-decay/compressibility driver.

Scientific responsibility:
    Analyze Schmidt decay and cumulative retained weight at a chosen cut.
Main inputs:
    Schwinger parameters (N, mass, coupling, chi), cut/plot settings, optional reference options.
Main outputs:
    `schmidt_decay_*.csv`, `schmidt_decay*.png`, and metadata JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "l2l").exists():
        sys.path.insert(0, str(_p))
        break

import matplotlib.pyplot as plt
import numpy as np

from l2l.entanglement import extract_schmidt_values as _extract_schmidt_values, cumulative_weight as _cumulative_weight
from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter
from l2l.tfim_adapter import tfim_ground_state


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


def check_outputs_exist(outdir: Path, tag_suffix: str, compare_reference: bool) -> list[Path]:
    """Return list of output files that already exist."""
    expected = [
        outdir / f"schmidt_decay{tag_suffix}.png",
        outdir / f"schmidt_decay_schwinger{tag_suffix}.csv",
        outdir / f"schmidt_decay_metadata{tag_suffix}.json",
    ]
    if compare_reference:
        expected.append(outdir / f"schmidt_decay_tfim{tag_suffix}.csv")
    return [p for p in expected if p.exists()]


def load_cached_state(state_path: Path):
    """Load cached MPS object from pickle path."""
    with open(state_path, "rb") as f:
        loaded = pickle.load(f)
    psi = loaded.get("psi0") if isinstance(loaded, dict) and "psi0" in loaded else loaded
    if psi is None or not hasattr(psi, "L"):
        raise ValueError(
            f"Cached state at '{state_path}' is invalid: expected pickled psi object or dict containing 'psi0'."
        )
    return psi


def load_or_compute_state(args: argparse.Namespace, model: str = "schwinger"):
    """Return (psi, E0_or_none) from either compute or load mode."""
    if model != "schwinger":
        raise ValueError(f"Unsupported model '{model}' for state loading.")
    if args.state_source == "load":
        if not args.state_path:
            raise ValueError("--state-path is required when --state-source=load.")
        state_path = Path(args.state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        return load_cached_state(state_path), None

    adapter = SchwingerMassGapAdapter(m_over_g=args.mass, E0=0.0)
    result = adapter.dmrg_solve_point(args.N, {"x": args.coupling}, chi=args.chi, return_mps=True)
    return result["psi0"], result["E0"]


def extract_schmidt_values(state, cut: int) -> np.ndarray:
    """Thin wrapper for Schmidt value extraction at a given cut."""
    return _extract_schmidt_values(state, cut)


def compute_cumulative_weight(lambdas: np.ndarray) -> np.ndarray:
    """Thin wrapper for cumulative Schmidt-weight computation."""
    return _cumulative_weight(lambdas)


def validate_cut(cut: int, L: int, label: str) -> int:
    """Validate MPS cut index against current state length."""
    min_cut = 0
    max_cut = L - 2
    if cut < min_cut or cut > max_cut:
        raise ValueError(
            f"{label} cut {cut} is outside valid range [{min_cut}, {max_cut}] for MPS length L={L}."
        )
    return cut


def map_reference_cut(primary_cut: int, source_L: int, target_L: int) -> int:
    """Map cut via rounded fractional bond-position mapping."""
    if source_L < 2 or target_L < 2:
        raise ValueError(f"Cut mapping requires source and target lengths >= 2 (got {source_L}, {target_L}).")
    if source_L == 2:
        return 0
    source_bond = primary_cut + 1
    source_pos = (source_bond - 1) / (source_L - 2)
    mapped_bond = int(np.floor(1.0 + source_pos * (target_L - 2) + 0.5))
    return mapped_bond - 1


def sorted_validated_schmidt(lambdas: np.ndarray, label: str) -> np.ndarray:
    """Sort descending and check Schmidt-value sanity."""
    lambdas = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    min_lambda = float(np.min(lambdas))
    if min_lambda < -1e-12:
        raise ValueError(f"{label}: significant negative Schmidt value encountered (min={min_lambda:.3e}).")
    if min_lambda < 0.0:
        lambdas = np.where(lambdas < 0.0, 0.0, lambdas)
    weight = float(np.sum(lambdas**2))
    if not np.isclose(weight, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{label}: Schmidt weights are not normalized (sum(lambda^2)={weight:.8f}).")
    return lambdas


def save_metadata(
    outdir: Path,
    args: argparse.Namespace,
    tag_suffix: str,
    *,
    actual_primary_cut: int,
    actual_reference_cut: int | None,
    output_files: dict[str, Path],
) -> Path:
    """Save run metadata to script-specific JSON."""
    metadata_path = outdir / f"schmidt_decay_metadata{tag_suffix}.json"
    serialized_outputs = {k: str(v) for k, v in output_files.items()}
    serialized_outputs["metadata"] = str(metadata_path)
    metadata = {
        "script": "schmidt_decay_analysis.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "output_directory": str(outdir.resolve()),
        "tag": args.tag or "",
        "state_source": args.state_source,
        "state_path": args.state_path,
        "bc": args.bc,
        "application_bundle_role": args.application_bundle_role,
        "actual_primary_cut": actual_primary_cut,
        "actual_reference_cut": actual_reference_cut,
        "entropy_max_cut": args.entropy_max_cut,
        "representative_cut_note": args.representative_cut_note,
        "outputs": serialized_outputs,
        "args": {
            "N": args.N,
            "m_over_g": args.mass,
            "x": args.coupling,
            "chi": args.chi,
            "bc": args.bc,
            "cut": args.cut,
            "nvals": args.nvals,
            "plot": args.plot,
            "entropy_max_cut": args.entropy_max_cut,
            "representative_cut_note": args.representative_cut_note,
            "compare_reference": args.compare_reference,
            "reference_model": args.reference_model if args.compare_reference else None,
            "reference_N": args.reference_N if args.compare_reference else None,
            "reference_J": args.reference_J if args.compare_reference else None,
            "reference_g": args.reference_g if args.compare_reference else None,
            "reference_chi": args.reference_chi if args.compare_reference else None,
            "force": args.force,
            "show": args.show,
            "application_bundle_role": args.application_bundle_role,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def save_decay_csv(
    outdir: Path,
    cut: int,
    lambdas: np.ndarray,
    cum: np.ndarray,
    model: str,
    model_params: dict,
    tag_suffix: str,
) -> Path:
    filename = f"schmidt_decay_{model}{tag_suffix}.csv"
    filepath = outdir / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        if model == "schwinger":
            writer.writerow(["rank", "lambda", "lambda_sq", "cum_weight", "cut", "model", "N", "m_over_g", "x", "chi"])
            for i, (lam, cw) in enumerate(zip(lambdas, cum)):
                writer.writerow([i, lam, lam**2, cw, cut, model, model_params["N"], model_params["m_over_g"], model_params["x"], model_params["chi"]])
        else:
            writer.writerow(["rank", "lambda", "lambda_sq", "cum_weight", "cut", "model", "N", "J", "g", "chi"])
            for i, (lam, cw) in enumerate(zip(lambdas, cum)):
                writer.writerow([i, lam, lam**2, cw, cut, model, model_params["N"], model_params["J"], model_params["g"], model_params["chi"]])

    return filepath


def save_decay_data(
    outdir: Path,
    cut: int,
    lambdas: np.ndarray,
    cum: np.ndarray,
    model: str,
    model_params: dict,
    tag_suffix: str,
) -> Path:
    """Thin wrapper for decay CSV persistence."""
    return save_decay_csv(outdir, cut, lambdas, cum, model, model_params, tag_suffix)


def plot_decay(
    outdir: Path,
    schwinger_data: dict,
    tfim_data: dict | None,
    args: argparse.Namespace,
    tag_suffix: str,
    representative_note: str | None = None,
) -> Path:
    filename = f"schmidt_decay{tag_suffix}.png"
    filepath = outdir / filename

    plot_decay_only = args.plot in ("decay", "both")
    plot_cum_only = args.plot in ("cumulative", "both")
    ncols = 2 if args.plot == "both" else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), dpi=150)
    if ncols == 1:
        axes = [axes]

    ax_idx = 0

    if plot_decay_only:
        ax = axes[ax_idx]
        ax_idx += 1

        lam_s = schwinger_data["lambdas"][:args.nvals]
        ranks = np.arange(len(lam_s))
        ax.plot(ranks, lam_s, "s-", markersize=5, label=f"Schwinger (m/g={args.mass})")

        if tfim_data is not None:
            lam_t = tfim_data["lambdas"][:args.nvals]
            ax.plot(np.arange(len(lam_t)), lam_t, "o--", markersize=4, label=f"TFIM (J={args.reference_J}, g={args.reference_g})")

        ax.set_xlabel("Rank $k$", fontsize=12)
        ax.set_ylabel(r"Schmidt value $\lambda_k$", fontsize=12)
        ax.set_title(f"Schmidt value decay (cut {args.cut})", fontsize=11)
        if args.logy:
            ax.set_yscale("log")
        if args.logx:
            ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if plot_cum_only:
        ax = axes[ax_idx]

        cum_s = schwinger_data["cum"][:args.nvals]
        ranks = np.arange(len(cum_s))
        ax.plot(ranks, cum_s, "s-", markersize=5, label=f"Schwinger (m/g={args.mass})")

        if tfim_data is not None:
            cum_t = tfim_data["cum"][:args.nvals]
            ax.plot(np.arange(len(cum_t)), cum_t, "o--", markersize=4, label=f"TFIM (J={args.reference_J}, g={args.reference_g})")

        ax.axhline(0.99, color="gray", linestyle=":", alpha=0.7, label="99% weight")
        ax.set_xlabel("Rank $k$", fontsize=12)
        ax.set_ylabel(r"Cumulative weight $\sum_{j \leq k} \lambda_j^2$", fontsize=12)
        ax.set_title(f"Cumulative retained weight (cut {args.cut})", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    if representative_note:
        fig.text(0.5, 0.01, representative_note, ha="center", va="bottom", fontsize=8)
        plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    else:
        plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()

    return filepath


def plot_cumulative(
    outdir: Path,
    schwinger_data: dict,
    tfim_data: dict | None,
    args: argparse.Namespace,
    tag_suffix: str,
    representative_note: str | None = None,
) -> Path:
    """Thin wrapper entrypoint for cumulative visualization mode."""
    return plot_decay(outdir, schwinger_data, tfim_data, args, tag_suffix, representative_note)


def plot_combined_comparison(
    outdir: Path,
    schwinger_data: dict,
    tfim_data: dict | None,
    args: argparse.Namespace,
    tag_suffix: str,
    representative_note: str | None = None,
) -> Path:
    """Thin wrapper entrypoint for decay/cumulative comparison plotting."""
    return plot_decay(outdir, schwinger_data, tfim_data, args, tag_suffix, representative_note)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Schmidt value decay for tensor-network compressibility."
    )
    model_args = parser.add_argument_group("Model Parameters")
    model_args.add_argument("--N", type=int, required=True, help="System size (number of lattice sites)")
    model_args.add_argument("--mass", type=float, required=True, help="Schwinger mass ratio m/g")
    model_args.add_argument("--coupling", type=float, required=True, help="Schwinger coupling x = 1/(ag)^2")
    model_args.add_argument("--chi", type=int, required=True, help="Maximum bond dimension chi")
    model_args.add_argument(
        "--bc",
        type=str,
        default="open",
        choices=["open", "periodic"],
        help="Boundary condition label (currently only 'open' is supported at runtime)",
    )

    state_args = parser.add_argument_group("State Source")
    state_args.add_argument(
        "--state-source",
        type=str,
        default="compute",
        choices=["compute", "load"],
        help="Ground-state source: compute via DMRG or load from cache",
    )
    state_args.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Path to cached pickled state (MPS object or dict containing key 'psi0')",
    )

    analysis_args = parser.add_argument_group("Decay Analysis")
    analysis_args.add_argument("--cut", type=int, required=True, help="Representative MPS cut index")
    analysis_args.add_argument("--nvals", type=int, default=32, help="Number of Schmidt values to display")
    analysis_args.add_argument("--plot", type=str, default="both", choices=["decay", "cumulative", "both"], help="Plot mode")
    analysis_args.add_argument("--entropy-max-cut", type=int, default=None, help="Entropy-profile maximum cut, if known")
    analysis_args.add_argument(
        "--representative-cut-note",
        type=str,
        default=None,
        help="Explicit note for representative cut choice when entropy max differs",
    )
    analysis_args.add_argument("--logx", action="store_true", help="Use logarithmic x-axis where applicable")
    analysis_args.add_argument("--logy", action="store_true", help="Use logarithmic y-axis where applicable")

    ref_args = parser.add_argument_group("Reference Comparison")
    ref_args.add_argument("--compare-reference", action="store_true", help="Enable TFIM reference comparison")
    ref_args.add_argument("--reference-model", type=str, default="tfim", choices=["tfim"], help="Reference model name")
    ref_args.add_argument("--reference-N", type=int, default=None, help="Reference system size (defaults to N)")
    ref_args.add_argument("--reference-J", type=float, default=1.0, help="Reference TFIM coupling J")
    ref_args.add_argument("--reference-g", type=float, default=1.0, help="Reference TFIM transverse field g")
    ref_args.add_argument("--reference-chi", type=int, default=None, help="Reference max bond dimension (defaults to chi)")

    io_args = parser.add_argument_group("Output and Run Control")
    io_args.add_argument("--outdir", type=str, required=True, help="Directory for output files")
    io_args.add_argument("--tag", type=str, default=None, help="Optional output tag suffix")
    io_args.add_argument(
        "--application-bundle-role",
        type=str,
        default="primary",
        choices=["primary", "secondary", "comparison"],
        help="Run role within an application result bundle",
    )
    io_args.add_argument("--force", action="store_true", help="Overwrite existing outputs if present")
    io_args.add_argument("--show", action="store_true", help="Display figure interactively")

    args = parser.parse_args()

    if args.bc != "open":
        raise ValueError(f"Unsupported boundary condition '{args.bc}'. Only 'open' is currently supported.")

    if args.reference_N is None:
        args.reference_N = args.N
    if args.reference_chi is None:
        args.reference_chi = args.chi

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_tag(args.tag)}" if args.tag else ""

    existing = check_outputs_exist(outdir, tag_suffix, args.compare_reference)
    if not args.force and existing:
        print(f"Error: Output files already exist: {[str(p) for p in existing]}")
        print("Use --force to overwrite.")
        sys.exit(1)

    if args.state_source == "compute":
        print(f"Computing Schwinger ground state: N={args.N}, m/g={args.mass}, x={args.coupling}, chi={args.chi}")
    else:
        print(f"Loading cached Schwinger state from: {args.state_path}")
    psi_schwinger, E0_s = load_or_compute_state(args, model="schwinger")
    if E0_s is not None:
        print(f"Schwinger E0 = {E0_s:.10f}")

    actual_primary_cut = validate_cut(args.cut, psi_schwinger.L, "Primary")

    print(f"Extracting Schmidt values at cut {actual_primary_cut}...")
    lambdas_s = extract_schmidt_values(psi_schwinger, actual_primary_cut)
    lambdas_s = sorted_validated_schmidt(lambdas_s, "Schwinger")
    cum_s = compute_cumulative_weight(lambdas_s)
    schwinger_data = {"lambdas": lambdas_s, "cum": cum_s}

    schwinger_csv = save_decay_data(
        outdir,
        actual_primary_cut,
        lambdas_s,
        cum_s,
        "schwinger",
        {"N": args.N, "m_over_g": args.mass, "x": args.coupling, "chi": args.chi},
        tag_suffix,
    )

    tfim_data = None
    tfim_csv = None
    actual_reference_cut = None
    if args.compare_reference:
        print(f"Computing TFIM ground state: N={args.reference_N}, J={args.reference_J}, g={args.reference_g}, chi={args.reference_chi}")
        E0_tfim, psi_tfim = tfim_ground_state(
            N=args.reference_N, J=args.reference_J, g=args.reference_g, chi=args.reference_chi
        )
        print(f"TFIM E0 = {E0_tfim:.10f}")

        if psi_tfim.L != psi_schwinger.L:
            mapped_cut = map_reference_cut(actual_primary_cut, psi_schwinger.L, psi_tfim.L)
        else:
            mapped_cut = actual_primary_cut
        actual_reference_cut = validate_cut(mapped_cut, psi_tfim.L, "Reference")

        lambdas_t = extract_schmidt_values(psi_tfim, actual_reference_cut)
        lambdas_t = sorted_validated_schmidt(lambdas_t, "TFIM")
        cum_t = compute_cumulative_weight(lambdas_t)
        tfim_data = {"lambdas": lambdas_t, "cum": cum_t}

        tfim_csv = save_decay_data(
            outdir,
            actual_reference_cut,
            lambdas_t,
            cum_t,
            "tfim",
            {"N": args.reference_N, "J": args.reference_J, "g": args.reference_g, "chi": args.reference_chi},
            tag_suffix,
        )

    representative_note_for_figure = None
    if args.entropy_max_cut is not None and args.entropy_max_cut != actual_primary_cut:
        representative_note_for_figure = (
            args.representative_cut_note.strip()
            if args.representative_cut_note
            else "Representative interior cut used for spectrum/decay; entropy maximum tracked separately."
        )

    png_file = plot_combined_comparison(
        outdir, schwinger_data, tfim_data, args, tag_suffix, representative_note=representative_note_for_figure
    )
    output_files = {
        "figure": png_file,
        "schwinger_csv": schwinger_csv,
    }
    if tfim_csv is not None:
        output_files["tfim_csv"] = tfim_csv
    metadata_file = save_metadata(
        outdir,
        args,
        tag_suffix,
        actual_primary_cut=actual_primary_cut,
        actual_reference_cut=actual_reference_cut,
        output_files=output_files,
    )

    print()
    print("Run completed: Schmidt decay analysis")
    print(f"N={args.N}, m/g={args.mass}, x={args.coupling}, chi={args.chi}, bc={args.bc}")
    print(f"Application bundle role: {args.application_bundle_role}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Cut: {actual_primary_cut}, Displayed {min(args.nvals, len(lambdas_s))} Schmidt values")
    print(f"Plot mode: {args.plot}")
    if representative_note_for_figure:
        print(f"Representative-cut note: {representative_note_for_figure}")
    if args.compare_reference:
        print(f"Reference comparison: TFIM enabled (reference cut {actual_reference_cut})")
    print(f"Saved figure: {png_file}")
    print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    main()
