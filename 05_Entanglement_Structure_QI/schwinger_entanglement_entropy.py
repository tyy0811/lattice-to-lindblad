#!/usr/bin/env python3
"""Schwinger entanglement entropy driver.

Scientific responsibility:
    Compute and visualize bipartite von Neumann entropy across all MPS cuts.
Main inputs:
    Schwinger parameters (N, mass, coupling, chi), state source, and output options.
Main outputs:
    `entropy_profile*.csv`, `entropy_profile*.png`, and script-specific metadata JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path
_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "l2l").exists():
        sys.path.insert(0, str(_p))
        break

import matplotlib.pyplot as plt
import numpy as np

from l2l.entanglement import compute_entropy_profile as _compute_entropy_profile
from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter


def sanitize_tag(tag: str) -> str:
    """Make tag safe for filenames."""
    return tag.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not available."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_HERE.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def check_outputs_exist(outdir: Path, tag_suffix: str) -> list[Path]:
    """Return list of output files that already exist."""
    expected = [
        outdir / f"entropy_profile{tag_suffix}.csv",
        outdir / f"entropy_profile{tag_suffix}.png",
        outdir / f"entropy_profile_metadata{tag_suffix}.json",
    ]
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


def load_or_compute_ground_state(args: argparse.Namespace):
    """Return (psi, E0_or_none) from either compute or load mode."""
    if args.state_source == "load":
        if not args.state_path:
            raise ValueError("--state-path is required when --state-source=load.")
        state_path = Path(args.state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        psi = load_cached_state(state_path)
        return psi, None

    adapter = SchwingerMassGapAdapter(m_over_g=args.mass, E0=0.0)
    result = adapter.dmrg_solve_point(
        args.N,
        {"x": args.coupling},
        chi=args.chi,
        return_mps=True,
    )
    return result["psi0"], result["E0"]


def compute_entropy_profile(state):
    """Thin wrapper for entropy extraction from an MPS state."""
    return _compute_entropy_profile(state)


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
    tie_policy: str = "mirrored_boundary_right",
    N: int | None = None,
) -> dict:
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


def save_metadata(
    outdir: Path,
    args: argparse.Namespace,
    script_name: str,
    tag_suffix: str,
    *,
    actual_primary_cut: int | None,
    actual_reference_cut: int | None,
    max_entropy: float,
    max_entropy_cut: int,
    tied_max_cuts: list[int],
    tie_policy: str,
    chosen_i_max: int,
    d_edge: int,
    output_files: dict[str, Path],
) -> Path:
    """Save run metadata to JSON."""
    metadata_path = outdir / f"entropy_profile_metadata{tag_suffix}.json"
    serialized_outputs = {k: str(v) for k, v in output_files.items()}
    serialized_outputs["metadata"] = str(metadata_path)
    metadata = {
        "script": script_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "output_directory": str(outdir.resolve()),
        "tag": args.tag or "",
        "state_source": args.state_source,
        "state_path": args.state_path,
        "bc": args.bc,
        "application_bundle_role": args.application_bundle_role,
        "max_entropy": max_entropy,
        "max_entropy_cut": max_entropy_cut,
        "tied_max_cuts": [int(c) for c in tied_max_cuts],
        "n_tied_max_cuts": len(tied_max_cuts),
        "tie_policy": tie_policy,
        "chosen_i_max": chosen_i_max,
        "d_edge": d_edge,
        "d_edge_definition": "minimum nearest-edge distance over tied maximum cuts",
        "chosen_i_max_definition": (
            "canonical representative of tied entropy maxima under mirrored_boundary_right policy"
        ),
        "actual_primary_cut": actual_primary_cut,
        "actual_reference_cut": actual_reference_cut,
        "outputs": serialized_outputs,
        "args": {
            "N": args.N,
            "m_over_g": args.mass,
            "x": args.coupling,
            "chi": args.chi,
            "bc": args.bc,
            "force": args.force,
            "show": args.show,
            "application_bundle_role": args.application_bundle_role,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def save_entropy_csv(
    outdir: Path,
    cuts: np.ndarray,
    entropies: np.ndarray,
    args: argparse.Namespace,
    tag_suffix: str,
) -> Path:
    """Save entropy profile to CSV."""
    filename = f"entropy_profile{tag_suffix}.csv"
    filepath = outdir / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cut", "entropy", "N", "m_over_g", "x", "chi"])
        for cut, ent in zip(cuts, entropies):
            writer.writerow([cut, ent, args.N, args.mass, args.coupling, args.chi])

    return filepath


def save_entropy_data(
    outdir: Path,
    cuts: np.ndarray,
    entropies: np.ndarray,
    args: argparse.Namespace,
    tag_suffix: str,
) -> Path:
    """Thin wrapper for entropy CSV persistence."""
    return save_entropy_csv(outdir, cuts, entropies, args, tag_suffix)


def plot_entropy_profile(
    outdir: Path,
    cuts: np.ndarray,
    entropies: np.ndarray,
    args: argparse.Namespace,
    tag_suffix: str,
) -> Path:
    """Generate entropy profile figure."""
    filename = f"entropy_profile{tag_suffix}.png"
    filepath = outdir / filename

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(cuts, entropies, "o-", markersize=4, linewidth=1.5, color="C0")

    ax.set_xlabel("MPS cut index $i$", fontsize=12)
    ax.set_ylabel(r"Entanglement entropy $S_{\rm vN}$", fontsize=12)
    ax.set_title(
        f"Schwinger model: N={args.N}, m/g={args.mass}, x={args.coupling}, χ={args.chi}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    peak_info = find_entropy_peak_info(cuts, entropies, N=args.N)
    ax.axvline(
        peak_info["chosen_i_max"],
        color="C1",
        linestyle="--",
        alpha=0.7,
        label=f"max at cut {peak_info['chosen_i_max']}",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()

    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute entanglement entropy profile for Schwinger model ground state."
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag_suffix = f"_{sanitize_tag(args.tag)}" if args.tag else ""

    existing = check_outputs_exist(outdir, tag_suffix)
    if not args.force and existing:
        print(f"Error: Output files already exist: {[str(p) for p in existing]}")
        print("Use --force to overwrite.")
        sys.exit(1)

    if args.state_source == "compute":
        print(f"Computing Schwinger ground state: N={args.N}, m/g={args.mass}, x={args.coupling}, chi={args.chi}")
    else:
        print(f"Loading cached Schwinger state from: {args.state_path}")
    psi, E0 = load_or_compute_ground_state(args)
    if E0 is not None:
        print(f"Ground state energy: E0 = {E0:.10f}")

    print("Computing entropy profile...")
    cuts, entropies = compute_entropy_profile(psi)

    tie_policy = "mirrored_boundary_right"
    peak_info = find_entropy_peak_info(cuts, entropies, tie_policy=tie_policy, N=args.N)
    max_entropy = float(peak_info["max_entropy"])
    tied_max_cuts = [int(c) for c in peak_info["tied_max_cuts"]]
    max_cut = int(peak_info["chosen_i_max"])
    d_edge = int(peak_info["d_edge"])

    csv_file = save_entropy_data(outdir, cuts, entropies, args, tag_suffix)
    png_file = plot_entropy_profile(outdir, cuts, entropies, args, tag_suffix)
    metadata_file = save_metadata(
        outdir,
        args,
        "schwinger_entanglement_entropy.py",
        tag_suffix,
        actual_primary_cut=None,
        actual_reference_cut=None,
        max_entropy=max_entropy,
        max_entropy_cut=max_cut,
        tied_max_cuts=tied_max_cuts,
        tie_policy=tie_policy,
        chosen_i_max=max_cut,
        d_edge=d_edge,
        output_files={"csv": csv_file, "figure": png_file},
    )

    print()
    print("Run completed: Schwinger entanglement entropy")
    print(f"N={args.N}, m/g={args.mass}, x={args.coupling}, chi={args.chi}, bc={args.bc}")
    print(f"Application bundle role: {args.application_bundle_role}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Computed entropy on {len(cuts)} bipartitions")
    print(f"Max entropy: {max_entropy:.4f} at cut {max_cut}")
    if len(tied_max_cuts) > 1:
        print(
            f"Tied maxima detected: cuts {tied_max_cuts} "
            f"(policy={tie_policy}, chosen_i_max={max_cut}, d_edge={d_edge})"
        )
    print(f"Saved data: {csv_file}")
    print(f"Saved figure: {png_file}")
    print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    main()
