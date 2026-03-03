#!/usr/bin/env python3
"""schwinger_dmrg.py (refactored wrapper)

Drop-in replacement wrapper for producing the DMRG mass-gap grid CSV used by the
Schwinger joint extrapolation pipeline.

Outputs (same filenames as before by default):
  - dmrg_massgap_results.csv (raw energies + optional ED reference gaps)
  - dmrg_massgap_grid.csv    (source,x,N,ag2,mg) for joint extrapolation
  - dmrg_massgap_plot.png    (quick validation plots)

This wrapper is deliberately thin: the heavy lifting is in l2l/ via the
SchwingerMassGapAdapter and the generic grid engine.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure repo root is importable when running from a stage folder
_HERE = _Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "utils_QOS.py").exists() or (_p / ".git").exists():
        sys.path.insert(0, str(_p))
        break

import argparse
import math
import csv

import pandas as pd

from l2l.grid_engines import DMRGRunSpec, run_dmrg_grid
from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter


# -------------------- Default config (edit here if you don't want CLI) --------------------
CONFIG = {
    "m_over_g": 0.0,
    "E0": 0.0,
    "chi": 80,
    "x_list": [4.0, 8.0, 12.0],
    "N_list": [4, 8, 12, 20, 30, 40],
    "include_ed_ref_for_N_le": 20,
    "out_results": "dmrg_massgap_results.csv",
    "out_grid": "dmrg_massgap_grid.csv",
    "out_plot": "dmrg_massgap_plot.png",
}


def plot_results(results_csv: str, out_png: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(results_csv, newline="") as f:
        for r in csv.DictReader(f):
            x = float(r["x"])
            N = int(r["N"])
            ga = 1.0 / math.sqrt(x)
            ag2 = ga ** 2
            gap_dmrg = float(r["gap_dmrg"])
            mg_dmrg = gap_dmrg * ga / 2.0

            mg_ed = None
            if r.get("gap_ed_ref") not in (None, "", "nan", "NaN"):
                try:
                    mg_ed = float(r["gap_ed_ref"]) * ga / 2.0
                except Exception:
                    mg_ed = None

            rows.append(dict(x=x, N=N, ag2=ag2, mg_dmrg=mg_dmrg, mg_ed=mg_ed))

    xs = sorted({r["x"] for r in rows})
    Ns = sorted({r["N"] for r in rows})
    exact = 1.0 / math.sqrt(math.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Left: finite-size vs 1/N per x
    for x in xs:
        pts = sorted([r for r in rows if r["x"] == x], key=lambda t: t["N"])
        invN = np.array([1.0 / p["N"] for p in pts])
        y_dm = np.array([p["mg_dmrg"] for p in pts])
        ax1.plot(invN, y_dm, "-", alpha=0.3)
        ax1.plot(invN, y_dm, "s", mfc="none", mew=1.5, ms=7, label=f"x={int(x)} (DMRG)")

        y_ed = [(1.0 / p["N"], p["mg_ed"]) for p in pts if p["mg_ed"] is not None]
        if y_ed:
            ax1.plot([t[0] for t in y_ed], [t[1] for t in y_ed], "o", ms=6, label=f"x={int(x)} (ED)")

    ax1.axhline(exact, color="black", ls="--", lw=1.2, label=r"Exact $1/\sqrt{\pi}$")
    ax1.set_xlabel(r"$1/N$")
    ax1.set_ylabel(r"$M_{\rm gap}/g$")
    ax1.set_title("Finite-size convergence")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    # Right: continuum vs ag2 per N
    for N in Ns:
        ptsN = sorted([r for r in rows if r["N"] == N], key=lambda t: t["ag2"])
        ag2 = [p["ag2"] for p in ptsN]

        if all(p["mg_ed"] is not None for p in ptsN):
            mg = [p["mg_ed"] for p in ptsN]
            ax2.plot(ag2, mg, "o-", alpha=0.35, lw=1, label=f"N={N} (ED)")
        else:
            mg = [p["mg_dmrg"] for p in ptsN]
            ax2.plot(ag2, mg, "s--", mfc="none", alpha=0.5, lw=1, label=f"N={N} (DMRG)")

    ax2.axhline(exact, color="black", ls="--", lw=1.2, label=r"Exact $1/\sqrt{\pi}$")
    ax2.set_xlabel(r"$(ag)^2 = 1/x$")
    ax2.set_ylabel(r"$M_{\rm gap}/g$")
    ax2.set_title("Continuum extrapolation")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle(r"Schwinger model mass gap: ED validated, DMRG extended ($m/g=0$)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    print(f"Saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", nargs="+", type=float, default=None)
    ap.add_argument("--N", nargs="+", type=int, default=None)
    ap.add_argument("--chi", type=int, default=None)
    ap.add_argument("--m_over_g", type=float, default=None)
    ap.add_argument("--E0", type=float, default=None)
    ap.add_argument("--out_results", default=None)
    ap.add_argument("--out_grid", default=None)
    ap.add_argument("--out_plot", default=None)
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.x is not None:
        cfg["x_list"] = [float(v) for v in args.x]
    if args.N is not None:
        cfg["N_list"] = [int(v) for v in args.N]
    if args.chi is not None:
        cfg["chi"] = int(args.chi)
    if args.m_over_g is not None:
        cfg["m_over_g"] = float(args.m_over_g)
    if args.E0 is not None:
        cfg["E0"] = float(args.E0)
    if args.out_results is not None:
        cfg["out_results"] = str(args.out_results)
    if args.out_grid is not None:
        cfg["out_grid"] = str(args.out_grid)
    if args.out_plot is not None:
        cfg["out_plot"] = str(args.out_plot)

    thr = int(cfg["include_ed_ref_for_N_le"])
    adapter = SchwingerMassGapAdapter(m_over_g=cfg["m_over_g"], E0=cfg["E0"])

    df = run_dmrg_grid(
        adapter,
        N_list=cfg["N_list"],
        params_list=[{"x": float(x)} for x in cfg["x_list"]],
        spec=DMRGRunSpec(chi=int(cfg["chi"])),
        include_ed_reference=True,
        include_ed_reference_max_N=thr,
        quiet=False,
    )

    # Mask ED references above threshold for parity with the older script
    if "gap_ed_ref" in df.columns:
        df.loc[df["N"] > thr, "gap_ed_ref"] = float("nan")

    out_results = cfg["out_results"]
    _Path(out_results).parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame({
        "x": df["x"].astype(float),
        "N": df["N"].astype(int),
        "E0_dmrg": df["E0"].astype(float),
        "E1_dmrg": df["E1"].astype(float),
        "gap_dmrg": df["gap"].astype(float),
        "gap_ed_ref": df.get("gap_ed_ref", pd.Series([float("nan")] * len(df))).astype(float),
    })
    raw.to_csv(out_results, index=False)
    print(f"Saved {len(raw)} rows to {out_results}")

    out_grid = cfg["out_grid"]
    _Path(out_grid).parent.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame({
        "source": "DMRG",
        "x": df["x"].astype(float),
        "N": df["N"].astype(int),
        "ag2": df["ag2"].astype(float),
        "mg": df["mg"].astype(float),
    })
    grid.to_csv(out_grid, index=False)
    print(f"Saved {out_grid}")

    _Path(cfg["out_plot"]).parent.mkdir(parents=True, exist_ok=True)
    plot_results(out_results, out_png=cfg["out_plot"])


if __name__ == "__main__":
    main()
