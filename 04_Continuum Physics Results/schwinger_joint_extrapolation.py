#!/usr/bin/env python3
"""schwinger_joint_extrapolation.py (refactored wrapper)

Drop-in replacement wrapper for joint thermodynamic + continuum extrapolation.

Inputs:
  - ED grid CSV from schwinger_continuum_massgap.py
  - DMRG grid CSV from schwinger_dmrg.py

Expected CSV schema (unchanged):
  source,x,N,ag2,mg

Fit variables (Schwinger conventions):
  u = 1/N
  v = (ag)^2  (here: ag2 column)

Policy:
  - ED points only for N <= max_ed_N
  - DMRG points only for N in fit_dmrg_N
  - ED wins on duplicate (x,N)

Uncertainty:
  - bootstrap on rows (optionally stratified)
  - leave-one-x-out jackknife (systematic)
  - model-spread across ansätze (systematic)
  - total reported as quadrature (transparent summary)
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from l2l.joint_extrapolation import (
    build_weights,
    fit_joint,
    bootstrap_joint,
    leave_one_x_out,
    model_spread,
    design_matrix,
)

TARGET = 1.0 / math.sqrt(math.pi)
STAGE_DIR = _HERE.parent


def load_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["x", "ag2", "mg", "N"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "ag2", "mg", "N"]).copy()
    df["N"] = df["N"].astype(int)
    if "source" not in df.columns:
        df["source"] = "UNK"
    return df[["source", "x", "N", "ag2", "mg"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed_csv", default=str(STAGE_DIR / "ed_massgap_grid.csv"))
    ap.add_argument("--dmrg_csv", default=str(STAGE_DIR / "dmrg_massgap_grid.csv"))
    ap.add_argument("--max_ed_N", type=int, default=20)
    ap.add_argument("--fit_dmrg_N", nargs="+", type=int, default=[30, 40])
    ap.add_argument("--min_fit_N", type=int, default=14)
    ap.add_argument("--weighting", default="x_source", choices=["none", "x", "x_source"])
    ap.add_argument("--bootstrap_stratify", default="x_source", choices=["none", "x", "x_source"])
    ap.add_argument("--models", nargs="+", default=["lin", "u2", "u3"])
    ap.add_argument("--baseline_model", default="u2", choices=["lin", "u2", "u3"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_png", default=str(STAGE_DIR / "massgap_joint_extrapolation.png"))
    ap.add_argument("--report_csv", default=str(STAGE_DIR / "joint_fit_report.csv"))
    args = ap.parse_args()

    ed = load_grid(args.ed_csv)
    dm = load_grid(args.dmrg_csv)

    # Plot dataset: keep all points (including ED/DMRG overlaps) for visibility.
    df_plot = pd.concat([ed, dm], ignore_index=True)

    # Fit dataset policy + min_fit_N
    ed_fit = ed[(ed["N"] <= args.max_ed_N) & (ed["N"] >= args.min_fit_N)].copy()
    dm_fit = dm[(dm["N"].isin(args.fit_dmrg_N)) & (dm["N"] >= args.min_fit_N)].copy()
    ed_fit_keys = set(zip(ed_fit["x"], ed_fit["N"]))
    dm_fit = dm_fit[~dm_fit.apply(lambda r: (r["x"], r["N"]) in ed_fit_keys, axis=1)]
    df_fit = pd.concat([ed_fit, dm_fit], ignore_index=True)
    if df_fit.empty:
        raise SystemExit("Fit dataset is empty after applying policy")

    df_fit = df_fit.copy()
    df_fit["u"] = 1.0 / df_fit["N"].to_numpy(dtype=float)
    df_fit["v"] = df_fit["ag2"].to_numpy(dtype=float)
    df_fit["y"] = df_fit["mg"].to_numpy(dtype=float)
    df_fit["x_source"] = df_fit["x"].astype(str) + "|" + df_fit["source"].astype(str)

    weights = build_weights(df_fit, weighting=args.weighting)

    base = fit_joint(df_fit, model=args.baseline_model, weights=weights)
    strat = None if args.bootstrap_stratify == "none" else args.bootstrap_stratify
    if strat == "x_source":
        strat = "x_source"
    elif strat == "x":
        strat = "x"

    betas = bootstrap_joint(
        df_fit,
        model=args.baseline_model,
        nboot=int(args.bootstrap),
        seed=int(args.seed),
        weights=weights,
        stratify=strat,
    )

    M00_s = betas[:, 0]
    M00_mean = float(np.mean(M00_s))
    M00_std = float(np.std(M00_s, ddof=1))
    M00_p16, M00_p84 = np.percentile(M00_s, [16, 84])
    M00_p025, M00_p975 = np.percentile(M00_s, [2.5, 97.5])

    m_models, sys_model = model_spread(df_fit, models=list(args.models), weights=weights)
    m_x, sys_x = leave_one_x_out(df_fit, model=args.baseline_model, weights=weights)
    total = float(math.sqrt(M00_std**2 + sys_model**2 + sys_x**2))

    rep = pd.DataFrame([{
        "baseline_model": args.baseline_model,
        "models": " ".join(args.models),
        "n_fit": int(len(df_fit)),
        "min_fit_N": int(args.min_fit_N),
        "max_ed_N": int(args.max_ed_N),
        "fit_dmrg_N": " ".join(map(str, args.fit_dmrg_N)),
        "weighting": args.weighting,
        "bootstrap_stratify": args.bootstrap_stratify,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "M00_mean": M00_mean,
        "M00_std": M00_std,
        "M00_p16": float(M00_p16),
        "M00_p84": float(M00_p84),
        "M00_p025": float(M00_p025),
        "M00_p975": float(M00_p975),
        "sys_model_std": float(sys_model),
        "sys_x_jackknife": float(sys_x),
        "total": total,
        "target": TARGET,
    }])
    _Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.report_csv, index=False)

    print("Baseline bootstrap (statistical) summary:")
    print(f"  M_gap/g at (u=0,v=0): {M00_mean:.6f} ± {M00_std:.6f}   (16–84%: [{M00_p16:.6f}, {M00_p84:.6f}])")
    print(f"  95%: [{M00_p025:.6f}, {M00_p975:.6f}]")
    print("Model-variation (systematic) summary:")
    print(f"  M00 models: {', '.join(f'{v:.6f}' for v in m_models)} ; std={sys_model:.6f}")
    print("Leave-one-x-out jackknife (systematic) summary:")
    if len(m_x):
        print(f"  M00 estimates: {', '.join(f'{v:.6f}' for v in m_x)}")
    print(f"  jackknife SE = {sys_x:.6f}")
    print("Combined error budget (quadrature; transparent, not sacred):")
    print(f"  M00 = {M00_mean:.6f} ± {total:.6f}   [stat={M00_std:.6f}, sys_model={sys_model:.6f}, sys_x={sys_x:.6f}]")
    print(f"Exact 1/sqrt(pi): {TARGET:.6f}")

    # Plot: u=0 curve with bootstrap bands
    v_grid = np.linspace(0, float(df_fit["v"].max()) * 1.05, 250)
    Xg, _ = design_matrix(np.zeros_like(v_grid), v_grid, args.baseline_model)
    curve_hat = Xg @ base.beta
    curve_samps = (Xg[None, :, :] @ betas[:, :, None]).squeeze(-1)
    band68_lo = np.percentile(curve_samps, 16, axis=0)
    band68_hi = np.percentile(curve_samps, 84, axis=0)
    band95_lo = np.percentile(curve_samps, 2.5, axis=0)
    band95_hi = np.percentile(curve_samps, 97.5, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    for x in sorted(df_plot["x"].unique()):
        dfx = df_plot[df_plot["x"] == x].sort_values("N")
        invN = 1.0 / dfx["N"].to_numpy()
        y = dfx["mg"].to_numpy()
        ax1.plot(invN, y, "-", alpha=0.25)
        # Draw DMRG first and ED crosses last to highlight overlapping points.
        for src, marker in [("DMRG", "s"), ("ED", "x")]:
            sub = dfx[dfx["source"] == src]
            if len(sub):
                kwargs = {"ms": 6, "label": f"x={int(x)} ({src})"}
                if src == "DMRG":
                    kwargs.update({"mfc": "none", "zorder": 2})
                else:
                    kwargs.update({"mew": 1.6, "zorder": 4})
                ax1.plot(1.0 / sub["N"], sub["mg"], marker, **kwargs)

    ax1.axhline(TARGET, color="black", ls="--", lw=1.2, label=r"Exact $1/\sqrt{\pi}$")
    ax1.set_xlabel(r"$1/N$")
    ax1.set_ylabel(r"$M_{\rm gap}/g$")
    ax1.set_title("Finite-size convergence")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    for N in sorted(df_plot["N"].unique()):
        dfN = df_plot[df_plot["N"] == N].sort_values("ag2")
        # classify curve source for labeling (usually all ED or all DMRG)
        sources = set(dfN["source"].astype(str).unique())
        src = list(sources)[0] if len(sources) == 1 else "mix"
        ax2.plot(
            dfN["ag2"], dfN["mg"],
            "o-", alpha=0.25, lw=1,
            label=f"N={N} ({src})"
        )

    ax2.fill_between(v_grid, band95_lo, band95_hi, alpha=0.15, label="Bootstrap 95% band (u=0)")
    ax2.fill_between(v_grid, band68_lo, band68_hi, alpha=0.25, label="Bootstrap 68% band (u=0)")
    ax2.plot(v_grid, curve_hat, lw=2, label=f"Joint fit (u=0): M(0,0)={M00_mean:.4f}")
    ax2.axhline(TARGET, color="black", ls="--", lw=1.2, label=r"Exact $1/\sqrt{\pi}$")

    ax2.set_xlabel(r"$(ag)^2 = 1/x$")
    ax2.set_ylabel(r"$M_{\rm gap}/g$")
    ax2.set_title("Continuum extrapolation")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle(r"Schwinger model mass gap: joint fit in $1/N$ and $(ag)^2$", fontsize=13, y=1.02)
    plt.tight_layout()
    _Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=250, bbox_inches="tight")
    print(f"Saved {args.out_png} and {args.report_csv}")


if __name__ == "__main__":
    main()
