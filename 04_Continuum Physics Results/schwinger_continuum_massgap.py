#!/usr/bin/env python3
"""schwinger_continuum_massgap.py (refactored wrapper)

Drop-in replacement wrapper for producing the ED mass-gap grid CSV used by the
Schwinger continuum/thermodynamic extrapolation pipeline.

Key points:
- Keeps the original CSV schema:
    source,x,N,ag2,mg,E0,E1,gap,dim
- Re-exports the ED helper API used for validation elsewhere:
    SchwingerParams, build_hamiltonian_sector, lowest_energies, lowest_energies_sector
- Internals are now reusable via the l2l/ library + a model adapter.
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
import pandas as pd

from l2l import schwinger_ed
from l2l.grid_engines import EDRunSpec, run_ed_grid
from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter

# Re-export ED helpers for compatibility
SchwingerParams = schwinger_ed.SchwingerParams
build_hamiltonian_sector = schwinger_ed.build_hamiltonian_sector
lowest_energies = schwinger_ed.lowest_energies
lowest_energies_sector = schwinger_ed.lowest_energies_sector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", nargs="+", type=float, default=[4, 8, 12], help="x values (x=1/(ag)^2)")
    ap.add_argument("--N", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20], help="Even system sizes")
    ap.add_argument("--m_over_g", type=float, default=0.0)
    ap.add_argument("--E0", type=float, default=0.0)
    ap.add_argument("--k_eigs", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=2_000_000)
    ap.add_argument("--out_grid_csv", default="ed_massgap_grid.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for N in args.N:
        if int(N) % 2 != 0:
            raise SystemExit("All ED N must be even")

    adapter = SchwingerMassGapAdapter(m_over_g=float(args.m_over_g), E0=float(args.E0))
    df = run_ed_grid(
        adapter,
        N_list=[int(v) for v in args.N],
        params_list=[{"x": float(x)} for x in args.x],
        spec=EDRunSpec(k_eigs=int(args.k_eigs), tol=float(args.tol), maxiter=int(args.maxiter)),
        quiet=bool(args.quiet),
    )

    # Preserve original schema/order
    cols = ["source", "x", "N", "ag2", "mg", "E0", "E1", "gap", "dim"]
    out = df.copy()
    out["source"] = "ED"
    out = out[cols]
    out.to_csv(args.out_grid_csv, index=False)

    print(f"Saved {len(out)} rows to {args.out_grid_csv}")


if __name__ == "__main__":
    main()