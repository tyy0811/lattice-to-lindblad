#!/usr/bin/env python3
"""vqe_runner.py

Modular benchmark runner for:
  • Ideal statevector VQE (parameter search)
  • Aer noisy simulation with explicit noise model (depolarizing + readout error)
  • Quantum Inspire hardware backends (optional)
  • Error mitigation: MEM, optional ZNE (Aer-only via noise scaling)
  • Error analysis: Aer ablation (gate-only vs readout-only vs both)

Project layout (what you asked for)
-----------------------------------
models/schwinger.py   models/tfim.py   models/xxz.py   models/npy.py
backends/aer_backend.py  backends/qi_backend.py
mitigation/mem.py     mitigation/zne.py
core/*                analysis/*       plotting/*

Docs references:
  • Aer noise model / ReadoutError: Qiskit Aer docs
  • Readout mitigation via assignment matrix: Qiskit Experiments docs
  • ZNE concept: Mitiq docs (curve fit + extrapolate)

"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eigvalsh

from models.schwinger import build_schwinger_full
from models.tfim import build_tfim
from models.xxz import build_heisenberg_xxz
from models.npy import load_hamiltonian_npy

from core.hamiltonian import pauli_decompose
from core.ansatz import build_ansatz_ry_cx_rz
from core.ideal_vqe import run_ideal_vqe
from core.measurement import (
    group_terms_by_basis,
    precompute_parity_vectors,
    evaluate_energy_shots,
    bootstrap_energy_se,
)

from mitigation.mem import build_assignment_matrix, make_apply_mem
from mitigation.zne import zne_extrapolate

from backends.aer_backend import build_aer_backend
from backends.qi_backend import get_qi_backend, list_qi_backends

from plotting.summary import make_summary_figure


def build_problem(args) -> Tuple[str, int, np.ndarray, float]:
    if args.model == "schwinger":
        H = build_schwinger_full(args.N, args.x, args.m_over_g)
        name = f"Schwinger(N={args.N}, x={args.x:g}, m/g={args.m_over_g:g})"
        N = args.N
    elif args.model == "tfim":
        H = build_tfim(args.N, J=args.J, h=args.h, pbc=args.pbc)
        name = f"TFIM(N={args.N}, J={args.J:g}, h={args.h:g}, pbc={args.pbc})"
        N = args.N
    elif args.model == "xxz":
        H = build_heisenberg_xxz(args.N, Jxy=args.Jxy, Jz=args.Jz, pbc=args.pbc)
        name = f"XXZ(N={args.N}, Jxy={args.Jxy:g}, Jz={args.Jz:g}, pbc={args.pbc})"
        N = args.N
    elif args.model == "npy":
        if not args.ham_npy:
            raise ValueError("--ham_npy PATH is required for --model npy")
        H = load_hamiltonian_npy(args.ham_npy)
        dim = H.shape[0]
        N = int(round(math.log2(dim)))
        if 2**N != dim:
            raise ValueError(f"Hamiltonian dimension {dim} is not 2^N")
        name = f"CustomNpy(N={N})"
    else:
        raise ValueError("Unknown model")
    E_ed = float(eigvalsh(H)[0])
    return name, N, H, E_ed


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--backend", choices=["aer", "qi", "both"], default="aer")

    # Model selection
    ap.add_argument("--model", choices=["schwinger", "tfim", "xxz", "npy"], default="schwinger")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--x", type=float, default=4.0)          # Schwinger
    ap.add_argument("--m_over_g", type=float, default=0.0)   # Schwinger
    ap.add_argument("--J", type=float, default=1.0)          # TFIM
    ap.add_argument("--h", type=float, default=1.0)          # TFIM
    ap.add_argument("--Jxy", type=float, default=1.0)        # XXZ
    ap.add_argument("--Jz", type=float, default=1.0)         # XXZ
    ap.add_argument("--pbc", action="store_true")
    ap.add_argument("--ham_npy", type=str, default="")

    # Ansatz / ideal VQE
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--no_neel", action="store_true")
    ap.add_argument("--ideal_restarts", type=int, default=5)
    ap.add_argument("--ideal_maxiter", type=int, default=800)

    # Measurement / decomposition
    ap.add_argument("--pauli_tol", type=float, default=1e-10)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--shots_cal", type=int, default=2048)
    ap.add_argument("--transpile_level", type=int, default=1)

    # Aer noise model params
    ap.add_argument("--p1q", type=float, default=1e-3)
    ap.add_argument("--p2q", type=float, default=1e-2)
    ap.add_argument("--p01", type=float, default=2e-2)
    ap.add_argument("--p10", type=float, default=2e-2)

    # QI
    ap.add_argument("--qi_backend", type=str, default="Tuna-5")
    ap.add_argument("--list_qi_backends", action="store_true")

    # Mitigation
    ap.add_argument("--do_mem", action="store_true")
    ap.add_argument("--do_zne", action="store_true")
    ap.add_argument("--zne_scales", type=str, default="1.0,1.5,2.0,2.5,3.0")

    # Analysis / outputs
    ap.add_argument("--error_analysis", action="store_true")
    ap.add_argument("--save_json", type=str, default="")
    ap.add_argument("--save_plot", action="store_true")

    args = ap.parse_args()

    if args.list_qi_backends:
        for b in list_qi_backends():
            print(b)
        return

    model_name, N, H, E_ed = build_problem(args)

    # Build Pauli decomposition + measurement grouping
    pauli_op = pauli_decompose(H, N, atol=args.pauli_tol)
    groups = group_terms_by_basis(pauli_op, N)
    parity_vec = precompute_parity_vectors(N)

    ansatz, params = build_ansatz_ry_cx_rz(N, args.layers, neel_init=not args.no_neel)
    n_cx = ansatz.count_ops().get("cx", 0)

    print("=" * 78)
    print(f"VQE benchmark | {model_name} | layers={args.layers} | backend={args.backend}")
    print(f"ED: {E_ed:.10f}")
    print(f"Pauli terms kept: {len(pauli_op)} | basis groups: {len(groups)}")
    print(f"Circuit params={len(params)}, CNOTs={n_cx}, depth~{ansatz.depth()}")
    print(f"Shots: energy={args.shots}, cal={args.shots_cal}")
    if args.backend in ("qi", "both"):
        print(f"QI backend: {args.qi_backend}")
    print("=" * 78)

    print(f"\n--- Ideal VQE (statevector, {args.ideal_restarts} restarts) ---")
    E_ideal, p_ideal = run_ideal_vqe(
        ansatz, params, H, restarts=args.ideal_restarts, maxiter=args.ideal_maxiter, seed=42
    )
    print(f"  Best ideal: {E_ideal:.10f} |dE|={abs(E_ideal - E_ed):.2e}")

    energies: Dict[str, float] = {"E_ed": float(E_ed), "E_ideal": float(E_ideal)}
    stderr: Dict[str, float] = {}

    # ------------- Aer -------------
    E_aer_raw: Optional[float] = None
    E_aer_mit: Optional[float] = None

    if args.backend in ("aer", "both"):
        print("\n--- Aer (depolarizing + readout) ---")
        aer = build_aer_backend(args.p1q, args.p2q, args.p01, args.p10)

        apply_mem = None
        if args.do_mem:
            print("  MEM calibration on Aer...")
            _, A_inv = build_assignment_matrix(aer, N, args.shots_cal, args.transpile_level)
            apply_mem = make_apply_mem(A_inv)

        # raw
        E_aer_raw, counts_ar, basis_ar = evaluate_energy_shots(
            aer, ansatz, params, p_ideal, groups, parity_vec, args.shots, None, args.transpile_level
        )
        print(f"  Aer raw:    {E_aer_raw:.10f}")
        stderr["E_aer_raw"] = bootstrap_energy_se(groups, parity_vec, basis_ar, counts_ar, N, args.shots, None)

        # mitigated
        if args.do_zne:
            scales = [float(s.strip()) for s in args.zne_scales.split(",") if s.strip()]
            zvals = []
            print("  ZNE on Aer (scaling depolarizing; readout fixed)")
            for lam in scales:
                aer_l = build_aer_backend(args.p1q * lam, args.p2q * lam, args.p01, args.p10)
                apply_mem_l = None
                if args.do_mem:
                    _, A_inv_l = build_assignment_matrix(aer_l, N, args.shots_cal, args.transpile_level)
                    apply_mem_l = make_apply_mem(A_inv_l)

                E_l, _, _ = evaluate_energy_shots(
                    aer_l, ansatz, params, p_ideal, groups, parity_vec, args.shots, apply_mem_l, args.transpile_level
                )
                zvals.append(E_l)
                print(f"    scale={lam:g}: E={E_l:.10f}")

            E_aer_mit = zne_extrapolate(scales, zvals, degree=2)
            print(f"  Aer ZNE->0: {E_aer_mit:.10f}")
            stderr["E_aer_mit"] = stderr.get("E_aer_raw", 0.0)
        elif args.do_mem:
            E_aer_mit, counts_am, basis_am = evaluate_energy_shots(
                aer, ansatz, params, p_ideal, groups, parity_vec, args.shots, apply_mem, args.transpile_level
            )
            print(f"  Aer + MEM:  {E_aer_mit:.10f}")
            stderr["E_aer_mit"] = bootstrap_energy_se(groups, parity_vec, basis_am, counts_am, N, args.shots, apply_mem)

        energies["E_aer_raw"] = float(E_aer_raw)
        if E_aer_mit is not None:
            energies["E_aer_mit"] = float(E_aer_mit)

        if args.error_analysis:
            print("\n--- Aer error-source ablation (using ideal params) ---")
            aer_gate = build_aer_backend(args.p1q, args.p2q, 0.0, 0.0)
            Eg, _, _ = evaluate_energy_shots(
                aer_gate, ansatz, params, p_ideal, groups, parity_vec, args.shots, None, args.transpile_level
            )

            aer_ro = build_aer_backend(0.0, 0.0, args.p01, args.p10)
            Er, _, _ = evaluate_energy_shots(
                aer_ro, ansatz, params, p_ideal, groups, parity_vec, args.shots, None, args.transpile_level
            )

            print(f"  Ideal:        {E_ideal:.10f}")
            print(f"  Gate-only:    {Eg:.10f}  Δ={Eg - E_ideal:+.3f}")
            print(f"  Readout-only: {Er:.10f}  Δ={Er - E_ideal:+.3f}")
            print(f"  Both:         {E_aer_raw:.10f}  Δ={E_aer_raw - E_ideal:+.3f}")
            energies["E_aer_gate_only"] = float(Eg)
            energies["E_aer_readout_only"] = float(Er)

    # ------------- QI -------------
    E_qi_raw: Optional[float] = None
    E_qi_mem: Optional[float] = None

    if args.backend in ("qi", "both"):
        print("\n--- Quantum Inspire ---")
        qi = get_qi_backend(args.qi_backend)

        apply_mem_qi = None
        if args.do_mem:
            print("  MEM calibration on QI...")
            _, A_inv_qi = build_assignment_matrix(qi, N, args.shots_cal, args.transpile_level)
            apply_mem_qi = make_apply_mem(A_inv_qi)

        E_qi_raw, counts_qr, basis_qr = evaluate_energy_shots(
            qi, ansatz, params, p_ideal, groups, parity_vec, args.shots, None, args.transpile_level
        )
        print(f"  QI raw:    {E_qi_raw:.10f}")
        stderr["E_qi_raw"] = bootstrap_energy_se(groups, parity_vec, basis_qr, counts_qr, N, args.shots, None)

        if args.do_mem:
            E_qi_mem, counts_qm, basis_qm = evaluate_energy_shots(
                qi, ansatz, params, p_ideal, groups, parity_vec, args.shots, apply_mem_qi, args.transpile_level
            )
            print(f"  QI + MEM: {E_qi_mem:.10f}")
            stderr["E_qi_mem"] = bootstrap_energy_se(groups, parity_vec, basis_qm, counts_qm, N, args.shots, apply_mem_qi)

        energies["E_qi_raw"] = float(E_qi_raw)
        if E_qi_mem is not None:
            energies["E_qi_mem"] = float(E_qi_mem)

    # ------------- Summary table -------------
    print("\n" + "=" * 78)
    print(f"{'Method':<18} {'Energy':>14} {'|dE|':>12} {'Error %':>9}")
    print("-" * 78)

    def pr(name: str, E: Optional[float]) -> None:
        if E is None:
            return
        err = abs(E - E_ed)
        pct = 100 * err / abs(E_ed) if E_ed != 0 else 0.0
        print(f"{name:<18} {E:14.10f} {err:12.2e} {pct:8.3f}%")

    pr("ED (exact)", E_ed)
    pr("Ideal VQE", E_ideal)
    pr("Aer noisy", E_aer_raw)
    pr("Aer mitigated", E_aer_mit)
    pr("QI raw", E_qi_raw)
    pr("QI + MEM", E_qi_mem)
    print("=" * 78)

    # ------------- Save JSON -------------
    if args.save_json:
        payload = {
            "model": args.model,
            "model_name": model_name,
            "N": N,
            "layers": args.layers,
            "ansatz": "RY-CX-RZ",
            "backend_mode": args.backend,
            "qi_backend": args.qi_backend if args.backend in ("qi", "both") else "",
            "shots": args.shots,
            "shots_cal": args.shots_cal,
            "aer_noise": {"p1q": args.p1q, "p2q": args.p2q, "p01": args.p01, "p10": args.p10},
            "mitigation": {"mem": bool(args.do_mem), "zne": bool(args.do_zne), "zne_scales": args.zne_scales},
            "energies": energies,
            "stderr_bootstrap": stderr,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {args.save_json}")

    if args.save_plot:
        out = "summary_vqe_gap.png"
        title = f"ED vs Ideal vs Aer vs QI ({model_name})"
        make_summary_figure(
            out,
            title,
            E_ed,
            E_ideal,
            E_aer_raw,
            E_aer_mit,
            E_qi_raw,
            E_qi_mem,
            stderr.get("E_aer_raw"),
            stderr.get("E_aer_mit"),
            stderr.get("E_qi_raw"),
            stderr.get("E_qi_mem"),
        )
        print(f"Saved plot: {out}")

if __name__ == "__main__":
    main()
