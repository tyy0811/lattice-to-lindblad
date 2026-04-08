#!/usr/bin/env python3
"""noisy_vqe_zne.py

Quantum Inspire (QI) hardware-faithful VQE *evaluation* for the N=4 Schwinger model.

What this script demonstrates (reviewer-facing):
  • Ideal VQE using a statevector (fast, local) to find good parameters.
  • Hardware-like energy estimation using *shots* and *counts* by measuring
    Pauli terms of the Hamiltonian (no density-matrix trace shortcuts).
  • Readout error mitigation (MEM) via a full assignment-matrix calibration.
  • Runs on Quantum Inspire backends via the Qiskit provider (Qiskit-QuantumInspire).

It is intentionally **evaluation-only** on hardware by default (running a full
shot-based VQE loop on real hardware is very expensive in circuit count).

Setup (once)
------------
Install the provider:
  pip install qiskit-quantuminspire

Login using the QI tool (recommended by QI docs):
  qi login

The QI tool stores credentials in ~/.quantuminspire/config.json, which the
provider reads automatically.

Run
---
  # Run on the QI simulator backend (cheap smoke test):
  python noisy_vqe_zne.py --qi_backend "QX emulator"

  # List backends, then run on a hardware backend:
  python noisy_vqe_zne.py --list_backends
  python noisy_vqe_zne.py --qi_backend "Starmon-7" --shots 2000 --shots_cal 4000

Outputs
-------
  • prints a summary table (ED, ideal VQE, QI raw, QI+MEM)
  • optionally saves a small plot and JSON file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import eigvalsh
    from scipy.optimize import minimize
except Exception:
    print("SciPy import failed. Install with:")
    print(f"  {sys.executable} -m pip install scipy")
    raise

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
except Exception:
    print("Qiskit import failed. Install with:")
    print(f"  {sys.executable} -m pip install qiskit")
    raise


# ========================== Quantum Inspire backend ==========================

def get_qi_provider():
    """Import and instantiate QIProvider (Qiskit-QuantumInspire)."""
    try:
        from qiskit_quantuminspire.qi_provider import QIProvider
    except Exception as e:
        raise RuntimeError(
            "Quantum Inspire provider not found. Install with:\n"
            "  pip install qiskit-quantuminspire\n"
            "Then login once using:\n"
            "  qi login\n"
        ) from e
    return QIProvider()


def _job_id_safe(job) -> str:
    try:
        jid = job.job_id()
        return str(jid) if jid else "<unknown>"
    except Exception:
        return "<unknown>"


def _job_result(job, label: str, timeout_s: Optional[float]):
    jid = _job_id_safe(job)
    print(f"  submitted {label} job id: {jid}")
    # qiskit-quantuminspire defaults to timeout=60s if not provided.
    # Map "no timeout" requests to a very large explicit timeout.
    effective_timeout = 86400.0 if timeout_s is None else float(timeout_s)
    try:
        return job.result(timeout=effective_timeout)
    except Exception as e:
        msg = str(e)
        is_timeout = "Timeout while waiting for job" in msg or e.__class__.__name__ == "JobTimeoutError"
        if is_timeout:
            tmsg = "24h (no-timeout mode)" if timeout_s is None else f"{timeout_s:g}s"
            raise RuntimeError(
                f"{label} job timed out after {tmsg} (job id: {jid}). "
                "Increase --job_timeout, reduce --shots/--shots_cal, or pick a less busy backend."
            ) from e
        raise


# ========================== Schwinger Hamiltonian ============================

def build_schwinger_full(
    N: int,
    x: float,
    m_over_g: float = 0.0,
    E0: float = 0.0,
    x_def: str = "tagliacozzo",
) -> np.ndarray:
    """Full 2^N x 2^N Schwinger Hamiltonian (Tagliacozzo mapping, no projection)."""
    ga = 1.0 / math.sqrt(x) if x_def == "tagliacozzo" else 1.0 / math.sqrt(2 * x)
    mu = 2.0 * m_over_g / ga
    stag = np.array([1 if n % 2 == 0 else -1 for n in range(N)], dtype=float)

    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)

    for s in range(dim):
        L = E0
        diag_e, diag_m = 0.0, 0.0
        for n in range(N):
            bit = (s >> n) & 1
            z = 1 - 2 * bit
            diag_m += 0.5 * mu * stag[n] * z
            qn = 0.5 * (z + stag[n])
            L += qn
            if n <= N - 2:
                diag_e += L * L
        H[s, s] = diag_e + diag_m

        for n in range(N - 1):
            bn = (s >> n) & 1
            bn1 = (s >> (n + 1)) & 1
            if bn != bn1:
                s2 = s ^ (1 << n) ^ (1 << (n + 1))
                H[s, s2] += x

    return 0.5 * (H + H.T)


def pauli_decompose(H: np.ndarray, n_qubits: int, atol: float = 1e-10) -> SparsePauliOp:
    op = SparsePauliOp.from_operator(Operator(H))
    mask = np.abs(op.coeffs) > atol
    op = SparsePauliOp(op.paulis[mask], op.coeffs[mask])
    op = SparsePauliOp(op.paulis, np.real(op.coeffs)).simplify()
    return op


# ========================== Ansatz ==========================================

def build_ansatz(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Neel init + layers of [RY(all) -> CX chain -> RZ(all)]."""
    qc = QuantumCircuit(n_qubits)
    for i in range(1, n_qubits, 2):
        qc.x(i)

    params: List[Parameter] = []
    for l in range(n_layers):
        for i in range(n_qubits):
            p = Parameter(f"ry{l}_{i}")
            params.append(p)
            qc.ry(p, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            p = Parameter(f"rz{l}_{i}")
            params.append(p)
            qc.rz(p, i)
    return qc, params


def energy_statevector(pvals: np.ndarray, ansatz: QuantumCircuit, params: List[Parameter], H: np.ndarray) -> float:
    bind = {params[i]: float(pvals[i]) for i in range(len(params))}
    qc = ansatz.assign_parameters(bind, inplace=False)
    psi = Statevector.from_instruction(qc).data
    return float(np.real(psi.conj().T @ (H @ psi)))


# ========================== Pauli measurement grouping =======================

def _label_to_ops_q0_first(label: str, n_qubits: int) -> List[str]:
    """Qiskit Pauli labels are big-endian; return ops in qubit order q0..qN-1."""
    return [label[n_qubits - 1 - i] for i in range(n_qubits)]


@dataclass
class Term:
    coeff: float
    mask: int


def group_terms_by_basis(pauli_op: SparsePauliOp, n_qubits: int) -> Dict[Tuple[str, ...], List[Term]]:
    """Group Hamiltonian terms by per-qubit measurement basis."""
    groups: Dict[Tuple[str, ...], List[Term]] = {}
    for p, c in zip(pauli_op.paulis, pauli_op.coeffs):
        coeff = float(np.real(c))
        ops = _label_to_ops_q0_first(p.to_label(), n_qubits)

        basis: List[str] = []
        mask = 0
        for q, op in enumerate(ops):
            if op == "X":
                basis.append("X")
                mask |= (1 << q)
            elif op == "Y":
                basis.append("Y")
                mask |= (1 << q)
            elif op == "Z":
                basis.append("Z")
                mask |= (1 << q)
            else:  # I
                basis.append("Z")

        b = tuple(basis)
        groups.setdefault(b, []).append(Term(coeff=coeff, mask=mask))
    return groups


def precompute_parity_vectors(n_qubits: int) -> Dict[int, np.ndarray]:
    dim = 2**n_qubits
    out: Dict[int, np.ndarray] = {}
    for mask in range(2**n_qubits):
        v = np.empty(dim, dtype=float)
        for s in range(dim):
            v[s] = -1.0 if ((s & mask).bit_count() % 2) else 1.0
        out[mask] = v
    return out


def build_measurement_circuit(ansatz: QuantumCircuit, basis: Tuple[str, ...]) -> QuantumCircuit:
    n = ansatz.num_qubits
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q, c)
    qc.compose(ansatz, qubits=q, inplace=True)

    for i, b in enumerate(basis):
        if b == "X":
            qc.h(q[i])
        elif b == "Y":
            qc.sdg(q[i])
            qc.h(q[i])

    for i in range(n):
        qc.measure(q[i], c[i])
    return qc


# ========================== Counts utilities + MEM ===========================

def counts_to_probvec(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    p = np.zeros(dim, dtype=float)
    shots = sum(counts.values()) if counts else 0
    if shots == 0:
        return p
    for bitstr, c in counts.items():
        bitstr = bitstr.replace(" ", "")
        idx = int(bitstr, 2)
        p[idx] += c / shots
    return p


def build_calibration_circuits(n_qubits: int) -> List[QuantumCircuit]:
    circs: List[QuantumCircuit] = []
    for s in range(2**n_qubits):
        q = QuantumRegister(n_qubits, "q")
        c = ClassicalRegister(n_qubits, "c")
        qc = QuantumCircuit(q, c)
        for i in range(n_qubits):
            if (s >> i) & 1:
                qc.x(q[i])
        for i in range(n_qubits):
            qc.measure(q[i], c[i])
        qc.name = f"cal_{s:0{n_qubits}b}"
        circs.append(qc)
    return circs


def build_assignment_matrix(
    backend,
    n_qubits: int,
    shots_cal: int,
    transpile_level: int = 1,
    job_timeout_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cal = build_calibration_circuits(n_qubits)
    cal_t = transpile(cal, backend, optimization_level=transpile_level)
    job = backend.run(cal_t, shots=shots_cal)
    res = _job_result(job, label="MEM calibration", timeout_s=job_timeout_s)

    dim = 2**n_qubits
    A = np.zeros((dim, dim), dtype=float)
    for true_state in range(dim):
        counts = res.get_counts(true_state)
        A[:, true_state] = counts_to_probvec(counts, n_qubits)
    A_inv = np.linalg.pinv(A)
    return A, A_inv


def apply_mitigator(p_meas: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
    p = A_inv @ p_meas
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s > 1e-12:
        p /= s
    return p


# ========================== Shot-based energy evaluation =====================

def evaluate_energy_shots(
    backend,
    ansatz: QuantumCircuit,
    params: List[Parameter],
    pvals: np.ndarray,
    groups: Dict[Tuple[str, ...], List[Term]],
    parity_vec: Dict[int, np.ndarray],
    shots: int,
    A_inv: np.ndarray | None,
    transpile_level: int = 1,
    job_timeout_s: Optional[float] = None,
) -> float:
    bind = {params[i]: float(pvals[i]) for i in range(len(params))}

    basis_list = list(groups.keys())
    circuits: List[QuantumCircuit] = []
    for b in basis_list:
        qc = build_measurement_circuit(ansatz, b)
        qc = qc.assign_parameters(bind, inplace=False)
        qc.name = "meas_" + "".join(b)
        circuits.append(qc)

    circuits_t = transpile(circuits, backend, optimization_level=transpile_level)
    job = backend.run(circuits_t, shots=shots)
    res = _job_result(job, label="energy", timeout_s=job_timeout_s)

    E = 0.0
    n_qubits = ansatz.num_qubits
    for k, b in enumerate(basis_list):
        counts = res.get_counts(k)
        p = counts_to_probvec(counts, n_qubits)
        if A_inv is not None:
            p = apply_mitigator(p, A_inv)
        for term in groups[b]:
            E += term.coeff * float(parity_vec[term.mask] @ p)
    return float(E)


# ========================== Optimizer (ideal only) ===========================

def run_ideal_vqe(
    ansatz: QuantumCircuit,
    params: List[Parameter],
    H: np.ndarray,
    restarts: int,
    maxiter: int,
    seed: int,
) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_E = float("inf")
    best_p: np.ndarray | None = None

    def cost(pv):
        return energy_statevector(pv, ansatz, params, H)

    for t in range(restarts):
        x0 = rng.uniform(-0.1, 0.1, len(params))
        t0 = time.time()
        res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        dt = time.time() - t0
        E = float(res.fun)
        print(f"  trial {t+1}: E={E:.10f}  ({int(res.nfev)} evals, {dt:.1f}s)")
        if E < best_E:
            best_E = E
            best_p = np.array(res.x, dtype=float)
    assert best_p is not None
    return best_E, best_p


# ========================== CLI / Main ======================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qi_backend", type=str, default="QX emulator",
                    help="Quantum Inspire backend name (e.g., 'QX emulator', 'Starmon-7').")
    ap.add_argument("--list_backends", action="store_true", help="List available QI backends and exit.")
    ap.add_argument("--shots", type=int, default=4000, help="Shots per Hamiltonian measurement circuit.")
    ap.add_argument("--shots_cal", type=int, default=8192, help="Shots per MEM calibration circuit.")
    ap.add_argument("--layers", type=int, default=4, help="Ansatz layers.")
    ap.add_argument("--ideal_restarts", type=int, default=5)
    ap.add_argument("--ideal_maxiter", type=int, default=800)
    ap.add_argument("--pauli_tol", type=float, default=1e-10)
    ap.add_argument("--transpile_level", type=int, default=1)
    ap.add_argument("--job_timeout", type=float, default=1800.0,
                    help="Seconds to wait for each backend job. Use <=0 for no timeout.")
    ap.add_argument("--save_plot", action="store_true")
    ap.add_argument("--save_json", type=str, default="", help="Path to save JSON results.")
    args = ap.parse_args()
    job_timeout_s = None if args.job_timeout <= 0 else args.job_timeout

    provider = get_qi_provider()
    if args.list_backends:
        print("Available Quantum Inspire backends:")
        for b in provider.backends():
            print(" -", b.name)
        return

    backend = provider.get_backend(args.qi_backend)

    # ----- Problem definition -----
    N = 4
    x = 4.0
    m_over_g = 0.0

    print("=" * 70)
    print(f"Quantum Inspire VQE evaluation + MEM | N={N}, x={x:g}, m/g={m_over_g:g}, layers={args.layers}")
    print(f"QI backend: {args.qi_backend}")
    print(f"Shots: energy={args.shots}, cal={args.shots_cal}")
    print("=" * 70)

    H = build_schwinger_full(N, x, m_over_g)
    E_ed = float(eigvalsh(H)[0])
    print(f"\nED ground-state energy: {E_ed:.10f}")

    ansatz, params = build_ansatz(N, args.layers)
    n_cx = ansatz.count_ops().get("cx", 0)
    print(f"Ansatz: {len(params)} params, {n_cx} CNOTs, depth ~{ansatz.depth()}")

    pauli_op = pauli_decompose(H, N, atol=args.pauli_tol)
    groups = group_terms_by_basis(pauli_op, N)
    parity_vec = precompute_parity_vectors(N)
    print(f"Pauli terms kept: {len(pauli_op)}")
    print(f"Measurement basis groups: {len(groups)}")

    print(f"\n--- Ideal VQE (statevector, {args.ideal_restarts} restarts) ---")
    E_ideal, p_ideal = run_ideal_vqe(
        ansatz=ansatz,
        params=params,
        H=H,
        restarts=args.ideal_restarts,
        maxiter=args.ideal_maxiter,
        seed=42,
    )
    print(f"  Best: {E_ideal:.10f}  |dE|={abs(E_ideal - E_ed):.2e}")

    print("\n--- Readout mitigation calibration (MEM) ---")
    t0 = time.time()
    A, A_inv = build_assignment_matrix(
        backend=backend,
        n_qubits=N,
        shots_cal=args.shots_cal,
        transpile_level=args.transpile_level,
        job_timeout_s=job_timeout_s,
    )
    dt = time.time() - t0
    cond = np.linalg.cond(A) if np.linalg.matrix_rank(A) == A.shape[0] else float("inf")
    print(f"  Cal complete in {dt:.1f}s | cond(A)~{cond:.2e}")

    print("\n--- QI energy evaluation (shots) ---")
    t0 = time.time()
    E_raw = evaluate_energy_shots(
        backend=backend,
        ansatz=ansatz,
        params=params,
        pvals=p_ideal,
        groups=groups,
        parity_vec=parity_vec,
        shots=args.shots,
        A_inv=None,
        transpile_level=args.transpile_level,
        job_timeout_s=job_timeout_s,
    )
    dt_raw = time.time() - t0
    print(f"  QI raw:     E={E_raw:.10f}  (elapsed {dt_raw:.1f}s)")

    t0 = time.time()
    E_mem = evaluate_energy_shots(
        backend=backend,
        ansatz=ansatz,
        params=params,
        pvals=p_ideal,
        groups=groups,
        parity_vec=parity_vec,
        shots=args.shots,
        A_inv=A_inv,
        transpile_level=args.transpile_level,
        job_timeout_s=job_timeout_s,
    )
    dt_mem = time.time() - t0
    print(f"  QI + MEM:   E={E_mem:.10f}  (elapsed {dt_mem:.1f}s)")

    print("\n" + "=" * 70)
    print(f"{'Method':<16} {'Energy':>14} {'|dE|':>12} {'Error %':>9}")
    print("-" * 70)
    rows = [
        ("ED (exact)", E_ed),
        ("Ideal VQE", E_ideal),
        ("QI raw", E_raw),
        ("QI + MEM", E_mem),
    ]
    for name, E in rows:
        err = abs(E - E_ed)
        pct = 100 * err / abs(E_ed) if E_ed != 0 else 0.0
        print(f"{name:<16} {E:14.10f} {err:12.2e} {pct:8.3f}%")
    print("=" * 70)

    if args.save_json:
        payload = {
            "N": N,
            "x": x,
            "m_over_g": m_over_g,
            "layers": args.layers,
            "qi_backend": args.qi_backend,
            "shots": args.shots,
            "shots_cal": args.shots_cal,
            "E_ed": E_ed,
            "E_ideal": E_ideal,
            "E_qi_raw": E_raw,
            "E_qi_mem": E_mem,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {args.save_json}")

    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = ["ED", "Ideal", "QI raw", "QI+MEM"]
            vals = [E_ed, E_ideal, E_raw, E_mem]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, vals)
            ax.set_ylabel("Energy")
            ax.set_title(f"N={N} Schwinger VQE on Quantum Inspire ({args.qi_backend})")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out = "qi_vqe_mem.png"
            fig.savefig(out, dpi=200)
            print(f"Saved plot: {out}")
        except Exception as e:
            print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
