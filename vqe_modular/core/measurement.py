from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

@dataclass(frozen=True)
class Term:
    coeff: float
    mask: int

def _label_to_ops_q0_first(label: str, n_qubits: int) -> List[str]:
    # Qiskit Pauli labels are big-endian. Convert to [q0..qN-1] order.
    return [label[n_qubits - 1 - i] for i in range(n_qubits)]

def group_terms_by_basis(pauli_op: SparsePauliOp, n_qubits: int) -> Dict[Tuple[str, ...], List[Term]]:
    """Group Pauli terms by required measurement basis (X/Y/Z/I per qubit)."""
    groups: Dict[Tuple[str, ...], List[Term]] = {}
    for pauli, coeff in zip(pauli_op.paulis, pauli_op.coeffs):
        label = pauli.to_label()
        ops = _label_to_ops_q0_first(label, n_qubits)

        basis: List[str] = []
        mask = 0
        for i, op in enumerate(ops):
            if op == "I":
                basis.append("I")
                continue
            basis.append(op)
            mask |= 1 << i

        groups.setdefault(tuple(basis), []).append(Term(float(np.real(coeff)), mask))
    return groups

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
        # Z or I => measure Z

    for i in range(n):
        qc.measure(q[i], c[i])
    return qc

def counts_to_probvec(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    p = np.zeros(dim, dtype=float)
    shots = sum(counts.values()) if counts else 0
    if shots == 0:
        return p
    for bitstr, c in counts.items():
        bitstr = bitstr.replace(" ", "")
        p[int(bitstr, 2)] += c / shots
    return p

def precompute_parity_vectors(n_qubits: int) -> Dict[int, np.ndarray]:
    """Return v_mask[s] = (-1)^{popcount(s & mask)} for all masks."""
    dim = 2**n_qubits
    out: Dict[int, np.ndarray] = {}
    for mask in range(2**n_qubits):
        v = np.empty(dim, dtype=float)
        for s in range(dim):
            v[s] = -1.0 if ((s & mask).bit_count() % 2) else 1.0
        out[mask] = v
    return out

def run_measurement_batch(backend, circuits: List[QuantumCircuit], shots: int, transpile_level: int) -> List[Dict[str, int]]:
    circuits_t = transpile(circuits, backend, optimization_level=transpile_level)
    job = backend.run(circuits_t, shots=shots)
    res = job.result()
    return [res.get_counts(k) for k in range(len(circuits))]

def evaluate_energy_shots(
    backend,
    ansatz: QuantumCircuit,
    params: List[Parameter],
    pvals: np.ndarray,
    groups: Dict[Tuple[str, ...], List[Term]],
    parity_vec: Dict[int, np.ndarray],
    shots: int,
    mitigator_apply_fn,
    transpile_level: int,
) -> Tuple[float, List[Dict[str, int]], List[Tuple[str, ...]]]:
    """Estimate energy by measuring grouped Pauli terms with shots.

    mitigator_apply_fn: None or a callable p_meas -> p_mitigated
    """
    bind = {params[i]: float(pvals[i]) for i in range(len(params))}
    basis_list = list(groups.keys())

    circuits = []
    for b in basis_list:
        qc = build_measurement_circuit(ansatz, b)
        circuits.append(qc.assign_parameters(bind, inplace=False))

    counts_list = run_measurement_batch(backend, circuits, shots=shots, transpile_level=transpile_level)

    n_qubits = ansatz.num_qubits
    E = 0.0
    for counts, b in zip(counts_list, basis_list):
        p = counts_to_probvec(counts, n_qubits)
        if mitigator_apply_fn is not None:
            p = mitigator_apply_fn(p)
        for term in groups[b]:
            E += term.coeff * float(parity_vec[term.mask] @ p)

    return float(E), counts_list, basis_list

def bootstrap_energy_se(
    groups: Dict[Tuple[str, ...], List[Term]],
    parity_vec: Dict[int, np.ndarray],
    basis_list: List[Tuple[str, ...]],
    counts_list: List[Dict[str, int]],
    n_qubits: int,
    shots: int,
    mitigator_apply_fn,
    n_boot: int = 200,
    seed: int = 123,
) -> float:
    """Bootstrap SE by resampling multinomials from measured distributions."""
    rng = np.random.default_rng(seed)

    P = [counts_to_probvec(c, n_qubits) for c in counts_list]
    if mitigator_apply_fn is not None:
        P = [mitigator_apply_fn(p) for p in P]

    terms_per_group = [groups[b] for b in basis_list]

    samples = np.empty(n_boot, dtype=float)
    for t in range(n_boot):
        E = 0.0
        for p, terms in zip(P, terms_per_group):
            c = rng.multinomial(shots, p)
            p_bs = c / shots
            for term in terms:
                E += term.coeff * float(parity_vec[term.mask] @ p_bs)
        samples[t] = E

    return float(np.std(samples, ddof=1))
