from __future__ import annotations

from typing import List, Tuple, Optional, Callable
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

from core.measurement import counts_to_probvec

def build_calibration_circuits(n_qubits: int) -> List[QuantumCircuit]:
    """Prepare each computational basis state and measure it to learn assignment matrix."""
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
    transpile_level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, A_pinv) where A_{y,x} = P(meas=y | prepared=x)."""
    cal = build_calibration_circuits(n_qubits)
    cal_t = transpile(cal, backend, optimization_level=transpile_level)
    job = backend.run(cal_t, shots=shots_cal)
    res = job.result()

    dim = 2**n_qubits
    A = np.zeros((dim, dim), dtype=float)
    for true_state in range(dim):
        A[:, true_state] = counts_to_probvec(res.get_counts(true_state), n_qubits)

    return A, np.linalg.pinv(A)

def make_apply_mem(A_inv: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function p_meas -> p_mitigated (clipped, renormalized)."""
    def apply(p_meas: np.ndarray) -> np.ndarray:
        p = A_inv @ p_meas
        p = np.clip(p, 0.0, None)
        s = float(np.sum(p))
        if s > 1e-12:
            p /= s
        return p
    return apply
