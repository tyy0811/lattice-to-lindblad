from __future__ import annotations

import numpy as np

from qiskit.quantum_info import Operator, SparsePauliOp

def pauli_decompose(H: np.ndarray, n_qubits: int, atol: float = 1e-10) -> SparsePauliOp:
    """Convert a full matrix Hamiltonian into a SparsePauliOp and drop tiny coefficients."""
    op = SparsePauliOp.from_operator(Operator(H))
    mask = np.abs(op.coeffs) > atol
    op = SparsePauliOp(op.paulis[mask], op.coeffs[mask])
    op = SparsePauliOp(op.paulis, np.real(op.coeffs)).simplify()
    return op
