from __future__ import annotations

from typing import List, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

def build_ansatz_ry_cx_rz(n_qubits: int, n_layers: int, neel_init: bool = True) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Neel init + layers of [RY(all) -> CX chain -> RZ(all)]."""
    qc = QuantumCircuit(n_qubits)
    if neel_init:
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
