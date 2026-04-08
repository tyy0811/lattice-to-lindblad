from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from core.ansatz import energy_statevector

def run_ideal_vqe(
    ansatz: QuantumCircuit,
    params: List[Parameter],
    H: np.ndarray,
    restarts: int,
    maxiter: int,
    seed: int,
    method: str = "COBYLA",
) -> Tuple[float, np.ndarray]:
    """Run a small multi-start ideal VQE (statevector cost) to get good parameters."""
    rng = np.random.default_rng(seed)
    best_E = float("inf")
    best_p: Optional[np.ndarray] = None

    def cost(pv):
        return energy_statevector(pv, ansatz, params, H)

    for t in range(restarts):
        x0 = rng.uniform(-0.1, 0.1, len(params))
        t0 = time.time()
        res = minimize(cost, x0, method=method, options={"maxiter": maxiter, "rhobeg": 0.3} if method == "COBYLA" else {"maxiter": maxiter})
        dt = time.time() - t0
        E = float(res.fun)
        print(f"  trial {t+1}: E={E:.10f}  ({int(res.nfev)} evals, {dt:.1f}s)")
        if E < best_E:
            best_E = E
            best_p = np.array(res.x, dtype=float)

    assert best_p is not None
    return best_E, best_p
