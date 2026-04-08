from __future__ import annotations

import numpy as np

def build_tfim(N: int, J: float = 1.0, h: float = 1.0, pbc: bool = False) -> np.ndarray:
    """Transverse-field Ising model.

    H = -J Σ Z_i Z_{i+1} - h Σ X_i

    Returns the full 2^N × 2^N Hamiltonian matrix.
    """
    if N < 2:
        raise ValueError("TFIM requires N>=2")

    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)

    def z(s: int, i: int) -> float:
        return 1.0 if ((s >> i) & 1) == 0 else -1.0

    for s in range(dim):
        zz = 0.0
        for i in range(N - 1):
            zz += z(s, i) * z(s, i + 1)
        if pbc and N > 2:
            zz += z(s, N - 1) * z(s, 0)
        H[s, s] += -J * zz

        for i in range(N):
            s2 = s ^ (1 << i)
            H[s, s2] += -h

    return 0.5 * (H + H.T)
