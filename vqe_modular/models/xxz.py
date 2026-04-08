from __future__ import annotations

import numpy as np

def build_heisenberg_xxz(N: int, Jxy: float = 1.0, Jz: float = 1.0, pbc: bool = False) -> np.ndarray:
    """Heisenberg XXZ chain.

    H = Jxy Σ (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Σ Z_i Z_{i+1}

    Implemented in the computational (Z) basis, returning the full 2^N × 2^N matrix.

    Notes
    -----
    In the Z basis, (X⊗X + Y⊗Y) couples |01> <-> |10> with matrix element 2.
    """
    if N < 2:
        raise ValueError("XXZ requires N>=2")

    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)

    edges = [(i, i + 1) for i in range(N - 1)]
    if pbc and N > 2:
        edges.append((N - 1, 0))

    def z(s: int, i: int) -> float:
        return 1.0 if ((s >> i) & 1) == 0 else -1.0

    for s in range(dim):
        # ZZ diagonal
        diag = 0.0
        for (i, j) in edges:
            diag += Jz * z(s, i) * z(s, j)
        H[s, s] += diag

        # XY flip-flop off-diagonal
        if Jxy != 0:
            for (i, j) in edges:
                bi = (s >> i) & 1
                bj = (s >> j) & 1
                if bi != bj:
                    s2 = s ^ (1 << i) ^ (1 << j)
                    H[s, s2] += 2.0 * Jxy

    return 0.5 * (H + H.T)
