from __future__ import annotations

import math
import numpy as np

def build_schwinger_full(
    N: int,
    x: float,
    m_over_g: float = 0.0,
    E0: float = 0.0,
    x_def: str = "tagliacozzo",
) -> np.ndarray:
    """Return the full 2^N × 2^N Schwinger Hamiltonian matrix (Tagliacozzo mapping, no projection).

    Notes
    -----
    This is the same "full Hilbert space" Hamiltonian used in your earlier scripts:
      - Electric term built from running gauge field L
      - Staggered mass term
      - Nearest-neighbor hopping term which flips adjacent opposite bits

    Parameters
    ----------
    N : int
        Number of qubits / sites.
    x : float
        Dimensionless coupling.
    m_over_g : float
        Mass / coupling.
    E0 : float
        Background electric field.
    x_def : str
        Convention for defining ga. (Keep default to match previous results.)

    Returns
    -------
    np.ndarray
        Real symmetric Hamiltonian matrix.
    """
    if N < 2:
        raise ValueError("Schwinger model requires N>=2")
    if x <= 0:
        raise ValueError("x must be > 0")

    ga = 1.0 / math.sqrt(x) if x_def == "tagliacozzo" else 1.0 / math.sqrt(2 * x)
    mu = 2.0 * m_over_g / ga
    stag = np.array([1 if n % 2 == 0 else -1 for n in range(N)], dtype=float)

    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)

    for s in range(dim):
        L = E0
        diag_e = 0.0
        diag_m = 0.0

        # diagonal electric + mass
        for n in range(N):
            bit = (s >> n) & 1
            z = 1 - 2 * bit  # +1 for |0>, -1 for |1>
            diag_m += 0.5 * mu * stag[n] * z
            qn = 0.5 * (z + stag[n])
            L += qn
            if n <= N - 2:
                diag_e += L * L

        H[s, s] = diag_e + diag_m

        # off-diagonal hopping term (flip opposite neighbors)
        for n in range(N - 1):
            bn = (s >> n) & 1
            bn1 = (s >> (n + 1)) & 1
            if bn != bn1:
                s2 = s ^ (1 << n) ^ (1 << (n + 1))
                H[s, s2] += x

    # symmetrize
    return 0.5 * (H + H.T)
