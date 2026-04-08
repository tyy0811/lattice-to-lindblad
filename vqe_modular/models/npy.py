from __future__ import annotations

import numpy as np

def load_hamiltonian_npy(path: str) -> np.ndarray:
    """Load a full Hamiltonian matrix from a .npy file."""
    H = np.load(path)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"Loaded matrix must be square; got {H.shape} from {path}")
    if not np.allclose(H, H.T.conj(), atol=1e-10):
        H = 0.5 * (H + H.T.conj())
    return np.real_if_close(H).astype(float)
