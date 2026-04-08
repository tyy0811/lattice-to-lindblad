from __future__ import annotations

from typing import Sequence
import numpy as np

def zne_extrapolate(scales: Sequence[float], energies: Sequence[float], degree: int = 2) -> float:
    """Polynomial fit in scale factor and extrapolate to scale=0."""
    xs = np.asarray(scales, dtype=float)
    ys = np.asarray(energies, dtype=float)
    if len(xs) < degree + 1:
        raise ValueError("Need at least degree+1 scale points")
    poly = np.poly1d(np.polyfit(xs, ys, deg=degree))
    return float(poly(0.0))
