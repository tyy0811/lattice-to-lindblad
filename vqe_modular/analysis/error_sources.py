from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from backends.aer_backend import build_aer_backend
from core.measurement import evaluate_energy_shots

def aer_error_source_ablation(
    *,
    ansatz,
    params,
    pvals: np.ndarray,
    groups,
    parity_vec,
    shots: int,
    transpile_level: int,
    p1q: float,
    p2q: float,
    p01: float,
    p10: float,
) -> Dict[str, float]:
    """Return energies for ideal-params evaluation with gate-only, readout-only, both."""
    aer_gate = build_aer_backend(p1q, p2q, 0.0, 0.0)
    Eg, _, _ = evaluate_energy_shots(aer_gate, ansatz, params, pvals, groups, parity_vec, shots, None, transpile_level)

    aer_ro = build_aer_backend(0.0, 0.0, p01, p10)
    Er, _, _ = evaluate_energy_shots(aer_ro, ansatz, params, pvals, groups, parity_vec, shots, None, transpile_level)

    aer_both = build_aer_backend(p1q, p2q, p01, p10)
    Eb, _, _ = evaluate_energy_shots(aer_both, ansatz, params, pvals, groups, parity_vec, shots, None, transpile_level)

    return {
        "E_aer_gate_only": float(Eg),
        "E_aer_readout_only": float(Er),
        "E_aer_both": float(Eb),
    }
