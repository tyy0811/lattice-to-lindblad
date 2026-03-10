"""Entanglement utilities for MPS states.

This module provides functions for analyzing bipartite entanglement:
- compute_entropy_profile: von Neumann entropy at all cuts
- extract_schmidt_values: Schmidt values at a single cut
- extract_schmidt_values_by_sector: Schmidt values grouped by bond-charge sector
- entanglement_levels: convert Schmidt values to entanglement levels
- cumulative_weight: cumulative retained weight from Schmidt values

Cut Indexing Convention:
    Sites are numbered 0, ..., L-1.
    cut = i means the bipartition between site i and site i+1.
    Valid cuts are 0, ..., L-2.
    Internal translation: TeNPy bond = cut + 1.
"""
from __future__ import annotations

import numpy as np


def compute_entropy_profile(psi) -> tuple[np.ndarray, np.ndarray]:
    """Compute von Neumann entropy at all bipartition cuts.

    Parameters
    ----------
    psi : MPS
        TeNPy MPS object with bc='finite'.

    Returns
    -------
    cuts : np.ndarray
        Integer array [0, 1, ..., L-2].
    entropies : np.ndarray
        Float array of S_vN at each cut.

    Raises
    ------
    ValueError
        If psi.bc != 'finite' or psi.L < 2.
    """
    if getattr(psi, "bc", None) != "finite":
        raise ValueError("compute_entropy_profile requires finite MPS (psi.bc == 'finite')")
    L = psi.L
    if L < 2:
        raise ValueError("compute_entropy_profile requires MPS with at least 2 sites")

    # TeNPy bond indices for finite MPS: 1, ..., L-1
    # Our cut convention: cut i is between site i and site i+1
    # Mapping: cut i -> TeNPy bond i+1
    bonds = list(range(1, L))
    entropies_raw = psi.entanglement_entropy(bonds=bonds)

    cuts = np.arange(L - 1)
    entropies = np.array(entropies_raw, dtype=np.float64)

    return cuts, entropies


def extract_schmidt_values(psi, cut: int) -> np.ndarray:
    """Extract Schmidt values at a single cut.

    Parameters
    ----------
    psi : MPS
        TeNPy MPS object with bc='finite'.
    cut : int
        Cut position (0 <= cut <= L-2).

    Returns
    -------
    lambdas : np.ndarray
        Float array of Schmidt values. sum(lambdas**2) ~= 1 for normalized MPS.
        Not guaranteed sorted; caller should sort if needed.

    Raises
    ------
    ValueError
        If cut is out of range or psi.bc != 'finite'.
    """
    if getattr(psi, "bc", None) != "finite":
        raise ValueError("extract_schmidt_values requires finite MPS (psi.bc == 'finite')")
    L = psi.L
    if not (0 <= cut <= L - 2):
        raise ValueError(f"cut must be in [0, {L - 2}], got {cut}")

    # TeNPy: get_SL(i) returns singular values on left of site i
    # Our cut between site `cut` and `cut+1` -> TeNPy bond = cut + 1
    bond = cut + 1
    sv = psi.get_SL(bond)
    if sv is None:
        raise RuntimeError(f"MPS singular values at bond {bond} are None; is MPS in canonical form?")

    return np.array(sv, dtype=np.float64).copy()


def entanglement_levels(lambdas: np.ndarray, cutoff: float = 1e-15) -> np.ndarray:
    """Convert Schmidt values to entanglement levels xi_i = -log(lambda_i^2).

    Parameters
    ----------
    lambdas : np.ndarray
        Schmidt values for one cut.
    cutoff : float, optional
        Values below cutoff are clipped before taking log to avoid inf.

    Returns
    -------
    xi : np.ndarray
        Entanglement levels.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    clipped = np.maximum(lambdas, cutoff)
    return -np.log(clipped**2)


def cumulative_weight(lambdas: np.ndarray) -> np.ndarray:
    """Compute cumulative retained weight from Schmidt values.

    Parameters
    ----------
    lambdas : np.ndarray
        Schmidt values (caller should sort descending for "top-k" semantics).

    Returns
    -------
    cum : np.ndarray
        cum[k] = sum_{j=0}^k lambda_j^2.
        For normalized input, cum[-1] ~= 1.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    return np.cumsum(lambdas**2)


def _normalize_charge_label(charge) -> str:
    """Convert TeNPy charge to stable string label."""
    if isinstance(charge, (int, np.integer)):
        return f"q{int(charge)}"
    elif hasattr(charge, '__len__'):
        if len(charge) == 1:
            return f"q{int(charge[0])}"
        else:
            return f"q({','.join(str(int(c)) for c in charge)})"
    else:
        return f"q{charge}"


def extract_schmidt_values_by_sector(psi, cut: int) -> dict[str, np.ndarray]:
    """Extract Schmidt values grouped by bond-charge sector.

    Parameters
    ----------
    psi : MPS
        Finite TeNPy MPS in canonical form.
    cut : int
        Cut index (bipartition between site cut and cut+1).

    Returns
    -------
    sector_to_lambdas : dict[str, np.ndarray]
        Mapping from normalized sector label (string) to sorted (descending) Schmidt values.
    """
    if getattr(psi, "bc", None) != "finite":
        raise ValueError("extract_schmidt_values_by_sector requires finite MPS (psi.bc == 'finite')")
    L = psi.L
    if not (0 <= cut <= L - 2):
        raise ValueError(f"cut must be in [0, {L - 2}], got {cut}")

    bond = cut + 1
    S = psi.get_SL(bond)
    if S is None:
        raise RuntimeError(f"Schmidt values at bond {bond} are None; is MPS in canonical form?")

    leg = psi.get_B(cut, form="B").get_leg("vR")

    sector_to_lambdas = {}
    for qi in range(leg.block_number):
        q = leg.charges[qi]
        slc = slice(leg.slices[qi], leg.slices[qi + 1])
        label = _normalize_charge_label(q)
        lambdas = np.array(S[slc], dtype=np.float64).copy()
        sector_to_lambdas[label] = np.sort(lambdas)[::-1]

    return sector_to_lambdas


def compute_sector_weights(sector_to_lambdas: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute total probability weight per sector.

    Parameters
    ----------
    sector_to_lambdas : dict[str, np.ndarray]
        Mapping from sector label to Schmidt values in that sector.

    Returns
    -------
    sector_to_weight : dict[str, float]
        Mapping from sector label to sum(lambda^2) for that sector.
    """
    return {
        label: float(np.sum(lambdas**2)) if len(lambdas) > 0 else 0.0
        for label, lambdas in sector_to_lambdas.items()
    }
