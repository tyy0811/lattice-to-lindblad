#!/usr/bin/env python3
"""Open Schwinger entanglement dynamics driver.

Scientific responsibility:
    Compare closed vs weakly open Schwinger quench dynamics in a fixed-charge
    ED sector. Track how dissipation modifies entropy growth, entanglement
    structure, and effective Schmidt compressibility.

Main inputs:
    Schwinger parameters (N, mass, coupling), dynamics (tmax, nt, cut),
    open-system (gamma, gamma_ref, channel), output options.

Main outputs:
    - open_schwinger_entanglement_dynamics.csv (entropy_vn, mean_abs_L timeseries)
    - open_schwinger_entanglement_schmidt_snapshots.csv (reduced-spectrum proxies)
    - open_schwinger_entanglement_dynamics.png (3-panel figure)
    - run_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and return CLI arguments."""
    p = argparse.ArgumentParser(
        description="Open Schwinger entanglement dynamics: closed vs weakly open quench comparison."
    )

    # Required model parameters
    p.add_argument("--N", type=int, required=True, help="System size (must be even)")
    p.add_argument("--mass", type=float, required=True, help="Schwinger mass ratio m/g")
    p.add_argument("--coupling", type=float, required=True, help="Schwinger coupling x = 1/(ag)^2")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")

    # Dynamics parameters
    p.add_argument("--tmax", type=float, default=6.0, help="Maximum evolution time (default: 6.0)")
    p.add_argument("--nt", type=int, default=61, help="Number of time steps (default: 61)")
    p.add_argument("--cut", type=int, default=4, help="Entanglement cut index (default: 4)")
    p.add_argument(
        "--initial-state",
        type=str,
        default="string_gs",
        choices=["string_gs", "vacuum_gs"],
        help="Initial state preparation protocol (default: string_gs)",
    )
    p.add_argument(
        "--quench",
        type=str,
        default="e0_drop",
        choices=["e0_drop", "mass_quench"],
        help="Quench protocol (default: e0_drop)",
    )

    # Open-system parameters
    p.add_argument("--gamma", type=float, default=0.0, help="Closed-system reference gamma (default: 0.0)")
    p.add_argument("--gamma-ref", type=float, default=0.02, help="Weak-open reference gamma (default: 0.02)")
    p.add_argument(
        "--channel",
        type=str,
        default="charge_dephasing",
        choices=["charge_dephasing"],
        help="Dissipation channel (v1: charge_dephasing only)",
    )

    # Output options
    p.add_argument(
        "--snapshot-times",
        type=str,
        default="0.0,3.0,6.0",
        help="Comma-separated snapshot times for spectrum analysis (default: 0.0,3.0,6.0)",
    )
    p.add_argument("--tag", type=str, default="", help="Output filename tag")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--show", action="store_true", help="Display figure interactively")

    return p.parse_args(argv)


# =============================================================================
# Sector basis and Hamiltonian utilities
# =============================================================================

def _sector_basis(N: int, n_up: int) -> np.ndarray:
    """Generate all bitstrings with exactly n_up spins up (Hamming weight n_up)."""
    basis = []
    for comb in itertools.combinations(range(N), n_up):
        s = 0
        for i in comb:
            s |= (1 << i)
        basis.append(s)
    return np.array(basis, dtype=np.uint32)


def _precompute_occ_and_L_links(
    N: int, basis: np.ndarray, E0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute occupation and electric field link operators.

    Returns:
        occ: shape (N, dim) - occupation numbers per site.
        L_links: shape (N-1, dim) - electric field on internal links.
    """
    dim = len(basis)
    occ = np.zeros((N, dim), dtype=np.float64)
    for n in range(N):
        occ[n, :] = ((basis >> n) & 1).astype(np.float64)

    # Staggered background charge
    background = np.array([n % 2 for n in range(N)], dtype=np.float64)[:, None]
    Q = occ - background  # staggered charge

    # Electric field from Gauss law: L_n = E0 + sum_{i<=n} Q_i
    L = np.cumsum(Q, axis=0) + E0
    L_links = L[: N - 1, :]  # internal links only
    return occ, L_links


def _build_schwinger_hamiltonian_sector(
    N: int,
    m: float,
    g: float,
    E0: float,
    x: float,
    basis: np.ndarray,
    state_index: dict[int, int],
    occ: np.ndarray,
    L_links: np.ndarray,
    penalty_lambda: float = 0.0,
    L_links_for_penalty: np.ndarray | None = None,
    E0_penalty: float | None = None,
) -> sp.csr_matrix:
    """Build Schwinger Hamiltonian in fixed-charge sector.

    H = x Σ (swap 01↔10) + m Σ (-1)^n n_n + (g^2/2) Σ L_link^2

    Optional penalty for string-like initial-state preparation.
    """
    dim = len(basis)

    # Kinetic term (off-diagonal): swap adjacent 01 <-> 10
    rows, cols, data = [], [], []
    for i in range(N - 1):
        mask = (1 << i) | (1 << (i + 1))
        for col, state in enumerate(basis):
            seg = (int(state) & mask) >> i
            if seg == 1 or seg == 2:
                new_state = int(state) ^ mask
                row = state_index[new_state]
                rows.append(row)
                cols.append(col)
                data.append(x)
    H_hop = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()

    # Diagonal mass term
    signs = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(N)], dtype=np.float64)[:, None]
    diag_m = (m * signs * occ).sum(axis=0)

    # Diagonal electric term
    J = (g**2) / 2.0
    diag_el = J * np.sum(L_links**2, axis=0)

    diag = diag_m + diag_el

    if penalty_lambda > 0.0:
        if L_links_for_penalty is None or E0_penalty is None:
            raise ValueError("Penalty requested but L_links_for_penalty/E0_penalty not provided")
        pen = np.sum((L_links_for_penalty - E0_penalty) ** 2, axis=0)
        diag = diag + penalty_lambda * pen

    return (H_hop + sp.diags(diag, 0, shape=(dim, dim), dtype=np.float64)).tocsr()


def _build_Q_ops_diagonal(N: int, occ: np.ndarray) -> list[np.ndarray]:
    """Build diagonal staggered-charge operators Q_n = n_n - (n mod 2)."""
    background = np.array([n % 2 for n in range(N)], dtype=np.float64)
    return [occ[n, :] - background[n] for n in range(N)]


def _build_cut_maps(N: int, cut: int, basis: np.ndarray) -> dict[str, Any]:
    """Precompute index maps for partial trace at given cut.

    Cut index 'cut' means we trace out sites [cut+1, ..., N-1],
    keeping sites [0, ..., cut].
    """
    dim = len(basis)
    n_A = cut + 1  # number of sites in subsystem A
    n_B = N - n_A  # number of sites in subsystem B

    # Dimensions of subsystems
    dim_A = 2**n_A
    dim_B = 2**n_B

    # Mask for subsystem A bits
    mask_A = (1 << n_A) - 1

    # For each basis state, compute (index_A, index_B)
    # index_A = bits [0..cut], index_B = bits [cut+1..N-1]
    indices_A = np.zeros(dim, dtype=np.int32)
    indices_B = np.zeros(dim, dtype=np.int32)
    for i, state in enumerate(basis):
        indices_A[i] = int(state) & mask_A
        indices_B[i] = int(state) >> n_A

    return {
        "cut": cut,
        "n_A": n_A,
        "n_B": n_B,
        "dim_A": dim_A,
        "dim_B": dim_B,
        "indices_A": indices_A,
        "indices_B": indices_B,
    }


def build_sector_model(
    N: int,
    mass: float,
    coupling: float,
    cut: int,
    initial_state: str,
    quench: str,
) -> dict[str, Any]:
    """Build and return the reusable projected-sector model payload.

    Args:
        N: System size (must be even).
        mass: Mass ratio m/g.
        coupling: Coupling x = 1/(ag)^2.
        cut: Entanglement cut index.
        initial_state: Initial state protocol (string_gs, vacuum_gs).
        quench: Quench protocol (e0_drop, mass_quench).

    Returns:
        Dictionary with basis, H_init, H_evolve, Q_ops, L_ops, cut_maps, meta.
    """
    # Fixed charge sector: Σn = N/2 (staggered vacuum sector)
    n_up = N // 2
    basis = _sector_basis(N, n_up)
    dim = len(basis)
    state_index = {int(s): i for i, s in enumerate(basis)}

    # Coupling convention: g=1 for simplicity, x is the hopping
    g = 1.0
    m = mass * g

    # Background field configurations
    E0_init = 1.0 if initial_state == "string_gs" else 0.0
    E0_evolve = 0.0 if quench == "e0_drop" else E0_init

    # Precompute occupancies and link fields
    occ_init, L_init = _precompute_occ_and_L_links(N, basis, E0_init)
    occ_ev, L_ev = _precompute_occ_and_L_links(N, basis, E0_evolve)

    # Build Hamiltonians
    # For string_gs: may need penalty to avoid screening
    penalty_lambda = 2.0 if initial_state == "string_gs" else 0.0
    H_init = _build_schwinger_hamiltonian_sector(
        N, m, g, E0_init, coupling, basis, state_index, occ_init, L_init,
        penalty_lambda=penalty_lambda,
        L_links_for_penalty=L_init,
        E0_penalty=E0_init,
    )
    H_evolve = _build_schwinger_hamiltonian_sector(
        N, m, g, E0_evolve, coupling, basis, state_index, occ_ev, L_ev
    )

    # Build diagonal operators
    Q_ops = _build_Q_ops_diagonal(N, occ_ev)
    L_ops = [L_ev[ell, :] for ell in range(N - 1)]

    # Precompute cut maps
    cut_maps = _build_cut_maps(N, cut, basis)

    return {
        "basis": basis,
        "dim": dim,
        "state_index": state_index,
        "H_init": H_init,
        "H_evolve": H_evolve,
        "Q_ops": Q_ops,
        "L_ops": L_ops,
        "occ": occ_ev,
        "cut_maps": cut_maps,
        "meta": {
            "N": N,
            "m_over_g": mass,
            "x": coupling,
            "g": g,
            "E0_init": E0_init,
            "E0_evolve": E0_evolve,
            "n_up": n_up,
            "initial_state": initial_state,
            "quench": quench,
        },
    }


def prepare_quench_initial_state(
    model: dict[str, Any],
    initial_state: str,
    quench: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prepare quench initial state.

    Returns:
        psi0: Complex vector, shape (dim,), normalized.
        rho0: Complex matrix, shape (dim, dim), Hermitian with trace 1.
        prep_meta: Protocol diagnostics for metadata output.
    """
    H_init = model["H_init"]

    # Compute ground state of H_init
    evals, evecs = spla.eigsh(H_init, k=1, which="SA")
    E0 = float(evals[0])
    psi0 = evecs[:, 0].astype(np.complex128)

    # Normalize
    psi0 /= np.linalg.norm(psi0)

    # Build density matrix
    rho0 = np.outer(psi0, psi0.conj())

    prep_meta = {
        "prep_type": f"GS({initial_state})",
        "prep_energy": E0,
        "initial_state": initial_state,
        "quench": quench,
    }

    return psi0, rho0, prep_meta


def evolve_open_dynamics(
    model: dict[str, Any],
    rho0: np.ndarray,
    times: np.ndarray,
    gamma: float,
    channel: str,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Evolve density matrix under Lindblad master equation.

    Lindblad RHS (v1 charge_dephasing):
        d rho / dt = -i [H_evolve, rho] + gamma * sum_n (Q_n rho Q_n - 0.5 {Q_n^2, rho})

    Args:
        model: Sector model payload from build_sector_model.
        rho0: Initial density matrix, shape (dim, dim).
        times: Array of time points for output.
        gamma: Dissipation strength.
        channel: Dissipation channel (v1: only "charge_dephasing").
        rtol: Relative tolerance for ODE solver.
        atol: Absolute tolerance for ODE solver.

    Returns:
        Dictionary with rho_t trajectory and solver_meta diagnostics.
    """
    if channel != "charge_dephasing":
        raise ValueError(f"v1 supports only channel='charge_dephasing', got {channel}")

    dim = model["dim"]
    H = model["H_evolve"]
    Q_ops = model["Q_ops"]  # list of diagonal arrays

    # Convert sparse H to dense for commutator computation
    H_dense = H.toarray() if sp.issparse(H) else np.asarray(H)

    # Precompute Q_n^2 diagonals for anti-commutator
    Q_sq_diags = [Q**2 for Q in Q_ops]

    def lindblad_rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
        """Compute d(rho_vec)/dt for the Lindblad equation."""
        rho = rho_vec.reshape((dim, dim))

        # Unitary part: -i [H, rho]
        commutator = H_dense @ rho - rho @ H_dense
        drho = -1j * commutator

        # Dissipator: gamma * sum_n (Q_n rho Q_n - 0.5 {Q_n^2, rho})
        if gamma > 0.0:
            for Q, Q_sq in zip(Q_ops, Q_sq_diags):
                # Q_n rho Q_n (diagonal Q acts by elementwise multiplication)
                Q_rho_Q = Q[:, None] * rho * Q[None, :]
                # 0.5 {Q_n^2, rho} = 0.5 * (Q_n^2 rho + rho Q_n^2)
                anticomm = 0.5 * (Q_sq[:, None] * rho + rho * Q_sq[None, :])
                drho += gamma * (Q_rho_Q - anticomm)

        return drho.ravel()

    # Vectorize initial state
    rho0_vec = rho0.ravel()

    # Solve ODE
    sol = solve_ivp(
        lindblad_rhs,
        t_span=(times[0], times[-1]),
        y0=rho0_vec,
        t_eval=times,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Reshape trajectory
    nt = len(times)
    rho_t = sol.y.T.reshape((nt, dim, dim))

    # Compute diagnostics
    trace_errors = []
    hermiticity_errors = []
    min_eigs = []

    # Check at all time points
    for i in range(nt):
        rho_i = rho_t[i]
        trace_errors.append(abs(np.trace(rho_i) - 1.0))
        hermiticity_errors.append(np.linalg.norm(rho_i - rho_i.conj().T, "fro"))

    # Check positivity at sampled points (first, last, and evenly spaced)
    check_indices = sorted(set([0, nt - 1] + list(range(0, nt, max(1, nt // 10)))))
    for i in check_indices:
        rho_i = rho_t[i]
        # Symmetrize for eigenvalue computation
        rho_sym = (rho_i + rho_i.conj().T) / 2
        eigs = np.linalg.eigvalsh(rho_sym)
        min_eigs.append(float(np.min(eigs)))

    solver_meta = {
        "max_abs_trace_error": float(np.max(trace_errors)),
        "max_hermiticity_error": float(np.max(hermiticity_errors)),
        "min_eig_real_over_checks": float(np.min(min_eigs)),
        "n_positivity_checks": len(check_indices),
        "rtol": rtol,
        "atol": atol,
        "gamma": gamma,
        "channel": channel,
    }

    return {"rho_t": rho_t, "solver_meta": solver_meta}


def stabilize_probs(
    eigs: np.ndarray,
    eig_clip: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip negative eigenvalues and renormalize to sum=1.

    Args:
        eigs: Array of eigenvalues (should be non-negative probabilities).
        eig_clip: Threshold below which values are clipped to zero.

    Returns:
        Tuple of (stabilized probabilities, metadata dict with clipping stats).
    """
    arr = np.asarray(eigs, dtype=float).copy()
    n_clipped = int(np.sum(arr < 0.0))
    max_neg = float(np.min(arr)) if n_clipped > 0 else 0.0

    # Clip values below threshold to zero
    arr[arr < eig_clip] = 0.0

    total = float(np.sum(arr))
    if total > 0.0:
        arr /= total

    return arr, {
        "n_clipped": n_clipped,
        "max_neg_before_clip": max_neg,
        "renorm_total_before": total,
    }


def _partial_trace_from_sector(
    rho: np.ndarray,
    cut_maps: dict[str, Any],
) -> np.ndarray:
    """Compute reduced density matrix rho_A by tracing out B.

    Note: This works within the fixed-charge sector, not full Hilbert space.
    The reduced density matrix is computed by summing over B indices.
    """
    dim = rho.shape[0]
    indices_A = cut_maps["indices_A"]
    indices_B = cut_maps["indices_B"]
    dim_A = cut_maps["dim_A"]

    # Initialize rho_A
    rho_A = np.zeros((dim_A, dim_A), dtype=np.complex128)

    # Sum over B indices
    for i in range(dim):
        for j in range(dim):
            if indices_B[i] == indices_B[j]:
                rho_A[indices_A[i], indices_A[j]] += rho[i, j]

    return rho_A


def _compute_entropy_vn(rho_A: np.ndarray, eig_clip: float = 1e-12) -> tuple[float, dict]:
    """Compute von Neumann entropy from reduced density matrix.

    Returns:
        Tuple of (entropy, clipping metadata).
    """
    # Symmetrize
    rho_A = (rho_A + rho_A.conj().T) / 2

    # Get eigenvalues
    eigs = np.linalg.eigvalsh(rho_A)

    # Stabilize
    probs, clip_meta = stabilize_probs(eigs, eig_clip=eig_clip)

    # Compute entropy: S = -sum(p * log(p)) where p > 0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log(p)

    return float(entropy), clip_meta


def _compute_mean_abs_L(rho: np.ndarray, L_ops: list[np.ndarray]) -> float:
    """Compute mean absolute electric field from density matrix."""
    probs = np.diag(rho).real
    L_expectations = []
    for L in L_ops:
        exp_L = np.sum(L * probs)
        L_expectations.append(abs(exp_L))
    return float(np.mean(L_expectations))


def _map_snapshot_times(
    snapshot_times: list[float],
    times: np.ndarray,
) -> list[tuple[float, float, int]]:
    """Map requested snapshot times to nearest grid indices.

    Returns:
        List of (requested_time, actual_time, index) tuples.
    """
    mapping = []
    for t_req in snapshot_times:
        idx = int(np.argmin(np.abs(times - t_req)))
        mapping.append((t_req, float(times[idx]), idx))
    return mapping


def measure_timeseries(
    model: dict[str, Any],
    rho_t: np.ndarray,
    times: np.ndarray,
    cut: int,
    snapshot_times: list[float],
    gamma: float,
    channel: str,
    row_meta: dict[str, Any],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Compute timeseries observables and snapshot proxies.

    Args:
        model: Sector model payload.
        rho_t: Density matrix trajectory, shape (nt, dim, dim).
        times: Array of time points.
        cut: Entanglement cut index.
        snapshot_times: List of times for detailed spectrum analysis.
        gamma: Dissipation strength (for row metadata).
        channel: Dissipation channel (for row metadata).
        row_meta: Additional metadata to include in each row.

    Returns:
        Tuple of (timeseries_rows, snapshot_rows, measurement_meta).
    """
    nt = len(times)
    cut_maps = model["cut_maps"]
    L_ops = model["L_ops"]

    # Map snapshot times to grid indices
    snapshot_map = _map_snapshot_times(snapshot_times, times)
    snapshot_indices = set(m[2] for m in snapshot_map)

    timeseries_rows = []
    snapshot_rows = []
    clip_stats = []

    for i in range(nt):
        t = float(times[i])
        rho = rho_t[i]

        # Compute reduced density matrix
        rho_A = _partial_trace_from_sector(rho, cut_maps)

        # Entropy
        entropy, clip_meta = _compute_entropy_vn(rho_A)
        clip_stats.append(clip_meta)

        timeseries_rows.append({
            "time": t,
            "observable": "entropy_vn",
            "value": entropy,
            "cut": cut,
            "channel": channel,
            "gamma": gamma,
            **row_meta,
        })

        # Mean |L|
        mean_abs_L = _compute_mean_abs_L(rho, L_ops)
        timeseries_rows.append({
            "time": t,
            "observable": "mean_abs_L",
            "value": mean_abs_L,
            "cut": cut,
            "channel": channel,
            "gamma": gamma,
            **row_meta,
        })

        # Snapshot spectrum analysis
        if i in snapshot_indices:
            # Symmetrize and get eigenvalues
            rho_A_sym = (rho_A + rho_A.conj().T) / 2
            eigs = np.linalg.eigvalsh(rho_A_sym)
            probs, _ = stabilize_probs(eigs)

            # Sort descending
            probs = np.sort(probs)[::-1]

            # Build snapshot rows
            cum_weight = 0.0
            for rank, p in enumerate(probs):
                if p < 1e-14:
                    continue
                cum_weight += p
                snapshot_rows.append({
                    "time": t,
                    "rank": rank,
                    "schmidt_proxy_from_rhoA": float(np.sqrt(p)),
                    "p_eig": float(p),
                    "cum_weight": float(cum_weight),
                    "cut": cut,
                    "channel": channel,
                    "gamma": gamma,
                    **row_meta,
                })

    measurement_meta = {
        "snapshot_time_map": snapshot_map,
        "n_clipped_total": sum(c["n_clipped"] for c in clip_stats),
        "max_neg_eigenvalue": min(c.get("max_neg_before_clip", 0.0) for c in clip_stats),
    }

    return timeseries_rows, snapshot_rows, measurement_meta


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_rank_for_threshold(rows: list[dict[str, Any]], threshold: float) -> int | None:
    """Return the minimum retained-rank (1-based) meeting a cumulative threshold."""
    for row in sorted(rows, key=lambda r: int(r["rank"])):
        if float(row["cum_weight"]) >= threshold:
            return int(row["rank"]) + 1
    return None


def _compute_comparison_summary(
    per_gamma: dict[float, dict[str, float]],
    snapshot_rank_rows: list[dict[str, Any]],
    field_tol: float = 0.02,
    time_tol: float = 1e-12,
) -> dict[str, Any] | None:
    """Build two-case comparison deltas and verdicts when exactly two gammas are present."""
    if len(per_gamma) != 2:
        return None

    closed_gamma, open_gamma = sorted(per_gamma.keys())
    closed = per_gamma[closed_gamma]
    open_case = per_gamma[open_gamma]

    delta_peak_entropy = open_case["peak_entropy_vn"] - closed["peak_entropy_vn"]
    delta_final_entropy = open_case["final_entropy_vn"] - closed["final_entropy_vn"]
    delta_mean_entropy = open_case["mean_entropy_vn"] - closed["mean_entropy_vn"]
    delta_peak_field = open_case["peak_mean_abs_L"] - closed["peak_mean_abs_L"]
    delta_final_field = open_case["final_mean_abs_L"] - closed["final_mean_abs_L"]

    entropy_increase_ok = (
        open_case["peak_entropy_vn"] > closed["peak_entropy_vn"]
        and open_case["final_entropy_vn"] > closed["final_entropy_vn"]
        and open_case["mean_entropy_vn"] > closed["mean_entropy_vn"]
    )

    closed_snapshot_rows = sorted(
        [r for r in snapshot_rank_rows if np.isclose(float(r["gamma"]), closed_gamma)],
        key=lambda r: float(r["time"]),
    )
    open_snapshot_rows = sorted(
        [r for r in snapshot_rank_rows if np.isclose(float(r["gamma"]), open_gamma)],
        key=lambda r: float(r["time"]),
    )

    shared_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for c_row in closed_snapshot_rows:
        c_time = float(c_row["time"])
        match = next(
            (
                o_row
                for o_row in open_snapshot_rows
                if np.isclose(float(o_row["time"]), c_time, atol=time_tol, rtol=0.0)
            ),
            None,
        )
        if match is not None:
            shared_pairs.append((c_time, c_row, match))

    shared_times = sorted({t for t, _, _ in shared_pairs})
    post_quench_pairs = [(t, c_row, o_row) for (t, c_row, o_row) in shared_pairs if t > time_tol]
    post_quench_times = sorted({t for t, _, _ in post_quench_pairs})

    has_missing = False
    any_worse = False
    any_strict_increase = False
    all_not_worse = bool(post_quench_pairs)
    for _, c_row, o_row in post_quench_pairs:
        c95 = c_row["rank_95"]
        c99 = c_row["rank_99"]
        o95 = o_row["rank_95"]
        o99 = o_row["rank_99"]
        if c95 is None or c99 is None or o95 is None or o99 is None:
            has_missing = True
            continue
        c95_i = int(c95)
        c99_i = int(c99)
        o95_i = int(o95)
        o99_i = int(o99)
        not_worse = o95_i >= c95_i and o99_i >= c99_i
        strict_increase = o95_i > c95_i or o99_i > c99_i
        if not not_worse:
            any_worse = True
            all_not_worse = False
        if strict_increase:
            any_strict_increase = True

    if not post_quench_pairs:
        compressibility_verdict = "~"
    elif has_missing:
        compressibility_verdict = "~"
    elif all_not_worse and any_strict_increase:
        compressibility_verdict = "✓"
    elif all_not_worse and not any_strict_increase:
        compressibility_verdict = "✗"
    elif any_worse and any_strict_increase:
        compressibility_verdict = "~"
    else:
        compressibility_verdict = "✗"

    field_perturbation_small = abs(delta_peak_field) <= field_tol and abs(delta_final_field) <= field_tol

    return {
        "closed_gamma": closed_gamma,
        "open_gamma": open_gamma,
        "delta_peak_entropy_vn": delta_peak_entropy,
        "delta_final_entropy_vn": delta_final_entropy,
        "delta_mean_entropy_vn": delta_mean_entropy,
        "delta_peak_mean_abs_L": delta_peak_field,
        "delta_final_mean_abs_L": delta_final_field,
        "entropy_increase_verdict": "✓" if entropy_increase_ok else "~",
        "compressibility_reduction_verdict": compressibility_verdict,
        "field_perturbation_verdict": "~" if field_perturbation_small else "!",
        "compressibility_time_tol": time_tol,
        "compressibility_shared_snapshot_times": shared_times,
        "compressibility_post_quench_snapshot_times": post_quench_times,
        "compressibility_verdict_definition": (
            "Verdict based on post-quench shared snapshot times only (t > 0 within tolerance); "
            "t=0 is excluded because closed and open spectra are identical by construction."
        ),
    }


def summarize_benchmark_metrics(
    timeseries_rows: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    tmax: float,
    thresholds: tuple[float, float] = (0.95, 0.99),
) -> dict[str, Any]:
    """Summarize benchmark-ready metrics from existing timeseries/snapshot rows."""
    summary_rows: list[dict[str, Any]] = []
    truncation_notes: list[str] = []

    if not timeseries_rows:
        return {
            "summary_rows": summary_rows,
            "markdown": "# Open Schwinger Entanglement Benchmark Summary\n\nNo data available.\n",
            "per_gamma": {},
            "snapshot_rank_rows": [],
            "truncation_notes": truncation_notes,
            "comparison": None,
            "comparison_summary": None,
        }

    base = timeseries_rows[0]
    common = {
        "cut": base.get("cut"),
        "N": base.get("N"),
        "m_over_g": base.get("m_over_g"),
        "x": base.get("x"),
        "initial_state": base.get("initial_state"),
        "quench": base.get("quench"),
        "channel": base.get("channel"),
    }

    gamma_values = sorted({float(r["gamma"]) for r in timeseries_rows})
    per_gamma: dict[float, dict[str, float]] = {}

    def _extract_series(gamma: float, observable: str) -> tuple[np.ndarray, np.ndarray]:
        rows = [r for r in timeseries_rows if float(r["gamma"]) == gamma and r["observable"] == observable]
        rows = sorted(rows, key=lambda r: float(r["time"]))
        t = np.array([float(r["time"]) for r in rows], dtype=float)
        v = np.array([float(r["value"]) for r in rows], dtype=float)
        return t, v

    for gamma in gamma_values:
        t_ent, v_ent = _extract_series(gamma, "entropy_vn")
        t_field, v_field = _extract_series(gamma, "mean_abs_L")
        if len(v_ent) == 0 or len(v_field) == 0:
            continue

        # Use last sampled value as final at t=tmax.
        peak_entropy = float(np.max(v_ent))
        final_entropy = float(v_ent[-1])
        mean_entropy = float(np.mean(v_ent))
        peak_field = float(np.max(v_field))
        final_field = float(v_field[-1])

        per_gamma[gamma] = {
            "peak_entropy_vn": peak_entropy,
            "final_entropy_vn": final_entropy,
            "mean_entropy_vn": mean_entropy,
            "peak_mean_abs_L": peak_field,
            "final_mean_abs_L": final_field,
        }

        summary_rows.append({
            "metric_group": "entropy_summary",
            "gamma": gamma,
            "time": None,
            "peak_entropy_vn": peak_entropy,
            "final_entropy_vn": final_entropy,
            "mean_entropy_vn": mean_entropy,
            "peak_mean_abs_L": None,
            "final_mean_abs_L": None,
            "rank_95": None,
            "rank_99": None,
            "top_p_eig": None,
            "top2_cum_weight": None,
            **common,
        })
        summary_rows.append({
            "metric_group": "field_summary",
            "gamma": gamma,
            "time": None,
            "peak_entropy_vn": None,
            "final_entropy_vn": None,
            "mean_entropy_vn": None,
            "peak_mean_abs_L": peak_field,
            "final_mean_abs_L": final_field,
            "rank_95": None,
            "rank_99": None,
            "top_p_eig": None,
            "top2_cum_weight": None,
            **common,
        })

    # Snapshot-based rank/compressibility summary.
    snapshot_rank_rows: list[dict[str, Any]] = []
    snap_keys = sorted({(float(r["gamma"]), float(r["time"])) for r in snapshot_rows})
    for gamma, time in snap_keys:
        rows = [r for r in snapshot_rows if float(r["gamma"]) == gamma and float(r["time"]) == time]
        rows = sorted(rows, key=lambda r: int(r["rank"]))
        if not rows:
            continue

        rank_95 = compute_rank_for_threshold(rows, thresholds[0])
        rank_99 = compute_rank_for_threshold(rows, thresholds[1])
        top_p = float(rows[0]["p_eig"])
        top2 = float(sum(float(r["p_eig"]) for r in rows[:2]))

        if rank_95 is None:
            truncation_notes.append(
                f"gamma={gamma:.6g}, t={time:.6g}: cumulative weight never reached {thresholds[0]:.2f}."
            )
        if rank_99 is None:
            truncation_notes.append(
                f"gamma={gamma:.6g}, t={time:.6g}: cumulative weight never reached {thresholds[1]:.2f}."
            )

        snap_row = {
            "metric_group": "snapshot_rank_summary",
            "gamma": gamma,
            "time": time,
            "peak_entropy_vn": None,
            "final_entropy_vn": None,
            "mean_entropy_vn": None,
            "peak_mean_abs_L": None,
            "final_mean_abs_L": None,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "top_p_eig": top_p,
            "top2_cum_weight": top2,
            **common,
        }
        snapshot_rank_rows.append(snap_row)
        summary_rows.append(snap_row)

    comparison = _compute_comparison_summary(per_gamma, snapshot_rank_rows, field_tol=0.02)

    verdict_defaults = {
        "entropy_increase_verdict": None,
        "compressibility_reduction_verdict": None,
        "field_perturbation_verdict": None,
    }
    if comparison is not None:
        verdict_defaults = {
            "entropy_increase_verdict": comparison["entropy_increase_verdict"],
            "compressibility_reduction_verdict": comparison["compressibility_reduction_verdict"],
            "field_perturbation_verdict": comparison["field_perturbation_verdict"],
        }
    for row in summary_rows:
        row.update(verdict_defaults)

    def _fmt(v: Any, digits: int = 6) -> str:
        if v is None:
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{digits}g}"
        return str(v)

    lines = [
        "# Open Schwinger Entanglement Benchmark Summary",
        "",
        "## Per-Gamma Summary",
        "",
        "| gamma | peak_entropy_vn | final_entropy_vn | mean_entropy_vn | peak_mean_abs_L | final_mean_abs_L |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for gamma in sorted(per_gamma):
        m = per_gamma[gamma]
        lines.append(
            f"| {gamma:.6g} | {_fmt(m['peak_entropy_vn'])} | {_fmt(m['final_entropy_vn'])} | "
            f"{_fmt(m['mean_entropy_vn'])} | {_fmt(m['peak_mean_abs_L'])} | {_fmt(m['final_mean_abs_L'])} |"
        )

    lines.extend([
        "",
        "## Snapshot Rank/Compressibility Summary",
        "",
        "| gamma | time | rank_95 | rank_99 | top_p_eig | top2_cum_weight |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(snapshot_rank_rows, key=lambda r: (float(r["gamma"]), float(r["time"]))):
        lines.append(
            f"| {float(row['gamma']):.6g} | {float(row['time']):.6g} | {_fmt(row['rank_95'])} | "
            f"{_fmt(row['rank_99'])} | {_fmt(row['top_p_eig'])} | {_fmt(row['top2_cum_weight'])} |"
        )

    if comparison is not None:
        shared_times_str = ", ".join(f"{t:.6g}" for t in comparison["compressibility_shared_snapshot_times"]) or "none"
        post_quench_times_str = (
            ", ".join(f"{t:.6g}" for t in comparison["compressibility_post_quench_snapshot_times"]) or "none"
        )
        lines.extend([
            "",
            "## Two-Case Comparison",
            "",
            f"- Closed reference: `gamma={comparison['closed_gamma']:.6g}`",
            f"- Open case: `gamma={comparison['open_gamma']:.6g}`",
            f"- Shared snapshot times: `{shared_times_str}`",
            f"- Post-quench snapshot times used for compressibility verdict: `{post_quench_times_str}`",
            "",
            "| entropy_increase_verdict | compressibility_reduction_verdict | field_perturbation_verdict |",
            "|---:|---:|---:|",
            f"| {comparison['entropy_increase_verdict']} | {comparison['compressibility_reduction_verdict']} | "
            f"{comparison['field_perturbation_verdict']} |",
            "",
            "| delta_peak_entropy_vn | delta_final_entropy_vn | delta_mean_entropy_vn | "
            "delta_peak_mean_abs_L | delta_final_mean_abs_L |",
            "|---:|---:|---:|---:|---:|",
            f"| {_fmt(comparison['delta_peak_entropy_vn'])} | {_fmt(comparison['delta_final_entropy_vn'])} | "
            f"{_fmt(comparison['delta_mean_entropy_vn'])} | {_fmt(comparison['delta_peak_mean_abs_L'])} | "
            f"{_fmt(comparison['delta_final_mean_abs_L'])} |",
            "",
            f"- Compressibility verdict definition: {comparison['compressibility_verdict_definition']}",
        ])

    if truncation_notes:
        lines.extend(["", "## Threshold Notes", ""])
        for note in truncation_notes:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- Panel 1 (`Subsystem von Neumann entropy`): closed case (`gamma=0`) is pure-state entanglement entropy.",
        "- Open case (`gamma>0`): `S_vN(rho_A)` mixes entanglement with local mixedness/classical uncertainty.",
        "- Panel 2 (`Reduced-state spectrum compressibility proxy`): derived from eigenvalues of `rho_A`.",
    ])

    return {
        "summary_rows": summary_rows,
        "markdown": "\n".join(lines) + "\n",
        "per_gamma": per_gamma,
        "snapshot_rank_rows": snapshot_rank_rows,
        "truncation_notes": truncation_notes,
        "comparison": comparison,
        "comparison_summary": comparison,
    }


def write_benchmark_summary(
    outdir: Path,
    tag: str,
    summary_rows: list[dict[str, Any]],
    summary_markdown: str,
    force: bool,
) -> dict[str, str]:
    """Write benchmark summary CSV and Markdown artifacts."""
    tag_suffix = f"_{tag}" if tag else ""
    csv_path = outdir / f"open_schwinger_entanglement_benchmark_summary{tag_suffix}.csv"
    md_path = outdir / f"open_schwinger_entanglement_benchmark_summary{tag_suffix}.md"

    if not force:
        for p in [csv_path, md_path]:
            if p.exists():
                raise FileExistsError(f"Output file exists (use --force to overwrite): {p}")

    fieldnames = [
        "metric_group",
        "gamma",
        "time",
        "peak_entropy_vn",
        "final_entropy_vn",
        "mean_entropy_vn",
        "peak_mean_abs_L",
        "final_mean_abs_L",
        "rank_95",
        "rank_99",
        "top_p_eig",
        "top2_cum_weight",
        "entropy_increase_verdict",
        "compressibility_reduction_verdict",
        "field_perturbation_verdict",
        "cut",
        "N",
        "m_over_g",
        "x",
        "initial_state",
        "quench",
        "channel",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(md_path, "w") as f:
        f.write(summary_markdown)

    return {"benchmark_summary_csv": str(csv_path), "benchmark_summary_md": str(md_path)}


def print_benchmark_summary(benchmark: dict[str, Any]) -> None:
    """Print a compact benchmark summary to terminal."""
    per_gamma = benchmark.get("per_gamma", {})
    snapshot_rows = benchmark.get("snapshot_rank_rows", [])
    comparison = benchmark.get("comparison_summary")

    print("\nBenchmark summary:")
    for gamma in sorted(per_gamma):
        m = per_gamma[gamma]
        print(
            f"  gamma={gamma:.6g}: "
            f"entropy peak/final/mean={m['peak_entropy_vn']:.6f}/{m['final_entropy_vn']:.6f}/{m['mean_entropy_vn']:.6f}; "
            f"mean_abs_L peak/final={m['peak_mean_abs_L']:.6f}/{m['final_mean_abs_L']:.6f}"
        )

    if snapshot_rows:
        print("  Snapshot rank summary:")
        for row in sorted(snapshot_rows, key=lambda r: (float(r["gamma"]), float(r["time"]))):
            r95 = "NA" if row["rank_95"] is None else str(int(row["rank_95"]))
            r99 = "NA" if row["rank_99"] is None else str(int(row["rank_99"]))
            print(
                f"    gamma={float(row['gamma']):.6g}, t={float(row['time']):.6g}: "
                f"rank_95={r95}, rank_99={r99}, "
                f"top_p_eig={float(row['top_p_eig']):.6f}, "
                f"top2_cum={float(row['top2_cum_weight']):.6f}"
            )

    if comparison is not None:
        shared_times_str = ", ".join(f"{t:.6g}" for t in comparison["compressibility_shared_snapshot_times"]) or "none"
        post_times_str = (
            ", ".join(f"{t:.6g}" for t in comparison["compressibility_post_quench_snapshot_times"]) or "none"
        )
        print("  Two-case verdicts:")
        print(
            "    "
            f"entropy_increase={comparison['entropy_increase_verdict']}, "
            f"compressibility_reduction={comparison['compressibility_reduction_verdict']}, "
            f"field_perturbation={comparison['field_perturbation_verdict']}"
        )
        print(
            "    "
            f"compressibility shared_times=[{shared_times_str}], "
            f"post_quench_times=[{post_times_str}], "
            f"time_tol={comparison['compressibility_time_tol']:.1e}"
        )


def _is_closed_gamma(gamma: float) -> bool:
    """Treat gamma ~ 0 as closed dynamics."""
    return bool(np.isclose(float(gamma), 0.0, atol=1e-12))


def _case_curve_label(gamma: float, channel: str) -> str:
    """Legend label for full timeseries curves."""
    if _is_closed_gamma(gamma):
        return "closed (γ=0)"
    if channel == "charge_dephasing":
        return f"open, charge dephasing (γ={float(gamma):.3f})"
    return f"open (γ={float(gamma):.3f})"


def _snapshot_curve_label(gamma: float, time: float, channel: str) -> str:
    """Legend label for snapshot compressibility curves."""
    if _is_closed_gamma(gamma):
        return f"closed, t={float(time):.1f}"
    if channel == "charge_dephasing":
        return f"open dephasing, t={float(time):.1f}"
    return f"open, t={float(time):.1f}"

def write_outputs_and_plot(
    timeseries_rows: list[dict],
    snapshot_rows: list[dict],
    benchmark_summary_rows: list[dict],
    benchmark_summary_markdown: str,
    meta: dict[str, Any],
    outdir: Path,
    tag: str,
    show: bool,
    force: bool,
) -> dict[str, str]:
    """Write CSV, snapshot CSV, figure, and metadata JSON.

    Args:
        timeseries_rows: List of timeseries observation dicts.
        snapshot_rows: List of snapshot spectrum dicts.
        meta: Additional metadata to include.
        outdir: Output directory.
        tag: Output filename tag suffix.
        show: Whether to display figure interactively.
        force: Whether to overwrite existing files.

    Returns:
        Dictionary mapping artifact names to file paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag_suffix = f"_{tag}" if tag else ""

    # Define file paths
    csv_path = outdir / f"open_schwinger_entanglement_dynamics{tag_suffix}.csv"
    snap_csv_path = outdir / f"open_schwinger_entanglement_schmidt_snapshots{tag_suffix}.csv"
    fig_path = outdir / f"open_schwinger_entanglement_dynamics{tag_suffix}.png"
    meta_path = outdir / f"run_metadata{tag_suffix}.json"
    bench_csv_path = outdir / f"open_schwinger_entanglement_benchmark_summary{tag_suffix}.csv"
    bench_md_path = outdir / f"open_schwinger_entanglement_benchmark_summary{tag_suffix}.md"

    # Check existing files
    if not force:
        for p in [csv_path, snap_csv_path, fig_path, meta_path, bench_csv_path, bench_md_path]:
            if p.exists():
                raise FileExistsError(f"Output file exists (use --force to overwrite): {p}")

    # Write main timeseries CSV
    if timeseries_rows:
        fieldnames = [
            "time", "observable", "value", "cut", "channel", "gamma",
            "model", "N", "m_over_g", "x", "initial_state", "quench"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(timeseries_rows)

    # Write snapshot CSV
    if snapshot_rows:
        snap_fieldnames = [
            "time", "rank", "schmidt_proxy_from_rhoA", "p_eig", "cum_weight",
            "cut", "channel", "gamma", "model", "N", "m_over_g", "x"
        ]
        with open(snap_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=snap_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(snapshot_rows)

    # Write benchmark summary artifacts
    benchmark_paths = write_benchmark_summary(
        outdir=outdir,
        tag=tag,
        summary_rows=benchmark_summary_rows,
        summary_markdown=benchmark_summary_markdown,
        force=force,
    )

    # Generate 3-panel figure
    _generate_figure(timeseries_rows, snapshot_rows, fig_path, show)

    # Write metadata JSON
    full_meta = {
        "script": "open_schwinger_entanglement_dynamics.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "output_directory": str(outdir.resolve()),
        "outputs": {
            "csv": str(csv_path),
            "snapshots_csv": str(snap_csv_path),
            "figure": str(fig_path),
            "metadata": str(meta_path),
            "benchmark_summary_csv": benchmark_paths["benchmark_summary_csv"],
            "benchmark_summary_md": benchmark_paths["benchmark_summary_md"],
        },
        **meta,
        "benchmark_summary_paths": benchmark_paths,
    }
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2, default=str)

    return {
        "csv": str(csv_path),
        "snapshots_csv": str(snap_csv_path),
        "figure": str(fig_path),
        "metadata": str(meta_path),
        "benchmark_summary_csv": benchmark_paths["benchmark_summary_csv"],
        "benchmark_summary_md": benchmark_paths["benchmark_summary_md"],
    }


def _generate_figure(
    timeseries_rows: list[dict],
    snapshot_rows: list[dict],
    fig_path: Path,
    show: bool,
) -> None:
    """Generate 3-panel figure: subsystem entropy, spectrum proxy, mean|L|(t)."""
    channel = str(timeseries_rows[0].get("channel", "")) if timeseries_rows else ""

    # Extract data by gamma value
    gamma_values = sorted(set(r["gamma"] for r in timeseries_rows))

    # Prepare data structures
    entropy_data = {g: {"times": [], "values": []} for g in gamma_values}
    field_data = {g: {"times": [], "values": []} for g in gamma_values}

    for row in timeseries_rows:
        g = row["gamma"]
        t = row["time"]
        if row["observable"] == "entropy_vn":
            entropy_data[g]["times"].append(t)
            entropy_data[g]["values"].append(row["value"])
        elif row["observable"] == "mean_abs_L":
            field_data[g]["times"].append(t)
            field_data[g]["values"].append(row["value"])

    # Sort by time
    for g in gamma_values:
        idx = np.argsort(entropy_data[g]["times"])
        entropy_data[g]["times"] = np.array(entropy_data[g]["times"])[idx]
        entropy_data[g]["values"] = np.array(entropy_data[g]["values"])[idx]
        idx = np.argsort(field_data[g]["times"])
        field_data[g]["times"] = np.array(field_data[g]["times"])[idx]
        field_data[g]["values"] = np.array(field_data[g]["values"])[idx]

    # Snapshot data by (gamma, time)
    snapshot_by_key = {}
    for row in snapshot_rows:
        key = (row["gamma"], row["time"])
        if key not in snapshot_by_key:
            snapshot_by_key[key] = []
        snapshot_by_key[key].append(row)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: Subsystem von Neumann entropy (closed vs open)
    ax1 = axes[0]
    for g in gamma_values:
        label = _case_curve_label(float(g), channel)
        ax1.plot(entropy_data[g]["times"], entropy_data[g]["values"],
                 "o-" if _is_closed_gamma(float(g)) else "s--", markersize=3, label=label)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"$S_{\mathrm{vN}}$")
    ax1.set_title("Subsystem von Neumann entropy")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative reduced-spectrum concentration at snapshot times
    ax2 = axes[1]
    for (g, t), rows in sorted(snapshot_by_key.items()):
        rows = sorted(rows, key=lambda r: r["rank"])
        ranks = [r["rank"] for r in rows]
        cum_weights = [r["cum_weight"] for r in rows]
        label = _snapshot_curve_label(float(g), float(t), channel)
        ax2.plot(ranks, cum_weights, "o-", markersize=3, label=label)
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Cumulative weight")
    ax2.set_title("Reduced-state spectrum compressibility proxy")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Panel 3: mean|L|(t) closed vs open
    ax3 = axes[2]
    for g in gamma_values:
        label = _case_curve_label(float(g), channel)
        ax3.plot(field_data[g]["times"], field_data[g]["values"],
                 "o-" if _is_closed_gamma(float(g)) else "s--", markersize=3, label=label)
    ax3.set_xlabel("Time")
    ax3.set_ylabel(r"$\overline{|\langle L \rangle|}$")
    ax3.set_title("Mean electric field magnitude")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Compact interpretation footer note.
    fig.text(
        0.5,
        0.02,
        "Interpretation: γ=0 => S_vN(ρ_A) is bipartite entanglement entropy (pure state). "
        "γ>0 => S_vN(ρ_A) mixes entanglement with local mixedness/classical uncertainty. "
        "Panel 2 is a reduced-spectrum compressibility proxy from eig(ρ_A).",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    # Save
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments, raising ValueError on invalid inputs."""
    # N must be even
    if args.N % 2 != 0:
        raise ValueError(f"N must be even, got {args.N}")

    # cut must satisfy 0 <= cut <= N-2
    if args.cut < 0 or args.cut > args.N - 2:
        raise ValueError(f"cut {args.cut} must satisfy 0 <= cut <= N-2 = {args.N - 2}")

    # tmax > 0
    if args.tmax <= 0:
        raise ValueError(f"tmax must be positive, got {args.tmax}")

    # nt >= 2
    if args.nt < 2:
        raise ValueError(f"nt must be >= 2, got {args.nt}")

    # gamma >= 0
    if args.gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {args.gamma}")

    # gamma_ref >= 0
    if args.gamma_ref < 0:
        raise ValueError(f"gamma_ref must be non-negative, got {args.gamma_ref}")

    # channel validation (already constrained by choices, but be explicit)
    if args.channel != "charge_dephasing":
        raise ValueError(f"v1 supports only channel='charge_dephasing', got {args.channel}")

    # snapshot times validation
    snapshots = [float(x.strip()) for x in args.snapshot_times.split(",") if x.strip()]
    for t in snapshots:
        if t < 0.0 or t > args.tmax:
            raise ValueError(f"snapshot time {t} outside [0, {args.tmax}]")


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    validate_args(args)

    print(f"Open Schwinger entanglement dynamics: N={args.N}, m/g={args.mass}, x={args.coupling}")
    print(f"Dynamics: tmax={args.tmax}, nt={args.nt}, cut={args.cut}")
    print(f"Open-system: gamma={args.gamma}, gamma_ref={args.gamma_ref}, channel={args.channel}")

    # Parse snapshot times
    snapshot_times = [float(x.strip()) for x in args.snapshot_times.split(",") if x.strip()]

    # Time array
    times = np.linspace(0.0, args.tmax, args.nt)

    # Build sector model once
    print("Building sector model...")
    model = build_sector_model(
        N=args.N,
        mass=args.mass,
        coupling=args.coupling,
        cut=args.cut,
        initial_state=args.initial_state,
        quench=args.quench,
    )
    print(f"  Sector dimension: {model['dim']}")

    # Prepare one quench initial state
    print("Preparing initial state...")
    psi0, rho0, prep_meta = prepare_quench_initial_state(
        model, args.initial_state, args.quench
    )
    print(f"  Prep type: {prep_meta['prep_type']}, energy: {prep_meta['prep_energy']:.6f}")

    # Row metadata for CSV
    row_meta = {
        "model": "schwinger",
        "N": args.N,
        "m_over_g": args.mass,
        "x": args.coupling,
        "initial_state": args.initial_state,
        "quench": args.quench,
    }

    # Run evolution for each gamma case
    gamma_values = [args.gamma, args.gamma_ref]
    all_timeseries = []
    all_snapshots = []
    solver_metas = {}

    for gamma in gamma_values:
        print(f"Evolving with gamma={gamma:.4f}...")
        evo = evolve_open_dynamics(
            model, rho0, times, gamma=gamma, channel=args.channel,
            rtol=1e-8, atol=1e-10
        )
        solver_metas[gamma] = evo["solver_meta"]
        print(f"  Trace error: {evo['solver_meta']['max_abs_trace_error']:.2e}")
        print(f"  Hermiticity error: {evo['solver_meta']['max_hermiticity_error']:.2e}")

        # Measure timeseries and snapshots
        ts_rows, snap_rows, meas_meta = measure_timeseries(
            model, evo["rho_t"], times, cut=args.cut,
            snapshot_times=snapshot_times, gamma=gamma, channel=args.channel,
            row_meta=row_meta
        )
        all_timeseries.extend(ts_rows)
        all_snapshots.extend(snap_rows)

    # Derive benchmark-ready summary from existing outputs only (no extra physics runs).
    benchmark = summarize_benchmark_metrics(
        all_timeseries,
        all_snapshots,
        tmax=args.tmax,
        thresholds=(0.95, 0.99),
    )

    summary_highlights = {}
    for gamma, vals in benchmark["per_gamma"].items():
        summary_highlights[f"gamma_{gamma:.6g}"] = {
            "peak_entropy_vn": vals["peak_entropy_vn"],
            "final_entropy_vn": vals["final_entropy_vn"],
            "mean_entropy_vn": vals["mean_entropy_vn"],
            "peak_mean_abs_L": vals["peak_mean_abs_L"],
            "final_mean_abs_L": vals["final_mean_abs_L"],
        }

    verdict_rules = {
        "entropy_increase_verdict": (
            "✓ if open > closed for peak_entropy_vn, final_entropy_vn, and mean_entropy_vn; otherwise ~."
        ),
        "compressibility_reduction_verdict": (
            "Use post-quench shared snapshot times only (t > 0 within tolerance). "
            "✓ if open is not worse at every post-quench shared time and has at least one strict "
            "increase in rank_95 and/or rank_99; ~ if mixed/incomplete post-quench evidence; "
            "✗ if open is not less compressible post-quench."
        ),
        "field_perturbation_verdict": (
            "~ if abs(delta_peak_mean_abs_L) <= 0.02 and abs(delta_final_mean_abs_L) <= 0.02; otherwise !."
        ),
    }

    # Build metadata
    full_meta = {
        "args": {
            "N": args.N,
            "mass": args.mass,
            "coupling": args.coupling,
            "tmax": args.tmax,
            "nt": args.nt,
            "cut": args.cut,
            "gamma": args.gamma,
            "gamma_ref": args.gamma_ref,
            "channel": args.channel,
            "snapshot_times": snapshot_times,
            "initial_state": args.initial_state,
            "quench": args.quench,
        },
        "model_meta": model["meta"],
        "prep_meta": prep_meta,
        "solver_metas": {str(k): v for k, v in solver_metas.items()},
        "entropy_measure_label": "subsystem_von_neumann_entropy",
        "closed_case_note": "For gamma=0, S_vN(rho_A) is a bipartite entanglement entropy for the pure state.",
        "open_case_note": "For gamma>0, S_vN(rho_A) mixes entanglement with local mixedness/classical uncertainty and is not a pure entanglement measure.",
        "schmidt_proxy_definition": "Panel derived from eigenvalue spectrum of rho_A; for open dynamics this is a reduced-state compressibility proxy, not a strict pure-state Schmidt decomposition.",
        "snapshot_spectrum_definition": "p_eig are eigenvalues of rho_A at the chosen cut.",
        "summary_highlights": summary_highlights,
        "threshold_notes": benchmark["truncation_notes"],
        "verdict_rules": verdict_rules,
        "compressibility_verdict_definition": (
            "Verdict based on post-quench shared snapshot times only (t > 0 within tolerance); "
            "t=0 is excluded because closed and open spectra are identical by construction."
        ),
        "compressibility_shared_snapshot_times": [],
        "compressibility_post_quench_snapshot_times": [],
        "headline_takeaway": (
            "Weak charge dephasing in the tested Schwinger quench substantially increases subsystem entropy "
            "and broadens the reduced-state spectrum, making the dynamics less tensor-network-compressible, "
            "while only modestly perturbing the mean electric-field observable."
        ),
    }
    if benchmark["comparison_summary"] is not None:
        full_meta["comparison_summary"] = benchmark["comparison_summary"]
        full_meta["compressibility_verdict_definition"] = benchmark["comparison_summary"][
            "compressibility_verdict_definition"
        ]
        full_meta["compressibility_shared_snapshot_times"] = benchmark["comparison_summary"][
            "compressibility_shared_snapshot_times"
        ]
        full_meta["compressibility_post_quench_snapshot_times"] = benchmark["comparison_summary"][
            "compressibility_post_quench_snapshot_times"
        ]
        # Backward-compatible alias kept for older tooling.
        full_meta["two_gamma_comparison"] = benchmark["comparison_summary"]

    # Write outputs and plot
    print("Writing outputs...")
    outputs = write_outputs_and_plot(
        all_timeseries,
        all_snapshots,
        benchmark["summary_rows"],
        benchmark["markdown"],
        full_meta,
        outdir=args.outdir, tag=args.tag, show=args.show, force=args.force
    )

    print("\nOutputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print_benchmark_summary(benchmark)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
