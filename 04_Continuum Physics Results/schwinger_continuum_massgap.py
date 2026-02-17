#!/usr/bin/env python3
"""
Massless Schwinger model mass gap (ED): finite-size control + continuum extrapolation.

Changes (per review comments):
  - Report BOTH linear and quadratic 1/N extrapolations (diagnostic).
  - More conservative model selection: avoid quadratic overfitting that pulls M_inf down systematically.
  - Add sensitivity scan over max_shift_frac thresholds without recomputing eigenvalues.
  - Make it easy to test continuum-fit robustness via --fit_ag2_max.

Conventions:
(A) x = 1/(ag)^2          ->  ga=1/sqrt(x),    (ag)^2=1/x
(B) x = 1/(2(ag)^2)       ->  ga=1/sqrt(2x),   (ag)^2=1/(2x)

Energy conversion:
  M_gap/g = ΔE_tilde * (ga)/2
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# -----------------------------
# Basis utilities (fixed weight)
# -----------------------------

def generate_basis_fixed_weight(N: int, n_ones: int) -> np.ndarray:
    if N <= 0 or n_ones < 0 or n_ones > N:
        raise ValueError("Invalid N or n_ones")
    if N > 63:
        raise ValueError("N too large for uint64 bit basis (max 63).")

    if n_ones == 0:
        return np.array([0], dtype=np.uint64)
    if n_ones == N:
        return np.array([(1 << N) - 1], dtype=np.uint64)

    x = (1 << n_ones) - 1
    limit = 1 << N
    states: List[int] = []
    while x < limit:
        states.append(x)
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r
    return np.array(states, dtype=np.uint64)


def bit_at(state: int, n: int) -> int:
    return (state >> n) & 1


def sigma_z_from_bit(b: int) -> int:
    return 1 if b == 1 else -1


# -----------------------------------------
# Conventions for x <-> (ag)^2 and ga
# -----------------------------------------

def ag2_from_x(x: float, x_def: str) -> float:
    if x_def == "1_over_ag2":
        return 1.0 / x
    if x_def == "1_over_2ag2":
        return 1.0 / (2.0 * x)
    raise ValueError(f"Unknown x_def: {x_def}")


def ga_from_x(x: float, x_def: str) -> float:
    if x_def == "1_over_ag2":
        return 1.0 / math.sqrt(x)
    if x_def == "1_over_2ag2":
        return 1.0 / math.sqrt(2.0 * x)
    raise ValueError(f"Unknown x_def: {x_def}")


# -----------------------------------------
# Gauge-eliminated Schwinger Hamiltonian ED
# -----------------------------------------

@dataclass(frozen=True)
class SchwingerParams:
    N: int
    x: float
    x_def: str
    m_over_g: float = 0.0
    E0: float = 0.0
    half_filling: bool = True


def build_hamiltonian_sector(params: SchwingerParams) -> Tuple[csr_matrix, np.ndarray]:
    N, x, m_over_g, E0, x_def = params.N, params.x, params.m_over_g, params.E0, params.x_def
    if N % 2 != 0:
        raise ValueError("Use even N for half-filling sector.")
    if x <= 0:
        raise ValueError("x must be positive.")

    n_ones = N // 2
    basis = generate_basis_fixed_weight(N, n_ones)
    index: Dict[int, int] = {int(s): i for i, s in enumerate(basis)}

    ga = ga_from_x(x, x_def)
    mu = 2.0 * m_over_g / ga

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    stag = np.array([1 if (n % 2 == 0) else -1 for n in range(N)], dtype=int)

    for i, s_u64 in enumerate(basis):
        s = int(s_u64)

        L = E0
        diag_electric = 0.0
        diag_mass = 0.0

        for n in range(N):
            b = bit_at(s, n)
            z = sigma_z_from_bit(b)

            diag_mass += 0.5 * mu * (stag[n] * z)

            qn = 0.5 * (z + stag[n])
            L += qn
            if n <= N - 2:
                diag_electric += L * L

        rows.append(i); cols.append(i); data.append(diag_electric + diag_mass)

        for n in range(N - 1):
            bn = bit_at(s, n)
            bn1 = bit_at(s, n + 1)
            if bn != bn1:
                s2 = s ^ (1 << n) ^ (1 << (n + 1))
                j = index.get(s2, None)
                if j is not None:
                    rows.append(i); cols.append(j); data.append(x)

    dim = len(basis)
    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)
    H = 0.5 * (H + H.T)
    return H.tocsr(), basis


def lowest_energies(H: csr_matrix, k: int = 6) -> np.ndarray:
    evals = eigsh(
        H, k=max(2, k), which="SA", return_eigenvectors=False,
        tol=1e-10, maxiter=2_000_000
    )
    return np.sort(np.real(evals))


def mass_gap_over_g_from_evals(evals: np.ndarray, x: float, x_def: str) -> Tuple[float, float, float]:
    E0 = float(evals[0])
    E1 = None
    for val in evals[1:]:
        if val > E0 + 1e-10:
            E1 = float(val)
            break
    if E1 is None:
        E1 = float(evals[1])

    gap_tilde = E1 - E0
    ga = ga_from_x(x, x_def)
    mg_over_g = gap_tilde * (ga / 2.0)
    return E0, E1, mg_over_g


# ---------------------------------
# Finite-size data + model fits
# ---------------------------------

def compute_gap_vs_N(
    x: float,
    N_list: List[int],
    x_def: str,
    m_over_g: float,
    E0: float,
    k_eigs: int,
    verbose: bool,
) -> List[Tuple[int, float, float, float]]:
    out = []
    for N in N_list:
        params = SchwingerParams(N=N, x=x, x_def=x_def, m_over_g=m_over_g, E0=E0)
        H, _ = build_hamiltonian_sector(params)
        evals = lowest_energies(H, k=k_eigs)
        E0t, E1t, mg = mass_gap_over_g_from_evals(evals, x=x, x_def=x_def)
        out.append((N, E0t, E1t, mg))
        if verbose:
            print(f"x={x:>4g}, N={N:>2d}, dim={H.shape[0]:>8d}  "
                  f"E0~{E0t: .6f}  E1~{E1t: .6f}  M_gap/g~{mg: .6f}")
    return out


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    OLS y = X beta.
    Returns beta, rss, se_intercept, rms_resid.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    rss = float(np.sum(resid**2))
    n, p = X.shape
    dof = max(1, n - p)
    s2 = rss / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = s2 * XtX_inv
    se0 = float(math.sqrt(max(cov[0, 0], 0.0)))
    rms = float(math.sqrt(rss / dof))
    return beta, rss, se0, rms


def fit_1overN_models(data: List[Tuple[int, float, float, float]], n_fit_points: int) -> Dict[str, Dict[str, float]]:
    tail = data[-min(n_fit_points, len(data)):]
    Ns = np.array([t[0] for t in tail], dtype=float)
    Ms = np.array([t[3] for t in tail], dtype=float)
    x1 = 1.0 / Ns

    out: Dict[str, Dict[str, float]] = {}

    X_lin = np.vstack([np.ones_like(x1), x1]).T
    beta, rss, se0, rms = _ols_fit(X_lin, Ms)
    n, p = X_lin.shape
    aic = (n * math.log(rss / n) + 2 * p) if rss > 0 else -np.inf
    out["linear"] = dict(M_inf=float(beta[0]), rss=rss, aic=float(aic), se=float(se0), rms=float(rms))

    X_q = np.vstack([np.ones_like(x1), x1, x1**2]).T
    beta, rss, se0, rms = _ols_fit(X_q, Ms)
    n, p = X_q.shape
    aic = (n * math.log(rss / n) + 2 * p) if rss > 0 else -np.inf
    out["quadratic"] = dict(M_inf=float(beta[0]), rss=rss, aic=float(aic), se=float(se0), rms=float(rms))

    return out


def choose_M_used(
    data: List[Tuple[int, float, float, float]],
    tol_rel: float,
    N_extrap_mode: str,
    n_fit_points: int,
    max_shift_frac: float,
    prefer_linear_if_quadratic_pulls_down: bool,
    verbose: bool,
) -> Tuple[float, float, str, Dict[str, Dict[str, float]]]:
    """
    Returns (M_used, M_err, method, fit_diag)
    method in {"stabilized","N_extrap_lin","N_extrap_quad","maxN_unstable"}.
    fit_diag always contains both model fits (if used).
    """
    M_last = float(data[-1][3])

    # Stabilization check based on last-step drift
    if len(data) >= 2:
        M_prev = float(data[-2][3])
        rel = abs((M_last - M_prev) / M_prev) if abs(M_prev) > 1e-14 else np.inf
        if verbose:
            print(f"  final-step rel change: {100*rel:.2f}% (tol={100*tol_rel:.2f}%)")
        if rel < tol_rel:
            return M_last, abs(M_last - M_prev), "stabilized", {}

    if N_extrap_mode == "off":
        err = abs(M_last - float(data[-2][3])) if len(data) >= 2 else 0.0
        return M_last, max(err, 0.0), "maxN_unstable", {}

    fits = fit_1overN_models(data, n_fit_points=n_fit_points)

    # pick model
    if N_extrap_mode == "linear":
        pick = "linear"
    elif N_extrap_mode == "quadratic":
        pick = "quadratic"
    else:
        pick = "quadratic" if fits["quadratic"]["aic"] < fits["linear"]["aic"] else "linear"

    # heuristic guard: quadratic often overfits tail noise and pulls M_inf down
    if prefer_linear_if_quadratic_pulls_down and pick == "quadratic":
        Mq = fits["quadratic"]["M_inf"]
        Ml = fits["linear"]["M_inf"]
        # if quadratic is noticeably lower than linear and not strongly justified by AIC, prefer linear
        if (Mq < Ml - 0.01) and (fits["linear"]["aic"] - fits["quadratic"]["aic"] < 2.0):
            pick = "linear"

    cand = float(fits[pick]["M_inf"])
    shift = abs(cand - M_last)
    se0 = float(fits[pick]["se"])
    rms = float(fits[pick]["rms"])

    unstable = False
    if shift > max_shift_frac * max(abs(M_last), 1e-12):
        unstable = True
    if se0 > max_shift_frac * max(abs(M_last), 1e-12):
        unstable = True

    if unstable:
        # fall back to maxN with honest error
        err1 = abs(M_last - float(data[-2][3])) if len(data) >= 2 else 0.0
        tail = np.array([t[3] for t in data[-min(4, len(data)):]], dtype=float)
        err2 = float(np.std(tail))
        err = max(err1, err2)
        if verbose:
            print(f"  N->∞ extrapolation UNSTABLE (shift={shift:.4f}, se={se0:.4f}). Using max-N with err~{err:.6f}.")
        return M_last, err, "maxN_unstable", fits

    # conservative error: max of shift, SE, RMS
    M_err = max(shift, se0, rms)
    tag = "N_extrap_lin" if pick == "linear" else "N_extrap_quad"
    if verbose:
        print(f"  using {tag}: M_inf={cand:.6f}, shift={shift:.6f}, se={se0:.6f}, rms={rms:.6f}, err={M_err:.6f}")
        print(f"    diag: M_lin={fits['linear']['M_inf']:.6f}, M_quad={fits['quadratic']['M_inf']:.6f}")
    return cand, M_err, tag, fits


def stabilized_or_extrapolated_mass_gap(
    x: float,
    N_list: List[int],
    x_def: str,
    tol_rel: float,
    m_over_g: float,
    E0: float,
    k_eigs: int,
    N_extrap_mode: str,
    N_fit_points: int,
    max_shift_frac: float,
    prefer_linear_if_quadratic_pulls_down: bool,
    verbose: bool,
) -> Tuple[int, float, float, float, float, str, Dict[str, Dict[str, float]]]:
    data = compute_gap_vs_N(x, N_list, x_def, m_over_g, E0, k_eigs, verbose)

    Nmax = int(data[-1][0])
    if verbose and Nmax < int(6 * math.sqrt(max(x, 1.0))):
        print(f"  WARNING: N_max={Nmax} may be too small for x={x} (sqrt(x)~{math.sqrt(x):.2f}). "
              f"Consider adding N=22,24...")

    M_used, M_err, method, fit_diag = choose_M_used(
        data=data,
        tol_rel=tol_rel,
        N_extrap_mode=N_extrap_mode,
        n_fit_points=N_fit_points,
        max_shift_frac=max_shift_frac,
        prefer_linear_if_quadratic_pulls_down=prefer_linear_if_quadratic_pulls_down,
        verbose=verbose,
    )

    N, E0t, E1t, _M_last = data[-1]
    return int(N), float(E0t), float(E1t), float(M_used), float(M_err), method, fit_diag


# -------------------------
# Continuum extrapolation
# -------------------------

def weighted_polyfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, deg: int) -> np.ndarray:
    V = np.vander(x, deg + 1)
    W = np.diag(w)
    coef, *_ = np.linalg.lstsq(W @ V, W @ y, rcond=None)
    return coef


def continuum_extrapolation_plot(
    xs: List[float],
    results: Dict[float, Tuple[int, float, float, float, float, str]],
    x_def: str,
    fit_order: int,
    fit_ag2_max: Optional[float],
    out_png: str,
    out_csv: str,
):
    ag2_all = np.array([ag2_from_x(x, x_def) for x in xs], dtype=float)
    M_all = np.array([results[x][3] for x in xs], dtype=float)
    Merr_all = np.array([max(results[x][4], 1e-6) for x in xs], dtype=float)
    N_used = np.array([results[x][0] for x in xs], dtype=int)
    method = [results[x][5] for x in xs]

    mask = np.ones_like(ag2_all, dtype=bool)
    if fit_ag2_max is not None:
        mask &= (ag2_all <= fit_ag2_max)

    ag2 = ag2_all[mask]
    M = M_all[mask]
    Merr = Merr_all[mask]
    w = 1.0 / (Merr**2)

    if len(ag2) < fit_order + 1:
        raise ValueError("Not enough points for continuum fit; add more x or relax fit_ag2_max.")

    coef = weighted_polyfit(ag2, M, w, deg=fit_order)
    poly = np.poly1d(coef)
    intercept = float(poly(0.0))

    ag2_fit = np.linspace(0.0, ag2_all.max() * 1.05, 250)
    M_fit = poly(ag2_fit)

    target = 1.0 / math.sqrt(math.pi)

    # CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("x,ag2,N_used,E0_tilde,E1_tilde,M_used,M_err,method\n")
        for x in xs:
            N, E0t, E1t, Mu, Me, meth = results[x]
            f.write(f"{x},{ag2_from_x(x, x_def)},{N},{E0t},{E1t},{Mu},{Me},{meth}\n")

    # Plot
    plt.figure(figsize=(7.2, 4.8), dpi=150)
    ax = plt.gca()

    ax.errorbar(ag2_all, M_all, yerr=Merr_all, fmt="o", capsize=3, label="ED (N control)")
    for xi, yi, Ni, meth in zip(ag2_all, M_all, N_used, method):
        tag = f"N={Ni}"
        if "N_extrap" in meth:
            tag += "*"
        if meth == "maxN_unstable":
            tag += "†"
        ax.annotate(tag, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=10)

    fit_label = "Linear" if fit_order == 1 else f"Poly(deg={fit_order})"
    ax.plot(ag2_fit, M_fit, "-", label=f"{fit_label} fit: intercept = {intercept:.4f}")
    ax.axhline(target, linestyle="--", linewidth=1.5,
               label=rf"Theory: $1/\sqrt{{\pi}}\approx{target:.4f}$")

    ax.set_xlabel(r"$(a g)^2$")
    ax.set_ylabel(r"$M_{\rm gap}/g$")
    ax.set_title("Massless Schwinger model mass gap: continuum extrapolation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    ax.set_xlim(left=0.0)

    conv_note = "x = 1/(ag)^2" if x_def == "1_over_ag2" else "x = 1/(2 (ag)^2)"
    fit_note = ""
    if fit_ag2_max is not None:
        fit_note = f"\nfit uses (ag)^2 ≤ {fit_ag2_max}"
    ax.text(0.02, 0.03, conv_note + "\n(* = N→∞ extrap)\n(† = max-N upper bound)" + fit_note,
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    used_x = [x for x, m in zip(xs, mask) if m]
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_csv}")
    print(f"Convention: {conv_note}")
    print(f"Continuum fit used x = {used_x}")
    print(f"Continuum intercept (ag^2->0): {intercept:.6f}")
    print(f"Theory 1/sqrt(pi):             {target:.6f}")
    print(f"Absolute error:               {abs(intercept-target):.6f}")


# -------------
# Main program
# -------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Massless Schwinger ED mass gap continuum extrapolation.")

    p.add_argument("--x", nargs="+", type=float,
                   default=[2, 3, 4, 5, 6, 8, 10, 12],
                   help="Numeric x values used in the Hamiltonian.")
    p.add_argument("--N", nargs="+", type=int,
                   default=[8, 10, 12, 14, 16, 18, 20],
                   help="Even N values to evaluate.")

    p.add_argument("--x_def", choices=["1_over_ag2", "1_over_2ag2"], default="1_over_2ag2",
                   help="Convention relating x to (ag)^2.")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Stabilization tolerance based on final-step relative change.")
    p.add_argument("--k_eigs", type=int, default=6, help="How many lowest eigenvalues to compute.")

    p.add_argument("--N_extrap_mode", choices=["off", "linear", "quadratic", "auto"], default="auto",
                   help="Finite-size extrapolation model if stabilization fails.")
    p.add_argument("--N_fit_points", type=int, default=6,
                   help="How many largest-N points to use for 1/N fits (>=6 recommended if you include N=22).")
    p.add_argument("--prefer_linear_if_quadratic_pulls_down", action="store_true", default=True,
                   help="If quadratic pulls M_inf notably below linear without strong AIC support, prefer linear.")
    p.add_argument("--max_shift_frac", type=float, default=0.25,
                   help="If N->∞ extrap shifts by > this fraction (or SE is huge), treat as unstable and fall back.")

    p.add_argument("--fit_order", type=int, default=1, choices=[1, 2],
                   help="Continuum fit order in (ag)^2.")
    p.add_argument("--fit_ag2_max", type=float, default=None,
                   help="Include only points with (ag)^2 <= this in continuum fit (robustness test).")

    # New: scan max_shift_frac thresholds (no recomputation; uses same per-x data)
    p.add_argument("--sensitivity_max_shift_frac", nargs="*", type=float, default=None,
                   help="Optional list, e.g. 0.15 0.35, to rerun selection+continuum fit with different thresholds.")

    p.add_argument("--report_all_Nfits", action="store_true",
                   help="Print per-x table including linear/quadratic 1/N fits and chosen method.")

    p.add_argument("--quiet", action="store_true", help="Reduce console output.")
    p.add_argument("--out_png", type=str, default="Figure4_massgap_continuum.png", help="Output figure filename.")
    p.add_argument("--out_csv", type=str, default="massgap_results.csv", help="Output CSV filename.")
    return p.parse_args()


def main():
    args = parse_args()
    xs = sorted(list(args.x))
    N_list = sorted(set(list(args.N)))
    for N in N_list:
        if N % 2 != 0:
            raise ValueError(f"N must be even (got {N}).")

    x_def = args.x_def
    verbose = not args.quiet

    conv_note = "x = 1/(ag)^2" if x_def == "1_over_ag2" else "x = 1/(2 (ag)^2)"
    if verbose:
        print(f"Using convention: {conv_note}")
        print(f"N grid: {N_list}")
        print(f"x grid: {xs}")
        print(f"Finite-size extrap mode: {args.N_extrap_mode}  (prefer_linear_guard={args.prefer_linear_if_quadratic_pulls_down})")
        print(f"max_shift_frac: {args.max_shift_frac}")
        if args.fit_ag2_max is not None:
            print(f"Continuum fit cutoff: fit_ag2_max={args.fit_ag2_max}")

    m_over_g = 0.0
    E0 = 0.0

    # Store full per-x fit diagnostics so we can rerun selection with different max_shift_frac
    per_x_data: Dict[float, Dict[str, object]] = {}

    # First pass: compute data and choose using the given threshold
    results: Dict[float, Tuple[int, float, float, float, float, str]] = {}
    fit_diags: Dict[float, Dict[str, Dict[str, float]]] = {}

    for x in xs:
        # compute full M(N) data once
        data = compute_gap_vs_N(x, N_list, x_def, m_over_g, E0, args.k_eigs, verbose)
        per_x_data[x] = {"data": data}

        # choose M_used with threshold
        M_used, M_err, method, diag = choose_M_used(
            data=data,
            tol_rel=float(args.tol),
            N_extrap_mode=args.N_extrap_mode,
            n_fit_points=int(args.N_fit_points),
            max_shift_frac=float(args.max_shift_frac),
            prefer_linear_if_quadratic_pulls_down=bool(args.prefer_linear_if_quadratic_pulls_down),
            verbose=verbose,
        )
        N, E0t, E1t, _ = data[-1]
        results[x] = (int(N), float(E0t), float(E1t), float(M_used), float(M_err), method)
        fit_diags[x] = diag

    if args.report_all_Nfits:
        print("\nPer-x finite-size diagnostics (tail fits):")
        hdr = "x   ag^2     M(Nmax)   M_lin    M_quad   AIC_lin  AIC_quad  used"
        print(hdr)
        print("-" * len(hdr))
        for x in xs:
            data = per_x_data[x]["data"]
            Mmax = float(data[-1][3])
            ag2 = ag2_from_x(x, x_def)
            if fit_diags[x]:
                Ml = fit_diags[x]["linear"]["M_inf"]
                Mq = fit_diags[x]["quadratic"]["M_inf"]
                Alic = fit_diags[x]["linear"]["aic"]
                Aqc = fit_diags[x]["quadratic"]["aic"]
            else:
                Ml = float("nan"); Mq = float("nan"); Alic = float("nan"); Aqc = float("nan")
            used = results[x][5]
            print(f"{x:<3g} {ag2:7.4f}  {Mmax:8.4f}  {Ml:7.4f}  {Mq:7.4f}  {Alic:7.2f}  {Aqc:7.2f}  {used}")

        print("\nNote: † points (maxN_unstable) are best interpreted as upper bounds at current N_max.")

    # Save main outputs
    continuum_extrapolation_plot(
        xs=xs,
        results=results,
        x_def=x_def,
        fit_order=int(args.fit_order),
        fit_ag2_max=args.fit_ag2_max,
        out_png=args.out_png,
        out_csv=args.out_csv,
    )

    # Sensitivity scan: rerun selection + continuum fit without recomputing eigenvalues
    if args.sensitivity_max_shift_frac:
        for thr in args.sensitivity_max_shift_frac:
            if verbose:
                print(f"\n--- Sensitivity scan: max_shift_frac={thr} ---")
            results_thr: Dict[float, Tuple[int, float, float, float, float, str]] = {}
            for x in xs:
                data = per_x_data[x]["data"]
                M_used, M_err, method, _diag = choose_M_used(
                    data=data,
                    tol_rel=float(args.tol),
                    N_extrap_mode=args.N_extrap_mode,
                    n_fit_points=int(args.N_fit_points),
                    max_shift_frac=float(thr),
                    prefer_linear_if_quadratic_pulls_down=bool(args.prefer_linear_if_quadratic_pulls_down),
                    verbose=False,
                )
                N, E0t, E1t, _ = data[-1]
                results_thr[x] = (int(N), float(E0t), float(E1t), float(M_used), float(M_err), method)

            out_png = args.out_png.replace(".png", f"_thr{thr:.2f}.png")
            out_csv = args.out_csv.replace(".csv", f"_thr{thr:.2f}.csv")
            continuum_extrapolation_plot(
                xs=xs,
                results=results_thr,
                x_def=x_def,
                fit_order=int(args.fit_order),
                fit_ag2_max=args.fit_ag2_max,
                out_png=out_png,
                out_csv=out_csv,
            )


if __name__ == "__main__":
    main()
