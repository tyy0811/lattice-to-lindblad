"""Gating physics-validation tests V1–V4 for Module 1.

If any test in this file fails, Module 1 is not complete. Do not loosen
tolerances; debug the implementation. See Module 1 spec §4 for the
tolerance rationale.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, TruncationParams
from dispersive_readout.physics.transmon import diagonalize_transmon, transmon_summary

_TWO_PI = 2.0 * math.pi


# -- V1: transmon eigenstructure ----------------------------------------------

def test_V1a_transmon_anharmonicity_matches_next_order():
    """Transmon anharmonicity matches the next-order perturbative formula.

    Primary gate (1% tolerance): α ≈ -E_C · (1 + sqrt(E_C/E_J)). This
    captures both the leading Koch-2007 α = -E_C limit and the dominant
    higher-order correction from the sextic term in the cos(φ) expansion.
    Numerically matches exact diagonalization to ~0.1% for E_J/E_C ≥ 50
    (verified at ratios 50, 74, 100, 200, 500). Cross-checked with
    brute-force N_charge=401 to rule out truncation artefacts.

    Deviation from spec §4 V1 (originally 5% against -E_C leading-order):
    at E_J/E_C ≈ 74 the leading-order gap is an inherent ~11.5% from exact
    diagonalization, so a 5%-on-leading-order gate is physically infeasible
    regardless of implementation. The next-order gate here is tighter (1%)
    and also catches real regressions more reliably than a wide tolerance on
    a formula that is known to be 10%+ away from the true anharmonicity.

    Secondary sanity check (15% leading-order, non-gating): bounds the
    direction and order of magnitude of α against -E_C. Kept as belt-and-
    braces for catastrophic bugs (sign flip, off-by-factor).
    """
    import math as _math
    d = REFERENCE_DEVICE
    s = transmon_summary(d.transmon, d.truncation)
    alpha_numerical = s["alpha"]

    # Primary gate: next-order perturbative formula, 1% tolerance.
    alpha_next_order = -d.transmon.E_C * (1.0 + _math.sqrt(d.transmon.E_C / d.transmon.E_J))
    rel_err_next = abs(alpha_numerical - alpha_next_order) / abs(alpha_next_order)
    assert rel_err_next < 0.01, (
        f"V1a FAIL (next-order): alpha/2π numerical = {alpha_numerical/_TWO_PI/1e6:.3f} MHz, "
        f"-E_C·(1+sqrt(E_C/E_J)) = {alpha_next_order/_TWO_PI/1e6:.3f} MHz, "
        f"rel err = {rel_err_next:.3%}."
    )

    # Secondary (leading-order, 15% sanity band): catches sign/scale bugs.
    alpha_leading = -d.transmon.E_C
    rel_err_leading = abs(alpha_numerical - alpha_leading) / abs(alpha_leading)
    assert rel_err_leading < 0.15, (
        f"V1a sanity FAIL (leading-order): alpha/2π = {alpha_numerical/_TWO_PI/1e6:.2f} MHz, "
        f"-E_C = {alpha_leading/_TWO_PI/1e6:.2f} MHz, rel err = {rel_err_leading:.3%}."
    )


def test_V1b_transmon_charge_dispersion_below_1kHz():
    """In the deep transmon regime, |ω_01(n_g=0.5) − ω_01(n_g=0)| < 1 kHz.

    Also acts as a N_charge sufficiency check — if charge dispersion is
    artifactually large, the charge ladder is truncating too tightly.
    """
    d = REFERENCE_DEVICE
    s = transmon_summary(d.transmon, d.truncation)
    charge_dispersion_hz = s["charge_dispersion_01"] / _TWO_PI
    assert charge_dispersion_hz < 1e3, (
        f"V1b FAIL: charge dispersion of |0⟩–|1⟩ transition = "
        f"{charge_dispersion_hz:.1f} Hz, expected < 1000 Hz."
    )


# -- truncation convergence (non-gating but required before moving on) --------

def test_N_charge_convergence_below_1e_6_relative():
    """omega_01 must change by < 1e-6 (relative) when N_charge: default → default+20.

    Deviation from plan (originally 13 → 21): the REFERENCE_DEVICE default
    was bumped to N_charge=31 in Task 5 to satisfy the Koch 2007 criterion,
    so the convergence check is now 31 → 51. Both endpoints are well into
    the converged regime and omega_01 is stable to numerical precision.
    """
    d = REFERENCE_DEVICE
    default_N = d.truncation.N_charge
    trunc_small = TruncationParams(N_charge=default_N, N_transmon=5, N_resonator=15)
    trunc_large = TruncationParams(N_charge=default_N + 20, N_transmon=5, N_resonator=15)
    e_small, _ = diagonalize_transmon(d.transmon, trunc_small)
    e_large, _ = diagonalize_transmon(d.transmon, trunc_large)
    omega_01_small = e_small[1] - e_small[0]
    omega_01_large = e_large[1] - e_large[0]
    rel = abs(omega_01_large - omega_01_small) / abs(omega_01_small)
    assert rel < 1e-6, f"N_charge not converged at default={default_N}: rel change = {rel:.2e}"


def test_computational_manifold_charge_dispersion_small():
    """Computational-manifold levels (|0⟩, |1⟩, |2⟩) must have charge
    dispersion < 10 kHz for REFERENCE_DEVICE.

    These are the levels used as qubit/readout states by Module 1 and by
    the shelving-readout work in Modules 2–4. For REFERENCE_DEVICE, the
    measured per-level dispersions are (0, ~190 Hz, ~7.5 kHz) at n_g=0.5,
    all well below the 10 kHz bound.

    Deviation from the plan, which originally applied this bound to the
    *top kept* level (j = N_transmon - 1). At E_J/E_C ≈ 74 with N_transmon
    = 5 the upper two levels (|3⟩, |4⟩) have physically large dispersion
    (~0.2 MHz and ~3 MHz) even though they remain eigenstates of the full
    charge-basis Hamiltonian — their cosine-potential wavefunctions are
    strongly mixed across adjacent charge wells at those energies, which
    produces large n_g sensitivity by construction. They are retained as a
    completeness basis for the perturbative sums used by `dispersive_shift_full`
    and for the dressed-state identification in `dispersive_shift_from_simulation`,
    not as converged qubit states. Applying the 10 kHz gate to them would
    fail on correct physics.

    Scope note: this test is specific to REFERENCE_DEVICE's
    computational-manifold size (3 levels). If the reference device or
    truncation changes materially, reconsider the manifold size.
    """
    from dataclasses import replace
    d = REFERENCE_DEVICE
    energies_0, _ = diagonalize_transmon(d.transmon, d.truncation)
    energies_half, _ = diagonalize_transmon(replace(d.transmon, n_g=0.5), d.truncation)
    COMPUTATIONAL_MANIFOLD = range(0, 3)  # |0⟩, |1⟩, |2⟩
    for j in COMPUTATIONAL_MANIFOLD:
        dispersion_hz = abs(energies_half[j] - energies_0[j]) / _TWO_PI
        assert dispersion_hz < 10e3, (
            f"computational-manifold level j={j} dispersion = {dispersion_hz:.1f} Hz "
            f"(expected < 10 kHz)."
        )


# -- V2: dispersive shift numerical vs analytic --------------------------------

from dispersive_readout.physics.dispersive import (  # noqa: E402
    dispersive_shift_from_simulation,
    dispersive_shift_full,
)
from dispersive_readout.physics.transmon import charge_operator_matrix_elements  # noqa: E402


def test_V2_chi_analytic_vs_numerical_at_reference_device():
    """Multi-level 2nd-order analytic χ vs exact dressed-JC numerical at
    REFERENCE_DEVICE coupling.

    Deviation from spec §4 V2, which listed 1e-4 relative tolerance. The
    analytic formula dispersive_shift_full is 2nd-order perturbation theory
    in the full non-RWA coupling; the numerical extractor is exact
    diagonalization. At REFERENCE_DEVICE's g/2π = 120 MHz, Δ/2π = −2.42 GHz,
    the 3rd-order residual (g/Δ)² ≈ 0.25%, amplified by transmon matrix-
    element factors to ~1.3% observed. 1e-4 is physically infeasible at
    this coupling regardless of implementation — would require either a
    higher-order analytic formula (impractical) or weaker coupling.

    Tolerance 2% catches the bug-classes the plan intended V2 to catch:
      * MINUS-sign bug in dispersive_shift_full (gives ~80% error)
      * Wrong dressed-state overlap identification (≥ 100% error)
      * Index errors in the analytic sum (>> 10% error)
      * ω_q ↔ ω_k swaps (sign flip, >> 100% error)
    The tight 1e-4 version of V2 lives in
    test_V2_chi_analytic_converges_at_weak_coupling below — exercising the
    same formula at g/2π = 8 MHz where the 3rd-order residual is ~6e-5.
    """
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_per_level = dispersive_shift_full(
        energies, n_mat, d.coupling.g, d.resonator.omega_r,
    )
    chi_analytic_half = (chi_per_level[1] - chi_per_level[0]) / 2.0
    chi_numerical_half = dispersive_shift_from_simulation(d)
    rel_error = abs(chi_analytic_half - chi_numerical_half) / abs(chi_analytic_half)
    assert rel_error < 0.02, (
        f"V2 FAIL (REFERENCE_DEVICE): chi analytic/2π = "
        f"{chi_analytic_half/_TWO_PI/1e6:.4f} MHz, numerical/2π = "
        f"{chi_numerical_half/_TWO_PI/1e6:.4f} MHz, rel err = {rel_error:.2e}"
    )


def test_V2_chi_analytic_converges_at_weak_coupling():
    """Tight (1e-4) analytic-vs-numerical χ agreement at reduced coupling.

    Same formula, same numerics, same device — only g/2π reduced from
    120 MHz to 8 MHz, shrinking the 3rd-order residual by (8/120)² ≈ 4.4e-3.
    Observed relative error at g/2π = 8 MHz: ~6e-5, safely inside the 1e-4
    gate. Agreement here is the cleanest gate that the PLUS-sign 2nd-order
    analytic formula is algebraically correct; V2 at REFERENCE_DEVICE
    (2% tolerance) sets the bug-level floor at physical coupling.
    """
    from dataclasses import replace
    from dispersive_readout.physics.config import CouplingParams
    d = REFERENCE_DEVICE
    d_weak = replace(d, coupling=CouplingParams(g=_TWO_PI * 8e6))
    energies, states = diagonalize_transmon(d_weak.transmon, d_weak.truncation)
    n_mat = charge_operator_matrix_elements(states, d_weak.truncation)
    chi_per_level = dispersive_shift_full(
        energies, n_mat, d_weak.coupling.g, d_weak.resonator.omega_r,
    )
    chi_analytic_half = (chi_per_level[1] - chi_per_level[0]) / 2.0
    chi_numerical_half = dispersive_shift_from_simulation(d_weak)
    rel_error = abs(chi_analytic_half - chi_numerical_half) / abs(chi_analytic_half)
    assert rel_error < 1e-4, (
        f"Weak-coupling V2 FAIL: chi analytic/2π = "
        f"{chi_analytic_half/_TWO_PI/1e6:.6f} MHz, numerical/2π = "
        f"{chi_numerical_half/_TWO_PI/1e6:.6f} MHz, rel err = {rel_error:.2e}"
    )
