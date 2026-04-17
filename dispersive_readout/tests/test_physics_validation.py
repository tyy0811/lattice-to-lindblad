"""Gating physics-validation tests V1–V4 for Module 1.

If any test in this file fails, Module 1 is not complete. Do not loosen
tolerances; debug the implementation. See Module 1 spec §4 for the
tolerance rationale.
"""
from __future__ import annotations

import math
from dataclasses import replace

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


@pytest.mark.slow
def test_V4a_T2_recovery_from_pure_dephasing():
    """V4a T2/gamma_phi recovery — validates the dephasing collapse operator.

    Same H=0 simplification as V3 (see V3 docstring for the full rationale).
    Here the qubit-diagonal drift would have produced a phase oscillation on
    |0><1|, but |rho_01(t)|² = coh_magnitude² is invariant under that
    phase rotation, so fitting the coherence *magnitude* decay gives T2
    exactly.

    Setup: initialize (|0> + |1>)/sqrt(2), set gamma_1 = 0 and n_th = 0 so
    only gamma_phi drives decay, evolve for 5/gamma_phi ≈ 150 us, fit
    exp(-gamma_phi·t) to |<0|rho|1>|.

    Gate: gamma_phi_fit matches gamma_phi input to 1% (spec §4 V4).

    Task 15 update: also sets κ = 0 to isolate γ_φ from Purcell-mediated
    coherence decay (γ_P/2 contribution). With κ > 0 the fit would include
    γ_P/2 on top of γ_φ.
    """
    import qutip as qt
    from dispersive_readout.physics.lindblad import build_collapse_operators

    d = REFERENCE_DEVICE
    d_deph = replace(
        d,
        decoherence=replace(d.decoherence, gamma_1=0.0, n_th=0.0),
        resonator=replace(d.resonator, kappa=0.0),
    )
    tr = d_deph.truncation
    Nq, Nr = tr.N_transmon, tr.N_resonator

    # Initial state: (|0>+|1>)/sqrt(2) ⊗ |vacuum>
    psi0 = qt.tensor(
        (qt.basis(Nq, 0) + qt.basis(Nq, 1)).unit(),
        qt.basis(Nr, 0),
    )
    rho0 = psi0 * psi0.dag()

    H_zero = qt.tensor(qt.qeye(Nq), qt.qeye(Nr)) * 0
    c_ops = build_collapse_operators(d_deph, Nq, Nr)

    # Coherence operator |0><1| ⊗ I_r — expectation is rho_01.
    coherence_op = qt.tensor(qt.basis(Nq, 0) * qt.basis(Nq, 1).dag(), qt.qeye(Nr))

    t_duration = 5.0 / d_deph.decoherence.gamma_phi
    t_list = np.linspace(0.0, t_duration, 200)

    result = qt.mesolve(
        H_zero, rho0, t_list, c_ops=c_ops, e_ops=[coherence_op],
        options={"nsteps": 10000, "atol": 1e-10, "rtol": 1e-8},
    )

    rho01 = np.asarray(result.expect[0], dtype=complex)
    coh_mag = np.abs(rho01)

    mask = coh_mag > 1e-3
    log_c = np.log(coh_mag[mask])
    coef = np.polyfit(t_list[mask], log_c, 1)
    gamma_phi_fit = -coef[0]

    rel_err = (
        abs(gamma_phi_fit - d_deph.decoherence.gamma_phi)
        / d_deph.decoherence.gamma_phi
    )
    assert rel_err < 0.01, (
        f"V4a FAIL: γ_phi_fit = {gamma_phi_fit:.3e}, γ_phi_input = "
        f"{d_deph.decoherence.gamma_phi:.3e}, rel err = {rel_err:.3%}."
    )


def test_V4b_purcell_formula_matches_dressed_state_overlap():
    """V4b: validate γ_P = (g|n_{01}|/Δ)² κ against the full-JC dressed-state
    resonator-component overlap.

    Refactor note (Task 15): build_hamiltonian now returns the dispersive-
    regime effective Hamiltonian with the transverse coupling transformed
    out. Purcell decay no longer happens automatically via
    coupling × κ-on-resonator — it must be added as an explicit
    Lindblad channel (see build_collapse_operators step 6). This test
    validates the FORMULA used by build_collapse_operators against an
    independent calculation: diagonalize the full (non-dispersive) JC
    Hamiltonian, find the dressed state adiabatically connected to |1, 0⟩,
    take its squared overlap with bare |0, 1⟩, and multiply by κ.

    This is a stronger V4b than the plan's original free-evolution fit
    (which, in the dispersive frame, would trivially reproduce whatever
    γ_P we put into the collapse op — the new test compares two
    independent calculations instead).

    Gate: 5% agreement (spec §4 V4, preserved).
    """
    import qutip as qt

    d = REFERENCE_DEVICE
    tr = d.truncation
    Nq, Nr = tr.N_transmon, tr.N_resonator

    energies, eigenstates = diagonalize_transmon(d.transmon, tr)
    from dispersive_readout.physics.transmon import charge_operator_matrix_elements
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    # Analytic formula: γ_P = (g |n_01| / Δ)² κ, Δ = ω_01 − ω_r
    g = d.coupling.g
    kappa = d.resonator.kappa
    omega_r = d.resonator.omega_r
    Delta = (energies[1] - energies[0]) - omega_r
    n_01 = abs(n_mat[0, 1])
    gamma_P_analytic = (g * n_01 / Delta) ** 2 * kappa

    # Dressed-overlap calculation: diagonalize full JC, pick dressed |1,0⟩,
    # compute | <0, 1|ψ_dressed> |².
    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    H_q = qt.tensor(qt.Qobj(np.diag(energies)), qt.qeye(Nr))
    H_r = omega_r * a.dag() * a
    n_op_q = qt.tensor(qt.Qobj(n_mat), qt.qeye(Nr))
    H_c = g * n_op_q * (a + a.dag())
    H_full = H_q + H_r + H_c

    eigvals, eigvecs = H_full.eigenstates()

    bare_10 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nr, 0))
    bare_01 = qt.tensor(qt.basis(Nq, 0), qt.basis(Nr, 1))
    overlaps_with_10 = np.array([abs(bare_10.overlap(v)) ** 2 for v in eigvecs])
    idx = int(np.argmax(overlaps_with_10))
    dressed_10 = eigvecs[idx]

    mix_01 = abs(bare_01.overlap(dressed_10)) ** 2
    gamma_P_dressed = mix_01 * kappa

    rel_err = abs(gamma_P_dressed - gamma_P_analytic) / gamma_P_analytic
    assert rel_err < 0.05, (
        f"V4b FAIL: γ_P analytic/2π = {gamma_P_analytic/_TWO_PI/1e3:.3f} kHz, "
        f"dressed-overlap/2π = {gamma_P_dressed/_TWO_PI/1e3:.3f} kHz, "
        f"rel err = {rel_err:.3%}"
    )


@pytest.mark.slow
def test_V3_T1_recovery_from_undriven_decay():
    """V3 T1 recovery test — validates the collapse-operator machinery.

    Uses H=0 (drive-free, trivial Hamiltonian) rather than calling
    simulate_readout with drive_amplitude=0. Rationale: at rotating frame
    ω_d = ω_r, transmon level j rotates at ω_j − j·ω_d ~ 2–10 GHz in the
    drift Hamiltonian. The Lindblad solver then needs ~100 ps timesteps,
    which makes 5·T1 = 150 μs integration fail with IntegratorException
    (excess work done) regardless of nsteps or solver choice.

    For population decay the qubit-diagonal drift commutes with |j⟩⟨j|, so
    setting H=0 gives physically identical populations to the full drift +
    zero drive case. The coupling term g n̂ (a+a†) contributes only
    Purcell-type corrections of order (g/Δ)² κ — that physics is tested
    separately against the analytic formula in V4b.

    This is a validation-test simplification; simulate_readout itself is
    unchanged and correct for short-pulse readout (Modules 2+). The bypass
    is isolated to V3 and V4a and documented in the docstring of each test.

    Gate: γ_fit matches γ_1 input to 1% (spec §4 V3).

    Task 15 update: also sets κ = 0 to isolate γ_1 from the Purcell channel
    (added explicitly to build_collapse_operators in the dispersive-frame
    refactor). With κ > 0 the measured decay is γ_1 + γ_P; the combined-
    rate check is covered separately by V4b + V3 together.
    """
    import qutip as qt
    from dispersive_readout.physics.lindblad import build_collapse_operators

    d = REFERENCE_DEVICE
    d_pure = replace(
        d,
        decoherence=replace(d.decoherence, gamma_phi=0.0, n_th=0.0),
        resonator=replace(d.resonator, kappa=0.0),
    )
    T1 = 1.0 / d_pure.decoherence.gamma_1
    tr = d_pure.truncation
    Nq, Nr = tr.N_transmon, tr.N_resonator

    H_zero = qt.tensor(qt.qeye(Nq), qt.qeye(Nr)) * 0
    c_ops = build_collapse_operators(d_pure, Nq, Nr)
    psi0 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nr, 0))
    e_ops = [
        qt.tensor(qt.basis(Nq, j) * qt.basis(Nq, j).dag(), qt.qeye(Nr))
        for j in range(Nq)
    ]
    t_list = np.linspace(0.0, 5.0 * T1, 300)

    result = qt.mesolve(
        H_zero, psi0, t_list, c_ops=c_ops, e_ops=e_ops,
        options={"nsteps": 10000, "atol": 1e-10, "rtol": 1e-8},
    )

    p1 = np.asarray(result.expect[1], dtype=float)
    mask = p1 > 1e-3
    coef = np.polyfit(t_list[mask], np.log(p1[mask]), 1)
    gamma_fit = -coef[0]

    rel_err = abs(gamma_fit - d_pure.decoherence.gamma_1) / d_pure.decoherence.gamma_1
    assert rel_err < 0.01, (
        f"V3 FAIL: γ_fit = {gamma_fit:.3e}, γ_input = {d_pure.decoherence.gamma_1:.3e}, "
        f"rel err = {rel_err:.3%}."
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
