"""Readout-model integration tests (dynamics + IQ separation + assignment fidelity).

All tests here drive a full Lindblad master-equation integration via QuTiP
mesolve. Marked @pytest.mark.slow so the fast TDD suite can skip them via
`pytest -m "not slow"`. Run on-demand with `pytest -m slow` or the default
`pytest`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.readout_model import (
    AssignmentFidelityResult,
    ReadoutResult,
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)

_TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _default_drive() -> DriveParams:
    return DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)


def test_simulate_readout_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    t_list = np.linspace(0.0, 500e-9, 101)
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0, t_list=t_list)
    assert isinstance(res, ReadoutResult)
    assert res.t.shape == (101,)
    assert res.a_expectation.shape == (101,)
    assert res.photon_number.shape == (101,)
    assert res.qubit_populations.shape == (101, d.truncation.N_transmon)
    assert res.drive_envelope.shape == (101,)


def test_simulate_readout_photon_number_is_nonnegative():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0)
    assert np.all(res.photon_number >= -1e-10)


def test_simulate_readout_populations_sum_to_unity():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=1)
    totals = res.qubit_populations.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-3)


def test_simulate_readout_iq_trajectories_separate_for_0_and_1():
    """The steady-state ⟨a⟩ for |0> and |1> must differ by a measurable amount."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    res0 = simulate_readout(d, drv, initial_qubit_state=0)
    res1 = simulate_readout(d, drv, initial_qubit_state=1)
    # Compare mean ⟨a⟩ over the last 20% of the window (after the rise transient)
    tail0 = res0.a_expectation[int(0.8 * len(res0.a_expectation)):]
    tail1 = res1.a_expectation[int(0.8 * len(res1.a_expectation)):]
    sep = abs(tail0.mean() - tail1.mean())
    assert sep > 0.05, f"IQ separation {sep:.4f} too small — dispersive regime lost?"


def test_assignment_fidelity_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    assert isinstance(f, AssignmentFidelityResult)
    assert 0.0 <= f.F_assign <= 1.0
    assert f.separation_distance > 0.0
    assert f.snr > 0.0


def test_assignment_fidelity_ideal_is_at_least_as_large_as_gaussian():
    """With no shot noise, fidelity is bounded above by the 'gaussian' noise case."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f_g = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    f_i = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="ideal")
    assert f_i.F_assign >= f_g.F_assign - 1e-9


def test_assignment_fidelity_analytic_matches_phi_snr_over_2():
    """noise_model='analytic' must return F = Φ(SNR/2) exactly.

    Pins the semantics of the 'analytic' mode from the closed-form side:
    direct comparison of the reported F against scipy's Gaussian CDF
    evaluated at snr/2 (the same SNR reported in the result).
    """
    from scipy.stats import norm
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f_a = compute_assignment_fidelity(r0, r1, window, n_shots=10_000, noise_model="analytic")
    expected = float(norm.cdf(f_a.snr / 2.0))
    assert abs(f_a.F_assign - expected) < 1e-12, (
        f"analytic F={f_a.F_assign} does not match Φ(SNR/2)={expected} "
        f"at SNR={f_a.snr:.4f}; definitional invariant violated."
    )


def test_assignment_fidelity_gaussian_converges_to_analytic_as_n_shots_grows():
    """Pin the invariant 'gaussian' → 'analytic' as n_shots → ∞.

    At large n_shots the empirical perpendicular-bisector F should agree
    with the closed-form F = Φ(SNR/2) to within the binomial SE √(F(1-F)/n).
    Tightens the 'ideal ≥ gaussian' bound above into a two-sided statement
    about how the three modes relate.
    """
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    n = 200_000
    rng = np.random.default_rng(seed=42)
    f_g = compute_assignment_fidelity(
        r0, r1, window, n_shots=n, noise_model="gaussian", rng=rng,
    )
    f_a = compute_assignment_fidelity(r0, r1, window, n_shots=n, noise_model="analytic")
    # 5σ binomial tolerance at n=2e5
    tol = 5.0 * math.sqrt(f_a.F_assign * (1.0 - f_a.F_assign) / n)
    assert abs(f_g.F_assign - f_a.F_assign) < tol, (
        f"gaussian F={f_g.F_assign:.5f} does not match analytic F={f_a.F_assign:.5f} "
        f"at n={n} within 5σ_binomial={tol:.2e}. Invariant 'gaussian → analytic "
        "as n → ∞' violated — shot-noise sampling and Φ(SNR/2) closed form drift."
    )


def test_assignment_fidelity_sanity_on_reference_device():
    """Reference device should hit ≥ 95% assignment fidelity at a drive
    amplitude that stays within the dispersive regime.

    Uses ε/2π = 10 MHz and a 450 ns integration window (50–500 ns). The
    SNR scales as ε × √T; at these parameters the physical SNR formula
    SNR² = 4κ |Δα|² T gives ≈ 3.9, which yields F ≳ 0.97 via the
    perpendicular-bisector discriminator. Mean intracavity photon number
    stays ~6 — below the N_resonator=15 warning threshold and well below
    the critical n_crit = (Δ/2g)² ≈ 100 for the dispersive approximation.

    The _default_drive (ε/2π = 2 MHz) is kept low for other tests so
    photons stay well under the Fock cutoff and those tests run fast.
    """
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 10e6, duration=500e-9, detuning=0.0)
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (50e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=20000, noise_model="gaussian")
    assert f.F_assign >= 0.95, f"Reference device fidelity {f.F_assign:.4f} below 0.95 — flag to human."


def test_snr_vs_integration_time_shape_and_monotone_rise():
    """SNR should rise roughly as sqrt(t) over short integrations and plateau."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    t_int = np.linspace(50e-9, 450e-9, 9)
    snr = snr_vs_integration_time(d, drv, t_int)
    assert snr.shape == (9,)
    # Monotone rise before plateau: first half must be non-decreasing (tolerating noise)
    early = snr[: len(snr) // 2]
    assert np.all(np.diff(early) >= -0.05), f"SNR not rising: {early}"
    # Final SNR should exceed the first SNR
    assert snr[-1] > snr[0]


# ---------------------------------------------------------------------------
# Module-1-side classify_iq refactor (Module 5b prerequisite)
# ---------------------------------------------------------------------------


def test_classify_iq_matches_compute_assignment_fidelity_logic():
    """Module-1-side refactor gate: extracting classify_iq from
    compute_assignment_fidelity must produce bit-identical classification
    for any (iq, centroid_g, centroid_e) triple. Failure here means the
    extraction changed the discriminator behavior — debug before any 5b
    code consumes the helper.
    """
    from dispersive_readout.physics.readout_model import classify_iq

    centroid_g = complex(1.0, 0.0)
    centroid_e = complex(3.0, 0.5)
    midpoint = 0.5 * (centroid_g + centroid_e)
    separation = abs(centroid_e - centroid_g)
    axis = (centroid_e - centroid_g) / separation

    def reference(iq):
        proj = np.real((iq - midpoint) * np.conj(axis))
        return 1 if proj > 0 else 0

    rng = np.random.default_rng(seed=20260428)
    test_points = [centroid_g, centroid_e, midpoint] + [
        complex(rng.standard_normal(), rng.standard_normal()) * 5 + midpoint
        for _ in range(50)
    ]
    for iq in test_points:
        assert classify_iq(iq, centroid_g, centroid_e) == reference(iq), (
            f"classify_iq disagreed with reference at iq={iq}"
        )


def test_classify_iq_at_centroids():
    """Edge-case sanity: an IQ point exactly at centroid_g classifies as 0;
    exactly at centroid_e classifies as 1. The midpoint itself produces
    proj == 0, which the > 0 rule classifies as 0 (g-side, by convention).
    """
    from dispersive_readout.physics.readout_model import classify_iq

    cg = complex(1.0, 0.0)
    ce = complex(3.0, 0.0)
    assert classify_iq(cg, cg, ce) == 0
    assert classify_iq(ce, cg, ce) == 1
    assert classify_iq(0.5 * (cg + ce), cg, ce) == 0  # midpoint → g
