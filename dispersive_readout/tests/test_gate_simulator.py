"""Gate simulator integration tests: V1 (Rabi trajectory), V2 (DRAG suppression),
V3 (truncation convergence), V4 (decoherence-free fidelity), V6 (sign flip).

Slow tests live in this file too (V5a/V5b α-scaling sweep).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from dispersive_readout.control.gate_simulator import simulate_x_gate
from dispersive_readout.control.pulses import (
    calibrate_pi_pulse_amplitude,
    sin2_windowed_gaussian,
)
from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    DecoherenceParams,
)


T_GATE = 20e-9


def _zero_decoherence() -> DecoherenceParams:
    return DecoherenceParams(gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False)


def test_x_gate_population_transfer_no_decoherence_trajectory():
    """V1 (spec §6, §7.2): at n_levels=2, no DRAG, no Lindblad, simulated P_1(t)
    must match sin²((1/2)∫_0^t Ω_x(t') dt') across the full pulse window to <1e-4."""
    sigma = T_GATE / 4.0
    A = calibrate_pi_pulse_amplitude(T_GATE, sigma)

    result = simulate_x_gate(
        device=REFERENCE_DEVICE,
        T_gate=T_GATE,
        n_levels=2,
        drag=False,
        beta=0.0,
        decoherence=_zero_decoherence(),
    )

    # Compute analytic P_1(t) = sin²((1/2) ∫_0^t Ω_x dt')
    t_grid = result.t_array
    analytic = np.empty_like(t_grid)
    for i, t in enumerate(t_grid):
        if t <= 0:
            analytic[i] = 0.0
            continue
        sub_grid = np.linspace(0.0, t, max(2, 1 + int(t / T_GATE * 2000)))
        running_integral = np.trapezoid(
            sin2_windowed_gaussian(sub_grid, A, T_GATE, sigma),
            sub_grid,
        )
        analytic[i] = math.sin(0.5 * running_integral) ** 2

    # Simulated population in |1⟩
    simulated = np.array([float((qt.basis(2, 1).proj() * rho).tr().real) for rho in result.rho_t])

    assert np.max(np.abs(simulated - analytic)) < 1e-4
    # Also assert endpoint inversion
    assert simulated[-1] == pytest.approx(1.0, abs=1e-4)


from dispersive_readout.analysis.gate_metrics import leakage_population


def test_drag_sign_flip_increases_leakage():
    """V6 (spec §6, §7.2): at REFERENCE_DEVICE α with decoherence zeroed, swapping
    β → −β must increase leakage relative to β = +1. Confirms the rotating-frame
    sign convention."""
    n_levels = 4
    T_gate = 10e-9  # short pulse — leakage is severe and DRAG sign matters most
    decoherence = _zero_decoherence()

    result_pos = simulate_x_gate(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=+1.0,
        decoherence=decoherence,
    )
    result_neg = simulate_x_gate(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=-1.0,
        decoherence=decoherence,
    )

    leak_pos = leakage_population(result_pos.rho_final, n_levels)
    leak_neg = leakage_population(result_neg.rho_final, n_levels)

    assert leak_neg > leak_pos, (
        f"DRAG sign convention bug: β=+1 gave leakage {leak_pos:.4e}, "
        f"β=-1 gave {leak_neg:.4e} (expected β=+1 to be smaller)."
    )


from dispersive_readout.analysis.gate_metrics import leakage_peak
from dispersive_readout.control.drag_calibration import calibrate_drag_beta


def test_v2a_drag_gate_error_below_1e4_at_headline():
    """V2a (spec §6, blocking — post-amendments N11 + N12): at the headline
    T_gate=20ns, REFERENCE α, decoherence zeroed, fidelity-optimal β_opt
    produces 1 − F_avg < 1e−4 where F_avg is averaged over the Pauli set
    {|0⟩, |1⟩, |+⟩, |+i⟩}. Empirical at v0: ~5.5e−5 (passes by ~18×).

    The β grid is restricted to [0, 1.2] (perturbative DRAG-1 range); the
    calibration objective is average gate fidelity (not one-way transfer or
    leakage ratios). Both guards together ensure the optimizer cannot select
    non-DRAG values that satisfy a loss but break the gate (see N11/N12).
    """
    from dispersive_readout.analysis.gate_metrics import average_gate_fidelity_x

    n_levels = 4
    T_gate = 20e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        n_levels=n_levels,
        decoherence=decoherence,
    )
    f_avg, _ = average_gate_fidelity_x(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=cal.beta_opt,
        decoherence=decoherence,
        sigma=sigma,
    )
    gate_error = 1.0 - f_avg
    assert gate_error < 1e-4, (
        f"V2a failed at headline T_gate=20ns: 1−F_avg={gate_error:.3e} at β_opt={cal.beta_opt:.3f}."
    )


@pytest.mark.parametrize("T_gate_ns", [10, 15, 20, 30])
def test_v2a_regime_structure_diagnostic(T_gate_ns):
    """V2a regime context (diagnostic, non-blocking): report 1−F_avg at
    calibrated β_opt across the panel-(b) range. This documents the regime
    structure (fidelity ramp from short-pulse non-perturbative regime to
    long-pulse DRAG-functional regime) without imposing a uniform threshold.
    """
    from dispersive_readout.analysis.gate_metrics import average_gate_fidelity_x

    n_levels = 4
    T_gate = T_gate_ns * 1e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        n_levels=n_levels,
        decoherence=decoherence,
    )
    f_avg, _ = average_gate_fidelity_x(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=cal.beta_opt,
        decoherence=decoherence,
        sigma=sigma,
    )
    gate_error = 1.0 - f_avg
    # Diagnostic: assert 1−F_avg < 1 (sanity only; documents regime, not threshold).
    assert gate_error < 1.0, f"Sanity violation at T_gate={T_gate_ns}ns: 1−F_avg={gate_error:.3e}"


def test_v2b_leakage_vs_fidelity_tradeoff_is_real():
    """V2b (spec §6, diagnostic — post-amendment N11): characterizes the
    leakage-vs-fidelity trade-off as a finding. At T_gate=20ns, the fidelity-
    optimal β_opt and the leakage-minimizing β values sit at materially
    different points on the perturbative β grid. The full curves
    (gate_error, p_final, p_peak) over β are exposed in
    `DragCalibrationResult` and published in the panel-(b) YAML.

    This test asserts that the trade-off is *present* (the minimizers diverge
    on the grid); the actual curves are the headline V2b deliverable.
    """
    n_levels = 4
    T_gate = 20e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        n_levels=n_levels,
        decoherence=decoherence,
    )
    # The trade-off: at headline T_gate, fidelity-optimal β ≠ leakage-optimal β.
    assert cal.beta_min_final_leak != cal.beta_opt or cal.beta_min_peak_leak != cal.beta_opt, (
        f"V2b expected trade-off: β_opt={cal.beta_opt:.3f}, "
        f"β_min_final_leak={cal.beta_min_final_leak:.3f}, "
        f"β_min_peak_leak={cal.beta_min_peak_leak:.3f}. "
        f"All three coinciding would mean no trade-off (unexpected at this regime)."
    )


def test_truncation_convergence():
    """V3 (spec §6, post-N11): at the headline T_gate=20ns under fidelity-optimal
    β_opt calibration, transfer fidelity is stable to <1e−5 across n_levels=4
    vs n_levels=5. Empirical at v0: spread ≈ 3.3e−8 (passes by ~300×).

    The N8 worst-case probe (T_gate=5ns, n∈{3,4,5} full spread) is dropped
    from V3's blocking criterion — at that T_gate the gate itself is broken
    (1−F ≈ 0.11) so truncation is not the dominant error source, and the
    regime is documented by V2a's regime sweep instead.
    """
    T_gate = 20e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        n_levels=4,
        decoherence=decoherence,
    )

    from dispersive_readout.analysis.gate_metrics import transfer_fidelity_0_to_1
    fidelities = {}
    for n in (4, 5):
        r = simulate_x_gate(
            device=REFERENCE_DEVICE,
            T_gate=T_gate,
            n_levels=n,
            drag=True,
            beta=cal.beta_opt,
            decoherence=decoherence,
            sigma=sigma,
        )
        fidelities[n] = transfer_fidelity_0_to_1(r.rho_final)

    spread = abs(fidelities[4] - fidelities[5])
    assert spread < 1e-5, f"V3 truncation convergence failed at T=20ns: |F(n=4) − F(n=5)| = {spread:.3e}, values {fidelities}."


def test_v4_decoherence_free_fidelity_ceiling_diagnostic():
    """V4 (spec §6, non-blocking diagnostic — post-N11/N12): with calibrated
    fidelity-optimal β_opt and zero decoherence, average-gate-fidelity error
    1 − F_avg < 1e−3 at the headline T_gate=20ns.

    V2a tests the (tighter) blocking bar; V4 reports the same physical
    quantity (decoherence-free ceiling) at the looser diagnostic threshold
    that the spec table already commits to.
    """
    from dispersive_readout.analysis.gate_metrics import average_gate_fidelity_x

    T_gate = 20e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        decoherence=decoherence,
    )
    f_avg, per_state = average_gate_fidelity_x(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=4,
        drag=True,
        beta=cal.beta_opt,
        decoherence=decoherence,
        sigma=sigma,
    )
    gate_error = 1.0 - f_avg
    print(f"V4 diagnostic: 1 − F_avg = {gate_error:.3e} at β_opt={cal.beta_opt:.3f}; "
          f"per-state F = [|0⟩→|1⟩: {per_state[0]:.6f}, |1⟩→|0⟩: {per_state[1]:.6f}, "
          f"|+⟩→|+⟩: {per_state[2]:.6f}, |+i⟩→|-i⟩: {per_state[3]:.6f}]")
    assert gate_error < 1e-3, f"V4 ceiling exceeded at headline T=20ns: 1−F_avg = {gate_error:.3e}."


import math
from dataclasses import replace

from dispersive_readout.physics.config import TransmonParams
from dispersive_readout.physics.transmon import transmon_summary


@pytest.mark.slow
def test_anharmonicity_scaling_full_sweep():
    """V5a (spec §6, blocking): fitted log-log slope of no-DRAG leakage vs |α|
    is negative across the swept range. V5b (spec §6, diagnostic — post-round-9):
    perturbative-half slope reported; the textbook -2 prediction assumes simple
    Rabi pulses, while the sin²-windowed envelope produces a much steeper
    observed slope due to spectral concentration. The qualitative finding (slope
    is steeply negative) is the deliverable; the precise power is envelope-dependent.

    Sweep |α| by varying device.transmon.E_C (per spec §4.3 / N3): re-extract α
    via transmon_summary at each point. Pulse amplitude A is determined by the
    π-pulse area condition (no free amplitude knob, per spec N1). Decoherence
    zeroed.
    """
    T_gate = 10e-9
    sigma = T_gate / 4.0
    n_levels = 5  # use highest truncation for V5 to capture leakage tail
    decoherence = _zero_decoherence()

    e_c_grid_hz = np.linspace(100e6, 500e6, 8)
    alphas = []
    leakages_no_drag = []

    for ec_hz in e_c_grid_hz:
        new_transmon = TransmonParams(
            E_C=2.0 * math.pi * float(ec_hz),
            E_J=REFERENCE_DEVICE.transmon.E_J,
            n_g=REFERENCE_DEVICE.transmon.n_g,
        )
        device_alt = replace(REFERENCE_DEVICE, transmon=new_transmon)
        alpha_value = float(transmon_summary(device_alt.transmon, device_alt.truncation)["alpha"])
        alphas.append(abs(alpha_value))

        r = simulate_x_gate(
            device=device_alt,
            T_gate=T_gate,
            n_levels=n_levels,
            drag=False,
            beta=0.0,
            decoherence=decoherence,
            sigma=sigma,
        )
        leakages_no_drag.append(leakage_population(r.rho_final, n_levels))

    alphas = np.asarray(alphas)
    leakages = np.asarray(leakages_no_drag)

    log_alpha = np.log(alphas)
    log_leak = np.log(leakages + 1e-30)
    slope, _ = np.polyfit(log_alpha, log_leak, 1)
    print(f"V5a: full-range fitted log-log slope = {slope:.3f} (negative required)")
    assert slope < 0.0, (
        f"V5a failed: fitted log-log slope {slope:.3f} not negative. "
        f"alphas/2π = {alphas / (2 * math.pi)}, leakages = {leakages}."
    )

    # V5b diagnostic — perturbative-half slope. Spec post-round-9: the textbook
    # -2 prediction assumes simple Rabi pulses; sin²-windowed envelopes have
    # spectral concentration that steepens the slope. Report the value; require
    # only that it's strongly negative (steeper than -1) — qualitative confirmation.
    n_pert = len(alphas) // 2
    slope_pert, _ = np.polyfit(log_alpha[-n_pert:], log_leak[-n_pert:], 1)
    print(f"V5b: perturbative-half fitted slope = {slope_pert:.3f} "
          f"(textbook -2 ± 0.5 assumes simple Rabi pulses; sin²-windowed envelope "
          f"is steeper — empirical envelope-dependent finding)")
    assert slope_pert < -1.0, (
        f"V5b sanity: perturbative slope {slope_pert:.3f} is not strongly negative; "
        f"expected steeper than -1 for sin²-windowed envelope."
    )
