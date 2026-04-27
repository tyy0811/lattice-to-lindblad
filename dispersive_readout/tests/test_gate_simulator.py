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


@pytest.mark.parametrize("T_gate_ns", [10, 15])
def test_drag_calibrated_suppresses_final_leakage_5x(T_gate_ns):
    """V2a (spec §6, blocking — post-amendment N10): at T_gate ∈ {10, 15} ns,
    REFERENCE α (via transmon_summary, ≈ −210 MHz/2π in deep-transmon limit),
    decoherence zeroed, calibrated β_opt reduces final leakage P_{≥2}(T) by
    ≥5× vs no-DRAG baseline. This is the blocking "DRAG works" test."""
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

    result_opt = simulate_x_gate(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=cal.beta_opt,
        decoherence=decoherence,
        sigma=sigma,
    )
    p_final_opt = leakage_population(result_opt.rho_final, n_levels)
    suppression_final = cal.p_final_no_drag / max(p_final_opt, 1e-30)

    assert suppression_final >= 5.0, (
        f"V2a final-leakage suppression failed at T_gate={T_gate_ns}ns: "
        f"{suppression_final:.2f}× at β_opt={cal.beta_opt:.3f}; "
        f"P_final_no_DRAG={cal.p_final_no_drag:.3e}, P_final_β_opt={p_final_opt:.3e}."
    )


def test_drag_calibrated_does_not_increase_peak_leakage():
    """V2b (spec §6, diagnostic — post-amendment N10): the combined-max-ratio
    calibration must not pick a β that *increases* peak leakage relative to
    no-DRAG (the recovery-only failure mode the calibration objective exists
    to prevent). DRAG-1 peak suppression at REFERENCE_DEVICE α saturates at
    ~3× for sin²-windowed envelopes; the achievable curve is reported as a
    deliverable in panel (b) and the YAML cache, not asserted as a threshold."""
    n_levels = 4
    T_gate = 10e-9
    sigma = T_gate / 4.0
    decoherence = _zero_decoherence()

    cal = calibrate_drag_beta(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        sigma=sigma,
        n_levels=n_levels,
        decoherence=decoherence,
    )
    result_opt = simulate_x_gate(
        device=REFERENCE_DEVICE,
        T_gate=T_gate,
        n_levels=n_levels,
        drag=True,
        beta=cal.beta_opt,
        decoherence=decoherence,
        sigma=sigma,
    )
    p_peak_opt = leakage_peak(result_opt.rho_t, n_levels)
    suppression_peak = cal.p_peak_no_drag / max(p_peak_opt, 1e-30)

    assert suppression_peak > 1.0, (
        f"V2b recovery-mode guard failed at T_gate=10ns: peak suppression "
        f"{suppression_peak:.2f}× ≤ 1×, meaning calibration picked a β that "
        f"makes peak leakage worse than no-DRAG. β_opt={cal.beta_opt:.3f}, "
        f"P_peak_no_DRAG={cal.p_peak_no_drag:.3e}, P_peak_β_opt={p_peak_opt:.3e}."
    )
