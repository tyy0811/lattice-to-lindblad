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
