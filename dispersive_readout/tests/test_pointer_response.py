"""Module 5b — pointer_response tests (analytic α-trajectory + V4a)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.pointer_response import _segment_integral_factor


def test_segment_integral_factor_taylor_fallback_matches_expm1_at_boundary():
    """At |rate·dt| just below 1e-8, the Taylor branch must agree with the
    expm1 branch to the precision of the fallback (Taylor truncated to
    O(x³) gives ~ |x|⁴/24 error). Boundary chosen so neither branch loses
    significant precision; agreement must hold to ~1e-13.
    """
    rate = complex(1e6, 1e6)  # rad/s scale
    # Pick dt so |rate·dt| ≈ 0.5e-8 (below the 1e-8 threshold)
    dt = 0.5e-8 / abs(rate)

    # Force Taylor branch
    taylor_value = _segment_integral_factor(rate, dt)

    # Reference: direct expm1 evaluation
    x = rate * dt
    expm1_value = -np.expm1(-x) / rate

    assert taylor_value == pytest.approx(expm1_value, rel=1e-12)


def test_segment_integral_factor_matches_steady_state_limit():
    """At large |rate·dt|, the integral factor approaches 1/rate
    (the steady-state limit: ∫ exp(-rate·t) dt over [0, ∞) = 1/rate).
    """
    rate = complex(1e7, 0)  # large κ/2
    dt = 1e-6  # |rate·dt| = 10 → exp(-10) ≈ 4.5e-5, so factor ≈ 1/rate
    factor = _segment_integral_factor(rate, dt)
    expected = 1.0 / rate
    assert factor == pytest.approx(expected, rel=1e-4)


def test_segment_integral_factor_matches_short_segment_limit():
    """At |rate·dt| → 0, the factor → dt (just integrating a constant α).
    Tests the leading-order Taylor term.
    """
    rate = complex(1e6, 0)
    dt = 1e-12  # |rate·dt| = 1e-6, deep in the Taylor regime
    factor = _segment_integral_factor(rate, dt)
    assert factor == pytest.approx(complex(dt, 0), rel=1e-6)


def test_segment_integral_factor_real_rate_returns_real():
    """For real rate (κ-only damping, no detuning), the factor is real."""
    rate = complex(2e6, 0)
    dt = 1e-7
    factor = _segment_integral_factor(rate, dt)
    assert abs(factor.imag) < 1e-15


from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.pointer_response import pointer_steady_state


def test_pointer_steady_state_alpha_inf_formula():
    """α_∞(s) = -i·ε / (κ/2 + i·δ_s) where δ_s = (ω_r − ω_d) + χ_s.

    For drive on resonance with bare cavity (detuning=0) and qubit in
    state s, the dispersive pull χ_s shifts the cavity off-resonance by
    χ_s, giving the steady-state α formula above.
    """
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=500e-9, detuning=0.0, edge_sigma=2e-9)
    alpha_g = pointer_steady_state(device, drive, qubit_state=0)
    alpha_e = pointer_steady_state(device, drive, qubit_state=1)

    # Both must be finite, non-zero complex numbers
    assert np.isfinite(alpha_g) and abs(alpha_g) > 0
    assert np.isfinite(alpha_e) and abs(alpha_e) > 0

    # The two pointer states must differ — that's the dispersive signature
    assert abs(alpha_g - alpha_e) > 0.01 * abs(alpha_g)


def test_pointer_steady_state_qubit_state_validation():
    """Only qubit_state ∈ {0, 1} is supported in v0."""
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=500e-9)
    with pytest.raises((ValueError, IndexError)):
        pointer_steady_state(device, drive, qubit_state=2)


from dispersive_readout.control.reset_protocol import QubitStateHistory
from dispersive_readout.physics.pointer_response import compute_alpha_trajectory


def test_compute_alpha_trajectory_constant_state_reaches_steady_state():
    """For constant qubit state and t_grid extending past several κ⁻¹,
    α(t) → α_∞(s) at the final time.
    """
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=5e-6, detuning=0.0)
    history = QubitStateHistory(segments=((0.0, 1),), t_total=5e-6)
    t_grid = np.linspace(0.0, 5e-6, 200)

    alpha_traj, integrated_iq = compute_alpha_trajectory(device, drive, history, t_grid)

    alpha_inf = pointer_steady_state(device, drive, qubit_state=1)
    # 5 µs >> 1/(κ/2) for REFERENCE κ ≈ 2π·5 MHz, so we should be in steady state
    assert alpha_traj[-1] == pytest.approx(alpha_inf, rel=1e-3)


def test_compute_alpha_trajectory_continuous_across_jump():
    """At a qubit jump, α(t) is continuous (the cavity remembers its phase-
    space coordinate); only δ_{s(t)} changes. The trajectory grid value at
    t_jump - epsilon and t_jump + epsilon should agree to high precision.
    """
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=2e-6, detuning=0.0)
    t_jump = 1e-6
    history = QubitStateHistory(
        segments=((0.0, 1), (t_jump, 0)),
        t_total=2e-6,
    )
    # eps small enough that rate·eps << 1e-5 at REFERENCE κ = 2π·5 MHz
    # (rate·1e-13 ≈ 1.6e-6, comfortably below the tolerance below)
    eps = 1e-13
    t_grid = np.array([t_jump - eps, t_jump + eps])
    alpha_traj, _ = compute_alpha_trajectory(device, drive, history, t_grid)
    # Continuity at the jump: α drifts by O(rate·eps) within each segment
    assert alpha_traj[0] == pytest.approx(alpha_traj[1], rel=1e-5)


def test_compute_alpha_trajectory_returns_tuple():
    """Returns (trajectory, integrated_iq); trajectory is array, IQ is scalar."""
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=500e-9, detuning=0.0)
    history = QubitStateHistory(segments=((0.0, 0),), t_total=500e-9)
    t_grid = np.array([0.0, 500e-9])
    result = compute_alpha_trajectory(device, drive, history, t_grid)
    assert isinstance(result, tuple) and len(result) == 2
    alpha_traj, integrated_iq = result
    assert isinstance(alpha_traj, np.ndarray) and alpha_traj.shape == (2,)
    assert isinstance(integrated_iq, complex)


def test_compute_alpha_trajectory_integrated_iq_independent_of_grid():
    """The closed-form integrated_iq is per-segment exact; it does NOT
    depend on t_grid resolution (t_grid is only for the trajectory output).
    """
    device = REFERENCE_DEVICE
    drive = DriveParams(amplitude=140e6, duration=1e-6, detuning=0.0)
    history = QubitStateHistory(
        segments=((0.0, 1), (5e-7, 0)),
        t_total=1e-6,
    )
    _, iq_2pt = compute_alpha_trajectory(
        device, drive, history, np.array([0.0, 1e-6]),
    )
    _, iq_200pt = compute_alpha_trajectory(
        device, drive, history, np.linspace(0.0, 1e-6, 200),
    )
    assert iq_2pt == pytest.approx(iq_200pt, rel=1e-12)


from dataclasses import replace

from dispersive_readout.physics.config import (
    DecoherenceParams,
    TruncationParams,
)
from dispersive_readout.physics.readout_model import simulate_readout


def test_pointer_response_matches_simulate_readout_in_no_jump_limit():
    """V4a (blocking integration gate): in the no-jump deterministic limit
    (γ_eff = 0, no Lindblad collapse), the integrated IQ from
    compute_alpha_trajectory must agree with the integrated IQ extracted
    from physics.readout_model.simulate_readout, to within mesolve's
    integration tolerance plus the erf-vs-square envelope mismatch.

    This is the consistency contract between the analytic semiclassical
    reduction (used by 5b) and the full mesolve calculation (Module 1
    canonical). A failure here means the semiclassical reduction's sign
    or phase conventions disagree with Module 1's, which is the most
    likely failure mode for 5b's joint-matrix output.

    Uses a smaller drive amplitude (40 MHz, not idx=18's 140 MHz) so the
    coherent-state cavity occupation |α|² stays well below the default
    N_resonator=15 Fock truncation. At 140 MHz the steady-state |α|² ≈ 17
    saturates the truncation ceiling and the mesolve α is artifically
    bounded — a numerical artifact, not a convention bug. The convention
    contract is amplitude-independent, so a smaller-amplitude check is
    fully sufficient.
    """
    no_decoherence = DecoherenceParams(
        gamma_1=0.0, gamma_phi=0.0, n_th=0.0, purcell_enabled=False,
    )
    device = replace(REFERENCE_DEVICE, decoherence=no_decoherence)
    drive = DriveParams(
        amplitude=40e6, duration=500e-9, detuning=0.0,
        edge_sigma=2e-9,
    )

    history_g = QubitStateHistory(segments=((0.0, 0),), t_total=500e-9)
    history_e = QubitStateHistory(segments=((0.0, 1),), t_total=500e-9)
    t_grid = np.linspace(0.0, 500e-9, 200)
    _, iq_g_analytic = compute_alpha_trajectory(device, drive, history_g, t_grid)
    _, iq_e_analytic = compute_alpha_trajectory(device, drive, history_e, t_grid)

    result_g = simulate_readout(device, drive, initial_qubit_state=0, t_list=t_grid)
    result_e = simulate_readout(device, drive, initial_qubit_state=1, t_list=t_grid)
    iq_g_mesolve = result_g.integrated_iq((0.0, 500e-9))
    iq_e_mesolve = result_e.integrated_iq((0.0, 500e-9))

    # erf-vs-square envelope mismatch + mesolve precision; tolerance set
    # generously to avoid masking convention bugs while absorbing envelope
    # difference. >5% failure → investigate sign/phase convention.
    rel_tol = 0.05
    assert iq_g_analytic == pytest.approx(iq_g_mesolve, rel=rel_tol), (
        f"V4a g-state IQ mismatch: analytic={iq_g_analytic}, mesolve={iq_g_mesolve}"
    )
    assert iq_e_analytic == pytest.approx(iq_e_mesolve, rel=rel_tol), (
        f"V4a e-state IQ mismatch: analytic={iq_e_analytic}, mesolve={iq_e_mesolve}"
    )
