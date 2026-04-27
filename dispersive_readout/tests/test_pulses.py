"""Pulse envelope, derivative, π-pulse calibration, and DRAG quadrature tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.control.pulses import (
    calibrate_pi_pulse_amplitude,
    sin2_windowed_gaussian,
    sin2_windowed_gaussian_derivative,
)

T_GATE = 20e-9
SIGMA = T_GATE / 4.0
A_TEST = 1.0e9  # rad/s, arbitrary nonzero amplitude for shape tests


def test_envelope_zero_at_boundaries():
    assert sin2_windowed_gaussian(0.0, A_TEST, T_GATE, SIGMA) == pytest.approx(0.0, abs=1e-20)
    assert sin2_windowed_gaussian(T_GATE, A_TEST, T_GATE, SIGMA) == pytest.approx(0.0, abs=1e-20)


def test_envelope_derivative_zero_at_boundaries():
    # V7 endpoint smoothness: Ω̇_x(0) = Ω̇_x(T) = 0 by construction. At t=0 the
    # implementation returns exact zero (sin(0) = 0 exactly). At t=T, np.sin(π)
    # carries ~1.22e-16 of float rounding, producing a residual ~A·π/T·1e-16
    # (machine precision relative to the peak |Ω̇| ≈ A·π/T). The tolerance
    # below is set to peak·1e-15 ≈ 160 rad/s², comfortably above the rounding
    # floor while still catching any real discontinuity.
    peak_omega_dot = A_TEST * math.pi / T_GATE
    tol = peak_omega_dot * 1e-15
    assert sin2_windowed_gaussian_derivative(0.0, A_TEST, T_GATE, SIGMA) == pytest.approx(0.0, abs=tol)
    assert sin2_windowed_gaussian_derivative(T_GATE, A_TEST, T_GATE, SIGMA) == pytest.approx(0.0, abs=tol)


def test_envelope_max_at_midpoint():
    """For sigma >= T/4 the Gaussian is wide enough that the sin² window dominates;
    the maximum sits at the midpoint to within numerical tolerance."""
    grid = np.linspace(0.0, T_GATE, 2001)
    values = sin2_windowed_gaussian(grid, A_TEST, T_GATE, SIGMA)
    idx_max = np.argmax(values)
    assert grid[idx_max] == pytest.approx(T_GATE / 2.0, abs=T_GATE / 2000.0)


def test_envelope_derivative_matches_finite_difference():
    """Analytic Ω̇_x(t) must match a centered finite-difference of Ω_x(t).

    Uses scale-relative tolerance: max |analytic - fd| / max |fd|. Pointwise
    relative error blows up where fd ≈ 0 by symmetry (notably t=T/2, where
    sin² is symmetric about its peak so fd is exactly 0 in float, but the
    analytic formula evaluates sin(π) → ~1.22e-16 ≠ 0). This is a pure
    floating-point artifact at the peak; the scale-relative comparison is
    the physically meaningful agreement check.
    """
    grid = np.linspace(1e-12, T_GATE - 1e-12, 401)
    h = 1e-13
    fd = (
        sin2_windowed_gaussian(grid + h, A_TEST, T_GATE, SIGMA)
        - sin2_windowed_gaussian(grid - h, A_TEST, T_GATE, SIGMA)
    ) / (2.0 * h)
    analytic = sin2_windowed_gaussian_derivative(grid, A_TEST, T_GATE, SIGMA)
    abs_err = np.max(np.abs(analytic - fd))
    fd_scale = np.max(np.abs(fd))
    assert abs_err / fd_scale < 1e-4


def test_pi_pulse_pulse_area():
    """Calibrated A must give ∫_0^T Ω_x(t) dt = π to <1e-6."""
    sigma = T_GATE / 4.0
    A = calibrate_pi_pulse_amplitude(T_GATE, sigma)
    grid = np.linspace(0.0, T_GATE, 100_001)
    integral = np.trapezoid(
        sin2_windowed_gaussian(grid, A, T_GATE, sigma),
        grid,
    )
    assert integral == pytest.approx(math.pi, abs=1e-6)


def test_pi_pulse_amplitude_positive():
    A = calibrate_pi_pulse_amplitude(T_GATE, T_GATE / 4.0)
    assert A > 0.0
