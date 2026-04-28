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
