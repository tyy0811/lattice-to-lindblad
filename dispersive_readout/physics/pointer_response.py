"""Analytic per-segment cavity-EOM integration for direct-jump active reset.

Solves dα/dt = -(κ/2 + i·δ_s)·α - i·ε in closed form on each segment of
a piecewise-constant qubit-state history. Returns both the time-resolved
α trajectory (for diagnostics) and the closed-form integrated IQ (the
quantity consumed by extract_joint_matrix's threshold step).

Convention: integrated IQ uses Module 1's `∫ α(t) dt` convention (no 1/τ
averaging factor), so Module 1's σ_per_quadrature = √(τ/(4κ)) noise scale
applies directly without rescaling. See compute_assignment_fidelity in
physics/readout_model.py for the canonical noise formula.
"""
from __future__ import annotations

import numpy as np


def _segment_integral_factor(rate: complex, dt: float) -> complex:
    """Returns (1 - exp(-rate·dt)) / rate, stably for all dt.

    For |rate·dt| < 1e-8, evaluates the Taylor expansion
        dt · (1 - rate·dt/2 + (rate·dt)²/6)
    to avoid catastrophic cancellation in `1 - exp(...)`. Otherwise uses
    np.expm1 for numerical accuracy.

    This matters at very short final segments (e.g., a qubit jump occurring
    at t_jump = τ_meas - ε with ε picosecond-scale), which are plausible
    across the V7 sweep range.
    """
    x = rate * dt
    if abs(x) < 1e-8:
        # Taylor: (1 - exp(-x)) / rate = dt · (1 - x/2 + x²/6 - ...)
        return dt * (1.0 - x / 2.0 + (x * x) / 6.0)
    return -np.expm1(-x) / rate
