"""Sin²-windowed Gaussian pulse envelope, analytic derivative, π-pulse calibration,
and DRAG quadrature.

Conventions (spec §3.2):
- Envelope:  Ω_x(t) = A · sin²(π t / T) · exp(-(t - T/2)² / (2 σ²)),  t ∈ [0, T].
- Both Ω_x(0) = Ω_x(T) = 0 and Ω̇_x(0) = Ω̇_x(T) = 0 hold by construction.
- All amplitudes Ω in rad/s; times in seconds.
- Symbol convention: `α` is transmon anharmonicity; `Δ` is qubit-resonator detuning.
"""
from __future__ import annotations

import math

import numpy as np


def sin2_windowed_gaussian(
    t: float | np.ndarray,
    A: float,
    T_gate: float,
    sigma: float,
) -> float | np.ndarray:
    """Sin²-windowed Gaussian envelope at time t (seconds), amplitude A (rad/s).

    Returns Ω_x(t) = A · sin²(π t / T) · exp(-(t - T/2)² / (2 σ²)).
    """
    sin_factor = np.sin(math.pi * t / T_gate) ** 2
    gauss_factor = np.exp(-((t - T_gate / 2.0) ** 2) / (2.0 * sigma ** 2))
    return A * sin_factor * gauss_factor


def sin2_windowed_gaussian_derivative(
    t: float | np.ndarray,
    A: float,
    T_gate: float,
    sigma: float,
) -> float | np.ndarray:
    """Analytic time-derivative of `sin2_windowed_gaussian`.

    Ω̇_x(t) = A · exp(-(t - T/2)²/(2σ²)) · [
        (π/T) sin(2π t / T)
      - sin²(π t / T) · (t - T/2) / σ²
    ]
    """
    gauss_factor = np.exp(-((t - T_gate / 2.0) ** 2) / (2.0 * sigma ** 2))
    term_window = (math.pi / T_gate) * np.sin(2.0 * math.pi * t / T_gate)
    term_gauss = np.sin(math.pi * t / T_gate) ** 2 * (t - T_gate / 2.0) / (sigma ** 2)
    return A * gauss_factor * (term_window - term_gauss)


def calibrate_pi_pulse_amplitude(
    T_gate: float,
    sigma: float,
    n_quadrature_points: int = 100_001,
) -> float:
    """Solve ∫_0^T Ω_x(t) dt = π for amplitude A given (T_gate, sigma).

    Since the envelope is linear in A, A_calibrated = π / I_unit where
    I_unit = ∫_0^T sin²(π t / T) · exp(-(t - T/2)²/(2σ²)) dt evaluated at A = 1.
    Trapezoidal quadrature on a fine grid is sufficient for the spec's
    <1e-6 tolerance at the default 100,001 points.
    """
    grid = np.linspace(0.0, T_gate, n_quadrature_points)
    unit_integral = np.trapezoid(
        sin2_windowed_gaussian(grid, 1.0, T_gate, sigma),
        grid,
    )
    return math.pi / unit_integral
