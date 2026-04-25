"""Module 3 — noise-model helpers.

Provides the full synthetic-trace noise stack:
  - NoiseModelParams: frozen config
  - generate_1f_drift: correlated 1/f^alpha drift across a scan
  - apply_shot_noise: binomial sampling
  - apply_readout_errors: classical bit-flip from Module 2's F_assign
  - load_reference_F_full: pulls F_full from Module 2's committed YAML

Amendment 7: F_assign is read from fig2_data.yaml at call time, not stored
in NoiseModelParams, so a stale cached value cannot silently persist in
serialized runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


_REFERENCE_F_FULL_PATH = Path("06_Dispersive_Readout/figures/fig2_data.yaml")


@dataclass(frozen=True)
class NoiseModelParams:
    """Frozen noise-stack configuration for synthetic trace generation.

    n_shots_per_point:         binomial shot count per trace point.
    drift_amplitude_Hz:        rms of the 1/f qubit-frequency drift across a scan.
    drift_alpha:               spectral exponent (1 = 1/f).
    drift_seed:                per-run drift seed; None = fresh each time (driven by the harness).
    readout_asymmetric:        if True, use P(0|1) != P(1|0) (not implemented — Module 3 follow-up).
    drive_amplitude_uncertainty: Gaussian SD of a once-per-run amplitude offset (Rabi only).
    """
    n_shots_per_point: int = 2000
    drift_amplitude_Hz: float = 1e4
    drift_alpha: float = 1.0
    drift_seed: int | None = None
    readout_asymmetric: bool = False
    drive_amplitude_uncertainty: float = 0.05


def load_reference_F_full() -> float:
    """Read F_full at REFERENCE_DEVICE from Module 2's committed artifact (amendment 7)."""
    with open(_REFERENCE_F_FULL_PATH) as f:
        budget = yaml.safe_load(f)
    return float(budget["F_full"])


def generate_1f_drift(
    n_points: int,
    amplitude_Hz: float,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an n-point 1/f^alpha realization with rms `amplitude_Hz`.

    Method: draw white Gaussian samples in frequency domain with amplitude
    proportional to f^(-alpha/2); inverse-FFT; rescale to target rms. DC bin
    set to zero (pure AC drift).
    """
    rng = np.random.default_rng(seed)
    N = int(n_points)
    freqs = np.fft.fftfreq(N)
    mag = np.zeros(N, dtype=float)
    nonzero = freqs != 0.0
    mag[nonzero] = np.abs(freqs[nonzero]) ** (-alpha / 2.0)
    re = rng.standard_normal(N)
    im = rng.standard_normal(N)
    X = (re + 1j * im) * mag
    X[0] = 0.0
    if N % 2 == 0:
        X[N // 2] = np.real(X[N // 2])
    x = np.fft.ifft(X).real
    current_rms = float(np.sqrt(np.mean(x**2)))
    if current_rms == 0.0:
        return x
    return x * (amplitude_Hz / current_rms)


def apply_shot_noise(
    P_true: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial shot-noise sampling. Returns observed P = k/n_shots ∈ [0, 1]."""
    P_clipped = np.clip(P_true, 0.0, 1.0)
    k = rng.binomial(n_shots, P_clipped)
    return k / n_shots


def apply_readout_errors(
    P_observed: np.ndarray,
    F_assign: float,
    asymmetric: bool = False,
) -> np.ndarray:
    """Classical bit-flip readout-error model; symmetric by default.

    P_out = (1 − p_flip) * P_in + p_flip * (1 − P_in), where p_flip = 1 − F_assign.
    """
    if asymmetric:
        raise NotImplementedError("Asymmetric readout errors are a Module 3 follow-up; use symmetric.")
    p_flip = 1.0 - F_assign
    return (1.0 - p_flip) * P_observed + p_flip * (1.0 - P_observed)
