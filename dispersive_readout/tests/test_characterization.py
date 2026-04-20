"""Module 3 — characterization tests (C1–C7). See MODULE_3_SPEC.md §5."""
from __future__ import annotations

import math

import numpy as np
import pytest


# -- C2: noise model sanity ---------------------------------------------------

def test_C2a_shot_noise_matches_binomial():
    """Shot-noise sampling variance matches p(1-p)/n_shots within 5% at n=5000 trials."""
    from dispersive_readout.characterization.noise import apply_shot_noise
    rng = np.random.default_rng(seed=42)
    P_true = np.array([0.3, 0.5, 0.7])
    n_shots = 5000
    n_trials = 5000
    samples = np.stack([apply_shot_noise(P_true, n_shots, rng) for _ in range(n_trials)])
    observed_var = samples.var(axis=0)
    expected_var = P_true * (1.0 - P_true) / n_shots
    rel = np.abs(observed_var - expected_var) / expected_var
    assert np.all(rel < 0.05), f"shot-noise variance mismatch: rel={rel}"


def test_C2b_1f_drift_psd_slope_approx_minus_one():
    """Log-log slope of averaged |FFT|² vs f lies in [-1.3, -0.7] for alpha=1.

    Single realization PSDs are extremely noisy; average over 200 realizations
    and fit a line to the log-log PSD.
    """
    from dispersive_readout.characterization.noise import generate_1f_drift
    n_points = 1024
    n_real = 200
    # Average |FFT|² across realizations, take positive-freq half.
    psd_sum = np.zeros(n_points // 2)
    for k in range(n_real):
        x = generate_1f_drift(n_points, amplitude_Hz=1e4, alpha=1.0, seed=1000 + k)
        X = np.fft.fft(x)
        psd = np.abs(X) ** 2
        psd_sum += psd[:n_points // 2]
    psd_mean = psd_sum / n_real
    # Fit log-log, skip DC bin (index 0).
    f = np.arange(1, n_points // 2)
    slope, _ = np.polyfit(np.log(f), np.log(psd_mean[1:]), 1)
    assert -1.3 < slope < -0.7, f"1/f slope = {slope:.3f}, expected ~-1"


def test_C2c_load_reference_F_full_matches_yaml():
    """load_reference_F_full returns the F_full value committed in fig2_data.yaml."""
    from dispersive_readout.characterization.noise import load_reference_F_full
    import yaml
    with open("06_Dispersive_Readout/figures/fig2_data.yaml") as f:
        budget = yaml.safe_load(f)
    assert abs(load_reference_F_full() - float(budget["F_full"])) < 1e-12
