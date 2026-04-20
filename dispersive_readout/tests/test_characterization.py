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


# -- C1a: Rabi round-trip ----------------------------------------------------

def test_C1a_rabi_round_trip():
    """Closed-form Rabi trace → fit pipeline (point-estimate only) recovers ε_π within 3%.

    Point-estimate sanity check; full uncertainty testing is in C3. Uses a
    light noise config (n_shots=5000, no drift, no amp uncertainty) so the
    round-trip is tight.
    """
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    noise = NoiseModelParams(
        n_shots_per_point=5000,
        drift_amplitude_Hz=0.0,
        drive_amplitude_uncertainty=0.0,
    )
    epsilon_pi_truth = 2 * math.pi * 50e6
    omega_q = 2 * math.pi * 4.5e9
    trace = generate_rabi_trace(epsilon_pi_truth, omega_q, noise, seed=0)
    P1 = trace.P1
    eps = trace.sweep_values
    idx_min = int(np.argmin(P1))
    eps_estimate = float(eps[idx_min])
    rel = abs(eps_estimate - epsilon_pi_truth) / epsilon_pi_truth
    assert rel < 0.03, f"Rabi round-trip: eps_est={eps_estimate:.3e}, truth={epsilon_pi_truth:.3e}, rel={rel:.3%}"


# -- Bundle round-trip (preps for schema validation in Task 6) ---------------

def test_trace_bundle_npz_round_trip(tmp_path):
    """save_trace_bundle → load_trace_bundle preserves all fields exactly."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import (
        generate_rabi_trace, save_trace_bundle, load_trace_bundle, TraceData,
    )
    noise = NoiseModelParams(n_shots_per_point=1000, drift_amplitude_Hz=0.0)
    trace = generate_rabi_trace(2 * math.pi * 30e6, 2 * math.pi * 4.8e9, noise, seed=123)
    path = tmp_path / "bundle.npz"
    save_trace_bundle([trace], str(path))
    loaded = load_trace_bundle(str(path))
    assert len(loaded) == 1
    t = loaded[0]
    assert t.protocol == trace.protocol
    assert t.sweep_axis == trace.sweep_axis
    np.testing.assert_array_equal(t.sweep_values, trace.sweep_values)
    np.testing.assert_array_equal(t.P1, trace.P1)
    np.testing.assert_array_equal(t.P1_uncertainty, trace.P1_uncertainty)
    assert t.metadata == trace.metadata
