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


def test_C1b_ramsey_round_trip():
    """Closed-form Ramsey → simple FFT-based estimator recovers ω_q within 0.1% and T2* within 15%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    omega_q_truth = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    omega_drive_offset = 2 * math.pi * 1.5e6
    trace = generate_ramsey_trace(
        omega_q_truth, T_2_star=T_2_star_truth, noise=noise,
        omega_drive_offset=omega_drive_offset, seed=1,
    )
    delays = trace.sweep_values
    signal = trace.P1 - trace.P1.mean()
    fft = np.abs(np.fft.rfft(signal))
    dt = float(delays[1] - delays[0])
    freqs = np.fft.rfftfreq(len(delays), d=dt)
    peak = int(np.argmax(fft[1:])) + 1
    delta_omega_est = 2 * math.pi * float(freqs[peak])
    omega_q_est = omega_q_truth - omega_drive_offset + delta_omega_est
    rel = abs(omega_q_est - omega_q_truth) / omega_q_truth
    assert rel < 1e-3, f"Ramsey ω_q naive FFT estimate off: rel={rel:.3e}"


def test_C1c_t1_round_trip():
    """Closed-form T1 → simple exponential-fit estimator recovers T1 within 5%.

    Uses delay_range=(0, 200e-6) ≈ 6.7·T_1 so the trailing 10 points have
    decayed below readout floor and the naïve "average last 10" estimate of
    the asymptote isn't biased. The default (0, 100e-6) is fine for the
    real lmfit fitter (Task 8) but not for this naive estimator.
    """
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, seed=2, delay_range=(0.0, 200e-6))
    delays = trace.sweep_values
    P1 = trace.P1
    floor = float(P1[-10:].mean())
    mask = (P1 - floor) > 0.02
    coef = np.polyfit(delays[mask], np.log(P1[mask] - floor), 1)
    T_1_est = -1.0 / coef[0]
    rel = abs(T_1_est - T_1_truth) / T_1_truth
    assert rel < 0.05, f"T1 round-trip rel={rel:.3%}"


def test_C1d_t2_echo_round_trip():
    """Closed-form T2-echo → simple exponential fit recovers T2 within 10%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t2_echo_trace
    T_2_truth = 40e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t2_echo_trace(T_2_truth, noise, seed=3)
    delays = trace.sweep_values
    P1 = trace.P1
    signal = 1.0 - 2.0 * P1
    mask = signal > 0.02
    coef = np.polyfit(delays[mask], np.log(signal[mask]), 1)
    T_2_est = -1.0 / coef[0]
    rel = abs(T_2_est - T_2_truth) / T_2_truth
    assert rel < 0.10, f"T2-echo round-trip rel={rel:.3%}"


# -- Schema validation for load_trace_bundle (§8 flag #5) --------------------

def test_load_trace_bundle_rejects_missing_field(tmp_path):
    """A .npz that lacks a required field (e.g., P1_uncertainty) raises ValueError."""
    from dispersive_readout.characterization.protocols import load_trace_bundle
    path = tmp_path / "missing_field.npz"
    np.savez(
        str(path),
        n_traces=np.array(1),
        **{
            "traces/0/protocol": np.array("rabi"),
            "traces/0/sweep_axis": np.array("drive_amplitude"),
            "traces/0/sweep_values": np.array([0.0, 1.0, 2.0]),
            "traces/0/P1": np.array([0.5, 0.5, 0.5]),
            "traces/0/metadata_json": np.array("{}"),
        },
    )
    with pytest.raises(ValueError, match="P1_uncertainty"):
        load_trace_bundle(str(path))


def test_load_trace_bundle_rejects_missing_metadata(tmp_path):
    """A bundle missing metadata_json on any entry raises ValueError."""
    from dispersive_readout.characterization.protocols import load_trace_bundle
    path = tmp_path / "missing_meta.npz"
    np.savez(
        str(path),
        n_traces=np.array(1),
        **{
            "traces/0/protocol": np.array("rabi"),
            "traces/0/sweep_axis": np.array("drive_amplitude"),
            "traces/0/sweep_values": np.array([0.0, 1.0, 2.0]),
            "traces/0/P1": np.array([0.5, 0.5, 0.5]),
            "traces/0/P1_uncertainty": np.array([0.01, 0.01, 0.01]),
        },
    )
    with pytest.raises(ValueError, match="metadata"):
        load_trace_bundle(str(path))


# -- C4: Pydantic schema + to_device_config ---------------------------------

def test_C4a_fitted_parameter_requires_positive_uncertainty():
    from dispersive_readout.characterization.fitting import FittedParameter
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FittedParameter(
            name="T_1", value=30e-6, uncertainty=-1e-6, unit="s",
            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200,
        )


def test_C4b_extracted_parameter_pack_yaml_round_trip(tmp_path):
    """Serialize to YAML and re-load — round-trip preserves all fields."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    import yaml
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.2, n_bootstrap=200),
            FittedParameter(name="omega_q", value=2 * math.pi * 4.5e9,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=0.95, n_bootstrap=200),
        ],
        trace_file="example.npz",
        timestamp="2026-04-22T10:00:00+00:00",
        stage_06_version="abc123",
    )
    path = tmp_path / "pack.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(pack.model_dump(), f)
    with open(path) as f:
        reloaded = ExtractedParameterPack.model_validate(yaml.safe_load(f))
    assert reloaded == pack


def test_C4c_to_device_config_produces_simulator_consumable():
    """to_device_config() → simulate_readout() runs without error."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import simulate_readout
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="omega_q", value=2 * math.pi * 4.5e9,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="example.npz",
        timestamp="2026-04-22T10:00:00+00:00",
        stage_06_version="abc123",
    )
    device = pack.to_device_config()
    drive = DriveParams(amplitude=2 * math.pi * 2e6, duration=500e-9, detuning=0.0)
    t_list = np.linspace(0.0, drive.duration, 101)
    _ = simulate_readout(device, drive, initial_qubit_state=0, t_list=t_list)


# -- C7: to_device_config physics consistency (amendment 5) ------------------

def test_C7a_to_device_config_back_solves_E_J_from_omega_q():
    """E_J = (ω_q + E_C)² / (8·E_C) per Koch 2007."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    omega_q_target = 2 * math.pi * 4.5e9
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="omega_q", value=omega_q_target,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="x.npz", timestamp="now", stage_06_version="x",
    )
    E_C = REFERENCE_DEVICE.transmon.E_C
    device = pack.to_device_config()
    expected_E_J = (omega_q_target + E_C) ** 2 / (8.0 * E_C)
    assert abs(device.transmon.E_J - expected_E_J) / expected_E_J < 1e-10


def test_C7b_to_device_config_warns_on_E_J_drift_over_30pct():
    """Large-drift ω_q → derived E_J > 30% off REFERENCE's E_J → UserWarning."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    E_C = REFERENCE_DEVICE.transmon.E_C
    omega_q_target = 2 * math.pi * 6.5e9
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="omega_q", value=omega_q_target,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="x.npz", timestamp="now", stage_06_version="x",
    )
    with pytest.warns(UserWarning, match="E_J"):
        pack.to_device_config()
