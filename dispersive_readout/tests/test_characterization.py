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


# -- Point-estimate fit tests (full bootstrap uncertainty lives in Task 9) ---

def test_fit_rabi_point_estimate_recovers_epsilon_pi_within_3pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    from dispersive_readout.characterization.fitting import fit_rabi
    eps_pi_truth = 2 * math.pi * 50e6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0, drive_amplitude_uncertainty=0.0)
    trace = generate_rabi_trace(eps_pi_truth, 2 * math.pi * 4.5e9, noise, seed=10)
    fp = fit_rabi(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - eps_pi_truth) / eps_pi_truth
    assert rel < 0.03, f"fit_rabi rel={rel:.3%}"
    assert fp.name == "epsilon_pi"


def test_fit_ramsey_point_estimate_recovers_omega_q_within_0_1pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q_truth = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_ramsey_trace(omega_q_truth, T_2_star=T_2_star_truth, noise=noise, seed=11)
    fp_omega, fp_T2star = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp_omega.value - omega_q_truth) / omega_q_truth
    assert rel < 1e-3, f"fit_ramsey omega_q rel={rel:.3e}"
    rel_T2 = abs(fp_T2star.value - T_2_star_truth) / T_2_star_truth
    assert rel_T2 < 0.15


def test_fit_t1_point_estimate_recovers_T1_within_5pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    from dispersive_readout.characterization.fitting import fit_t1
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, seed=12)
    fp = fit_t1(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_1_truth) / T_1_truth
    assert rel < 0.05


def test_fit_t2_echo_point_estimate_recovers_T2_within_5pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t2_echo_trace
    from dispersive_readout.characterization.fitting import fit_t2_echo
    T_2_truth = 40e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t2_echo_trace(T_2_truth, noise, seed=13)
    fp = fit_t2_echo(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_2_truth) / T_2_truth
    assert rel < 0.05


def test_parametric_bootstrap_produces_nonzero_uncertainty_on_noisy_trace():
    """With non-zero drift + shot noise, bootstrap uncertainty must be > 0 and
    larger than the covariance-matrix SE by at least a factor of 1.5 (the
    gap amendment 3 is designed to reveal)."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q_truth = 2 * math.pi * 4.5e9
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    trace = generate_ramsey_trace(omega_q_truth, T_2_star=20e-6, noise=noise, seed=20)
    fp_omega_pe, _ = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    fp_omega_bs, _ = fit_ramsey(trace, bootstrap_samples=50, seed=42)
    assert fp_omega_bs.n_bootstrap == 50
    assert fp_omega_bs.uncertainty > 0
    assert fp_omega_bs.uncertainty > 1.5 * fp_omega_pe.uncertainty, (
        f"bootstrap SE {fp_omega_bs.uncertainty:.3e} not > 1.5× covariance SE {fp_omega_pe.uncertainty:.3e}"
    )


# -- Recovery harness --------------------------------------------------------

def test_fit_one_device_returns_four_RecoveryResults():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import DeviceGroundTruth, fit_one_device
    d = DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6, omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6, thermal_offset=0.0,
        ramsey_detuning=2 * math.pi * 1e6,
    )
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    out = fit_one_device(d, noise, seed=42)
    assert len(out) == 4
    names = {r.parameter_name for r in out}
    assert names == {"T_1", "T_2_echo", "omega_q", "epsilon_pi"}


def test_generate_synthetic_device_family_rejects_T2_gt_2T1():
    from dispersive_readout.characterization.recovery import generate_synthetic_device_family
    devices = generate_synthetic_device_family(n_devices=50, seed=42)
    assert len(devices) == 50
    for d in devices[2:]:
        assert d.T_2_echo <= 2.0 * d.T_1 * 0.95 + 1e-18, (
            f"Device with T_2={d.T_2_echo:.2e} exceeds 2·T_1·0.95={2 * d.T_1 * 0.95:.2e}"
        )
    assert devices[0].ramsey_detuning == 0.0
    assert devices[1].thermal_offset == 0.08


def test_fit_one_device_is_deterministic_under_same_seed():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import DeviceGroundTruth, fit_one_device
    d = DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6, omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
    )
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    a = fit_one_device(d, noise, seed=123)
    b = fit_one_device(d, noise, seed=123)
    for ra, rb in zip(a, b):
        assert ra.parameter_name == rb.parameter_name
        assert ra.fitted_value == rb.fitted_value


# -- C3: recovery-coverage regression gate (amendment 9) ---------------------

@pytest.mark.slow
def test_C3_recovery_coverage_matches_committed_artifact():
    """Re-run 50-device harness at SEED=42 and match the committed artifact
    within ±2% per parameter. Regression gate; if this fails, diagnose the
    fitter before regenerating the artifact."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import (
        run_recovery_harness, load_committed_coverage_report,
    )
    observed_reports, _ = run_recovery_harness(n_devices=50, noise=NoiseModelParams(), seed=42)
    committed = load_committed_coverage_report(
        "06_Dispersive_Readout/figures/recovery_coverage_report.yaml"
    )
    for name, rep in observed_reports.items():
        ref = committed[name]
        for field_name in ("coverage_1_sigma", "coverage_2_sigma"):
            delta = abs(getattr(rep, field_name) - getattr(ref, field_name))
            assert delta < 0.02, (
                f"{name}.{field_name} regression: observed {getattr(rep, field_name):.2%} "
                f"vs committed {getattr(ref, field_name):.2%} (Δ={delta:.2%})"
            )


# -- C5: CLI smoke tests ----------------------------------------------------

def _run_cli(args: list[str]) -> int:
    from dispersive_readout.characterization.cli import main
    return main(argv=args)


def test_C5a_cli_generate_synthetic(tmp_path):
    out = tmp_path / "synthetic.npz"
    rc = _run_cli(["--generate-synthetic", "--output", str(out), "--seed", "42"])
    assert rc == 0
    from dispersive_readout.characterization.protocols import load_trace_bundle
    traces = load_trace_bundle(str(out))
    assert {t.protocol for t in traces} == {"rabi", "ramsey", "t1", "t2_echo"}


def test_C5b_cli_full_pipeline_generate_then_fit(tmp_path):
    bundle = tmp_path / "synth.npz"
    params = tmp_path / "params.yaml"
    rc1 = _run_cli(["--generate-synthetic", "--output", str(bundle), "--seed", "42"])
    assert rc1 == 0
    rc2 = _run_cli(["--traces", str(bundle), "--output", str(params), "--bootstrap-samples", "20"])
    assert rc2 == 0
    import yaml
    with open(params) as f:
        data = yaml.safe_load(f)
    names = {p["name"] for p in data["fitted_parameters"]}
    assert {"T_1", "T_2_echo", "omega_q", "epsilon_pi"}.issubset(names)


def test_C5c_cli_help_has_no_todo(capsys):
    with pytest.raises(SystemExit):
        _run_cli(["--help"])
    out = capsys.readouterr().out
    for forbidden in ("TODO", "TBD", "FIXME", "XXX"):
        assert forbidden not in out, f"--help text contains '{forbidden}'"


def test_C5d_cli_rejects_conflicting_flags(tmp_path):
    """--traces + --generate-synthetic is ambiguous; must exit non-zero with a clear error."""
    rc = _run_cli(["--traces", "x.npz", "--generate-synthetic", "--output", str(tmp_path / "o.yaml")])
    assert rc != 0


# -- C6: edge cases ----------------------------------------------------------

def test_C6a_ramsey_zero_detuning_envelope_only_path():
    """Ramsey with Δω=0 uses the envelope-only fallback and returns a T2* within 20%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_ramsey_trace(omega_q, T_2_star=T_2_star_truth, noise=noise,
                                  omega_drive_offset=0.0, seed=99)
    fp_omega, fp_T2 = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    assert fp_T2.name == "T_2_star"
    rel = abs(fp_T2.value - T_2_star_truth) / T_2_star_truth
    assert rel < 0.20


def test_C6b_t1_with_elevated_thermal_no_downward_bias():
    """T1 fit with thermal_offset=0.08 recovers T1 within 10% (thermal absorbed by A)."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    from dispersive_readout.characterization.fitting import fit_t1
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, thermal_offset=0.08, seed=7)
    fp = fit_t1(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_1_truth) / T_1_truth
    assert rel < 0.10


def test_readout_asymmetric_true_fails_loudly_not_silently_symmetric():
    """NoiseModelParams(readout_asymmetric=True) must cause the generator to
    raise NotImplementedError, not silently produce symmetric traces (Codex
    adversarial finding #2)."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import (
        generate_rabi_trace, generate_ramsey_trace,
        generate_t1_trace, generate_t2_echo_trace,
    )
    noise = NoiseModelParams(n_shots_per_point=1000, drift_amplitude_Hz=0.0,
                             readout_asymmetric=True)
    with pytest.raises(NotImplementedError, match="symmetric"):
        generate_rabi_trace(2 * math.pi * 50e6, 2 * math.pi * 4.5e9, noise, seed=0)
    with pytest.raises(NotImplementedError, match="symmetric"):
        generate_ramsey_trace(2 * math.pi * 4.5e9, T_2_star=20e-6, noise=noise, seed=0)
    with pytest.raises(NotImplementedError, match="symmetric"):
        generate_t1_trace(30e-6, noise, seed=0)
    with pytest.raises(NotImplementedError, match="symmetric"):
        generate_t2_echo_trace(40e-6, noise, seed=0)


def test_binomial_ci_at_perfect_coverage_does_not_collapse_to_unit_interval():
    """Wilson CI at n=50, k=50 (p=1) must produce a lower bound below 0.95,
    so "2σ CI includes 95%" is a meaningful gate at the boundary.
    Wald CI trivially gave [1, 1] here; Wilson gives roughly [0.93, 1]."""
    from dispersive_readout.characterization.recovery import _binomial_2sigma_ci
    lo, hi = _binomial_2sigma_ci(1.0, 50)
    assert hi <= 1.0 + 1e-12
    assert lo < 0.95, f"Wilson lower bound at p=1, n=50 is {lo:.3f}, should be < 0.95"
    assert lo > 0.85, f"Wilson lower bound at p=1, n=50 is {lo:.3f}, should be > 0.85"
    lo0, hi0 = _binomial_2sigma_ci(0.0, 50)
    assert lo0 >= -1e-12
    assert hi0 > 0.05, f"Wilson upper bound at p=0, n=50 is {hi0:.3f}, should be > 0.05"


def test_C6c_rabi_amplitude_span_too_small_sets_reject_flag():
    """Spec §1.1: <1.5 visible oscillations → reject_flag set.

    Under-sampled Rabi (0 → 0.6·ε_π, half an oscillation) must land a
    fit but set ``reject_flag='insufficient_oscillations'`` so downstream
    coverage aggregation excludes it (Codex F3 follow-up).
    """
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    from dispersive_readout.characterization.fitting import fit_rabi
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0, drive_amplitude_uncertainty=0.0)
    eps_pi_truth = 2 * math.pi * 50e6
    trace = generate_rabi_trace(eps_pi_truth, 2 * math.pi * 4.5e9, noise, seed=8, amplitude_span_mult=(0.0, 0.6))
    fp = fit_rabi(trace, bootstrap_samples=0, seed=42)
    assert fp.goodness_of_fit >= 0
    assert fp.reject_flag == "insufficient_oscillations", (
        f"expected reject_flag='insufficient_oscillations', got {fp.reject_flag!r}"
    )


def test_rabi_full_span_does_not_trigger_reject_flag():
    """Sanity: the default 2.5·ε_π span (2.5 oscillations) must NOT flag."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    from dispersive_readout.characterization.fitting import fit_rabi
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0, drive_amplitude_uncertainty=0.0)
    eps_pi_truth = 2 * math.pi * 50e6
    trace = generate_rabi_trace(eps_pi_truth, 2 * math.pi * 4.5e9, noise, seed=8)
    fp = fit_rabi(trace, bootstrap_samples=0, seed=42)
    assert fp.reject_flag is None


def test_ramsey_auto_escalates_to_stretched_on_gaussian_envelope():
    """F1: synthetic trace with true Gaussian envelope (n=2) — the
    auto-escalator must fire and recover n within ±0.3 of 2.

    Built manually (generate_ramsey_trace only produces plain-exp envelope)
    to validate the fitter's stretched-exp capability independently of
    whether spec-default 1/f drift produces envelope mismatch in practice.
    """
    from dispersive_readout.characterization.fitting import fit_ramsey
    from dispersive_readout.characterization.protocols import TraceData
    omega_q = 2 * math.pi * 4.5e9
    omega_drive_offset = 2 * math.pi * 1e6
    T_2_true = 20e-6
    n_true = 2.0
    delays = np.linspace(0.0, 40e-6, 101)
    envelope = np.exp(-((delays / T_2_true) ** n_true))
    P_true = 0.5 - 0.5 * envelope * np.cos(omega_drive_offset * delays)
    rng = np.random.default_rng(42)
    n_shots = 5000
    k = rng.binomial(n_shots, np.clip(P_true, 0.0, 1.0))
    P_obs = k / n_shots
    P_se = np.sqrt(np.clip(P_true, 1e-12, 1 - 1e-12) * (1 - np.clip(P_true, 1e-12, 1 - 1e-12)) / n_shots)
    trace = TraceData(
        protocol="ramsey", sweep_axis="delay",
        sweep_values=delays, P1=P_obs, P1_uncertainty=P_se,
        metadata={
            "ground_truth": {"omega_q": omega_q, "T_2_star": T_2_true,
                             "omega_drive_offset": omega_drive_offset},
            "noise": {"n_shots_per_point": n_shots},
            "seed": 42,
        },
    )
    fp_o, fp_t = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    assert fp_t.envelope_model == "stretched", (
        f"expected stretched on Gaussian-envelope trace, got {fp_t.envelope_model}"
    )
    assert fp_t.stretch_exponent is not None
    assert abs(fp_t.stretch_exponent - 2.0) < 0.3, (
        f"recovered n = {fp_t.stretch_exponent:.3f}, expected 2.0 ± 0.3"
    )


def test_ramsey_clean_trace_does_not_escalate():
    """F1: a clean Ramsey trace (plain exp envelope, no drift) must NOT
    trigger the stretched escalation."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    # Zero drift + high shot count → redchi ~ 1, escalation should not fire.
    noise = NoiseModelParams(n_shots_per_point=10000, drift_amplitude_Hz=0.0)
    trace = generate_ramsey_trace(2 * math.pi * 4.5e9, T_2_star=20e-6, noise=noise, seed=99)
    fp_o, fp_t = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    assert fp_t.envelope_model == "exponential"
    assert fp_t.stretch_exponent is None
    assert fp_t.goodness_of_fit < 3.0, (
        f"clean-trace redchi = {fp_t.goodness_of_fit:.2f}, expected < 3"
    )


def test_ramsey_fig3_seed_escalation_evaluated_but_rejected():
    """F1 null finding: on the fig3 seed (SEED=42, spec-default noise with
    10 kHz 1/f drift), the stretched fit is evaluated but does NOT beat
    plain — residual structure under 1/f drift is phase-jitter, not
    envelope-shape mismatch. Locking this as a regression: if this test
    starts returning 'stretched', the noise model or fitter changed."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    noise = NoiseModelParams()
    trace = generate_ramsey_trace(2 * math.pi * 4.5e9, T_2_star=20e-6, noise=noise, seed=42)
    fp_o, fp_t = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    assert fp_t.envelope_model == "exponential"
    assert fp_t.goodness_of_fit > 3.0  # (auto-gate WAS evaluated)


def test_fit_t2_echo_force_stretched_returns_stretched():
    """F1.2: force_stretched=True bypasses the redchi gate for testing."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t2_echo_trace
    from dispersive_readout.characterization.fitting import fit_t2_echo
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t2_echo_trace(40e-6, noise, seed=3)
    fp = fit_t2_echo(trace, force_stretched=True, bootstrap_samples=0, seed=42)
    assert fp.envelope_model == "stretched"
    assert fp.stretch_exponent is not None
    # n should land near 1 since the ground-truth envelope IS plain exp.
    assert 0.5 < fp.stretch_exponent < 1.5


def test_coverage_report_tallies_rejects_and_excludes_from_on_accepted():
    """Inject a reject-flagged RecoveryResult directly and verify
    CoverageReport.coverage_{1,2}_sigma_on_accepted excludes it while
    raw coverage still includes it (for comparison)."""
    from dispersive_readout.characterization.recovery import (
        CoverageReport, RecoveryResult, _binomial_2sigma_ci,
    )
    records = [
        RecoveryResult("epsilon_pi", 1.0, 1.0, 0.01, 0.0, True, True, reject_flag=None),
        RecoveryResult("epsilon_pi", 1.0, 1.0, 0.01, 0.0, True, True, reject_flag=None),
        # Flagged fit: huge bogus uncertainty → trivially "within 1σ"
        RecoveryResult("epsilon_pi", 1.0, 10.0, 1000.0, 0.009, True, True,
                       reject_flag="insufficient_oscillations"),
    ]
    n = len(records)
    cov1 = sum(r.within_1_sigma for r in records) / n
    accepted = [r for r in records if r.reject_flag is None]
    cov1_acc = sum(r.within_1_sigma for r in accepted) / len(accepted)
    c1_lo, c1_hi = _binomial_2sigma_ci(cov1, n)
    rep = CoverageReport(
        parameter_name="epsilon_pi", n_devices=n,
        coverage_1_sigma=cov1, coverage_2_sigma=cov1,
        coverage_1_sigma_ci_low=c1_lo, coverage_1_sigma_ci_high=c1_hi,
        coverage_2_sigma_ci_low=c1_lo, coverage_2_sigma_ci_high=c1_hi,
        bias=0.0, bias_uncertainty=0.0,
        n_rejected=1, coverage_1_sigma_on_accepted=cov1_acc,
        coverage_2_sigma_on_accepted=cov1_acc,
    )
    assert rep.n_rejected == 1
    assert rep.coverage_1_sigma == 1.0  # raw includes the flagged one
    assert rep.coverage_1_sigma_on_accepted == 1.0  # 2/2 accepted
    # Contrast with a case where the flagged fit legitimately misses:
    records2 = list(records[:2]) + [
        RecoveryResult("epsilon_pi", 1.0, 5.0, 0.001, 4000.0, False, False,
                       reject_flag="insufficient_oscillations"),
    ]
    cov1_raw = sum(r.within_1_sigma for r in records2) / len(records2)
    acc2 = [r for r in records2 if r.reject_flag is None]
    cov1_on_acc = sum(r.within_1_sigma for r in acc2) / len(acc2)
    assert cov1_raw == 2 / 3
    assert cov1_on_acc == 1.0
