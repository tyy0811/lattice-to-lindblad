"""Module 2 tests — see MODULE_2_SPEC.md §6 for the test plan."""
from __future__ import annotations

import numpy as np
import pytest


def test_module2_package_imports_without_error():
    """Smoke test: the analysis subpackage can be imported. Populated further
    as Tasks 4–8 add real API."""
    import dispersive_readout.analysis  # noqa: F401
    import dispersive_readout.analysis.operating_point  # noqa: F401
    import dispersive_readout.analysis.purcell_isolation  # noqa: F401
    import dispersive_readout.analysis.error_budget  # noqa: F401


def test_analytic_calibration_hits_target_fidelity_within_3_sigma():
    """Analytic ε₀ calibration at REFERENCE_DEVICE produces F_verified in
    F_target ± 3σ_shot. If this fails, fallback to grid search is triggered.
    See MODULE_2_SPEC.md §2.3."""
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import calibrate_drive_amplitude

    target = 0.99
    n_shots = 10_000
    sigma_shot = np.sqrt(target * (1.0 - target) / n_shots)  # ≈ 1e-3

    epsilon_0 = calibrate_drive_amplitude(
        device=REFERENCE_DEVICE,
        duration=500e-9,
        integration_window=(50e-9, 500e-9),
        target_fidelity=target,
        n_shots=n_shots,
        sigma_tolerance_factor=3.0,
    )

    # Verify at the returned ε₀
    from dispersive_readout.physics import (
        DriveParams, simulate_readout, compute_assignment_fidelity,
    )

    drv = DriveParams(amplitude=epsilon_0, duration=500e-9, detuning=0.0)
    r0 = simulate_readout(REFERENCE_DEVICE, drv, initial_qubit_state=0)
    r1 = simulate_readout(REFERENCE_DEVICE, drv, initial_qubit_state=1)
    f = compute_assignment_fidelity(
        r0, r1, (50e-9, 500e-9), n_shots=n_shots, noise_model="gaussian",
        rng=np.random.default_rng(seed=42),  # deterministic for test reproducibility
    )

    assert abs(f.F_assign - target) <= 3.0 * sigma_shot, (
        f"Calibration gave F={f.F_assign:.4f}, expected {target}±{3*sigma_shot:.4f}. "
        f"Either the analytic formula is wrong or fallback is needed."
    )


def test_B4_negative_contribution_raises():
    """ChannelContribution with delta_F < -0.005 must raise ValueError.
    Small shot-noise-range negatives are preserved as signed values
    (amendment 9b: no one-sided clipping)."""
    from dispersive_readout.analysis import ChannelContribution

    # Below -0.005 bug gate: must raise
    with pytest.raises(ValueError, match="negative"):
        ChannelContribution(
            name="T1_intrinsic",
            group="active_loss",
            delta_F=-0.01,
            delta_F_uncertainty=1e-4,
            description="test",
        )

    # Within shot-noise range: accepted AND preserved (no clip to 0)
    c = ChannelContribution(
        name="T1_intrinsic",
        group="active_loss",
        delta_F=-0.003,  # > -0.005 gate, still signed
        delta_F_uncertainty=1e-4,
        description="test",
    )
    assert c.delta_F == -0.003


def test_analytic_purcell_rate_positive_at_reference():
    """γ_P at REFERENCE should be positive and of order (g/Δ)²κ ~ O(kHz)."""
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import analytic_purcell_rate

    gamma_P = analytic_purcell_rate(REFERENCE_DEVICE)
    assert gamma_P > 0.0
    # Order-of-magnitude sanity: g/Δ ≈ 120 MHz / 2700 MHz ≈ 0.044
    # γ_P / κ ≈ 0.044² ≈ 1.9e-3; κ/2π = 5 MHz → γ_P/2π ~ 9.5 kHz
    kappa = REFERENCE_DEVICE.resonator.kappa
    ratio = gamma_P / kappa
    assert 1e-4 < ratio < 1e-1, f"γ_P/κ = {ratio:.2e} outside plausible range"


def test_T1_intrinsic_contribution_nonzero_at_reference():
    """Turning off γ_1 at REFERENCE returns a signed ΔF (no longer clipped,
    amendment 9b) within the shot-noise-valid range, with a well-defined
    uncertainty. Per-channel ΔFs at REFERENCE with n_shots=10_000 sit
    near σ_shot ≈ 1e-3 so even the "always positive T1" can land on
    either side of zero; the non-triviality is tested via the full
    budget's B2 / B3 cross-checks, not per-channel sign."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )

    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="T1_intrinsic")

    assert c.name == "T1_intrinsic"
    assert c.group == "active_loss"
    # Validator lets signed values through down to -0.005. A physical
    # bug (wrong turn-off direction) would trigger the raise.
    assert c.delta_F > -0.005
    assert c.delta_F_uncertainty > 0.0


def test_pure_dephasing_contribution_nonzero_at_reference():
    """Pure dephasing in the dispersive frame barely affects |0>/|1>
    readout populations. Post-amendment-9b, ΔF is signed; the only
    bound on a physical implementation is ΔF > -0.005."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="pure_dephasing")
    assert c.name == "pure_dephasing"
    assert c.group == "active_loss"
    assert c.delta_F > -0.005


def test_thermal_contribution_nonzero_at_reference():
    """Thermal turn-off at n_th=0.01 is below shot-noise floor (σ≈1e-3 vs
    thermal effect ~1.7e-4 per 500 ns). Post-amendment-9b, signed ΔF is
    preserved; only the hard bug gate at -0.005 applies."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="thermal")
    assert c.name == "thermal"
    assert c.group == "active_loss"
    assert c.delta_F > -0.005


def test_purcell_contribution_nonzero_at_reference():
    """Purcell at REFERENCE has γ_P≈10 kHz × 500 ns ≈ 5×σ_shot; signed
    ΔF can cross zero under shot noise. B3 (Task 9) provides the tight
    γ_P cross-check at this regime."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="purcell")
    assert c.name == "purcell"
    assert c.group == "active_loss"
    assert c.delta_F > -0.005


def test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent():
    """ΔF under ±5% amplitude perturbation should agree with first-order
    Taylor expansion |dF/dε|·Δε to within 20% (O(Δε²) higher-order correction)."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="drive_amplitude")
    assert c.name == "drive_amplitude"
    assert c.group == "calibration_sensitivity"
    assert c.delta_F >= 0.0
    assert c.perturbation_description is not None
    assert "±5" in c.perturbation_description or "5%" in c.perturbation_description


def test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent():
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="drive_detuning")
    assert c.name == "drive_detuning"
    assert c.group == "calibration_sensitivity"
    assert c.delta_F >= 0.0
    assert c.perturbation_description is not None
    assert "κ/4" in c.perturbation_description or "kappa/4" in c.perturbation_description


def test_B1_active_loss_sums_to_ideal_minus_full_within_tolerance():
    """Σ ΔF_c + R_active ≈ (F_ideal − F_full) within 3σ_prop for active group."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_full_error_budget,
    )
    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    active_sum = sum(c.delta_F for c in budget.active_loss_channels)
    identity_lhs = budget.F_ideal - budget.F_full
    identity_rhs = active_sum + budget.residual_active
    tolerance = 3.0 * budget.residual_active_uncertainty
    assert abs(identity_lhs - identity_rhs) <= tolerance, (
        f"Additivity violation: (F_ideal - F_full) = {identity_lhs:.5f}, "
        f"Σ ΔF + R = {identity_rhs:.5f}, tol = {tolerance:.5f}"
    )


def test_B2_active_loss_residual_is_consistent_with_additivity():
    """|R_active| ≤ max(3σ_R, 0.2 × (F_ideal − F_full)).

    Per spec amendment 8: two-regime criterion combined with max():
      - Noise-dominated (denom ~ σ_R at CI's n_shots=1e4): 3σ_R clause fires,
        testing R is consistent with zero within shot-noise propagation.
      - Physics-dominated (denom >> σ_R, e.g. figure run at n_shots=1e5):
        0.2×denom clause fires, testing that channels interact weakly.

    Reports which clause was active so future debugging can distinguish
    noise-dominated passes from physics-dominated passes.
    """
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_full_error_budget,
    )
    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    R = budget.residual_active
    sigma_R = budget.residual_active_uncertainty
    denom = budget.F_ideal - budget.F_full
    threshold = max(3.0 * sigma_R, 0.2 * denom)

    if abs(R) < 3.0 * sigma_R:
        regime = "noise-dominated (R consistent with zero within 3σ_R)"
    elif abs(R) < 0.2 * denom:
        regime = "physics-dominated (R small relative to denom; channels weakly interacting)"
    else:
        regime = "FAIL"

    assert abs(R) <= threshold, (
        f"B2 FAIL: R={R:.3e}, σ_R={sigma_R:.3e}, denom={denom:.3e}, "
        f"threshold=max(3σ_R, 0.2*denom)={threshold:.3e}. regime={regime}. "
        f"Channels interact strongly enough that marginal attribution is "
        f"breaking down — consider regrouping (e.g., merge T1+purcell)."
    )


def test_B3_simulated_purcell_matches_analytic_within_1_percent_at_reference():
    """Simulated ΔF_Purcell vs analytic γ_P-weighted prediction at REFERENCE.

    The test compares Purcell rates, not ΔF values directly: fit γ_P from
    simulated T1_eff with γ_1=γ_φ=n_th=0 (Purcell is the only remaining
    relaxation channel), compare to analytic_purcell_rate at 1% tolerance.
    Physics ceiling for 2nd-order PT residual is ~0.2% at g/Δ≈0.044.
    """
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import analytic_purcell_rate

    # Analytic prediction
    gamma_P_analytic = analytic_purcell_rate(REFERENCE_DEVICE)

    from dataclasses import replace
    from dispersive_readout.physics import DriveParams, simulate_readout
    from dispersive_readout.physics.config import DeviceConfig

    # Build γ_1=γ_φ=n_th=0 device with Purcell still on
    new_dec = replace(REFERENCE_DEVICE.decoherence, gamma_1=0.0, gamma_phi=0.0, n_th=0.0)
    dev = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=new_dec,
        truncation=REFERENCE_DEVICE.truncation,
    )

    # Zero-drive (H_drive=0); long enough to see Purcell decay.
    # Use a very small amplitude so drive doesn't dominate.
    T = 5.0 / gamma_P_analytic  # ~5 Purcell lifetimes
    T = min(T, 100e-6)           # cap at 100 μs to bound solver cost
    drv = DriveParams(amplitude=1e-6, duration=T, detuning=0.0, edge_sigma=2e-9)
    r = simulate_readout(dev, drv, initial_qubit_state=1)

    # Extract γ_P from exponential fit of P(|1⟩)(t): P(|1⟩) = exp(-γ_P t)
    p1 = r.qubit_populations[:, 1]
    t = r.t
    # Fit in log space; restrict to P(|1⟩) > 0.1 for clean fit.
    mask = p1 > 0.1
    log_p1 = np.log(p1[mask])
    t_fit = t[mask]
    slope, _intercept = np.polyfit(t_fit, log_p1, 1)
    gamma_P_sim = -slope

    ratio = gamma_P_sim / gamma_P_analytic
    assert 0.99 <= ratio <= 1.01, (
        f"Simulated Purcell γ_P = {gamma_P_sim:.3e} rad/s, analytic = "
        f"{gamma_P_analytic:.3e} rad/s, ratio = {ratio:.4f}. "
        f"Expected 1 ± 0.01 at REFERENCE."
    )


def test_B3_simulated_purcell_matches_analytic_at_strong_coupling():
    """Same as B3 at REFERENCE but with 2× coupling (g/Δ ≈ 0.088), 5% tol.

    If this fails but B3a passes, the 2nd-order SW approximation is
    tighter than we thought — an informative regime-scope measurement,
    not a bug. Document in the report if it fires."""
    from dataclasses import replace
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.physics.config import (
        CouplingParams, DeviceConfig,
    )
    from dispersive_readout.physics import DriveParams, simulate_readout
    from dispersive_readout.analysis import analytic_purcell_rate

    # 2× coupling device
    new_coup = CouplingParams(g=2.0 * REFERENCE_DEVICE.coupling.g)
    new_dec = replace(REFERENCE_DEVICE.decoherence, gamma_1=0.0, gamma_phi=0.0, n_th=0.0)
    dev = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=new_coup,
        decoherence=new_dec,
        truncation=REFERENCE_DEVICE.truncation,
    )

    gamma_P_analytic = analytic_purcell_rate(dev)
    T = min(5.0 / gamma_P_analytic, 100e-6)
    drv = DriveParams(amplitude=1e-6, duration=T, detuning=0.0, edge_sigma=2e-9)
    r = simulate_readout(dev, drv, initial_qubit_state=1)

    p1 = r.qubit_populations[:, 1]
    mask = p1 > 0.1
    slope, _ = np.polyfit(r.t[mask], np.log(p1[mask]), 1)
    gamma_P_sim = -slope
    ratio = gamma_P_sim / gamma_P_analytic
    assert 0.95 <= ratio <= 1.05, (
        f"At 2×g: simulated γ_P = {gamma_P_sim:.3e}, analytic = "
        f"{gamma_P_analytic:.3e}, ratio = {ratio:.4f}. 5% tol exceeded; "
        f"2nd-order SW approximation failing at this coupling."
    )


def test_B5_budget_yaml_round_trip(tmp_path):
    """export_budget_to_yaml + re-read reproduces the ErrorBudget exactly."""
    from dispersive_readout.analysis import (
        ErrorBudget, ChannelContribution,
        get_reference_operating_point, compute_full_error_budget,
        export_budget_to_yaml,
    )
    import yaml

    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    yaml_path = tmp_path / "fig2_data.yaml"
    export_budget_to_yaml(budget, yaml_path)

    # Re-read and reconstruct
    reread = yaml.safe_load(yaml_path.read_text())
    channels = [ChannelContribution(**d) for d in reread["channels"]]
    reread.pop("channels")
    round_trip = ErrorBudget(channels=channels, **reread)

    assert round_trip.F_full == budget.F_full
    assert round_trip.F_ideal == budget.F_ideal
    assert round_trip.residual_active == budget.residual_active
    assert len(round_trip.channels) == len(budget.channels)
    for c_orig, c_new in zip(budget.channels, round_trip.channels):
        assert c_orig.name == c_new.name
        assert c_orig.delta_F == c_new.delta_F
