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
