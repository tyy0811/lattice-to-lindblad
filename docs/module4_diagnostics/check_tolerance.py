"""Module 4 diagnostic — Check 1: tolerance independence of |S_γ1|.

Reproduces the Day-10 verification that the empirical |S_γ1| at the
T_1 = 0.22 µs stress point is not compressed by the default mesolve
tolerances (atol=1e-10, rtol=1e-8). Runs |S_γ1| at default and at
100× tighter tolerances (atol=1e-12, rtol=1e-10).

Pass criterion: relative change < 5%.

Reproduction:
    python docs/module4_diagnostics/check_tolerance.py

Expected output: both computations agree to ~0.000% (F values bit-identical).
"""
from __future__ import annotations

import math
from dataclasses import replace

from scipy.stats import norm

from dispersive_readout.physics.config import REFERENCE_DEVICE
from dispersive_readout.physics.readout_model import simulate_readout
from dispersive_readout.analysis.operating_point import get_reference_operating_point


def _F_analytic(c0, c1, window, kappa):
    separation = abs(c1 - c0)
    if separation == 0.0:
        return 0.5
    t0, t1 = window
    sigma = math.sqrt((t1 - t0) / (4.0 * kappa))
    return float(norm.cdf(separation / sigma / 2.0))


def _S_gamma_1(device, drive, window, solver_opts):
    h = 0.05
    gamma_ref = device.decoherence.gamma_1
    F = {}
    for label, frac in [("ref", 1.0), ("plus", 1.0 + h), ("minus", 1.0 - h)]:
        new_dec = replace(device.decoherence, gamma_1=gamma_ref * frac)
        new_dev = replace(device, decoherence=new_dec)
        r0 = simulate_readout(new_dev, drive, initial_qubit_state=0, solver_options=solver_opts)
        r1 = simulate_readout(new_dev, drive, initial_qubit_state=1, solver_options=solver_opts)
        F[label] = _F_analytic(
            r0.integrated_iq(window), r1.integrated_iq(window), window,
            new_dev.resonator.kappa,
        )
    return (math.log(F["plus"]) - math.log(F["minus"])) / (2.0 * h), F


def main() -> None:
    ref_op = get_reference_operating_point(n_shots=10_000)
    stress_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=1.0 / (0.22e-6),
        purcell_enabled=False,
    )
    stress_dev = replace(REFERENCE_DEVICE, decoherence=stress_dec)

    S_def, F_def = _S_gamma_1(stress_dev, ref_op.drive, ref_op.integration_window, None)
    S_tight, F_tight = _S_gamma_1(
        stress_dev, ref_op.drive, ref_op.integration_window,
        {"atol": 1e-12, "rtol": 1e-10, "nsteps": 50_000},
    )

    print("Check 1 — Tolerance independence at T_1 = 0.22 µs, Purcell-off")
    print()
    print(f"Default (atol=1e-10, rtol=1e-8):")
    print(f"  F_ref={F_def['ref']:.8f}  F+={F_def['plus']:.8f}  F-={F_def['minus']:.8f}")
    print(f"  |S_γ1| = {abs(S_def):.5f}")
    print()
    print(f"Tight (atol=1e-12, rtol=1e-10):")
    print(f"  F_ref={F_tight['ref']:.8f}  F+={F_tight['plus']:.8f}  F-={F_tight['minus']:.8f}")
    print(f"  |S_γ1| = {abs(S_tight):.5f}")
    print()
    rel = abs(abs(S_tight) - abs(S_def)) / max(abs(S_def), 1e-12)
    print(f"Relative change: {rel*100:.3f}%")
    print(f"Verdict: {'PASS' if rel < 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
