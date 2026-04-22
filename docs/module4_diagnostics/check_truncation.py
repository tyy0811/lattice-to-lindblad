"""Module 4 diagnostic — Check 2: Hilbert-space truncation independence.

Reproduces the Day-10 verification that the empirical |S_γ1| at the
T_1 = 0.22 µs stress point is not compressed by Hilbert-space truncation.
Default REFERENCE uses (N_transmon=5, N_resonator=15); this also runs
(N_transmon=7, N_resonator=25) — the latter eliminates the N_r=15 photon-
truncation warning at this stress point (mean photon peaks at ~5.8).

Pass criterion: relative change < 5%.

Reproduction:
    python docs/module4_diagnostics/check_truncation.py

Expected output: 1.4% change in |S_γ1| (F_ref shifts slightly, sensitivity
changes only marginally).
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


def _S_gamma_1(device, drive, window):
    h = 0.05
    gamma_ref = device.decoherence.gamma_1
    F = {}
    for label, frac in [("ref", 1.0), ("plus", 1.0 + h), ("minus", 1.0 - h)]:
        new_dec = replace(device.decoherence, gamma_1=gamma_ref * frac)
        new_dev = replace(device, decoherence=new_dec)
        r0 = simulate_readout(new_dev, drive, initial_qubit_state=0)
        r1 = simulate_readout(new_dev, drive, initial_qubit_state=1)
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
    dev_default = replace(REFERENCE_DEVICE, decoherence=stress_dec)

    big_trunc = replace(dev_default.truncation, N_transmon=7, N_resonator=25)
    dev_big = replace(dev_default, truncation=big_trunc)

    S_def, F_def = _S_gamma_1(dev_default, ref_op.drive, ref_op.integration_window)
    S_big, F_big = _S_gamma_1(dev_big, ref_op.drive, ref_op.integration_window)

    print("Check 2 — Truncation independence at T_1 = 0.22 µs, Purcell-off")
    print()
    print(f"Default (N_q={dev_default.truncation.N_transmon}, N_r={dev_default.truncation.N_resonator}):")
    print(f"  F_ref={F_def['ref']:.8f}  |S_γ1| = {abs(S_def):.5f}")
    print()
    print(f"Enlarged (N_q={dev_big.truncation.N_transmon}, N_r={dev_big.truncation.N_resonator}):")
    print(f"  F_ref={F_big['ref']:.8f}  |S_γ1| = {abs(S_big):.5f}")
    print()
    rel = abs(abs(S_big) - abs(S_def)) / max(abs(S_def), 1e-12)
    print(f"Relative change: {rel*100:.3f}%")
    print(f"Verdict: {'PASS' if rel < 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
