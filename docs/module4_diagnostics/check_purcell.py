"""Module 4 diagnostic — Check 3: pure-γ_1 verification with coupling.g = 0.

Reproduces the Day-10 verification that the "Purcell-off" stress tests
have no residual-decay leak. Zeros coupling.g (decouples qubit from
resonator entirely), sets γ_φ = 0 and n_th = 0, and fits γ_eff from
P_|1>(t) decay of an initial |1, vacuum> state. If γ_eff exactly matches
1/T_1 to numerical precision, there are no residual decay channels.

Pass criterion: |γ_eff − γ_1_true| / γ_1_true < 1 ppm.

Reproduction:
    python docs/module4_diagnostics/check_purcell.py

Expected output: relative deviation ~0.000000% (fit residual is from
finite-sample log-linear regression, not from any physical channel).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import qutip as qt

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.lindblad import build_collapse_operators, build_hamiltonian


def main() -> None:
    T1_us = 5.0
    gamma_1_true = 1.0 / (T1_us * 1e-6)

    zero_drive = DriveParams(amplitude=0.0, duration=5e-6, detuning=0.0)
    zero_coupling = replace(REFERENCE_DEVICE.coupling, g=0.0)
    iso_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=gamma_1_true,
        gamma_phi=0.0,
        n_th=0.0,
        purcell_enabled=False,
    )
    dev = replace(REFERENCE_DEVICE, decoherence=iso_dec, coupling=zero_coupling)

    Nq = dev.truncation.N_transmon
    Nr = dev.truncation.N_resonator
    H0, _ = build_hamiltonian(dev, zero_drive)
    c_ops = build_collapse_operators(dev, Nq, Nr)

    psi0 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nr, 0))
    P1 = qt.tensor(qt.basis(Nq, 1) * qt.basis(Nq, 1).dag(), qt.qeye(Nr))

    t_list = np.linspace(0, 3.0 * T1_us * 1e-6, 500)
    result = qt.mesolve(
        H=H0, rho0=psi0, tlist=t_list, c_ops=c_ops, e_ops=[P1],
        options={"nsteps": 100_000, "atol": 1e-14, "rtol": 1e-12},
    )
    P1_t = np.asarray(result.expect[0], dtype=float)

    mask = (t_list > 0) & (t_list < 2.0 * T1_us * 1e-6) & (P1_t > 0.01)
    coeffs = np.polyfit(t_list[mask], np.log(P1_t[mask]), 1)
    gamma_eff = -coeffs[0]
    rel = (gamma_eff - gamma_1_true) / gamma_1_true

    print("Check 3 — Pure-γ_1 with coupling.g = 0, γ_φ = 0, n_th = 0")
    print()
    print(f"  γ_1_true    = {gamma_1_true:.10e} 1/s")
    print(f"  γ_eff (fit) = {gamma_eff:.10e} 1/s")
    print(f"  relative    = {rel*100:+.8f}%")
    print()
    print(f"Verdict: {'PASS' if abs(rel) < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
