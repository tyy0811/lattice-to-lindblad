#!/usr/bin/env python3
"""Stage 06 Figure 1 driver — dispersive-readout simulator demonstration.

Produces `figures/dispersive_readout_simulation.png` with three panels:
  (a) IQ trajectories for initial |0> and |1>
  (b) SNR vs integration time, with short-τ ∝ √τ asymptote overlay
  (c) Assignment fidelity vs κ/|χ|, with vertical marker at κ/|χ| = 2

The numerical χ used in panel (c) is extracted from the dressed JC
spectrum (dispersive.dispersive_shift_from_simulation).

Matches the 01–05 stage-script convention: walks up from the script
location to the repo root (identified by the presence of a sibling
``dispersive_readout`` package or a ``.git`` directory), prepends that
path to sys.path, then imports from the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "dispersive_readout").exists() or (_p / ".git").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    DriveParams,
    ResonatorParams,
)
from dispersive_readout.physics.dispersive import dispersive_shift_from_simulation
from dispersive_readout.physics.readout_model import (
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)

_TWO_PI = 2.0 * math.pi

OUTPUT = Path(__file__).resolve().parent / "figures" / "dispersive_readout_simulation.png"

# Drive amplitude for panels (a) and (b): small enough to keep mean photon
# number ~0.2, well below the N_resonator=15 Fock cutoff. Panel (c)
# sweeps κ/|χ| and uses the same drive.
_DRIVE = DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)


def _panel_a_iq_trajectories(ax) -> None:
    r0 = simulate_readout(REFERENCE_DEVICE, _DRIVE, initial_qubit_state=0)
    r1 = simulate_readout(REFERENCE_DEVICE, _DRIVE, initial_qubit_state=1)
    i0, q0 = r0.a_expectation.real, r0.a_expectation.imag
    i1, q1 = r1.a_expectation.real, r1.a_expectation.imag
    ax.plot(i0, q0, color="#1f77b4", lw=1.6, label="|0⟩")
    ax.plot(i1, q1, color="#d62728", lw=1.6, label="|1⟩")
    ax.plot(i0[0], q0[0], "o", color="#1f77b4", markersize=5)
    ax.plot(i0[-1], q0[-1], "s", color="#1f77b4", markersize=5)
    ax.plot(i1[0], q1[0], "o", color="#d62728", markersize=5)
    ax.plot(i1[-1], q1[-1], "s", color="#d62728", markersize=5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("(a) IQ trajectories")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)


def _panel_b_snr(ax) -> None:
    t_int = np.linspace(30e-9, 450e-9, 30)
    snr = snr_vs_integration_time(REFERENCE_DEVICE, _DRIVE, t_int)
    ax.loglog(t_int * 1e9, snr, "-", color="black", lw=1.6, label="simulation")
    # Short-τ asymptote: SNR ∝ √τ (with prefactor fit to first few points).
    prefactor = snr[0] / np.sqrt(t_int[0])
    asym = prefactor * np.sqrt(t_int)
    ax.loglog(t_int * 1e9, asym, "--", color="#999999", lw=1.2, label=r"$\propto\sqrt{\tau}$")
    ax.set_xlabel("integration time  τ (ns)")
    ax.set_ylabel("SNR")
    ax.set_title("(b) SNR vs integration time")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, which="both")


def _panel_c_fidelity_vs_kappa_over_chi(ax) -> None:
    chi = dispersive_shift_from_simulation(REFERENCE_DEVICE)
    chi_abs = abs(chi)
    ratios = np.logspace(-1.0, 1.0, 11)  # κ/|χ| from 0.1 to 10
    fidelities = np.zeros_like(ratios)
    window = (50e-9, 500e-9)
    # Use a higher drive here so fidelity approaches 1 in the optimal regime;
    # mean photon ~5, still well below N_resonator=15 cutoff.
    drive_for_fidelity = DriveParams(
        amplitude=_TWO_PI * 10e6, duration=500e-9, detuning=0.0
    )
    for i, ratio in enumerate(ratios):
        new_kappa = ratio * chi_abs
        dev = replace(
            REFERENCE_DEVICE,
            resonator=ResonatorParams(
                omega_r=REFERENCE_DEVICE.resonator.omega_r, kappa=new_kappa
            ),
        )
        r0 = simulate_readout(dev, drive_for_fidelity, initial_qubit_state=0)
        r1 = simulate_readout(dev, drive_for_fidelity, initial_qubit_state=1)
        f = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
        fidelities[i] = f.F_assign
    ax.semilogx(ratios, fidelities, "-o", color="black", lw=1.4, markersize=4)
    ax.axvline(2.0, color="#ca0020", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"$\kappa / |\chi|$")
    ax.set_ylabel(r"$F_{\mathrm{assign}}$")
    ax.set_title("(c) Assignment fidelity vs κ/|χ|")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.2, which="both")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    _panel_a_iq_trajectories(axes[0])
    _panel_b_snr(axes[1])
    _panel_c_fidelity_vs_kappa_over_chi(axes[2])
    fig.suptitle(
        "Dispersive-readout simulation — reference device (Marxer arXiv:2508.16437)",
        fontsize=12,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
