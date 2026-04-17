"""Collapse operators and Hamiltonian builder for the readout simulation.

Collapse operators are constructed in the dressed transmon eigenbasis, not
the bare charge basis and not a 2-level approximation. This matters for
Module 2 leakage tracking. Pure dephasing in the multi-level transmon uses
per-level projectors L_j = sqrt(γ_φ) |j><j| (single-projector gauge of the
standard σ_z dephasing convention) so that every |j><k| coherence decays
at γ_φ regardless of Nq, avoiding the spurious cross-dephasing that the
(|j><j| − |0><0|) gauge produces in the N-level transmon. Adapted from
Blais et al. RMP 93, 025005 (2021) §III.E.

Frame convention (Task 15 refactor, approved by plan author):

build_hamiltonian returns the *dispersive-regime effective Hamiltonian* in
the fully-rotating frame: each transmon level rotates at its bare
frequency ω_j, the resonator rotates at ω_d = ω_r + detuning. A 2nd-order
Schrieffer-Wolff transformation eliminates the bare coupling g n̂(a + a†);
its residual effect is the per-level Lamb shift Δω_j and the dispersive
pull χ_j a†a. In this frame the stiff GHz oscillations of the rotating-
frame-at-ω_r approach are gone, and long-timescale (T1, Purcell) Lindblad
integrations become ~100× faster.

Since the transverse coupling is no longer explicit, spontaneous qubit
emission via the resonator ("Purcell decay") must be added as an explicit
collapse operator: γ_P_{j→j-1} = (g|⟨j-1|n̂|j⟩|/Δ_{j,j-1})² κ where
Δ_{j,j-1} = ω_j − ω_{j-1} − ω_r. V4b validates this formula against the
dressed-state resonator-component overlap of the full JC Hamiltonian.

This refactor is necessary for Modules 2–4: at the original frame's
~115 s/readout call, Module 3's 10,000 calls would require 13 days;
the dispersive frame brings it to ~1 s/call and ~3 hours total.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import qutip as qt
from scipy.special import erf

from .config import DeviceConfig, DriveParams
from .transmon import charge_operator_matrix_elements, diagonalize_transmon


def build_collapse_operators(
    device: DeviceConfig,
    transmon_basis_dim: int,
    resonator_dim: int,
) -> list[qt.Qobj]:
    """Lindblad collapse operators in the (transmon ⊗ resonator) Hilbert space.

    Channels:
      1. Resonator decay:   sqrt(κ (1 + n_th))  a
      2. Resonator heating: sqrt(κ  n_th)       a†     (only if n_th > 0)
      3. Qubit relaxation:  per-transition amplitudes in dressed transmon basis,
         scaled by |<j|n̂|k>|² relative to |<0|n̂|1>|² for |j+1⟩ → |j⟩ transitions.
      4. Qubit pure dephasing: sqrt(2 γ_φ) (|j><j| − |0><0|) for j = 1, ..., Nq−1.
      5. Qubit thermal heating: reverse of (3) scaled by n_th (only if n_th > 0).
    """
    tr = device.truncation
    Nq = transmon_basis_dim
    Nr = resonator_dim
    kappa = device.resonator.kappa
    gamma_1 = device.decoherence.gamma_1
    gamma_phi = device.decoherence.gamma_phi
    n_th = device.decoherence.n_th

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    c_ops: list[qt.Qobj] = []

    # 1. Resonator decay
    c_ops.append(np.sqrt(kappa * (1.0 + n_th)) * a)

    # 2. Resonator heating (only if bath is warm)
    if n_th > 0:
        c_ops.append(np.sqrt(kappa * n_th) * a.dag())

    # Build charge matrix elements for relaxation scaling in the dressed basis.
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    # Normalize so |<0|n̂|1>|² is the reference scale (rate γ_1 applies to |1>→|0>).
    ref_sq = abs(n_mat[0, 1]) ** 2

    # 3. Qubit relaxation: |j+1> -> |j> for j = 0, 1, ..., Nq-2
    for j in range(Nq - 1):
        scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
        rate = gamma_1 * scale * (1.0 + n_th)
        if rate > 0:
            op = qt.basis(Nq, j) * qt.basis(Nq, j + 1).dag()
            c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    # 4. Qubit pure dephasing: one projector per level, L_j = sqrt(γ_φ) |j><j|.
    #
    # Deviation from plan, which used sqrt(2γ_φ) × (|j><j| − |0><0|) for
    # j = 1..Nq-1. That form cross-couples through the shared |0><0| term:
    # every L_j contributes to dephasing of ρ_{01}, giving an effective rate
    # of (Nq+2) γ_φ on the |0>-|1> coherence rather than γ_φ. At Nq=5 the
    # V4a test fit ~7× the input γ_φ, matching the 4γ_φ + (Nq-2)γ_φ
    # arithmetic exactly. The per-level-projector form below gives γ_φ
    # decay on every coherence ρ_{jk}, j≠k, independent of Nq — the
    # expected behavior when "gamma_phi" is the measured dephasing rate.
    if gamma_phi > 0:
        for j in range(Nq):
            proj = qt.basis(Nq, j) * qt.basis(Nq, j).dag()
            c_ops.append(np.sqrt(gamma_phi) * qt.tensor(proj, qt.qeye(Nr)))

    # 5. Qubit thermal heating (reverse direction)
    if n_th > 0:
        for j in range(Nq - 1):
            scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
            rate = gamma_1 * scale * n_th
            if rate > 0:
                op = qt.basis(Nq, j + 1) * qt.basis(Nq, j).dag()
                c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    # 6. Purcell decay |j> -> |j-1> at rate (g|n_{j-1,j}| / Delta_{j,j-1})^2 kappa.
    #
    # In the lab/rotating-frame Hamiltonian the transverse coupling produces
    # Purcell decay implicitly: dressed |j,0> hybridizes with |j-1,1>, which
    # decays at κ. The dispersive-frame Hamiltonian returned by
    # build_hamiltonian has the transverse coupling transformed out, so that
    # implicit pathway is gone and Purcell must be added explicitly. Rate
    # derived from 2nd-order perturbation: the amplitude of |j-1, 1> in the
    # dressed |j, 0> state is (g |n_{j-1,j}|) / (ω_j − ω_{j-1} − ω_r), so
    # κ a acting on that admixture decays the qubit at rate |amplitude|² × κ.
    # Verified in test_V4b against the dressed-state overlap of the full
    # Jaynes-Cummings Hamiltonian.
    for j in range(1, Nq):
        delta_j = energies[j] - energies[j - 1] - device.resonator.omega_r
        n_elem = abs(n_mat[j - 1, j])
        # Include (1 + n_th) thermal factor for consistency with qubit relaxation.
        gamma_P = ((device.coupling.g * n_elem) / delta_j) ** 2 * kappa * (1.0 + n_th)
        if gamma_P > 0:
            op = qt.basis(Nq, j - 1) * qt.basis(Nq, j).dag()
            c_ops.append(np.sqrt(gamma_P) * qt.tensor(op, qt.qeye(Nr)))

    return c_ops


def build_hamiltonian(
    device: DeviceConfig,
    drive_params: DriveParams,
    frame: Literal["rotating", "dispersive"] = "rotating",
) -> tuple[qt.Qobj, list]:
    """Dispersive-regime effective Hamiltonian in the fully-rotating frame.

    Frame: each transmon level j rotates at its bare frequency ω_j, the
    resonator rotates at ω_d = ω_r + detuning. 2nd-order Schrieffer-Wolff
    eliminates the transverse coupling g n̂(a + a†); its residual is the
    per-level Lamb shift Δω_j and dispersive pull χ_j a†a:

        H_eff = Σ_j Δω_j |j><j|                       (Lamb shift, diagonal qubit)
              + Σ_j χ_j |j><j| a†a                    (dispersive shift)
              + (ω_r − ω_d) a†a                       (resonator in its rot frame)
              + ε(t) (a + a†)                         (drive, RWA-reduced)

    with
        Δω_j = Σ_{k≠j} |g ⟨j|n̂|k⟩|² / (ω_j − ω_k − ω_r)
        χ_j  = Σ_{k≠j} |g ⟨j|n̂|k⟩|² × [1/(ω_j − ω_k − ω_r) + 1/(ω_j − ω_k + ω_r)]
              (computed by dispersive_shift_full; includes Bloch-Siegert).

    Since the qubit diagonal is absorbed by the rotating-frame transform, the
    remaining diagonal is Lamb + χ·n_photon ~ tens of MHz rather than the
    previous ~GHz. This makes the ODE non-stiff and ~100× faster.

    'rotating' and 'dispersive' are aliases in this implementation (the only
    frame currently supported). Other frames raise NotImplementedError —
    callers that need the full bare JC Hamiltonian must add it explicitly.
    """
    if frame not in ("rotating", "dispersive"):
        raise NotImplementedError(
            f"frame '{frame}' not supported — 'rotating' and 'dispersive' "
            f"are aliases for the dispersive-regime effective Hamiltonian."
        )

    from .dispersive import dispersive_shift_full

    tr = device.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    g = device.coupling.g
    omega_r = device.resonator.omega_r
    omega_d = omega_r + drive_params.detuning

    # Per-level chi_j (non-RWA 2nd-order PT, includes Bloch-Siegert term)
    chi_per_level = dispersive_shift_full(energies, n_mat, g, omega_r)

    # Lamb shift: Δω_j = Σ |g n_jk|² / (ω_j − ω_k − ω_r)
    # Same sum as χ_j but keeping only the near-resonant (−ω_r) denominator.
    lamb_shifts = np.zeros(Nq, dtype=float)
    for j in range(Nq):
        total = 0.0
        for k in range(Nq):
            if k == j:
                continue
            coupling_sq = (g * abs(n_mat[j, k])) ** 2
            delta_jk = energies[j] - energies[k]
            total += coupling_sq / (delta_jk - omega_r)
        lamb_shifts[j] = total

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    n_ph = a.dag() * a

    # Resonator drift (vanishes on drive resonance)
    H_r = (omega_r - omega_d) * n_ph

    # Qubit Lamb shift: diagonal on qubit subspace
    H_q_lamb = qt.tensor(qt.Qobj(np.diag(lamb_shifts)), qt.qeye(Nr))

    # Dispersive shift: Σ_j χ_j |j><j| ⊗ a†a
    H_chi = 0 * qt.tensor(qt.qeye(Nq), qt.qeye(Nr))
    for j in range(Nq):
        proj_j = qt.tensor(qt.basis(Nq, j) * qt.basis(Nq, j).dag(), qt.qeye(Nr))
        H_chi = H_chi + chi_per_level[j] * proj_j * n_ph

    # Symmetrize against float roundoff (~1e-6 rad/s at scales of 1e10 rad/s)
    H0_raw = H_r + H_q_lamb + H_chi
    H0 = 0.5 * (H0_raw + H0_raw.dag())

    # Drive operator (same ε(t) convention as before; unchanged between frames)
    drive_op = a + a.dag()

    eps_0 = drive_params.amplitude
    t_end = drive_params.duration
    sigma = drive_params.edge_sigma
    t_rise = 3.0 * sigma
    t_fall = t_end - t_rise
    if t_fall <= t_rise + 2.0 * sigma:
        raise ValueError(
            f"Drive duration {t_end*1e9:.1f} ns too short for rise/fall "
            f"width {sigma*1e9:.1f} ns; need t_end > 6*sigma + 2*sigma."
        )

    def envelope(t: float, args: dict) -> float:
        return 0.5 * eps_0 * (erf((t - t_rise) / sigma) - erf((t - t_fall) / sigma))

    return H0, [drive_op, envelope]
