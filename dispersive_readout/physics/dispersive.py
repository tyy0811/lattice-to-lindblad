"""Analytic and numerical dispersive-shift formulas.

χ convention: χ ≡ (χ_1 − χ_0)/2, the half-splitting observable in readout.
dispersive_shift_full returns per-level χ_j; the caller computes the
half-splitting from those as needed.
"""
from __future__ import annotations

import numpy as np

from .config import DeviceConfig


def dispersive_shift_two_level(g: float, Delta: float) -> float:
    """Two-level-limit dispersive shift: χ = g² / Δ.

    Inputs are in rad/s; output in rad/s. For Δ < 0 (qubit below resonator,
    the reference device's regime) this is negative.
    """
    return (g ** 2) / Delta


def dispersive_shift_full(
    energies: np.ndarray,
    n_matrix: np.ndarray,
    g: float,
    omega_r: float,
) -> np.ndarray:
    """Multi-level per-level dispersive shifts χ_j (non-RWA, 2nd order).

    Derived from 2nd-order perturbation theory on the full transverse
    coupling V = g n̂ (a + a†). For state |q, N⟩ the energy correction is

        ΔE(q, N) = Σ_{q'≠q} |g n_{q'q}|² × [
            N     / (ω_q − ω_{q'} + ω_r)           (from |q', N−1⟩)
          + (N+1) / (ω_q − ω_{q'} − ω_r)           (from |q', N+1⟩)
        ]

    The photon-linear piece — the cavity shift — is the coefficient of N:

        χ_q = Σ_{k≠q} |g <q|n̂|k>|² × [
            1/(ω_q − ω_k + ω_r) + 1/(ω_q − ω_k − ω_r)
        ].

    The observable readout shift is χ ≡ (χ_1 − χ_0)/2. The PLUS between the
    two denominator terms keeps both the near-resonant (JC) and counter-
    rotating (Bloch-Siegert) contributions — correct for the non-RWA
    Hamiltonian used throughout this package.

    Plan note: the original plan draft had a MINUS between the two terms,
    which would give χ = 0 identically in the two-level limit (the JC and
    Bloch-Siegert contributions cancel exactly). Fixed during Task 9 after
    V2 failure; verified at weak coupling (g/2π = 12 MHz) where the formula
    agrees with the exact dressed-JC diagonalization to 1.3e-4 relative,
    and at REFERENCE_DEVICE coupling where the ~1.3% residual scales as
    (g/Δ)² and is correctly identified as 3rd-order perturbative.
    """
    N = len(energies)
    chi = np.zeros(N, dtype=float)
    for j in range(N):
        total = 0.0
        for k in range(N):
            if k == j:
                continue
            coupling_sq = (g * abs(n_matrix[j, k])) ** 2
            delta_jk = energies[j] - energies[k]
            denom_minus = delta_jk - omega_r
            denom_plus = delta_jk + omega_r
            if denom_minus == 0.0 or denom_plus == 0.0:
                raise ValueError(
                    f"Degeneracy in denominators at j={j}, k={k}: "
                    f"delta={delta_jk}, omega_r={omega_r}"
                )
            total += coupling_sq * (1.0 / denom_minus + 1.0 / denom_plus)
        chi[j] = total
    return chi


def dispersive_shift_from_simulation(device: DeviceConfig) -> float:
    """Extract χ ≡ (χ₁ − χ₀)/2 from the dressed Jaynes-Cummings spectrum.

    Builds the full zero-drive Hamiltonian in the
    (transmon ⊗ resonator) basis, diagonalizes it, identifies the dressed
    states adiabatically connected to the bare product states
    |q,n⟩ for q ∈ {0,1} and n ∈ {0,1} (by overlap), and returns
        ((E(1,1) − E(1,0)) − (E(0,1) − E(0,0))) / 2.

    Requires N_transmon >= 2 and N_resonator >= 2, since the definition uses
    bare kets |1⟩_q and |1⟩_r. Raises ValueError otherwise — these are
    narrower than the global TruncationParams contract, which allows =1 on
    both dimensions for callers that only need the subspace they actually
    have.
    """
    import qutip as qt

    from .transmon import charge_operator_matrix_elements, diagonalize_transmon

    tr = device.truncation
    if tr.N_transmon < 2:
        raise ValueError(
            f"dispersive_shift_from_simulation requires N_transmon >= 2 "
            f"(got {tr.N_transmon}); the chi definition uses bare qubit |1⟩."
        )
    if tr.N_resonator < 2:
        raise ValueError(
            f"dispersive_shift_from_simulation requires N_resonator >= 2 "
            f"(got {tr.N_resonator}); the chi definition uses bare photon |1⟩."
        )
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    Nq = tr.N_transmon
    Nr = tr.N_resonator

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    H_q = qt.tensor(qt.Qobj(np.diag(energies)), qt.qeye(Nr))
    H_r = device.resonator.omega_r * a.dag() * a
    n_op_q = qt.tensor(qt.Qobj(n_mat), qt.qeye(Nr))
    H_c = device.coupling.g * n_op_q * (a + a.dag())
    H = H_q + H_r + H_c

    eigvals, eigvecs = H.eigenstates()

    # Identify dressed states by max-overlap with bare product kets.
    # .overlap returns a complex scalar in QuTiP 4 and 5; magnitude-squared is
    # the fidelity.
    bare_energies = {}
    for q in (0, 1):
        for n in (0, 1):
            bare_ket = qt.tensor(qt.basis(Nq, q), qt.basis(Nr, n))
            overlaps = np.array([abs(bare_ket.overlap(v)) ** 2 for v in eigvecs])
            idx = int(np.argmax(overlaps))
            # H is Hermitian, so eigvals are real modulo float roundoff; take .real
            # to silence the cast-to-real warning rather than mask a bug.
            bare_energies[(q, n)] = float(np.real(eigvals[idx]))

    return (
        (bare_energies[(1, 1)] - bare_energies[(1, 0)])
        - (bare_energies[(0, 1)] - bare_energies[(0, 0)])
    ) / 2.0
