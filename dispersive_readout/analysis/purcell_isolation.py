"""Analytic Purcell rate for cross-validation of the simulated Purcell channel.

See MODULE_2_SPEC.md §5.2. Post-blocker-6, only analytic_purcell_rate is
exported; effective_T1_from_device and decomposed_T1 from the original spec
are YAGNI.

Reference: Blais et al., Rev. Mod. Phys. 93, 025005 (2021), §III.E.
"""
from __future__ import annotations

from ..physics.config import DeviceConfig
from ..physics.transmon import charge_operator_matrix_elements, diagonalize_transmon


def analytic_purcell_rate(device: DeviceConfig) -> float:
    """γ_Purcell for the |1⟩→|0⟩ transition from (g |⟨0|n̂|1⟩| / Δ_{10})² κ.

    Uses the dressed transmon basis (N-level), not the 2-level estimate.
    Δ_{10} = ω_1 − ω_0 − ω_r is the detuning of the |1>→|0> transition
    from the resonator.

    Returns
    -------
    gamma_P : float
        Purcell rate in rad/s (equivalently, 1/s for rates).
    """
    tr = device.truncation
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    g = device.coupling.g
    kappa = device.resonator.kappa
    omega_r = device.resonator.omega_r

    delta_10 = energies[1] - energies[0] - omega_r
    n_elem = abs(n_mat[0, 1])
    gamma_P = ((g * n_elem) / delta_10) ** 2 * kappa
    return float(gamma_P)
