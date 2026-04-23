"""Closed-form analytic regime-map surface and boundary functions.

See MODULE_4_SPEC.md §3.2. The map is an analytic evaluation of the
dispersive-readout SNR formula (Bengtsson 2024 PRL §II, cross-checked
against Blais RMP 2021 §V.B §V.B), not a Lindblad grid — the 100× chi/kappa
range would otherwise extrapolate the 2nd-order SW dispersive PT well
outside its validity envelope (Q3 lock).
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm


ArrayLike = Union[float, np.ndarray]


def f_analytic_dispersive(
    chi_over_kappa: ArrayLike,
    gamma_1_tau: ArrayLike,
    n_phot: float,
) -> ArrayLike:
    """Closed-form F_assign per Bengtsson 2024 PRL §II.

    Parameters
    ----------
    chi_over_kappa : float or ndarray
        χ/κ at each evaluation point. Non-broadcastable axes must match.
    gamma_1_tau : float or ndarray
        γ_1 · τ_readout (dimensionless decoherence budget).
    n_phot : float
        Steady-state resonator photon number (scalar; held fixed across
        the grid and quoted on Figure 4 Panel (b)'s subtitle).

    Returns
    -------
    F : float or ndarray
        Assignment fidelity Φ(SNR_eff / 2).

    Notes
    -----
    Decoherence envelope is linearized: (1 − γ_1·τ/2)^(1/2). Within 1%
    of exp(−γ_1·τ/4) over the spec's y-axis range [1e-4, 1e-1]; a
    unit test asserts this.
    """
    chi_k = np.asarray(chi_over_kappa, dtype=float)
    g_t = np.asarray(gamma_1_tau, dtype=float)
    snr_steady = 4.0 * chi_k * np.sqrt(n_phot) / (1.0 + (2.0 * chi_k) ** 2)
    envelope = np.sqrt(np.clip(1.0 - g_t / 2.0, 0.0, 1.0))
    snr_eff = snr_steady * envelope
    return norm.cdf(snr_eff / 2.0)


from dataclasses import dataclass
from ..physics.config import REFERENCE_DEVICE


@dataclass(frozen=True)
class DevicePoint:
    """A published device's position on the (χ/κ, γ_1·τ_readout) regime map."""
    label: str
    citation: str
    chi_over_kappa: float
    gamma_1_tau: float
    reported_F_assign: float | None
    marker: str                    # matplotlib marker code
    marker_color: str              # "warm_orange" or "red" per Q6 encoding
    estimated: bool = False        # True → grey-hatched marker fill (Q5)
    estimated_fields: tuple[str, ...] = ()


PUBLISHED_DEVICE_POINTS: list[DevicePoint] = [
    DevicePoint(
        label="Marxer Q1 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437 p.15 device table + §V.3 Table 1",
        chi_over_kappa=2.5 / 6.1,              # 0.41
        gamma_1_tau=280e-9 / 86e-6,            # 3.26e-3
        reported_F_assign=0.99943,
        marker="*",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Marxer Q2 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437 p.15",
        chi_over_kappa=2.6 / 3.4,              # 0.76
        gamma_1_tau=280e-9 / 102e-6,           # 2.75e-3
        reported_F_assign=0.99946,
        marker="D",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Bengtsson (Google, 2024)",
        citation=(
            "Bengtsson et al., PRL 132 100603 (2024) / arXiv:2308.02079 Eq. 3; "
            "κ ∈ [4,8] MHz from Sank arXiv:2402.00413 §IV; "
            "T_1 ≈ 20 µs from Arute 2019 Sycamore-typical"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 20e-6,            # 2.5e-2
        reported_F_assign=0.985,
        marker="o",
        marker_color="red",
        estimated=True,
        estimated_fields=("T_1",),
    ),
    DevicePoint(
        label="Garnet (IQM, 2024)",
        citation=(
            "Abdurakhimov et al., arXiv:2408.12433 p.9 (F_assign) + p.13 (T_1); "
            "χ/κ and τ_readout are IQM design-family estimates"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 40e-6,            # 1.25e-2
        reported_F_assign=0.97,
        marker="s",
        marker_color="red",
        estimated=True,
        estimated_fields=("chi_over_kappa", "tau_readout"),
    ),
]


# ---------- Analytic boundaries (MODULE_4_SPEC.md §3.2 post-Nit-2) ----------


def _reference_purcell_rate_per_kappa() -> float:
    """(g_REF / Δ_REF)² — the dimensionless factor in γ_Purcell = κ · (g/Δ)²."""
    # Δ = |ω_q − ω_r|; from REFERENCE_DEVICE's transmon parameters (Koch limit)
    # ω_q ≈ sqrt(8 E_J E_C) − E_C. Compute once via Module 1's diagonalize_transmon.
    from ..physics.transmon import diagonalize_transmon
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    omega_q = float(energies[1] - energies[0])
    delta = abs(omega_q - REFERENCE_DEVICE.resonator.omega_r)
    return (REFERENCE_DEVICE.coupling.g / delta) ** 2


def _reference_chi_magnitude() -> float:
    """|χ_01| at REFERENCE, used to relate x = χ/κ → κ along the boundary."""
    from ..physics.transmon import diagonalize_transmon, charge_operator_matrix_elements
    from ..physics.dispersive import dispersive_shift_full
    energies, eigenstates = diagonalize_transmon(
        REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, REFERENCE_DEVICE.truncation)
    chi = dispersive_shift_full(
        energies, n_mat, REFERENCE_DEVICE.coupling.g, REFERENCE_DEVICE.resonator.omega_r,
    )
    return abs(chi[0] - chi[1])


def purcell_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Purcell limit locus: γ_Purcell · τ_readout = 0.1, (g, Δ) at REFERENCE.

    With χ held at REFERENCE's dispersive-computed value, κ(x) = χ_REF / x,
    γ_Purcell(x) = κ(x) · (g_REF/Δ_REF)², and τ_readout(x) = 0.1/γ_Purcell(x).
    Plotted y = γ_1_REF · τ_readout(x).
    """
    chi_ref = _reference_chi_magnitude()
    g_over_delta_sq = _reference_purcell_rate_per_kappa()
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1

    kappa_x = chi_ref / np.asarray(chi_over_kappa, dtype=float)
    gamma_P_x = kappa_x * g_over_delta_sq
    tau_readout_x = 0.1 / gamma_P_x
    return gamma_1_ref * tau_readout_x


def dispersive_breakdown_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Dispersive breakdown locus: χ · τ_readout = 2π.

    With κ held at REFERENCE, χ(x) = x · κ_REF, τ_readout(x) = 2π/χ(x).
    Plotted y = γ_1_REF · τ_readout(x).
    """
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    chi_x = np.asarray(chi_over_kappa, dtype=float) * kappa_ref
    tau_readout_x = (2.0 * np.pi) / chi_x
    return gamma_1_ref * tau_readout_x


def resonator_too_slow_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Resonator-too-slow locus: κ · τ_readout = 1.

    κ held at REFERENCE; horizontal line in (x, y) at y = γ_1_REF / κ_REF.
    """
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    y_const = gamma_1_ref / kappa_ref
    return np.full_like(np.asarray(chi_over_kappa, dtype=float), y_const)
