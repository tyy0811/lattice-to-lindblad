"""Closed-form analytic regime-map surface and boundary functions.

See MODULE_4_SPEC.md §3.2 (post-Day-11 amendment item 15) and the per-level
derivation at docs/module4_diagnostics/per_level_analytic_derivation.md.

The closed-form `f_analytic_dispersive` evaluates the integrated readout SNR
for a transmon with per-level dispersive shifts (χ_0, χ_1), not the textbook
two-level antisymmetric ±χ/2 approximation. The chart parametrization
(χ_diff/κ, γ_1·τ) is REFERENCE-family-anchored: at each chart point, REFERENCE's
per-level χ values and drive amplitude are held fixed, and κ is rescaled to
hit the target χ_diff/κ ratio. Validated against the Lindblad simulator at 3
operating points (Marxer Q1, mid-range, weak-decoherence) to <5%.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np
from scipy.stats import norm

from ..physics.config import REFERENCE_DEVICE


ArrayLike = Union[float, np.ndarray]


# ------------------------------------------------------------------------
# REFERENCE-anchored physical constants (cached; read once per process)
# ------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _reference_per_level_chi() -> tuple[float, float]:
    """Per-level dispersive shifts (χ_0, χ_1) at REFERENCE, in rad/s.

    Returns (chi_g, chi_e) — both positive in general for a transmon (Koch
    2007 §V), with chi_g > chi_e for REFERENCE's E_J/E_C ratio. The textbook
    ±χ/2 antisymmetric approximation (chi_e = -chi_g) does NOT hold; REFERENCE
    has chi_g/chi_e ≈ 1.47 (both same sign).
    """
    from ..physics.transmon import diagonalize_transmon, charge_operator_matrix_elements
    from ..physics.dispersive import dispersive_shift_full
    energies, eigenstates = diagonalize_transmon(
        REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, REFERENCE_DEVICE.truncation)
    chi = dispersive_shift_full(
        energies, n_mat, REFERENCE_DEVICE.coupling.g, REFERENCE_DEVICE.resonator.omega_r,
    )
    return float(chi[0]), float(chi[1])


def _reference_chi_magnitude() -> float:
    """|χ_diff| = |χ_0 − χ_1| at REFERENCE, in rad/s.

    This is the "χ" that the regime map's x-axis labels (the g-e dispersive
    splitting). NOT the per-level shifts (which are (chi_g, chi_e) above).
    """
    chi_0, chi_1 = _reference_per_level_chi()
    return abs(chi_0 - chi_1)


@lru_cache(maxsize=1)
def _reference_drive_and_window() -> tuple[float, float]:
    """REFERENCE's calibrated (drive amplitude ε, integration window length T) in (rad/s, s)."""
    from ..analysis.operating_point import get_reference_operating_point
    op = get_reference_operating_point(n_shots=10_000)
    epsilon = float(op.drive.amplitude)
    T_window = float(op.integration_window[1] - op.integration_window[0])
    return epsilon, T_window


# ------------------------------------------------------------------------
# Per-level closed-form F_assign (workhorse)
# ------------------------------------------------------------------------


def f_analytic_dispersive_per_level(
    chi_0: ArrayLike,
    chi_1: ArrayLike,
    kappa: ArrayLike,
    epsilon: float,
    T_window: float,
    gamma_1_tau: ArrayLike,
) -> ArrayLike:
    """Closed-form F_assign for a transmon with per-level dispersive shifts.

    Implements the per-level integrated readout SNR derived in
    docs/module4_diagnostics/per_level_analytic_derivation.md §6:

        |Δα|_ss = |ε · (χ_0 − χ_1)| / [√((κ/2)² + χ_0²) · √((κ/2)² + χ_1²)]
        SNR_M1 = 2·√(κ·T_window) · |Δα|_ss
        F      = Φ(SNR_M1·√(1 − γ_1·τ/2) / 2)

    Module 1 SNR convention (η=1 implicit, T_window = integration window length).

    All non-broadcastable axes between chi_0, chi_1, kappa, gamma_1_tau must
    match. epsilon and T_window are scalars (held at REFERENCE for the chart).
    """
    chi_0_arr = np.asarray(chi_0, dtype=float)
    chi_1_arr = np.asarray(chi_1, dtype=float)
    kappa_arr = np.asarray(kappa, dtype=float)
    g_t_arr = np.asarray(gamma_1_tau, dtype=float)

    half_kappa = kappa_arr / 2.0
    denom_g = np.sqrt(half_kappa ** 2 + chi_0_arr ** 2)
    denom_e = np.sqrt(half_kappa ** 2 + chi_1_arr ** 2)
    chi_diff = np.abs(chi_0_arr - chi_1_arr)
    delta_alpha_ss = epsilon * chi_diff / (denom_g * denom_e)

    snr_M1_steady = 2.0 * np.sqrt(kappa_arr * T_window) * delta_alpha_ss
    envelope = np.sqrt(np.clip(1.0 - g_t_arr / 2.0, 0.0, 1.0))
    snr_eff = snr_M1_steady * envelope
    return norm.cdf(snr_eff / 2.0)


# ------------------------------------------------------------------------
# Chart-coordinate wrapper (REFERENCE-family-anchored)
# ------------------------------------------------------------------------


def f_analytic_dispersive(
    chi_over_kappa: ArrayLike,
    gamma_1_tau: ArrayLike,
) -> ArrayLike:
    """Closed-form F_assign at chart coordinates (χ_diff/κ, γ_1·τ).

    REFERENCE-family anchored: at each chart point, REFERENCE's per-level χ
    values and drive amplitude (ε, T_window) are held fixed, and κ is
    rescaled to hit the target χ_diff/κ ratio. The chart is therefore
    specific to the REFERENCE transmon family (Koch E_J/E_C, g/Δ); other
    families with different per-level structure will shift the surface.

    See docs/module4_diagnostics/per_level_analytic_derivation.md §7 for the
    chart parametrization and the rationale for choosing REFERENCE as anchor.

    Parameters
    ----------
    chi_over_kappa : float or ndarray
        χ_diff/κ ≡ |χ_0 − χ_1|/κ at each evaluation point.
    gamma_1_tau : float or ndarray
        γ_1 · τ_readout (dimensionless decoherence budget).

    Returns
    -------
    F : float or ndarray
        Assignment fidelity Φ(SNR_eff / 2).
    """
    chi_k = np.asarray(chi_over_kappa, dtype=float)
    chi_0_ref, chi_1_ref = _reference_per_level_chi()
    chi_diff_ref = abs(chi_0_ref - chi_1_ref)
    epsilon, T_window = _reference_drive_and_window()

    # κ = χ_diff_REF / x at each chart point (holding REFERENCE per-level shifts).
    kappa = chi_diff_ref / chi_k
    # Per-level shifts unchanged from REFERENCE (the structure that determines
    # the per-state Lorentzian-response detuning).
    chi_0 = np.broadcast_to(chi_0_ref, kappa.shape).astype(float)
    chi_1 = np.broadcast_to(chi_1_ref, kappa.shape).astype(float)

    return f_analytic_dispersive_per_level(
        chi_0=chi_0, chi_1=chi_1, kappa=kappa,
        epsilon=epsilon, T_window=T_window,
        gamma_1_tau=gamma_1_tau,
    )


# ------------------------------------------------------------------------
# Published device markers (Q5 lock; unchanged by item-15 amendment)
# ------------------------------------------------------------------------


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


# ------------------------------------------------------------------------
# Analytic boundaries (MODULE_4_SPEC.md §3.2 post-Nit-2; unchanged)
# ------------------------------------------------------------------------


def _reference_purcell_rate_per_kappa() -> float:
    """(g_REF / Δ_REF)² — the dimensionless factor in γ_Purcell = κ · (g/Δ)²."""
    from ..physics.transmon import diagonalize_transmon
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    omega_q = float(energies[1] - energies[0])
    delta = abs(omega_q - REFERENCE_DEVICE.resonator.omega_r)
    return (REFERENCE_DEVICE.coupling.g / delta) ** 2


def purcell_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Purcell limit locus: γ_Purcell · τ_readout = 0.1, (g, Δ) at REFERENCE."""
    chi_ref = _reference_chi_magnitude()
    g_over_delta_sq = _reference_purcell_rate_per_kappa()
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    kappa_x = chi_ref / np.asarray(chi_over_kappa, dtype=float)
    gamma_P_x = kappa_x * g_over_delta_sq
    tau_readout_x = 0.1 / gamma_P_x
    return gamma_1_ref * tau_readout_x


def dispersive_breakdown_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Dispersive breakdown locus: χ · τ_readout = 2π."""
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    chi_x = np.asarray(chi_over_kappa, dtype=float) * kappa_ref
    tau_readout_x = (2.0 * np.pi) / chi_x
    return gamma_1_ref * tau_readout_x


def resonator_too_slow_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Resonator-too-slow locus: κ · τ_readout = 1."""
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    y_const = gamma_1_ref / kappa_ref
    return np.full_like(np.asarray(chi_over_kappa, dtype=float), y_const)


# ------------------------------------------------------------------------
# Chart compute + Lindblad validation (per-level)
# ------------------------------------------------------------------------


def compute_analytic_regime_map(
    chi_over_kappa_range: tuple[float, float] = (0.1, 10.0),
    gamma_1_tau_range: tuple[float, float] = (1e-4, 1e-1),
    n_chi: int = 20,
    n_gamma: int = 20,
) -> dict:
    """Return dict with 'chi_over_kappa_axis', 'gamma_1_tau_axis', 'F_grid',
    'epsilon', 'T_window', 'chi_per_level_anchor'.

    The chart is REFERENCE-family-anchored per item-15 amendment: REFERENCE's
    per-level (χ_0, χ_1) and (ε, T_window) are held fixed across the grid,
    and κ varies to hit each chart point's χ_diff/κ. Sub-second; pure analytic
    (no sim calls beyond the cached REFERENCE-anchor lookups).
    """
    x_axis = np.logspace(
        np.log10(chi_over_kappa_range[0]),
        np.log10(chi_over_kappa_range[1]),
        n_chi,
    )
    y_axis = np.logspace(
        np.log10(gamma_1_tau_range[0]),
        np.log10(gamma_1_tau_range[1]),
        n_gamma,
    )
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    F = f_analytic_dispersive(X, Y)

    chi_0_ref, chi_1_ref = _reference_per_level_chi()
    epsilon, T_window = _reference_drive_and_window()
    return {
        "chi_over_kappa_axis": x_axis,
        "gamma_1_tau_axis": y_axis,
        "F_grid": F,
        "epsilon": epsilon,
        "T_window": T_window,
        "chi_per_level_anchor": (chi_0_ref, chi_1_ref),
    }


def validate_analytic_vs_lindblad(
    points: list[tuple[float, float]] | None = None,
) -> dict:
    """Per-level item-15 amendment: evaluate F_sim at specified (χ/κ, γ_1·τ)
    points and compare to the per-level analytic F.

    F_sim is computed by rescaling REFERENCE's resonator κ to hit target χ/κ
    (holding REFERENCE per-level χ_j unchanged) and rescaling γ_1 to hit target
    γ_1·τ (holding REFERENCE drive duration). F_analytic is computed on the
    same rescaled device via f_analytic_dispersive_per_level.

    Default points are O3a (Marxer Q1), O3b (mid-range), O3c (weak-decoherence).
    """
    from dataclasses import replace

    from ..analysis.operating_point import get_reference_operating_point
    from ..physics.readout_model import simulate_readout, compute_assignment_fidelity

    if points is None:
        q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
        points = [
            (q1.chi_over_kappa, q1.gamma_1_tau),  # O3a: Marxer Q1
            (1.0, 0.01),                          # O3b: mid-range
            (0.5, 1e-3),                          # O3c: weak-decoherence at dispersive optimum
        ]

    op = get_reference_operating_point(n_shots=10_000)
    chi_0_ref, chi_1_ref = _reference_per_level_chi()
    chi_diff_ref = abs(chi_0_ref - chi_1_ref)
    epsilon, T_window = _reference_drive_and_window()
    tau = op.drive.duration

    per_point = []
    for (target_chi_over_k, target_gamma_tau) in points:
        target_kappa = chi_diff_ref / target_chi_over_k
        target_gamma_1 = target_gamma_tau / tau
        new_res = replace(op.device.resonator, kappa=target_kappa)
        new_dec = replace(op.device.decoherence, gamma_1=target_gamma_1)
        new_device = replace(op.device, resonator=new_res, decoherence=new_dec)

        r0 = simulate_readout(new_device, op.drive, initial_qubit_state=0)
        r1 = simulate_readout(new_device, op.drive, initial_qubit_state=1)
        F_sim = compute_assignment_fidelity(
            r0, r1, op.integration_window, n_shots=op.n_shots, noise_model="analytic",
        ).F_assign

        # Per-level analytic F at the same rescaled-κ device. REFERENCE per-level
        # χ_j unchanged because we did not modify the transmon — only the resonator.
        F_analytic = float(
            f_analytic_dispersive_per_level(
                chi_0=chi_0_ref, chi_1=chi_1_ref, kappa=target_kappa,
                epsilon=epsilon, T_window=T_window,
                gamma_1_tau=target_gamma_tau,
            )
        )
        per_point.append({
            "chi_over_kappa": float(target_chi_over_k),
            "gamma_1_tau": float(target_gamma_tau),
            "F_analytic": float(F_analytic),
            "F_lindblad": float(F_sim),
            "deviation_fractional": float(abs(F_sim - F_analytic) / F_sim),
        })

    max_dev = max(p["deviation_fractional"] for p in per_point)
    return {
        "per_point": per_point,
        "max_deviation_fractional": max_dev,
        "epsilon": epsilon,
        "T_window": T_window,
        "chi_per_level_anchor": (chi_0_ref, chi_1_ref),
    }
