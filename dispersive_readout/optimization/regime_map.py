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


def _infer_n_phot_at_reference() -> float:
    """Infer steady-state photon number at REFERENCE operating point.

    Reuses Module 2's calibration path: get_reference_operating_point returns
    the calibrated drive; average photon number over the last 20% of the
    integration window is the steady-state estimate.
    """
    from ..analysis.operating_point import get_reference_operating_point
    from ..physics.readout_model import simulate_readout

    op = get_reference_operating_point(n_shots=10_000)
    r0 = simulate_readout(op.device, op.drive, initial_qubit_state=0)
    # Average photon number over last 20% of the integration window
    t = r0.t
    t0, t1 = op.integration_window
    window_mask = (t >= t0 + 0.8 * (t1 - t0)) & (t <= t1)
    return float(np.mean(r0.photon_number[window_mask]))


def compute_analytic_regime_map(
    chi_over_kappa_range: tuple[float, float] = (0.1, 10.0),
    gamma_1_tau_range: tuple[float, float] = (1e-4, 1e-1),
    n_chi: int = 20,
    n_gamma: int = 20,
    n_phot: float | None = None,
) -> dict:
    """Return dict with 'chi_over_kappa_axis', 'gamma_1_tau_axis',
    'F_grid', 'n_phot_used'. Sub-second; pure analytic (no sim calls)."""
    if n_phot is None:
        n_phot = _infer_n_phot_at_reference()

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
    # 2D grid via broadcasting: x on axis=0, y on axis=1
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    F = f_analytic_dispersive(X, Y, n_phot=n_phot)

    return {
        "chi_over_kappa_axis": x_axis,
        "gamma_1_tau_axis": y_axis,
        "F_grid": F,
        "n_phot_used": float(n_phot),
    }


def validate_analytic_vs_lindblad(
    points: list[tuple[float, float]] | None = None,
) -> dict:
    """Q3 Refinement 2: evaluate F_sim at specified (χ/κ, γ_1·τ) points and
    compare to F_analytic.

    F_sim is computed at REFERENCE-with-resonator-κ-rescaled-to-hit-target-χ/κ
    (holding χ at REFERENCE's dispersive-computed value) and decoherence-γ_1-
    rescaled-to-hit-target-γ_1·τ (holding τ at REFERENCE's drive.duration).
    Caption cites max deviation.
    """
    from dataclasses import replace

    from ..analysis.operating_point import get_reference_operating_point
    from ..physics.readout_model import simulate_readout, compute_assignment_fidelity

    if points is None:
        # Defaults: Marxer Q1 + mid-range (χ/κ=1, γ_1·τ=0.01)
        q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
        points = [(q1.chi_over_kappa, q1.gamma_1_tau), (1.0, 0.01)]

    op = get_reference_operating_point(n_shots=10_000)
    n_phot = _infer_n_phot_at_reference()
    chi_ref = _reference_chi_magnitude()
    tau = op.drive.duration

    per_point = []
    for (target_chi_over_k, target_gamma_tau) in points:
        # Construct device with rescaled κ and γ_1 to hit target coordinates
        target_kappa = chi_ref / target_chi_over_k
        target_gamma_1 = target_gamma_tau / tau
        new_res = replace(op.device.resonator, kappa=target_kappa)
        new_dec = replace(op.device.decoherence, gamma_1=target_gamma_1)
        new_device = replace(op.device, resonator=new_res, decoherence=new_dec)

        r0 = simulate_readout(new_device, op.drive, initial_qubit_state=0)
        r1 = simulate_readout(new_device, op.drive, initial_qubit_state=1)
        F_sim = compute_assignment_fidelity(
            r0, r1, op.integration_window, n_shots=op.n_shots, noise_model="analytic",
        ).F_assign

        F_analytic = float(
            f_analytic_dispersive(
                np.asarray(target_chi_over_k), np.asarray(target_gamma_tau), n_phot=n_phot,
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
    return {"per_point": per_point, "max_deviation_fractional": max_dev, "n_phot_used": n_phot}
