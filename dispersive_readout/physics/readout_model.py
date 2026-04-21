"""Pulsed readout simulation, IQ trajectories, assignment fidelity.

simulate_readout integrates the Lindblad master equation with QuTiP mesolve.
The observable is <a>(t) — the homodyne signal. Runtime-checks that the
mean photon number during readout stays below an N_resonator-dependent
ceiling and prints a warning if not.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt

from .config import DeviceConfig, DriveParams
from .lindblad import build_collapse_operators, build_hamiltonian


@dataclass(frozen=True)
class ReadoutResult:
    """Single readout trajectory.

    All arrays share first-axis length T = len(t).
    a_expectation is complex (homodyne-observable resonator coherent amplitude).
    photon_number is real (for truncation monitoring).
    qubit_populations is (T, N_transmon).
    """
    t: np.ndarray
    a_expectation: np.ndarray
    photon_number: np.ndarray
    qubit_populations: np.ndarray
    drive_envelope: np.ndarray
    device: DeviceConfig
    drive_params: DriveParams
    initial_qubit_state: int

    def integrated_iq(self, window: tuple[float, float]) -> complex:
        """Return the integrated complex IQ amplitude over [window[0], window[1]]."""
        t0, t1 = window
        mask = (self.t >= t0) & (self.t <= t1)
        if mask.sum() < 2:
            raise ValueError(f"Window {window} contains fewer than 2 samples")
        return np.trapezoid(self.a_expectation[mask], self.t[mask])


@dataclass(frozen=True)
class AssignmentFidelityResult:
    F_assign: float
    F_assign_uncertainty: float
    centroid_0: complex
    centroid_1: complex
    snr: float
    separation_distance: float
    integration_window: tuple[float, float]
    n_shots: int
    noise_model: str


_MAX_PHOTON_RATIO = 0.33  # warn if mean photon > 1/3 of N_resonator


def simulate_readout(
    device: DeviceConfig,
    drive_params: DriveParams,
    initial_qubit_state: int,
    initial_resonator_state: str = "vacuum",
    t_list: np.ndarray | None = None,
    solver_options: dict | None = None,
    chi_scale: float = 1.0,
) -> ReadoutResult:
    """Integrate the Lindblad ME for the transmon-resonator system under a pulsed drive.

    initial_qubit_state = 0 or 1 selects the dressed transmon eigenket at t=0.
    initial_resonator_state = 'vacuum' is the only supported option in Module 1.
    """
    if initial_qubit_state not in (0, 1):
        raise ValueError("initial_qubit_state must be 0 or 1.")
    if initial_resonator_state != "vacuum":
        raise NotImplementedError(f"only 'vacuum' supported, got '{initial_resonator_state}'.")

    tr = device.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    H0, drive_spec = build_hamiltonian(device, drive_params, frame="rotating", chi_scale=chi_scale)
    c_ops = build_collapse_operators(device, Nq, Nr)

    psi0 = qt.tensor(qt.basis(Nq, initial_qubit_state), qt.basis(Nr, 0))

    if t_list is None:
        t_list = np.linspace(0.0, drive_params.duration, 501)

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    n_photon = a.dag() * a
    # Populations: P(|j⟩) = Tr_r(|j><j| ⊗ I · ρ)
    e_ops_pop = [
        qt.tensor(qt.basis(Nq, j) * qt.basis(Nq, j).dag(), qt.qeye(Nr))
        for j in range(Nq)
    ]

    # QuTiP 5.x: options is a plain dict. The plan's pattern
    # `opts = qt.Options(); opts.nsteps = 10000` would AttributeError here
    # because qt.Options() now returns an empty dict.
    opts = {"nsteps": 10000, "atol": 1e-10, "rtol": 1e-8}
    if solver_options:
        opts.update(solver_options)

    result = qt.mesolve(
        H=[H0, drive_spec],
        rho0=psi0,
        tlist=t_list,
        c_ops=c_ops,
        e_ops=[a, n_photon, *e_ops_pop],
        options=opts,
    )

    a_exp = np.asarray(result.expect[0], dtype=complex)
    n_exp = np.asarray(result.expect[1], dtype=float)
    pops = np.stack([np.asarray(result.expect[2 + j], dtype=float) for j in range(Nq)], axis=1)

    # Runtime check — flag if we're close to Fock truncation.
    max_photon = float(n_exp.max())
    if max_photon > _MAX_PHOTON_RATIO * Nr:
        warnings.warn(
            f"Mean photon number peaked at {max_photon:.2f} with N_resonator={Nr}. "
            f"Truncation may be insufficient — consider N_resonator={Nr + 10}.",
            RuntimeWarning,
        )

    # Record drive envelope (convenience — same callable used in drive_spec)
    _, envelope_fn = drive_spec
    drive_env = np.array([envelope_fn(ti, {}) for ti in t_list], dtype=float)

    return ReadoutResult(
        t=np.asarray(t_list, dtype=float),
        a_expectation=a_exp,
        photon_number=n_exp,
        qubit_populations=pops,
        drive_envelope=drive_env,
        device=device,
        drive_params=drive_params,
        initial_qubit_state=initial_qubit_state,
    )


def compute_assignment_fidelity(
    result_ground: ReadoutResult,
    result_excited: ReadoutResult,
    integration_window: tuple[float, float],
    n_shots: int = 10000,
    noise_model: Literal["ideal", "gaussian"] = "gaussian",
    rng: np.random.Generator | None = None,
) -> AssignmentFidelityResult:
    """Single-shot assignment fidelity from two simulated trajectories.

    Integrates ⟨a⟩(t) over the window for each of |0> and |1> to get
    deterministic centroids; adds per-shot circular Gaussian noise in IQ space
    (when noise_model='gaussian'); classifies shots with the perpendicular-
    bisector discriminator; returns F = 1 - (P(1|0) + P(0|1)) / 2.

    Parameters
    ----------
    rng : np.random.Generator | None, optional
        RNG for shot-noise draws. If None (default), an ephemeral RNG is
        created per call, giving independent draws across successive calls.
        Pass a seeded RNG for deterministic tests.

    Notes
    -----
    The analytic F_assign_uncertainty returned in the result assumes
    independent shot draws between successive calls with the same (c0, c1).
    Passing the *same* rng object to multiple calls will advance its state
    and correlate the draws, violating that assumption — Module 2's
    error-budget decomposition relies on default rng=None.
    """
    if noise_model not in ("ideal", "gaussian"):
        raise ValueError(f"noise_model must be 'ideal' or 'gaussian', got {noise_model!r}")

    c0 = result_ground.integrated_iq(integration_window)
    c1 = result_excited.integrated_iq(integration_window)
    separation = abs(c1 - c0)
    if separation == 0:
        raise ValueError("IQ centroids coincide — dispersive regime lost or window too short.")

    t0, t1 = integration_window
    window_duration = t1 - t0
    kappa = result_ground.device.resonator.kappa
    # Shot-noise σ per quadrature for integrated homodyne output.
    # Homodyne photocurrent variance is T/2 per quadrature in the convention
    # where Gambetta 2008's integrated output is s = √(2κ) ∫⟨a⟩ dt. Scaled
    # into the |Δc| = |∫⟨a⟩ dt| integrated units, σ_per_quadrature = √(T/(4κ)),
    # giving SNR = |Δc|/σ_per_quadrature = 2√(κ/T) × |Δc| which matches
    # the standard dispersive-readout formula SNR² = 4κ |Δα|² T for
    # well-separated steady states. Perpendicular-bisector fidelity then
    # follows F = 1 − Q(SNR/2) for equal-prior two-state discrimination.
    # Plan draft's σ = √(κT/2) reversed κ ↔ 1/κ and gave SNR values that
    # didn't match Gambetta — caught when reference device gave 50% fidelity
    # instead of the ≥95% the plan expected at these parameters.
    sigma = np.sqrt(window_duration / (4.0 * kappa)) if noise_model == "gaussian" else 0.0

    if rng is None:
        rng = np.random.default_rng()

    if sigma == 0.0:
        # Ideal case: all shots fall on the centroid; F = 1 if centroids differ.
        F = 1.0
        F_unc = 0.0
    else:
        # Per-quadrature noise: each of Re, Im gets an independent σ-Gaussian.
        draws_0 = c0 + sigma * (
            rng.standard_normal(n_shots) + 1j * rng.standard_normal(n_shots)
        )
        draws_1 = c1 + sigma * (
            rng.standard_normal(n_shots) + 1j * rng.standard_normal(n_shots)
        )
        # Perpendicular-bisector discriminator:
        # decision axis = unit vector from c0 to c1; midpoint = (c0+c1)/2.
        axis = (c1 - c0) / separation
        midpoint = 0.5 * (c0 + c1)
        proj_0 = np.real((draws_0 - midpoint) * np.conj(axis))
        proj_1 = np.real((draws_1 - midpoint) * np.conj(axis))
        # Classify: proj > 0 → predicted |1>
        wrong_0 = np.mean(proj_0 > 0)   # P(1|0)
        wrong_1 = np.mean(proj_1 <= 0)  # P(0|1)
        F = 1.0 - 0.5 * (wrong_0 + wrong_1)
        # Bootstrap uncertainty (binomial-standard-error of F)
        F_unc = np.sqrt(F * (1.0 - F) / n_shots)

    return AssignmentFidelityResult(
        F_assign=float(F),
        F_assign_uncertainty=float(F_unc),
        centroid_0=complex(c0),
        centroid_1=complex(c1),
        snr=float(separation / sigma) if sigma > 0 else float("inf"),
        separation_distance=float(separation),
        integration_window=(float(t0), float(t1)),
        n_shots=int(n_shots),
        noise_model=noise_model,
    )


def snr_vs_integration_time(
    device: DeviceConfig,
    drive_params: DriveParams,
    t_integration_values: np.ndarray,
) -> np.ndarray:
    """SNR(t_int) using the physical homodyne formula SNR = |Δc|/σ_per_quadrature
    with σ_per_quadrature = √(t_int / (4κ)).

    Runs one |0⟩ trajectory and one |1⟩ trajectory out to the maximum
    t_integration value, then computes SNR for each window (0, t_int).
    """
    if np.any(t_integration_values <= 0):
        raise ValueError("t_integration_values must be strictly positive.")

    t_max = float(t_integration_values.max())
    if t_max > drive_params.duration:
        raise ValueError(
            f"t_integration max {t_max*1e9:.1f} ns exceeds drive duration "
            f"{drive_params.duration*1e9:.1f} ns."
        )

    # Use a fine grid so cumulative-trapezoid integration is accurate at all windows.
    t_list = np.linspace(0.0, drive_params.duration, 1001)
    r0 = simulate_readout(device, drive_params, initial_qubit_state=0, t_list=t_list)
    r1 = simulate_readout(device, drive_params, initial_qubit_state=1, t_list=t_list)

    snrs = np.zeros_like(t_integration_values, dtype=float)
    kappa = device.resonator.kappa
    for i, t_int in enumerate(t_integration_values):
        c0 = r0.integrated_iq((0.0, float(t_int)))
        c1 = r1.integrated_iq((0.0, float(t_int)))
        sep = abs(c1 - c0)
        sigma = np.sqrt(float(t_int) / (4.0 * kappa))
        snrs[i] = sep / sigma if sigma > 0 else float("inf")
    return snrs
