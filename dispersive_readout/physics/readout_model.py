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
        return np.trapz(self.a_expectation[mask], self.t[mask])


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

    H0, drive_spec = build_hamiltonian(device, drive_params, frame="rotating")
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
) -> AssignmentFidelityResult:
    raise NotImplementedError  # Task 17


def snr_vs_integration_time(
    device: DeviceConfig,
    drive_params: DriveParams,
    t_integration_values: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError  # Task 18
