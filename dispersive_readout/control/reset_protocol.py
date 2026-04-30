"""Module 5b — direct-jump joint transition-readout active reset.

Direct-jump v0 reset model: samples T₁/Purcell jump times exponentially,
analytically integrates the dispersive cavity equation of motion conditioned
on the resulting piecewise qubit-state history (via
dispersive_readout.physics.pointer_response), adds Module-1-consistent
Gaussian IQ noise, classifies via classify_iq (Module 1's perpendicular-
bisector discriminator), and produces a JointMatrix(P(s_f, m | s_i)) that
the closed-form reset_residual_single_cycle formula consumes.

v0 explicitly excludes mcsolve — a v1.5 extension may add mcsolve-based
jump-history sampling for richer non-Markovian effects, but cavity response
would still flow through pointer_response. See test_no_mcsolve_in_reset_
protocol for the lint-grade enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    DecoherenceParams,
    DeviceConfig,
    DriveParams,
    TransmonParams,
)


@dataclass(frozen=True)
class QubitStateHistory:
    """Piecewise-constant qubit-state history over [0, t_total].

    segments: tuple of (t_start, qubit_state) pairs. Validated at
    construction:
      - segments[0][0] == 0.0 (first segment starts at 0)
      - t_start values strictly monotonically increasing
      - all t_start < t_total
      - qubit_state ∈ {0, 1} (v0 has no thermal/leakage states)

    The last segment runs from its t_start to t_total. v0 has at most two
    segments (initial state + optional one jump); the dataclass shape
    extends naturally to multi-jump for v1.5 mcsolve sampling.
    """
    segments: tuple[tuple[float, int], ...]
    t_total: float

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must be non-empty")
        if self.segments[0][0] != 0.0:
            raise ValueError(
                f"first segment must start at 0.0 (got {self.segments[0][0]})"
            )
        for i, (t_start, q) in enumerate(self.segments):
            if q not in (0, 1):
                raise ValueError(
                    f"qubit_state ∈ {{0, 1}} required (got {q} at segment {i})"
                )
            if t_start >= self.t_total:
                raise ValueError(
                    f"segment {i} t_start={t_start} exceeds t_total={self.t_total}"
                )
            if i > 0 and t_start <= self.segments[i - 1][0]:
                raise ValueError(
                    f"segments must be strictly monotonic in t_start "
                    f"(segment {i} t_start={t_start} <= segment {i-1} "
                    f"t_start={self.segments[i-1][0]})"
                )


# ---------------------------------------------------------------------------
# Day 2.1 — operating-point helpers
# ---------------------------------------------------------------------------

_CLOSED_LOOP_YAML_PATH = (
    Path(__file__).parent.parent.parent
    / "06_Dispersive_Readout"
    / "figures"
    / "closed_loop_demo_device.yaml"
)

_FIG5A_DATA_YAML_PATH = (
    Path(__file__).parent.parent.parent
    / "06_Dispersive_Readout"
    / "figures"
    / "fig5a_drag_leakage_data.yaml"
)

# v0 explicitly assumes zero thermal qubit population: extract_joint_matrix
# does not sample thermal g→e excitation, so any device with n_th > 0 would
# silently drop a transition pathway that physics.lindblad.build_collapse_
# operators *does* model (line 108-114). The reset model is consistent only
# at strict zero temperature. device_idx18 enforces this by overriding
# REFERENCE_DEVICE.decoherence.n_th to 0; extract_joint_matrix raises
# NotImplementedError if it ever sees n_th > 0.


def closed_loop_demo_drive_params(duration: float) -> DriveParams:
    """DriveParams for the closed-loop demo device idx=18.

    Parameter named `duration` to match the DriveParams field it sets.
    "tau_meas" is 5b's physics-side terminology for the same quantity
    since v0 has the drive-on window equal to the integration window.

    Updates only DriveParams.duration. amplitude (140 MHz from idx=18
    Pareto optimum), detuning (0.0 = on resonance with bare cavity),
    and edge_sigma (REFERENCE default 2 ns) are fixed across the entire
    τ_meas sweep.
    """
    return DriveParams(
        amplitude=140e6,
        duration=duration,
        detuning=0.0,
        edge_sigma=2e-9,
    )


def device_idx18(yaml_path: Path | None = None) -> DeviceConfig:
    """Construct closed-loop demo device idx=18 from Module 4's YAML.

    Inherits (κ, g, ω_r, n̄_q, transmon E_C) from REFERENCE_DEVICE;
    overrides (T₁ → γ_1, T₂_echo → γ_φ, ω_q → E_J_derived) from the
    'chosen' block of yaml_path. ε_drive is exposed via DriveParams,
    not the device — see closed_loop_demo_drive_params.

    yaml_path defaults to the canonical Module 4 figure path, resolved
    relative to __file__. Tests inject an alternate path via the kwarg.

    v0 zero-temp invariant: n_th is set to 0.0 in the returned device,
    overriding REFERENCE_DEVICE.decoherence.n_th (typically 0.01). This
    matches the v0 reset model in extract_joint_matrix, which does not
    sample thermal g→e excitation. Thermal-aware reset is v1.5 territory;
    devices with n_th > 0 would silently drop a transition pathway the
    underlying Lindblad simulator does model.

    Raises:
      FileNotFoundError if yaml_path missing (with regeneration hint).
      KeyError if the 'chosen' block schema has changed.
    """
    if yaml_path is None:
        yaml_path = _CLOSED_LOOP_YAML_PATH

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"closed_loop_demo_device.yaml not found at {yaml_path}. "
            f"Run 06_Dispersive_Readout/scripts/fig4_optimization.py to regenerate."
        )

    data = yaml.safe_load(yaml_path.read_text())
    chosen = data['chosen']

    T_1 = chosen['T_1_us'] * 1e-6
    T_2_echo = chosen['T_2_echo_us'] * 1e-6
    omega_q_target = chosen['omega_q_GHz'] * 1e9 * 2 * np.pi

    gamma_1 = 1.0 / T_1
    # γ_φ = 1/T_2_echo - γ_1/2 per the standard echo convention
    gamma_phi = 1.0 / T_2_echo - gamma_1 / 2.0
    if gamma_phi < 0:
        raise ValueError(
            f"Negative γ_φ derived from T_2_echo={T_2_echo}, T_1={T_1}; "
            f"check YAML."
        )

    # E_J derived from ω_q target while holding E_C fixed at REFERENCE
    # via the simple transmon dispersion ω_q ≈ √(8 E_J E_C) - E_C, so
    # E_J = (ω_q + E_C)² / (8 E_C). For idx=18 ω_q≈4.72 GHz the derived
    # E_J differs from REFERENCE's by only the proportional rescale needed
    # to hit the target frequency.
    E_C = REFERENCE_DEVICE.transmon.E_C
    E_J_derived = (omega_q_target + E_C) ** 2 / (8.0 * E_C)

    transmon = TransmonParams(
        E_C=E_C, E_J=E_J_derived, n_g=REFERENCE_DEVICE.transmon.n_g,
    )
    decoherence = DecoherenceParams(
        gamma_1=gamma_1,
        gamma_phi=gamma_phi,
        n_th=0.0,  # v0 zero-temp reset model (overrides REFERENCE_DEVICE.n_th)
        purcell_enabled=REFERENCE_DEVICE.decoherence.purcell_enabled,
    )
    return DeviceConfig(
        transmon=transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=decoherence,
        truncation=REFERENCE_DEVICE.truncation,
    )


def load_eps_x_5a(
    t_gate: float = 20e-9,
    yaml_path: Path | None = None,
) -> tuple[float, dict]:
    """Load 5a's ε_X = 1 - F_avg at the given T_gate from fig5a's data YAML.

    Returns (eps_x, provenance), where provenance is a dict capturing:
      - source_yaml:    string path of the YAML
      - source_mtime:   mtime stamp at load time (staleness detection)
      - T_gate_ns:      the T_gate row used
      - beta_opt:       the matching β_opt from fig5a's calibration
                        (sourced from beta_opt_fidelity, the F_avg-optimal
                        β grid in the 5a YAML)
      - F_avg_drag_opt: 1 - eps_x (the underlying fidelity)

    Provenance flows into fig5b's data YAML under epsilon_x_5a_provenance
    for full lineage from idx=18 reset → 5a's gate calibration.

    yaml_path defaults to the canonical fig5a output path, resolved
    relative to __file__. Tests inject.

    Raises:
      FileNotFoundError if yaml_path missing.
      ValueError if t_gate not in 5a's sweep_T_gate_ns grid.
      KeyError if the YAML schema has changed.
    """
    if yaml_path is None:
        yaml_path = _FIG5A_DATA_YAML_PATH

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"5a data YAML not found at {yaml_path}. "
            f"Run 06_Dispersive_Readout/scripts/fig5a_drag_leakage.py first."
        )

    mtime_at_load = yaml_path.stat().st_mtime
    data = yaml.safe_load(yaml_path.read_text())

    sweep = data['sweep_T_gate_ns']
    eps_x_curve = data['epsilon_x_drag_opt']
    beta_opt_curve = data['beta_opt_fidelity']

    t_gate_ns = t_gate * 1e9
    if t_gate_ns not in sweep:
        raise ValueError(
            f"T_gate={t_gate_ns}ns not in 5a's sweep grid {sweep}; "
            f"available T_gate values: {sweep}."
        )

    idx = sweep.index(t_gate_ns)
    eps_x = float(eps_x_curve[idx])

    provenance = {
        'source_yaml': str(yaml_path),
        'source_mtime': mtime_at_load,
        'T_gate_ns': float(t_gate_ns),
        'beta_opt': float(beta_opt_curve[idx]),
        'F_avg_drag_opt': 1.0 - eps_x,
    }
    return eps_x, provenance


# ---------------------------------------------------------------------------
# Day 2.3 — purcell_rate_1_to_0
# ---------------------------------------------------------------------------

from dispersive_readout.physics.transmon import (
    charge_operator_matrix_elements,
    diagonalize_transmon,
)


from dispersive_readout.physics.dispersive import dispersive_shift_full
from dispersive_readout.physics.joint_matrix import JointMatrix
from dispersive_readout.physics.pointer_response import (
    compute_alpha_trajectory,
)
from dispersive_readout.physics.readout_model import classify_iq


def purcell_rate_1_to_0(device: DeviceConfig) -> float:
    """Purcell decay rate for the |1⟩→|0⟩ transition.

    γ_P = (g · |⟨0|n̂|1⟩| / Δ_{1,0})² · κ · (1 + n_th)

    where Δ_{1,0} = ω_1 − ω_0 − ω_r per the dispersive frame's Purcell
    construction. Sourced from the same formula used in
    physics.lindblad.build_collapse_operators (j=1 row); single source
    of truth so 5b and Module 1 simulate consistent γ_P.

    Used by extract_joint_matrix to construct γ_eff = γ_1 + γ_P for
    direct-jump exponential sampling. The factor (1 + n_th) is included
    for consistency with Module 2's qubit-relaxation channel.
    """
    energies, eigenstates = diagonalize_transmon(
        device.transmon, device.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, device.truncation)
    delta_10 = energies[1] - energies[0] - device.resonator.omega_r
    n_elem_01 = abs(n_mat[0, 1])
    return (
        (device.coupling.g * n_elem_01 / delta_10) ** 2
        * device.resonator.kappa
        * (1.0 + device.decoherence.n_th)
    )


# ---------------------------------------------------------------------------
# Day 2.4 — extract_joint_matrix direct-jump sampler
# ---------------------------------------------------------------------------


def extract_joint_matrix(
    device: DeviceConfig,
    drive_params: DriveParams,
    n_trajectories: int = 1000,
    threshold: Literal['midpoint'] = 'midpoint',
    rng: np.random.Generator | None = None,
) -> JointMatrix:
    """Direct-jump joint matrix extraction.

    Measurement window: tau_meas := drive_params.duration in v0 (single
    source of truth). v1.5 may add an integration_window arg for sub-
    interval integration (e.g., likelihood-ratio threshold optimization).

    Per s_i ∈ {0, 1} (in SEPARATE outer loops with rng.spawn(2) for
    substream independence):
      1. Sample t_jump ~ Exp(γ_eff) if s_i=1, else no jump (v0 has no
         thermal excitation from |g⟩).
      2. Build QubitStateHistory.
      3. compute_alpha_trajectory → integrated_iq.
      4. Add Gaussian shot noise σ = √(tau_meas/(4κ)) (Module 1 noise
         model, η=1 implicit).
      5. classify_iq → m ∈ {0, 1}.
      6. Tally (s_i, s_f, m).

    threshold is Literal['midpoint'] in v0; v1.5 may add 'likelihood_ratio'.

    rng=None defaults to np.random.default_rng() per numpy convention.
    """
    if threshold != 'midpoint':
        raise ValueError(
            f"v0 supports only threshold='midpoint' (got {threshold!r}); "
            f"likelihood_ratio is v1.5 territory."
        )

    if device.decoherence.n_th > 0:
        raise NotImplementedError(
            f"v0 reset model assumes zero qubit thermal population "
            f"(got n_th = {device.decoherence.n_th}). The s_i=0 branch hard-"
            f"codes |g⟩-stays-|g⟩ for the entire measurement window, which "
            f"silently drops the thermal g→e excitation pathway that "
            f"physics.lindblad.build_collapse_operators models. Either pass "
            f"a device with n_th=0 (e.g., from device_idx18, which sets "
            f"n_th=0 explicitly) or wait for v1.5 thermal-aware sampling."
        )

    if rng is None:
        rng = np.random.default_rng()

    tau_meas = drive_params.duration
    kappa = device.resonator.kappa
    gamma_1 = device.decoherence.gamma_1
    gamma_purcell = (
        purcell_rate_1_to_0(device)
        if device.decoherence.purcell_enabled
        else 0.0
    )
    gamma_eff = gamma_1 + gamma_purcell

    # Pre-compute pure-g and pure-e centroids (consumed by classify_iq)
    t_grid_2pt = np.array([0.0, tau_meas])
    history_g = QubitStateHistory(segments=((0.0, 0),), t_total=tau_meas)
    history_e = QubitStateHistory(segments=((0.0, 1),), t_total=tau_meas)
    _, centroid_g = compute_alpha_trajectory(device, drive_params, history_g, t_grid_2pt)
    _, centroid_e = compute_alpha_trajectory(device, drive_params, history_e, t_grid_2pt)

    # Module 1's noise model: σ_per_quadrature = √(τ_meas / (4κ))
    sigma_iq = float(np.sqrt(tau_meas / (4.0 * kappa)))

    # Independent substreams for s_i=0 and s_i=1
    rng_g, rng_e = rng.spawn(2)

    counts: dict[tuple[int, int, int], int] = {
        (s_i, s_f, m): 0
        for s_i in (0, 1) for s_f in (0, 1) for m in (0, 1)
    }

    for s_i, rng_si in ((0, rng_g), (1, rng_e)):
        for _ in range(n_trajectories):
            if s_i == 0:
                # v0: no thermal excitation from |g⟩
                history = QubitStateHistory(segments=((0.0, 0),), t_total=tau_meas)
                s_f = 0
            else:
                if gamma_eff > 0:
                    t_jump = rng_si.exponential(scale=1.0 / gamma_eff)
                else:
                    t_jump = float('inf')  # no jumps possible

                if t_jump < tau_meas:
                    history = QubitStateHistory(
                        segments=((0.0, 1), (t_jump, 0)),
                        t_total=tau_meas,
                    )
                    s_f = 0
                else:
                    history = QubitStateHistory(segments=((0.0, 1),), t_total=tau_meas)
                    s_f = 1

            _, integrated_iq = compute_alpha_trajectory(
                device, drive_params, history, t_grid_2pt,
            )

            # Gaussian shot noise per quadrature
            noise = sigma_iq * (
                rng_si.standard_normal() + 1j * rng_si.standard_normal()
            )
            noisy_iq = integrated_iq + noise

            m = classify_iq(noisy_iq, centroid_g, centroid_e)
            counts[(s_i, s_f, m)] += 1

    probabilities = {
        key: counts[key] / n_trajectories for key in counts
    }
    binomial_se = {
        key: float(np.sqrt(p * (1 - p) / n_trajectories))
        for key, p in probabilities.items()
    }

    # Operating-point metadata
    energies, eigenstates = diagonalize_transmon(
        device.transmon, device.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, device.truncation)
    chi_per_level = dispersive_shift_full(
        energies, n_mat, device.coupling.g, device.resonator.omega_r,
    )

    return JointMatrix(
        probabilities=probabilities,
        binomial_se=binomial_se,
        n_trajectories=n_trajectories,
        operating_point={
            'tau_meas': float(tau_meas),
            'kappa': float(kappa),
            'eps_drive': float(drive_params.amplitude),
            'delta_drive': float(drive_params.detuning),
            'chi_g': float(chi_per_level[0]),
            'chi_e': float(chi_per_level[1]),
            'gamma_1': float(gamma_1),
            'gamma_purcell': float(gamma_purcell),
            'gamma_eff': float(gamma_eff),
        },
    )


# ---------------------------------------------------------------------------
# Day 3.1 — reset formulas
# ---------------------------------------------------------------------------


def passive_reset_residual(T1: float, tau: float) -> float:
    """Closed-form passive baseline: P_e(τ) = exp(-τ/T₁) for |e⟩-prepared
    qubit decaying freely over duration τ.

    Matched-duration comparison uses τ = τ_meas + τ_gate, where τ_gate is
    the gate duration consumed by the conditional X-flip (5a's headline
    is 20 ns).
    """
    if T1 <= 0:
        raise ValueError(f"T1 must be positive (got {T1})")
    return float(np.exp(-tau / T1))


def reset_residual_single_cycle(
    p_e: float,
    joint: JointMatrix,
    gate_error: float = 0.0,
) -> float:
    """p_e' from the three-term direct-jump formula:

      p_e' = p_e · [P(s_f=e, m=0|e) + P(s_f=e, m=1|e)·ε_X +
                    P(s_f=g, m=1|e)·(1-ε_X)]
           + (1-p_e) · [P(s_f=e, m=0|g) + P(s_f=e, m=1|g)·ε_X +
                        P(s_f=g, m=1|g)·(1-ε_X)]

    Three terms per branch:
      missed-excited:   P(s_f=e, m=0 | s_i) — readout missed; no flip
      gate-failure:     P(s_f=e, m=1 | s_i) · ε_X — flip failed
      false-positive:   P(s_f=g, m=1 | s_i) · (1-ε_X) — wrong flip g→e

    Note: the fourth combination P(s_f=g, m=0 | s_i) does NOT appear:
    qubit decayed, readout correctly said 0, no flip, ends in |g⟩.

    gate_error: classical bit-flip failure probability ε_X. Default 0.0
    is the v0 stub. The natural value when 5a has shipped is 1 - F_avg
    from 5a (8.12e-4 at T_gate=20ns); fig5b renders three traces, two of
    which call this function with eps_x=0 and eps_x=8.12e-4 respectively.
    """
    if not 0.0 <= p_e <= 1.0:
        raise ValueError(f"p_e must be in [0, 1] (got {p_e})")
    if not 0.0 <= gate_error <= 1.0:
        raise ValueError(f"gate_error must be in [0, 1] (got {gate_error})")

    p = joint.probabilities

    branch_e = (
        p[(1, 1, 0)]                          # missed-excited
        + p[(1, 1, 1)] * gate_error           # gate failure
        + p[(1, 0, 1)] * (1.0 - gate_error)   # false-positive on decayed
    )
    branch_g = (
        p[(0, 1, 0)]                          # missed-excited (from g — usually ~0)
        + p[(0, 1, 1)] * gate_error           # gate failure (from g — usually ~0)
        + p[(0, 0, 1)] * (1.0 - gate_error)   # false-positive on |g⟩ → flips → |e⟩
    )

    return p_e * branch_e + (1.0 - p_e) * branch_g
