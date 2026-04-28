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

_THERMAL_REGIME_THRESHOLD = 0.05  # v0 enforces n̄_q < 0.05; v1.5 territory above


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

    Raises:
      FileNotFoundError if yaml_path missing (with regeneration hint).
      KeyError if the 'chosen' block schema has changed.
      NotImplementedError if device.decoherence.n_th >= 0.05 — v1.5
        thermal-excitation territory.
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

    n_th = REFERENCE_DEVICE.decoherence.n_th
    if n_th >= _THERMAL_REGIME_THRESHOLD:
        raise NotImplementedError(
            f"v0 reset model assumes thermal n̄_q < {_THERMAL_REGIME_THRESHOLD}; "
            f"got n̄_q = {n_th}. Thermal-excitation initial-state preparation "
            f"is v1.5 territory."
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
        n_th=n_th,
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
