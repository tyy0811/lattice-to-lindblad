"""Module 3 — closed-form synthetic trace generators and bundle I/O.

Amendment 1: traces are CLOSED-FORM ANALYTIC (P₁ as an exact function of the
ground-truth parameters), not Lindblad-simulated. The recovery harness tests
the fitter's statistical behavior; Module 1 V3/V4a/V4b already validate the
Lindblad dynamics.

Amendment 2: Rabi fit form is `P₁(ε) = A + B·cos(π·ε/ε_π + φ)` with no T_R
envelope — T_R is unidentifiable from an amplitude sweep at fixed τ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    load_reference_F_full,
)


@dataclass(frozen=True)
class TraceData:
    """Container for one protocol's measurement trace.

    On disk (.npz): one file per bundle, one entry per trace with a JSON-
    serialized metadata blob (so arbitrary Python types survive the round-trip).
    """
    protocol: str
    sweep_axis: str
    sweep_values: np.ndarray
    P1: np.ndarray
    P1_uncertainty: np.ndarray
    metadata: dict


_REQUIRED_TRACE_FIELDS = ("protocol", "sweep_axis", "sweep_values", "P1", "P1_uncertainty", "metadata")


def generate_rabi_trace(
    epsilon_pi: float,
    omega_q: float,
    noise: NoiseModelParams,
    n_points: int = 101,
    amplitude_span_mult: tuple[float, float] = (0.0, 2.5),
    seed: int | None = None,
) -> TraceData:
    """Closed-form Rabi trace.

    Form: P₁(ε) = 0.5 + 0.5·cos(π·ε/ε_π·(1 + δ_amp)) per spec §4.2
    (A + B·cos form with A≈B≈0.5, φ=0; P₁=1 at ε=0, dips to 0 at ε=ε_π).
    δ_amp is a once-per-run Gaussian calibration offset of SD
    `drive_amplitude_uncertainty`. The 1/f drift does NOT enter at leading
    order — Rabi rate depends on transmon dipole, not ω_q.

    Noise stack: (1) amplitude calibration offset (scalar per run); (2) binomial
    shot noise per point; (3) symmetric readout errors using Module 2's F_full.
    """
    rng = np.random.default_rng(seed)
    F_assign = load_reference_F_full()
    eps = np.linspace(
        amplitude_span_mult[0] * epsilon_pi,
        amplitude_span_mult[1] * epsilon_pi,
        n_points,
    )
    delta_amp = rng.normal(0.0, noise.drive_amplitude_uncertainty) if noise.drive_amplitude_uncertainty > 0 else 0.0
    eps_effective = eps * (1.0 + delta_amp)
    P_true = 0.5 + 0.5 * np.cos(np.pi * eps_effective / epsilon_pi)
    P_after_readout = apply_readout_errors(P_true, F_assign)
    P_observed = apply_shot_noise(P_after_readout, noise.n_shots_per_point, rng)
    P_ro_c = np.clip(P_after_readout, 1e-12, 1 - 1e-12)
    P_se = np.sqrt(P_ro_c * (1 - P_ro_c) / noise.n_shots_per_point)
    return TraceData(
        protocol="rabi",
        sweep_axis="drive_amplitude",
        sweep_values=eps,
        P1=P_observed,
        P1_uncertainty=P_se,
        metadata={
            "ground_truth": {"epsilon_pi": epsilon_pi, "omega_q": omega_q},
            "noise": {
                "n_shots_per_point": noise.n_shots_per_point,
                "drift_amplitude_Hz": noise.drift_amplitude_Hz,
                "drift_alpha": noise.drift_alpha,
                "drive_amplitude_uncertainty": noise.drive_amplitude_uncertainty,
                "F_assign": F_assign,
            },
            "seed": seed,
            "delta_amp_realization": float(delta_amp),
        },
    )


# -- Bundle I/O ---------------------------------------------------------------

def save_trace_bundle(traces: list[TraceData], path: str | Path) -> None:
    """Save a list of traces to .npz."""
    payload: dict[str, np.ndarray] = {"n_traces": np.array(len(traces))}
    for i, t in enumerate(traces):
        payload[f"traces/{i}/protocol"] = np.array(t.protocol)
        payload[f"traces/{i}/sweep_axis"] = np.array(t.sweep_axis)
        payload[f"traces/{i}/sweep_values"] = np.asarray(t.sweep_values)
        payload[f"traces/{i}/P1"] = np.asarray(t.P1)
        payload[f"traces/{i}/P1_uncertainty"] = np.asarray(t.P1_uncertainty)
        payload[f"traces/{i}/metadata_json"] = np.array(json.dumps(t.metadata))
    np.savez(path, **payload)


def load_trace_bundle(path: str | Path) -> list[TraceData]:
    """Load a .npz trace bundle; raises ValueError on missing fields.

    Schema validation per §8 flag #5.
    """
    raw = np.load(path, allow_pickle=False)
    n = int(raw["n_traces"])
    out: list[TraceData] = []
    for i in range(n):
        for field_name in _REQUIRED_TRACE_FIELDS:
            key = f"traces/{i}/{field_name}" if field_name != "metadata" else f"traces/{i}/metadata_json"
            if key not in raw:
                raise ValueError(f"Trace bundle missing required field '{field_name}' on entry {i}")
        out.append(TraceData(
            protocol=str(raw[f"traces/{i}/protocol"]),
            sweep_axis=str(raw[f"traces/{i}/sweep_axis"]),
            sweep_values=np.array(raw[f"traces/{i}/sweep_values"]),
            P1=np.array(raw[f"traces/{i}/P1"]),
            P1_uncertainty=np.array(raw[f"traces/{i}/P1_uncertainty"]),
            metadata=json.loads(str(raw[f"traces/{i}/metadata_json"])),
        ))
    return out
