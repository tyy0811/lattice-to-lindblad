"""Module 3 — lmfit-based parameter extraction.

This module has two layers:
  1. Pydantic schemas (FittedParameter, ExtractedParameterPack) with a
     to_device_config bridge that back-solves E_J from ω_q per Koch 2007
     (amendment 5).
  2. lmfit wrappers + parametric_bootstrap (amendment 3). The wrappers
     arrive in Task 8; bootstrap in Task 9.
"""
from __future__ import annotations

import math
import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FittedParameter(BaseModel):
    """One fitted device parameter with bootstrap uncertainty."""
    name: Literal["T_1", "T_2_echo", "T_2_star", "omega_q", "epsilon_pi"]
    value: float
    uncertainty: float
    unit: Literal["s", "rad/s"]
    protocol_source: Literal["rabi", "ramsey", "t1", "t2_echo"]
    goodness_of_fit: float = Field(ge=0.0)
    n_bootstrap: int = Field(ge=0)

    @field_validator("uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("uncertainty must be strictly positive")
        return v


class ExtractedParameterPack(BaseModel):
    """All parameters extracted from one trace bundle."""
    fitted_parameters: list[FittedParameter]
    trace_file: str
    timestamp: str
    stage_06_version: str

    def _get(self, name: str) -> FittedParameter | None:
        for p in self.fitted_parameters:
            if p.name == name:
                return p
        return None

    def to_device_config(self, E_J_tolerance_rel: float = 0.30):
        """Bridge fitted parameters to Module 1's DeviceConfig (amendment 5).

        Policy:
          - E_C held fixed at REFERENCE_DEVICE (geometric, not fit).
          - E_J back-solved from fitted ω_q via Koch 2007:
                E_J = (ω_q + E_C)² / (8·E_C)
          - γ_1 = 1 / T_1; γ_φ from T_2_echo via 1/T_2 = γ_1/2 + γ_φ.
          - resonator, coupling, truncation inherited from REFERENCE_DEVICE.
          - UserWarning if |E_J − E_J_REFERENCE| / E_J_REFERENCE > 30%.
        """
        from dispersive_readout.physics.config import (
            DecoherenceParams, DeviceConfig, REFERENCE_DEVICE, TransmonParams,
        )
        omega_q_fp = self._get("omega_q")
        T_1_fp = self._get("T_1")
        T_2_echo_fp = self._get("T_2_echo")
        if omega_q_fp is None or T_1_fp is None or T_2_echo_fp is None:
            raise ValueError(
                "to_device_config requires omega_q, T_1, and T_2_echo fits. "
                "Missing: " + ", ".join(n for n, v in (
                    ("omega_q", omega_q_fp), ("T_1", T_1_fp), ("T_2_echo", T_2_echo_fp),
                ) if v is None)
            )
        E_C = REFERENCE_DEVICE.transmon.E_C
        omega_q = omega_q_fp.value
        E_J_derived = (omega_q + E_C) ** 2 / (8.0 * E_C)
        E_J_reference = REFERENCE_DEVICE.transmon.E_J
        rel_drift = abs(E_J_derived - E_J_reference) / E_J_reference
        if rel_drift > E_J_tolerance_rel:
            warnings.warn(
                f"Derived E_J/2π = {E_J_derived / (2 * math.pi) / 1e9:.3f} GHz is "
                f"{rel_drift:.1%} off REFERENCE's E_J/2π = "
                f"{E_J_reference / (2 * math.pi) / 1e9:.3f} GHz — check the fit.",
                UserWarning,
                stacklevel=2,
            )
        transmon = TransmonParams(E_C=E_C, E_J=E_J_derived, n_g=REFERENCE_DEVICE.transmon.n_g)
        gamma_1 = 1.0 / T_1_fp.value
        gamma_phi = max(1.0 / T_2_echo_fp.value - 0.5 * gamma_1, 0.0)
        decoherence = DecoherenceParams(
            gamma_1=gamma_1, gamma_phi=gamma_phi,
            n_th=REFERENCE_DEVICE.decoherence.n_th,
            purcell_enabled=REFERENCE_DEVICE.decoherence.purcell_enabled,
        )
        return DeviceConfig(
            transmon=transmon,
            resonator=REFERENCE_DEVICE.resonator,
            coupling=REFERENCE_DEVICE.coupling,
            decoherence=decoherence,
            truncation=REFERENCE_DEVICE.truncation,
        )
