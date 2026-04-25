"""Frozen-dataclass config for the dispersive-readout simulator.

All rates and frequencies are stored in angular-frequency units (rad/s).
Hz values appear only at I/O boundaries (property accessors, display,
docstrings). See conversation plan header for citation trail.

REFERENCE_DEVICE values follow Marxer et al., arXiv:2508.16437 (IQM Munich,
Aug 2025) tunable-coupler + shelving-readout device; Bengtsson et al.,
Phys. Rev. Lett. 132, 100603 (2024) is the secondary cross-check reference.
Where Marxer does not tabulate an exact value (bare g, Δ), we use the
mid-range of IQM published values and mark the derivation in the field
comment. No proprietary data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TransmonParams:
    """Transmon qubit parameters (rad/s for energies)."""
    E_C: float
    E_J: float
    n_g: float = 0.0

    @property
    def E_C_Hz(self) -> float:
        return self.E_C / _TWO_PI

    @property
    def E_J_Hz(self) -> float:
        return self.E_J / _TWO_PI


@dataclass(frozen=True)
class ResonatorParams:
    """Readout resonator parameters."""
    omega_r: float  # resonator frequency, rad/s
    kappa: float    # total linewidth, rad/s


@dataclass(frozen=True)
class CouplingParams:
    """Transmon-resonator bare coupling."""
    g: float  # rad/s


@dataclass(frozen=True)
class DecoherenceParams:
    """Incoherent error channels.

    gamma_1:          qubit relaxation rate (1/s, equivalently rad/s for rates).
    gamma_phi:        pure dephasing rate; from T2_echo after subtracting gamma_1/2.
    n_th:             bath thermal population (dimensionless).
    purcell_enabled:  if False, omit Purcell collapse operators in
                      build_collapse_operators. Used by Module 2's Purcell
                      turn-off channel to isolate the Purcell contribution.
    """
    gamma_1: float
    gamma_phi: float
    n_th: float = 0.01
    purcell_enabled: bool = True


@dataclass(frozen=True)
class DriveParams:
    """Readout drive pulse parameters.

    amplitude:  epsilon_0, rad/s.
    duration:   total pulse length, seconds.
    detuning:   omega_drive - omega_resonator, rad/s (0 = on resonance).
    edge_sigma: Gaussian-edge width for the erf-difference envelope, seconds.
    """
    amplitude: float
    duration: float
    detuning: float = 0.0
    edge_sigma: float = 2e-9


@dataclass(frozen=True)
class TruncationParams:
    """Hilbert-space truncation sizes.

    N_charge:    # charge states in [-N_charge//2, +N_charge//2]; must be odd.
                 Koch 2007 requires N_charge >> sqrt(8 E_J / E_C) for charge
                 dispersion to converge. For REFERENCE_DEVICE (E_J/E_C ≈ 74)
                 sqrt(8 * 74) ≈ 24, so default = 31 gives ~6 points of margin.
    N_transmon:  transmon levels kept after diagonalization.
                 Must satisfy 1 <= N_transmon <= N_charge.
    N_resonator: resonator Fock basis size. Runtime-checked against
                 mean photon number during readout (readout_model.py).
    """
    N_charge: int = 31
    N_transmon: int = 5
    N_resonator: int = 15

    def __post_init__(self) -> None:
        if self.N_charge < 3:
            raise ValueError(f"N_charge must be >= 3 (got {self.N_charge}).")
        if self.N_charge % 2 == 0:
            raise ValueError(
                f"N_charge must be odd (got {self.N_charge}) so the ladder is symmetric about zero."
            )
        if self.N_transmon < 1:
            raise ValueError(f"N_transmon must be >= 1 (got {self.N_transmon}).")
        if self.N_transmon > self.N_charge:
            raise ValueError(
                f"N_transmon ({self.N_transmon}) cannot exceed N_charge ({self.N_charge})."
            )
        if self.N_resonator < 1:
            raise ValueError(f"N_resonator must be >= 1 (got {self.N_resonator}).")


@dataclass(frozen=True)
class DeviceConfig:
    """Complete device spec. Bundles the five param groups above."""
    transmon: TransmonParams
    resonator: ResonatorParams
    coupling: CouplingParams
    decoherence: DecoherenceParams
    truncation: TruncationParams = field(default_factory=TruncationParams)


# T1 = 30 us → γ1 = 1/T1 (≈ 5.3 kHz in "/2π" display units)
_T1_SEC = 30e-6
_T2_ECHO_SEC = 40e-6
_GAMMA_1 = 1.0 / _T1_SEC
# γφ from T2_echo relation: 1/T2 = γ1/2 + γφ  →  γφ = 1/T2 − γ1/2
_GAMMA_PHI = max(1.0 / _T2_ECHO_SEC - 0.5 * _GAMMA_1, 0.0)


REFERENCE_DEVICE: DeviceConfig = DeviceConfig(
    transmon=TransmonParams(
        E_C=_TWO_PI * 210e6,     # 210 MHz — Marxer 2508.16437 anharmonicity range
        E_J=_TWO_PI * 15.5e9,    # 15.5 GHz — gives E_J/E_C ≈ 74 (deep transmon, Koch 2007)
        n_g=0.0,                 # sweet spot
    ),
    resonator=ResonatorParams(
        omega_r=_TWO_PI * 7.3e9, # 7.3 GHz — within IQM tunable-coupler arch readout band
        kappa=_TWO_PI * 5e6,     # 5 MHz — fast-readout regime
    ),
    coupling=CouplingParams(
        g=_TWO_PI * 120e6,       # 120 MHz — mid-range IQM value (derived from reported χ, κ)
    ),
    decoherence=DecoherenceParams(
        gamma_1=_GAMMA_1,        # from T1 = 30 μs
        gamma_phi=_GAMMA_PHI,    # from T2_echo = 40 μs after γ1/2 subtraction
        n_th=0.01,               # ~30 mK base temperature
    ),
    truncation=TruncationParams(),
)
