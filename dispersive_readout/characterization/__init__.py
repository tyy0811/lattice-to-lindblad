"""Stage 06 Module 3 — parameter characterization protocols.

Public API (post-Task 1):
    - NoiseModelParams
    - generate_1f_drift, apply_shot_noise, apply_readout_errors, load_reference_F_full

Additional exports land as subsequent tasks (protocols, fitting, recovery, CLI).
"""
from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    generate_1f_drift,
    load_reference_F_full,
)

__all__ = [
    "NoiseModelParams",
    "apply_readout_errors",
    "apply_shot_noise",
    "generate_1f_drift",
    "load_reference_F_full",
]
