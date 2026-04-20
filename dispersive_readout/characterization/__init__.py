"""Stage 06 Module 3 — parameter characterization protocols."""
from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    generate_1f_drift,
    load_reference_F_full,
)
from .fitting import (
    ExtractedParameterPack,
    FittedParameter,
    fit_rabi,
    fit_ramsey,
    fit_t1,
    fit_t2_echo,
)
from .protocols import (
    TraceData,
    generate_rabi_trace,
    generate_ramsey_trace,
    generate_t1_trace,
    generate_t2_echo_trace,
    load_trace_bundle,
    save_trace_bundle,
)

__all__ = [
    "ExtractedParameterPack",
    "FittedParameter",
    "NoiseModelParams",
    "fit_rabi",
    "fit_ramsey",
    "fit_t1",
    "fit_t2_echo",
    "TraceData",
    "apply_readout_errors",
    "apply_shot_noise",
    "generate_1f_drift",
    "generate_rabi_trace",
    "generate_ramsey_trace",
    "generate_t1_trace",
    "generate_t2_echo_trace",
    "load_reference_F_full",
    "load_trace_bundle",
    "save_trace_bundle",
]
