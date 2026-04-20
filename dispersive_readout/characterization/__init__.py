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
    fit_all,
    fit_rabi,
    fit_ramsey,
    fit_t1,
    fit_t2_echo,
    parametric_bootstrap,
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
from .recovery import (
    CoverageReport,
    DeviceGroundTruth,
    RecoveryResult,
    fit_one_device,
    generate_synthetic_device_family,
)

__all__ = [
    "CoverageReport",
    "DeviceGroundTruth",
    "ExtractedParameterPack",
    "FittedParameter",
    "NoiseModelParams",
    "RecoveryResult",
    "fit_one_device",
    "generate_synthetic_device_family",
    "fit_all",
    "fit_rabi",
    "fit_ramsey",
    "fit_t1",
    "fit_t2_echo",
    "parametric_bootstrap",
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
