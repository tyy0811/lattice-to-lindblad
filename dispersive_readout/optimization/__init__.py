"""Stage 06 Module 4 — sensitivity + Pareto + closed-loop optimization layer.

See 06_Dispersive_Readout/MODULE_4_SPEC.md for the design contract.

Public API is populated incrementally across Tasks 2–21. At end-of-Module-4
this __init__.py re-exports:
    - Policy constants: SENSITIVITY_FD_STEP, SENSITIVITY_RENDER_BAR_THRESHOLD,
                        SENSITIVITY_WARNING_THRESHOLD
    - Schemas: SensitivityResult, ParetoPoint, DevicePoint, RecommendationReport
    - Functions: compute_all_sensitivities, compute_pareto_frontier,
                 recommend_from_fitted_parameters, pareto_one_tuple
    - Data: PUBLISHED_DEVICE_POINTS, PARETO_DEVICE_VARIANTS
"""
from .sensitivity import (
    SENSITIVITY_FD_STEP,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
    SENSITIVITY_WARNING_THRESHOLD,
)

__all__ = [
    "SENSITIVITY_FD_STEP",
    "SENSITIVITY_RENDER_BAR_THRESHOLD",
    "SENSITIVITY_WARNING_THRESHOLD",
]
