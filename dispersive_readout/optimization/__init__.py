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
    ParameterName,
    SensitivityResult,
    compute_log_sensitivity,
    compute_all_sensitivities,
    rank_sensitivities,
)
from .regime_map import (
    DevicePoint,
    PUBLISHED_DEVICE_POINTS,
    f_analytic_dispersive,
    f_analytic_dispersive_per_level,
    purcell_boundary,
    dispersive_breakdown_boundary,
    resonator_too_slow_boundary,
    compute_analytic_regime_map,
    validate_analytic_vs_lindblad,
)
from .pareto import (
    ParetoPoint,
    PARETO_DEVICE_VARIANTS,
    TAU_MAX_GRID_NS,
    build_variant,
    find_pareto_point,
    compute_pareto_frontier,
)
from .modal_pareto import app, pareto_one_tuple
from .recommend import (
    RecommendationReport,
    recommend_from_fitted_parameters,
    generate_narrative,
    export_recommendation_to_yaml,
)

__all__ = [
    "SENSITIVITY_FD_STEP",
    "SENSITIVITY_RENDER_BAR_THRESHOLD",
    "SENSITIVITY_WARNING_THRESHOLD",
    "ParameterName",
    "SensitivityResult",
    "compute_log_sensitivity",
    "compute_all_sensitivities",
    "rank_sensitivities",
    "DevicePoint",
    "PUBLISHED_DEVICE_POINTS",
    "f_analytic_dispersive",
    "f_analytic_dispersive_per_level",
    "purcell_boundary",
    "dispersive_breakdown_boundary",
    "resonator_too_slow_boundary",
    "compute_analytic_regime_map",
    "validate_analytic_vs_lindblad",
    "ParetoPoint",
    "PARETO_DEVICE_VARIANTS",
    "TAU_MAX_GRID_NS",
    "build_variant",
    "find_pareto_point",
    "compute_pareto_frontier",
    "app",
    "pareto_one_tuple",
    "RecommendationReport",
    "recommend_from_fitted_parameters",
    "generate_narrative",
    "export_recommendation_to_yaml",
]
