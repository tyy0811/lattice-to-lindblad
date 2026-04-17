"""Public API for the dispersive-readout physics foundation (Module 1).

Stable entry points for scripts and downstream modules:
    - Config dataclasses and REFERENCE_DEVICE
    - simulate_readout, ReadoutResult
    - compute_assignment_fidelity, AssignmentFidelityResult
    - snr_vs_integration_time
    - transmon_summary (for quick device sanity checks)
    - dispersive_shift_{two_level, full, from_simulation} (for validation)
"""
from .config import (
    REFERENCE_DEVICE,
    CouplingParams,
    DecoherenceParams,
    DeviceConfig,
    DriveParams,
    ResonatorParams,
    TransmonParams,
    TruncationParams,
)
from .dispersive import (
    dispersive_shift_from_simulation,
    dispersive_shift_full,
    dispersive_shift_two_level,
)
from .readout_model import (
    AssignmentFidelityResult,
    ReadoutResult,
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)
from .transmon import (
    charge_basis_hamiltonian,
    charge_operator_matrix_elements,
    diagonalize_transmon,
    transmon_summary,
)

__all__ = [
    # config
    "REFERENCE_DEVICE",
    "CouplingParams",
    "DecoherenceParams",
    "DeviceConfig",
    "DriveParams",
    "ResonatorParams",
    "TransmonParams",
    "TruncationParams",
    # transmon
    "charge_basis_hamiltonian",
    "charge_operator_matrix_elements",
    "diagonalize_transmon",
    "transmon_summary",
    # dispersive
    "dispersive_shift_from_simulation",
    "dispersive_shift_full",
    "dispersive_shift_two_level",
    # readout model
    "AssignmentFidelityResult",
    "ReadoutResult",
    "compute_assignment_fidelity",
    "simulate_readout",
    "snr_vs_integration_time",
]
