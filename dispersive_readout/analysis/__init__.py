"""Stage 06 Module 2 — error-budget decomposition and Figure 2 data model.

See 06_Dispersive_Readout/MODULE_2_SPEC.md for the design contract.
"""
from .operating_point import (
    OperatingPoint,
    calibrate_drive_amplitude,
    get_reference_operating_point,
)
from .purcell_isolation import analytic_purcell_rate
