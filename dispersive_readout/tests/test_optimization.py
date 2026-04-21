"""Stage 06 Module 4 — optimization layer tests (O1–O24).

Test catalog per MODULE_4_SPEC.md §6.1. Convention: each test function's
docstring cites the spec test ID it implements.
"""
from __future__ import annotations


def test_policy_constants_present_and_frozen():
    """Policy constants must live in source with locked values (Q6 lock)."""
    from dispersive_readout.optimization.sensitivity import (
        SENSITIVITY_FD_STEP,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
        SENSITIVITY_WARNING_THRESHOLD,
    )
    assert SENSITIVITY_FD_STEP == 0.05, (
        f"SENSITIVITY_FD_STEP changed from spec-locked 0.05 to {SENSITIVITY_FD_STEP}; "
        "requires spec amendment"
    )
    assert SENSITIVITY_RENDER_BAR_THRESHOLD == 0.03, (
        f"SENSITIVITY_RENDER_BAR_THRESHOLD changed from spec-locked 0.03 "
        f"to {SENSITIVITY_RENDER_BAR_THRESHOLD}; requires spec amendment"
    )
    assert SENSITIVITY_WARNING_THRESHOLD == 2.0, (
        f"SENSITIVITY_WARNING_THRESHOLD changed from spec-locked 2.0 "
        f"to {SENSITIVITY_WARNING_THRESHOLD}; requires spec amendment"
    )
