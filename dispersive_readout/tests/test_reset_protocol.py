"""Module 5b — direct-jump joint transition-readout active reset tests."""
from __future__ import annotations

import numpy as np
import pytest

from dispersive_readout.control.reset_protocol import QubitStateHistory


def test_qubit_state_history_validates_monotonicity():
    """t_start values must be strictly monotonically increasing."""
    # Valid: monotonic
    QubitStateHistory(segments=((0.0, 1), (5e-7, 0)), t_total=1e-6)

    # Invalid: t_start values equal
    with pytest.raises(ValueError, match="monotonic"):
        QubitStateHistory(segments=((0.0, 1), (0.0, 0)), t_total=1e-6)

    # Invalid: t_start values decreasing
    with pytest.raises(ValueError, match="monotonic"):
        QubitStateHistory(segments=((0.0, 1), (5e-7, 0), (3e-7, 1)), t_total=1e-6)


def test_qubit_state_history_rejects_nonzero_start():
    """First segment must start at t=0.0."""
    with pytest.raises(ValueError, match="must start at 0"):
        QubitStateHistory(segments=((1e-9, 0),), t_total=1e-6)


def test_qubit_state_history_rejects_segment_past_t_total():
    """All t_start values must be < t_total."""
    with pytest.raises(ValueError, match="t_total"):
        QubitStateHistory(segments=((0.0, 1), (2e-6, 0)), t_total=1e-6)


def test_qubit_state_history_rejects_invalid_qubit_state():
    """qubit_state ∈ {0, 1} (v0 has no thermal; no |2⟩+ states)."""
    with pytest.raises(ValueError, match="qubit_state"):
        QubitStateHistory(segments=((0.0, 2),), t_total=1e-6)


def test_qubit_state_history_frozen():
    """Mutation should raise FrozenInstanceError."""
    h = QubitStateHistory(segments=((0.0, 1),), t_total=1e-6)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        h.t_total = 2e-6  # type: ignore
