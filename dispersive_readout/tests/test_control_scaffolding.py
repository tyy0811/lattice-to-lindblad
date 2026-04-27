"""Smoke test that the control submodule imports cleanly."""
from __future__ import annotations


def test_control_submodule_imports():
    import dispersive_readout.control  # noqa: F401
