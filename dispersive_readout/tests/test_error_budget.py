"""Module 2 tests — see MODULE_2_SPEC.md §6 for the test plan."""
from __future__ import annotations


def test_module2_package_imports_without_error():
    """Smoke test: the analysis subpackage can be imported. Populated further
    as Tasks 4–8 add real API."""
    import dispersive_readout.analysis  # noqa: F401
    import dispersive_readout.analysis.operating_point  # noqa: F401
    import dispersive_readout.analysis.purcell_isolation  # noqa: F401
    import dispersive_readout.analysis.error_budget  # noqa: F401
