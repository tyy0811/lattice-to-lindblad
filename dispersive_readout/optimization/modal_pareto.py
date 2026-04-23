"""Modal-parallelized Pareto per-point dispatch.

Public module (not `_modal_pareto`): parallelism boundary is a first-class
architectural surface per Q7. The Day-11 afternoon pre-warm task (Q2 lock)
builds the image, verifies credentials, and runs one smoke dispatch via
test O10 — surfaces infra rot on Day 11 (not Day 12 morning when Pareto
needs to run).

The actual Pareto dispatch lands in Task 14 (after Tasks 12-13 fill in
ParetoPoint schema, build_variant, and find_pareto_point implementation).
"""
from __future__ import annotations

import modal


# Extends Module 3's image with qutip + scipy so the inner find_pareto_point
# call can run Lindblad-solver + SLSQP on the Modal worker.
stage_06_module4_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3.0",
        "scipy>=1.11,<2.0",
        "qutip>=5.0,<6.0",
        "pydantic>=2.0,<3.0",
        "pyyaml>=6.0,<7.0",
    )
    .add_local_python_source("dispersive_readout")
)


app = modal.App("stage06-module4-pareto", image=stage_06_module4_image)


@app.function(cpu=2.0, memory=4096)
def pareto_one_tuple(device, tau_max: float):
    """Single-tuple Pareto-point computation.

    Pure function: no global state, no filesystem side effects. Receives
    `device: DeviceConfig` and `tau_max: float`; returns a `ParetoPoint`.
    Delegates to `dispersive_readout.optimization.pareto.find_pareto_point`,
    which currently ships as a placeholder stub (Task 11) and lands a real
    SLSQP implementation in Task 13.
    """
    from .pareto import find_pareto_point
    return find_pareto_point(device, tau_max)
