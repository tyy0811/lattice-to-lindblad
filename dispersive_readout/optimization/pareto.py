"""Pareto frontier — skeleton committed in Task 11 (Modal smoke); full
ParetoPoint schema + build_variant land in Task 12; SLSQP find_pareto_point
in Task 13; Modal-dispatched compute_pareto_frontier in Task 14.

The placeholder ParetoPoint schema and find_pareto_point stub below exist
solely so the Day-11 Modal smoke (test O10) can confirm credentials,
image build, and round-trip serialization without depending on the
fuller implementation arriving first.
"""
from __future__ import annotations

from pydantic import BaseModel


class ParetoPoint(BaseModel):
    """Placeholder schema — full definition in Task 12."""
    device_id: str
    tau_max: float
    epsilon_0_opt: float
    tau_opt: float
    F_assign_opt: float
    F_assign_uncertainty: float
    dominant_loss_channel: str
    solver_converged: bool


def find_pareto_point(device, tau_max: float) -> ParetoPoint:
    """Placeholder implementation so O10 smoke succeeds. Task 13 replaces
    the body with SLSQP + 5×5 warm-start over (epsilon_0, tau).

    Returns a deterministic ParetoPoint that round-trips through Modal's
    serializer; the F_assign_opt = 0.5 + dominant_loss_channel = 'placeholder'
    + solver_converged = False values are intentionally diagnostic of the
    stub state — any caller that sees these in committed artifacts has
    a wiring bug.
    """
    return ParetoPoint(
        device_id="placeholder",
        tau_max=float(tau_max),
        epsilon_0_opt=0.0,
        tau_opt=float(tau_max),
        F_assign_opt=0.5,
        F_assign_uncertainty=0.01,
        dominant_loss_channel="placeholder",
        solver_converged=False,
    )
