"""Sensitivity-analysis policy constants and (later) compute functions.

Policy constants (Q1, Q4, Q6 locks) are defined here — not in figure scripts —
so they are auditable, test-targeted, and version-controlled alongside the
numbers they gate.
"""
from __future__ import annotations


# Central finite-difference fractional perturbation.
# Rationale: large enough to beat simulator numerical noise; small enough
# that higher-order FD error remains <1% (confirmed by O2 step-independence).
SENSITIVITY_FD_STEP: float = 0.05

# Below this, render sensitivity as point-with-errorbar (not filled bar).
# Rationale (Q6/β): 10× below the spec's 0.3 dominance threshold; deterministic
# across runs (avoids filled-bar flicker between 0.025 and 0.035 replicates).
SENSITIVITY_RENDER_BAR_THRESHOLD: float = 0.03

# Above this, emit a boundary-proximity warning in RecommendationReport.
# Rationale (Q4): signals devices where linearized sensitivity is locally
# unreliable — regime-change boundary (Purcell, dispersive breakdown) is near.
SENSITIVITY_WARNING_THRESHOLD: float = 2.0
