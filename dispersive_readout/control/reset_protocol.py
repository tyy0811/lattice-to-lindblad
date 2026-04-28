"""Module 5b — direct-jump joint transition-readout active reset.

Direct-jump v0 reset model: samples T₁/Purcell jump times exponentially,
analytically integrates the dispersive cavity equation of motion conditioned
on the resulting piecewise qubit-state history (via
dispersive_readout.physics.pointer_response), adds Module-1-consistent
Gaussian IQ noise, classifies via classify_iq (Module 1's perpendicular-
bisector discriminator), and produces a JointMatrix(P(s_f, m | s_i)) that
the closed-form reset_residual_single_cycle formula consumes.

v0 explicitly excludes mcsolve — a v1.5 extension may add mcsolve-based
jump-history sampling for richer non-Markovian effects, but cavity response
would still flow through pointer_response. See test_no_mcsolve_in_reset_
protocol for the lint-grade enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QubitStateHistory:
    """Piecewise-constant qubit-state history over [0, t_total].

    segments: tuple of (t_start, qubit_state) pairs. Validated at
    construction:
      - segments[0][0] == 0.0 (first segment starts at 0)
      - t_start values strictly monotonically increasing
      - all t_start < t_total
      - qubit_state ∈ {0, 1} (v0 has no thermal/leakage states)

    The last segment runs from its t_start to t_total. v0 has at most two
    segments (initial state + optional one jump); the dataclass shape
    extends naturally to multi-jump for v1.5 mcsolve sampling.
    """
    segments: tuple[tuple[float, int], ...]
    t_total: float

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must be non-empty")
        if self.segments[0][0] != 0.0:
            raise ValueError(
                f"first segment must start at 0.0 (got {self.segments[0][0]})"
            )
        for i, (t_start, q) in enumerate(self.segments):
            if q not in (0, 1):
                raise ValueError(
                    f"qubit_state ∈ {{0, 1}} required (got {q} at segment {i})"
                )
            if t_start >= self.t_total:
                raise ValueError(
                    f"segment {i} t_start={t_start} exceeds t_total={self.t_total}"
                )
            if i > 0 and t_start <= self.segments[i - 1][0]:
                raise ValueError(
                    f"segments must be strictly monotonic in t_start "
                    f"(segment {i} t_start={t_start} <= segment {i-1} "
                    f"t_start={self.segments[i-1][0]})"
                )
