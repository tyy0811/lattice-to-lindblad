"""Module 5b — JointMatrix dataclass for joint transition-readout output.

A JointMatrix is the physics-tier data structure produced by
extract_joint_matrix and consumed by reset_residual_single_cycle and
analysis/reset_metrics. It encodes P(s_f, m | s_i) for s_i, s_f ∈ {0, 1}
and m ∈ {0, 1}, with rows over (s_f, m) summing to 1 for each s_i.

Frozen by convention: the operating-point cache layer keys joint matrices
by operating point, and downstream code must not mutate cached instances.
Mutation raises FrozenInstanceError at write-time rather than producing
a Heisenbug across cycles.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointMatrix:
    """P(s_f, m | s_i) for s_i, s_f ∈ {0,1}, m ∈ {0,1}.

    Attributes
    ----------
    probabilities : dict[tuple[int, int, int], float]
        Keys are (s_i, s_f, m) triples. Values in [0, 1]. Rows over
        (s_f, m) sum to 1 for each s_i.
    binomial_se : dict[tuple[int, int, int], float]
        Per-entry binomial standard error √(p(1-p)/N). Same key shape
        as probabilities.
    n_trajectories : int
        N used in the underlying Monte Carlo extraction.
    operating_point : dict
        Diagnostic metadata identifying the (κ, χ, ε_drive, τ_meas, ...)
        tuple this JointMatrix was extracted at. Keys per §8 of
        MODULE_5b_SPEC.md: tau_meas, kappa, eps_drive, delta_drive,
        chi_g, chi_e, gamma_1, gamma_purcell, gamma_eff.
    """
    probabilities: dict[tuple[int, int, int], float]
    binomial_se: dict[tuple[int, int, int], float]
    n_trajectories: int
    operating_point: dict

    def marginal_confusion_matrix(self) -> dict[tuple[int, int], float]:
        """Sum over s_f to recover the plain confusion matrix P(m | s_i).

        The plain confusion matrix is what Module 1 reports; conflates
        qubit-stayed-excited-and-missed (reset failure) with qubit-
        decayed-and-correctly-classified (reset success). Used in the
        diagnostic markdown to show the joint matrix's added structure.
        """
        confusion: dict[tuple[int, int], float] = {}
        for s_i in (0, 1):
            for m in (0, 1):
                confusion[(s_i, m)] = sum(
                    self.probabilities[(s_i, s_f, m)] for s_f in (0, 1)
                )
        return confusion

    def joint_ideal_gate_floor(self) -> float:
        """The two-term ideal-gate (ε_X = 0) reset residual at p_e = 1.

        p_e' |_{ε_X=0} = P(s_f=e, m=0 | e) + P(s_f=g, m=1 | e)

        Both terms are non-negligible when T₁-during-measurement is
        active. The missed-excited term is reset failure (no flip
        applied); the false-positive-on-decayed term is unnecessary
        flip mapping g→e. The second term does NOT vanish at ε_X = 0;
        it is maximal there.
        """
        return (
            self.probabilities[(1, 1, 0)]   # P(s_f=e, m=0 | s_i=e)
            + self.probabilities[(1, 0, 1)]  # P(s_f=g, m=1 | s_i=e)
        )
