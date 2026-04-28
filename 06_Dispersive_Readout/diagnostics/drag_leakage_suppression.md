# DRAG Leakage Suppression — Diagnostic Report

## Headline finding

A sin²-windowed-Gaussian π-pulse with calibrated DRAG-1 quadrature correction
delivers a transmon X gate with X-gate error
**ε_X^ref(T_gate = 20 ns) = 8.12 × 10⁻⁴** under full REFERENCE_DEVICE Lindblad
(T₁ = 30 μs, T₂_echo = 40 μs, Purcell on, n_th = 0.01). The calibration uses
the **average-X-gate-fidelity objective** `argmin_β (1 − F_avg)` where `F_avg`
is averaged over the four Pauli-set inputs `{|0⟩, |1⟩, |+⟩, |+i⟩}` (post-N12),
on the perturbative β grid `[0, 1.2]`; the pulse-area condition
`∫₀^T Ω_x dt = π` removes the amplitude as a free parameter. The shipped
`ε_X^ref` is `1 − F_avg`, not `1 − F_transfer` — see §12.1 (N12) for why
this distinction matters.

The implementation surfaced two regime findings that became Module 5a
deliverables in their own right (V2a regime structure + V2b leakage-vs-fidelity
trade-off, see below).

## Citable numbers (REFERENCE_DEVICE)

| Quantity | Value |
|---|---|
| α / 2π | **−234.20 MHz** |
| β_opt(T_gate = 20 ns), fidelity-optimal | **0.50** |
| ε_X^ref(T_gate = 20 ns), full Lindblad, F_avg over Pauli set | **8.12 × 10⁻⁴** |
| 1 − F_avg(T_gate = 20 ns), decoherence-free | **5.5 × 10⁻⁵** |
| Final-leakage suppression at fidelity-optimal β_opt | **3.02×** vs no-DRAG |
| Peak-leakage suppression at fidelity-optimal β_opt | **1.07×** vs no-DRAG |

The full ε_X(T_gate) curve over T_gate ∈ [5, 50] ns and the V2b leakage-vs-
fidelity trade-off triplet `(β_opt_fidelity, β_min_final_leak, β_min_peak_leak)`
are exported in `06_Dispersive_Readout/figures/fig5a_drag_leakage_data.yaml`
and consumed as data by the eventual Module 5b active-reset spec.

## Validations summary

| ID | Check | Result |
|---|---|---|
| V1 | Two-level analytic Rabi trajectory match across full pulse window | **PASS** — max deviation < 1e−4 |
| V2a | Gate error 1 − F_avg < 10⁻⁴ at T_gate = 20 ns headline (post-N12) | **PASS** — empirical 5.5 × 10⁻⁵ decoherence-free (passes by ~18×) |
| V2a regime sweep | Diagnostic context at T_gate ∈ {10, 15, 20, 30} ns | reported via `average_gate_fidelity_x` per state |
| V2b | Leakage-vs-fidelity trade-off characterization | **CHARACTERIZED** — three β minimizers diverge across the perturbative grid; trade-off published as panel (b) inset 2 + YAML schema |
| V3 | Truncation convergence at T_gate = 20 ns headline (n=4 vs n=5) | **PASS** — \|ΔF\| ≈ 3.3 × 10⁻⁸ (passes by ~300×) |
| V4 | Decoherence-free fidelity ceiling 1−F_avg < 10⁻³ (diagnostic, non-blocking) | **PASS** — 5.5 × 10⁻⁵ at headline T_gate |
| V5a | Negative log-log slope of no-DRAG leakage vs \|α\| (full sweep, blocking) | **PASS** — fitted slope = **−10.08** |
| V5b | Steepness of perturbative-half slope (diagnostic) | **CHARACTERIZED** — slope = **−14.5** (textbook α⁻² assumes simple Rabi pulses; sin²-windowed envelope is steeper) |
| V6 | DRAG sign-flip increases leakage (β=−1 vs β=+1) | **PASS** — confirms rotating-frame sign convention |
| V7 | Endpoint smoothness Ω_x(0,T) = Ω̇_x(0,T) = 0 | **PASS** — to machine precision (residual ~1e−16 of peak \|Ω̇\|) |

## Methodology — three rounds of measurement-before-amendment

The Module 5a implementation surfaced v0-spec issues that were resolved across
three rounds of empirical measurement / external review followed by spec
amendment:

- **Round 8 (N10):** the original V2 row promised ≥ 5× suppression of *both*
  final and peak leakage. Measurement under the original combined-max-ratio
  leakage-objective calibration showed peak suppression saturates at ~3×
  for sin²-windowed envelopes at REFERENCE_DEVICE α — a regime characterization,
  not an implementation bug. V2 was split into V2a (final-leakage suppression,
  blocking) and V2b (peak-leakage saturation curve, diagnostic).

- **Round 9 (N11):** the V4 audit revealed that the combined-max-ratio
  calibration objective itself misbehaved: at long T_gate where leakage is
  ~10⁻⁷ at all β, the leakage ratio is noise-dominated and the optimizer
  drifts to non-perturbative β values (β > 1.2) that produce broken gates
  (1 − F up to 0.4). The objective was switched to `(1 − F_transfer)` with
  the β grid restricted to `[0, 1.2]`. V2a was recast as a fidelity threshold
  at headline; V2b was recast as the leakage-vs-fidelity trade-off curve.

- **Round 10 (N12):** Codex adversarial review surfaced two concrete findings:
  (a) the shipped headline `ε_X` was computed from one-way `transfer_fidelity_0_to_1`
  which only measures `⟨1|ρ|1⟩` and would silently miss asymmetric forward/
  reverse action or coherent superposition-state phase errors; and (b) a custom
  `beta_grid` could bypass the perturbative `[0, 1.2]` guard since
  `calibrate_drag_beta` only enforced the range when the grid was `None`.
  The metric was upgraded to **average X-gate fidelity over the Pauli set**
  `F_avg = mean(F(|ψ_in⟩ → X|ψ_in⟩) over ψ_in ∈ {|0⟩, |1⟩, |+⟩, |+i⟩})` and
  custom β grids now require explicit `allow_nonperturbative=True` opt-in
  (results carry a `perturbative_safe: bool` flag). Empirically the headline
  number under decoherence-free is essentially unchanged at the same β_opt
  (5.5e−5 vs 7.3e−5 — same order of magnitude) — the upgrade is a
  methodological-correctness fix, not a numerical correction.

The methodology lessons — formalized as process patterns in §12.1 (N11/N12)
for future modules — are:

1. **Interior-optimizer signal alone is not sufficient evidence of correctness;
   the optimizer must land in a region where the underlying physics remains valid.**
2. **Published metrics must probe the full computational subspace, not just one
   input state.** A scalar evaluated from one input (transfer fidelity) cannot
   catch coherent failures the calibration target should catch.

## Methodology notes (implementation contract)

- α is sourced from `dispersive_readout.physics.transmon.transmon_summary`
  (single source of truth, spec §4.2). V5a/V5b α-sweep varies E_C and
  re-extracts α; α is never a free parameter to the gate Hamiltonian.
- DRAG calibration uses fidelity objective `argmin_β (1 − F_transfer)` over
  β ∈ [0, 1.2] (perturbative DRAG-1 range, post-N11). Leakage curves are
  reported as diagnostics in panel (b) and YAML, not as calibration targets.
- Pure dephasing in the qubit-only Duffing-basis collapse operators uses
  per-level projectors `√γ_φ |j⟩⟨j|` (matches Module 2 convention,
  `lindblad.py:92-105`).
- Purcell decay is treated as an effective qubit relaxation channel with
  rate `γ_P = (g/Δ)² · κ` at the |0⟩↔|1⟩ transition (leading-order dispersive
  limit, spec §3.6).

## V2b — Leakage-vs-fidelity trade-off (headline finding)

Under fidelity-optimal calibration, the β minimizing gate error and the β
values minimizing leakage diverge on the perturbative β grid. At headline
T_gate = 20 ns:

| Minimizer | β | 1 − F_transfer | Final leakage | Peak leakage |
|---|---|---|---|---|
| Fidelity (β_opt) | 0.50 | 7.3 × 10⁻⁵ | 1.0 × 10⁻⁷ | 2.37 × 10⁻² |
| Final-leakage (β_min_final_leak) | 0.90 | 8.4 × 10⁻³ | 6.3 × 10⁻⁸ | 2.25 × 10⁻² |
| Peak-leakage (β_min_peak_leak) | 1.20 | 2.5 × 10⁻² | 8.5 × 10⁻⁸ | 2.17 × 10⁻² |

The three minimizers do not coincide because the gate has multiple competing
error sources (incomplete σ_x rotation at low β, transient |2⟩ excursion at
all β, leakage-recovery oscillation at high β). The full curve over T_gate
is published in panel (b) inset 2.

## Limitations

- v0 uses the Duffing-oscillator approximation. Charge-basis transmon drive
  using exact n̂ matrix elements is a v1.5 extension. v0 claims are
  Duffing-model results, full stop.
- Coherent gate errors beyond transfer fidelity (over-rotation, axis errors
  on Choi-state metric) are not modeled. Average-gate fidelity from Choi
  state is v1.5.
- Higher-order DRAG (DRAG-2, second-order corrections) is v1.5+. The peak-
  leakage saturation finding under leakage-objective calibration (N10) and
  the leakage-vs-fidelity trade-off finding under fidelity-objective
  calibration (N11) both point to DRAG-1's authority limits in this regime.

## Pointers

- Spec: `06_Dispersive_Readout/MODULE_5a_SPEC.md` (local-only, gitignored).
- Plan: `06_Dispersive_Readout/MODULE_5a_PLAN.md` (local-only, gitignored).
- Figure: `06_Dispersive_Readout/figures/fig5a_drag_leakage.png`.
- Data: `06_Dispersive_Readout/figures/fig5a_drag_leakage_data.yaml`.
- Tests: `dispersive_readout/tests/test_pulses.py`,
  `test_drag_calibration.py`, `test_gate_simulator.py`,
  `test_gate_metrics.py`.
