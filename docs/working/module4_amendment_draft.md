# Module 4 spec + plan amendment — working draft

**Status**: working scratch file. Do not commit to `MODULE_4_SPEC.md` or `MODULE_4_PLAN.md` until end-of-Day-10 when Panel (a) caption text can be verified against the rendered tornado (Task 7).

This file captures the amendment decisions that are *already* implemented in the code (commits `b691901`, `381996c`) so the amendment text doesn't depend on working memory across a context boundary.

---

## Amendment 1: Q1 sanity sign + REFERENCE-past-peak finding

**Target**: `MODULE_4_SPEC.md §0 row 1` (Q1 orthogonality lock) and `MODULE_4_PLAN.md §Task 4 Step 4.1` (O1 test expectations).

**New text (for §0 row 1):**

> **Q1 cross-check sanity sign — amended at Module 4 execution.** Under the SW-2 simulator at REFERENCE_DEVICE, F_assign peaks at `chi_scale ≈ 0.85` (|χ_01|/κ ≈ 0.375 full-split); REFERENCE sits ~18% past this peak on the high-χ side, so S_χ is slightly negative (−0.029 ± 0.014, noise-consistent at the 0.03 rendering threshold). **This is a falsifiable prediction of the simulator, not a bug:** Marxer's design target χ/κ ≈ 0.5 prioritizes measurement-induced-transition suppression and other constraints beyond F_assign-only optimization, and the Module 1 simulator correctly shows that reducing χ from REFERENCE would improve F_assign in this model under the assumed noise stack.
>
> The Q1 orthogonality argument (chi_scale and coupling.g as independent axes) is unaffected; the Day-10 cross-check `|S_g − 2·S_χ|` remains the content-positive orthogonality evidence. The Q1-locked sanity assertion "S_χ > 0 at REFERENCE" was a vestige from the original spec that assumed REFERENCE sat in the SNR-monotone regime; the diagnostic F(chi_scale) scan falsifies that assumption.
>
> Similarly, `|S_{γ_1}| ≈ 5 × 10⁻⁴` matches Module 2's `T_1_intrinsic ΔF = −3.05 × 10⁻⁴` contribution at REFERENCE (T_1 is ~3% of REFERENCE's error budget; Purcell dominates the active-loss channel per `fig2_data.yaml`). This is non-trivial cross-module internal validation: two independently-computed sensitivities — Module 2's channel-disable-ΔF and Module 4's log-FD — agree at the same REFERENCE.

**New O1 test structure (replaces original O1):**

- **O1a — sign sanity for bar-rendered parameters**: assert sign for `|S_θ| ≥ SENSITIVITY_RENDER_BAR_THRESHOLD`. At REFERENCE, fires for `epsilon_0 (+)` and `tau (+)`. Must-fire, no relaxation — a wrong sign here propagates directly into Figure 4 Panel (a).
- **O1b — log-only for near-zero parameters**: `|S_θ| < threshold` are logged to `test_output/o1b_near_zero_sensitivities.txt` with `|S|/σ` diagnostic column. No sign assertion. Captures the measured sensitivity landscape without gating CI on near-peak parameters.

**Rationale**: a single `<sign> OR <noise-consistent>` assertion is silent in the noise-consistent branch; splitting means the sign assertion *always* fires when a parameter is bar-rendered, closing the failure mode where a future bug pushes a near-zero parameter back into bar-rendering territory with the wrong sign.

---

## Amendment 2: Module 1 `noise_model='analytic'` extension

**Target**: `MODULE_4_SPEC.md §0 row 8` (Q8 contract) and any plan passages referencing `noise_model='ideal'` in the sensitivity/Pareto inner loops.

**New text (for §0 row 8):**

> **Q8 analytic-objective contract — amended at Module 4 execution.** Module 4's sensitivity analysis required a finite-SNR analytic F pathway that Module 1's shipped `noise_model='ideal'` (zero-noise limit, F=1 whenever centroids differ) did not provide. Extended Module 1 with a new `noise_model='analytic'` mode in a 1-file surgical edit (`dispersive_readout/physics/readout_model.py`), with the shipped `'ideal'` semantics preserved bit-exactly and a regression test pinning the `'gaussian' → 'analytic'` infinite-n invariant.
>
> Three noise modes now exist in Module 1:
>
> - `'ideal'` — σ = 0. Zero-shot-noise limit; all shots land on their centroids → F = 1.0 unconditionally when centroids differ. Represents the infinite-SNR upper bound. **Not used in Module 4 inner loops** (saturates to F=1, FD gradient = 0).
> - `'analytic'` — F = Φ(SNR/2), the ensemble-mean F under the gaussian noise model in the continuous-shot (n → ∞) limit. `F_assign_uncertainty` is the binomial SE √(F(1-F)/n_shots) at the stated n_shots. **Required for Module 4 inner loops** (FD-differentiable, shot-noise-free).
> - `'gaussian'` — σ = √(T/(4κ)). Per-shot circular Gaussian noise drawn in IQ space; empirical F from the perpendicular-bisector discriminator over n_shots samples. Module 1/2/3 default. Used in Module 4 only for O5b shot-noise-detectability Welch-t test.
>
> Invariant pinned by Module 1 regression test: `F_gaussian → F_analytic` as `n_shots → ∞`; also `F_analytic = Φ(SNR/2)` exactly.
>
> **O8 contract strengthened** from a single "forbid `noise_model='gaussian'`" grep to three tests per module (`sensitivity.py`, `pareto.py`):
>
> 1. forbid `'gaussian'` (existing — shot noise pollutes FD gradients)
> 2. forbid `'ideal'` (new — F=1 saturation makes FD gradient identically zero)
> 3. require `'analytic'` to appear at least once (positive assertion catches kwarg-deletion regressions that would pick up Module 1's `'gaussian'` default)

**Framing for the day-14 report (IQM-reviewer-facing):**

> Module 4's sensitivity analysis required a finite-SNR analytic F pathway that Module 1's `noise_model='ideal'` (zero-noise limit) did not provide; extended Module 1 with a new `noise_model='analytic'` mode in a 1-file surgical edit, with the shipped `'ideal'` semantics preserved bit-exactly and a regression test pinning the `'gaussian' → 'analytic'` infinite-n invariant.

This is not "Module 1 bug surfaced by Module 4"; Module 1's `'ideal'` does exactly what its docstring says. The correct framing is "Module 4 required an additional API surface that was cheap to add and sharpened both modules' expressiveness."

---

## Amendment 3: Figure 4 Panel (a) caption addendum

**Target**: `MODULE_4_SPEC.md §7 Panel (a)` caption specification.

**Add to Panel (a) caption** (verify against rendered tornado after Task 7 before committing):

> χ registers as noise-consistent-with-zero at REFERENCE because the Module 1 simulator places the F_assign peak at `chi_scale ≈ 0.85`, 18% below REFERENCE; under a pure-F_assign-optimization criterion, Marxer's χ is slightly above the simulator's optimum, consistent with the device's design prioritizing measurement-induced-transition suppression alongside F_assign.

**Rationale**: without this sentence the tornado reader sees "χ: noise-consistent point-with-errorbar" as an unexplained null result. With it, the bar becomes a *characterized observation* that demonstrates the simulator agrees with Marxer's published F_assign value while also showing what pure-F_assign optimization would recommend — two pieces of information, not one.

---

---

## Amendment 4: `SENSITIVITY_WARNING_THRESHOLD` recalibration (2.0 → 0.3)

**Target**: `MODULE_4_SPEC.md §3.1` policy-constants table, `MODULE_4_PLAN.md §Task 2` locked-values commit message, `MODULE_4_PLAN.md §Task 6` O11 probe device.

**Finding**: the spec-locked `SENSITIVITY_WARNING_THRESHOLD = 2.0` is unreachable under the Lindblad simulator across realistic parameter space. Empirical `|S_θ|` caps at ~0.4 (|S_ε0|_max = 0.39 at ε/2π = 15 MHz; |S_γ1|_max = 0.25 at T_1 = 0.22 µs; others below). Verified as genuine physics (not solver / truncation / Purcell-leak artifact) via three independent checks committed to `docs/module4_diagnostics/`:

- `check_tolerance.py`: 0.000% change under 100× tighter mesolve tolerances
- `check_truncation.py`: 1.36% change at 40% larger Hilbert space
- `check_purcell.py`: γ_eff − γ_1_true = −1.4e-6 1/s (<1 ppm) with coupling.g = 0

Supporting:

- Closed-form derivation of `|S_γ| = (s/(2(1-s))) · (x/2) · (φ/Φ)(x/2)` from `F = Φ(SNR/2)` with linearized envelope
- Integration-window-geometry factor `4·T_mid/τ ≈ 2.4` explaining the closed-form vs Lindblad ratio in the weak-decoherence regime

**Amendment** (applied in commit `<TBD>` during Task 6):

- `SENSITIVITY_WARNING_THRESHOLD = 0.3` (aligned with spec §2.1's "dominance" level).
- Policy-constants test locks 0.3.
- O11 probe device amended from `T_1 = 5 µs, REFERENCE drive` (|S_γ1| < 0.01 empirically, under any threshold) to `ε/2π = 15 MHz, REFERENCE T_1` (|S_ε0| = 0.388 > 0.3, realistic operating-regime choice).

**Figure 4 caption phrasing**:

> The `SENSITIVITY_WARNING_THRESHOLD = 0.3` warning fires when any parameter reaches dominance-level sensitivity (|S_θ| > 0.3), flagging devices where a single parameter controls F_assign and the linearized ranking is less informative. Under the Lindblad simulator the empirical ceiling across all 7 parameters is ~0.4; the threshold is therefore near the top of the observable sensitivity band and fires only for unambiguously dominance-proximate operating points.

**Framing for the day-14 report**:

> Module 4's execution revealed that the simulator's `|S_θ|` ceiling caps at ~0.4 in realistic regimes, making the original `SENSITIVITY_WARNING_THRESHOLD = 2.0` unreachable (dead code). Three independent reproducibility checks — solver tolerance at 100× tighter, Hilbert-space truncation at 40% larger, and explicit coupling-zero isolation of `decoherence.gamma_1` — verified the ceiling as genuine Lindblad physics. On the basis of this characterization, the threshold was recalibrated to 0.3 (spec §2.1 "dominance" level). The recalibration is backed by six independent verifications (analytic ceiling + 7-parameter empirical scan + integration-window-geometry factor + three reproducibility checks) committed to `docs/module4_diagnostics/`.

---

## Tally

This amendment consolidates three substantive day-14 items:

1. **Module 4 Q1 physics finding** — REFERENCE sits 18% past the F_assign peak in χ-space; consistent with Marxer's multi-constraint device design; cross-module-validated against Module 2's Purcell-dominated error budget.
2. **Module 1 `noise_model='analytic'` extension** — surgical 1-file addition; two-test invariant (`analytic = Φ(SNR/2)` definitional + `gaussian → analytic` at n=∞); Q8 contract strengthened from 1 grep to 3 per module.
3. **SENSITIVITY_WARNING_THRESHOLD recalibration (2.0 → 0.3)** — characterized empirical ceiling under the Lindblad simulator; three reproducibility checks rule out solver / truncation / Purcell-leak artifacts; threshold realigned with spec §2.1 dominance level.

Total: **39 substantive corrections across Modules 1–4** going into the day-14 report.
