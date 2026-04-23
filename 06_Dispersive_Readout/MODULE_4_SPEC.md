# Stage 06 Module 4 — Sensitivity and Pareto Optimization

**Status:** amended design, 2026-04-21. Supersedes the original Module 4 draft pasted into the brainstorming session on the same date.

**Goal.** Given the Module 1 simulator and Module 3 characterization pipeline, compute which device and control parameters dominate readout fidelity (sensitivity analysis), map the speed-fidelity landscape analytically with published-device overlays (regime map), trace the speed-fidelity Pareto frontier for three parameter-anchored device variants (Pareto frontier), and close the loop from fitted parameters → recommended readout configuration (closed-loop demonstration). Ship as a single composite Figure 4 plus a YAML recommendation artifact.

**Budget.** 4 working days (days 10–13 in the Implementation Plan timeline).

**Prerequisites.** Module 1 simulator complete (V1a, V2, V3, V4a/b passing); Module 2 shipped at tag `stage06-module2` with `ErrorBudget` YAML committed at `06_Dispersive_Readout/figures/fig2_data.yaml`; Module 3 shipped at tag `stage06-module3` with `recovery_coverage_report.yaml` committed. Module 1 public API (`dispersive_readout.physics`) exposes `DeviceConfig`, `DriveParams`, `DecoherenceParams`, `REFERENCE_DEVICE`, `simulate_readout`, `compute_assignment_fidelity`. Module 2 public API (`dispersive_readout.analysis`) exposes `OperatingPoint`, `ErrorBudget`, `compute_full_error_budget`. Module 3 public API (`dispersive_readout.characterization`) exposes `ExtractedParameterPack.to_device_config()`.

**Scope lock.** Physics decisions below are locked post-amendment. No expansion to simulation-based inference (SBI), Bayesian MCMC, multi-qubit extensions, adjoint-state optimal control, full JAX rewrite of the Lindblad solver, or resonator/coupling characterization beyond what Module 3 already provides. Implementation discoveries that challenge a locked decision are raised as blockers, not silently resolved.

---

## 0. Amendments applied to the original spec

Nine substantive decisions were surfaced during adversarial brainstorming (Q1–Q9, 2026-04-21) and resolved with spec amendments before implementation. The rest of this document is the *post-amendment* spec; this section records what changed and why, so the delta to the original is traceable.

| # | Amendment | Driver |
|---|---|---|
| 1 | χ-sensitivity implemented via `chi_scale: float = 1.0` kwarg on `build_hamiltonian` (`dispersive_readout/physics/lindblad.py:191`, one-line multiplicative rescale of `chi_per_level` array) and threaded through `simulate_readout`. Not via perturbing `coupling.g`. A Day-10 cross-check computes `S_g` via ±5% perturbation of `g` (re-deriving γ_Purcell) and compares to `2 · S_χ`; the numerical `abs(S_g − 2·S_χ)` is logged into the Figure 4 caption. | Q1 — Orthogonality: perturbing `g` simultaneously moves `γ_Purcell` (which is part of total `γ_1`), so `S_χ`-via-g-bumping silently carries a Purcell component that also appears in `S_gamma_1`. The two tornado axes would overlap, destroying the "independent lever" interpretation. Standard sensitivity-analysis practice treats effective parameters as independent axes via partial derivatives; `chi_scale` makes that explicit in the Hamiltonian. |
| 2 | Pareto frontier parallelized via Modal `pareto_one_tuple.map(...)` reusing Module 3's `.map()` scaffolding. Pareto compute is 3 parameter-anchored device variants × 10 τ_max values (locked, not the original spec's "3–5" range). Modal image pre-warm task on Day 11 afternoon ensures qutip+scipy deps are baked before Day 12's Pareto run. | Q2 — Iteration velocity on Day 12: serial Pareto at ~50 min wall-clock burns one of four effective working hours per attempt; Modal collapses that to ~5 min, preserving capacity for fix-cycles. Modal infrastructure is already shipped in Module 3, so this is a copy-paste-refactor rather than a new infrastructure dependency. |
| 3 | Regime map (Panel b) is a closed-form analytic surface from the dispersive-SNR formula (Bengtsson 2024 PRL §II, cross-checked against Blais RMP 2021 §V.B), not a Lindblad-simulated grid. Marxer Q1 annotated with `F_sim = {F_ref:.4f}` computed once from the Module 1 simulator at REFERENCE. A 2-point Lindblad-vs-analytic validation at (Marxer Q1, χ/κ=1·γ₁τ=0.01) runs on Day 11; the maximum observed deviation is printed in the caption. | Q3 — A `chi_scale` sweep over 100× range at REFERENCE-everything-else produces a synthetic Hamiltonian (rescaled χ while Lamb shifts and Purcell stay at original-g values) that doesn't correspond to any real device at the extrema. The published-device overlays on a simulated grid would also be apples-to-oranges: the grid is REFERENCE-derived, the markers are at foreign full-parameter sets. Closed-form analytic sidesteps both issues and matches the publishable genre of this plot (Bengtsson 2024, Sank 2024). |
| 4 | Closed-loop scope narrowed: `recommend_from_fitted_parameters` accepts fitted (T₁, T₂, ω_q) but inherits REFERENCE values for (`κ`, `g`, `ω_r`, truncation) because Module 3's four protocols don't constrain those. `RecommendationReport` adds `sensitivity_warnings: list[str]` that fires when `abs(S_θ) > 2.0` (threshold in `SENSITIVITY_WARNING_THRESHOLD` policy constant), signalling boundary-proximate devices where linearized sensitivity is locally unreliable. Demo device for the closed-loop arrow is picked on Day 13 from `recovery_coverage_report.yaml` (SEED=42 stable), selecting whichever hard-recovery-harness device shifts the Pareto optimum most visibly from REFERENCE's. | Q4 — `to_device_config` inherits REFERENCE resonator/coupling/truncation because Module 3 only fits decoherence + frequency. Claiming "closed-loop recommendation on the fitted device" without scope narrowing overclaims — reviewer asks "where's the fitted κ?" and the answer is "it wasn't fit." Picking a non-REFERENCE demo device ensures the arrow demonstrates responsiveness, not just drive re-optimization around REFERENCE. |
| 5 | Panel (b) overlays 4 markers: Marxer Q1 (★ orange, primary anchor, `F_sim` annotated), Marxer Q2 (◆ orange, same-chip fabrication-variation point), Bengtsson (● red), Garnet (□ red, grey-hatched to signal estimated coordinates). Hazra 2407.10934 is cited in the reference list as the methodological template for repeated-measurement benchmarking but **not plotted** on the regime map. Bengtsson's κ range is cited from Sank 2402.00413 §IV (companion paper, measured on same Sycamore-class hardware); Marxer parameters from p.15 of 2508.16437 (measured device table); Garnet T₁=40 µs and F_assign=0.97 from Abdurakhimov 2408.12433 pp. 9, 13, with χ/κ and τ_readout as IQM-design-family estimates (caveated). | Q5 — Hazra's only T₁ number is a Purcell-bound upper estimate (2.6 s), not a device-relaxation measurement; its dimon (two-mode Josephson) hardware is non-standard-transmon and the χ-mediation is nonlinear. Plotting on a transmon regime map would be apples-to-pears. Public source access failed for full Garnet device table (PDF binary not parseable without poppler/pypdf; direct web sources blocked); citing what is extractable with explicit caveats is honest. |
| 6 | Figure 4 applies `feedback_figure_presentation.md` rules. Panel (a): cool palette for sensitivities; `SENSITIVITY_RENDER_BAR_THRESHOLD = 0.03` policy constant forces point-with-errorbar rendering for any parameter with `abs(S_θ) < 0.03` regardless of bootstrap CI (deterministic across runs; avoids filled-bar flicker near noise floor). Panel (b): viridis colormap; greyed analytic boundaries; saturated warm markers; Marxer Q1/Q2 share warm-orange (shape-family encoding) while Bengtsson and Garnet are red (distinct-lab). Panel (c): points-on-curve + shaded envelope for Pareto (Bengtsson Fig 3 / Sank Fig 6 pattern). Cross-panel anchor `n_shots = 10⁴` lives in the figure-wide caption, not repeated in per-panel subtitles. | Q6 — External-reader clarity standard per memory entry: readers who have not seen the work must interpret each panel in 5–10 seconds. Locked threshold constants (0.03 for rendering, 2.0 for warnings) must be auditable in source code, not magic numbers in a figure script. |
| 7 | Pareto variants in Panel (c) are **parameter-anchored**, not device-anchored. Locked variants: V1 = REFERENCE (Marxer-Q1-class); V2 = REFERENCE with `decoherence.gamma_1 = 1/40µs` (Garnet-like T₁, κ at REFERENCE); V3 = REFERENCE with `decoherence.gamma_1 = 1/20µs` and `resonator.kappa = 2π·6 MHz` (Bengtsson-like T₁ and κ). Curve labels on Figure 4 name the parameter substitution, not the device. Figure-wide caption includes the foreclosing sentence: *"Curves V1–V3 represent the Pareto frontier predicted by this work's simulator under parameter substitution — not the frontier achievable on the cited devices' native hardware."* The V1↔V2 T₁-offset is reframed as a positive Panel-(a)↔Panel-(c) cross-tie. | Q7 — Plugging foreign (T₁, κ) pairs into REFERENCE's remaining parameters and running this work's SLSQP over (ε₀, τ) is **not** a prediction of what each lab achieves. "Garnet's Pareto frontier" would be an overclaim: Garnet doesn't use Marxer's resonator architecture, Bengtsson's chip uses Google's Purcell-filter topology, and the simulator doesn't know. Parameter-anchored labels make the scope of the claim literal. |
| 8 | All finite-differencing in Module 4 (Pareto SLSQP function evaluations; sensitivity ±5% FD probes) runs against `compute_assignment_fidelity(..., noise_model='ideal')`. Shot-noise sampling (`noise_model='gaussian'`) appears only at (i) the analytic binomial SE envelope on reported `F_opt` and `S_θ` uncertainties via closed-form `σ = sqrt(F(1−F)/n_shots)`, (ii) test O5b's Welch-t shot-noise detectability assertion, and (iii) verification-style single-point checks mirroring Module 2's calibration verifier. Test O8 asserts `noise_model='gaussian'` does not appear inside `optimization/pareto.py` or `optimization/sensitivity.py` call sites. This **extends** Module 2's analytic-calibration-verified-by-shot-noise pattern to a finite-difference context Module 2 did not face. | Q8 — A shot-noisy objective pollutes SLSQP's finite-difference gradient estimates; the optimizer can wander, settle at local minima, or return biased optima. `noise_model='ideal'` is already first-class on Module 1's public API (`dispersive_readout/physics/readout_model.py:152`); using it is a zero-new-code fix. Keeps finite-differencing reliable at the ±5% sensitivity perturbation scale. |
| 9 | (a) Autodiff add-on has three concrete abort signals: (i) JAX-compatible Lindblad forward pass not producing a finite F within 90 min, (ii) gradient-vs-finite-diff agreement failing at the 10% level on `S_χ` after 3 hr, (iii) any of the three baseline deliverables having unresolved bugs at the 4-hr cap. Any trigger → immediate revert. (b) Recommendation narrative template: IQM-table rounding (T₁/T₂ to integer µs, ω_q/2π to 3 decimal GHz, ε₀ to 2 sig fig MHz/2π, τ to integer ns, F to 4 decimals, ΔF to 4 decimals); σ rounded up to 1 sig fig with F's last-decimal position matched (metrology standard); asymmetric uncertainty handled if Module 3's `FittedParameter` schema exposes it. (c) 28 tests locked (29 with O7 autodiff-contingent); O5 splits into O5a (modeled improvement) / O5b (shot-noise detectability) for cleaner diagnostics; O9 (YAML regression gate), O10 (Modal-image smoke), O11 (sensitivity_warnings fires) added. | Q9 — "Contingent" without explicit abort triggers slides into "just 30 more minutes" under deadline pressure; three concrete failure modes force a real decision. Metrology-standard σ convention avoids the `0.0002 vs 0.00022` ambiguity that vaguer rounding rules produce. Split-test O5a/O5b yields cleaner failure diagnostics because the two assertions have independent failure modes. |

### 0.1 Module 4 execution-time amendments (Day 10, 2026-04-22)

Three substantive amendments were surfaced during Day-10 execution. Each is grounded in committed diagnostic artifacts under `docs/module4_diagnostics/`. The rest of this document is the *pre-Day-10-execution* spec; this section records the deltas, so the original text remains readable alongside the amended interpretation.

| # | Execution-time amendment | Driver |
|---|---|---|
| 10 | **Q1 S_χ sign expectation**: the original Q1 wrote "S_χ > 0 at REFERENCE" as a physics-falsifiable sanity sign. Execution finding: under the SW-2 simulator at REFERENCE, F_assign peaks at `chi_scale ≈ 0.85` (\|χ_01\|/κ ≈ 0.375 full-split); REFERENCE sits ~18% past this peak on the high-χ side, so S_χ is slightly negative (`−0.029 ± 0.014`, noise-consistent at the 0.03 rendering threshold). This is a **falsifiable prediction of the simulator, not a bug**: Marxer's design target χ/κ ≈ 0.5 prioritizes measurement-induced-transition suppression alongside F_assign; the Module 1 simulator correctly shows that reducing χ from REFERENCE would improve F_assign in this model. The Q1 orthogonality argument is unaffected; the Day-10 cross-check `\|S_g − 2·S_χ\| / \|2·S_χ\| = 6.4%` is the content-positive orthogonality evidence. **O1 restructured**: O1a asserts spec-predicted signs only for parameters whose `\|S_θ\|` rises above `SENSITIVITY_RENDER_BAR_THRESHOLD` (would render as filled bars on the tornado — must-fire, no relaxation); O1b logs near-zero parameters to `test_output/o1b_near_zero_sensitivities.txt` with measured values, no sign assertion. Cross-module validation: `\|S_γ1\| ≈ 5 × 10⁻⁴` matches Module 2's `T_1_intrinsic ΔF = −3.05 × 10⁻⁴` at REFERENCE (T_1 is ~3% of the error budget; Purcell dominates per `fig2_data.yaml`). | Execution finding: the committed REFERENCE's F-landscape puts the operating point past the χ-peak under this simulator, not in the SNR-monotone regime the original Q1 assumed. |
| 11 | **Module 1 `noise_model='analytic'` extension**: Module 1's shipped `noise_model='ideal'` is the zero-shot-noise limit (F=1.0 unconditionally when centroids differ), which is useless for finite-difference sensitivity analysis (log F saturates, FD gradient = 0). Module 4's sensitivity analysis required a finite-SNR analytic F pathway that `'ideal'` did not provide. Extended `compute_assignment_fidelity` with a new `noise_model='analytic'` mode (`F = Φ(SNR/2)`, ensemble-mean F under the gaussian noise model in the continuous-shot limit) in a 1-file surgical edit. Shipped `'ideal'` semantics preserved bit-exactly; two regression tests pin the invariants (`F_analytic == Φ(SNR/2)` definitionally; `F_gaussian → F_analytic` as `n_shots → ∞` within 5σ_binomial at n=2×10⁵). **Q8 contract strengthened** from a single `"forbid 'gaussian'"` grep to three tests per module (`sensitivity.py`, `pareto.py`): forbid `'gaussian'`, forbid `'ideal'`, require `'analytic'` to appear at least once (positive assertion catches kwarg-deletion regressions). All occurrences of `noise_model='ideal'` in §2.2, §3.1, §3.3 should be read as `noise_model='analytic'` per this amendment. | Execution finding: Module 4 needed a specific F pathway (finite-SNR analytic F) that Module 1 didn't expose. Additive API extension was cleaner than repurposing `'ideal'` semantics (which would have silently changed a shipped contract). |
| 12 | **`SENSITIVITY_WARNING_THRESHOLD` recalibrated 2.0 → 0.3**: the spec-locked 2.0 was unreachable under the Lindblad simulator across realistic parameter space. Empirical `\|S_θ\|` caps at ~0.4 (`\|S_ε0\|_max = 0.39` at ε/2π = 15 MHz; `\|S_γ1\|_max = 0.25` at T_1 = 0.22 µs; others below). Verified as genuine Lindblad physics (not solver / truncation / Purcell-leak artifact) via three independent reproducibility checks committed at `docs/module4_diagnostics/check_{tolerance,truncation,purcell}.py`: tolerance 0.000% change at 100× tighter; truncation 1.36% change at 40% larger Hilbert space; Purcell isolation via `coupling.g = 0` gives `γ_eff = 1/T_1` to <1 ppm. `0.3` aligns with spec §2.1's "dominance" level. **O11 probe device amended** from `T_1 = 5 µs` at REFERENCE drive (\|S_γ1\| < 0.01 empirically, under any threshold) to `ε/2π = 15 MHz` at REFERENCE T_1 (\|S_ε0\| = 0.388 > 0.3, realistic operating-regime choice). Figure 4 caption phrasing: *"fires when any parameter reaches dominance-level sensitivity (\|S_θ\| > 0.3), flagging devices where a single parameter controls F_assign and the linearized ranking is less informative."* Amended references: §0 row 4 (`SENSITIVITY_WARNING_THRESHOLD = 2.0` → 0.3); §3.1 policy-constants table (2.0 → 0.3); §0 row 6 (locked threshold constants list). | Six independent verifications: analytic ceiling derivation, empirical 7-parameter scan, integration-window geometry factor `4·T_mid/τ ≈ 2.4`, plus the three reproducibility checks. Original 2.0 was a pre-execution estimate from the linearized-envelope closed form that does not match the simulator's actual F(γ) curve. |

**Panel (a) subtitle concrete values** (measured at standalone tornado render `06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py`): `F_ref = 0.9899`, `τ_int = 450 ns` (integration window `(50 ns, 500 ns)`, not 500 ns drive duration), `n̄_phot = 2.24`, `n_shots = 10⁴`.

**Panel (a) caption addendum** (chi noise-consistent framing): *"χ registers as noise-consistent-with-zero at REFERENCE because the Module 1 simulator places the F_assign peak at `chi_scale ≈ 0.85`, 18% below REFERENCE; under a pure-F_assign-optimization criterion, Marxer's χ is slightly above the simulator's optimum, consistent with the device's design prioritizing measurement-induced-transition suppression alongside F_assign."*

**Day-14 narrative-template note for Task 16**: the original `top_3_sensitivities` renders the top-3 parameters by `\|S_θ\|` regardless of bar/point status. At REFERENCE the top-3 by `\|S\|` are `ε₀ (+0.050)`, `τ (+0.030)`, `chi_scale (−0.029)` — but under the corrected dispatcher only ε₀ is bar-rendered (τ and chi_scale sit at the threshold edge, noise-consistent). Task 16's narrative should render as "top-3 bar-rendered parameters" when not all top-3 are above the rendering threshold, so the narrative doesn't imply τ or χ is a dominant lever at REFERENCE.

### 0.2 Codex adversarial-review amendment (Day 10, 2026-04-23)

| # | Amendment | Driver |
|---|---|---|
| 13 | **τ-window mismatch in sensitivity FD dispatcher (high severity)**: `_perturbed_device_drive_scale` rescaled `drive.duration` for τ probes but kept `op.integration_window` fixed. The ±5% probes compared integrals over mismatched physical windows, silently biasing `S_τ` by ~20%. Fix: `_perturbed_device_drive_window_scale` threads a per-probe integration window through the dispatcher; for τ probes, `window[1]` co-perturbs to `drive.duration` while `window[0]` (κ-ramp-up exclusion) stays fixed. `S_τ` now measures the package-deal "longer pulse + longer integration" sensitivity consistent with the landscape `find_pareto_point` navigates. New regression test `test_fd_dispatcher_probe_configuration_self_consistent` asserts for all 7 parameters that the dispatcher returns self-consistent (device, drive, window, chi_scale) triples. **Measured Δ**: `S_τ = +0.0369 → +0.0297`, dropping τ from bar-rendered to point-with-errorbar (|S_τ|/σ = 2.08 — τ's upper-1σ cap at 0.044 still extends into bar territory, so τ remains statistically resolvable). All other sensitivities bit-identical before/after. See `docs/module4_diagnostics/tau_window_correction.md`. | Codex adversarial review. Test suite (O1a, O2, O8) did not interrogate FD-dispatcher parameter-configuration self-consistency, so the bias passed existing gates. |
| 14 | **Zero-reference multiplicative-collapse guard (code hardening)**: `compute_log_sensitivity` now raises `ValueError` with Koch-back-solve-aware guidance when `reference_value == 0.0`. Previously, multiplicative perturbation `θ·(1±h)` collapsed to 0 at both FD branches, producing silent `S = 0`. Specifically named in the error text: the γ_φ scenario where Koch back-solve with `T_2_echo = 2·T_1` gives `γ_φ = 0` exactly (a regime that V2/V3 Pareto variants per §3.3 could reach in the wild). Not a numerical correction — no parameter at REFERENCE triggers it — but closes a latent failure mode in the sensitivity engine. | Codex adversarial review finding #2 (medium severity). Defensible programming; no observed failure in current artifacts. |

**τ-at-the-edge caption phrasing**: the corrected tornado has one bar-rendered parameter (ε₀) and six point-with-errorbar. The Panel (a) caption should describe this as *"ε₀ is the dominant control parameter; τ and χ sit at the rendering-threshold edge with statistically significant support (|S|/σ ≈ 2), but neither has the central-value dominance that would warrant a filled bar."* Draft language subject to final verification against the re-rendered PNG.

---

## 1. Philosophy

Module 4 is the differentiating module. It exists to land the ML-engineering reframe — Jane computes sensitivities, maps the optimization landscape, traces Pareto frontiers, and closes the characterization-to-recommendation loop. It is also the module most at risk of scope creep: autodiff is tempting, Pareto can be overbuilt, the recommendation can balloon into a rule engine, and Figure 4 can drift into a six-panel monstrosity.

The discipline is:

1. **Three concrete deliverables, one composite figure.** Sensitivity tornado, regime map, Pareto frontier, Figure 4 — nothing else ships. A fourth deliverable is cut or pushed to post-submission.
2. **Deterministic first, autodiff second.** The baseline uses scipy.optimize + central finite differences. JAX autodiff is a 4-hour contingent extension, not a dependency for shipping.
3. **Analytic surfaces over simulation surfaces where defensible.** The regime map (§3.2) is closed-form analytic because the alternative — a Lindblad grid at fixed REFERENCE-minus-χ — produces a synthetic landscape at the corners. The publishable genre of this plot (Bengtsson 2024, Sank 2024) is analytic.
4. **Recommendation is a YAML-emitting report, not a rule engine.** Given fitted parameters from Module 3, return an optimal (ε₀, τ), the top-3 sensitivities at the per-device optimum, warnings when linearized sensitivity is locally unreliable, and a template-rendered narrative. No orchestration, no LLM in the loop.
5. **Honest scope at every surface.** The closed-loop arrow is annotated with its scope (fitted T₁/T₂/ω_q over REFERENCE resonator); the Pareto variants are parameter-anchored (not device-claimed); the regime map captions which parameters are measured vs. estimated per marker; the sensitivity tornado captions the orthogonal-axes methodology vs. fabrication-noise-response.

---

## 2. Physical scope

### 2.1 Parameters swept

Seven parameters, each treated as an independent axis for sensitivity analysis (Q1 orthogonality locked):

**Device parameters** (five, varied in sensitivity):

- **χ** (dispersive shift) — perturbed via `chi_scale: float` kwarg on `build_hamiltonian`. Multiplicative rescale of the per-level χ_j array. At the reference, `chi_scale = 1.0` reproduces the Module 1 simulator exactly.
- **κ** (resonator linewidth) — perturbed directly via `dataclasses.replace(device.resonator, kappa=...)`.
- **γ₁** (intrinsic qubit relaxation rate, Purcell excluded) — perturbed directly via `decoherence.gamma_1`.
- **γ_φ** (pure dephasing rate) — perturbed directly via `decoherence.gamma_phi`.
- **n̄_th** (thermal bath population) — perturbed directly via `decoherence.n_th`.

**Control parameters** (two, varied in both sensitivity and Pareto):

- **ε₀** (drive amplitude) — `drive.amplitude`.
- **τ** (drive duration) — `drive.duration`.

**Held fixed** (not varied in Module 4):

- E_C, E_J, E_J/E_C ratio, n_g (transmon parameters);
- drive detuning (assumed calibrated to optimum from Module 2 characterization);
- `edge_sigma` (Gaussian-edge width on `DriveParams` — held fixed in the baseline Pareto and sensitivity sweeps; varied only in the contingent autodiff add-on per §3.5);
- truncation dimensions.

Note: Module 1's pulse envelope is a Gaussian-edged square parameterized by `(amplitude, duration, edge_sigma, detuning)` — there is no independent "plateau duration" field on `DriveParams`. The plateau length is derived as `duration − 6·edge_sigma` from the erf-difference envelope (`dispersive_readout/physics/lindblad.py:239`). Consequently, the autodiff add-on's pulse-shape DoFs are `(edge_sigma)` with `duration` held at the Pareto-converged `τ_opt` — not a distinct plateau parameter. See §3.5.

### 2.2 Objective function

Primary objective: assignment fidelity `F_assign` at fixed integration window and shot count, computed analytically (no shot-noise sampling inside optimization or sensitivity loops, per Q8 lock):

```python
def objective(control_params, device_params, config) -> float:
    """F_assign (analytic) at (control_params, device_params), higher is better."""
    r0 = simulate_readout(
        device=build_device(device_params),
        drive=build_drive(control_params),
        initial_qubit_state=0,
    )
    r1 = simulate_readout(device=..., drive=..., initial_qubit_state=1)
    return compute_assignment_fidelity(
        r0, r1,
        integration_window=config.integration_window,
        n_shots=config.n_shots,
        noise_model="ideal",   # << Q8-locked: no shot noise inside optimizer
    ).F_assign
```

Secondary objective (Pareto): readout duration τ (lower is better). The Pareto surface is 2D: (τ, F_assign).

---

## 3. Methodology

### 3.1 Sensitivity analysis

**Normalized logarithmic sensitivities** at the reference operating point:

```
S_θ := ∂(ln F_assign) / ∂(ln θ)   evaluated at θ = θ_ref
```

Dimensionless by construction, directly comparable across the 7 parameters.

**Computation.** Central finite differences with step size `h = 0.05`:

```
S_θ ≈ [ln F(θ·(1+h)) − ln F(θ·(1−h))] / (2h)
```

All F evaluations use `noise_model='ideal'` (Q8). Uncertainty on S_θ is propagated from the analytic binomial SE on F_ref:

```
σ(S_θ) ≈ sqrt(2) · σ(F_ref) / (h · F_ref)
```

with `σ(F_ref) = sqrt(F_ref · (1 − F_ref) / n_shots)`.

**Day-10 cross-check (Q1).** In addition to S_χ via `chi_scale ± 0.05`, compute S_g via ±5% perturbation of `coupling.g` (which re-derives γ_Purcell through `build_collapse_operators`). Under the transmon `χ ≈ 2 g² α / (Δ(Δ+α))` at fixed (κ, α), `S_g ≈ 2 · S_χ` holds at leading order. The measured `|S_g − 2 · S_χ|` quantifies Purcell-coupling contamination in an (A)-style χ-sensitivity, and is logged into the Figure 4 caption.

**Policy constants** (in `optimization/sensitivity.py`):

```python
SENSITIVITY_FD_STEP = 0.05                 # central finite-difference fractional step
SENSITIVITY_RENDER_BAR_THRESHOLD = 0.03    # |S_θ| < this → point-with-errorbar, not bar
SENSITIVITY_WARNING_THRESHOLD = 2.0        # |S_θ| > this → sensitivity_warnings fires
```

All three constants are auditable, test-targeted, and cited in the Figure 4 caption.

**Sanity signs.** At REFERENCE:

- `S_χ > 0` (positive sensitivity: increasing χ improves SNR).
- `S_{γ_1} < 0`, `S_{γ_φ} < 0`, `S_{n̄_th} < 0` (increasing any loss channel degrades F).
- Signs falsifiable by Test O1; wrong signs mean the simulator is broken, not the sensitivity code.

### 3.2 Regime map (analytic)

**Closed-form F_assign** over (χ/κ, γ₁·τ_readout) from the dispersive-SNR formula (Bengtsson 2024 PRL §II; Blais RMP 2021 §V.B cross-check for the steady-state derivation):

```
SNR_steady(χ/κ, n̄_phot) = 4 · (χ/κ) · sqrt(n̄_phot) / (1 + (2χ/κ)²)
SNR_eff(χ/κ, γ₁τ, n̄_phot) = SNR_steady · (1 − γ₁τ/2)^(1/2)
F_analytic = Φ(SNR_eff / 2)       # Φ = standard normal CDF
```

where `n̄_phot` is the steady-state resonator photon number, held fixed at a value chosen to reproduce REFERENCE's operating point and quoted on Figure 4 Panel (b)'s subtitle.

Linear decoherence-envelope `(1 − γ₁τ/2)^(1/2)` is within 1% of the exponential form `exp(−γ₁τ/4)` across the grid's y-axis range `[1e-4, 1e-1]` — confirmed once in a unit test and annotated in the caption.

**Grid.** χ/κ axis log-spaced from 0.1 to 10 (20 points); γ₁·τ_readout axis log-spaced from 1e-4 to 1e-1 (20 points); 400 analytic evaluations (no simulator calls, sub-second total).

**Analytic boundaries** drawn on the map (grey dashed). The grid has only two degrees of freedom (χ/κ, γ₁·τ_readout); all other device parameters are held at REFERENCE values to render the boundary curves well-defined. Each boundary is parameterized as:

1. **Purcell limit** — locus where `γ_Purcell · τ_readout = 0.1`. Under the 2nd-order SW dispersive formula `γ_Purcell = κ · (g/Δ)²`, holding `(g, Δ)` at REFERENCE gives `γ_Purcell` as a function of `κ` alone. Holding χ at REFERENCE's dispersive-computed value fixes `κ(x) = χ_REF / x` along the x-axis. Substituting `τ_readout(x) = 0.1 / γ_Purcell(x)` and `γ_1_REF` yields the y-axis coordinate: `y_Purcell(x) = γ_1_REF · 0.1 / γ_Purcell(x)`. Below the line, Purcell dominates intrinsic T₁.
2. **Dispersive breakdown** — locus where `χ · τ_readout = 2π`, i.e., the drive phase accumulates a full cycle within one dispersive period. With `χ = x · κ_REF` (holding κ at REFERENCE to resolve x-axis) and `τ_readout = 2π / χ(x)`, the y-axis coordinate is `y_disp(x) = γ_1_REF · 2π / (x · κ_REF)`. Above the line, the pulse is too selective and the dispersive approximation is self-inconsistent.
3. **Resonator-too-slow** — locus where `κ · τ_readout = 1`, i.e., the resonator cannot complete one response period within the pulse. With `κ = κ_REF` held at REFERENCE, `τ_readout = 1 / κ_REF`, giving `y_slow = γ_1_REF / κ_REF` as a horizontal line in (x, y) space.

All three are closed-form functions of x ∈ [0.1, 10] at fixed REFERENCE values of `(κ, g, Δ, γ_1)`. `regime_map.py`'s boundary functions take no arguments — REFERENCE values are sourced from `REFERENCE_DEVICE` at module import time.

**Lindblad-vs-analytic validation (Q3 Refinement 2).** On Day 11, evaluate F_assign at two points with the full Module 1 Lindblad simulator:

- Point 1: REFERENCE Q1 (corresponds to Marxer Q1 marker on the map).
- Point 2: (χ/κ = 1, γ₁τ = 0.01) — mid-range validity point.

Compute `F_analytic − F_Lindblad` at each; the maximum deviation is cited in the Figure 4 caption. If the deviation exceeds 5%, the analytic formula gets a leading-order correction term and the validation re-runs.

**Marxer F_sim annotation (Q3 Refinement 1).** Marxer Q1's published (χ/κ, γ₁τ) = (0.41, 3.3e-3) is also computed with the Module 1 simulator at REFERENCE parameters; the resulting `F_sim = {F_ref:.4f}` is printed next to the ★ marker on Panel (b). This grounds the IQM-anchor marker with a simulator-backed number at no extra compute.

**Published device overlays (Q5 lock).**

| Device | χ/κ | γ₁·τ_readout | F_assign | Marker | Source |
|---|---|---|---|---|---|
| Marxer Q1 | 0.41 | 3.3e-3 | 0.99943 | ★ (warm orange, primary) | Marxer 2508.16437 p.15 (measured device table) + Sec V.3 Table 1 |
| Marxer Q2 | 0.76 | 2.7e-3 | 0.99946 | ◆ (warm orange, same-chip) | Marxer 2508.16437 p.15 |
| Bengtsson | 0.5 | ~2.5e-2 | 0.985 | ● (red) | χ/κ from Bengtsson 2308.02079 Eq. 3 (design criterion 2χ=κ); κ/2π ∈ [4, 8] MHz from Sank 2402.00413 §IV (companion paper, same Sycamore class); T₁ ≈ 20 µs estimated from Arute 2019 Sycamore-typical |
| Garnet | ~0.5 (estimated) | ~1.25e-2 (estimated) | 0.97 | □ (red, grey-hatched) | T₁ = 40 µs from Abdurakhimov 2408.12433 p.13; F_assign = 1 − 3×10⁻² from p.9; χ/κ and τ_readout are IQM-design-family estimates flagged in caption |

Hazra 2407.10934 is explicitly **not** plotted: dimon two-mode-Josephson device, non-standard χ-mediation, no measured (non-Purcell-bound) T₁. Cited in §11 as methodological reference for repeated-measurement benchmarking.

Optional Day-10 tightening: if Bengtsson 2308.02079v2 Fig. 1 (per-qubit κ, χ) is accessible via institutional arxiv, upgrade Bengtsson marker from "estimated" to "measured" and update the caption.

### 3.3 Pareto frontier

**Problem.** For each of three parameter-anchored device variants, trace the optimal (ε₀, τ) vs. τ_max constraint curve:

```
maximize   F_assign(ε₀, τ; device)
subject to τ ≤ τ_max
           ε₀ ∈ [ε_min, ε_max]
           τ ∈ [τ_min, τ_max]
```

Varying τ_max over 10 log-spaced values from 100 ns to 2 μs traces the frontier.

**Solver.** `scipy.optimize.minimize` with SLSQP method. Warm-started from a 5×5 coarse grid over (ε₀, τ) to avoid local optima. All function evaluations use `noise_model='ideal'` (Q8 lock). F_opt is reported as the analytic Gaussian-overlap value at the converged (ε₀_opt, τ_opt); uncertainty is post-hoc analytic binomial SE (§3.6).

**Parameter-anchored device variants (Q7 lock).**

| Variant | Label on figure | `dataclasses.replace` construction | Narrative role |
|---|---|---|---|
| V1 | `REFERENCE (≈ Marxer Q1)` | unchanged REFERENCE_DEVICE | Anchor |
| V2 | `T₁ = 40 µs (Garnet-like)` | `replace(REFERENCE.decoherence, gamma_1=1/40e-6, gamma_phi=...)` | Decoherence sensitivity at fixed resonator |
| V3 | `T₁ = 20 µs, κ/2π = 6 MHz (Bengtsson-like)` | `replace(REFERENCE.decoherence, gamma_1=1/20e-6, gamma_phi=...)` + `replace(REFERENCE.resonator, kappa=2π·6e6)` | Decoherence + resonator combined |

The γ_φ values for V2 and V3 are computed via the Koch-back-solve pathway (Module 3's convention): `gamma_phi = 1/T_2_echo - gamma_1/2`, using T_2_echo at the reference's value unless otherwise specified — consistent with the closed-loop arrow's fitted-device construction (§3.4).

**Bootstrap uncertainty — analytic binomial SE.** Per Q8 lock:

```
σ(F_opt) = sqrt(F_opt · (1 − F_opt) / n_shots)
```

Zero extra compute. Rendered as a shaded envelope around the points-on-curve Pareto trace (Panel c).

**Parallelism.** `compute_pareto_frontier` dispatches 3 × 10 = 30 (device, τ_max) tuples via `pareto_one_tuple.map(...)` on Modal, reusing Module 3's `.map()` pattern. Local serial fallback: `list(map(pareto_one_tuple, ...))`. Zero code change between modes.

### 3.4 Recommendation pipeline (closed loop, narrow scope)

**Closed loop (Q4 lock).** Given `ExtractedParameterPack` from Module 3 (fitted T₁, T₂_echo, ω_q, ε_π):

```
ExtractedParameterPack
   │ to_device_config()             (Module 3 bridge; Koch back-solve for E_J;
   ▼                                 REFERENCE-inherited for κ, g, ω_r, truncation)
DeviceConfig  (fitted decoherence + frequency; REFERENCE resonator/coupling)
   │ find_pareto_point(device, tau_max=500e-9)
   ▼
ParetoPoint  (epsilon_0_opt, tau_opt, F_opt, dominant_loss_channel)
   │ compute_all_sensitivities(operating_point_at_optimum)
   ▼
list[SensitivityResult]  (7 params at the PER-DEVICE optimum, not REFERENCE)
   │ take top-3 by |S_θ|; emit warnings where |S_θ| > SENSITIVITY_WARNING_THRESHOLD
   ▼
RecommendationReport  + YAML export
```

Scope is explicitly narrow: fitted (T₁, T₂, ω_q) over REFERENCE resonator+coupling. Spec amendment 4 records the rationale; Figure 4's caption makes the scope visible to the reader.

**Narrative template** (IQM-table rounding convention + metrology σ convention per Q9b):

```python
def generate_narrative(report: RecommendationReport) -> str:
    return (
        f"For the fitted device (T_1 = {T1_us:d} ± {T1_sigma:s} µs, "
        f"T_2_echo = {T2_us:d} ± {T2_sigma:s} µs, "
        f"ω_q/2π = {omega_q_GHz:.3f} GHz), the recommended readout configuration "
        f"is ε_0/2π = {eps_0_MHz:s} MHz at τ = {tau_ns:d} ns; predicted "
        f"F_assign = {F_opt:.4f} ± {F_sigma:s}. The dominant remaining loss "
        f"channel at this optimum is {dominant}; the top-3 parameters by "
        f"|S_θ| are {top_3_formatted}. "
        f"{'[WARNING: %s]' % '; '.join(warnings) if warnings else ''}"
        f"The highest-leverage device improvement would be to change "
        f"{best_parameter}, with a projected ΔF = {delta_F:.4f}."
    )
```

Rounding rules:

- T₁, T₂ to integer µs; ω_q/2π to 3 decimal places GHz; ε₀/2π to 2 sig fig MHz; τ to integer ns.
- F_assign, ΔF to 4 decimal places.
- σ rounded up to 1 sig fig; F's last-decimal position matched to σ's last-decimal position (metrology standard; avoids "σ = 0.0002 vs. 0.00022" ambiguity).
- Asymmetric uncertainty handled: if `FittedParameter` schema exposes `sigma_lo`/`sigma_hi`, display as `value +σ_hi / −σ_lo`; else `value ± σ`.

**Preflight check before template locks.** One-liner on Day 13: `grep -R "sigma_lo\|sigma_hi" dispersive_readout/characterization/fitting.py` — confirms Module 3's schema before template assumes symmetric.

Not an LLM, not an agentic recommender, not an orchestration layer. Template-rendered f-string.

### 3.5 Day-12/13 autodiff add-on (CONTINGENT — 4-hour cap, three abort signals)

**Trigger.** Ship the add-on only if, at end of Day 11:

- All three baseline deliverables on track (tornado + regime map committed standalone; Pareto structure in place).
- No unresolved blockers in Modules 1–3.
- Modal image pre-warm task (Day 11 afternoon) has succeeded.

**Target.** Gradient-based refinement of `edge_sigma` (the only pulse-shape DoF held fixed in the baseline — see §2.1 note on why there is no independent `pulse_plateau_duration` parameter). `duration` is held at the Pareto-converged `τ_opt` for the variant being refined, so autodiff explores pulse *shape* orthogonally to the baseline `(ε_0, τ)` optimization. Compares autodiff-refined F_assign to the grid-search result at the same `(τ_max, ε_0_opt)`.

**Abort signals (Q9a lock).** Any one trips → immediate revert:

1. JAX-compatible Lindblad forward pass does not produce a finite F within the first **90 minutes**.
2. Autodiff gradient vs. central-finite-diff agreement fails at the **10% level on S_χ** after the first **3 hours**.
3. Any of the three baseline deliverables has unresolved bugs at the **4-hour hard cap**.

**If shipped.** Render as an inset on Figure 4 Panel (c), showing the (σ_edge, plateau) trajectory converging to the grid-search optimum. README framing: *"The Module 1 simulator is backend-agnostic; a small autodiff-based pulse-edge refinement is included as a proof of concept, demonstrating the optimization layer supports gradient-based extensions."* Not framed as a headline.

**If aborted.** Cleanly revert `optimization/autodiff_addon.py` and any Panel-(c) inset wiring. No half-shipped state.

### 3.6 Uncertainty model (unified across sensitivity, regime, Pareto, recommend)

Single convention, analytic throughout:

- `σ(F) = sqrt(F · (1 − F) / n_shots)` at `n_shots = 10⁴`.
- `σ(S_θ) = sqrt(2) · σ(F) / (h · F)` at h = 0.05.
- No bootstrap resampling machinery (Q8 Option I lock).

Shot-noise sampling appears only at (i) O5b Welch-t detectability assertion, (ii) any verification-style single-point sanity check mirroring `operating_point.py`'s pattern.

---

## 4. Module structure

```
dispersive_readout/                            (existing package root)
├── optimization/                              (NEW Module 4)
│   ├── __init__.py                            # exposes compute_all_sensitivities,
│   │                                           # compute_pareto_frontier,
│   │                                           # recommend_from_fitted_parameters,
│   │                                           # pareto_one_tuple (Modal-ready public fn)
│   ├── sensitivity.py                         # §5.1
│   ├── regime_map.py                          # §5.2
│   ├── pareto.py                              # §5.3
│   ├── modal_pareto.py                        # §5.4  (public, not _underscored; Q-note)
│   ├── recommend.py                           # §5.5
│   └── autodiff_addon.py                      # §5.6 (CONTINGENT)
├── physics/                                   (existing; two one-line additions)
│   ├── lindblad.py                            # +1 line: chi_scale kwarg on build_hamiltonian (line 191)
│   └── readout_model.py                       # +1 line: thread chi_scale through simulate_readout
└── tests/
    └── test_optimization.py                   # O1–O24 (28–29 tests total)

06_Dispersive_Readout/
├── scripts/
│   └── fig4_optimization.py                   # NEW; renders Figure 4 composite
├── figures/
│   ├── fig4_optimization.png                  # Module 4 artifact
│   ├── fig4_data.yaml                         # Test O9 regression gate artifact
│   └── recommendation.yaml                    # RecommendationReport export (closed-loop demo)
├── MODULE_4_SPEC.md                           # this file
└── MODULE_4_PLAN.md                           # written by writing-plans skill
```

One-line change to Module 1 public API: `chi_scale: float = 1.0` kwarg on `build_hamiltonian` and `simulate_readout`. Default reproduces current behavior bit-exactly. No breaking change; existing callers (Module 2 `error_budget.py`, Module 3 `protocols.py`) unaffected.

---

## 5. Detailed component specs

### 5.1 `optimization/sensitivity.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Callable
import numpy as np
from pydantic import BaseModel, Field, field_validator
from ..physics.config import DeviceConfig, DriveParams
from ..physics.readout_model import simulate_readout, compute_assignment_fidelity
from ..analysis.operating_point import OperatingPoint

# Policy constants (Q1, Q4, Q6 locks)
SENSITIVITY_FD_STEP: float = 0.05                 # central-FD fractional perturbation
SENSITIVITY_RENDER_BAR_THRESHOLD: float = 0.03    # below this: point-with-errorbar render
SENSITIVITY_WARNING_THRESHOLD: float = 2.0        # above this: warning fires

ParameterName = Literal[
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
]


class SensitivityResult(BaseModel):
    """Normalized log-sensitivity of F_assign to one parameter."""
    parameter: ParameterName
    reference_value: float                  # θ_ref in θ's native units
    reference_unit: str
    sensitivity: float                      # S_θ = ∂ln F / ∂ln θ
    sensitivity_uncertainty: float          # σ(S_θ) from analytic SE propagation
    F_reference: float                      # F at θ_ref
    step_size_used: float = SENSITIVITY_FD_STEP
    method: Literal["finite_diff", "autodiff"] = "finite_diff"
    noise_consistent_with_zero: bool = False  # True iff |S| < RENDER_BAR_THRESHOLD

    @field_validator("sensitivity_uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v < 0:
            raise ValueError("sensitivity_uncertainty must be >= 0")
        return v


def compute_log_sensitivity(
    operating_point: OperatingPoint,
    parameter: ParameterName,
    step_size: float = SENSITIVITY_FD_STEP,
) -> SensitivityResult:
    """Compute S_θ = ∂ln F / ∂ln θ via central finite differences.

    All simulator calls use noise_model='ideal' (Q8 lock).
    """
    ...  # impl in Task-level plan


def compute_all_sensitivities(
    operating_point: OperatingPoint,
    parameters: list[ParameterName] | None = None,
) -> list[SensitivityResult]:
    """Compute sensitivities for all 7 parameters by default."""
    ...


def rank_sensitivities(results: list[SensitivityResult]) -> list[SensitivityResult]:
    """Sort by |sensitivity|, descending."""
    ...


def day_10_cross_check_s_g_vs_s_chi(operating_point: OperatingPoint) -> dict:
    """Day-10 Q1 cross-check: compute S_g via ±5% on coupling.g and compare to 2·S_χ.

    Returns a dict with keys 'S_chi', 'S_g', 'predicted_S_g', 'residual',
    'residual_fractional'. The residual_fractional goes into the Figure 4 caption.
    """
    ...
```

### 5.2 `optimization/regime_map.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.stats import norm

@dataclass(frozen=True)
class DevicePoint:
    """A published device's position on the (χ/κ, γ₁·τ_readout) regime map."""
    label: str
    citation: str
    chi_over_kappa: float
    gamma_1_tau: float
    reported_F_assign: float | None
    marker: str                    # matplotlib marker code
    marker_color: str              # per Q6: Marxer pair warm-orange; Bengtsson/Garnet red
    estimated: bool = False        # True → grey-hatched marker fill (Q5)
    estimated_fields: tuple[str, ...] = ()


PUBLISHED_DEVICE_POINTS: list[DevicePoint] = [
    DevicePoint(
        label="Marxer Q1 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437, p.15 device table + §V.3 Table 1",
        chi_over_kappa=2.5 / 6.1,           # 0.41
        gamma_1_tau=280e-9 / 86e-6,         # 3.3e-3
        reported_F_assign=0.99943,
        marker="*",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Marxer Q2 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437, p.15",
        chi_over_kappa=2.6 / 3.4,           # 0.76
        gamma_1_tau=280e-9 / 102e-6,        # 2.7e-3
        reported_F_assign=0.99946,
        marker="D",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Bengtsson (Google, 2024)",
        citation=(
            "Bengtsson et al., Phys. Rev. Lett. 132 100603 (2024) / "
            "arXiv:2308.02079 Eq. 3; κ from Sank 2402.00413 §IV; "
            "T1 estimate from Arute 2019 Sycamore-typical"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 20e-6,         # 2.5e-2
        reported_F_assign=0.985,
        marker="o",
        marker_color="red",
        estimated=True,
        estimated_fields=("T_1",),
    ),
    DevicePoint(
        label="Garnet (IQM, 2024)",
        citation=(
            "Abdurakhimov et al., arXiv:2408.12433 p.9 (F_assign) + p.13 (T1); "
            "χ/κ and τ_readout are IQM design-family estimates"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 40e-6,         # 1.25e-2
        reported_F_assign=0.97,
        marker="s",
        marker_color="red",
        estimated=True,
        estimated_fields=("chi_over_kappa", "tau_readout"),
    ),
]


def f_analytic_dispersive(
    chi_over_kappa: np.ndarray,
    gamma_1_tau: np.ndarray,
    n_phot: float,
) -> np.ndarray:
    """Closed-form F_assign per Bengtsson 2024 PRL §II with Blais RMP §V.B
    cross-check. Linear decoherence envelope (within 1% of exp form over the
    grid range)."""
    snr_steady = 4.0 * chi_over_kappa * np.sqrt(n_phot) / (1.0 + (2.0 * chi_over_kappa) ** 2)
    snr_eff = snr_steady * np.sqrt(np.clip(1.0 - gamma_1_tau / 2.0, 0.0, 1.0))
    return norm.cdf(snr_eff / 2.0)


def compute_analytic_regime_map(
    chi_over_kappa_range: tuple[float, float] = (0.1, 10.0),
    gamma_1_tau_range: tuple[float, float] = (1e-4, 1e-1),
    n_chi: int = 20,
    n_gamma: int = 20,
    n_phot: float | None = None,     # default: inferred from REFERENCE operating point
) -> dict:
    """Return dict with 'chi_over_kappa_axis', 'gamma_1_tau_axis', 'F_grid',
    'n_phot_used'. No Lindblad calls; sub-second."""
    ...


def validate_analytic_vs_lindblad(
    validation_points: list[tuple[float, float]] | None = None,
) -> dict:
    """Day-11 Q3 Refinement 2: evaluate F_sim at specified points and compare
    to F_analytic. Default: Marxer Q1 + (chi/kappa=1, gamma_1_tau=1e-2).
    Returns {'max_deviation_percent': ..., 'per_point': [...]}. Cited in caption."""
    ...


def purcell_boundary(...) -> np.ndarray: ...
def dispersive_breakdown_boundary(...) -> np.ndarray: ...
def resonator_too_slow_boundary(...) -> np.ndarray: ...
```

### 5.3 `optimization/pareto.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from pydantic import BaseModel, field_validator
from scipy.optimize import minimize
from ..physics.config import DeviceConfig, DriveParams

# Parameter-anchored variants (Q7 lock)
PARETO_DEVICE_VARIANTS: list[dict] = [
    {"label": "REFERENCE (≈ Marxer Q1)", "T1_us": None, "kappa_MHz": None},  # unchanged
    {"label": "T₁ = 40 µs (Garnet-like)", "T1_us": 40.0, "kappa_MHz": None},
    {"label": "T₁ = 20 µs, κ/2π = 6 MHz (Bengtsson-like)", "T1_us": 20.0, "kappa_MHz": 6.0},
]

TAU_MAX_GRID_NS: np.ndarray = np.logspace(np.log10(100), np.log10(2000), 10)  # 100-2000 ns


class ParetoPoint(BaseModel):
    """Optimal (ε₀, τ) at one τ_max constraint, for one device."""
    device_id: str                    # hash of replaced DeviceConfig
    device_label: str
    tau_max: float
    epsilon_0_opt: float
    tau_opt: float
    F_assign_opt: float               # analytic Gaussian-overlap F at the optimum
    F_assign_uncertainty: float       # analytic binomial SE at n_shots=10⁴
    dominant_loss_channel: str        # from Module 2 ErrorBudget at this operating point
    solver_converged: bool

    @field_validator("tau_opt")
    @classmethod
    def _tau_opt_le_tau_max(cls, v: float, info) -> float:
        if v > info.data.get("tau_max", float("inf")) * 1.001:  # 0.1% tol for solver slop
            raise ValueError(f"tau_opt ({v}) exceeds tau_max ({info.data['tau_max']})")
        return v


def find_pareto_point(
    device: DeviceConfig,
    tau_max: float,
    epsilon_0_bounds: tuple[float, float] = (1e6, 1e9),
    tau_bounds: tuple[float, float] = (50e-9, None),   # upper = tau_max
    n_warm_start_grid: int = 25,                        # 5×5
) -> ParetoPoint:
    """Coarse-grid-warm-started SLSQP.

    All objective evaluations use noise_model='ideal' (Q8 lock).
    F_assign_uncertainty is analytic binomial SE at n_shots=10⁴.
    """
    ...


def compute_pareto_frontier(
    device: DeviceConfig,
    tau_max_values: np.ndarray = TAU_MAX_GRID_NS * 1e-9,
) -> list[ParetoPoint]:
    """Trace one device's Pareto frontier across tau_max values.

    Delegates per-point work to pareto_one_tuple.map(...) (Modal) or serial
    fallback. See optimization/modal_pareto.py."""
    ...


def build_variant(variant_spec: dict) -> DeviceConfig:
    """Construct a PARETO_DEVICE_VARIANTS entry via dataclasses.replace.

    γ_φ is recomputed via Koch-style back-solve: gamma_phi = 1/T_2_echo - gamma_1/2,
    matching Module 3's to_device_config convention."""
    ...
```

### 5.4 `optimization/modal_pareto.py`

```python
"""Modal-parallelized Pareto per-point dispatch.

Public module (not _underscored): parallelism boundary is a first-class
architectural surface. Matches Module 3's recovery.py/fit_one_device precedent.
"""
from __future__ import annotations
import modal
from ..physics.config import DeviceConfig
from .pareto import ParetoPoint, find_pareto_point

# Module 4 image extends Module 3's with qutip, scipy, scipy.optimize
stage_06_module4_image = modal.Image.debian_slim().pip_install(
    "numpy", "scipy", "qutip", "pydantic", "pyyaml",
)
app = modal.App("stage06-module4-pareto", image=stage_06_module4_image)


@app.function(cpu=2, memory=4096)
def pareto_one_tuple(device: DeviceConfig, tau_max: float) -> ParetoPoint:
    """Single-tuple Pareto-point computation. Parallel-safe: pure function,
    no global state, no filesystem side effects. Module 3 convention."""
    return find_pareto_point(device, tau_max)
```

### 5.5 `optimization/recommend.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import yaml
from pydantic import BaseModel
from ..physics.config import DriveParams
from ..characterization.fitting import ExtractedParameterPack
from .sensitivity import SensitivityResult, SENSITIVITY_WARNING_THRESHOLD
from .pareto import ParetoPoint


class RecommendationReport(BaseModel):
    """Closed-loop output: fit → recommend → report."""
    device_parameters_fitted: dict        # ExtractedParameterPack as dict
    optimal_drive: dict                   # DriveParams as dict
    predicted_F_assign: float
    predicted_F_uncertainty: float
    top_3_sensitivities: list[SensitivityResult]
    all_sensitivities: list[SensitivityResult]
    dominant_loss_channel: str
    sensitivity_warnings: list[str]       # fires when |S_θ| > SENSITIVITY_WARNING_THRESHOLD
    recommendation_narrative: str
    scope_caveat: str = (
        "Closed-loop scope: fitted (T₁, T₂, ω_q) over fixed REFERENCE resonator "
        "and coupling. Full closed-loop including resonator spectroscopy and "
        "AC-Stark characterization is post-submission roadmap."
    )


def recommend_from_fitted_parameters(
    fitted: ExtractedParameterPack,
    tau_max: float = 500e-9,
) -> RecommendationReport:
    """Pipeline: bridge → Pareto → sensitivities → narrative."""
    ...


def export_recommendation_to_yaml(report: RecommendationReport, path: str) -> None:
    ...


def generate_narrative(report: RecommendationReport) -> str:
    """IQM-table rounding + metrology σ convention (Q9b).

    Delegates per-value formatting to _format_value_with_sigma (below) so the
    metrology σ convention is applied consistently across all fields with
    uncertainty, rather than reinvented per f-string token.
    """
    ...


def _format_value_with_sigma(
    value: float,
    sigma: float,
    unit_exponent: int = 0,       # e.g., -6 for µs display, 6 for MHz, 9 for GHz
    sigma_lo: float | None = None,  # for asymmetric CIs from FittedParameter
    sigma_hi: float | None = None,
) -> tuple[str, str]:
    """Return (value_str, sigma_str) with:
      - σ rounded UP to 1 significant figure;
      - value rounded to the same decimal position as σ's last digit
        (metrology standard; eliminates "0.0002 vs 0.00022" ambiguity);
      - asymmetric form `value +σ_hi / −σ_lo` when sigma_lo/sigma_hi given
        (Module 3 `FittedParameter` schema may expose these; checked by
        Day-13 preflight grep).
    """
    ...
```

### 5.6 `optimization/autodiff_addon.py` (CONTINGENT)

```python
"""CONTINGENT Day-12/13 add-on: autodiff refinement of Gaussian-edge pulse parameters.

Hard 4-hour cap; three abort signals:
  (i)   JAX Lindblad forward pass not producing finite F within 90 min.
  (ii)  Autodiff-vs-FD gradient disagreement > 10% on S_χ after 3 hours.
  (iii) Unresolved bugs in any baseline deliverable at the 4-hour mark.
Any trigger → immediate revert; this module unloads cleanly from Figure 4.
"""
from __future__ import annotations
from typing import Any


def autodiff_refine_pulse_edges(
    device: Any,                        # DeviceConfig
    tau_max: float,
    initial_edges: tuple[float, float], # (sigma_edge, plateau_duration)
    n_steps: int = 50,
    learning_rate: float = 1e-3,
) -> dict:
    """Refine pulse-edge parameters via jax.grad + Adam.

    Returns
    -------
    dict with keys:
      'trajectory':           list[(sigma_edge, plateau, F_assign)]
      'final_F_assign':       float
      'grid_search_F_assign': float            (for comparison at same tau_max)
      'improvement_fraction': float
      'aborted':              bool
      'abort_reason':         str | None      (one of 'forward_pass_90min',
                                               'grad_agreement_3hr',
                                               'baseline_bugs_4hr', None)
    """
    ...
```

### 5.7 Module 1 API extension: `chi_scale` kwarg

One-line change to `dispersive_readout/physics/lindblad.py:191`:

```python
# Before
chi_per_level = dispersive_shift_full(energies, n_mat, g, omega_r)

# After
chi_per_level = chi_scale * dispersive_shift_full(energies, n_mat, g, omega_r)
```

with `chi_scale: float = 1.0` added as a kwarg on `build_hamiltonian`. Threaded through `simulate_readout` as a passthrough kwarg (same default). Default behavior bit-exactly unchanged; existing Module 2/3 callers unaffected.

---

## 6. Validation tests

### 6.1 Test catalog

| Test | Covers | Compute |
|---|---|---|
| **O1** | Sensitivity sign sanity: `S_χ > 0`, `S_{γ_1} < 0`, `S_{γ_φ} < 0`, `S_{n̄_th} < 0` | < 1 min |
| **O2** | Sensitivity step-independence: S at h=0.05 vs h=0.025 within 10% | < 1 min |
| **O3a** | Analytic-vs-Lindblad regime map at Marxer Q1: `abs(F_analytic − F_Lindblad) < 5%` | 1 sim call |
| **O3b** | Analytic-vs-Lindblad regime map at (χ/κ=1, γ₁τ=0.01): same 5% tolerance | 1 sim call |
| **O4** | Pareto monotonicity: F_opt non-decreasing in tau_max along each variant's curve | uses O5's variants |
| **O5a** | Closed-loop modeled improvement: `F_opt_analytic − F_default_analytic > 0.005` | moderate |
| **O5b** | Closed-loop shot-noise detectability: Welch-t on n_shots=10⁴ samples, p < 0.05 | 2 extra sim calls |
| **O6.1** | Pydantic: SensitivityResult requires finite, uncertainty >= 0 | < 1 s |
| **O6.2** | Pydantic: ParetoPoint.tau_opt <= tau_max (within solver tol) | < 1 s |
| **O6.3** | Pydantic: RecommendationReport requires non-empty all_sensitivities | < 1 s |
| **O7** | Autodiff-vs-grid agreement within 1% at same tau_max (CONTINGENT, per §3.5) | contingent |
| **O8** | Q8 analytic-objective contract: `grep noise_model='gaussian'` returns empty inside `optimization/pareto.py` and `optimization/sensitivity.py` | < 1 s |
| **O9** | Regression gate: regenerate fig4_data.yaml at SEED=42 and compare to committed artifact (±2% per sensitivity, ±2% per Pareto point) | moderate |
| **O10** | Modal image smoke: `pareto_one_tuple.map([one_tuple])` returns valid `ParetoPoint` | < 30 s once pre-warmed |
| **O11** | `sensitivity_warnings` fires on a boundary-proximate device (e.g., T₁ = 5 µs forces `abs(S_{γ_1}) > 2.0`) | < 1 min |
| **O12–O18** | Per-parameter sensitivity unit checks (chi_scale, kappa, gamma_1, gamma_phi, n_th, epsilon_0, tau) | < 1 min |
| **O19** | Pareto edge case: tau_max at lower boundary (100 ns) returns feasible point | < 1 min |
| **O20** | Pareto edge case: tau_max at upper boundary (2 µs) returns feasible point | < 1 min |
| **O21** | Pareto edge case: infeasibility detection when drive bounds exclude all F > 0.5 | < 1 min |
| **O22** | Bridge round-trip: `to_device_config(fitted_for_V2)` gives expected `gamma_1` | < 1 s |
| **O23** | Bridge round-trip: `to_device_config(fitted_for_V3)` gives expected `gamma_1`, `kappa` | < 1 s |
| **O24** | Day-10 cross-check: `abs(S_g − 2·S_χ) / (2·S_χ)` logged to caption | 4 sim calls |

**Total: 28 tests** (29 with O7 contingent). Comparable to Module 2 (22) and Module 3 (25).

### 6.2 Policy contracts enforced by tests

- **O8 (analytic-objective contract)**: no `noise_model='gaussian'` or finite `n_shots` in optimization-inner-loop call sites.
- **Policy constants live in source, not figure scripts**: `SENSITIVITY_FD_STEP`, `SENSITIVITY_RENDER_BAR_THRESHOLD`, `SENSITIVITY_WARNING_THRESHOLD` are imported into both `fig4_optimization.py` and the test suite; any change to a constant value is caught by O9 (regression gate).
- **Seed discipline**: all stochastic tests use `SEED=42` (Module 3 convention).

---

## 7. Figure 4 specification

**File:** `scripts/fig4_optimization.py` → `figures/fig4_optimization.png`
**Layout:** 3 horizontal panels (a, b, c), 1400 px wide, 150 DPI, white background.

### Panel (a) — Sensitivity tornado

- **Horizontal bar chart** of all 7 S_θ, sorted by `abs(S_θ)` descending from top.
- **Cool palette** (memory rule: sensitivity = cool). Positive/negative S share hue; sign conveyed by bar direction.
- **Point-with-errorbar rendering** when `abs(S_θ) < SENSITIVITY_RENDER_BAR_THRESHOLD = 0.03`. Memory rule: noise-consistent values don't get filled bars.
- **Numeric annotation** above each bar (S value, 3 sig figs). Memory rule.
- **y-axis labels include perturbation scale**: `χ (via chi_scale, ±5%)`, `κ (±5%)`, `T_1 (±5%)`, `T_φ (±5%)`, `n̄_th (±5%)`, `ε_0 (±5%)`, `τ (±5%)`.
- **x-axis label**: `Normalized log-sensitivity S_θ = ∂ln F / ∂ln θ`.
- **Vertical zero line**: grey dashed.
- **Anchoring subtitle**: `F_ref = 0.99XX, τ_int = 500 ns, n̄_phot = X`.
- **Title**: `Parameter sensitivity of F_assign at REFERENCE (Marxer 2508.16437)`.

### Panel (b) — Analytic regime map

- **Viridis heatmap** of `F_analytic(χ/κ, γ₁·τ_readout)`, log-log axes.
- **Contour lines** at F = 0.95, 0.99, 0.999: white dashed, inline-labeled.
- **Analytic boundaries** (grey dashed, memory rule): Purcell limit, dispersive breakdown, resonator-too-slow. Each labeled inline.
- **4 device markers** per PUBLISHED_DEVICE_POINTS:
  - Marxer Q1 ★ warm-orange, labeled `F_sim = 0.99XX` (Q3 Refinement 1 annotation).
  - Marxer Q2 ◆ warm-orange (same-color shape-family encoding per Q6).
  - Bengtsson ● red.
  - Garnet □ red, grey-hatched fill (per Q5 to signal estimated coordinates).
- **Anchoring subtitle**: `Dispersive-analytic F (Bengtsson 2024 PRL §II); Lindblad-validated at 2 points, max deviation Y%; n̄_phot = X`.
- **Title**: `Regime map of dispersive readout fidelity`.

### Panel (c) — Pareto frontier

- **x-axis log-scale**: `Readout duration τ_opt (ns)`. **y-axis**: `F_assign at optimum`.
- **3 curves**, one per variant in PARETO_DEVICE_VARIANTS. Neutral-category colors (dark grey / slate / charcoal with line-style differentiation). Parameter-anchored labels per Q7.
- **Points-on-curve + shaded envelope** per Q6 ζ. 10 points per curve (TAU_MAX_GRID_NS).
- **Closed-loop arrow** from `(τ_default, F_default)` at the fitted-demo-device to `(τ_opt, F_opt)` on the REFERENCE-with-fitted-decoherence curve. Arrow label: `Closed-loop recommendation: ΔF = +X% at fitted device`. Small italic under arrow: `Scope: fitted (T₁, T₂, ω_q) over fixed REFERENCE resonator`.
- **Anchoring subtitle**: `ε_0 ∈ [ε_min, ε_max], SLSQP + 5×5 warm-start, Modal`.
- **Optional autodiff inset** (contingent §3.5): upper-right, shows (σ_edge, plateau) trajectory converging to grid optimum.
- **Title**: `Speed–fidelity Pareto frontier`.

### Figure-wide caption

> **Figure 4.** Optimization layer for dispersive transmon readout. (a) Normalized log-sensitivities of F_assign to 7 device and control parameters at REFERENCE (Marxer arXiv:2508.16437); sensitivities computed with parameters treated as independent axes via `chi_scale` (see text); positive and negative bars share the cool palette, sign is direction. Day-10 cross-check |S_g − 2·S_χ| = Z%; Purcell coupling contributes W% to an (A)-style χ-sensitivity. (b) Analytic regime map using the dispersive-SNR closed form (Bengtsson 2024 PRL §II; Blais RMP 2021 §V.B); decoherence envelope linearized (deviation from exp form < 1% over y-axis range). Grey dashed: Purcell limit, dispersive breakdown, resonator-too-slow boundary. Published devices: Marxer Q1 (★, F_sim = 0.99XX), Marxer Q2 (◆, same-chip fabrication spread), Bengtsson (●, κ cited from Sank 2402.00413 §IV, T₁ estimated from Arute 2019 Sycamore-typical), Garnet (□, hatched: χ/κ and τ_readout are IQM design-family estimates). Hazra 2407.10934 (dimon device, non-standard χ-mediation) cited in reference list but not plotted. Lindblad-vs-analytic validation at Marxer Q1 and (χ/κ=1, γ₁τ=0.01): max deviation Y%. (c) Pareto frontiers for three parameter-anchored variants of REFERENCE (V1=REFERENCE, V2=T₁=40µs, V3=T₁=20µs + κ/2π=6 MHz). Curves represent the Pareto frontier predicted by this work's simulator under parameter substitution — not the frontier achievable on the cited devices' native hardware. Arrow: closed-loop recommendation from Module 3's fitted parameters (T₁, T₂, ω_q over fixed REFERENCE resonator and coupling; full closed-loop including resonator spectroscopy and AC-Stark characterization is post-submission roadmap) to the Module 4 optimum. Shaded envelope: analytic binomial SE at n_shots = 10⁴. The V1↔V2 T₁-offset illustrates the linear sensitivity of F_assign to T₁ at fixed κ, consistent with panel (a)'s ranking. Module 1 simulator used throughout; drive uses `noise_model='ideal'` inside the optimizer (analytic objective); shot-noise enters only at the envelope and at test O5b.

### Style locks

- Consistent colormap family with Figures 1–3.
- 150 DPI, 1400 px wide.
- All axes labeled with units. All published device points cited in caption.
- Top/right spines off. White background.
- No emojis.

---

## 8. Day-by-day breakdown

### Day 10 (Sat Apr 25) — Sensitivity analysis

**Morning:**
- Add `chi_scale: float = 1.0` kwarg to `build_hamiltonian` + thread through `simulate_readout`.
- Write `optimization/sensitivity.py` with policy constants (`SENSITIVITY_FD_STEP`, `SENSITIVITY_RENDER_BAR_THRESHOLD`, `SENSITIVITY_WARNING_THRESHOLD`).
- Write `SensitivityResult` Pydantic schema; tests O1 (sign sanity), O6.1 (schema validation), O8 (analytic-objective contract on sensitivity.py).
- Run `compute_all_sensitivities` at REFERENCE.

**Afternoon:**
- Test O2 (step-independence at h=5% vs h=2.5%) passing.
- **O24 Day-10 cross-check**: compute S_g via ±5% on `coupling.g`, log `|S_g − 2·S_χ|` to `docs/day10_cross_check.txt`.
- **O11 sensitivity_warnings firing test** (moved from Day 13 per Q9c Change 4).
- First-pass tornado plot rendered standalone.
- Begin `optimization/regime_map.py` — analytic F_assign function and boundary fns.

**End-of-day checkpoint:** All 7 sensitivities committed with correct signs; Day-10 cross-check result in repo; tornado plot standalone; O1, O2, O6.1, O8, O11, O24 passing.

### Day 11 (Sun Apr 26) — Regime map + Modal infrastructure

**Morning:**
- Finish `regime_map.py`: `compute_analytic_regime_map`, analytic boundary functions, `PUBLISHED_DEVICE_POINTS` populated from §3.2 table.
- **O3a, O3b Lindblad-vs-analytic validation** at Marxer Q1 and (χ/κ=1, γ₁τ=0.01). If max deviation > 5%, add correction term to analytic formula and re-run.
- **Optional tightening**: if Bengtsson 2308.02079v2 Fig. 1 per-qubit κ, χ is accessible via institutional arXiv, upgrade Bengtsson marker from "estimated" to "measured."

**Afternoon:**
- **Modal image pre-warm task** (per Q2 lock): build image with qutip + scipy + scipy.optimize on top of Module 3's base; run `pareto_one_tuple.map([one_tuple])` smoke call.
- Test O10 (Modal smoke) passing.
- Regime map rendered standalone.
- Verify 4 device overlays sit in plausible F regions (Marxer ≥ 0.99; Bengtsson ~0.98; Garnet ~0.97).

**End-of-day checkpoint:** Regime map committed standalone; Modal image pre-warmed and smoke-tested; Bengtsson tightening done or flagged; O3a, O3b, O10 passing.

### Day 12 (Mon Apr 27) — Pareto frontier (Modal-parallelized)

**Morning:**
- Write `optimization/pareto.py`: `ParetoPoint` schema, `find_pareto_point` with 5×5 warm-start SLSQP, `build_variant`, `compute_pareto_frontier`.
- Write `optimization/modal_pareto.py` — `pareto_one_tuple` Modal function.
- Run 3 variants × 10 τ_max on Modal (~5 min wall-clock).
- Test O4 (Pareto monotonicity) + O6.2 (ParetoPoint schema) + O19/O20/O21 (edge cases) passing.

**Afternoon:**
- Pareto results committed; Panel (c) standalone plot generated.
- Begin `optimization/recommend.py`: `RecommendationReport` schema; preflight check on `FittedParameter` for asymmetric-σ support.

**End-of-day checkpoint:** Pareto frontier committed standalone; recommend.py skeleton in place; O4, O6.2, O19, O20, O21 passing. **CV/CL work moved to Day 13 afternoon per Q9c Change 3.**

### Day 13 (Tue Apr 28) — Closed loop + Figure 4 + contingent autodiff + CV/CL

**Morning:**
- Finish `recommend.py`: `recommend_from_fitted_parameters`, `generate_narrative` (IQM rounding + metrology σ), `export_recommendation_to_yaml`.
- Pick demo device from `recovery_coverage_report.yaml` (SEED=42 stable; select the device whose fitted (T₁, T₂, ω_q) shifts the Pareto optimum most visibly from REFERENCE's).
- **O5a + O5b closed-loop tests** + **O6.3 schema** + **O22/O23 bridge round-trips** passing.
- **Contingent autodiff add-on** starts (hard 4-hour cap, three abort signals per §3.5). If any abort signal trips at 90 min / 3 hr / 4 hr → revert cleanly, move on.

**Afternoon:**
- Write `scripts/fig4_optimization.py` — composite 3-panel figure.
- Render Figure 4; compare style to Figures 1–3; tight polish.
- Run **O9 regression gate**: regenerate fig4_data.yaml at SEED=42, compare to committed artifact.
- Commit Figure 4 + all Module 4 code.
- **CV v1 and cover letter v1 drafting** (moved from Day 12 afternoon per Q9c Change 3). Lower-pressure slot after Figure 4 ships.

**End-of-day checkpoint:** Figure 4 committed at 150 DPI, 1400 px wide, matching Stage 06 style family. Closed-loop test passing. Module 4 complete. Commit message: `"Stage 06 Module 4: sensitivity + analytic regime map + Modal-parallelized Pareto frontier + closed-loop recommendation (Q1–Q9 amendments applied)"`. CV/CL v1 drafted.

---

## 9. What to flag to the human

1. **If sensitivity signs are wrong (test O1 fails).** The simulator or the sensitivity code has a bug. Do not "fix" by flipping signs in the figure.
2. **If Day-10 cross-check shows `|S_g − 2·S_χ| / (2·S_χ) > 10%`.** Purcell contamination of an (A)-style χ-sensitivity is non-negligible at REFERENCE — that's the quantitative evidence for the Q1 orthogonality decision; it goes into the caption, not fixed.
3. **If analytic-vs-Lindblad regime map deviation > 5% at either validation point (O3a, O3b).** The analytic formula needs a leading-order correction term before the figure ships; do not publish the uncorrected version.
4. **If Pareto is non-monotonic (test O4 fails).** SLSQP is finding local optima. Increase `n_warm_start_grid` from 25 to 100. If still non-monotonic, investigate the objective landscape before changing solvers.
5. **If Modal smoke test (O10) fails after image pre-warm.** The Module 4 image deps are broken; do not let this slip into Day 12 morning — fix on Day 11 afternoon or the Pareto run is blocked.
6. **If the closed-loop recommendation significantly disagrees with the manual Pareto result.** The bridge `to_device_config → Pareto` is miscalibrated; check the Koch back-solve and the γ_φ recomputation.
7. **If any autodiff abort signal trips (90 min, 3 hr, 4 hr).** Revert cleanly. The Module 4 baseline is already shippable; do not sacrifice Figure 4's ship-ready state for the add-on.
8. **If the recommendation narrative reads like an LLM hallucination.** The f-string template is not being populated with real numbers; fix the formatting, not the text.
9. **If Figure 4 has more than three panels** (ignoring the contingent autodiff inset). Cut.
10. **If the amendments table in §0 renders incorrectly on GitHub** (unescaped pipes, column shifts): the spec self-review failed; fix before proceeding to writing-plans.

---

## 10. Review checklist before advancing to the report (Day 14)

- [ ] All 28 Module 4 tests (O1–O24 with O5 split into O5a/O5b, plus O9/O10/O11) passing. O7 passing or cleanly reverted.
- [ ] Tornado plot shows 7 parameters with correct signs; `SENSITIVITY_RENDER_BAR_THRESHOLD` applied; numeric annotations visible; `(±5%)` on axis labels.
- [ ] Regime map rendered with 4 published device points overlaid and correctly cited. Hazra omitted from plot but cited in reference list.
- [ ] Marxer Q1 annotated with `F_sim`; Lindblad-validation deviation printed in caption.
- [ ] Pareto frontiers for 3 parameter-anchored variants traced; 10 τ_max points per curve; points-on-curve + shaded envelope rendering; parameter-anchored labels.
- [ ] Closed-loop arrow annotated with ΔF and scope caveat; demo device pick recorded in commit message.
- [ ] `RecommendationReport` round-trips (fit → recommend → YAML); IQM-table rounding + metrology σ applied; `sensitivity_warnings` field populated correctly.
- [ ] Test O5a (modeled improvement) and O5b (shot-noise detectability) both passing.
- [ ] Test O8 (analytic-objective contract) passing: `grep noise_model='gaussian'` inside `optimization/pareto.py` and `optimization/sensitivity.py` returns empty.
- [ ] Test O9 (YAML regression gate) passing at SEED=42 within ±2% per value.
- [ ] Figure 4 rendered at 150 DPI, 1400 px wide, in same style family as Figures 1–3.
- [ ] Figure 4 caption contains all three locked caveats (orthogonality, analytic-regime, closed-loop-scope) plus the Q7 honesty-preserving sentence.
- [ ] `optimization/__init__.py` exposes `compute_all_sensitivities`, `compute_pareto_frontier`, `recommend_from_fitted_parameters`, and `pareto_one_tuple`.
- [ ] Autodiff add-on either shipped cleanly or cleanly reverted — no half-shipped state.
- [ ] Policy constants (`SENSITIVITY_FD_STEP`, `SENSITIVITY_RENDER_BAR_THRESHOLD`, `SENSITIVITY_WARNING_THRESHOLD`) in source, imported by both figure script and tests.
- [ ] CV v1 and cover letter v1 drafted (moved to Day 13 afternoon per Q9c Change 3).

If any item is unchecked, the report does not start.

---

## 11. Reference list for Module 4

- **Bengtsson et al., Phys. Rev. Lett. 132, 100603 (2024)** / arXiv:2308.02079 — Model-based readout optimization. Methodological template for this entire module (spec §10); source for the χ/κ = 0.5 design-criterion marker on Panel (b).
- **Sank et al., arXiv:2402.00413 (2024)** — Companion paper to Bengtsson: measured κ distribution on the same Sycamore class. Primary source for Bengtsson's κ ∈ [4, 8] MHz range on Panel (b).
- **Gautier et al., arXiv:2403.14765 (v3 Feb 2025)** — Adjoint-state optimal control for transmon readout. Reference for the contingent autodiff upgrade path (§3.5); not required for baseline.
- **Marxer et al., arXiv:2508.16437 (Aug 2025)** — Primary REFERENCE_DEVICE anchor. Panel (b) Marxer Q1 and Q2 markers from p.15 device table + §V.3 Table 1.
- **Hazra et al., Phys. Rev. Lett. 134, 100601 (2025)** / arXiv:2407.10934 — Repeated-measurement benchmarking; methodological reference for Panel (c) closed-loop scope discussion. **Not plotted** on regime map (dimon device, non-standard χ-mediation).
- **Abdurakhimov et al., arXiv:2408.12433 (2024)** — IQM Garnet 20-qubit benchmarks. Source for Panel (b) Garnet marker (T₁ = 40 µs, F = 0.97; χ/κ and τ_readout estimated).
- **Blais et al., Rev. Mod. Phys. 93, 025005 (2021)** — Cross-check reference for the steady-state dispersive SNR formula used in Panel (b)'s analytic regime map.
- **Arute et al., Nature 574, 505 (2019)** — Sycamore device characterization (via Klimov-class Google papers). Source for the Bengtsson T₁ ≈ 20 µs estimate on Panel (b).
- **Koch et al., Phys. Rev. A 76, 042319 (2007)** — Deep-transmon dispersion formula used by Module 3's `to_device_config` to back-solve E_J from ω_q; Module 4 inherits that bridge for V2/V3 construction and the closed-loop demo device.

These sources govern Module 4. Bengtsson is the primary methodological template; Sank is the κ-distribution source; Gautier is the forward-looking bridge to the post-submission autodiff direction; Marxer is the REFERENCE anchor.
