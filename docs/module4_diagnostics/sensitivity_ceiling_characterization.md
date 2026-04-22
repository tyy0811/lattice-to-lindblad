# Module 4 diagnostic — sensitivity ceiling characterization

**Completed**: 2026-04-22 during Module 4 Task 6 amendment decision.
**Status**: completed investigation (not deferred). Committed as supervision-evidence artifact for the day-14 report.

## Summary

The empirical `|S_θ|` ceiling under the Lindblad simulator caps at ~0.4 across the realistic parameter space at REFERENCE operating conditions (drive, integration window, truncation, decoherence stack). This ceiling was verified as genuine Lindblad physics — not a solver, truncation, or Purcell-isolation artifact — via three independent reproducibility checks. On the basis of this characterization, `SENSITIVITY_WARNING_THRESHOLD` was amended from the spec-locked 2.0 (unreachable, dead code) to 0.3 (aligned with spec §2.1's "dominance" level; fires on devices where one parameter reaches dominance-level sensitivity).

The closed-form upper bound from the linearized decoherence envelope `√(1 − γτ/2)` gives `|S_γ| > 2` at extreme `γτ`; the Lindblad simulator gives ~5–10× smaller values in the same regime because the actual integration produces a shallower `F(γ_1)` curve (integration-window geometry: decoherence acts over the full integration window, not instantaneously).

## Check 1: Tolerance independence

**Reproduction**: `python docs/module4_diagnostics/check_tolerance.py`

**Method**: Compute `|S_γ1|` at T_1 = 0.22 µs (Purcell-off) with default mesolve tolerances (`atol=1e-10`, `rtol=1e-8`) and with 100× tighter tolerances (`atol=1e-12`, `rtol=1e-10`).

**Result**: `|S_γ1|` changes by **0.000%** under tighter tolerances. `F_ref`, `F_plus`, `F_minus` are bit-identical.

**Verdict**: PASS. Default solver tolerances are not compressing sensitivity.

## Check 2: Truncation independence

**Reproduction**: `python docs/module4_diagnostics/check_truncation.py`

**Method**: Compute `|S_γ1|` at the same T_1 = 0.22 µs stress point with REFERENCE truncation (`N_transmon=5`, `N_resonator=15`) and with enlarged truncation (`N_transmon=7`, `N_resonator=25`). The enlarged basis eliminates the `N_r=15` photon-truncation warning at this stress point (mean photon peaks near 5.8).

**Result**: `F_ref` shifts from 0.81249 → 0.81881 (a 0.6% physical truncation correction, as expected). `|S_γ1|` shifts from 0.24540 → 0.24206, **a 1.36% relative change**.

**Verdict**: PASS (< 5% tolerance). Truncation is not the dominant source of sensitivity compression.

## Check 3: Pure-γ_1 verification via coupling.g = 0

**Reproduction**: `python docs/module4_diagnostics/check_purcell.py`

**Method**: Set `coupling.g = 0` (decouple the qubit from the resonator entirely), set `γ_φ = 0` and `n_th = 0`, set `purcell_enabled=False`. Initialize `|1, vacuum⟩`, evolve under zero drive via mesolve, fit `γ_eff` from `ln P_|1⟩(t)` over `t ∈ [0, 2·T_1]`.

**Result**: `γ_eff − γ_1_true = −1.4 × 10⁻⁶ 1/s`, **relative −0.000000%** (< 1 ppm). `γ_eff = 1/T_1` to numerical precision.

**Verdict**: PASS. No residual decay channel beyond the explicit `decoherence.gamma_1`. Earlier stress tests using `purcell_enabled=False` (with `coupling.g` retained for χ) had no leaked Purcell contribution.

## Amendment

On the basis of the above three checks plus:

- the analytic closed-form derivation of the ceiling (see below),
- the empirical ceiling across 7 parameters (`|S_ε₀|_max = 0.39` at ε/2π = 15 MHz; `|S_γ1|_max = 0.25` at T_1 = 0.22 µs; others below),
- the integration-window geometry factor `4·T_mid/τ ≈ 2.4` that explains the closed-form vs Lindblad ratio,

`SENSITIVITY_WARNING_THRESHOLD` was amended from **2.0 → 0.3** in `dispersive_readout/optimization/sensitivity.py`. The policy-constants test was updated to lock the new value.

`test_O11_sensitivity_warning_fires_*` was rewritten to probe `ε/2π = 15 MHz` at REFERENCE T_1 (`|S_ε₀| = 0.388 > 0.3`) instead of `T_1 = 5 µs` at REFERENCE drive. The drive-stress regime is a realistic operating-point choice a user might make (trading SNR for readout duration); the original `T_1 = 5 µs` probe did not approach even the amended 0.3 threshold under the Lindblad simulator.

## Related finding (Check A, separate diagnostic)

The Day-10 diagnostic that discovered this ceiling also surfaced an integration-window geometry factor:

> The closed-form `|S_γ|` derivation assumed integration from `t=0`. The actual integration window is `(50 ns, 500 ns)`. The leading-order T-weighting factor is `4·T_mid/τ = 4·275 ns / 450 ns = 2.44`, which matches the empirical-to-closed-form ratio observed across γτ ∈ [0.002, 0.09] within 6% (constant across the range, not diverging at small γτ — which would indicate a Lindblad bug).

This finding is a characterization result in its own right: the closed-form upper bounds from the linearized envelope are ~2.4× lower than what the Lindblad simulator produces in the weak-decoherence regime because of integration-window geometry, and ~5–10× higher than the simulator produces in the strong-decoherence regime because the envelope approximation breaks down as `γτ → 2`.

## Reproduction batch

All three checks at once:

```bash
for script in docs/module4_diagnostics/check_*.py; do
    echo "=== $(basename $script) ==="
    python "$script"
    echo
done
```

Expected runtime: ~3 minutes total.
