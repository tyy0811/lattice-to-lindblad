# Deferred investigation — C3 regression failure on `stage-06-module-3-characterization`

**Filed**: 2026-04-21 during Module 4 Task 4 post-implementation regression check.
**Observed at commit**: `381996c` (Module 4 Task 4 landed) on branch `stage-06-module-4-optimization`.
**Status**: open, not a Module 4 blocker.

## Symptom

`test_C3_recovery_coverage_matches_committed_artifact` (marked `@pytest.mark.slow`, ~9-min wall-clock) fails:

```
AssertionError: T_1.coverage_1_sigma regression: observed 82.00% vs committed 74.00% (Δ=8.00%)
```

Tolerance is ±2% per parameter per field (`coverage_1_sigma`, `coverage_2_sigma`). 8% delta on T_1's 1σ coverage trips the gate. Direction is "coverage improved" (82 > 74), so the failure is *not* a fit-quality regression — it's a drift between the currently-computed coverage and the committed artifact.

## What was ruled out

Pre-existing on this branch, NOT introduced by Module 4 work:

- `dispersive_readout/characterization/recovery.py` and `dispersive_readout/characterization/protocols.py` import **nothing** from `dispersive_readout/physics/readout_model.py` or `dispersive_readout/physics/lindblad.py` (confirmed via grep at filing time).
- Module 4's edits to Module 1 (`chi_scale` kwarg on `build_hamiltonian`; `noise_model='analytic'` on `compute_assignment_fidelity`) are therefore not on the Module 3 recovery-harness code path.
- Module 4 does not touch `dispersive_readout/characterization/` at all.

## What likely changed

Per the Branch Summary, C3 was green at the `stage06-module3` tag. Something drifted between that tag and `381996c`. Two candidate hypotheses:

1. **Harness non-determinism at `SEED=42`** that wasn't caught by C3's first committed run (e.g., dict-ordering, floating-point accumulation order, scipy/lmfit optimizer internal seeding). If so, C3's ±2% tolerance is tighter than the harness's actual reproducibility floor.
2. **Silent local-env dependency drift** since the artifact was regenerated (commit `f42873e`, "regenerate recovery artifact under F1 envelope-escalation"). A scipy / lmfit / numpy minor-version bump can shift optimizer convergence paths, changing per-device fit residuals and thereby the coverage ratio.

## Resolution path (post-Module-4)

1. Run `pip freeze` inside the current venv and diff against the dependency pins at tag `stage06-module3` (if recorded). Any change in `scipy`, `lmfit`, `numpy`, or their transitive pins is a candidate cause.
2. If no env drift is found: rerun the recovery harness 5 times at `SEED=42` with cold RNGs; if per-run coverage varies outside ±2%, the tolerance is under-specified and should widen to match the observed reproducibility floor.
3. If env drift IS found: regenerate `06_Dispersive_Readout/figures/recovery_coverage_report.yaml` against the current env, note the env pins in the regeneration commit, and keep ±2% as the tolerance.

## Why this is not a Module 4 blocker

- Module 4 does not import, call, or depend on Module 3's recovery harness for any computation.
- Module 4's closed-loop recommendation pipeline (Task 17) uses the *ground-truth* device parameters + `to_device_config()` bridge, not the fitted pack's coverage statistics.
- C3 fires on the committed artifact's match; it does not validate downstream outputs. Module 4's outputs are validated by their own O1–O24 test catalog.

Flag for the day-14 deviations paragraph as a pre-existing Module 3 artifact-drift item, distinct from Module 4 work.
