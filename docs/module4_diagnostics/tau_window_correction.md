# Module 4 diagnostic — τ-window FD correction (Codex adversarial finding)

**Filed**: 2026-04-23.
**Driver**: Codex adversarial review on branch `stage-06-module-4-optimization` surfaced a high-severity FD-dispatcher bug in `dispersive_readout/optimization/sensitivity.py`.
**Status**: fixed in commit `c16dbe5`; this artifact records the before/after deltas.

## The bug

`_perturbed_device_drive_scale` rescaled `drive.duration` for `parameter='tau'` probes but kept `op.integration_window` fixed at REFERENCE's `(50 ns, 500 ns)`. At τ ± 5%:

- **+5% probe**: `drive.duration = 525 ns`, `integration_window = (50, 500) ns`. Integrates a 525 ns pulse over 500 ns — last 25 ns of the pulse is unmeasured.
- **−5% probe**: `drive.duration = 475 ns`, `integration_window = (50, 500) ns`. Integrates a 475 ns pulse over 500 ns — last 25 ns of the window is post-pulse zero-drive resonator ring-down.

The two FD branches therefore compared *physically different observables*, and `S_τ` silently mixed "τ-sensitivity" with "window-mismatch artifact."

## The fix

`_perturbed_device_drive_window_scale` now threads a per-probe integration window through. For τ probes, `window[1]` co-perturbs to match `drive.duration`; `window[0]` (κ-ramp-up exclusion) stays fixed. Both FD branches integrate the full pulse. `S_τ` now measures the "longer pulse + longer integration" package-deal sensitivity — the same landscape `find_pareto_point` navigates when optimizing τ.

## Before / after at REFERENCE

| Parameter | S (before, biased) | S (after, corrected) | σ(S) | Rendering before | Rendering after |
|---|---|---|---|---|---|
| ε₀ | +0.0503 | +0.0503 | 0.0143 | BAR | BAR |
| τ | **+0.0369** | **+0.0297** | 0.0143 | BAR | **point** |
| χ | -0.0292 | -0.0292 | 0.0143 | point | point |
| κ | +0.0105 | +0.0105 | 0.0143 | point | point |
| γ₁ | -0.0005 | -0.0005 | 0.0143 | point | point |
| n_th | -0.0001 | -0.0001 | 0.0143 | point | point |
| γ_φ | +0.0000 | +0.0000 | 0.0143 | point | point |

Only `S_τ` shifted (from 0.0369 to 0.0297, a Δ of −0.0072 ≈ −20%). All other sensitivities bit-identical before/after, consistent with the fix being τ-specific.

## Threshold crossing

τ **crossed the `SENSITIVITY_RENDER_BAR_THRESHOLD = 0.03` threshold from above to below**. Before fix: two bar-rendered parameters (ε₀, τ); after fix: one bar-rendered parameter (ε₀). `|S_τ|/σ` is 2.08 — τ's upper-1σ cap (0.030 + 0.014 = 0.044) still extends well into bar territory, so τ is statistically resolvable from zero, just no longer central-value-above-threshold.

Panel (a) caption implication: the pre-fix caption language could have described "ε₀ and τ as the dominant control parameters"; the post-fix caption should say "ε₀ is the dominant control parameter; τ sits at the rendering-threshold edge with significant statistical support (|S|/σ ≈ 2)." The spec's §0.1 amendment for Panel (a) caption is updated to reflect this.

## Reproduction

All seven sensitivities at REFERENCE under the corrected dispatcher:

```bash
PYTHONPATH=. python 06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py
```

Expected output:

```
  F_ref  = 0.9899
  τ_int  = 450 ns
  n̄_phot = 2.241
  Top-3 by |S|:
    epsilon_0     S=+0.0503 ± 0.0143
    tau           S=+0.0297 ± 0.0143
    chi_scale     S=-0.0292 ± 0.0143
```

## Why the test suite didn't catch this

O1a (sign sanity): passed because S_τ was still positive. O2 (step-independence): passed because the window-mismatch bias was consistent across h = 0.05 and h = 0.025. O8 (analytic-objective contract): passed because `noise_model='analytic'` was correctly used. None of these assertions interrogated the *dispatcher's parameter-configuration self-consistency*.

Commit `c16dbe5` adds `test_fd_dispatcher_probe_configuration_self_consistent`, which asserts that for all 7 parameters, the dispatcher returns a `(device, drive, integration_window, chi_scale)` triple with non-perturbed fields unchanged and perturbed fields aligned. Specifically: for τ probes, `drive.duration == integration_window[1]`; for non-τ probes, `drive.duration` and `integration_window` both equal the reference operating point. This closes the class of FD-dispatcher coupling bugs Codex caught.

## Supervision-evidence framing (day-14 report)

> The Codex adversarial review flagged a high-severity bug in the sensitivity FD dispatcher: τ probes rescaled the drive duration without co-perturbing the integration window, mixing τ-sensitivity with a window-mismatch artifact. The bug had silently biased `S_τ` upward by ~20% (from +0.030 to +0.037, crossing the tornado rendering threshold from below to above). The fix threads a per-probe integration window through the dispatcher; `S_τ` now measures the package-deal sensitivity consistent with how the Pareto solver navigates τ. A new dispatcher-self-consistency regression test closes the class of FD-coupling bugs the original test suite did not interrogate.
