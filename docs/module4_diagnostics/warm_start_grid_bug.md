# Pareto solver bug — Day-13 finding (Amendment #10)

## Summary

`find_pareto_point` shipped with a **5-point linear warm-start grid**
over the 3-decade `eps_0` domain, structurally unable to sample F's
multimodal landscape in the dispersive-saturation regime. Task 14's
30 Pareto points (shipped at commit e6d2e09) all reported F_opt values
~6 F-percentage-points below truth. Surfaced by the Day-13 Task-17
picker diagnostic when every synthetic device returned
`eps_opt = 2.5075e8 exactly`.

The fix required three stacked changes, documented below.

## Root cause

F(eps_0, tau) at REFERENCE, tau=500ns has **two local maxima**
separated by a sharp valley. Verified by a 50-point linear F(eps)
scan at tau=500ns on REFERENCE (evidence: `F_scan_50pts_tau500_reference.npy`
alongside this doc), and by cross-device 50-point scans on T_1
extremes (low_T_1 = 5.4 us at harness index 18, high_T_1 = 91.9 us at
harness index 41; evidence in `diagnostic_peak_ordering.yaml`):

| Feature         | REFERENCE (T_1=30us) | low_T_1 (5.4us)  | high_T_1 (91.9us) |
|-----------------|-----------------------|-------------------|--------------------|
| Peak #1 (secondary) | eps=7.76e7, F=0.99324 | eps=7.76e7, F=0.99116 | eps=7.76e7, F=0.99351 |
| Valley          | eps=1.05e8, F=0.90853 | eps=1.05e8, F=0.89911 | eps=1.05e8, F=0.90990 |
| Peak #2 (global) | eps=1.51e8, F=0.99421 | eps=1.51e8, F=0.99274 | eps=1.51e8, F=0.99441 |

Peak locations identical across devices; peak #2 consistently global
by ~0.001 F margin (directionally stable across the T_1 variation
span spanned).

The shipped n=5 linear warm-start placed eps at
`[1e6, 2.5075e8, 5.005e8, 7.5025e8, 1e9]`. None of these grid points
is inside peak #2's basin (approximately eps in [1.3e8, 1.7e8]).
SLSQP initialized at 2.5075e8 settled at a local max on the
descending tail past peak #2 (F=0.929).

## Progressive fix attempts

Three fix attempts on the way to the shipped solver:

| Attempt | Grid topology | Multi-start | REFERENCE F_opt at tau_max=500ns |
|---|---|---|---|
| (1) n=10 log-spaced, single-start SLSQP | log over 3 decades | no | 0.961 (warm-start winner at eps=2.15e8; SLSQP stuck) |
| (2) n=10 log-spaced, K=5 multi-start SLSQP, AND-based (eps, tau) filter | log over 3 decades | yes | 0.961 (all top-K starts in same basin; no start inside peak #2) |
| (3) **n=10 log-spaced, K=5 multi-start, eps-only filter, per-start sub-grid refinement** | log over 3 decades | yes | **0.9938** (matches n=20 linear reference 0.9930 within shot noise) |

Also ruled out along the way:
- **Increasing SLSQP finite-difference step** (eps=1e7): did not help.
  SLSQP at (2.15e8, 500ns) is at a genuine local maximum, gradient
  magnitude is small, line search produces no improving step,
  `nfev=3` termination regardless of FD step size.
- **n=20 linear** (same cost as n=20 log, ~400 warm-start evals):
  works by coincidence (grid point at 1.5874e8 lands near peak #2),
  but double the cost of n=10 log + sub-grid and no more principled.

## Fix (shipped)

Three components in `find_pareto_point`:

1. **Warm-start grid: n=10 log-spaced on eps_0, linear on tau.**
   Log-spacing is structurally correct for the 3-decade eps domain.
2. **Top-K=5 diverse candidates selected by F** with an **eps-only**
   symmetric-ratio separation filter
   (`max(eps_i/eps_j, eps_j/eps_i) >= 1.2`). Enforcing separation in
   tau is counterproductive — two starts at the same eps with
   different tau converge to the same eps basin, wasting K slots.
3. **Per-start sub-grid refinement** before SLSQP: 5-point linear
   sweep in eps centered on the warm-start winner, +/- 40% half-width,
   tau held fixed. Pulls each start into the nearest local-max basin's
   interior. SLSQP from sub-grid best refines cleanly.

Plus a **zero-width integration window guard in `_F_analytic_at`**:
when callers probe at `tau <= integration_window[0]`, return F=0.5
instead of propagating the `ValueError` raised by `integrated_iq`.
Separate latent bug, surfaced by SLSQP finite-difference probes at the
tau lower bound (the old warm-start's try/except masked it).

## Verification

Local single-device sanity check at REFERENCE, tau_max=500ns,
`find_pareto_point` with the shipped solver:

| Check                       | Result     | Target                    | Status |
|-----------------------------|------------|----------------------------|--------|
| F_opt                       | 0.9938     | >= 0.99                    | PASS   |
| eps_opt                     | 1.40e+08   | 1.3e8-1.7e8 (peak #2)       | PASS   |
| tau_opt                     | 500.00 ns  | 500 ns boundary            | PASS   |
| wall-clock                  | 8.3 min    | < 30 min                   | PASS   |
| converged                   | True       | True                       | PASS   |

Test: `test_warm_start_resolves_basin_at_reference` in
`dispersive_readout/tests/test_optimization.py`, marked `@pytest.mark.slow`.
Asserts F_opt >= 0.99 at REFERENCE, tau_max=500ns. Regresses if any of
the three components above reverts.

## Impact

- **Task 14 Panel (c) cache** (`06_Dispersive_Readout/figures/fig4_panel_c_data.yaml`,
  shipped at commit e6d2e09): regenerated at corrected solver.
  Previous F values were ~6 percentage points below truth.
  Panel-(c) rendering code unchanged; only numerical content.
- **Task 17 picker** (`pick_closed_loop_demo_device.py`): re-dispatched
  on all 51 devices at corrected solver. Original result (every
  device "drift=0.0%, eps_opt=2.5075e8") was an artifact of the bug.
- **Task 20 regression artifact** (`fig4_data.yaml`): generated on
  corrected data; O9 gate pins corrected sensitivities.

## Artifacts

- `F_scan_50pts_tau500_reference.npy` — 50-point F(eps) scan at
  REFERENCE, tau=500ns, eps in [5e7, 5e8] linear. Two-peak structure
  reproducible from this artifact via numpy.load.
- `06_Dispersive_Readout/figures/diagnostic_pareto_structure.yaml` — expanded-B
  diagnostic output: 3 devices x {n=5, n=20} warm-start comparison +
  21x21 F-surface scan summary.
- `06_Dispersive_Readout/figures/diagnostic_peak_ordering.yaml` — cross-device
  peak-ordering verification on low_T_1 + high_T_1 extremes.
- `06_Dispersive_Readout/scripts/diagnostic_pareto_structure.py`,
  `06_Dispersive_Readout/scripts/diagnostic_peak_ordering_across_devices.py` —
  scripts reproducing the diagnostics.

## Tally attribution

Amendment #10 on the Module-4 running amendments list.
Separate tally entry for the two-peak physics finding (Amendment #11,
published-grade observation that peak #1 corresponds to a low-photon
dispersive-shift-clean regime while peak #2 sits in the
dispersive-saturation regime; Marxer's REFERENCE is tuned for peak #2,
consistent with Q3 Panel-(a)'s chi-overprovisioning observation).

Execution finding — caught by designing a downstream diagnostic
(Day-13 closed-loop picker) rather than by adversarial review. The
fix is traceable to a principled decomposition (log topology for the
grid + multi-start for local-max escape + sub-grid for basin
capture + zero-width guard for solver robustness) rather than
empirical tuning.
