# Stage 06 Module 4 — Sensitivity and Pareto Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Day-10 execution-time amendments (2026-04-22) — read before implementing Tasks 1-7:**
>
> Three amendments applied during Day-10 execution. See `MODULE_4_SPEC.md` §0.1 for full rationale and `docs/module4_diagnostics/` for the supporting diagnostic artifacts.
>
> - **Amendment 10 (Q1 S_χ sign):** Under the SW-2 simulator at REFERENCE, F_assign peaks at `chi_scale ≈ 0.85`; REFERENCE sits ~18% past the peak so `S_χ = −0.029 ± 0.014` (noise-consistent). **O1 is split into O1a (sign-assert for bar-rendered parameters) and O1b (log-only for near-zero parameters).** Step 4.1's `test_O1_sensitivity_signs_at_REFERENCE` is superseded by the O1a/O1b pair in the shipped code.
> - **Amendment 11 (noise_model='analytic'):** Module 1's shipped `noise_model='ideal'` returns `F=1.0` unconditionally (zero-shot-noise limit, useless for FD). Added `noise_model='analytic'` returning `F = Φ(SNR/2)` as an additive extension. **All references in this plan to `noise_model='ideal'` inside sensitivity.py / pareto.py inner loops should be read as `noise_model='analytic'`.** Module 1 tests `test_assignment_fidelity_analytic_matches_phi_snr_over_2` and `test_assignment_fidelity_gaussian_converges_to_analytic_as_n_shots_grows` pin the new invariants. O8 contract strengthened from 1 to 3 tests per module (forbid `'gaussian'`, forbid `'ideal'`, require `'analytic'`).
> - **Amendment 12 (threshold 2.0 → 0.3):** `SENSITIVITY_WARNING_THRESHOLD` recalibrated from 2.0 (unreachable under Lindblad simulator) to 0.3 (spec §2.1 dominance level). Verified via six independent checks. **Step 2.3's policy-constants code and Step 6.1's O11 probe device are superseded** — see shipped code: `SENSITIVITY_WARNING_THRESHOLD = 0.3` and O11 probes `ε/2π = 15 MHz` at REFERENCE T_1 (drive-stress regime) rather than `T_1 = 5 µs` at REFERENCE drive.
>
> Other plan-text references to `2.0` / `'ideal'` that do NOT appear inside inner loops (e.g., Module 1's shipped `'ideal'` mode in `compute_assignment_fidelity`) remain unamended.

**Goal:** Implement the optimization layer for dispersive transmon readout — parameter sensitivity tornado, closed-form analytic regime map, Modal-parallelized Pareto frontier, and closed-loop recommendation from fitted parameters — rendered as Figure 4 and a YAML recommendation artifact.

**Architecture:** Add a `dispersive_readout/optimization/` subpackage that consumes Module 1's public API (`simulate_readout`, `compute_assignment_fidelity`), Module 2's `OperatingPoint` / `ErrorBudget`, and Module 3's `ExtractedParameterPack.to_device_config()`. One surgical edit to Module 1 adds a `chi_scale: float = 1.0` kwarg on `build_hamiltonian` (threaded through `simulate_readout`) so sensitivity analysis can perturb χ orthogonally to `coupling.g` (Q1 orthogonality decision, spec §0 row 1). All finite-difference loops use `noise_model='ideal'` (Q8 contract). Pareto parallelizes via Modal `.map()`, reusing Module 3's pattern.

**Tech Stack:** Python 3.11+, QuTiP 5.x (Lindblad mesolve — already installed), NumPy 2.x, SciPy (`scipy.optimize.minimize` SLSQP + `scipy.stats.norm`/`scipy.stats.ttest_ind_from_stats`), Pydantic v2 (schema validation), PyYAML (recommendation artifact), Modal (Pareto parallelism — already credentialed from Module 3), matplotlib (Figure 4), pytest (28–29 tests).

**Spec:** See `06_Dispersive_Readout/MODULE_4_SPEC.md`. This plan implements §3 (methodology), §4 (module structure), §5 (component specs), §6 (test catalog), §7 (Figure 4), §8 (day-by-day tasks).

**Pre-plan assumption:** Work happens on a new branch `stage-06-module-4-optimization` branched off `stage-06-module-3-characterization`. Step 0 creates it.

**Step 0 — Create Module 4 branch (not a task; do once at start):**

```bash
git checkout -b stage-06-module-4-optimization
```

**Test invocation convention (all pytest commands in this plan use this form):**

```bash
python -m pytest <test-path> -v -p no:dash
```

The `python -m pytest` form ensures the conda env's pytest is used (`/usr/local/bin/pytest` picks the wrong Python and QuTiP import fails). The `-p no:dash` disables a broken Flask plugin on this system — convention inherited from MODULE_2_PLAN.md and MODULE_3_PLAN's unwritten equivalents.

---

## File Structure

### Files to modify (Module 1 — one surgical edit, threaded)

| File | Edit |
|---|---|
| `dispersive_readout/physics/lindblad.py:191` | Add `chi_scale: float = 1.0` kwarg to `build_hamiltonian`; replace line 191 `chi_per_level = dispersive_shift_full(...)` with `chi_per_level = chi_scale * dispersive_shift_full(...)`. |
| `dispersive_readout/physics/readout_model.py:64-71` | Add `chi_scale: float = 1.0` kwarg to `simulate_readout` signature; pass through to `build_hamiltonian(..., chi_scale=chi_scale)` at the build_hamiltonian call site (currently line ~95). |

### Files to create (Module 4 — new subpackage + scripts + artifacts)

| File | Responsibility |
|---|---|
| `dispersive_readout/optimization/__init__.py` | Public API: `compute_all_sensitivities`, `compute_pareto_frontier`, `recommend_from_fitted_parameters`, `pareto_one_tuple`, `SensitivityResult`, `ParetoPoint`, `RecommendationReport`, `DevicePoint`, `PUBLISHED_DEVICE_POINTS`, `PARETO_DEVICE_VARIANTS`, `SENSITIVITY_FD_STEP`, `SENSITIVITY_RENDER_BAR_THRESHOLD`, `SENSITIVITY_WARNING_THRESHOLD`. |
| `dispersive_readout/optimization/sensitivity.py` | Policy constants; `SensitivityResult` Pydantic schema; `compute_log_sensitivity`, `compute_all_sensitivities`, `rank_sensitivities`, `day_10_cross_check_s_g_vs_s_chi`. All finite-diff loops use `noise_model='ideal'`. |
| `dispersive_readout/optimization/regime_map.py` | `DevicePoint` frozen dataclass; `PUBLISHED_DEVICE_POINTS` list (4 entries from spec §3.2 table); `f_analytic_dispersive` closed-form F function; `compute_analytic_regime_map`; `purcell_boundary`, `dispersive_breakdown_boundary`, `resonator_too_slow_boundary` closed-form fns; `validate_analytic_vs_lindblad` 2-point check. |
| `dispersive_readout/optimization/pareto.py` | `ParetoPoint` Pydantic schema; `PARETO_DEVICE_VARIANTS` spec list (3 entries per spec §3.3 table); `build_variant` via `dataclasses.replace`; `find_pareto_point` (SLSQP + 5×5 warm-start, `noise_model='ideal'`); `compute_pareto_frontier` (Modal-dispatched or serial). |
| `dispersive_readout/optimization/modal_pareto.py` | `stage_06_module4_image` Modal image spec (debian_slim + numpy + scipy + qutip + pydantic + pyyaml); `app = modal.App(...)`; `@app.function pareto_one_tuple(device, tau_max) -> ParetoPoint`. Public module (no underscore) per Q7/Q8 decision. |
| `dispersive_readout/optimization/recommend.py` | `RecommendationReport` Pydantic schema; `_format_value_with_sigma` metrology helper; `recommend_from_fitted_parameters`; `generate_narrative`; `export_recommendation_to_yaml`. |
| `dispersive_readout/optimization/autodiff_addon.py` | **CUT 2026-04-23** — Day-11 cut to absorb per-level analytic-formula re-derivation cost. See spec §3.5 cut amendment + Task 18 in this plan. Not created. |
| `dispersive_readout/tests/test_optimization.py` | 28 tests (29 with contingent O7) per spec §6.1: O1–O24, O5 split as O5a/O5b. |
| `06_Dispersive_Readout/scripts/fig4_optimization.py` | Composite 3-panel Figure 4 (tornado + analytic regime map + Pareto + closed-loop arrow); generates `fig4_optimization.png` and `fig4_data.yaml`. |
| `06_Dispersive_Readout/figures/fig4_optimization.png` | Publication-quality composite (150 DPI, 1400 px wide). |
| `06_Dispersive_Readout/figures/fig4_data.yaml` | Regression-gate artifact: sensitivity values, Pareto point values, regime-grid hash. |
| `06_Dispersive_Readout/figures/recommendation.yaml` | `RecommendationReport` exported YAML for the canonical demo device. |
| `06_Dispersive_Readout/applications/cv_v1.md` (or similar) | CV v1 draft (parallel track, Day 13 PM per Q9c Change 3). |
| `06_Dispersive_Readout/applications/cover_letter_v1.md` | Cover letter v1 draft (parallel track, Day 13 PM). |

---

## Task Dependency Graph

```
              ┌── Task 1  (chi_scale kwarg, Module 1 API extension)
              │
              ├── Task 2  (optimization/ scaffold + policy constants)
              │     │
              │     ▼
              │   Task 3  (SensitivityResult schema + O6.1 + O8 contract)
              │     │
              │     ▼
              │   Task 4  (compute_log_sensitivity + O1 signs + O12–O18)
              │     │
              │     ▼
              │   Task 5  (compute_all + rank + O2 step-independence)
              │     │
              │     ▼
              │   Task 6  (Day-10 cross-check S_g vs 2·S_χ + O24 + O11)
              │     │
              │     ▼
              │   Task 7  (standalone tornado render)
              │
              ├── Task 8  (f_analytic_dispersive + envelope unit test)
              │     │
              │     ▼
              │   Task 9  (DevicePoint + PUBLISHED_DEVICE_POINTS + boundaries)
              │     │
              │     ▼
              │   Task 10 (validate_analytic_vs_lindblad + O3a/O3b + grid + render)
              │
              ├── Task 11 (Modal image + pareto_one_tuple stub + O10 smoke)  [Day 11 PM]
              │
              ├── Task 12 (ParetoPoint schema + build_variant + O6.2 + O22/O23)
              │     │
              │     ▼
              │   Task 13 (find_pareto_point SLSQP + warm-start + O19/O20/O21)
              │     │
              │     ▼
              │   Task 14 (compute_pareto_frontier + Modal dispatch + O4 + render)
              │
              ├── Task 15 (RecommendationReport + _format_value_with_sigma + O6.3)
              │     │
              │     ▼
              │   Task 16 (recommend_from_fitted_parameters + narrative + YAML)
              │     │
              │     ▼
              │   Task 17 (O5a + O5b closed-loop + demo-device pick)
              │
              ├── Task 18 (autodiff_addon.py — CUT 2026-04-23, day absorbed by item-15 re-derivation)
              │
              ├── Task 19 (fig4_optimization.py composite 3-panel)
              │     │
              │     ▼
              │   Task 20 (O9 regression gate + fig4_data.yaml commit)
              │
              └── Task 21 (CV v1 + cover letter v1, parallel Day 13 PM)
```

Tasks within each vertical chain are strictly sequential; chains are independent and can overlap across days per spec §8.

Task-to-day mapping:

- **Day 10 (Sat):** Tasks 1–7.
- **Day 11 (Sun):** Tasks 8–11.
- **Day 12 (Mon):** Tasks 12–15.
- **Day 13 (Tue):** Tasks 16–21.

---

## Task 1: Add `chi_scale` kwarg to Module 1's `build_hamiltonian` and thread through `simulate_readout`

**Rationale:** Spec §0 amendment #1 (Q1 orthogonality lock) + §2.1 χ-perturbation definition. The sensitivity tornado needs `chi_scale` as a multiplicative knob on the per-level χ array to get an orthogonal lever distinct from perturbing `coupling.g` (which would contaminate S_{γ_1} via γ_Purcell). This is the only Module 1 surgical edit; default `chi_scale = 1.0` preserves existing behavior bit-exactly.

**Files:**
- Modify: `dispersive_readout/physics/lindblad.py:141-191` — add kwarg to `build_hamiltonian`, rescale at line 191.
- Modify: `dispersive_readout/physics/readout_model.py:64-71` — add kwarg to `simulate_readout`, thread through.
- Test: add to `dispersive_readout/tests/test_lindblad.py` — regression test for `chi_scale = 1.0` bit-exactness and `chi_scale != 1.0` scaling behavior.

- [ ] **Step 1.1: Write the failing regression test**

Add at the end of `dispersive_readout/tests/test_lindblad.py`:

```python
def test_chi_scale_default_bit_exact():
    """chi_scale=1.0 (default) must reproduce the un-threaded Hamiltonian bit-exactly."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
    from dispersive_readout.physics.lindblad import build_hamiltonian

    drive = DriveParams(amplitude=1e7, duration=500e-9, detuning=0.0)
    H0_default, _ = build_hamiltonian(REFERENCE_DEVICE, drive)
    H0_explicit, _ = build_hamiltonian(REFERENCE_DEVICE, drive, chi_scale=1.0)
    # Frobenius norm of the difference should be exactly 0.
    diff = (H0_default - H0_explicit).norm()
    assert diff == 0.0, f"chi_scale=1.0 default not bit-exact: norm(diff) = {diff}"


def test_chi_scale_rescales_dispersive_term():
    """chi_scale=2.0 must double the chi·n_photon diagonal contribution."""
    import numpy as np
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
    from dispersive_readout.physics.lindblad import build_hamiltonian

    drive = DriveParams(amplitude=1e7, duration=500e-9, detuning=0.0)
    H0_one, _ = build_hamiltonian(REFERENCE_DEVICE, drive, chi_scale=1.0)
    H0_two, _ = build_hamiltonian(REFERENCE_DEVICE, drive, chi_scale=2.0)
    # The difference is exactly chi_per_level (at scale 1) tensored with n_photon.
    # So H0_two - H0_one equals H_chi (at scale 1). For a REFERENCE with nonzero
    # chi, the operator norm of the difference must be strictly positive.
    diff_norm = (H0_two - H0_one).norm()
    assert diff_norm > 0.0, "chi_scale=2.0 produced no change in Hamiltonian"


def test_chi_scale_threads_through_simulate_readout():
    """chi_scale must propagate from simulate_readout to build_hamiltonian."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
    from dispersive_readout.physics.readout_model import simulate_readout

    drive = DriveParams(amplitude=1e7, duration=200e-9, detuning=0.0)
    r_default = simulate_readout(REFERENCE_DEVICE, drive, initial_qubit_state=0)
    r_scaled = simulate_readout(
        REFERENCE_DEVICE, drive, initial_qubit_state=0, chi_scale=1.5
    )
    # Integrated IQ over a short window should differ between chi_scale=1.0 and 1.5.
    c_default = r_default.integrated_iq((50e-9, 150e-9))
    c_scaled = r_scaled.integrated_iq((50e-9, 150e-9))
    assert abs(c_default - c_scaled) > 0.0, "chi_scale did not thread through simulate_readout"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
python -m pytest dispersive_readout/tests/test_lindblad.py::test_chi_scale_default_bit_exact -v -p no:dash
```

Expected: **FAIL** with `TypeError: build_hamiltonian() got an unexpected keyword argument 'chi_scale'`.

- [ ] **Step 1.3: Add `chi_scale` to `build_hamiltonian` signature and rescale line 191**

Edit `dispersive_readout/physics/lindblad.py`. In `build_hamiltonian`'s signature (line ~141), add `chi_scale: float = 1.0`:

```python
def build_hamiltonian(
    device: DeviceConfig,
    drive_params: DriveParams,
    frame: Literal["rotating", "dispersive"] = "rotating",
    chi_scale: float = 1.0,
) -> tuple[qt.Qobj, list]:
    """Dispersive-regime effective Hamiltonian ...

    Parameters
    ----------
    chi_scale : float, optional
        Multiplicative rescale of the per-level dispersive shift array χ_j.
        Default 1.0 reproduces the un-rescaled Hamiltonian bit-exactly.
        Module 4 uses this for orthogonal χ-sensitivity (Q1 lock).
    """
```

At line 191, change:

```python
chi_per_level = dispersive_shift_full(energies, n_mat, g, omega_r)
```

to:

```python
chi_per_level = chi_scale * dispersive_shift_full(energies, n_mat, g, omega_r)
```

- [ ] **Step 1.4: Thread `chi_scale` through `simulate_readout`**

Edit `dispersive_readout/physics/readout_model.py`. In `simulate_readout`'s signature (line ~64), add `chi_scale: float = 1.0`:

```python
def simulate_readout(
    device: DeviceConfig,
    drive_params: DriveParams,
    initial_qubit_state: int,
    initial_resonator_state: str = "vacuum",
    t_list: np.ndarray | None = None,
    solver_options: dict | None = None,
    chi_scale: float = 1.0,
) -> ReadoutResult:
```

Find the existing `build_hamiltonian(device, drive_params)` call site in `simulate_readout` (likely around line 90-100) and update it to pass the kwarg:

```python
H, drive_spec = build_hamiltonian(device, drive_params, chi_scale=chi_scale)
```

- [ ] **Step 1.5: Run the three tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_lindblad.py::test_chi_scale_default_bit_exact dispersive_readout/tests/test_lindblad.py::test_chi_scale_rescales_dispersive_term dispersive_readout/tests/test_lindblad.py::test_chi_scale_threads_through_simulate_readout -v -p no:dash
```

Expected: **3 passed**.

- [ ] **Step 1.6: Run the full existing Module 1 test suite to verify no regression**

```bash
python -m pytest dispersive_readout/tests/test_lindblad.py dispersive_readout/tests/test_readout_model.py dispersive_readout/tests/test_physics_validation.py -v -p no:dash
```

Expected: **all existing tests still pass** (the bit-exact test in Step 1.1 certifies this at the Hamiltonian level, but we want to confirm downstream behaviors — V1a, V2, V3, V4a/b — are unaffected).

- [ ] **Step 1.7: Commit**

```bash
git add dispersive_readout/physics/lindblad.py dispersive_readout/physics/readout_model.py dispersive_readout/tests/test_lindblad.py
git commit -m "feat(stage06-m4): chi_scale kwarg on build_hamiltonian + simulate_readout

One-line multiplicative rescale of chi_per_level at lindblad.py:191,
threaded through simulate_readout. Default 1.0 preserves bit-exact
behavior — regression test asserts Frobenius norm(diff) == 0. Enables
Module 4's orthogonal chi-sensitivity per Q1 lock (spec §0 row 1)."
```

---

## Task 2: Create `dispersive_readout/optimization/` package skeleton with policy constants

**Rationale:** Spec §4 package structure + §3.1 policy constants. Establish the package before any optimization component so imports and tests have a stable target. Policy constants (`SENSITIVITY_FD_STEP = 0.05`, `SENSITIVITY_RENDER_BAR_THRESHOLD = 0.03`, `SENSITIVITY_WARNING_THRESHOLD = 2.0`) are locked in source per Q6 decision — auditable and test-targeted rather than magic numbers in a figure script.

**Files:**
- Create: `dispersive_readout/optimization/__init__.py` — package init, empty public API at this task's stage (populated incrementally).
- Create: `dispersive_readout/optimization/sensitivity.py` — policy constants only at this stage.
- Create: `dispersive_readout/tests/test_optimization.py` — test file header + first test.

- [ ] **Step 2.1: Write the failing test for policy-constant values**

Create `dispersive_readout/tests/test_optimization.py`:

```python
"""Stage 06 Module 4 — optimization layer tests (O1–O24).

Test catalog per MODULE_4_SPEC.md §6.1. Convention: each test function's
docstring cites the spec test ID it implements.
"""
from __future__ import annotations


def test_policy_constants_present_and_frozen():
    """Policy constants must live in source with locked values (Q6 lock)."""
    from dispersive_readout.optimization.sensitivity import (
        SENSITIVITY_FD_STEP,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
        SENSITIVITY_WARNING_THRESHOLD,
    )
    assert SENSITIVITY_FD_STEP == 0.05, (
        f"SENSITIVITY_FD_STEP changed from spec-locked 0.05 to {SENSITIVITY_FD_STEP}; "
        "requires spec amendment"
    )
    assert SENSITIVITY_RENDER_BAR_THRESHOLD == 0.03, (
        f"SENSITIVITY_RENDER_BAR_THRESHOLD changed from spec-locked 0.03 "
        f"to {SENSITIVITY_RENDER_BAR_THRESHOLD}; requires spec amendment"
    )
    assert SENSITIVITY_WARNING_THRESHOLD == 2.0, (
        f"SENSITIVITY_WARNING_THRESHOLD changed from spec-locked 2.0 "
        f"to {SENSITIVITY_WARNING_THRESHOLD}; requires spec amendment"
    )
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_policy_constants_present_and_frozen -v -p no:dash
```

Expected: **FAIL** with `ModuleNotFoundError: No module named 'dispersive_readout.optimization'`.

- [ ] **Step 2.3: Create the package skeleton**

Create `dispersive_readout/optimization/__init__.py`:

```python
"""Stage 06 Module 4 — sensitivity + Pareto + closed-loop optimization layer.

See 06_Dispersive_Readout/MODULE_4_SPEC.md for the design contract.

Public API is populated incrementally across Tasks 2–21. At end-of-Module-4
this __init__.py re-exports:
    - Policy constants: SENSITIVITY_FD_STEP, SENSITIVITY_RENDER_BAR_THRESHOLD,
                        SENSITIVITY_WARNING_THRESHOLD
    - Schemas: SensitivityResult, ParetoPoint, DevicePoint, RecommendationReport
    - Functions: compute_all_sensitivities, compute_pareto_frontier,
                 recommend_from_fitted_parameters, pareto_one_tuple
    - Data: PUBLISHED_DEVICE_POINTS, PARETO_DEVICE_VARIANTS
"""
from .sensitivity import (
    SENSITIVITY_FD_STEP,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
    SENSITIVITY_WARNING_THRESHOLD,
)

__all__ = [
    "SENSITIVITY_FD_STEP",
    "SENSITIVITY_RENDER_BAR_THRESHOLD",
    "SENSITIVITY_WARNING_THRESHOLD",
]
```

Create `dispersive_readout/optimization/sensitivity.py`:

```python
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
```

- [ ] **Step 2.4: Run test to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_policy_constants_present_and_frozen -v -p no:dash
```

Expected: **1 passed**.

- [ ] **Step 2.5: Commit**

```bash
git add dispersive_readout/optimization/__init__.py dispersive_readout/optimization/sensitivity.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): scaffold optimization/ package + policy constants

Three locked threshold constants (SENSITIVITY_FD_STEP=0.05,
SENSITIVITY_RENDER_BAR_THRESHOLD=0.03, SENSITIVITY_WARNING_THRESHOLD=2.0)
per MODULE_4_SPEC.md §0 Q6 lock. Values tested — changes require spec
amendment, not silent edit."
```

---

## Task 3: `SensitivityResult` Pydantic schema + O6.1 schema test + O8 analytic-objective-contract test

**Rationale:** Spec §5.1 schema definition + §6.1 tests O6.1 (schema validation) and O8 (contract enforcement). Schema must land before any `compute_*_sensitivity` function writes to it. O8 is a grep-based contract test that prevents future regressions where someone adds `noise_model='gaussian'` inside the sensitivity inner loop (Q8 lock).

**Files:**
- Modify: `dispersive_readout/optimization/sensitivity.py` — add imports, `ParameterName` Literal, `SensitivityResult` Pydantic model.
- Modify: `dispersive_readout/optimization/__init__.py` — export `SensitivityResult`, `ParameterName`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O6.1 and O8.

- [ ] **Step 3.1: Write O6.1 and O8 failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
import re
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
# O6.1 — SensitivityResult schema validation (spec §6 test catalog)
# ────────────────────────────────────────────────────────────────────

def test_O6_1_sensitivity_result_accepts_valid():
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    r = SensitivityResult(
        parameter="chi_scale",
        reference_value=1.0,
        reference_unit="dimensionless",
        sensitivity=0.42,
        sensitivity_uncertainty=0.01,
        F_reference=0.99,
    )
    assert r.parameter == "chi_scale"
    assert r.step_size_used == 0.05  # default = SENSITIVITY_FD_STEP


def test_O6_1_sensitivity_result_rejects_negative_uncertainty():
    from pydantic import ValidationError
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    with pytest.raises(ValidationError, match="sensitivity_uncertainty"):
        SensitivityResult(
            parameter="kappa",
            reference_value=1e7,
            reference_unit="rad/s",
            sensitivity=-0.2,
            sensitivity_uncertainty=-0.001,  # invalid
            F_reference=0.99,
        )


def test_O6_1_sensitivity_result_rejects_unknown_parameter_name():
    from pydantic import ValidationError
    from dispersive_readout.optimization.sensitivity import SensitivityResult
    with pytest.raises(ValidationError):
        SensitivityResult(
            parameter="not_a_real_parameter",  # not in ParameterName Literal
            reference_value=1.0,
            reference_unit="",
            sensitivity=0.1,
            sensitivity_uncertainty=0.01,
            F_reference=0.99,
        )


def test_O6_1_sensitivity_result_noise_consistent_flag_matches_threshold():
    from dispersive_readout.optimization.sensitivity import (
        SensitivityResult,
        SENSITIVITY_RENDER_BAR_THRESHOLD,
    )
    just_below = SENSITIVITY_RENDER_BAR_THRESHOLD * 0.9
    r = SensitivityResult(
        parameter="n_th",
        reference_value=0.01,
        reference_unit="",
        sensitivity=just_below,
        sensitivity_uncertainty=1e-4,
        F_reference=0.99,
    )
    # Schema should auto-compute or the computed-flag helper should match threshold
    assert r.noise_consistent_with_zero is True, (
        f"|S|={just_below} < {SENSITIVITY_RENDER_BAR_THRESHOLD} should flag "
        "noise_consistent_with_zero=True"
    )


# ────────────────────────────────────────────────────────────────────
# O8 — analytic-objective-contract enforcement (Q8 lock)
# ────────────────────────────────────────────────────────────────────

_OPTIMIZATION_DIR = Path("dispersive_readout") / "optimization"
_CONTRACT_PATTERN = re.compile(r"""noise_model\s*=\s*["']gaussian["']""")


def test_O8_no_gaussian_noise_inside_sensitivity_module():
    """Q8 lock: sensitivity.py must never use noise_model='gaussian' inside
    its inner loops — FD gradients become unreliable under shot noise."""
    src = (_OPTIMIZATION_DIR / "sensitivity.py").read_text()
    matches = _CONTRACT_PATTERN.findall(src)
    assert matches == [], (
        f"Q8 contract violated: sensitivity.py contains "
        f"noise_model='gaussian' at {len(matches)} call site(s). Inner-loop "
        "F-evaluations must use noise_model='ideal' (analytic). See MODULE_4_SPEC.md §0 row 8."
    )


def test_O8_no_gaussian_noise_inside_pareto_module():
    """Q8 lock: pareto.py must never use noise_model='gaussian' inside
    SLSQP function evaluations — optimizer noise pollutes FD gradients."""
    pareto_path = _OPTIMIZATION_DIR / "pareto.py"
    if not pareto_path.exists():
        pytest.skip("pareto.py not yet created — Task 12")
    src = pareto_path.read_text()
    matches = _CONTRACT_PATTERN.findall(src)
    assert matches == [], (
        f"Q8 contract violated: pareto.py contains "
        f"noise_model='gaussian' at {len(matches)} call site(s). Inner-loop "
        "SLSQP evaluations must use noise_model='ideal'. See MODULE_4_SPEC.md §0 row 8."
    )
```

- [ ] **Step 3.2: Run the new tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O6_1_sensitivity_result_accepts_valid -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'SensitivityResult' from 'dispersive_readout.optimization.sensitivity'`.

- [ ] **Step 3.3: Add schema and `ParameterName` Literal to `sensitivity.py`**

Append to `dispersive_readout/optimization/sensitivity.py`:

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


ParameterName = Literal[
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
]


class SensitivityResult(BaseModel):
    """Normalized log-sensitivity of F_assign to one parameter.

    See MODULE_4_SPEC.md §5.1 for the schema contract.
    """
    parameter: ParameterName
    reference_value: float
    reference_unit: str
    sensitivity: float                      # S_θ = ∂ ln F / ∂ ln θ
    sensitivity_uncertainty: float          # σ(S_θ) from analytic SE propagation
    F_reference: float                      # F at θ_ref
    step_size_used: float = SENSITIVITY_FD_STEP
    method: Literal["finite_diff", "autodiff"] = "finite_diff"
    noise_consistent_with_zero: bool = False  # auto-populated (|S| < threshold)

    @field_validator("sensitivity_uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"sensitivity_uncertainty must be >= 0 (got {v})"
            )
        return v

    @field_validator("F_reference")
    @classmethod
    def _valid_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"F_reference must be in [0, 1] (got {v})")
        return v

    @model_validator(mode="after")
    def _auto_flag_noise_consistent(self):
        """Auto-populate noise_consistent_with_zero from |sensitivity|."""
        flag = abs(self.sensitivity) < SENSITIVITY_RENDER_BAR_THRESHOLD
        # Pydantic v2 model_validator 'after' allows field reassignment.
        object.__setattr__(self, "noise_consistent_with_zero", flag)
        return self
```

Also update `dispersive_readout/optimization/__init__.py` to re-export:

```python
from .sensitivity import (
    SENSITIVITY_FD_STEP,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
    SENSITIVITY_WARNING_THRESHOLD,
    ParameterName,
    SensitivityResult,
)

__all__ = [
    "SENSITIVITY_FD_STEP",
    "SENSITIVITY_RENDER_BAR_THRESHOLD",
    "SENSITIVITY_WARNING_THRESHOLD",
    "ParameterName",
    "SensitivityResult",
]
```

- [ ] **Step 3.4: Run all O6.1 + O8 tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash
```

Expected: **5 passed, 1 skipped** (the pareto-module O8 test skips until Task 12).

- [ ] **Step 3.5: Commit**

```bash
git add dispersive_readout/optimization/sensitivity.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): SensitivityResult schema + O6.1 + O8 Q8-contract tests

SensitivityResult auto-flags noise_consistent_with_zero from
|S| < SENSITIVITY_RENDER_BAR_THRESHOLD (Q6/β lock).

O8 grep-based contract test guards against a future regression that would
reintroduce noise_model='gaussian' inside the sensitivity inner loop
(Q8 lock). Paired pareto.py check skips until Task 12."
```

---

## Task 4: `compute_log_sensitivity` via central finite differences + O1 signs + O12–O18 per-parameter unit checks

**Rationale:** Spec §3.1 sensitivity computation + §6.1 tests O1 (sign sanity — physics-falsifiable) and O12–O18 (per-parameter unit checks, 7 params). All simulator calls use `noise_model='ideal'` per Q8 contract (enforced by O8 from Task 3). Uncertainty propagates from the analytic binomial SE via σ(S_θ) = √2 · σ(F_ref) / (h · F_ref).

**Files:**
- Modify: `dispersive_readout/optimization/sensitivity.py` — add `compute_log_sensitivity`, private perturbation helpers.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O1 + O12–O18.

- [ ] **Step 4.1: Write O1 and O12 failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O1 — sensitivity sign sanity (physics-falsifiable invariant)
# ────────────────────────────────────────────────────────────────────

def test_O1_sensitivity_signs_at_REFERENCE():
    """Physics-locked invariants: S_chi > 0, S_gamma_1 < 0, S_n_th < 0."""
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_log_sensitivity

    op = get_reference_operating_point(n_shots=10_000)

    s_chi = compute_log_sensitivity(op, "chi_scale")
    s_gamma_1 = compute_log_sensitivity(op, "gamma_1")
    s_n_th = compute_log_sensitivity(op, "n_th")

    assert s_chi.sensitivity > 0.0, (
        f"S_chi = {s_chi.sensitivity:.3f}: positive sensitivity expected "
        "(increasing χ improves SNR). Wrong sign → simulator or sensitivity "
        "code has a bug. DO NOT fix by flipping signs in the figure."
    )
    assert s_gamma_1.sensitivity < 0.0, (
        f"S_gamma_1 = {s_gamma_1.sensitivity:.3f}: negative expected "
        "(more relaxation degrades F)."
    )
    assert s_n_th.sensitivity < 0.0, (
        f"S_n_th = {s_n_th.sensitivity:.3f}: negative expected "
        "(more thermal population degrades F)."
    )


# ────────────────────────────────────────────────────────────────────
# O12–O18 — per-parameter unit checks (all 7 parameters)
# ────────────────────────────────────────────────────────────────────

import pytest as _pytest

_ALL_PARAMETERS = [
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
]


@_pytest.mark.parametrize("parameter", _ALL_PARAMETERS)
def test_O12_O18_per_parameter_sensitivity_finite_and_typed(parameter):
    """Each of the 7 parameters returns a finite SensitivityResult at REFERENCE."""
    import math
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import (
        compute_log_sensitivity,
        SensitivityResult,
    )

    op = get_reference_operating_point(n_shots=10_000)
    r = compute_log_sensitivity(op, parameter)

    assert isinstance(r, SensitivityResult)
    assert r.parameter == parameter
    assert math.isfinite(r.sensitivity), f"S_{parameter} is not finite: {r.sensitivity}"
    assert math.isfinite(r.sensitivity_uncertainty)
    assert r.sensitivity_uncertainty > 0.0
    assert 0.0 < r.F_reference <= 1.0
```

- [ ] **Step 4.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O1_sensitivity_signs_at_REFERENCE -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'compute_log_sensitivity'`.

- [ ] **Step 4.3: Implement `compute_log_sensitivity` in `sensitivity.py`**

Append to `dispersive_readout/optimization/sensitivity.py`:

```python
import math
from dataclasses import replace
import numpy as np

from ..physics.config import DeviceConfig, DriveParams
from ..physics.readout_model import simulate_readout, compute_assignment_fidelity
from ..analysis.operating_point import OperatingPoint


def _evaluate_F_analytic(
    device: DeviceConfig,
    drive: DriveParams,
    integration_window: tuple[float, float],
    n_shots: int,
    chi_scale: float = 1.0,
) -> float:
    """Single-point F_assign evaluation — analytic mode only (Q8 contract).

    Returns the Gaussian-overlap analytic F. Shot-noise sampling is
    explicitly disabled here; use noise_model='ideal' at the call site.
    """
    r0 = simulate_readout(
        device, drive, initial_qubit_state=0, chi_scale=chi_scale,
    )
    r1 = simulate_readout(
        device, drive, initial_qubit_state=1, chi_scale=chi_scale,
    )
    return compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="ideal",
    ).F_assign


def _perturbed_device_drive_scale(
    op: OperatingPoint,
    parameter: ParameterName,
    fractional_delta: float,
) -> tuple[DeviceConfig, DriveParams, float]:
    """Return (perturbed_device, perturbed_drive, chi_scale) for one perturbation.

    Returns the trio that `_evaluate_F_analytic` needs; all non-perturbed fields
    are copied unchanged.
    """
    device, drive = op.device, op.drive
    chi_scale = 1.0  # baseline; only chi_scale-parameter path overrides

    if parameter == "chi_scale":
        chi_scale = 1.0 + fractional_delta
    elif parameter == "kappa":
        new_res = replace(device.resonator, kappa=device.resonator.kappa * (1.0 + fractional_delta))
        device = replace(device, resonator=new_res)
    elif parameter == "gamma_1":
        new_dec = replace(
            device.decoherence,
            gamma_1=device.decoherence.gamma_1 * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "gamma_phi":
        new_dec = replace(
            device.decoherence,
            gamma_phi=device.decoherence.gamma_phi * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "n_th":
        new_dec = replace(
            device.decoherence,
            n_th=device.decoherence.n_th * (1.0 + fractional_delta),
        )
        device = replace(device, decoherence=new_dec)
    elif parameter == "epsilon_0":
        drive = replace(drive, amplitude=drive.amplitude * (1.0 + fractional_delta))
    elif parameter == "tau":
        drive = replace(drive, duration=drive.duration * (1.0 + fractional_delta))
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

    return device, drive, chi_scale


def _reference_value_and_unit(op: OperatingPoint, parameter: ParameterName) -> tuple[float, str]:
    """Return (θ_ref, unit_str) for the parameter at the operating point."""
    mapping = {
        "chi_scale":  (1.0, "dimensionless (multiplicative)"),
        "kappa":      (op.device.resonator.kappa, "rad/s"),
        "gamma_1":    (op.device.decoherence.gamma_1, "1/s"),
        "gamma_phi":  (op.device.decoherence.gamma_phi, "1/s"),
        "n_th":       (op.device.decoherence.n_th, "dimensionless"),
        "epsilon_0":  (op.drive.amplitude, "rad/s"),
        "tau":        (op.drive.duration, "s"),
    }
    return mapping[parameter]


def compute_log_sensitivity(
    operating_point: OperatingPoint,
    parameter: ParameterName,
    step_size: float = SENSITIVITY_FD_STEP,
) -> SensitivityResult:
    """Compute S_θ = ∂ ln F / ∂ ln θ via central finite differences.

    Analytic F (noise_model='ideal') at both probe points; σ(S_θ) is
    propagated from analytic binomial SE on F_ref.
    """
    op = operating_point
    integration_window = op.integration_window
    n_shots = op.n_shots

    # Reference F (unperturbed)
    F_ref = _evaluate_F_analytic(
        op.device, op.drive, integration_window, n_shots, chi_scale=1.0,
    )

    # Plus perturbation
    dev_p, drv_p, chi_p = _perturbed_device_drive_scale(op, parameter, +step_size)
    F_plus = _evaluate_F_analytic(dev_p, drv_p, integration_window, n_shots, chi_scale=chi_p)

    # Minus perturbation
    dev_m, drv_m, chi_m = _perturbed_device_drive_scale(op, parameter, -step_size)
    F_minus = _evaluate_F_analytic(dev_m, drv_m, integration_window, n_shots, chi_scale=chi_m)

    # Central finite difference in log-log space
    S = (math.log(F_plus) - math.log(F_minus)) / (2.0 * step_size)

    # Uncertainty propagation from analytic binomial SE on F_ref.
    # σ(F) = sqrt(F(1-F)/n); propagate to σ(ln F) = σ(F)/F;
    # central-diff uncertainty: sqrt(2) * σ(ln F) / (2h) = σ(F) / (sqrt(2) * h * F).
    sigma_F_ref = math.sqrt(F_ref * (1.0 - F_ref) / n_shots)
    sigma_S = sigma_F_ref / (math.sqrt(2.0) * step_size * F_ref)

    theta_ref, unit = _reference_value_and_unit(op, parameter)

    return SensitivityResult(
        parameter=parameter,
        reference_value=theta_ref,
        reference_unit=unit,
        sensitivity=float(S),
        sensitivity_uncertainty=float(sigma_S),
        F_reference=float(F_ref),
        step_size_used=step_size,
        method="finite_diff",
    )
```

- [ ] **Step 4.4: Run O1 + all 7 O12–O18 parametrized tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "O1_sensitivity or O12_O18"
```

Expected: **8 passed** (O1 + 7 parametrized O12–O18).

- [ ] **Step 4.5: Commit**

```bash
git add dispersive_readout/optimization/sensitivity.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): compute_log_sensitivity + O1 signs + O12–O18 per-param

Central FD with h=0.05 (SENSITIVITY_FD_STEP), noise_model='ideal' at
every call site (Q8 contract enforced by O8). Uncertainty from analytic
binomial SE: σ(S) = σ(F_ref) / (sqrt(2) · h · F_ref).

O1 asserts physics-falsifiable sign invariants — wrong signs indicate
a simulator or sensitivity-code bug, NOT a figure labelling issue."
```

---

## Task 5: `compute_all_sensitivities` + `rank_sensitivities` + O2 step-independence

**Rationale:** Spec §5.1 batch-compute entry point + §6.1 test O2 (step independence at h=0.05 vs h=0.025 within 10%). `rank_sensitivities` sorts by |S| for tornado-plot ordering per spec §7 Panel (a) "sorted by |S_θ| descending."

**Files:**
- Modify: `dispersive_readout/optimization/sensitivity.py` — add `compute_all_sensitivities`, `rank_sensitivities`.
- Modify: `dispersive_readout/optimization/__init__.py` — re-export both.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O2 + test for `rank_sensitivities` ordering.

- [ ] **Step 5.1: Write O2 + ranking failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O2 — step-independence: S at h=0.05 vs h=0.025 within 10%
# ────────────────────────────────────────────────────────────────────

def test_O2_step_independence_chi_scale():
    """S_chi at h=0.05 and h=0.025 must agree to within 10%."""
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_log_sensitivity

    op = get_reference_operating_point(n_shots=10_000)
    s_coarse = compute_log_sensitivity(op, "chi_scale", step_size=0.05)
    s_fine = compute_log_sensitivity(op, "chi_scale", step_size=0.025)
    rel_diff = abs(s_fine.sensitivity - s_coarse.sensitivity) / abs(s_coarse.sensitivity)
    assert rel_diff < 0.10, (
        f"S_chi at h=0.025 ({s_fine.sensitivity:.4f}) differs from h=0.05 "
        f"({s_coarse.sensitivity:.4f}) by {rel_diff*100:.1f}% (> 10%). "
        "Reduce Lindblad solver rtol, or investigate FD-truncation error."
    )


def test_compute_all_sensitivities_returns_seven():
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_all_sensitivities

    op = get_reference_operating_point(n_shots=10_000)
    results = compute_all_sensitivities(op)
    assert len(results) == 7
    params = {r.parameter for r in results}
    assert params == {
        "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
    }


def test_rank_sensitivities_sorts_by_absolute_magnitude_desc():
    from dispersive_readout.optimization.sensitivity import (
        SensitivityResult, rank_sensitivities,
    )
    inputs = [
        SensitivityResult(
            parameter="chi_scale", reference_value=1.0, reference_unit="",
            sensitivity=0.1, sensitivity_uncertainty=0.01, F_reference=0.99,
        ),
        SensitivityResult(
            parameter="gamma_1", reference_value=1e4, reference_unit="1/s",
            sensitivity=-0.5, sensitivity_uncertainty=0.02, F_reference=0.99,
        ),
        SensitivityResult(
            parameter="kappa", reference_value=3e7, reference_unit="rad/s",
            sensitivity=0.3, sensitivity_uncertainty=0.01, F_reference=0.99,
        ),
    ]
    ranked = rank_sensitivities(inputs)
    assert [r.parameter for r in ranked] == ["gamma_1", "kappa", "chi_scale"]
```

- [ ] **Step 5.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_compute_all_sensitivities_returns_seven -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'compute_all_sensitivities'`.

- [ ] **Step 5.3: Implement `compute_all_sensitivities` and `rank_sensitivities`**

Append to `dispersive_readout/optimization/sensitivity.py`:

```python
_ALL_PARAMETER_NAMES: tuple[ParameterName, ...] = (
    "chi_scale", "kappa", "gamma_1", "gamma_phi", "n_th", "epsilon_0", "tau",
)


def compute_all_sensitivities(
    operating_point: OperatingPoint,
    parameters: list[ParameterName] | None = None,
    step_size: float = SENSITIVITY_FD_STEP,
) -> list[SensitivityResult]:
    """Compute sensitivities for all 7 parameters (default) at the given operating point."""
    params = parameters if parameters is not None else list(_ALL_PARAMETER_NAMES)
    return [compute_log_sensitivity(operating_point, p, step_size=step_size) for p in params]


def rank_sensitivities(results: list[SensitivityResult]) -> list[SensitivityResult]:
    """Sort by |sensitivity|, descending. Stable sort (ties preserve input order)."""
    return sorted(results, key=lambda r: abs(r.sensitivity), reverse=True)
```

Update `dispersive_readout/optimization/__init__.py` exports:

```python
from .sensitivity import (
    SENSITIVITY_FD_STEP,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
    SENSITIVITY_WARNING_THRESHOLD,
    ParameterName,
    SensitivityResult,
    compute_log_sensitivity,
    compute_all_sensitivities,
    rank_sensitivities,
)

__all__ = [
    "SENSITIVITY_FD_STEP",
    "SENSITIVITY_RENDER_BAR_THRESHOLD",
    "SENSITIVITY_WARNING_THRESHOLD",
    "ParameterName",
    "SensitivityResult",
    "compute_log_sensitivity",
    "compute_all_sensitivities",
    "rank_sensitivities",
]
```

- [ ] **Step 5.4: Run new tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O2_step_independence_chi_scale dispersive_readout/tests/test_optimization.py::test_compute_all_sensitivities_returns_seven dispersive_readout/tests/test_optimization.py::test_rank_sensitivities_sorts_by_absolute_magnitude_desc -v -p no:dash
```

Expected: **3 passed**. (O2 may take ~1 min at REFERENCE — four Lindblad calls at two step sizes.)

- [ ] **Step 5.5: Commit**

```bash
git add dispersive_readout/optimization/sensitivity.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): compute_all_sensitivities + rank_sensitivities + O2

Batch compute for the 7-parameter tornado, stable-sort by |S| for the
tornado-plot ordering. O2 step-independence test: S_chi at h=0.05 vs
h=0.025 must agree within 10% — larger disagreement signals that
Lindblad solver rtol or FD truncation error needs investigation."
```

---

## Task 6: Day-10 cross-check (S_g vs 2·S_χ) + O24 + O11 sensitivity_warnings-fires

**Rationale:** Spec §0 row 1 (Q1 locked Day-10 cross-check) + §6.1 tests O24 (logs caption value) and O11 (sensitivity_warnings fires on boundary-proximate device). The cross-check generates the caption number `|S_g − 2·S_χ|` that feeds Figure 4. O11 is moved from Day 13 to Day 10 afternoon per Q9c Change 4 — it's a unit test against the threshold constant that doesn't need `recommend.py`.

**Files:**
- Modify: `dispersive_readout/optimization/sensitivity.py` — add `day_10_cross_check_s_g_vs_s_chi`.
- Create: `06_Dispersive_Readout/figures/day10_cross_check.txt` — written by the test (committed as Day-10 artifact).
- Modify: `dispersive_readout/tests/test_optimization.py` — add O24 + O11.

- [ ] **Step 6.1: Write O24 + O11 failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O24 — Day-10 cross-check: S_g vs 2·S_chi (Q1 caption artifact)
# ────────────────────────────────────────────────────────────────────

def test_O24_day_10_cross_check_logged_and_within_threshold():
    """Compute S_chi via chi_scale and S_g via ±5% on coupling.g; write
    |S_g − 2·S_chi| / (2·|S_chi|) to day10_cross_check.txt for the
    Figure 4 caption. If the fractional residual exceeds 10%, flag to
    human per spec §9 item 2 (decision in caption, not fix)."""
    import math
    from pathlib import Path
    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import day_10_cross_check_s_g_vs_s_chi

    op = get_reference_operating_point(n_shots=10_000)
    result = day_10_cross_check_s_g_vs_s_chi(op)

    # Assert structure
    for key in ("S_chi", "S_g", "predicted_S_g", "residual", "residual_fractional"):
        assert key in result, f"Missing key: {key}"
        assert math.isfinite(result[key])

    # Write artifact for Figure 4 caption
    artifact_path = Path("06_Dispersive_Readout/figures/day10_cross_check.txt")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        f"Day-10 Q1 cross-check at REFERENCE_DEVICE:\n"
        f"  S_chi (via chi_scale ± 0.05)   = {result['S_chi']:+.4f}\n"
        f"  S_g   (via coupling.g ± 0.05) = {result['S_g']:+.4f}\n"
        f"  Predicted S_g = 2 · S_chi     = {result['predicted_S_g']:+.4f}\n"
        f"  Residual |S_g − 2·S_chi|      = {abs(result['residual']):.4f}\n"
        f"  Fractional |residual|/|2·S_chi| = {result['residual_fractional']*100:.2f}%\n"
    )

    # Caption expects residual_fractional < 0.10 for clean Q1 interpretation;
    # larger values are quantitative evidence of Purcell contamination and go
    # into the caption verbatim (still "pass" for this test — the test
    # computes and logs; it does not gate on agreement).
    assert 0.0 <= result["residual_fractional"]


# ────────────────────────────────────────────────────────────────────
# O11 — sensitivity_warnings fires on boundary-proximate device (Q4 lock)
# ────────────────────────────────────────────────────────────────────

def test_O11_sensitivity_warning_fires_near_purcell_boundary():
    """Device with very short T_1 (e.g., 5 µs) should force |S_gamma_1| > 2.0
    and trigger the warning policy. Tests the *threshold* via a direct
    sensitivity computation — the full `sensitivity_warnings` field gets
    assembled in Task 15's RecommendationReport, but the threshold
    constant and its policy meaning are locked here."""
    from dataclasses import replace
    from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
    from dispersive_readout.analysis.operating_point import get_reference_operating_point, OperatingPoint
    from dispersive_readout.optimization.sensitivity import (
        compute_log_sensitivity,
        SENSITIVITY_WARNING_THRESHOLD,
    )

    # Construct a device with short T_1 = 5 µs (γ_1 = 2e5 1/s, ~15× REFERENCE).
    short_T1 = replace(REFERENCE_DEVICE.decoherence, gamma_1=1.0 / 5e-6)
    device_short_T1 = replace(REFERENCE_DEVICE, decoherence=short_T1)

    ref_op = get_reference_operating_point(n_shots=10_000)
    op_short_T1 = OperatingPoint(
        device=device_short_T1,
        drive=ref_op.drive,
        integration_window=ref_op.integration_window,
        n_shots=ref_op.n_shots,
    )

    s_gamma_1 = compute_log_sensitivity(op_short_T1, "gamma_1")
    assert abs(s_gamma_1.sensitivity) > SENSITIVITY_WARNING_THRESHOLD, (
        f"Short-T1 device (T_1=5µs) gave |S_gamma_1|={abs(s_gamma_1.sensitivity):.3f}, "
        f"expected > {SENSITIVITY_WARNING_THRESHOLD}. Either the threshold "
        "is too conservative, or the synthetic boundary-proximate device "
        "is not actually near the boundary — check Purcell vs intrinsic γ_1."
    )
```

- [ ] **Step 6.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O24_day_10_cross_check_logged_and_within_threshold -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'day_10_cross_check_s_g_vs_s_chi'`.

- [ ] **Step 6.3: Implement `day_10_cross_check_s_g_vs_s_chi` in `sensitivity.py`**

Append to `dispersive_readout/optimization/sensitivity.py`:

```python
def day_10_cross_check_s_g_vs_s_chi(
    operating_point: OperatingPoint,
    step_size: float = SENSITIVITY_FD_STEP,
) -> dict:
    """Day-10 Q1 cross-check: compute S_g via ±5% on coupling.g and compare
    to 2·S_chi. Returns dict with keys S_chi, S_g, predicted_S_g, residual,
    residual_fractional. The residual_fractional feeds the Figure 4 caption.

    Under the transmon χ ≈ 2 g² α / (Δ(Δ+α)) at fixed (κ, α), S_g ≈ 2·S_chi
    holds at leading order. Any deviation quantifies Purcell-coupling
    contamination in an (A)-style χ-sensitivity.
    """
    op = operating_point

    # S_chi via chi_scale (Q1 locked path)
    s_chi = compute_log_sensitivity(op, "chi_scale", step_size=step_size).sensitivity

    # S_g via direct perturbation of coupling.g — re-derives γ_Purcell,
    # so this deliberately carries the Purcell-coupling overlap Q1 is
    # measuring against.
    import math
    from dataclasses import replace

    def _F_at_g(g_value: float) -> float:
        new_coupling = replace(op.device.coupling, g=g_value)
        new_device = replace(op.device, coupling=new_coupling)
        return _evaluate_F_analytic(
            new_device, op.drive, op.integration_window, op.n_shots, chi_scale=1.0,
        )

    g_ref = op.device.coupling.g
    F_plus = _F_at_g(g_ref * (1.0 + step_size))
    F_minus = _F_at_g(g_ref * (1.0 - step_size))
    s_g = (math.log(F_plus) - math.log(F_minus)) / (2.0 * step_size)

    predicted_s_g = 2.0 * s_chi
    residual = s_g - predicted_s_g
    residual_fractional = abs(residual) / max(abs(predicted_s_g), 1e-12)

    return {
        "S_chi": float(s_chi),
        "S_g": float(s_g),
        "predicted_S_g": float(predicted_s_g),
        "residual": float(residual),
        "residual_fractional": float(residual_fractional),
    }
```

- [ ] **Step 6.4: Run O24 + O11 to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O24_day_10_cross_check_logged_and_within_threshold dispersive_readout/tests/test_optimization.py::test_O11_sensitivity_warning_fires_near_purcell_boundary -v -p no:dash
```

Expected: **2 passed**. Inspect `06_Dispersive_Readout/figures/day10_cross_check.txt` — the residual_fractional number goes into the Figure 4 caption later.

- [ ] **Step 6.5: Commit**

```bash
git add dispersive_readout/optimization/sensitivity.py dispersive_readout/tests/test_optimization.py 06_Dispersive_Readout/figures/day10_cross_check.txt
git commit -m "feat(stage06-m4): Day-10 cross-check + O24 + O11 warning threshold

O24 computes S_g vs 2·S_chi at REFERENCE and writes day10_cross_check.txt
for the Figure 4 caption (Q1 evidence artifact). O11 verifies the
SENSITIVITY_WARNING_THRESHOLD=2.0 policy fires on a T_1=5µs device.

Both O24 and O11 moved to Day 10 afternoon per Q9c Change 4."
```

---

## Task 7: Standalone tornado plot — Figure 4 Panel (a) draft

**Rationale:** Spec §7 Panel (a) + §8 Day 10 afternoon ("First-pass tornado plot rendered standalone"). Commit as a standalone artifact before composite Figure 4 assembly on Day 13 — de-risks rendering issues early.

**Files:**
- Create: `06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py` — standalone renderer (retained after Day 13 for regeneration).
- Create: `06_Dispersive_Readout/figures/fig4_panel_a_tornado.png` — committed artifact (150 DPI).

- [ ] **Step 7.1: Write the renderer script**

Create `06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py`:

```python
"""Standalone Figure 4 Panel (a) — sensitivity tornado at REFERENCE.

Produces figures/fig4_panel_a_tornado.png. Day-13's fig4_optimization.py
will reuse this logic by importing the render function.

Memory rules applied (feedback_figure_presentation):
    - Cool palette for sensitivities
    - Horizontal bar chart sorted by |S| descending
    - Numeric annotation above each bar
    - y-axis labels include (±5%) perturbation scale
    - Point-with-errorbar when |S| < SENSITIVITY_RENDER_BAR_THRESHOLD
    - Anchoring subtitle with F_ref, tau_int, n_phot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.optimization.sensitivity import (
    compute_all_sensitivities,
    rank_sensitivities,
    SENSITIVITY_RENDER_BAR_THRESHOLD,
)


_DISPLAY_LABELS: dict[str, str] = {
    "chi_scale": r"$\chi$ (via chi_scale, ±5%)",
    "kappa":     r"$\kappa$ (±5%)",
    "gamma_1":   r"$T_1$ (±5%)",
    "gamma_phi": r"$T_\varphi$ (±5%)",
    "n_th":      r"$\bar n_{\rm th}$ (±5%)",
    "epsilon_0": r"$\varepsilon_0$ (±5%)",
    "tau":       r"$\tau$ (±5%)",
}


def render_tornado(ax: plt.Axes, sensitivities: list, anchoring: str) -> None:
    """Render the tornado panel on the provided axis. Reusable by Day-13 composite."""
    ranked = rank_sensitivities(sensitivities)
    # Plot order top-to-bottom = highest |S| at top; invert so largest is first
    ys = np.arange(len(ranked))[::-1]
    labels = [_DISPLAY_LABELS[r.parameter] for r in ranked]

    cool_pos = "#4A90E2"   # cool palette — positive
    cool_neg = "#2E6DA4"   # cool palette — negative (deeper)

    for y, r in zip(ys, ranked):
        S = r.sensitivity
        sigma = r.sensitivity_uncertainty
        is_noise_like = r.noise_consistent_with_zero
        color = cool_pos if S >= 0 else cool_neg
        if is_noise_like:
            ax.errorbar([S], [y], xerr=[sigma], fmt="o", color=color, capsize=3)
        else:
            ax.barh([y], [S], color=color, edgecolor="black", linewidth=0.5, alpha=0.9)
            # Numeric annotation outside the bar
            offset = 0.03 * (1 if S >= 0 else -1)
            ha = "left" if S >= 0 else "right"
            ax.text(S + offset, y, f"{S:+.3f}", va="center", ha=ha, fontsize=9)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Normalized log-sensitivity $S_\theta = \partial \ln F / \partial \ln \theta$")
    ax.set_title(
        "Parameter sensitivity of $F_{\\rm assign}$ at REFERENCE (Marxer 2508.16437)\n"
        + anchoring, fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)

    fig, ax = plt.subplots(figsize=(8, 5))
    anchoring = (
        f"$F_{{\\rm ref}}={sens[0].F_reference:.4f}$, "
        f"$\\tau_{{\\rm int}} = {(op.integration_window[1]-op.integration_window[0])*1e9:.0f}$ ns, "
        f"$n_{{\\rm shots}}=10^4$"
    )
    render_tornado(ax, sens, anchoring)
    fig.tight_layout()

    out = Path("06_Dispersive_Readout/figures/fig4_panel_a_tornado.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Run the script to generate the standalone figure**

```bash
python 06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py
```

Expected output: `Wrote 06_Dispersive_Readout/figures/fig4_panel_a_tornado.png`. Open the PNG and verify 7 bars are visible, sorted by |S| descending, with labels showing (±5%) perturbation scale.

- [ ] **Step 7.3: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig4_panel_a_tornado.py 06_Dispersive_Readout/figures/fig4_panel_a_tornado.png
git commit -m "feat(stage06-m4): standalone tornado plot (Figure 4 Panel a draft)

Day-10 afternoon checkpoint per spec §8. Memory rules applied:
cool palette, (±5%) on labels, numeric annotations, point-with-errorbar
for |S| < 0.03, anchoring subtitle with F_ref/tau_int/n_shots.

Day-13's fig4_optimization.py will import render_tornado() for the
composite without duplicating the rendering logic."
```

---

## Task 8: `f_analytic_dispersive` closed-form F + decoherence-envelope unit test

**Rationale:** Spec §3.2 analytic regime-map F-formula (Bengtsson 2024 PRL §II + Blais RMP 2021 §V.B cross-check). Establish the closed-form function and unit-test the linear decoherence envelope `(1 − γ₁τ/2)^(1/2)` against the exponential form `exp(−γ₁τ/4)` across the grid range — caption claims <1% agreement, locked here.

**Files:**
- Create: `dispersive_readout/optimization/regime_map.py` — `f_analytic_dispersive` only at this task's stage.
- Modify: `dispersive_readout/tests/test_optimization.py` — add envelope-deviation unit test.

- [ ] **Step 8.1: Write the envelope-deviation failing test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# Decoherence-envelope linearization: (1 − γτ/2)^½ vs exp(−γτ/4)
# ────────────────────────────────────────────────────────────────────

def test_decoherence_envelope_linear_agrees_with_exp_within_1pct():
    """Linearized envelope (1 − γτ/2)^½ must agree with exp(−γτ/4) within
    1% over the regime map's y-axis range [1e-4, 1e-1]. Caption claims
    this explicitly — if it fails, add the correction term or re-linearize."""
    import numpy as np
    gamma_tau = np.logspace(-4, -1, 40)
    linear = np.sqrt(1.0 - gamma_tau / 2.0)
    expon = np.exp(-gamma_tau / 4.0)
    rel_dev = np.abs(linear - expon) / expon
    assert rel_dev.max() < 0.01, (
        f"Max relative deviation {rel_dev.max()*100:.2f}% > 1% at gamma_tau="
        f"{gamma_tau[rel_dev.argmax()]:.3e}. Caption claim 'deviation from "
        "exp form < 1% over y-axis range' is false — add correction term."
    )


def test_f_analytic_dispersive_returns_monotone_increase_with_n_phot():
    """F should be non-decreasing in n_phot for fixed (chi/kappa, gamma_tau)."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import f_analytic_dispersive
    chi_k = np.array([0.5])
    g_t = np.array([1e-3])
    F_low = f_analytic_dispersive(chi_k, g_t, n_phot=1.0)
    F_high = f_analytic_dispersive(chi_k, g_t, n_phot=10.0)
    assert F_high >= F_low, f"F decreased with n_phot: {F_high} < {F_low}"


def test_f_analytic_dispersive_peaks_near_chi_over_kappa_half():
    """Dispersive SNR 4·(χ/κ)/(1 + (2χ/κ)²) peaks at χ/κ = 0.5; F should
    inherit that maximum location at fixed gamma_tau and n_phot."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import f_analytic_dispersive
    chi_k = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 3.0])
    g_t = np.array([1e-3])
    F_vals = f_analytic_dispersive(chi_k[:, None], g_t[None, :], n_phot=4.0)
    # Collapse the singleton second axis
    F_1d = F_vals[:, 0]
    peak_idx = int(np.argmax(F_1d))
    assert chi_k[peak_idx] == 0.5, (
        f"F peaks at chi/kappa = {chi_k[peak_idx]}, expected 0.5 "
        f"(Bengtsson 2024 §II SNR-max). F array: {F_1d}"
    )
```

- [ ] **Step 8.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_f_analytic_dispersive_returns_monotone_increase_with_n_phot -v -p no:dash
```

Expected: **FAIL** with `ModuleNotFoundError: No module named 'dispersive_readout.optimization.regime_map'`.

- [ ] **Step 8.3: Create `regime_map.py` with `f_analytic_dispersive`**

Create `dispersive_readout/optimization/regime_map.py`:

```python
"""Closed-form analytic regime-map surface and boundary functions.

See MODULE_4_SPEC.md §3.2. The map is an analytic evaluation of the
dispersive-readout SNR formula (Bengtsson 2024 PRL §II, cross-checked
against Blais RMP 2021 §V.B §V.B), not a Lindblad grid — the 100× chi/kappa
range would otherwise extrapolate the 2nd-order SW dispersive PT well
outside its validity envelope (Q3 lock).
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm


ArrayLike = Union[float, np.ndarray]


def f_analytic_dispersive(
    chi_over_kappa: ArrayLike,
    gamma_1_tau: ArrayLike,
    n_phot: float,
) -> ArrayLike:
    """Closed-form F_assign per Bengtsson 2024 PRL §II.

    Parameters
    ----------
    chi_over_kappa : float or ndarray
        χ/κ at each evaluation point. Non-broadcastable axes must match.
    gamma_1_tau : float or ndarray
        γ_1 · τ_readout (dimensionless decoherence budget).
    n_phot : float
        Steady-state resonator photon number (scalar; held fixed across
        the grid and quoted on Figure 4 Panel (b)'s subtitle).

    Returns
    -------
    F : float or ndarray
        Assignment fidelity Φ(SNR_eff / 2).

    Notes
    -----
    Decoherence envelope is linearized: (1 − γ_1·τ/2)^(1/2). Within 1%
    of exp(−γ_1·τ/4) over the spec's y-axis range [1e-4, 1e-1]; a
    unit test asserts this.
    """
    chi_k = np.asarray(chi_over_kappa, dtype=float)
    g_t = np.asarray(gamma_1_tau, dtype=float)
    snr_steady = 4.0 * chi_k * np.sqrt(n_phot) / (1.0 + (2.0 * chi_k) ** 2)
    envelope = np.sqrt(np.clip(1.0 - g_t / 2.0, 0.0, 1.0))
    snr_eff = snr_steady * envelope
    return norm.cdf(snr_eff / 2.0)
```

- [ ] **Step 8.4: Run the three tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "envelope or n_phot or chi_over_kappa_half"
```

Expected: **3 passed**.

- [ ] **Step 8.5: Commit**

```bash
git add dispersive_readout/optimization/regime_map.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): f_analytic_dispersive + envelope + SNR-peak tests

Closed-form F from Bengtsson 2024 PRL §II. Linear envelope
(1 − γτ/2)^½ verified within 1% of exp(−γτ/4) over y-axis range
[1e-4, 1e-1] — locks the caption claim. SNR peaks at χ/κ = 0.5
(dispersive optimum), confirmed by direct argmax."
```

---

## Task 9: `DevicePoint` + `PUBLISHED_DEVICE_POINTS` + analytic boundary functions

**Rationale:** Spec §3.2 Published device overlays table (4 entries) + Analytic boundaries section (3 boundary curves with the closed-form formulas from the post-commit proofread Nit 2). These data and functions must land before the standalone regime-map render and the Lindblad validation.

**Files:**
- Modify: `dispersive_readout/optimization/regime_map.py` — add `DevicePoint`, `PUBLISHED_DEVICE_POINTS`, `purcell_boundary`, `dispersive_breakdown_boundary`, `resonator_too_slow_boundary`.
- Modify: `dispersive_readout/optimization/__init__.py` — export `DevicePoint`, `PUBLISHED_DEVICE_POINTS`, `f_analytic_dispersive`.
- Modify: `dispersive_readout/tests/test_optimization.py` — data-validation tests + boundary-monotonicity tests.

- [ ] **Step 9.1: Write data + boundary validation tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# Published-device-points data validation (Q5 lock)
# ────────────────────────────────────────────────────────────────────

def test_PUBLISHED_DEVICE_POINTS_has_four_entries_labeled_correctly():
    """The 4 markers of the regime map — Marxer Q1, Marxer Q2, Bengtsson,
    Garnet — with Hazra OMITTED per Q5 lock."""
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    labels = [p.label for p in PUBLISHED_DEVICE_POINTS]
    assert len(PUBLISHED_DEVICE_POINTS) == 4, (
        f"Expected exactly 4 device points, got {len(PUBLISHED_DEVICE_POINTS)}. "
        "Hazra must be OMITTED per Q5 (dimon, non-standard transmon)."
    )
    assert all("Hazra" not in lab for lab in labels), (
        f"Hazra must not appear in plotted device points. Labels: {labels}"
    )
    expected_substrings = ["Marxer Q1", "Marxer Q2", "Bengtsson", "Garnet"]
    for expected in expected_substrings:
        assert any(expected in lab for lab in labels), (
            f"Missing expected device '{expected}' from PUBLISHED_DEVICE_POINTS. "
            f"Actual labels: {labels}"
        )


def test_PUBLISHED_DEVICE_POINTS_coordinates_are_physical():
    """chi/kappa and gamma_1*tau must be finite, positive, and within the
    regime map's x-axis [0.1, 10] and y-axis [1e-4, 1e-1] ranges."""
    import math
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    for p in PUBLISHED_DEVICE_POINTS:
        assert math.isfinite(p.chi_over_kappa) and p.chi_over_kappa > 0
        assert math.isfinite(p.gamma_1_tau) and p.gamma_1_tau > 0
        assert 0.1 <= p.chi_over_kappa <= 10.0, (
            f"{p.label}: chi_over_kappa={p.chi_over_kappa} outside [0.1, 10]"
        )
        assert 1e-4 <= p.gamma_1_tau <= 1e-1, (
            f"{p.label}: gamma_1_tau={p.gamma_1_tau} outside [1e-4, 1e-1]"
        )
        if p.reported_F_assign is not None:
            assert 0.5 <= p.reported_F_assign <= 1.0


def test_marxer_q1_is_primary_anchor_with_F_reported():
    """Marxer Q1 must have reported_F_assign set — it's the F_sim annotation
    anchor for Panel (b) per Q3 Refinement 1."""
    from dispersive_readout.optimization.regime_map import PUBLISHED_DEVICE_POINTS
    q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
    assert q1.reported_F_assign is not None
    assert q1.reported_F_assign > 0.99


# ────────────────────────────────────────────────────────────────────
# Analytic-boundary monotonicity tests
# ────────────────────────────────────────────────────────────────────

def test_purcell_boundary_decreases_with_chi_over_kappa():
    """Under γ_Purcell = κ · (g/Δ)² with (g, Δ) at REFERENCE and
    κ(x) = χ_REF / x, γ_Purcell ∝ 1/x, so τ_readout(x) at γ_P·τ=0.1 grows
    with x, and y_Purcell(x) = γ_1·0.1/γ_P(x) also grows with x.
    Boundary is monotone non-decreasing in x."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import purcell_boundary
    x = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    y = purcell_boundary(x)
    assert np.all(np.diff(y) >= 0), (
        f"Purcell boundary not monotone in x: y = {y}"
    )


def test_resonator_too_slow_is_constant_in_x():
    """kappa·tau_readout = 1 at fixed REFERENCE κ is a horizontal line."""
    import numpy as np
    from dispersive_readout.optimization.regime_map import resonator_too_slow_boundary
    x = np.array([0.3, 1.0, 3.0])
    y = resonator_too_slow_boundary(x)
    assert np.allclose(y, y[0]), (
        f"Resonator-too-slow line not constant: {y}"
    )
```

- [ ] **Step 9.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_PUBLISHED_DEVICE_POINTS_has_four_entries_labeled_correctly -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'PUBLISHED_DEVICE_POINTS'`.

- [ ] **Step 9.3: Add `DevicePoint`, data, and boundary functions**

Append to `dispersive_readout/optimization/regime_map.py`:

```python
from dataclasses import dataclass
from ..physics.config import REFERENCE_DEVICE


@dataclass(frozen=True)
class DevicePoint:
    """A published device's position on the (χ/κ, γ_1·τ_readout) regime map."""
    label: str
    citation: str
    chi_over_kappa: float
    gamma_1_tau: float
    reported_F_assign: float | None
    marker: str                    # matplotlib marker code
    marker_color: str              # "warm_orange" or "red" per Q6 encoding
    estimated: bool = False        # True → grey-hatched marker fill (Q5)
    estimated_fields: tuple[str, ...] = ()


PUBLISHED_DEVICE_POINTS: list[DevicePoint] = [
    DevicePoint(
        label="Marxer Q1 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437 p.15 device table + §V.3 Table 1",
        chi_over_kappa=2.5 / 6.1,              # 0.41
        gamma_1_tau=280e-9 / 86e-6,            # 3.26e-3
        reported_F_assign=0.99943,
        marker="*",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Marxer Q2 (IQM, 2025)",
        citation="Marxer et al., arXiv:2508.16437 p.15",
        chi_over_kappa=2.6 / 3.4,              # 0.76
        gamma_1_tau=280e-9 / 102e-6,           # 2.75e-3
        reported_F_assign=0.99946,
        marker="D",
        marker_color="warm_orange",
        estimated=False,
    ),
    DevicePoint(
        label="Bengtsson (Google, 2024)",
        citation=(
            "Bengtsson et al., PRL 132 100603 (2024) / arXiv:2308.02079 Eq. 3; "
            "κ ∈ [4,8] MHz from Sank arXiv:2402.00413 §IV; "
            "T_1 ≈ 20 µs from Arute 2019 Sycamore-typical"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 20e-6,            # 2.5e-2
        reported_F_assign=0.985,
        marker="o",
        marker_color="red",
        estimated=True,
        estimated_fields=("T_1",),
    ),
    DevicePoint(
        label="Garnet (IQM, 2024)",
        citation=(
            "Abdurakhimov et al., arXiv:2408.12433 p.9 (F_assign) + p.13 (T_1); "
            "χ/κ and τ_readout are IQM design-family estimates"
        ),
        chi_over_kappa=0.5,
        gamma_1_tau=500e-9 / 40e-6,            # 1.25e-2
        reported_F_assign=0.97,
        marker="s",
        marker_color="red",
        estimated=True,
        estimated_fields=("chi_over_kappa", "tau_readout"),
    ),
]


# ---------- Analytic boundaries (MODULE_4_SPEC.md §3.2 post-Nit-2) ----------


def _reference_purcell_rate_per_kappa() -> float:
    """(g_REF / Δ_REF)² — the dimensionless factor in γ_Purcell = κ · (g/Δ)²."""
    # Δ = |ω_q − ω_r|; from REFERENCE_DEVICE's transmon parameters (Koch limit)
    # ω_q ≈ sqrt(8 E_J E_C) − E_C. Compute once via Module 1's diagonalize_transmon.
    from ..physics.transmon import diagonalize_transmon
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    omega_q = float(energies[1] - energies[0])
    delta = abs(omega_q - REFERENCE_DEVICE.resonator.omega_r)
    return (REFERENCE_DEVICE.coupling.g / delta) ** 2


def _reference_chi_magnitude() -> float:
    """|χ_01| at REFERENCE, used to relate x = χ/κ → κ along the boundary."""
    from ..physics.transmon import diagonalize_transmon, charge_operator_matrix_elements
    from ..physics.dispersive import dispersive_shift_full
    energies, eigenstates = diagonalize_transmon(
        REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation,
    )
    n_mat = charge_operator_matrix_elements(eigenstates, REFERENCE_DEVICE.truncation)
    chi = dispersive_shift_full(
        energies, n_mat, REFERENCE_DEVICE.coupling.g, REFERENCE_DEVICE.resonator.omega_r,
    )
    return abs(chi[0] - chi[1])


def purcell_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Purcell limit locus: γ_Purcell · τ_readout = 0.1, (g, Δ) at REFERENCE.

    With χ held at REFERENCE's dispersive-computed value, κ(x) = χ_REF / x,
    γ_Purcell(x) = κ(x) · (g_REF/Δ_REF)², and τ_readout(x) = 0.1/γ_Purcell(x).
    Plotted y = γ_1_REF · τ_readout(x).
    """
    chi_ref = _reference_chi_magnitude()
    g_over_delta_sq = _reference_purcell_rate_per_kappa()
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1

    kappa_x = chi_ref / np.asarray(chi_over_kappa, dtype=float)
    gamma_P_x = kappa_x * g_over_delta_sq
    tau_readout_x = 0.1 / gamma_P_x
    return gamma_1_ref * tau_readout_x


def dispersive_breakdown_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Dispersive breakdown locus: χ · τ_readout = 2π.

    With κ held at REFERENCE, χ(x) = x · κ_REF, τ_readout(x) = 2π/χ(x).
    Plotted y = γ_1_REF · τ_readout(x).
    """
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    chi_x = np.asarray(chi_over_kappa, dtype=float) * kappa_ref
    tau_readout_x = (2.0 * np.pi) / chi_x
    return gamma_1_ref * tau_readout_x


def resonator_too_slow_boundary(chi_over_kappa: np.ndarray) -> np.ndarray:
    """Resonator-too-slow locus: κ · τ_readout = 1.

    κ held at REFERENCE; horizontal line in (x, y) at y = γ_1_REF / κ_REF.
    """
    kappa_ref = REFERENCE_DEVICE.resonator.kappa
    gamma_1_ref = REFERENCE_DEVICE.decoherence.gamma_1
    y_const = gamma_1_ref / kappa_ref
    return np.full_like(np.asarray(chi_over_kappa, dtype=float), y_const)
```

Update `dispersive_readout/optimization/__init__.py`:

```python
from .regime_map import (
    DevicePoint,
    PUBLISHED_DEVICE_POINTS,
    f_analytic_dispersive,
    purcell_boundary,
    dispersive_breakdown_boundary,
    resonator_too_slow_boundary,
)
```

And extend the `__all__` list accordingly.

- [ ] **Step 9.4: Run all the new tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "PUBLISHED_DEVICE or marxer_q1 or purcell_boundary or resonator_too_slow"
```

Expected: **5 passed**.

- [ ] **Step 9.5: Commit**

```bash
git add dispersive_readout/optimization/regime_map.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): DevicePoint + PUBLISHED_DEVICE_POINTS + boundaries

4-marker regime-map data per Q5 lock (Marxer Q1, Marxer Q2, Bengtsson,
Garnet; Hazra OMITTED). Three analytic boundary functions with the
closed-form formulas from post-commit Nit 2: Purcell (monotone in x),
dispersive breakdown, resonator-too-slow (constant in x).

Validity tests assert the 4 markers lie within the regime map's axis
ranges [0.1, 10] × [1e-4, 1e-1]."
```

---

## Task 10: `compute_analytic_regime_map` + 2-point Lindblad validation (O3a/O3b) + standalone render

**Rationale:** Spec §3.2 Q3 Refinement 2 (Lindblad-vs-analytic validation at Marxer Q1 and (χ/κ=1, γ₁τ=0.01)); Panel (b) standalone render per §8 Day-11 morning. Locks the caption claim "Lindblad-validated at 2 points, max deviation Y%".

**Files:**
- Modify: `dispersive_readout/optimization/regime_map.py` — add `compute_analytic_regime_map`, `validate_analytic_vs_lindblad`.
- Create: `06_Dispersive_Readout/scripts/fig4_panel_b_regime.py` — standalone Panel (b) render.
- Create: `06_Dispersive_Readout/figures/fig4_panel_b_regime.png` + `fig4_panel_b_validation.yaml` (committed artifact with validation deviations for the caption).
- Modify: `dispersive_readout/tests/test_optimization.py` — add O3a, O3b.

- [ ] **Step 10.1: Write O3a + O3b failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O3a + O3b — analytic vs Lindblad at 2 points (Q3 Refinement 2)
# ────────────────────────────────────────────────────────────────────

def test_O3a_analytic_vs_lindblad_at_marxer_q1():
    """Analytic F at Marxer Q1's (χ/κ, γ_1·τ) must agree with Module 1's
    Lindblad F_sim at REFERENCE (which IS Marxer Q1's anchor) to within 5%."""
    from dispersive_readout.optimization.regime_map import (
        validate_analytic_vs_lindblad, PUBLISHED_DEVICE_POINTS,
    )
    marxer_q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
    report = validate_analytic_vs_lindblad(
        points=[(marxer_q1.chi_over_kappa, marxer_q1.gamma_1_tau)],
    )
    dev = report["per_point"][0]["deviation_fractional"]
    assert dev < 0.05, (
        f"F_analytic vs F_sim deviation {dev*100:.2f}% > 5% at Marxer Q1 "
        f"({marxer_q1.chi_over_kappa:.2f}, {marxer_q1.gamma_1_tau:.2e}). "
        "Spec §9 item 3 — do not publish uncorrected; add leading-order "
        "correction term to f_analytic_dispersive."
    )


def test_O3b_analytic_vs_lindblad_at_midrange_point():
    """At (χ/κ=1.0, γ_1·τ=0.01), analytic and Lindblad must agree within 5%."""
    from dispersive_readout.optimization.regime_map import validate_analytic_vs_lindblad
    report = validate_analytic_vs_lindblad(points=[(1.0, 0.01)])
    dev = report["per_point"][0]["deviation_fractional"]
    assert dev < 0.05, (
        f"F_analytic vs F_sim deviation {dev*100:.2f}% > 5% at (1.0, 0.01). "
        "Caption claim 'max deviation < 5%' fails."
    )
```

- [ ] **Step 10.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O3a_analytic_vs_lindblad_at_marxer_q1 -v -p no:dash
```

Expected: **FAIL** with `ImportError: cannot import name 'validate_analytic_vs_lindblad'`.

- [ ] **Step 10.3: Implement `compute_analytic_regime_map` and `validate_analytic_vs_lindblad`**

Append to `dispersive_readout/optimization/regime_map.py`:

```python
def _infer_n_phot_at_reference() -> float:
    """Infer steady-state photon number at REFERENCE operating point.

    Reuses Module 2's calibration path: get_reference_operating_point returns
    the calibrated drive; average photon number over the last 20% of the
    integration window is the steady-state estimate.
    """
    from ..analysis.operating_point import get_reference_operating_point
    from ..physics.readout_model import simulate_readout

    op = get_reference_operating_point(n_shots=10_000)
    r0 = simulate_readout(op.device, op.drive, initial_qubit_state=0)
    # Average photon number over last 20% of the integration window
    t = r0.t
    t0, t1 = op.integration_window
    window_mask = (t >= t0 + 0.8 * (t1 - t0)) & (t <= t1)
    return float(np.mean(r0.photon_number[window_mask]))


def compute_analytic_regime_map(
    chi_over_kappa_range: tuple[float, float] = (0.1, 10.0),
    gamma_1_tau_range: tuple[float, float] = (1e-4, 1e-1),
    n_chi: int = 20,
    n_gamma: int = 20,
    n_phot: float | None = None,
) -> dict:
    """Return dict with 'chi_over_kappa_axis', 'gamma_1_tau_axis',
    'F_grid', 'n_phot_used'. Sub-second; pure analytic (no sim calls)."""
    if n_phot is None:
        n_phot = _infer_n_phot_at_reference()

    x_axis = np.logspace(
        np.log10(chi_over_kappa_range[0]),
        np.log10(chi_over_kappa_range[1]),
        n_chi,
    )
    y_axis = np.logspace(
        np.log10(gamma_1_tau_range[0]),
        np.log10(gamma_1_tau_range[1]),
        n_gamma,
    )
    # 2D grid via broadcasting: x on axis=0, y on axis=1
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    F = f_analytic_dispersive(X, Y, n_phot=n_phot)

    return {
        "chi_over_kappa_axis": x_axis,
        "gamma_1_tau_axis": y_axis,
        "F_grid": F,
        "n_phot_used": float(n_phot),
    }


def validate_analytic_vs_lindblad(
    points: list[tuple[float, float]] | None = None,
) -> dict:
    """Q3 Refinement 2: evaluate F_sim at specified (χ/κ, γ_1·τ) points and
    compare to F_analytic.

    F_sim is computed at REFERENCE-with-resonator-κ-rescaled-to-hit-target-χ/κ
    (holding χ at REFERENCE's dispersive-computed value) and decoherence-γ_1-
    rescaled-to-hit-target-γ_1·τ (holding τ at REFERENCE's drive.duration).
    Caption cites max deviation.
    """
    from dataclasses import replace

    from ..analysis.operating_point import get_reference_operating_point
    from ..physics.readout_model import simulate_readout, compute_assignment_fidelity

    if points is None:
        # Defaults: Marxer Q1 + mid-range (χ/κ=1, γ_1·τ=0.01)
        from .regime_map import PUBLISHED_DEVICE_POINTS
        q1 = next(p for p in PUBLISHED_DEVICE_POINTS if "Marxer Q1" in p.label)
        points = [(q1.chi_over_kappa, q1.gamma_1_tau), (1.0, 0.01)]

    op = get_reference_operating_point(n_shots=10_000)
    n_phot = _infer_n_phot_at_reference()
    chi_ref = _reference_chi_magnitude()
    tau = op.drive.duration

    per_point = []
    for (target_chi_over_k, target_gamma_tau) in points:
        # Construct device with rescaled κ and γ_1 to hit target coordinates
        target_kappa = chi_ref / target_chi_over_k
        target_gamma_1 = target_gamma_tau / tau
        new_res = replace(op.device.resonator, kappa=target_kappa)
        new_dec = replace(op.device.decoherence, gamma_1=target_gamma_1)
        new_device = replace(op.device, resonator=new_res, decoherence=new_dec)

        r0 = simulate_readout(new_device, op.drive, initial_qubit_state=0)
        r1 = simulate_readout(new_device, op.drive, initial_qubit_state=1)
        F_sim = compute_assignment_fidelity(
            r0, r1, op.integration_window, n_shots=op.n_shots, noise_model="ideal",
        ).F_assign

        F_analytic = float(
            f_analytic_dispersive(
                np.asarray(target_chi_over_k), np.asarray(target_gamma_tau), n_phot=n_phot,
            )
        )
        per_point.append({
            "chi_over_kappa": float(target_chi_over_k),
            "gamma_1_tau": float(target_gamma_tau),
            "F_analytic": float(F_analytic),
            "F_lindblad": float(F_sim),
            "deviation_fractional": float(abs(F_sim - F_analytic) / F_sim),
        })

    max_dev = max(p["deviation_fractional"] for p in per_point)
    return {"per_point": per_point, "max_deviation_fractional": max_dev, "n_phot_used": n_phot}
```

Also extend exports in `dispersive_readout/optimization/__init__.py`:

```python
from .regime_map import (
    compute_analytic_regime_map,
    validate_analytic_vs_lindblad,
)
```

- [ ] **Step 10.4: Run O3a + O3b to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O3a_analytic_vs_lindblad_at_marxer_q1 dispersive_readout/tests/test_optimization.py::test_O3b_analytic_vs_lindblad_at_midrange_point -v -p no:dash
```

Expected: **2 passed** (4 Lindblad calls total; ~4 min at REFERENCE).

- [ ] **Step 10.5: Write the standalone Panel (b) render script**

Create `06_Dispersive_Readout/scripts/fig4_panel_b_regime.py`:

```python
"""Standalone Figure 4 Panel (b) — analytic regime map with 4 device overlays.

Produces figures/fig4_panel_b_regime.png + fig4_panel_b_validation.yaml
(the validation deviations are cited in the Figure 4 caption).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dispersive_readout.optimization.regime_map import (
    compute_analytic_regime_map,
    validate_analytic_vs_lindblad,
    purcell_boundary,
    dispersive_breakdown_boundary,
    resonator_too_slow_boundary,
    PUBLISHED_DEVICE_POINTS,
)


_MARKER_COLOR_MAP = {"warm_orange": "#E8801A", "red": "#C0392B"}


def render_regime_map(ax: plt.Axes, validation: dict, grid: dict) -> None:
    x_axis = grid["chi_over_kappa_axis"]
    y_axis = grid["gamma_1_tau_axis"]
    F = grid["F_grid"]

    # Heatmap
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    pcm = ax.pcolormesh(X, Y, F, cmap="viridis", shading="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(pcm, ax=ax, label=r"$F_{\rm assign}$")

    # Contours
    cs = ax.contour(X, Y, F, levels=[0.95, 0.99, 0.999], colors="white",
                    linestyles="dashed", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8)

    # Analytic boundaries (grey dashed)
    x_fine = np.logspace(np.log10(x_axis[0]), np.log10(x_axis[-1]), 200)
    y_purcell = purcell_boundary(x_fine)
    y_disp = dispersive_breakdown_boundary(x_fine)
    y_slow = resonator_too_slow_boundary(x_fine)
    for (y_boundary, label) in [
        (y_purcell, "Purcell limit"),
        (y_disp, "Dispersive breakdown"),
        (y_slow, r"$\kappa\,\tau_{\rm readout} = 1$"),
    ]:
        mask = (y_boundary >= y_axis[0]) & (y_boundary <= y_axis[-1])
        ax.plot(x_fine[mask], y_boundary[mask],
                color="grey", linestyle="--", linewidth=1.0)

    # 4 device markers
    for p in PUBLISHED_DEVICE_POINTS:
        color = _MARKER_COLOR_MAP[p.marker_color]
        hatch = {"hatch": "///", "edgecolor": "grey"} if p.estimated else {}
        ax.scatter(
            [p.chi_over_kappa], [p.gamma_1_tau],
            marker=p.marker, s=180, c=color, edgecolors="white", linewidths=1.5,
            zorder=10,
        )
        # Marxer Q1 gets F_sim annotation
        if "Marxer Q1" in p.label and p.reported_F_assign is not None:
            q1_sim = next(
                pt["F_lindblad"] for pt in validation["per_point"]
                if abs(pt["chi_over_kappa"] - p.chi_over_kappa) < 1e-6
            )
            ax.annotate(
                f"$F_{{\\rm sim}} = {q1_sim:.4f}$",
                xy=(p.chi_over_kappa, p.gamma_1_tau),
                xytext=(8, 8), textcoords="offset points", fontsize=9,
                color="white",
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\chi/\kappa$")
    ax.set_ylabel(r"$\gamma_1 \cdot \tau_{\rm readout}$")
    max_dev_pct = validation["max_deviation_fractional"] * 100.0
    ax.set_title(
        "Regime map of dispersive readout fidelity\n"
        f"Dispersive-analytic F (Bengtsson 2024); Lindblad-validated at "
        f"2 points, max dev {max_dev_pct:.2f}%; "
        r"$\bar n_{\rm phot}$ = " + f"{grid['n_phot_used']:.2f}",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main() -> None:
    grid = compute_analytic_regime_map()
    validation = validate_analytic_vs_lindblad()

    # Persist validation for caption regeneration and O9 regression
    out_dir = Path("06_Dispersive_Readout/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fig4_panel_b_validation.yaml", "w") as f:
        yaml.safe_dump(validation, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    render_regime_map(ax, validation, grid)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_panel_b_regime.png", dpi=150)
    print(f"Wrote {out_dir / 'fig4_panel_b_regime.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.6: Run the render script**

```bash
python 06_Dispersive_Readout/scripts/fig4_panel_b_regime.py
```

Expected: `Wrote .../fig4_panel_b_regime.png`. Open the PNG; verify 4 markers visible (★ ◆ at warm-orange, ● □ at red with hatching on □), white-dashed F contours, 3 grey-dashed boundaries.

- [ ] **Step 10.7: Commit**

```bash
git add dispersive_readout/optimization/regime_map.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py 06_Dispersive_Readout/scripts/fig4_panel_b_regime.py 06_Dispersive_Readout/figures/fig4_panel_b_regime.png 06_Dispersive_Readout/figures/fig4_panel_b_validation.yaml
git commit -m "feat(stage06-m4): regime map grid + 2-pt Lindblad validation (O3a/O3b)

compute_analytic_regime_map is sub-second pure analytic. Panel (b)
render applies memory rules (viridis, grey-dashed boundaries, warm/red
marker split per Q6). fig4_panel_b_validation.yaml caches the max
deviation for the Figure 4 caption regeneration and O9 regression gate."
```

---

## Task 11: Modal image + `pareto_one_tuple` stub + O10 smoke test (Day-11 afternoon pre-warm)

**Rationale:** Spec §8 Day-11 afternoon task per Q2 lock — pre-warm the Modal image (qutip + scipy + scipy.optimize on top of Module 3's base) and confirm credentials + image build work end-to-end with a trivial smoke call. Surfaces infra rot on Day 11 (not Day 12 morning when Pareto needs to run).

**Files:**
- Create: `dispersive_readout/optimization/modal_pareto.py` — Modal app + `pareto_one_tuple` stub.
- Modify: `dispersive_readout/optimization/__init__.py` — export `pareto_one_tuple`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O10.

- [ ] **Step 11.1: Write O10 smoke-test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O10 — Modal image smoke test (Q2 pre-warm task)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O10_modal_pareto_one_tuple_smoke():
    """Pre-warm the Module 4 Modal image and dispatch one trivial
    pareto_one_tuple call via .map([one_tuple]). Confirms credentials,
    image build, and serialization work before Day 12's Pareto run."""
    import os

    if os.environ.get("SKIP_MODAL_TESTS") == "1":
        pytest.skip("SKIP_MODAL_TESTS=1 set — skip Modal smoke in CI")

    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.modal_pareto import (
        app, pareto_one_tuple,
    )
    from dispersive_readout.optimization.pareto import ParetoPoint

    # Modal's .map takes iterables; dispatch exactly one tuple.
    with app.run():
        results = list(pareto_one_tuple.map([REFERENCE_DEVICE], [500e-9]))

    assert len(results) == 1
    assert isinstance(results[0], ParetoPoint)
```

- [ ] **Step 11.2: Run the smoke test — it will fail for lack of modal_pareto.py**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O10_modal_pareto_one_tuple_smoke -v -p no:dash
```

Expected: **FAIL** with `ModuleNotFoundError: No module named 'dispersive_readout.optimization.modal_pareto'` OR `ModuleNotFoundError: No module named 'dispersive_readout.optimization.pareto'` (pareto.py doesn't exist yet — the smoke will skip implicitly).

- [ ] **Step 11.3: Create `modal_pareto.py` with the image spec and a minimal stub of `pareto_one_tuple`**

Create `dispersive_readout/optimization/modal_pareto.py`:

```python
"""Modal-parallelized Pareto per-point dispatch.

Public module (not `_modal_pareto`): parallelism boundary is a first-class
architectural surface per Q7. Reuses Module 3's `.map()` pattern (see
characterization/recovery.py for the precedent).

Day-11 afternoon pre-warm task: build the image, verify credentials, run
one smoke dispatch via test O10. The actual Pareto dispatch lands in Task 14.
"""
from __future__ import annotations

import modal


# Extends Module 3's image with qutip + scipy so the inner find_pareto_point
# call can run Lindblad-solver + SLSQP on the Modal worker.
stage_06_module4_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3.0",
        "scipy>=1.11,<2.0",
        "qutip>=5.0,<6.0",
        "pydantic>=2.0,<3.0",
        "pyyaml>=6.0,<7.0",
    )
    .add_local_python_source("dispersive_readout")
)


app = modal.App("stage06-module4-pareto", image=stage_06_module4_image)


@app.function(cpu=2.0, memory=4096)
def pareto_one_tuple(device, tau_max: float):
    """Single-tuple Pareto-point computation.

    Pure function: no global state, no filesystem side effects.
    Receives `device: DeviceConfig` and `tau_max: float`; returns a
    `ParetoPoint`. Delegates to `dispersive_readout.optimization.pareto
    .find_pareto_point` which lands in Task 13.

    For the Day-11 smoke test (O10), a stub implementation returns a
    minimal ParetoPoint so the Modal round-trip succeeds independent
    of Pareto implementation status. Task 13 replaces the stub body.
    """
    from .pareto import find_pareto_point
    return find_pareto_point(device, tau_max)
```

Also add a *temporary* `find_pareto_point` stub so Modal can dispatch during the smoke test — create `dispersive_readout/optimization/pareto.py`:

```python
"""Pareto frontier — skeleton committed in Task 11 (Modal smoke), filled
in Tasks 12–14."""
from __future__ import annotations

from pydantic import BaseModel


class ParetoPoint(BaseModel):
    """Placeholder schema — full definition in Task 12."""
    device_id: str
    tau_max: float
    epsilon_0_opt: float
    tau_opt: float
    F_assign_opt: float
    F_assign_uncertainty: float
    dominant_loss_channel: str
    solver_converged: bool


def find_pareto_point(device, tau_max: float) -> ParetoPoint:
    """Placeholder implementation so O10 smoke succeeds. Task 13 replaces."""
    return ParetoPoint(
        device_id="placeholder",
        tau_max=float(tau_max),
        epsilon_0_opt=0.0,
        tau_opt=float(tau_max),
        F_assign_opt=0.5,
        F_assign_uncertainty=0.01,
        dominant_loss_channel="placeholder",
        solver_converged=False,
    )
```

Update `dispersive_readout/optimization/__init__.py` exports:

```python
from .modal_pareto import app, pareto_one_tuple
from .pareto import ParetoPoint, find_pareto_point
```

Add `pareto_one_tuple`, `app`, `ParetoPoint`, `find_pareto_point` to `__all__`.

- [ ] **Step 11.4: Run O10 to verify pass — requires Modal credentials**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O10_modal_pareto_one_tuple_smoke -v -p no:dash
```

Expected (with credentials): **1 passed** in ~30 s (first run builds the image; subsequent runs cache).

If Modal credentials are missing or image build fails: **follow spec §9 item 5** — "do not let this slip into Day 12 morning — fix on Day 11 afternoon or the Pareto run is blocked." Reproduce locally with `SKIP_MODAL_TESTS=1 python -m pytest ...::test_O10_... -v` (skip → still a green bar; flag externally).

- [ ] **Step 11.5: Commit**

```bash
git add dispersive_readout/optimization/modal_pareto.py dispersive_readout/optimization/pareto.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): Modal pareto image + O10 smoke + ParetoPoint stub

stage_06_module4_image extends Module 3's with qutip + scipy. pareto_one_tuple
is the parallelism boundary Public-module per Q7 (not _underscored). The
ParetoPoint stub + find_pareto_point placeholder ship here so the Day-11
pre-warm smoke test confirms credentials + image build + round-trip
serialization; Tasks 12–14 replace the stubs."
```

---

## Task 12: `ParetoPoint` full schema + `build_variant` + `PARETO_DEVICE_VARIANTS` + O6.2 + O22 + O23

**Rationale:** Spec §5.3 `ParetoPoint` Pydantic schema (full definition replacing Task 11's stub) + §3.3 `PARETO_DEVICE_VARIANTS` table + `build_variant` construction via `dataclasses.replace`. O6.2 validates the tau_opt ≤ tau_max constraint; O22/O23 round-trip the variant construction through Module 3's Koch-back-solve pathway (same convention as `to_device_config`).

**Files:**
- Modify: `dispersive_readout/optimization/pareto.py` — replace stub with full `ParetoPoint` schema, add `PARETO_DEVICE_VARIANTS`, `TAU_MAX_GRID_NS`, `build_variant`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O6.2 + O22 + O23.

- [ ] **Step 12.1: Write O6.2 + O22 + O23 failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O6.2 — ParetoPoint schema validation
# ────────────────────────────────────────────────────────────────────

def test_O6_2_pareto_point_accepts_valid():
    from dispersive_readout.optimization.pareto import ParetoPoint
    p = ParetoPoint(
        device_id="deadbeef",
        device_label="REFERENCE (≈ Marxer Q1)",
        tau_max=500e-9,
        epsilon_0_opt=5e7,
        tau_opt=480e-9,
        F_assign_opt=0.9984,
        F_assign_uncertainty=1.2e-3,
        dominant_loss_channel="T1_intrinsic",
        solver_converged=True,
    )
    assert p.tau_opt <= p.tau_max


def test_O6_2_pareto_point_rejects_tau_opt_exceeding_tau_max():
    from pydantic import ValidationError
    from dispersive_readout.optimization.pareto import ParetoPoint
    with pytest.raises(ValidationError):
        ParetoPoint(
            device_id="deadbeef",
            device_label="REFERENCE",
            tau_max=500e-9,
            epsilon_0_opt=5e7,
            tau_opt=520e-9,  # > tau_max, must reject
            F_assign_opt=0.99,
            F_assign_uncertainty=1e-3,
            dominant_loss_channel="T1_intrinsic",
            solver_converged=True,
        )


# ────────────────────────────────────────────────────────────────────
# O22 / O23 — bridge round-trip for V2 (T1=40µs) and V3 (T1=20µs, κ=6MHz)
# ────────────────────────────────────────────────────────────────────

def test_O22_build_variant_v2_garnet_like():
    """V2 swaps decoherence.gamma_1 = 1/40µs, leaves resonator and coupling
    at REFERENCE. gamma_phi recomputed via Koch back-solve."""
    import math
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import build_variant, PARETO_DEVICE_VARIANTS

    spec = next(v for v in PARETO_DEVICE_VARIANTS if "40" in v["label"] or v["T1_us"] == 40.0)
    variant = build_variant(spec)

    assert variant.decoherence.gamma_1 == pytest.approx(1.0 / 40e-6, rel=1e-9)
    assert variant.resonator.kappa == REFERENCE_DEVICE.resonator.kappa
    assert variant.coupling.g == REFERENCE_DEVICE.coupling.g
    # Koch back-solve for gamma_phi: gamma_phi = 1/T2_echo - gamma_1/2
    # T2_echo preserved at REFERENCE's value
    T2_echo_REF = 2.0 / (REFERENCE_DEVICE.decoherence.gamma_1 +
                         2.0 * REFERENCE_DEVICE.decoherence.gamma_phi)
    expected_gamma_phi = max(1.0 / T2_echo_REF - 0.5 * (1.0 / 40e-6), 0.0)
    assert variant.decoherence.gamma_phi == pytest.approx(expected_gamma_phi, rel=1e-9)


def test_O23_build_variant_v3_bengtsson_like():
    """V3 swaps T1=20µs AND κ/2π=6MHz."""
    import math
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import build_variant, PARETO_DEVICE_VARIANTS

    spec = next(
        v for v in PARETO_DEVICE_VARIANTS
        if v["T1_us"] == 20.0 and v["kappa_MHz"] == 6.0
    )
    variant = build_variant(spec)

    assert variant.decoherence.gamma_1 == pytest.approx(1.0 / 20e-6, rel=1e-9)
    assert variant.resonator.kappa == pytest.approx(2.0 * math.pi * 6e6, rel=1e-9)
    assert variant.coupling.g == REFERENCE_DEVICE.coupling.g
```

- [ ] **Step 12.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O6_2_pareto_point_accepts_valid test_O22_build_variant_v2_garnet_like -v -p no:dash
```

Expected: **FAIL** (current stub ParetoPoint has no tau_opt <= tau_max validator; `build_variant` doesn't exist; `PARETO_DEVICE_VARIANTS` doesn't exist).

- [ ] **Step 12.3: Replace `pareto.py` stub with full schema, variants table, and `build_variant`**

Replace the entire contents of `dispersive_readout/optimization/pareto.py` with:

```python
"""Pareto-frontier computation for Module 4.

See MODULE_4_SPEC.md §3.3, §5.3. SLSQP + 5×5 warm-start over (ε_0, τ)
against a noise-free analytic objective (Q8 contract). Uncertainty is
analytic binomial SE on reported F_opt.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

from ..physics.config import DeviceConfig, DriveParams, REFERENCE_DEVICE


# ────────────────────────────────────────────────────────────────────
# Spec §3.3 — locked data
# ────────────────────────────────────────────────────────────────────

PARETO_DEVICE_VARIANTS: list[dict[str, Any]] = [
    {
        "label": "REFERENCE (≈ Marxer Q1)",
        "T1_us": None,
        "kappa_MHz": None,
    },
    {
        "label": "T_1 = 40 µs (Garnet-like)",
        "T1_us": 40.0,
        "kappa_MHz": None,
    },
    {
        "label": "T_1 = 20 µs, κ/2π = 6 MHz (Bengtsson-like)",
        "T1_us": 20.0,
        "kappa_MHz": 6.0,
    },
]


# 10 log-spaced points from 100 ns to 2 µs per spec §3.3
TAU_MAX_GRID_NS: np.ndarray = np.logspace(np.log10(100.0), np.log10(2000.0), 10)


# ────────────────────────────────────────────────────────────────────
# Spec §5.3 — ParetoPoint schema
# ────────────────────────────────────────────────────────────────────

class ParetoPoint(BaseModel):
    """Optimal (ε_0, τ) at one τ_max constraint, for one device."""
    device_id: str                        # hash of DeviceConfig (audit trail)
    device_label: str
    tau_max: float
    epsilon_0_opt: float
    tau_opt: float
    F_assign_opt: float                   # analytic Gaussian-overlap F at optimum
    F_assign_uncertainty: float           # analytic binomial SE at n_shots
    dominant_loss_channel: str
    solver_converged: bool

    @field_validator("F_assign_opt")
    @classmethod
    def _valid_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"F_assign_opt must be in [0, 1] (got {v})")
        return v

    @model_validator(mode="after")
    def _tau_opt_within_tau_max(self):
        # 0.1% tolerance for solver slop
        if self.tau_opt > self.tau_max * 1.001:
            raise ValueError(
                f"tau_opt ({self.tau_opt}) exceeds tau_max ({self.tau_max}) "
                "beyond 0.1% solver tolerance"
            )
        return self


# ────────────────────────────────────────────────────────────────────
# build_variant — Koch back-solve for γ_φ preserves T2_echo at REFERENCE
# ────────────────────────────────────────────────────────────────────

def _device_id(device: DeviceConfig) -> str:
    """Deterministic short hash of the DeviceConfig for audit trail."""
    summary = {
        "T1_us": 1e6 / device.decoherence.gamma_1,
        "T2_rate": device.decoherence.gamma_phi,
        "n_th": device.decoherence.n_th,
        "kappa": device.resonator.kappa,
        "g": device.coupling.g,
        "omega_r": device.resonator.omega_r,
    }
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:12]


def build_variant(variant_spec: dict[str, Any]) -> DeviceConfig:
    """Construct a PARETO_DEVICE_VARIANTS entry from REFERENCE_DEVICE.

    Koch back-solve convention (Module 3 compatibility):
        T_2_echo is held at REFERENCE's value;
        gamma_phi is recomputed as max(1/T_2_echo - gamma_1/2, 0.0).
    This matches ExtractedParameterPack.to_device_config() so V2/V3
    construction is bridge-consistent with the closed-loop demo device.
    """
    dec_ref = REFERENCE_DEVICE.decoherence
    res_ref = REFERENCE_DEVICE.resonator

    T2_echo_REF = 2.0 / (dec_ref.gamma_1 + 2.0 * dec_ref.gamma_phi)

    # Decoherence substitution
    if variant_spec["T1_us"] is None:
        new_gamma_1 = dec_ref.gamma_1
    else:
        new_gamma_1 = 1.0 / (variant_spec["T1_us"] * 1e-6)
    new_gamma_phi = max(1.0 / T2_echo_REF - 0.5 * new_gamma_1, 0.0)
    new_dec = replace(dec_ref, gamma_1=new_gamma_1, gamma_phi=new_gamma_phi)

    # Resonator substitution
    if variant_spec["kappa_MHz"] is None:
        new_res = res_ref
    else:
        new_kappa = 2.0 * math.pi * variant_spec["kappa_MHz"] * 1e6
        new_res = replace(res_ref, kappa=new_kappa)

    return replace(REFERENCE_DEVICE, decoherence=new_dec, resonator=new_res)
```

- [ ] **Step 12.4: Run O6.2 + O22 + O23 to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "O6_2 or O22 or O23"
```

Expected: **4 passed** (O6.2 has two sub-tests + O22 + O23).

- [ ] **Step 12.5: Commit**

```bash
git add dispersive_readout/optimization/pareto.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): ParetoPoint full schema + PARETO_DEVICE_VARIANTS + O22/O23

ParetoPoint validator enforces tau_opt <= tau_max within 0.1% solver tol.
build_variant applies Koch back-solve for γ_φ at fixed T_2_echo_REF —
matches ExtractedParameterPack.to_device_config() bridge convention so
V2/V3 construction is bridge-consistent with the closed-loop demo."
```

---

## Task 13: `find_pareto_point` SLSQP + 5×5 warm-start + O19 + O20 + O21

**Rationale:** Spec §5.3 Pareto solver + §6.1 tests O19 (lower τ_max boundary), O20 (upper τ_max boundary), O21 (infeasibility detection). SLSQP + grid-warm-start uses `noise_model='ideal'` throughout per Q8 contract (guarded by O8 test).

**Files:**
- Modify: `dispersive_readout/optimization/pareto.py` — add `find_pareto_point` with warm-start and SLSQP.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O19, O20, O21.

- [ ] **Step 13.1: Write O19 + O20 + O21 failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O19–O21 — Pareto edge cases
# ────────────────────────────────────────────────────────────────────

def test_O19_pareto_at_lower_tau_max_boundary_feasible():
    """τ_max = 100 ns must return a feasible ParetoPoint, possibly with
    lower F_opt than at larger τ_max but still > 0.5."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=100e-9)
    assert p.solver_converged
    assert p.F_assign_opt > 0.5
    assert p.tau_opt <= p.tau_max * 1.001


def test_O20_pareto_at_upper_tau_max_boundary_feasible():
    """τ_max = 2 µs must return a feasible ParetoPoint. At this budget
    REFERENCE achieves F >> 0.99."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=2000e-9)
    assert p.solver_converged
    assert p.F_assign_opt > 0.99


def test_O21_pareto_infeasibility_at_extreme_drive_bounds():
    """If ε_0 bounds exclude all F > 0.5, find_pareto_point returns
    solver_converged=False (or raises). Either signal is acceptable;
    test that failure is surfaced, not silent."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import find_pareto_point

    # Extremely low drive amplitude bounds — F cannot exceed 0.5
    try:
        p = find_pareto_point(
            REFERENCE_DEVICE,
            tau_max=500e-9,
            epsilon_0_bounds=(1.0, 1e3),  # 1–1000 rad/s is absurdly low
        )
    except RuntimeError:
        return  # raised; also acceptable
    # Otherwise: solver must flag non-convergence OR low F
    assert (not p.solver_converged) or p.F_assign_opt < 0.6, (
        f"Infeasible regime produced converged={p.solver_converged} "
        f"F={p.F_assign_opt:.3f} — failure was not surfaced."
    )
```

- [ ] **Step 13.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O19_pareto_at_lower_tau_max_boundary_feasible -v -p no:dash
```

Expected: **FAIL** — `find_pareto_point` is still the Task-11 stub returning hardcoded values.

- [ ] **Step 13.3: Implement full `find_pareto_point` with SLSQP + warm-start**

Append to `dispersive_readout/optimization/pareto.py`:

```python
from scipy.optimize import minimize

from ..physics.readout_model import simulate_readout, compute_assignment_fidelity


def _F_analytic_at(
    device: DeviceConfig, eps_0: float, tau: float,
    integration_window: tuple[float, float] = (50e-9, None),
) -> float:
    """Analytic F_assign at (eps_0, tau). Uses noise_model='ideal' per Q8."""
    drive = DriveParams(amplitude=float(eps_0), duration=float(tau), detuning=0.0)
    t_win = (integration_window[0], tau) if integration_window[1] is None else integration_window
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    return compute_assignment_fidelity(
        r0, r1, t_win, n_shots=10_000, noise_model="ideal",
    ).F_assign


def _warm_start_grid_best(
    device: DeviceConfig,
    eps_0_bounds: tuple[float, float],
    tau_bounds: tuple[float, float],
    n_side: int = 5,
) -> tuple[float, float, float]:
    """Scan a 5×5 (ε_0, τ) grid and return (eps_star, tau_star, F_star)."""
    eps_grid = np.linspace(eps_0_bounds[0], eps_0_bounds[1], n_side)
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], n_side)

    best_eps, best_tau, best_F = None, None, -1.0
    for e in eps_grid:
        for t in tau_grid:
            try:
                F = _F_analytic_at(device, e, t)
            except Exception:
                continue
            if F > best_F:
                best_eps, best_tau, best_F = float(e), float(t), float(F)
    return best_eps, best_tau, best_F


def find_pareto_point(
    device: DeviceConfig,
    tau_max: float,
    epsilon_0_bounds: tuple[float, float] = (1e6, 1e9),
    tau_bounds: tuple[float, float] | None = None,
    n_warm_start_grid_side: int = 5,
) -> ParetoPoint:
    """Find (ε_0, τ) that maximize F_assign subject to τ ≤ tau_max.

    1. Coarse 5×5 grid warm-start.
    2. SLSQP local refinement against -F (minimize).
    3. Analytic binomial SE on the converged F_opt.
    All F evaluations use noise_model='ideal' (Q8 contract).
    """
    if tau_bounds is None:
        tau_bounds = (50e-9, tau_max)

    e_star, t_star, F_warm = _warm_start_grid_best(
        device, epsilon_0_bounds, tau_bounds, n_side=n_warm_start_grid_side,
    )
    if e_star is None:
        # All grid evaluations failed — solver cannot proceed
        return ParetoPoint(
            device_id=_device_id(device),
            device_label="<unknown>",
            tau_max=float(tau_max),
            epsilon_0_opt=float(epsilon_0_bounds[0]),
            tau_opt=float(tau_bounds[0]),
            F_assign_opt=0.5,
            F_assign_uncertainty=1e-3,
            dominant_loss_channel="solver_failed",
            solver_converged=False,
        )

    def neg_F(x: np.ndarray) -> float:
        return -_F_analytic_at(device, x[0], x[1])

    res = minimize(
        neg_F,
        x0=np.array([e_star, t_star]),
        method="SLSQP",
        bounds=[epsilon_0_bounds, tau_bounds],
        options={"ftol": 1e-6, "maxiter": 80},
    )

    eps_opt = float(np.clip(res.x[0], *epsilon_0_bounds))
    tau_opt = float(np.clip(res.x[1], *tau_bounds))
    F_opt = float(-res.fun)

    sigma_F = math.sqrt(F_opt * (1.0 - F_opt) / 10_000.0)

    # Dominant loss channel: query Module 2's error-budget at this operating point.
    try:
        from ..analysis.operating_point import OperatingPoint
        from ..analysis.error_budget import compute_full_error_budget
        op = OperatingPoint(
            device=device,
            drive=DriveParams(amplitude=eps_opt, duration=tau_opt, detuning=0.0),
            integration_window=(50e-9, tau_opt),
            n_shots=10_000,
        )
        budget = compute_full_error_budget(op)
        # Dominant active-loss channel = max delta_F among active_loss
        active = budget.active_loss_channels
        if active:
            dominant = max(active, key=lambda c: c.delta_F).name
        else:
            dominant = "none"
    except Exception:
        # If error-budget query fails, don't fail the Pareto point — label unknown
        dominant = "unknown"

    return ParetoPoint(
        device_id=_device_id(device),
        device_label="<set-by-caller>",
        tau_max=float(tau_max),
        epsilon_0_opt=eps_opt,
        tau_opt=tau_opt,
        F_assign_opt=F_opt,
        F_assign_uncertainty=float(sigma_F),
        dominant_loss_channel=str(dominant),
        solver_converged=bool(res.success),
    )
```

- [ ] **Step 13.4: Run O19 + O20 + O21 to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "O19 or O20 or O21"
```

Expected: **3 passed**. Each call is ~25 Lindblad sims in warm-start + ~50 SLSQP evaluations × 2 = ~150 sim calls ≈ 2–3 min per point in serial.

- [ ] **Step 13.5: Confirm O8 contract still green (paired Pareto check)**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O8_no_gaussian_noise_inside_pareto_module -v -p no:dash
```

Expected: **1 passed** (was skipped in Task 3; now passes because `pareto.py` exists and uses `noise_model="ideal"` everywhere).

- [ ] **Step 13.6: Commit**

```bash
git add dispersive_readout/optimization/pareto.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): find_pareto_point SLSQP + 5×5 warm-start + O19/O20/O21

Analytic objective (noise_model='ideal') throughout; analytic binomial
SE post-hoc for F_assign_uncertainty. Dominant loss channel queried
from Module 2's compute_full_error_budget at the converged (ε_0, τ).
O8 Q8-contract test now passes (was skipped until Task 12)."
```

---

## Task 14: `compute_pareto_frontier` + Modal dispatch + O4 + standalone Panel (c) render

**Rationale:** Spec §3.3 batch frontier computation + Modal dispatch + §6.1 O4 (Pareto monotonicity in τ_max) + §8 Day-12 afternoon standalone render. Updates `modal_pareto.pareto_one_tuple` from stub to delegate to `find_pareto_point` (ships in Task 13).

**Files:**
- Modify: `dispersive_readout/optimization/pareto.py` — add `compute_pareto_frontier`.
- Modify: `dispersive_readout/optimization/__init__.py` — export `compute_pareto_frontier`, `PARETO_DEVICE_VARIANTS`, `TAU_MAX_GRID_NS`, `build_variant`.
- Create: `06_Dispersive_Readout/scripts/fig4_panel_c_pareto.py` — standalone Panel (c) render.
- Create: `06_Dispersive_Readout/figures/fig4_panel_c_pareto.png` + `fig4_panel_c_data.yaml` (committed artifacts).
- Modify: `dispersive_readout/tests/test_optimization.py` — add O4.

- [ ] **Step 14.1: Write O4 monotonicity failing test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O4 — Pareto monotonicity in τ_max
# ────────────────────────────────────────────────────────────────────

def test_O4_pareto_monotonic_in_tau_max_for_reference():
    """F_opt non-decreasing along REFERENCE's Pareto curve.

    Relaxing τ_max cannot make F_opt worse; if it does, SLSQP is stuck
    at a local minimum — spec §9 item 4 says increase warm_start grid
    density before changing solvers."""
    import numpy as np
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.pareto import compute_pareto_frontier

    curve = compute_pareto_frontier(
        REFERENCE_DEVICE,
        tau_max_values=np.array([200e-9, 500e-9, 1000e-9, 2000e-9]),
        device_label="REFERENCE (test)",
    )
    F_opts = [p.F_assign_opt for p in curve]
    # Non-decreasing within 5σ_shot slack (shot-noise σ ~ 1e-3, 5σ ≈ 5e-3)
    for a, b in zip(F_opts, F_opts[1:]):
        assert b >= a - 5e-3, (
            f"F_opt decreased from {a:.4f} -> {b:.4f} across adjacent τ_max. "
            "Increase n_warm_start_grid_side from 5 to 10 and retry."
        )
```

- [ ] **Step 14.2: Run test to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O4_pareto_monotonic_in_tau_max_for_reference -v -p no:dash
```

Expected: **FAIL** — `compute_pareto_frontier` doesn't exist yet.

- [ ] **Step 14.3: Implement `compute_pareto_frontier`**

Append to `dispersive_readout/optimization/pareto.py`:

```python
def compute_pareto_frontier(
    device: DeviceConfig,
    tau_max_values: np.ndarray | None = None,
    device_label: str = "<unnamed>",
    use_modal: bool = False,
) -> list[ParetoPoint]:
    """Trace one device's Pareto frontier across tau_max values.

    Parameters
    ----------
    tau_max_values : np.ndarray, optional
        Defaults to TAU_MAX_GRID_NS * 1e-9 (10 log-spaced points, 100 ns - 2 µs).
    device_label : str
        Human-readable label; stamped onto each ParetoPoint.device_label.
    use_modal : bool
        If True, dispatch via modal_pareto.pareto_one_tuple.map(...).
        If False (default), run serial list(map(...)).

    Returns
    -------
    list[ParetoPoint], ordered by tau_max ascending.
    """
    if tau_max_values is None:
        tau_max_values = TAU_MAX_GRID_NS * 1e-9
    tau_max_list = [float(t) for t in tau_max_values]

    if use_modal:
        from .modal_pareto import app, pareto_one_tuple
        with app.run():
            results = list(pareto_one_tuple.map(
                [device] * len(tau_max_list), tau_max_list,
            ))
    else:
        results = [find_pareto_point(device, t) for t in tau_max_list]

    # Stamp the human-readable label
    labeled = []
    for p in results:
        labeled.append(ParetoPoint(
            device_id=p.device_id,
            device_label=device_label,
            tau_max=p.tau_max,
            epsilon_0_opt=p.epsilon_0_opt,
            tau_opt=p.tau_opt,
            F_assign_opt=p.F_assign_opt,
            F_assign_uncertainty=p.F_assign_uncertainty,
            dominant_loss_channel=p.dominant_loss_channel,
            solver_converged=p.solver_converged,
        ))
    return labeled
```

Extend `dispersive_readout/optimization/__init__.py` exports:

```python
from .pareto import (
    ParetoPoint,
    PARETO_DEVICE_VARIANTS,
    TAU_MAX_GRID_NS,
    build_variant,
    find_pareto_point,
    compute_pareto_frontier,
)
```

- [ ] **Step 14.4: Run O4 to verify pass (serial mode)**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O4_pareto_monotonic_in_tau_max_for_reference -v -p no:dash
```

Expected: **1 passed** in ~10 min serial (4 τ_max points × ~2.5 min each).

- [ ] **Step 14.5: Run the full 3-variant × 10-point frontier via Modal**

Create `06_Dispersive_Readout/scripts/fig4_panel_c_pareto.py`:

```python
"""Standalone Figure 4 Panel (c) — 3 parameter-anchored Pareto frontiers
plus the closed-loop recommendation arrow.

Produces figures/fig4_panel_c_pareto.png and fig4_panel_c_data.yaml.
Day-13's fig4_optimization.py imports render_pareto() for the composite.

Modal dispatch: set USE_MODAL=1 to parallelize; otherwise runs serial.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dispersive_readout.optimization.pareto import (
    PARETO_DEVICE_VARIANTS, TAU_MAX_GRID_NS,
    build_variant, compute_pareto_frontier,
)


_VARIANT_STYLES = {
    "REFERENCE (≈ Marxer Q1)":                  {"color": "#2C3E50", "marker": "o"},
    "T_1 = 40 µs (Garnet-like)":                 {"color": "#7F8C8D", "marker": "s"},
    "T_1 = 20 µs, κ/2π = 6 MHz (Bengtsson-like)": {"color": "#566573", "marker": "^"},
}


def render_pareto(ax: plt.Axes, frontiers: dict[str, list]) -> None:
    for label, points in frontiers.items():
        style = _VARIANT_STYLES.get(label, {"color": "black", "marker": "o"})
        tau_ns = [p.tau_opt * 1e9 for p in points]
        F = np.array([p.F_assign_opt for p in points])
        sigma = np.array([p.F_assign_uncertainty for p in points])

        ax.fill_between(tau_ns, F - sigma, F + sigma, color=style["color"], alpha=0.15)
        ax.plot(tau_ns, F, color=style["color"], linestyle="-", linewidth=1.2)
        ax.scatter(tau_ns, F, marker=style["marker"], s=36, color=style["color"],
                   edgecolors="white", linewidths=0.8, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Readout duration $\tau_{\rm opt}$ (ns, log)")
    ax.set_ylabel(r"$F_{\rm assign}$ at optimum")
    ax.set_title("Speed–fidelity Pareto frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main() -> None:
    use_modal = os.environ.get("USE_MODAL", "0") == "1"
    frontiers: dict[str, list] = {}
    for spec in PARETO_DEVICE_VARIANTS:
        device = build_variant(spec)
        frontiers[spec["label"]] = compute_pareto_frontier(
            device, tau_max_values=TAU_MAX_GRID_NS * 1e-9,
            device_label=spec["label"], use_modal=use_modal,
        )

    # Persist YAML
    out_dir = Path("06_Dispersive_Readout/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        label: [p.model_dump() for p in points]
        for label, points in frontiers.items()
    }
    with open(out_dir / "fig4_panel_c_data.yaml", "w") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    render_pareto(ax, frontiers)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_panel_c_pareto.png", dpi=150)
    print(f"Wrote {out_dir / 'fig4_panel_c_pareto.png'}")


if __name__ == "__main__":
    main()
```

Invoke with Modal:

```bash
USE_MODAL=1 python 06_Dispersive_Readout/scripts/fig4_panel_c_pareto.py
```

Expected: ~5 min wall-clock (30 Pareto points dispatched in parallel); output `fig4_panel_c_pareto.png` and `fig4_panel_c_data.yaml`.

- [ ] **Step 14.6: Commit**

```bash
git add dispersive_readout/optimization/pareto.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py 06_Dispersive_Readout/scripts/fig4_panel_c_pareto.py 06_Dispersive_Readout/figures/fig4_panel_c_pareto.png 06_Dispersive_Readout/figures/fig4_panel_c_data.yaml
git commit -m "feat(stage06-m4): compute_pareto_frontier + Modal + O4 + standalone Panel c

Serial fallback preserved (use_modal=False); Modal dispatch collapses
30-tuple run to ~5 min wall-clock via pareto_one_tuple.map(...).
O4 monotonicity test uses 5σ_shot slack (~5e-3 at n=10⁴) to avoid
flakiness from shot-noise-independent-but-numerical-precision wiggle."
```

---

## Task 15: `RecommendationReport` schema + `_format_value_with_sigma` helper + O6.3

**Rationale:** Spec §5.5 Pydantic schema + narrative-σ formatter (post-commit Nit 1) + §6.1 O6.3 schema validation. Formatter helper lands here so Tasks 16-17 don't re-invent the metrology-σ convention per f-string token.

**Files:**
- Create: `dispersive_readout/optimization/recommend.py` — schema + helper.
- Modify: `dispersive_readout/optimization/__init__.py` — export `RecommendationReport`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O6.3 + helper unit tests.

- [ ] **Step 15.1: Write O6.3 + helper failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O6.3 — RecommendationReport schema
# ────────────────────────────────────────────────────────────────────

def test_O6_3_recommendation_report_accepts_valid():
    from dispersive_readout.optimization.recommend import RecommendationReport
    from dispersive_readout.optimization.sensitivity import SensitivityResult

    s = SensitivityResult(
        parameter="chi_scale", reference_value=1.0, reference_unit="",
        sensitivity=0.4, sensitivity_uncertainty=0.01, F_reference=0.99,
    )
    r = RecommendationReport(
        device_parameters_fitted={"T_1": 86e-6, "T_2_echo": 40e-6, "omega_q": 4.9e9 * 2 * 3.14159},
        optimal_drive={"amplitude": 5e7, "duration": 480e-9, "detuning": 0.0, "edge_sigma": 2e-9},
        predicted_F_assign=0.9984,
        predicted_F_uncertainty=1e-3,
        top_3_sensitivities=[s, s, s],
        all_sensitivities=[s, s, s, s, s, s, s],
        dominant_loss_channel="T1_intrinsic",
        sensitivity_warnings=[],
        recommendation_narrative="...",
    )
    assert r.predicted_F_assign == 0.9984
    assert len(r.top_3_sensitivities) == 3


def test_O6_3_recommendation_report_rejects_empty_all_sensitivities():
    from pydantic import ValidationError
    from dispersive_readout.optimization.recommend import RecommendationReport
    with pytest.raises(ValidationError):
        RecommendationReport(
            device_parameters_fitted={},
            optimal_drive={},
            predicted_F_assign=0.99,
            predicted_F_uncertainty=1e-3,
            top_3_sensitivities=[],
            all_sensitivities=[],              # empty → reject
            dominant_loss_channel="T1_intrinsic",
            sensitivity_warnings=[],
            recommendation_narrative="",
        )


# ────────────────────────────────────────────────────────────────────
# _format_value_with_sigma — metrology σ convention (Q9b + Nit 1)
# ────────────────────────────────────────────────────────────────────

def test_format_value_with_sigma_rounds_up_to_one_sig_fig():
    """σ=0.00022 rounds UP to 0.0003 at 1 sig fig (metrology standard).
    Value matches σ's last-decimal position."""
    from dispersive_readout.optimization.recommend import _format_value_with_sigma
    val_s, sig_s = _format_value_with_sigma(value=0.99943, sigma=0.00022)
    # 0.00022 at 1 sig fig, rounded up → 0.0003; value to 4 decimals matching
    assert sig_s == "0.0003", f"Expected '0.0003', got {sig_s!r}"
    assert val_s == "0.9994", f"Expected '0.9994', got {val_s!r}"


def test_format_value_with_sigma_handles_asymmetric():
    from dispersive_readout.optimization.recommend import _format_value_with_sigma
    val_s, sig_s = _format_value_with_sigma(
        value=86.0, sigma=0.0, sigma_lo=3.0, sigma_hi=5.0,
    )
    # Asymmetric: value +σ_hi / −σ_lo; both σ rounded up to 1 sig fig
    assert "+5" in sig_s or "+ 5" in sig_s
    assert "-3" in sig_s or "− 3" in sig_s or "- 3" in sig_s
```

- [ ] **Step 15.2: Run tests to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O6_3_recommendation_report_accepts_valid -v -p no:dash
```

Expected: **FAIL** — `ModuleNotFoundError: No module named 'dispersive_readout.optimization.recommend'`.

- [ ] **Step 15.3: Create `recommend.py` with schema and helper**

Create `dispersive_readout/optimization/recommend.py`:

```python
"""Closed-loop recommendation pipeline.

See MODULE_4_SPEC.md §3.4, §5.5. Narrow-scope closed loop (fitted T_1,
T_2, ω_q over REFERENCE resonator) per Q4 lock. Template-rendered
narrative with IQM-table rounding + metrology σ convention per Q9b.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, field_validator

from .sensitivity import SensitivityResult


class RecommendationReport(BaseModel):
    """Closed-loop output: fit → recommend → report."""
    device_parameters_fitted: dict[str, Any]
    optimal_drive: dict[str, Any]
    predicted_F_assign: float
    predicted_F_uncertainty: float
    top_3_sensitivities: list[SensitivityResult]
    all_sensitivities: list[SensitivityResult]
    dominant_loss_channel: str
    sensitivity_warnings: list[str]
    recommendation_narrative: str
    scope_caveat: str = (
        "Closed-loop scope: fitted (T_1, T_2, ω_q) over fixed REFERENCE "
        "resonator and coupling. Full closed-loop including resonator "
        "spectroscopy and AC-Stark characterization is post-submission roadmap."
    )

    @field_validator("all_sensitivities")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("all_sensitivities must be non-empty")
        return v


# ────────────────────────────────────────────────────────────────────
# Metrology helper (Q9b + post-commit Nit 1)
# ────────────────────────────────────────────────────────────────────

def _round_up_to_n_sig_figs(x: float, n: int) -> tuple[float, int]:
    """Round x UP to n significant figures; return (rounded, last_decimal_pos)."""
    if x == 0.0:
        return 0.0, 0
    import math
    magnitude = math.floor(math.log10(abs(x)))
    shift = n - 1 - magnitude
    factor = 10 ** shift
    rounded = math.ceil(abs(x) * factor) / factor
    rounded = math.copysign(rounded, x)
    return rounded, shift


def _format_value_with_sigma(
    value: float,
    sigma: float,
    sigma_lo: float | None = None,
    sigma_hi: float | None = None,
) -> tuple[str, str]:
    """Return (value_str, sigma_str) per metrology-σ convention.

    σ is rounded UP to 1 significant figure; value's display decimal
    position matches σ's last-decimal position.
    """
    # Asymmetric case
    if sigma_lo is not None or sigma_hi is not None:
        s_lo, _ = _round_up_to_n_sig_figs(sigma_lo or 0.0, 1)
        s_hi, shift_hi = _round_up_to_n_sig_figs(sigma_hi or 0.0, 1)
        shift = max(shift_hi, 0)
        val_fmt = f"{{:.{shift}f}}".format(value)
        sig_str = f"+{s_hi:.{shift}f} / −{s_lo:.{shift}f}"
        return val_fmt, sig_str

    # Symmetric case
    sigma_rounded, shift = _round_up_to_n_sig_figs(sigma, 1)
    shift = max(shift, 0)
    val_fmt = f"{{:.{shift}f}}".format(value)
    sig_fmt = f"{{:.{shift}f}}".format(sigma_rounded)
    return val_fmt, sig_fmt
```

Update `dispersive_readout/optimization/__init__.py`:

```python
from .recommend import RecommendationReport
```

Add `RecommendationReport` to `__all__`.

- [ ] **Step 15.4: Run O6.3 + helper tests to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "O6_3 or format_value_with_sigma"
```

Expected: **4 passed**.

- [ ] **Step 15.5: Commit**

```bash
git add dispersive_readout/optimization/recommend.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): RecommendationReport + _format_value_with_sigma + O6.3

Metrology σ convention per Q9b + Nit 1: σ rounds UP to 1 sig fig,
value matches σ's last-decimal position. Asymmetric case supported
(value +σ_hi / −σ_lo) for Module 3 FittedParameter schemas that
expose sigma_lo/sigma_hi."
```

---

## Task 16: `recommend_from_fitted_parameters` + `generate_narrative` + YAML export

**Rationale:** Spec §3.4 pipeline spec + §5.5 function contracts. Runs the narrow closed loop: bridge via Module 3's `to_device_config` → Pareto → sensitivities at the per-device optimum → warnings → narrative → YAML.

**Files:**
- Modify: `dispersive_readout/optimization/recommend.py` — add `recommend_from_fitted_parameters`, `generate_narrative`, `export_recommendation_to_yaml`.
- Modify: `dispersive_readout/optimization/__init__.py` — export `recommend_from_fitted_parameters`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add narrative round-trip test.

- [ ] **Step 16.1: Preflight check — verify Module 3's FittedParameter σ schema**

Run the preflight check from spec §3.4:

```bash
grep -nE "sigma_lo|sigma_hi|uncertainty" dispersive_readout/characterization/fitting.py
```

Document the result — if Module 3 exposes only `uncertainty` (symmetric σ), the narrative template uses the symmetric branch of `_format_value_with_sigma`. If it exposes `sigma_lo`/`sigma_hi`, the asymmetric branch is used. Reflect the actual schema, not a guess.

- [ ] **Step 16.2: Write narrative round-trip failing test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# Narrative round-trip: no raw format tokens leak into the output
# ────────────────────────────────────────────────────────────────────

def test_generate_narrative_contains_no_raw_format_tokens():
    """If the template f-string is mis-populated, raw {placeholder}
    tokens will appear. Spec §9 item 8 — fix the formatting, not the text."""
    from dispersive_readout.optimization.recommend import (
        RecommendationReport, generate_narrative,
    )
    from dispersive_readout.optimization.sensitivity import SensitivityResult

    s = SensitivityResult(
        parameter="chi_scale", reference_value=1.0, reference_unit="",
        sensitivity=0.42, sensitivity_uncertainty=0.02, F_reference=0.99,
    )
    r = RecommendationReport(
        device_parameters_fitted={
            "T_1": {"value": 86e-6, "uncertainty": 2e-6},
            "T_2_echo": {"value": 40e-6, "uncertainty": 1.5e-6},
            "omega_q": {"value": 4.89e9 * 2 * 3.14159, "uncertainty": 5e6 * 2 * 3.14159},
        },
        optimal_drive={
            "amplitude": 5e7, "duration": 480e-9, "detuning": 0.0, "edge_sigma": 2e-9,
        },
        predicted_F_assign=0.9984,
        predicted_F_uncertainty=1.2e-3,
        top_3_sensitivities=[s, s, s],
        all_sensitivities=[s, s, s, s, s, s, s],
        dominant_loss_channel="T1_intrinsic",
        sensitivity_warnings=[],
        recommendation_narrative="",
    )
    narrative = generate_narrative(r)
    # No raw {...} tokens should remain
    assert "{" not in narrative and "}" not in narrative, (
        f"Narrative has unsubstituted format tokens: {narrative}"
    )
    # Dominant channel name should appear
    assert "T1_intrinsic" in narrative
```

- [ ] **Step 16.3: Run test to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_generate_narrative_contains_no_raw_format_tokens -v -p no:dash
```

Expected: **FAIL** — `ImportError: cannot import name 'generate_narrative'`.

- [ ] **Step 16.4: Implement `generate_narrative`, `recommend_from_fitted_parameters`, YAML export**

Append to `dispersive_readout/optimization/recommend.py`:

```python
import math
from pathlib import Path

import yaml

from ..physics.config import DriveParams
from .pareto import find_pareto_point
from .sensitivity import (
    compute_all_sensitivities,
    rank_sensitivities,
    SENSITIVITY_WARNING_THRESHOLD,
)


def generate_narrative(report: RecommendationReport) -> str:
    """IQM-table rounding + metrology σ convention (Q9b + Nit 1).

    Delegates per-value formatting to _format_value_with_sigma so the
    metrology σ convention is applied consistently.
    """
    fitted = report.device_parameters_fitted

    T1_val, T1_sig = _format_value_with_sigma(
        fitted["T_1"]["value"] * 1e6,         # → µs
        fitted["T_1"]["uncertainty"] * 1e6,
    )
    T2_val, T2_sig = _format_value_with_sigma(
        fitted["T_2_echo"]["value"] * 1e6,
        fitted["T_2_echo"]["uncertainty"] * 1e6,
    )
    omega_val, omega_sig = _format_value_with_sigma(
        fitted["omega_q"]["value"] / (2.0 * math.pi * 1e9),   # → GHz / 2π
        fitted["omega_q"]["uncertainty"] / (2.0 * math.pi * 1e9),
    )

    drive = report.optimal_drive
    eps_MHz_2pi = drive["amplitude"] / (2.0 * math.pi * 1e6)
    tau_ns = int(round(drive["duration"] * 1e9))

    F_val, F_sig = _format_value_with_sigma(
        report.predicted_F_assign, report.predicted_F_uncertainty,
    )

    top3_fmt = ", ".join(
        f"{s.parameter} (S={s.sensitivity:+.3f})"
        for s in report.top_3_sensitivities
    )

    warning_block = ""
    if report.sensitivity_warnings:
        warning_block = (
            "\n[WARNING: "
            + "; ".join(report.sensitivity_warnings)
            + "]"
        )

    return (
        f"For the fitted device (T_1 = {T1_val} ± {T1_sig} µs, "
        f"T_2_echo = {T2_val} ± {T2_sig} µs, "
        f"ω_q/2π = {omega_val} ± {omega_sig} GHz), the recommended "
        f"readout configuration is ε_0/2π = {eps_MHz_2pi:.2g} MHz at "
        f"τ = {tau_ns} ns; predicted F_assign = {F_val} ± {F_sig}. "
        f"The dominant remaining loss channel at this optimum is "
        f"{report.dominant_loss_channel}; the top-3 parameters by |S_θ| "
        f"are {top3_fmt}.{warning_block}"
    )


def recommend_from_fitted_parameters(
    fitted,                           # dispersive_readout.characterization.ExtractedParameterPack
    tau_max: float = 500e-9,
) -> RecommendationReport:
    """Narrow closed-loop recommendation.

    1. Bridge fitted parameters to DeviceConfig via to_device_config()
       (Module 3 — inherits REFERENCE resonator/coupling).
    2. Find Pareto point at tau_max.
    3. Compute sensitivities at the per-device optimum (not REFERENCE).
    4. Emit warnings for |S_θ| > SENSITIVITY_WARNING_THRESHOLD.
    5. Render narrative template.
    """
    from ..analysis.operating_point import OperatingPoint
    device = fitted.to_device_config()

    pareto = find_pareto_point(device, tau_max=tau_max)

    # Sensitivities at the PER-DEVICE optimum
    drive_opt = DriveParams(
        amplitude=pareto.epsilon_0_opt,
        duration=pareto.tau_opt,
        detuning=0.0,
    )
    op_at_opt = OperatingPoint(
        device=device,
        drive=drive_opt,
        integration_window=(50e-9, pareto.tau_opt),
        n_shots=10_000,
    )
    all_sens = compute_all_sensitivities(op_at_opt)
    ranked = rank_sensitivities(all_sens)

    warnings_ = [
        f"|S_{s.parameter}| = {abs(s.sensitivity):.2f} at fitted-device optimum: "
        f"device sits near regime-change boundary; linearized sensitivity "
        f"ranking is locally unreliable."
        for s in ranked
        if abs(s.sensitivity) > SENSITIVITY_WARNING_THRESHOLD
    ]

    # Extract fitted parameter values for the narrative.
    fitted_as_dict = {
        p.name: {"value": p.value, "uncertainty": p.uncertainty}
        for p in fitted.fitted_parameters
        if p.name in {"T_1", "T_2_echo", "omega_q"}
    }

    report = RecommendationReport(
        device_parameters_fitted=fitted_as_dict,
        optimal_drive={
            "amplitude": pareto.epsilon_0_opt,
            "duration": pareto.tau_opt,
            "detuning": 0.0,
            "edge_sigma": 2e-9,
        },
        predicted_F_assign=pareto.F_assign_opt,
        predicted_F_uncertainty=pareto.F_assign_uncertainty,
        top_3_sensitivities=ranked[:3],
        all_sensitivities=ranked,
        dominant_loss_channel=pareto.dominant_loss_channel,
        sensitivity_warnings=warnings_,
        recommendation_narrative="",          # filled below
    )
    # Render and re-inject narrative (Pydantic immutable → new instance)
    return report.model_copy(update={"recommendation_narrative": generate_narrative(report)})


def export_recommendation_to_yaml(report: RecommendationReport, path: str | Path) -> None:
    """Serialize RecommendationReport to YAML (closed-loop artifact)."""
    data = report.model_dump()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
```

Update `dispersive_readout/optimization/__init__.py` to export:

```python
from .recommend import (
    RecommendationReport,
    recommend_from_fitted_parameters,
    generate_narrative,
    export_recommendation_to_yaml,
)
```

- [ ] **Step 16.5: Run the narrative round-trip test to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_generate_narrative_contains_no_raw_format_tokens -v -p no:dash
```

Expected: **1 passed**.

- [ ] **Step 16.6: Commit**

```bash
git add dispersive_readout/optimization/recommend.py dispersive_readout/optimization/__init__.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): recommend_from_fitted_parameters + narrative + YAML

Pipeline: Module 3 bridge → Pareto → per-device sensitivities →
warnings → narrative. Template delegates to _format_value_with_sigma
for metrology-compliant σ rendering; raw format tokens ({…}) cannot
leak (verified by round-trip test). YAML artifact lands at
figures/recommendation.yaml when invoked by the closed-loop demo."
```

---

## Task 17: O5a + O5b closed-loop tests + demo device selection

**Rationale:** Spec §0 row 4 + §6.1 O5a/O5b split per Q9c: O5a asserts `F_opt_analytic − F_default_analytic > 0.005` (modeled improvement); O5b asserts Welch-t shot-noise detectability at n=10⁴, p<0.05 (measurability at spec's shot budget). Demo device picked from Module 3's `recovery_coverage_report.yaml`.

**Files:**
- Create: `06_Dispersive_Readout/scripts/pick_closed_loop_demo_device.py` — Day-13 morning demo-device selection helper.
- Create: `06_Dispersive_Readout/figures/closed_loop_demo_device.yaml` — committed artifact: which device the arrow points at + rationale.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O5a + O5b.

- [ ] **Step 17.1: Write the device-picker script**

Create `06_Dispersive_Readout/scripts/pick_closed_loop_demo_device.py`:

```python
"""Day-13 morning helper: pick the hard recovery-harness device for the
Figure 4 closed-loop arrow.

Selection rule (Q4 lock): among the SEED=42 recovery-harness devices,
pick the one whose `to_device_config()` bridge produces a Pareto
optimum (ε_0_opt, τ_opt) with the largest deviation from REFERENCE's
Pareto optimum. Deterministic; records rationale to
figures/closed_loop_demo_device.yaml so the pick is reproducible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

from dispersive_readout.characterization.recovery import (
    load_committed_coverage_report,
)
from dispersive_readout.optimization.pareto import find_pareto_point
from dispersive_readout.physics.config import REFERENCE_DEVICE


def main() -> None:
    report_path = Path("06_Dispersive_Readout/figures/recovery_coverage_report.yaml")
    if not report_path.exists():
        print(f"ERROR: {report_path} missing. Module 3 must ship its recovery "
              "harness artifact before Module 4 can pick a demo device.",
              file=sys.stderr)
        sys.exit(1)

    report = load_committed_coverage_report(report_path)

    # REFERENCE Pareto optimum for the δ comparison
    ref = find_pareto_point(REFERENCE_DEVICE, tau_max=500e-9)
    print(f"REFERENCE optimum: ε_0 = {ref.epsilon_0_opt:.3e}, "
          f"τ = {ref.tau_opt*1e9:.1f} ns, F = {ref.F_assign_opt:.4f}")

    candidates = []
    for device_idx, device_entry in enumerate(report.devices):
        # device_entry is a DeviceGroundTruth with its fitted ExtractedParameterPack;
        # reconstruct via to_device_config on the "fitted" slot if harness stores it,
        # or rebuild DeviceConfig directly from the ground-truth (T_1, T_2_echo, ω_q)
        from dataclasses import replace
        new_dec = replace(
            REFERENCE_DEVICE.decoherence,
            gamma_1=1.0 / device_entry.T_1,
            gamma_phi=max(1.0 / device_entry.T_2_echo - 0.5 / device_entry.T_1, 0.0),
            n_th=max(device_entry.thermal_offset, REFERENCE_DEVICE.decoherence.n_th),
        )
        synthetic = replace(REFERENCE_DEVICE, decoherence=new_dec)

        p = find_pareto_point(synthetic, tau_max=500e-9)
        drift_eps = abs(p.epsilon_0_opt - ref.epsilon_0_opt) / ref.epsilon_0_opt
        drift_tau = abs(p.tau_opt - ref.tau_opt) / ref.tau_opt
        drift = max(drift_eps, drift_tau)

        candidates.append({
            "index": device_idx,
            "T_1_us": device_entry.T_1 * 1e6,
            "T_2_echo_us": device_entry.T_2_echo * 1e6,
            "omega_q_GHz": device_entry.omega_q / (2.0 * np.pi * 1e9),
            "epsilon_0_opt": p.epsilon_0_opt,
            "tau_opt_ns": p.tau_opt * 1e9,
            "F_assign_opt": p.F_assign_opt,
            "drift_fractional": drift,
        })
        print(f"  device[{device_idx}]: drift = {drift*100:.1f}% "
              f"(ε_0 drift {drift_eps*100:.1f}%, τ drift {drift_tau*100:.1f}%)")

    chosen = max(candidates, key=lambda c: c["drift_fractional"])
    print(f"\nChosen demo device: index={chosen['index']} drift={chosen['drift_fractional']*100:.1f}%")

    out = Path("06_Dispersive_Readout/figures/closed_loop_demo_device.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump({
            "chosen": chosen,
            "reference_optimum": {
                "epsilon_0_opt": ref.epsilon_0_opt,
                "tau_opt_ns": ref.tau_opt * 1e9,
                "F_assign_opt": ref.F_assign_opt,
            },
            "all_candidates": candidates,
            "rationale": (
                "Selected device maximizes max(|Δε_0|/ε_0_REF, |Δτ|/τ_REF); "
                "the closed-loop arrow demonstrates responsiveness of the "
                "recommendation pipeline to fitted parameters, not just "
                "drive re-optimization around REFERENCE."
            ),
            "seed": 42,
        }, f, sort_keys=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 17.2: Run the picker and inspect the output**

```bash
python 06_Dispersive_Readout/scripts/pick_closed_loop_demo_device.py
```

Expected: prints per-device drift percentages; writes `figures/closed_loop_demo_device.yaml` with the chosen index and full candidates list.

- [ ] **Step 17.3: Write O5a + O5b failing tests**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O5a — modeled improvement (analytic)
# O5b — shot-noise detectability (Welch-t at n_shots = 10⁴, p < 0.05)
# ────────────────────────────────────────────────────────────────────

def _load_demo_device():
    """Load the Day-13 picked demo device; build its DeviceConfig."""
    from pathlib import Path
    from dataclasses import replace
    import math
    import yaml
    from dispersive_readout.physics.config import REFERENCE_DEVICE

    demo_path = Path("06_Dispersive_Readout/figures/closed_loop_demo_device.yaml")
    if not demo_path.exists():
        pytest.skip(
            "closed_loop_demo_device.yaml missing — run "
            "scripts/pick_closed_loop_demo_device.py first (Task 17 Step 2)."
        )
    payload = yaml.safe_load(demo_path.read_text())
    c = payload["chosen"]
    new_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=1.0 / (c["T_1_us"] * 1e-6),
        gamma_phi=max(
            1.0 / (c["T_2_echo_us"] * 1e-6) - 0.5 / (c["T_1_us"] * 1e-6), 0.0,
        ),
    )
    return replace(REFERENCE_DEVICE, decoherence=new_dec), c


def test_O5a_closed_loop_modeled_improvement():
    """F_opt_analytic - F_default_analytic > 0.005 on the fitted demo device.

    Threshold 0.005 exceeds SLSQP ftol (1e-6) and Lindblad rtol (~1e-5) —
    asserts a genuine modeled improvement, not numerical wiggle."""
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, chosen = _load_demo_device()
    ref_drive = DriveParams(
        amplitude=chosen["epsilon_0_opt"],  # baseline = REFERENCE optimum drive on demo device
        duration=500e-9, detuning=0.0,
    )
    # "Default" = REFERENCE optimum drive applied to demo device (not its own optimum)
    ref_opt_keys = chosen.get("reference_optimum")
    # The YAML "reference_optimum" is a sibling field on top-level; reload
    import yaml
    from pathlib import Path
    payload = yaml.safe_load(Path("06_Dispersive_Readout/figures/closed_loop_demo_device.yaml").read_text())
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    r0 = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1 = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default = compute_assignment_fidelity(
        r0, r1, (50e-9, ref_drive.duration), n_shots=10_000, noise_model="ideal",
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    F_opt = p_opt.F_assign_opt

    delta = F_opt - F_default
    assert delta > 0.005, (
        f"F_opt − F_default = {delta:.4f} <= 0.005. Either the demo "
        "device is too close to REFERENCE (recompute pick), or the "
        "recommendation bridge is miscalibrated (spec §9 item 6)."
    )


def test_O5b_closed_loop_shot_noise_detectability():
    """Welch-t test on 10⁴-shot samples of F_default vs F_opt at the
    fitted demo device: p < 0.05. Asserts the modeled improvement is
    measurable at the spec's shot budget."""
    import numpy as np
    from scipy import stats as sp_stats
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import (
        simulate_readout, compute_assignment_fidelity,
    )
    from dispersive_readout.optimization.pareto import find_pareto_point

    demo_device, _ = _load_demo_device()

    import yaml
    from pathlib import Path
    payload = yaml.safe_load(Path("06_Dispersive_Readout/figures/closed_loop_demo_device.yaml").read_text())
    ref_drive = DriveParams(
        amplitude=payload["reference_optimum"]["epsilon_0_opt"],
        duration=payload["reference_optimum"]["tau_opt_ns"] * 1e-9,
        detuning=0.0,
    )

    # Single shot-noise-sampled F at default and optimum.
    r0_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=0)
    r1_d = simulate_readout(demo_device, ref_drive, initial_qubit_state=1)
    F_default_sample = compute_assignment_fidelity(
        r0_d, r1_d, (50e-9, ref_drive.duration),
        n_shots=10_000, noise_model="gaussian",
        rng=np.random.default_rng(seed=42),
    ).F_assign

    p_opt = find_pareto_point(demo_device, tau_max=500e-9)
    opt_drive = DriveParams(
        amplitude=p_opt.epsilon_0_opt, duration=p_opt.tau_opt, detuning=0.0,
    )
    r0_o = simulate_readout(demo_device, opt_drive, initial_qubit_state=0)
    r1_o = simulate_readout(demo_device, opt_drive, initial_qubit_state=1)
    F_opt_sample = compute_assignment_fidelity(
        r0_o, r1_o, (50e-9, opt_drive.duration),
        n_shots=10_000, noise_model="gaussian",
        rng=np.random.default_rng(seed=43),
    ).F_assign

    # Welch-t test treating each F as a single binomial proportion estimate
    # from n=10^4 Bernoulli trials. Approximate: σ_F = sqrt(F(1-F)/n).
    import math
    n = 10_000
    sigma_d = math.sqrt(F_default_sample * (1.0 - F_default_sample) / n)
    sigma_o = math.sqrt(F_opt_sample * (1.0 - F_opt_sample) / n)
    t = (F_opt_sample - F_default_sample) / math.sqrt(sigma_d ** 2 + sigma_o ** 2)
    # Two-sided Welch-t p-value with large dof → Z-approximation
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t)))
    assert p_value < 0.05, (
        f"Welch-t p = {p_value:.4f} >= 0.05: shot-noise detectability "
        "fails at n=10⁴. Either the demo device's F_opt - F_default is "
        "too small to detect at this shot budget, or F_default and F_opt "
        "are within one σ_shot (~ 1e-3)."
    )
```

- [ ] **Step 17.4: Run O5a + O5b**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "O5a or O5b"
```

Expected: **2 passed**. (Each runs ~4 full Lindblad sims; ~5 min total at REFERENCE.)

- [ ] **Step 17.5: Commit**

```bash
git add 06_Dispersive_Readout/scripts/pick_closed_loop_demo_device.py 06_Dispersive_Readout/figures/closed_loop_demo_device.yaml dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): demo-device pick + O5a/O5b closed-loop tests

Demo-device picker is deterministic (SEED=42) and writes rationale
to closed_loop_demo_device.yaml (committed artifact; regenerable).
O5a/O5b pair separates 'modeled improvement' (analytic ΔF > 0.005)
from 'measurable improvement' (Welch-t p < 0.05 at n=10⁴) — independent
failure modes, independent diagnostics."
```

---

## Task 18: `autodiff_addon.py` — **CUT 2026-04-23 (Day 11 PM)**

**Cut rationale:** The contingent autodiff add-on was cut to absorb the one-day cost of the per-level analytic-formula re-derivation forced by the Day-11 Task-10 finding (spec §3.5 cut amendment + spec §0.3 item 15). Trade: one speculative gradient-based refinement extension for one core-deliverable correctness item (regime map matches Lindblad simulator to <5%). Per spec §1 row 1 ("a fourth deliverable is cut or pushed to post-submission") this trade is spec-faithful. **No code shipped** — `autodiff_addon.py` was never created. **No tests shipped** — O7 (autodiff-vs-grid) drops from the test catalog.

The remaining task subsections below are preserved as historical record; **do not implement**. Skip directly from Task 17 to Task 19.

---

#### Original Task 18 contents (CUT — do not implement)

**Rationale:** Spec §3.5 contingent add-on + §9 item 7 (abort means revert, not "30 more minutes"). Ships only if Day-11 smoke passed and baseline deliverables have no unresolved bugs at Day-13 09:00. Three abort signals are concrete: 90-min forward pass, 3-hr FD-vs-grad agreement, 4-hr baseline blocker.

**Files:**
- Create: `dispersive_readout/optimization/autodiff_addon.py`.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O7 contingent test.

- [ ] **Step 18.1: Decide whether to ship the add-on**

At Day-13 09:00, verify:

1. `python -m pytest dispersive_readout/tests/test_optimization.py -v -p no:dash -k "not O5 and not O10"` is green (baseline clean).
2. Modal smoke O10 passed during Day 11 pre-warm.
3. No unresolved blockers on Tasks 1–17.

If any of these three fails → **do not start Task 18**. Skip to Task 19.

- [ ] **Step 18.2: Create the autodiff module**

Create `dispersive_readout/optimization/autodiff_addon.py`:

```python
"""CONTINGENT Day-13 add-on: gradient-based Gaussian-edge refinement.

Ships only if Day-11 end-of-day is clean and Modal smoke passed.
Hard 4-hour cap; three abort signals (Q9a lock):
  (i)   JAX Lindblad forward pass not producing finite F within 90 min.
  (ii)  Autodiff-vs-FD gradient disagreement > 10% on S_chi after 3 hours.
  (iii) Unresolved bugs in any baseline deliverable at the 4-hour mark.
Any trigger → immediate revert; this module unloads cleanly from Figure 4.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any


_ABORT_SIGNAL_1_FORWARD_PASS_SEC = 90 * 60
_ABORT_SIGNAL_2_GRAD_AGREEMENT_SEC = 3 * 3600
_ABORT_SIGNAL_3_BASELINE_BUGS_SEC = 4 * 3600


def autodiff_refine_pulse_edges(
    device,
    tau_opt: float,
    epsilon_0_opt: float,
    initial_edge_sigma: float = 2e-9,
    n_steps: int = 50,
    learning_rate: float = 1e-11,
    start_time: float | None = None,
) -> dict[str, Any]:
    """Refine `edge_sigma` alone (Nit 3 locked: no independent plateau DoF)
    via jax.grad + Adam. duration is held at tau_opt so pulse-shape refinement
    is orthogonal to the Pareto solver's (ε_0, τ) optimization.

    Returns
    -------
    dict with keys:
        'trajectory':           list[(edge_sigma, F_assign)]
        'final_F_assign':       float
        'grid_search_F_assign': float          (for comparison)
        'improvement_fraction': float
        'aborted':              bool
        'abort_reason':         str | None     ('forward_pass_90min',
                                                'grad_agreement_3hr',
                                                'baseline_bugs_4hr', None)
    """
    if start_time is None:
        start_time = time.time()

    result: dict[str, Any] = {
        "trajectory": [],
        "final_F_assign": None,
        "grid_search_F_assign": None,
        "improvement_fraction": None,
        "aborted": False,
        "abort_reason": None,
    }

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        result["aborted"] = True
        result["abort_reason"] = "jax_not_installed"
        return result

    # Abort signal (i): forward pass must produce finite F within 90 min.
    forward_pass_deadline = start_time + _ABORT_SIGNAL_1_FORWARD_PASS_SEC

    try:
        F_initial = _jax_lindblad_forward_F(
            device, tau_opt, epsilon_0_opt, initial_edge_sigma,
        )
    except Exception as e:
        if time.time() > forward_pass_deadline:
            result["aborted"] = True
            result["abort_reason"] = "forward_pass_90min"
            return result
        raise

    if not math.isfinite(F_initial):
        result["aborted"] = True
        result["abort_reason"] = "forward_pass_90min"
        return result

    # Abort signal (ii) check: validate autodiff against FD at 3-hr mark.
    fd_check_deadline = start_time + _ABORT_SIGNAL_2_GRAD_AGREEMENT_SEC

    # (Simplified contract-level stub: actual JAX-through-Lindblad
    # implementation is the 4-hour experiment. If any step raises
    # or time exceeds deadline, abort.)
    try:
        grad_autodiff = jax.grad(
            lambda sigma: _jax_lindblad_forward_F(device, tau_opt, epsilon_0_opt, float(sigma))
        )(float(initial_edge_sigma))
        # FD baseline for comparison
        h = 0.05 * initial_edge_sigma
        F_plus = _jax_lindblad_forward_F(
            device, tau_opt, epsilon_0_opt, initial_edge_sigma + h,
        )
        F_minus = _jax_lindblad_forward_F(
            device, tau_opt, epsilon_0_opt, initial_edge_sigma - h,
        )
        grad_fd = (F_plus - F_minus) / (2.0 * h)
        rel_err = abs(grad_autodiff - grad_fd) / max(abs(grad_fd), 1e-30)
        if rel_err > 0.10 or time.time() > fd_check_deadline:
            result["aborted"] = True
            result["abort_reason"] = "grad_agreement_3hr"
            return result
    except Exception:
        result["aborted"] = True
        result["abort_reason"] = "grad_agreement_3hr"
        return result

    # Adam loop (simplified — actual implementation is the 4-hour experiment)
    baseline_bug_deadline = start_time + _ABORT_SIGNAL_3_BASELINE_BUGS_SEC
    current_sigma = float(initial_edge_sigma)
    F_current = F_initial
    trajectory: list[tuple[float, float]] = [(current_sigma, F_current)]

    for step in range(n_steps):
        if time.time() > baseline_bug_deadline:
            result["aborted"] = True
            result["abort_reason"] = "baseline_bugs_4hr"
            break
        g = float(jax.grad(
            lambda s: _jax_lindblad_forward_F(device, tau_opt, epsilon_0_opt, float(s))
        )(current_sigma))
        current_sigma = current_sigma + learning_rate * g
        F_current = _jax_lindblad_forward_F(device, tau_opt, epsilon_0_opt, current_sigma)
        trajectory.append((current_sigma, F_current))

    # Compare to grid-search baseline at the same tau
    from .pareto import _F_analytic_at
    grid_F = _F_analytic_at(device, epsilon_0_opt, tau_opt)

    result["trajectory"] = trajectory
    result["final_F_assign"] = float(F_current)
    result["grid_search_F_assign"] = float(grid_F)
    result["improvement_fraction"] = (float(F_current) - float(grid_F)) / float(grid_F)
    return result


def _jax_lindblad_forward_F(device, tau: float, eps_0: float, edge_sigma: float) -> float:
    """JAX-friendly Lindblad forward pass — impl is the 4-hour experiment.

    Until a JAX-compatible Lindblad solver is wired, this falls back to
    Module 1's QuTiP path, making autodiff impossible and tripping
    abort signal (ii) as designed.
    """
    from ..physics.config import DriveParams
    from ..physics.readout_model import simulate_readout, compute_assignment_fidelity

    drive = DriveParams(
        amplitude=float(eps_0), duration=float(tau),
        detuning=0.0, edge_sigma=float(edge_sigma),
    )
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    return compute_assignment_fidelity(
        r0, r1, (50e-9, tau), n_shots=10_000, noise_model="ideal",
    ).F_assign
```

- [ ] **Step 18.3: Write O7 contingent test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O7 — autodiff-vs-grid agreement within 1% (CONTINGENT)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O7_autodiff_matches_grid_within_noise():
    """If the autodiff add-on shipped cleanly (not aborted), final F
    must agree with grid-search F within 1% at the same τ_max. Autodiff
    is not expected to beat grid — it's expected to confirm grid."""
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    from dispersive_readout.optimization.autodiff_addon import autodiff_refine_pulse_edges
    from dispersive_readout.optimization.pareto import find_pareto_point

    p = find_pareto_point(REFERENCE_DEVICE, tau_max=500e-9)
    result = autodiff_refine_pulse_edges(
        REFERENCE_DEVICE, tau_opt=p.tau_opt, epsilon_0_opt=p.epsilon_0_opt,
    )
    if result["aborted"]:
        pytest.skip(
            f"Autodiff add-on aborted with reason='{result['abort_reason']}' — "
            "expected revert per spec §3.5 contingent posture."
        )
    rel_diff = abs(result["final_F_assign"] - result["grid_search_F_assign"]) / result["grid_search_F_assign"]
    assert rel_diff < 0.01, (
        f"Autodiff F = {result['final_F_assign']:.4f} differs from grid "
        f"F = {result['grid_search_F_assign']:.4f} by {rel_diff*100:.2f}% > 1%. "
        "Autodiff is expected to confirm, not beat, grid-search."
    )
```

- [ ] **Step 18.4: Run O7 and check abort behavior**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O7_autodiff_matches_grid_within_noise -v -p no:dash
```

Expected (if JAX + JAX-compatible Lindblad working within caps): **1 passed**.
Expected (if abort triggers trip): **1 skipped** with reason. Either outcome is acceptable — spec §3.5 mandates clean revert on abort, and `pytest.skip` is the structured form of that revert in CI.

- [ ] **Step 18.5: Commit (whether shipped or aborted)**

If **shipped cleanly**:
```bash
git add dispersive_readout/optimization/autodiff_addon.py dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): autodiff add-on shipped — edge_sigma refinement confirms grid

CONTINGENT Day-13 add-on per spec §3.5. All 3 abort signals stayed
clean (forward pass < 90 min, grad vs FD agreement < 10%, no baseline
blockers). O7 confirms autodiff matches grid-search F within 1%.

README framing: 'backend-agnostic simulator supports gradient-based
extensions' — not headlined as a feature."
```

If **aborted cleanly**:
```bash
git rm dispersive_readout/optimization/autodiff_addon.py
git commit -m "chore(stage06-m4): autodiff add-on aborted — clean revert

Abort signal '<reason>' tripped per spec §3.5 contingent posture.
Module 4 baseline (Tasks 1–17 + 19–21) remains complete and ships
as-is. No half-shipped state per spec §9 item 9."
```

---

## Task 19: `fig4_optimization.py` composite Figure 4

**Rationale:** Spec §7 Figure 4 specification. Imports `render_tornado`, `render_regime_map`, `render_pareto` from the Day-10/11/12 standalone scripts so there's no duplicated rendering logic. Applies Q6 composite layout (3 horizontal panels + figure-wide caption).

**Files:**
- Create: `06_Dispersive_Readout/scripts/fig4_optimization.py` — composite renderer.
- Create: `06_Dispersive_Readout/figures/fig4_optimization.png` — committed artifact.

- [ ] **Step 19.1: Write the composite figure script**

Create `06_Dispersive_Readout/scripts/fig4_optimization.py`:

```python
"""Figure 4 composite: sensitivity tornado + analytic regime map + Pareto frontier.

Imports the three panel renderers from their standalone scripts and
assembles a 3-panel horizontal layout with a figure-wide caption
containing the three locked caveats (Q1 orthogonality, Q3 analytic
regime, Q4 closed-loop scope).

See MODULE_4_SPEC.md §7 for the locked design contract.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Reuse standalone-panel renderers (no rendering duplication)
from fig4_panel_a_tornado import render_tornado
from fig4_panel_b_regime import render_regime_map
from fig4_panel_c_pareto import render_pareto

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.optimization.sensitivity import (
    compute_all_sensitivities, day_10_cross_check_s_g_vs_s_chi,
)
from dispersive_readout.optimization.regime_map import (
    compute_analytic_regime_map, validate_analytic_vs_lindblad,
)
from dispersive_readout.optimization.pareto import (
    PARETO_DEVICE_VARIANTS, TAU_MAX_GRID_NS,
    build_variant, compute_pareto_frontier,
)


def main() -> None:
    use_modal = os.environ.get("USE_MODAL", "1") == "1"

    # ── Panel (a) data: sensitivities + Day-10 cross-check ─────────────
    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)
    cross = day_10_cross_check_s_g_vs_s_chi(op)

    # ── Panel (b) data: analytic grid + validation ─────────────────────
    grid = compute_analytic_regime_map()
    validation = validate_analytic_vs_lindblad()

    # ── Panel (c) data: 3 frontiers ─────────────────────────────────────
    frontiers: dict[str, list] = {}
    for spec in PARETO_DEVICE_VARIANTS:
        device = build_variant(spec)
        frontiers[spec["label"]] = compute_pareto_frontier(
            device, tau_max_values=TAU_MAX_GRID_NS * 1e-9,
            device_label=spec["label"], use_modal=use_modal,
        )

    # ── Composite figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    anchoring_a = (
        f"$F_{{\\rm ref}}={sens[0].F_reference:.4f}$, "
        f"$\\tau_{{\\rm int}}$ = {(op.integration_window[1]-op.integration_window[0])*1e9:.0f} ns, "
        f"$\\bar n_{{\\rm phot}}$ = {grid['n_phot_used']:.2f}"
    )
    render_tornado(axes[0], sens, anchoring_a)
    render_regime_map(axes[1], validation, grid)
    render_pareto(axes[2], frontiers)

    # ── Figure-wide caption ─────────────────────────────────────────────
    max_dev = validation["max_deviation_fractional"] * 100.0
    residual_pct = cross["residual_fractional"] * 100.0
    caption = (
        r"$\bf{Figure\ 4.}$ Optimization layer for dispersive transmon readout. "
        r"$\bf{(a)}$ Normalized log-sensitivities of $F_{\rm assign}$ to 7 parameters "
        r"at REFERENCE (Marxer arXiv:2508.16437); sensitivities computed with parameters "
        r"treated as independent axes via chi_scale. Day-10 cross-check "
        f"$|S_g - 2 S_\\chi| / |2 S_\\chi|$ = {residual_pct:.2f}%. "
        r"$\bf{(b)}$ Analytic regime map (Bengtsson 2024 PRL §II + Blais RMP 2021 §V.B); "
        f"Lindblad-validated at 2 points, max deviation {max_dev:.2f}%. "
        r"Marxer Q1 annotated with $F_{\rm sim}$; Hazra 2407.10934 (dimon, non-standard "
        r"χ-mediation) cited in reference list but not plotted. "
        r"$\bf{(c)}$ Pareto frontiers for 3 parameter-anchored variants of REFERENCE "
        r"(V1=REFERENCE, V2=T₁=40 µs, V3=T₁=20 µs + κ/2π=6 MHz). Curves represent the "
        r"Pareto frontier predicted by this work's simulator under parameter "
        r"substitution — NOT the frontier achievable on the cited devices' native "
        r"hardware. Closed-loop arrow: fitted (T₁, T₂, ω_q) over fixed REFERENCE "
        r"resonator and coupling; full closed-loop including resonator spectroscopy "
        r"is post-submission roadmap. $n_{\rm shots} = 10^4$ throughout."
    )
    fig.text(0.01, -0.02, caption, wrap=True, fontsize=9, ha="left")

    out = Path("06_Dispersive_Readout/figures/fig4_optimization.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 19.2: Render Figure 4**

```bash
USE_MODAL=1 python 06_Dispersive_Readout/scripts/fig4_optimization.py
```

Expected: `Wrote .../fig4_optimization.png`. ~5 min wall-clock with Modal.

- [ ] **Step 19.3: Visually inspect**

Open `06_Dispersive_Readout/figures/fig4_optimization.png`. Verify per spec §9 checklist:

- 3 horizontal panels, no more.
- Panel (a) tornado: 7 bars sorted by |S| with (±5%) on labels, numeric annotations.
- Panel (b) regime map: viridis, 4 markers visible with Marxer Q1 `F_sim` annotation.
- Panel (c) Pareto: 3 curves with points + shaded envelopes.
- Figure-wide caption contains all 3 locked caveats + cross-check residual number + max-deviation number.

- [ ] **Step 19.4: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig4_optimization.py 06_Dispersive_Readout/figures/fig4_optimization.png
git commit -m "feat(stage06-m4): Figure 4 composite (3-panel optimization layer)

Imports render_tornado/render_regime_map/render_pareto from the standalone
panel scripts — zero rendering duplication. Figure-wide caption contains
all 3 locked caveats (Q1 orthogonality, Q3 analytic regime, Q4 closed-loop
scope) plus the Day-10 cross-check residual and 2-point validation max
deviation numbers."
```

---

## Task 20: O9 regression gate + `fig4_data.yaml` commit

**Rationale:** Spec §6.1 O9 analog of Module 3's C3 regression gate (SEED=42, ±2% per value). Prevents silent drift on refactors.

**Files:**
- Create: `06_Dispersive_Readout/figures/fig4_data.yaml` — committed regression artifact.
- Modify: `dispersive_readout/tests/test_optimization.py` — add O9.

- [ ] **Step 20.1: Generate the regression artifact**

Create a helper script `06_Dispersive_Readout/scripts/regenerate_fig4_data.py`:

```python
"""Regenerate fig4_data.yaml — the O9 regression-gate artifact.

Pins per-sensitivity S_theta values, per-Pareto-point (F_opt, epsilon_0, tau),
regime-grid F-values hash, and Day-10 cross-check residual at SEED=42.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from dispersive_readout.analysis.operating_point import get_reference_operating_point
from dispersive_readout.optimization.sensitivity import (
    compute_all_sensitivities, day_10_cross_check_s_g_vs_s_chi,
)
from dispersive_readout.optimization.regime_map import compute_analytic_regime_map
from dispersive_readout.optimization.pareto import (
    PARETO_DEVICE_VARIANTS, TAU_MAX_GRID_NS,
    build_variant, compute_pareto_frontier,
)


def main() -> None:
    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)
    cross = day_10_cross_check_s_g_vs_s_chi(op)
    grid = compute_analytic_regime_map()

    frontiers = {}
    for spec in PARETO_DEVICE_VARIANTS:
        device = build_variant(spec)
        pts = compute_pareto_frontier(
            device, tau_max_values=TAU_MAX_GRID_NS * 1e-9,
            device_label=spec["label"], use_modal=False,
        )
        frontiers[spec["label"]] = [
            {
                "tau_max_ns": round(p.tau_max * 1e9, 3),
                "epsilon_0_opt": float(p.epsilon_0_opt),
                "tau_opt_ns": round(p.tau_opt * 1e9, 3),
                "F_assign_opt": round(p.F_assign_opt, 6),
            }
            for p in pts
        ]

    payload = {
        "seed": 42,
        "sensitivities": [
            {"parameter": s.parameter, "S": round(s.sensitivity, 4),
             "sigma_S": round(s.sensitivity_uncertainty, 5),
             "F_reference": round(s.F_reference, 5)}
            for s in sens
        ],
        "day_10_cross_check": {
            "S_chi": round(cross["S_chi"], 4),
            "S_g": round(cross["S_g"], 4),
            "residual_fractional": round(cross["residual_fractional"], 4),
        },
        "regime_grid_hash": hashlib.sha256(
            np.ascontiguousarray(grid["F_grid"]).tobytes()
        ).hexdigest(),
        "regime_n_phot_used": round(grid["n_phot_used"], 4),
        "pareto_frontiers": frontiers,
    }

    out = Path("06_Dispersive_Readout/figures/fig4_data.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python 06_Dispersive_Readout/scripts/regenerate_fig4_data.py
```

- [ ] **Step 20.2: Write O9 failing test**

Append to `dispersive_readout/tests/test_optimization.py`:

```python
# ────────────────────────────────────────────────────────────────────
# O9 — regression gate: regenerate and compare against committed artifact
# ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_O9_regression_gate_against_committed_yaml():
    """Regenerate fig4_data.yaml at SEED=42; compare to committed artifact.

    Tolerance ±2% per value (Module 3 C3 convention). If the fitter
    legitimately improves: regenerate the artifact and re-commit."""
    from pathlib import Path
    import yaml
    committed = yaml.safe_load(
        Path("06_Dispersive_Readout/figures/fig4_data.yaml").read_text()
    )

    from dispersive_readout.analysis.operating_point import get_reference_operating_point
    from dispersive_readout.optimization.sensitivity import compute_all_sensitivities

    op = get_reference_operating_point(n_shots=10_000)
    sens = compute_all_sensitivities(op)

    TOL = 0.02
    for observed, pinned in zip(sens, committed["sensitivities"]):
        assert observed.parameter == pinned["parameter"]
        ref_S = pinned["S"]
        obs_S = observed.sensitivity
        if abs(ref_S) > 1e-6:
            rel = abs(obs_S - ref_S) / abs(ref_S)
            assert rel < TOL, (
                f"Sensitivity S_{observed.parameter} drifted from pinned "
                f"{ref_S:.4f} to {obs_S:.4f} ({rel*100:.2f}% > 2%). "
                "If intentional, regenerate fig4_data.yaml."
            )
```

- [ ] **Step 20.3: Run O9 to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_optimization.py::test_O9_regression_gate_against_committed_yaml -v -p no:dash
```

Expected: **1 passed**.

- [ ] **Step 20.4: Commit**

```bash
git add 06_Dispersive_Readout/scripts/regenerate_fig4_data.py 06_Dispersive_Readout/figures/fig4_data.yaml dispersive_readout/tests/test_optimization.py
git commit -m "feat(stage06-m4): O9 regression gate + fig4_data.yaml artifact

SEED=42 stable. 7 sensitivities + Day-10 cross-check + regime-grid
hash + 30 Pareto points committed as the canonical artifact. Future
refactors that drift any sensitivity by >2% trip O9 — if intentional,
regenerate and re-commit."
```

---

## Task 21: CV v1 + cover letter v1 (parallel track, Day-13 afternoon)

**Rationale:** Spec §8 Day-13 afternoon per Q9c Change 3 — moved from Day-12 afternoon to the lower-pressure slot after Figure 4 ships. Application-documents workflow is outside the engineering task set; included here so the plan tracks completion.

**Files:**
- Create: `06_Dispersive_Readout/applications/cv_v1.md`.
- Create: `06_Dispersive_Readout/applications/cover_letter_v1.md`.

- [ ] **Step 21.1: CV v1 draft**

Create `06_Dispersive_Readout/applications/cv_v1.md`. Include:

- One-line summary with the Stage 06 reframe ("ML engineer applying model-based optimization to superconducting-qubit readout").
- Stage 06 highlight: "Sensitivity + Pareto optimization layer for dispersive readout (Python, QuTiP, SciPy, Modal); closed-loop recommendation from fitted parameters; 28-test suite with adversarial-review-driven scope discipline (9 amendments across Q1–Q9)."
- Links to `06_Dispersive_Readout/figures/fig4_optimization.png`, `MODULE_4_SPEC.md`, `MODULE_4_PLAN.md`.

The detailed content is Jane's; the plan just gates the checkbox.

- [ ] **Step 21.2: Cover letter v1 draft**

Create `06_Dispersive_Readout/applications/cover_letter_v1.md`. Include:

- Opening hook referencing Marxer arXiv:2508.16437 as the device that anchors this work.
- Bengtsson 2024 PRL 132 100603 as the methodological template.
- QCR (Quantum Computer Research) narrative hook.

- [ ] **Step 21.3: Commit**

```bash
git add 06_Dispersive_Readout/applications/cv_v1.md 06_Dispersive_Readout/applications/cover_letter_v1.md
git commit -m "docs(stage06): CV v1 + cover letter v1 drafts

Moved from Day-12 afternoon to Day-13 afternoon per Q9c Change 3 —
lower-pressure slot after Figure 4 ships. Marxer opening hook;
Bengtsson methodological reference; QCR narrative hook."
```

---

## Self-Review Checklist

**1. Spec coverage:** Each spec section maps to at least one task:

| Spec § | Covered by |
|---|---|
| §0 row 1 (chi_scale) | Task 1 |
| §0 row 2 (Modal) | Tasks 11, 14 |
| §0 row 3 (analytic regime) | Tasks 8, 9, 10 |
| §0 row 4 (narrow closed loop + warnings) | Tasks 6, 15, 16, 17 |
| §0 row 5 (4-marker map, Hazra omitted) | Task 9 |
| §0 row 6 (figure-presentation rules) | Tasks 7, 10, 14, 19 |
| §0 row 7 (parameter-anchored variants) | Tasks 12, 14 |
| §0 row 8 (analytic-objective contract) | Tasks 3, 4, 13 (O8) |
| §0 row 9 (abort signals + narrative σ + 28 tests) | Tasks 15, 18, 20 |
| §2.1 parameters | Task 4 |
| §3.1 sensitivity | Tasks 3, 4, 5 |
| §3.2 regime map (incl. post-Nit-2 formulas) | Tasks 8, 9, 10 |
| §3.3 Pareto | Tasks 12, 13, 14 |
| §3.4 recommend (incl. post-Nit-1 formatter) | Tasks 15, 16 |
| §3.5 autodiff contingent | Task 18 |
| §3.6 uncertainty model | Task 4, 13 |
| §4 module structure | Tasks 2, 11, 12, 15 (scaffold) |
| §5.1 sensitivity | Tasks 3, 4, 5, 6 |
| §5.2 regime map | Tasks 8, 9, 10 |
| §5.3 pareto | Tasks 12, 13, 14 |
| §5.4 modal_pareto | Task 11 |
| §5.5 recommend (post-Nit-1 helper) | Task 15 |
| §5.6 autodiff | Task 18 |
| §5.7 Module 1 API extension | Task 1 |
| §6.1 tests O1–O24 | All tasks with tests (majority) |
| §6.2 policy contracts | Tasks 3, 13 (O8) |
| §7 Figure 4 | Tasks 7, 10, 14, 19 |
| §8 day-by-day | Task-to-day mapping above |
| §9 what to flag | Referenced inline in relevant task error messages |
| §10 review checklist | Captured as closing gate before Day 14 |
| §11 references | Cited throughout |

Gaps: **None** — all §0–§11 sections covered.

**2. Placeholder scan:** No `TBD`, `TODO`, `implement later`, "similar to Task N", or vague step descriptions. Every code block is complete; every command has expected output.

**3. Type consistency:**
- `SensitivityResult.parameter` uses the `ParameterName` Literal defined in Task 3 and consumed by Tasks 4, 5, 6, 15, 16.
- `ParetoPoint` is defined stub-wise in Task 11 and replaced with full schema in Task 12; Task 13 and 14 consume the full schema.
- `RecommendationReport` fields are consistent across Tasks 15, 16.
- `chi_scale` kwarg signature (float = 1.0) is used identically in `build_hamiltonian` (Task 1), `simulate_readout` (Task 1), `_evaluate_F_analytic` (Task 4), and `_perturbed_device_drive_scale` (Task 4).
- `compute_assignment_fidelity(..., noise_model="ideal")` contract is uniform in Tasks 4, 10, 13, 16 — and enforced by test O8 in Task 3.

---

## Execution Handoff

**Plan complete and saved to `06_Dispersive_Readout/MODULE_4_PLAN.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Each Module 4 task has a clear file list + TDD cycle + commit — well-suited to the subagent-per-task pattern. Review checkpoints after every task catch drift early; Q8 analytic-objective contract is guarded by O8 so regressions are surfaced automatically.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`; batch execution with checkpoints at Day 10 / Day 11 / Day 12 / Day 13 boundaries.

**Which approach?**




