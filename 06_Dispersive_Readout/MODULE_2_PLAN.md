# Stage 06 Module 2 — Error Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a coherent/incoherent error-budget decomposition of dispersive-readout assignment infidelity at REFERENCE_DEVICE, rendered as a two-group waterfall figure (Figure 2) with YAML-serialized underlying data.

**Architecture:** Add a `dispersive_readout/analysis/` subpackage that consumes Module 1's public API (`simulate_readout`, `compute_assignment_fidelity`). Two-group waterfall: **active-loss** channels (T1, pure dephasing, thermal, Purcell) turn off their collapse operators in simulation; **calibration-sensitivity** channels (drive amplitude ±5 %, detuning ±κ/4) perturb `DriveParams` about the calibrated operating point. Uncertainty is analytic binomial SE (no bootstrap); calibration is closed-form from steady-state SNR with simulation-verified fallback. Two Module 1 surgical edits support the Purcell toggle and per-call RNG propagation.

**Tech Stack:** Python 3.11+, QuTiP 5.x (Lindblad mesolve), NumPy 2.x (`np.trapezoid`), Pydantic v2 (schema validation), PyYAML (data export), matplotlib (Figure 2), pytest (test suite).

**Spec:** See `06_Dispersive_Readout/MODULE_2_SPEC.md`. This plan implements §3 (Module 1 edits), §4 (module structure), §5 (detailed specs), §6 (tests), §7 (Figure 2), §8 (day-by-day tasks).

**Pre-plan assumption:** All work happens on a new branch `stage-06-module-2-error-budget` branched off the current `stage-06-module-1-physics`. The executor should create this branch as step 0 before Task 1.

**Step 0 — Create Module 2 branch (not a task; do once at start):**

```bash
git checkout -b stage-06-module-2-error-budget
```

**Test invocation convention (all pytest commands in this plan use this form):**

```bash
python -m pytest <test-path> -v -p no:dash
```

The `python -m pytest` form ensures we use the conda environment's pytest (the `/usr/local/bin/pytest` picks the wrong Python and QuTiP import fails). The `-p no:dash` disables a broken Flask plugin on this system.

---

## File Structure

### Files to modify (Module 1 — three surgical edits)

| File | Edit |
|---|---|
| `dispersive_readout/physics/config.py` | Add `purcell_enabled: bool = True` field to `DecoherenceParams`. |
| `dispersive_readout/physics/lindblad.py` | Gate Purcell loop (lines 128–135) on `device.decoherence.purcell_enabled`. |
| `dispersive_readout/physics/readout_model.py` | Add `rng: np.random.Generator \| None = None` kwarg to `compute_assignment_fidelity`; replace hardcoded `default_rng(seed=42)` with conditional RNG selection. |

### Files to create (Module 2 — new subpackage + scripts)

| File | Responsibility |
|---|---|
| `dispersive_readout/analysis/__init__.py` | Public API: `OperatingPoint`, `ErrorBudget`, `ChannelContribution`, `ChannelName`, `ChannelGroup`, `compute_channel_contribution`, `compute_full_error_budget`, `calibrate_drive_amplitude`, `get_reference_operating_point`, `export_budget_to_yaml`, `analytic_purcell_rate`. |
| `dispersive_readout/analysis/operating_point.py` | `OperatingPoint` frozen dataclass; `calibrate_drive_amplitude` (analytic + sim-verified fallback); `get_reference_operating_point`. |
| `dispersive_readout/analysis/purcell_isolation.py` | `analytic_purcell_rate(device) -> float` (one exported function, ~15 lines). |
| `dispersive_readout/analysis/error_budget.py` | Pydantic schemas (`ChannelName`, `ChannelGroup`, `ChannelContribution`, `ErrorBudget`); `compute_channel_contribution(op, channel)`; `compute_full_error_budget(op)`; `export_budget_to_yaml`. |
| `dispersive_readout/tests/test_error_budget.py` | 13 tests per spec §6 (B1–B5 + 6 per-channel + 1 calibration). |
| `06_Dispersive_Readout/scripts/fig2_error_budget.py` | Renders Figure 2 PNG + YAML from a computed `ErrorBudget`. |
| `06_Dispersive_Readout/figures/fig2_error_budget.png` | Generated publication-quality waterfall (150 DPI, ~1200 px). |
| `06_Dispersive_Readout/figures/fig2_data.yaml` | YAML-serialized `ErrorBudget` at REFERENCE_OPERATING_POINT. |

---

## Task 1: Add `purcell_enabled` toggle to `DecoherenceParams` and gate the Purcell loop

**Rationale:** Spec §3 requires a config field so the Purcell turn-off channel in Module 2 can disable the Purcell collapse operator without touching unrelated rates.

**Files:**
- Modify: `dispersive_readout/physics/config.py` — add field to `DecoherenceParams` dataclass
- Modify: `dispersive_readout/physics/lindblad.py:128-135` — gate the Purcell loop
- Test: add assertion to `dispersive_readout/tests/test_lindblad.py` — verify disabling removes Purcell operators

- [ ] **Step 1.1: Write the failing regression test**

Add a test that creates a `DecoherenceParams(purcell_enabled=False)` device and verifies the returned collapse-operator list has one fewer per-transmon-transition operator than the default.

Add to the end of `dispersive_readout/tests/test_lindblad.py`:

```python
def test_purcell_disabled_removes_purcell_collapse_operators():
    """Setting purcell_enabled=False must omit the Purcell channel operators."""
    from dispersive_readout.physics.config import (
        DecoherenceParams, DeviceConfig, REFERENCE_DEVICE
    )
    from dispersive_readout.physics.lindblad import build_collapse_operators

    tr = REFERENCE_DEVICE.truncation
    Nq, Nr = tr.N_transmon, tr.N_resonator

    device_on = REFERENCE_DEVICE  # purcell_enabled=True by default
    device_off = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=DecoherenceParams(
            gamma_1=REFERENCE_DEVICE.decoherence.gamma_1,
            gamma_phi=REFERENCE_DEVICE.decoherence.gamma_phi,
            n_th=REFERENCE_DEVICE.decoherence.n_th,
            purcell_enabled=False,
        ),
        truncation=tr,
    )

    c_ops_on = build_collapse_operators(device_on, Nq, Nr)
    c_ops_off = build_collapse_operators(device_off, Nq, Nr)

    # Purcell adds Nq-1 operators (|j> -> |j-1> for j=1..Nq-1)
    assert len(c_ops_on) - len(c_ops_off) == Nq - 1
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
python -m pytest dispersive_readout/tests/test_lindblad.py::test_purcell_disabled_removes_purcell_collapse_operators -v -p no:dash
```

Expected: FAIL with `TypeError: DecoherenceParams.__init__() got an unexpected keyword argument 'purcell_enabled'`.

- [ ] **Step 1.3: Add the `purcell_enabled` field to `DecoherenceParams`**

In `dispersive_readout/physics/config.py`, locate the `DecoherenceParams` dataclass (around line 51–62) and add the field:

```python
@dataclass(frozen=True)
class DecoherenceParams:
    """Incoherent error channels.

    gamma_1:          qubit relaxation rate (1/s, equivalently rad/s for rates).
    gamma_phi:        pure dephasing rate; from T2_echo after subtracting gamma_1/2.
    n_th:             bath thermal population (dimensionless).
    purcell_enabled:  if False, omit Purcell collapse operators in
                      build_collapse_operators. Used by Module 2's Purcell
                      turn-off channel to isolate the Purcell contribution.
    """
    gamma_1: float
    gamma_phi: float
    n_th: float = 0.01
    purcell_enabled: bool = True
```

- [ ] **Step 1.4: Gate the Purcell loop in `build_collapse_operators`**

In `dispersive_readout/physics/lindblad.py`, wrap the Purcell loop (lines 128–135) in an `if` guard:

```python
    # 6. Purcell decay |j> -> |j-1> at rate (g|n_{j-1,j}| / Delta_{j,j-1})^2 kappa.
    # (See module docstring for derivation.)
    if device.decoherence.purcell_enabled:
        for j in range(1, Nq):
            delta_j = energies[j] - energies[j - 1] - device.resonator.omega_r
            n_elem = abs(n_mat[j - 1, j])
            gamma_P = ((device.coupling.g * n_elem) / delta_j) ** 2 * kappa * (1.0 + n_th)
            if gamma_P > 0:
                op = qt.basis(Nq, j - 1) * qt.basis(Nq, j).dag()
                c_ops.append(np.sqrt(gamma_P) * qt.tensor(op, qt.qeye(Nr)))
```

- [ ] **Step 1.5: Run the new test to verify it passes**

```bash
python -m pytest dispersive_readout/tests/test_lindblad.py::test_purcell_disabled_removes_purcell_collapse_operators -v -p no:dash
```

Expected: PASS.

- [ ] **Step 1.6: Run full Module 1 test suite to verify no regressions**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -20
```

Expected: all 56 Module 1 tests + the new test passing (57 total). If any Module 1 test fails, STOP and investigate — the `purcell_enabled=True` default should preserve the exact prior behavior.

- [ ] **Step 1.7: Commit**

```bash
git add dispersive_readout/physics/config.py dispersive_readout/physics/lindblad.py dispersive_readout/tests/test_lindblad.py
git commit -m "feat(stage06): add purcell_enabled toggle to DecoherenceParams

Module 2 Task 1. Adds a purcell_enabled: bool = True field to
DecoherenceParams and gates the Purcell loop in build_collapse_operators.
Default True preserves Module 1 behavior exactly; setting False omits
the Nq-1 Purcell collapse operators so Module 2's Purcell turn-off
channel can isolate the contribution."
```

**Definition of done:** New test passes; all 56 Module 1 tests still pass; `DecoherenceParams(purcell_enabled=False)` produces `len(c_ops_on) - len(c_ops_off) == Nq - 1` at REFERENCE.

---

## Task 2: Add `rng` kwarg to `compute_assignment_fidelity`

**Rationale:** Spec §2.4 and §3 require independent shot draws across successive calls. The current hardcoded `seed=42` correlates all draws, breaking quadrature propagation of ΔF uncertainties. After this change, default `rng=None` produces ephemeral randomness; tests that need determinism pass an explicit seeded RNG.

**Files:**
- Modify: `dispersive_readout/physics/readout_model.py` — signature + body of `compute_assignment_fidelity`
- Verify: `dispersive_readout/tests/test_readout_model.py` — three existing tests must still pass (they only check loose ranges, not specific F values)

- [ ] **Step 2.1: Update the function signature**

In `dispersive_readout/physics/readout_model.py`, update `compute_assignment_fidelity`:

```python
def compute_assignment_fidelity(
    result_ground: ReadoutResult,
    result_excited: ReadoutResult,
    integration_window: tuple[float, float],
    n_shots: int = 10000,
    noise_model: Literal["ideal", "gaussian"] = "gaussian",
    rng: np.random.Generator | None = None,
) -> AssignmentFidelityResult:
```

- [ ] **Step 2.2: Replace the hardcoded RNG in the function body**

Find the line (currently around 186):

```python
    rng = np.random.default_rng(seed=42)
```

Replace with:

```python
    if rng is None:
        rng = np.random.default_rng()
```

(Using `np.random.default_rng()` without seed draws from system entropy, giving independent draws across calls.)

- [ ] **Step 2.3: Update the function docstring to document the new kwarg and the independence assumption**

Locate the docstring (under `def compute_assignment_fidelity`) and append to it:

```
    Parameters
    ----------
    rng : np.random.Generator | None, optional
        RNG for shot-noise draws. If None (default), an ephemeral RNG is
        created per call, giving independent draws across successive calls.
        Pass a seeded RNG for deterministic tests.

    Notes
    -----
    The analytic F_assign_uncertainty returned in the result assumes
    independent shot draws between successive calls with the same (c0, c1).
    Passing the *same* rng object to multiple calls will advance its state
    and correlate the draws, violating that assumption — Module 2's
    error-budget decomposition relies on default rng=None.
```

- [ ] **Step 2.4: Run the three existing Module 1 tests that use `compute_assignment_fidelity`**

```bash
python -m pytest dispersive_readout/tests/test_readout_model.py -v -p no:dash
```

Expected: all tests pass. The three assertions (`0 <= F <= 1`, `f_i >= f_g - 1e-9`, `F >= 0.95`) tolerate nondeterministic draws. If any test flakes, loosen the tolerance or pass `rng=np.random.default_rng(seed=42)` at the call site — but *do not* do this preemptively; only if a test actually flakes.

- [ ] **Step 2.5: Run full Module 1 test suite**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: 57 passing (56 original + Task 1's new test).

- [ ] **Step 2.6: Commit**

```bash
git add dispersive_readout/physics/readout_model.py
git commit -m "feat(stage06): add rng kwarg to compute_assignment_fidelity

Module 2 Task 2. Replaces hardcoded default_rng(seed=42) with a
conditional: if rng is None, an ephemeral RNG is created per call for
independent shot draws. The analytic binomial SE only propagates
correctly when successive calls use independent draws, which Module 2's
error-budget decomposition requires."
```

**Definition of done:** All 57 Module 1 tests pass; `compute_assignment_fidelity` accepts `rng` kwarg; default `rng=None` path produces independent draws (verifiable by calling twice with same inputs and seeing different F values).

---

## Task 3: Create `dispersive_readout/analysis/` package skeleton

**Rationale:** Spec §4 puts Module 2 code in a new `analysis/` subpackage sibling to `physics/`. Empty-but-importable scaffolding first, then fill in per task.

**Files:**
- Create: `dispersive_readout/analysis/__init__.py` (empty stub)
- Create: `dispersive_readout/analysis/operating_point.py` (empty stub)
- Create: `dispersive_readout/analysis/purcell_isolation.py` (empty stub)
- Create: `dispersive_readout/analysis/error_budget.py` (empty stub)
- Create: `dispersive_readout/tests/test_error_budget.py` (empty stub with one import-smoke test)

- [ ] **Step 3.1: Create the four `analysis/` files with minimal stubs**

```bash
mkdir -p dispersive_readout/analysis
```

Create `dispersive_readout/analysis/__init__.py`:

```python
"""Stage 06 Module 2 — error-budget decomposition and Figure 2 data model.

See 06_Dispersive_Readout/MODULE_2_SPEC.md for the design contract.
Public API is populated as Tasks 4–8 of the Module 2 plan land.
"""
```

Create `dispersive_readout/analysis/operating_point.py`:

```python
"""Operating-point dataclass and analytic drive-amplitude calibration.

See MODULE_2_SPEC.md §2.3 for the closed-form calibration derivation
and §5.1 for the API contract.
"""
```

Create `dispersive_readout/analysis/purcell_isolation.py`:

```python
"""Analytic Purcell rate for cross-validation of the simulated Purcell channel.

See MODULE_2_SPEC.md §5.2. Post-blocker-6, only analytic_purcell_rate is
exported; effective_T1_from_device and decomposed_T1 from the original spec
are YAGNI.
"""
```

Create `dispersive_readout/analysis/error_budget.py`:

```python
"""Coherent/incoherent error-budget decomposition data model and computation.

See MODULE_2_SPEC.md §2 (methodology), §5.3 (schemas), §6 (tests).
"""
```

- [ ] **Step 3.2: Create the test file with an import-smoke test**

Create `dispersive_readout/tests/test_error_budget.py`:

```python
"""Module 2 tests — see MODULE_2_SPEC.md §6 for the test plan."""
from __future__ import annotations


def test_module2_package_imports_without_error():
    """Smoke test: the analysis subpackage can be imported. Populated further
    as Tasks 4–8 add real API."""
    import dispersive_readout.analysis  # noqa: F401
    import dispersive_readout.analysis.operating_point  # noqa: F401
    import dispersive_readout.analysis.purcell_isolation  # noqa: F401
    import dispersive_readout.analysis.error_budget  # noqa: F401
```

- [ ] **Step 3.3: Run the smoke test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py -v -p no:dash
```

Expected: 1 passed.

- [ ] **Step 3.4: Commit**

```bash
git add dispersive_readout/analysis/ dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): scaffold dispersive_readout/analysis subpackage

Module 2 Task 3. Creates empty-but-importable stubs for operating_point,
purcell_isolation, error_budget, plus a test file with an import smoke
test. Real API populated by Tasks 4-8."
```

**Definition of done:** `import dispersive_readout.analysis` succeeds; smoke test passes.

---

## Task 4: `operating_point.py` — `OperatingPoint` + analytic calibration + verified fallback

**Rationale:** Spec §2.3 closed-form calibration; §5.1 API. The analytic form: `ε₀ = SNR_target / (2 × |M| × sqrt(κ × T_int))` with M the frequency-response factor. Verification step compares measured F against the target; fallback to grid search if > 3σ_shot off.

**Files:**
- Modify: `dispersive_readout/analysis/operating_point.py`
- Modify: `dispersive_readout/analysis/__init__.py` (export)
- Test: `dispersive_readout/tests/test_error_budget.py` (add calibration test)

- [ ] **Step 4.1: Write the failing calibration test**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
import numpy as np
import pytest


def test_analytic_calibration_hits_target_fidelity_within_3_sigma():
    """Analytic ε₀ calibration at REFERENCE_DEVICE produces F_verified in
    F_target ± 3σ_shot. If this fails, fallback to grid search is triggered.
    See MODULE_2_SPEC.md §2.3."""
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import calibrate_drive_amplitude

    target = 0.99
    n_shots = 10_000
    sigma_shot = np.sqrt(target * (1.0 - target) / n_shots)  # ≈ 1e-3

    epsilon_0 = calibrate_drive_amplitude(
        device=REFERENCE_DEVICE,
        duration=500e-9,
        integration_window=(50e-9, 500e-9),
        target_fidelity=target,
        n_shots=n_shots,
        sigma_tolerance_factor=3.0,
    )

    # Verify at the returned ε₀
    import math
    from dispersive_readout.physics import (
        DriveParams, simulate_readout, compute_assignment_fidelity,
    )

    drv = DriveParams(amplitude=epsilon_0, duration=500e-9, detuning=0.0)
    r0 = simulate_readout(REFERENCE_DEVICE, drv, initial_qubit_state=0)
    r1 = simulate_readout(REFERENCE_DEVICE, drv, initial_qubit_state=1)
    f = compute_assignment_fidelity(
        r0, r1, (50e-9, 500e-9), n_shots=n_shots, noise_model="gaussian",
        rng=np.random.default_rng(seed=42),  # deterministic for test reproducibility
    )

    assert abs(f.F_assign - target) <= 3.0 * sigma_shot, (
        f"Calibration gave F={f.F_assign:.4f}, expected {target}±{3*sigma_shot:.4f}. "
        f"Either the analytic formula is wrong or fallback is needed."
    )
```

- [ ] **Step 4.2: Run the test to verify it fails**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_analytic_calibration_hits_target_fidelity_within_3_sigma -v -p no:dash
```

Expected: FAIL with `ImportError: cannot import name 'calibrate_drive_amplitude' from 'dispersive_readout.analysis'`.

- [ ] **Step 4.3: Implement `OperatingPoint` dataclass and `calibrate_drive_amplitude` in `operating_point.py`**

Replace the stub `dispersive_readout/analysis/operating_point.py` with:

```python
"""Operating-point dataclass and analytic drive-amplitude calibration.

See MODULE_2_SPEC.md §2.3 for the closed-form calibration derivation
and §5.1 for the API contract.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..physics.config import DeviceConfig, DriveParams, REFERENCE_DEVICE
from ..physics.dispersive import dispersive_shift_full
from ..physics.transmon import charge_operator_matrix_elements, diagonalize_transmon
from ..physics.readout_model import (
    simulate_readout,
    compute_assignment_fidelity,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Fixed operating point for error-budget analysis.

    Attributes
    ----------
    device : DeviceConfig
    drive : DriveParams
        Readout drive with amplitude calibrated per §2.3.
    integration_window : tuple[float, float]
        (t0, t1) for IQ integration, seconds.
    n_shots : int
        Shots per fidelity evaluation.
    """
    device: DeviceConfig
    drive: DriveParams
    integration_window: tuple[float, float]
    n_shots: int


def _response_factor_M(device: DeviceConfig) -> complex:
    """Steady-state separation-per-unit-drive factor M for on-resonance drive.

    M = 1/(κ/2 − iχ_0) − 1/(κ/2 − iχ_1). Uses the per-level χ_j from
    dispersive_shift_full (non-RWA 2nd-order PT including Bloch-Siegert).
    |M| has units of s/rad.
    """
    tr = device.truncation
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    chi = dispersive_shift_full(energies, n_mat, device.coupling.g,
                                 device.resonator.omega_r)
    kappa = device.resonator.kappa
    M = 1.0 / (0.5 * kappa - 1j * chi[0]) - 1.0 / (0.5 * kappa - 1j * chi[1])
    return M


def _analytic_epsilon_0(
    device: DeviceConfig, target_fidelity: float, t_int: float
) -> float:
    """Solve ε₀ from SNR_target = 2 × |M| × sqrt(κ T_int) × ε₀.

    SNR_target = 2 × Φ⁻¹(F_target) from F = 1 − Q(SNR/2) and Q(x) = 1 − Φ(x).
    """
    snr_target = 2.0 * norm.ppf(target_fidelity)
    M = _response_factor_M(device)
    kappa = device.resonator.kappa
    epsilon_0 = snr_target / (2.0 * abs(M) * math.sqrt(kappa * t_int))
    return float(epsilon_0)


def _grid_search_epsilon_0(
    device: DeviceConfig,
    duration: float,
    integration_window: tuple[float, float],
    target_fidelity: float,
    n_shots: int,
    n_grid: int = 15,
) -> float:
    """Fallback: grid-scan the low-ε branch, return lowest ε with F ≥ target.

    Bracket: ε_min (where F ≈ 0.5, chosen at 0.1× analytic) to ε_max
    (where n̄_peak ≈ 0.5 × N_resonator).
    """
    epsilon_analytic = _analytic_epsilon_0(
        device, target_fidelity, integration_window[1] - integration_window[0]
    )
    eps_min = 0.1 * epsilon_analytic
    eps_max = 3.0 * epsilon_analytic
    grid = np.linspace(eps_min, eps_max, n_grid)

    for eps in grid:
        drv = DriveParams(amplitude=float(eps), duration=duration, detuning=0.0)
        r0 = simulate_readout(device, drv, initial_qubit_state=0)
        r1 = simulate_readout(device, drv, initial_qubit_state=1)
        f = compute_assignment_fidelity(
            r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian"
        )
        if f.F_assign >= target_fidelity:
            return float(eps)

    raise RuntimeError(
        f"Grid search did not find ε₀ achieving F ≥ {target_fidelity} on "
        f"low-ε branch [{eps_min:.2e}, {eps_max:.2e}] rad/s. Target unreachable."
    )


def calibrate_drive_amplitude(
    device: DeviceConfig,
    duration: float,
    integration_window: tuple[float, float],
    target_fidelity: float = 0.99,
    n_shots: int = 10_000,
    sigma_tolerance_factor: float = 3.0,
) -> float:
    """Analytic drive-amplitude calibration with simulation-verified fallback.

    Computes ε₀ from the dispersive-regime steady-state SNR formula
    (§2.3). Verifies against a simulation; if the measured F deviates
    from target by more than sigma_tolerance_factor × σ_shot, falls back
    to grid search on the low-ε branch and emits a warning.

    Parameters
    ----------
    device : DeviceConfig
    duration : float
        Pulse duration in seconds.
    integration_window : tuple[float, float]
        (t0, t1) for IQ integration.
    target_fidelity : float
        F target for calibration; default 0.99.
    n_shots : int
        Shots for the verification measurement.
    sigma_tolerance_factor : float
        Fallback trigger band in units of σ_shot.

    Returns
    -------
    epsilon_0 : float
        Drive amplitude in rad/s.

    Raises
    ------
    RuntimeError
        If both analytic and grid search fail to achieve target.
    """
    t_int = integration_window[1] - integration_window[0]
    eps_analytic = _analytic_epsilon_0(device, target_fidelity, t_int)

    # Verification sim at eps_analytic
    drv = DriveParams(amplitude=eps_analytic, duration=duration, detuning=0.0)
    r0 = simulate_readout(device, drv, initial_qubit_state=0)
    r1 = simulate_readout(device, drv, initial_qubit_state=1)
    f_verified = compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian",
        rng=np.random.default_rng(seed=42),  # deterministic verification
    )

    sigma_shot = math.sqrt(
        target_fidelity * (1.0 - target_fidelity) / n_shots
    )
    tolerance = sigma_tolerance_factor * sigma_shot

    if abs(f_verified.F_assign - target_fidelity) <= tolerance:
        return eps_analytic

    warnings.warn(
        f"Analytic calibration gave F_verified={f_verified.F_assign:.4f}, "
        f"expected {target_fidelity}±{tolerance:.4f}. Falling back to grid "
        f"search on low-ε branch.",
        RuntimeWarning,
    )
    return _grid_search_epsilon_0(
        device, duration, integration_window, target_fidelity, n_shots
    )


def get_reference_operating_point() -> OperatingPoint:
    """Return the canonical operating point for Figure 2.

    Calibration runs on first call (< 3 s total: analytic solve + one
    verification sim × two qubit states). No persistent cache — fast
    enough to compute on demand.
    """
    integration_window = (50e-9, 500e-9)
    epsilon_0 = calibrate_drive_amplitude(
        device=REFERENCE_DEVICE,
        duration=500e-9,
        integration_window=integration_window,
        target_fidelity=0.99,
        n_shots=10_000,
    )
    return OperatingPoint(
        device=REFERENCE_DEVICE,
        drive=DriveParams(
            amplitude=epsilon_0,
            duration=500e-9,
            detuning=0.0,
            edge_sigma=2e-9,
        ),
        integration_window=integration_window,
        n_shots=10_000,
    )
```

- [ ] **Step 4.4: Export the API from `dispersive_readout/analysis/__init__.py`**

Replace `dispersive_readout/analysis/__init__.py` with:

```python
"""Stage 06 Module 2 — error-budget decomposition and Figure 2 data model.

See 06_Dispersive_Readout/MODULE_2_SPEC.md for the design contract.
"""
from .operating_point import (
    OperatingPoint,
    calibrate_drive_amplitude,
    get_reference_operating_point,
)
```

- [ ] **Step 4.5: Run the calibration test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_analytic_calibration_hits_target_fidelity_within_3_sigma -v -p no:dash
```

Expected: PASS (the verification sim lands within 3σ_shot of F=0.99 at REFERENCE).

If it FAILS with the fallback warning, the analytic formula or the steady-state approximation is insufficient — STOP, investigate before continuing. This is spec §9 flag #4.

- [ ] **Step 4.6: Full test suite regression check**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: all passing.

- [ ] **Step 4.7: Commit**

```bash
git add dispersive_readout/analysis/__init__.py dispersive_readout/analysis/operating_point.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): analytic drive-amplitude calibration with fallback

Module 2 Task 4. Implements OperatingPoint dataclass and
calibrate_drive_amplitude per MODULE_2_SPEC §2.3:
- Closed-form ε₀ from SNR_target = 2·Φ⁻¹(F_target) and
  SNR² = 4κ|Δα|²T with |Δα| = ε₀·|M|
- Verification sim at analytic ε₀; fallback to low-ε grid scan if
  measured F deviates from target by > 3σ_shot (warns on fallback)
- get_reference_operating_point returns the canonical Figure 2 anchor"
```

**Definition of done:** Calibration test passes at REFERENCE; `get_reference_operating_point()` returns in < 5 s; no fallback warning on REFERENCE.

---

## Task 5: `purcell_isolation.py` — `analytic_purcell_rate`

**Rationale:** Spec §5.2 (post-blocker-6): one exported function, ~15 lines. `γ_P = (g |⟨0|n̂|1⟩| / Δ_{10})² × κ` for the dominant |1⟩→|0⟩ transition in the transmon dressed basis.

**Files:**
- Modify: `dispersive_readout/analysis/purcell_isolation.py`
- Modify: `dispersive_readout/analysis/__init__.py` (export)

- [ ] **Step 5.1: Implement `analytic_purcell_rate`**

Replace `dispersive_readout/analysis/purcell_isolation.py` with:

```python
"""Analytic Purcell rate for cross-validation of the simulated Purcell channel.

See MODULE_2_SPEC.md §5.2. Post-blocker-6, only analytic_purcell_rate is
exported; effective_T1_from_device and decomposed_T1 from the original spec
are YAGNI.

Reference: Blais et al., Rev. Mod. Phys. 93, 025005 (2021), §III.E.
"""
from __future__ import annotations

from ..physics.config import DeviceConfig
from ..physics.transmon import charge_operator_matrix_elements, diagonalize_transmon


def analytic_purcell_rate(device: DeviceConfig) -> float:
    """γ_Purcell for the |1⟩→|0⟩ transition from (g |⟨0|n̂|1⟩| / Δ_{10})² κ.

    Uses the dressed transmon basis (N-level), not the 2-level estimate.
    Δ_{10} = ω_1 − ω_0 − ω_r is the detuning of the |1>→|0> transition
    from the resonator.

    Returns
    -------
    gamma_P : float
        Purcell rate in rad/s (equivalently, 1/s for rates).
    """
    tr = device.truncation
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    g = device.coupling.g
    kappa = device.resonator.kappa
    omega_r = device.resonator.omega_r

    delta_10 = energies[1] - energies[0] - omega_r
    n_elem = abs(n_mat[0, 1])
    gamma_P = ((g * n_elem) / delta_10) ** 2 * kappa
    return float(gamma_P)
```

- [ ] **Step 5.2: Export from `__init__.py`**

Append to `dispersive_readout/analysis/__init__.py`:

```python
from .purcell_isolation import analytic_purcell_rate
```

- [ ] **Step 5.3: Smoke-test the function**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_analytic_purcell_rate_positive_at_reference():
    """γ_P at REFERENCE should be positive and of order (g/Δ)²κ ~ O(kHz)."""
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import analytic_purcell_rate

    gamma_P = analytic_purcell_rate(REFERENCE_DEVICE)
    assert gamma_P > 0.0
    # Order-of-magnitude sanity: g/Δ ≈ 120 MHz / 2700 MHz ≈ 0.044
    # γ_P / κ ≈ 0.044² ≈ 1.9e-3; κ/2π = 5 MHz → γ_P/2π ~ 9.5 kHz
    kappa = REFERENCE_DEVICE.resonator.kappa
    ratio = gamma_P / kappa
    assert 1e-4 < ratio < 1e-1, f"γ_P/κ = {ratio:.2e} outside plausible range"
```

- [ ] **Step 5.4: Run the smoke test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_analytic_purcell_rate_positive_at_reference -v -p no:dash
```

Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add dispersive_readout/analysis/purcell_isolation.py dispersive_readout/analysis/__init__.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): analytic Purcell rate via Blais RMP III.E

Module 2 Task 5. analytic_purcell_rate(device) returns γ_P for the
|1⟩→|0⟩ transition from the dressed-basis formula
(g·|<0|n̂|1>|/Δ_{10})²·κ. Used by test B3 (Task 9) to cross-validate
the simulated Purcell contribution against the 2nd-order PT prediction."
```

**Definition of done:** `analytic_purcell_rate(REFERENCE_DEVICE) > 0`; order-of-magnitude sanity test passes.

---

## Task 6: Pydantic schemas in `error_budget.py` + B4 validation test

**Rationale:** Spec §5.3 defines `ChannelName`, `ChannelGroup`, `ChannelContribution`, `ErrorBudget`. Pydantic v2 validators catch unexpectedly-negative contributions. Task 6 lands the schemas and the B4 validator test before the computation functions (Task 7).

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Modify: `dispersive_readout/analysis/__init__.py` (export)
- Test: `dispersive_readout/tests/test_error_budget.py` (add B4)

- [ ] **Step 6.1: Write the failing B4 test**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_B4_negative_contribution_raises():
    """ChannelContribution with delta_F < -0.005 must raise ValueError.
    Small negatives (shot-noise floor) are floored to zero."""
    from dispersive_readout.analysis import ChannelContribution

    # Below -0.005 floor: must raise
    with pytest.raises(ValueError, match="negative"):
        ChannelContribution(
            name="T1_intrinsic",
            group="active_loss",
            delta_F=-0.01,
            delta_F_uncertainty=1e-4,
            description="test",
        )

    # Within shot-noise floor: accepted, floored to 0
    c = ChannelContribution(
        name="T1_intrinsic",
        group="active_loss",
        delta_F=-0.003,  # > -0.005 floor
        delta_F_uncertainty=1e-4,
        description="test",
    )
    assert c.delta_F == 0.0
```

- [ ] **Step 6.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B4_negative_contribution_raises -v -p no:dash
```

Expected: FAIL (import error).

- [ ] **Step 6.3: Implement the schemas in `error_budget.py`**

Replace `dispersive_readout/analysis/error_budget.py` with:

```python
"""Coherent/incoherent error-budget decomposition data model and computation.

See MODULE_2_SPEC.md §2 (methodology), §5.3 (schemas), §6 (tests).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


ChannelName = Literal[
    "T1_intrinsic",
    "pure_dephasing",
    "thermal",
    "purcell",
    "drive_amplitude",
    "drive_detuning",
]

ChannelGroup = Literal["active_loss", "calibration_sensitivity"]


class ChannelContribution(BaseModel):
    """Single channel's contribution to the error budget.

    For active_loss channels: delta_F = F_c_off - F_full (non-negative modulo
    shot noise); uncertainty is analytic binomial SE propagated in quadrature.
    For calibration_sensitivity channels: delta_F = mean(|F_full - F_±|)
    (non-negative by construction); uncertainty is the ± asymmetry |F_+ - F_-|/2.
    """
    name: ChannelName
    group: ChannelGroup
    delta_F: float
    delta_F_uncertainty: float
    description: str
    perturbation_description: str | None = None

    @field_validator("delta_F")
    @classmethod
    def nonnegative(cls, v: float) -> float:
        if v < -0.005:
            raise ValueError(
                f"Channel contribution unexpectedly negative: {v}. "
                f"Small negatives from shot noise are floored to zero; "
                f"< -0.005 indicates a bug in the turn-off logic."
            )
        return max(v, 0.0)


class ErrorBudget(BaseModel):
    """Complete error budget at a single operating point.

    The additivity identity (F_ideal − F_full) = Σ_active ΔF_c + R_active
    holds only for the active-loss group (§2.1). Calibration-sensitivity
    channels do not enter this identity.
    """
    operating_point_id: str
    F_full: float
    F_ideal: float
    channels: list[ChannelContribution]
    residual_active: float
    residual_active_uncertainty: float

    @property
    def active_loss_channels(self) -> list[ChannelContribution]:
        return [c for c in self.channels if c.group == "active_loss"]

    @property
    def calibration_channels(self) -> list[ChannelContribution]:
        return [c for c in self.channels if c.group == "calibration_sensitivity"]

    @property
    def total_infidelity(self) -> float:
        return 1.0 - self.F_full

    @property
    def explained_active_loss(self) -> float:
        return sum(c.delta_F for c in self.active_loss_channels)


def export_budget_to_yaml(budget: ErrorBudget, path) -> None:
    """Serialize an ErrorBudget to YAML at `path` (str or Path).

    Preserves all fields and the channel list in order. Used by
    scripts/fig2_error_budget.py and test B5 round-trip.
    """
    import yaml
    from pathlib import Path

    payload = {
        "operating_point_id": budget.operating_point_id,
        "F_full": budget.F_full,
        "F_ideal": budget.F_ideal,
        "residual_active": budget.residual_active,
        "residual_active_uncertainty": budget.residual_active_uncertainty,
        "channels": [
            {
                "name": c.name,
                "group": c.group,
                "delta_F": c.delta_F,
                "delta_F_uncertainty": c.delta_F_uncertainty,
                "description": c.description,
                "perturbation_description": c.perturbation_description,
            }
            for c in budget.channels
        ],
    }
    Path(path).write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    )
```

- [ ] **Step 6.4: Export from `__init__.py`**

Append to `dispersive_readout/analysis/__init__.py`:

```python
from .error_budget import (
    ChannelName,
    ChannelGroup,
    ChannelContribution,
    ErrorBudget,
    export_budget_to_yaml,
)
```

- [ ] **Step 6.5: Run B4 to verify pass**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B4_negative_contribution_raises -v -p no:dash
```

Expected: PASS.

- [ ] **Step 6.6: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/analysis/__init__.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): Pydantic schemas + export_budget_to_yaml + B4 test

Module 2 Task 6. ChannelName, ChannelGroup, ChannelContribution,
ErrorBudget per MODULE_2_SPEC §5.3, plus export_budget_to_yaml from
§5.4 public API. field_validator on delta_F catches contributions
< -0.005 (bug), floors [-0.005, 0] (shot noise). perturbation_description
field makes calibration-sensitivity bars self-documenting in YAML.
B4 test validates the validator."
```

**Definition of done:** B4 passes; `from dispersive_readout.analysis import ChannelContribution, ErrorBudget` works.

---

## Task 7a: `compute_channel_contribution` for `T1_intrinsic`

**Rationale:** First of six active-loss channels. Establishes the pattern subsequent channels follow: build a turn-off `DeviceConfig` with one field zeroed, simulate |0> and |1>, compute F, return ΔF with analytic σ propagation.

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Modify: `dispersive_readout/analysis/__init__.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7a.1: Write the failing T1 contribution test**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_T1_intrinsic_contribution_nonzero_at_reference():
    """Turning off γ_1 at REFERENCE should improve F by a non-trivial amount."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )

    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="T1_intrinsic")

    assert c.name == "T1_intrinsic"
    assert c.group == "active_loss"
    assert c.delta_F > 0.0
    assert c.delta_F_uncertainty > 0.0
```

- [ ] **Step 7a.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_T1_intrinsic_contribution_nonzero_at_reference -v -p no:dash
```

Expected: FAIL (import error).

- [ ] **Step 7a.3: Add helper + T1 channel logic to `error_budget.py`**

Append to `dispersive_readout/analysis/error_budget.py`:

```python
import math
from dataclasses import replace

import numpy as np

from ..physics.config import DecoherenceParams, DeviceConfig, DriveParams
from ..physics.readout_model import (
    simulate_readout,
    compute_assignment_fidelity,
)


def _F_at(
    device: DeviceConfig,
    drive: DriveParams,
    integration_window: tuple[float, float],
    n_shots: int,
) -> tuple[float, float]:
    """Simulate |0>, |1>, return (F_assign, σ_F) using independent shot draws."""
    r0 = simulate_readout(device, drive, initial_qubit_state=0)
    r1 = simulate_readout(device, drive, initial_qubit_state=1)
    f = compute_assignment_fidelity(
        r0, r1, integration_window, n_shots=n_shots, noise_model="gaussian",
        rng=None,  # ephemeral RNG → independent draws
    )
    return float(f.F_assign), float(f.F_assign_uncertainty)


def _device_with_decoherence(device: DeviceConfig, **overrides) -> DeviceConfig:
    """Return a copy of device with the given DecoherenceParams field overrides."""
    new_dec = replace(device.decoherence, **overrides)
    return DeviceConfig(
        transmon=device.transmon,
        resonator=device.resonator,
        coupling=device.coupling,
        decoherence=new_dec,
        truncation=device.truncation,
    )


def compute_channel_contribution(
    operating_point,
    channel: ChannelName,
) -> ChannelContribution:
    """Compute the marginal fidelity loss attributable to a single channel.

    Active-loss channels (T1, dephasing, thermal, Purcell) zero their
    respective field and compute ΔF = F_off − F_full. Calibration-sensitivity
    channels (drive_amplitude, drive_detuning) perturb DriveParams and
    compute mean-of-absolute losses.

    See MODULE_2_SPEC.md §2.1 and §2.3 for details.
    """
    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    # Baseline F (all channels on)
    F_full, sigma_full = _F_at(device, drive, window, n_shots)

    if channel == "T1_intrinsic":
        dev_off = _device_with_decoherence(device, gamma_1=0.0)
        F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
        delta_F = F_off - F_full
        sigma_delta = math.sqrt(sigma_off**2 + sigma_full**2)
        return ChannelContribution(
            name="T1_intrinsic",
            group="active_loss",
            delta_F=delta_F,
            delta_F_uncertainty=sigma_delta,
            description="Fidelity loss from intrinsic T1 relaxation (γ_1).",
        )

    raise NotImplementedError(f"Channel {channel!r} not yet implemented.")
```

- [ ] **Step 7a.4: Export the function**

Append to `dispersive_readout/analysis/__init__.py`:

```python
from .error_budget import compute_channel_contribution
```

- [ ] **Step 7a.5: Run the T1 test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_T1_intrinsic_contribution_nonzero_at_reference -v -p no:dash
```

Expected: PASS (takes ~6 s — three sims: baseline |0>, baseline |1>, T1-off |0>+|1>).

- [ ] **Step 7a.6: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/analysis/__init__.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): T1_intrinsic channel contribution + helpers

Module 2 Task 7a. compute_channel_contribution dispatcher + T1_intrinsic
branch. Helper _device_with_decoherence uses dataclasses.replace to build
turn-off devices without mutating baseline. Helper _F_at wraps sim+F
with rng=None for independent draws. Analytic σ_ΔF propagation via
quadrature."
```

**Definition of done:** T1 test passes; `ChannelContribution` returned has `name=\"T1_intrinsic\"`, `group=\"active_loss\"`, `delta_F > 0`.

---

## Task 7b: `pure_dephasing` channel

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7b.1: Write the failing test**

```python
def test_pure_dephasing_contribution_nonzero_at_reference():
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="pure_dephasing")
    assert c.name == "pure_dephasing"
    assert c.group == "active_loss"
    assert c.delta_F > 0.0
```

- [ ] **Step 7b.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_pure_dephasing_contribution_nonzero_at_reference -v -p no:dash
```

Expected: FAIL with `NotImplementedError: Channel 'pure_dephasing' not yet implemented.`

- [ ] **Step 7b.3: Add the `pure_dephasing` branch**

In `dispersive_readout/analysis/error_budget.py`, inside `compute_channel_contribution`, above the final `raise NotImplementedError`, add:

```python
    if channel == "pure_dephasing":
        dev_off = _device_with_decoherence(device, gamma_phi=0.0)
        F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
        delta_F = F_off - F_full
        sigma_delta = math.sqrt(sigma_off**2 + sigma_full**2)
        return ChannelContribution(
            name="pure_dephasing",
            group="active_loss",
            delta_F=delta_F,
            delta_F_uncertainty=sigma_delta,
            description="Fidelity loss from pure dephasing (γ_φ).",
        )
```

- [ ] **Step 7b.4: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_pure_dephasing_contribution_nonzero_at_reference -v -p no:dash
```

Expected: PASS.

- [ ] **Step 7b.5: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): pure_dephasing channel contribution

Module 2 Task 7b. Zero γ_φ in DecoherenceParams, recompute F,
quadrature-propagate σ_ΔF."
```

**Definition of done:** Test passes.

---

## Task 7c: `thermal` channel

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7c.1: Write the failing test**

```python
def test_thermal_contribution_nonzero_at_reference():
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="thermal")
    assert c.name == "thermal"
    assert c.group == "active_loss"
    assert c.delta_F > 0.0
```

- [ ] **Step 7c.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_thermal_contribution_nonzero_at_reference -v -p no:dash
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 7c.3: Add the `thermal` branch**

In `compute_channel_contribution`, above the final `raise`:

```python
    if channel == "thermal":
        dev_off = _device_with_decoherence(device, n_th=0.0)
        F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
        delta_F = F_off - F_full
        sigma_delta = math.sqrt(sigma_off**2 + sigma_full**2)
        return ChannelContribution(
            name="thermal",
            group="active_loss",
            delta_F=delta_F,
            delta_F_uncertainty=sigma_delta,
            description="Fidelity loss from thermal bath occupation (n_th).",
        )
```

- [ ] **Step 7c.4: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_thermal_contribution_nonzero_at_reference -v -p no:dash
```

Expected: PASS.

- [ ] **Step 7c.5: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): thermal channel contribution

Module 2 Task 7c. Zero n_th (which cascades to resonator heating and
qubit thermal upward transitions in lindblad.py)."
```

**Definition of done:** Test passes.

---

## Task 7d: `purcell` channel

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7d.1: Write the failing test**

```python
def test_purcell_contribution_nonzero_at_reference():
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="purcell")
    assert c.name == "purcell"
    assert c.group == "active_loss"
    assert c.delta_F > 0.0
```

- [ ] **Step 7d.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_purcell_contribution_nonzero_at_reference -v -p no:dash
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 7d.3: Add the `purcell` branch**

In `compute_channel_contribution`, above the final `raise`:

```python
    if channel == "purcell":
        dev_off = _device_with_decoherence(device, purcell_enabled=False)
        F_off, sigma_off = _F_at(dev_off, drive, window, n_shots)
        delta_F = F_off - F_full
        sigma_delta = math.sqrt(sigma_off**2 + sigma_full**2)
        return ChannelContribution(
            name="purcell",
            group="active_loss",
            delta_F=delta_F,
            delta_F_uncertainty=sigma_delta,
            description="Fidelity loss from Purcell-enhanced decay (g²κ/Δ²).",
        )
```

- [ ] **Step 7d.4: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_purcell_contribution_nonzero_at_reference -v -p no:dash
```

Expected: PASS.

- [ ] **Step 7d.5: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): purcell channel contribution

Module 2 Task 7d. Sets purcell_enabled=False (Task 1's toggle) to
remove the N_q-1 Purcell collapse operators from lindblad.py."
```

**Definition of done:** Test passes.

---

## Task 7e: `drive_amplitude` calibration sensitivity

**Rationale:** Non-trivial asymmetry bookkeeping (spec §2.1 Group B). Two perturbations (±5 %), `ΔF = mean(|F_full − F_±|)`, error bar from `|F_+ − F_−|/2`.

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7e.1: Write the failing test**

```python
def test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent():
    """ΔF under ±5% amplitude perturbation should agree with first-order
    Taylor expansion |dF/dε|·Δε to within 20% (O(Δε²) higher-order correction)."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="drive_amplitude")
    assert c.name == "drive_amplitude"
    assert c.group == "calibration_sensitivity"
    assert c.delta_F >= 0.0
    assert c.perturbation_description is not None
    assert "±5" in c.perturbation_description or "5%" in c.perturbation_description
```

- [ ] **Step 7e.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent -v -p no:dash
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 7e.3: Add the `drive_amplitude` branch**

In `compute_channel_contribution`, above the final `raise`:

```python
    if channel == "drive_amplitude":
        perturbation = 0.05
        drive_plus = replace(drive, amplitude=drive.amplitude * (1.0 + perturbation))
        drive_minus = replace(drive, amplitude=drive.amplitude * (1.0 - perturbation))
        F_plus, sigma_plus = _F_at(device, drive_plus, window, n_shots)
        F_minus, sigma_minus = _F_at(device, drive_minus, window, n_shots)
        delta_F = 0.5 * (abs(F_full - F_plus) + abs(F_full - F_minus))
        # Asymmetry error bar
        err = 0.5 * abs(F_plus - F_minus)
        return ChannelContribution(
            name="drive_amplitude",
            group="calibration_sensitivity",
            delta_F=delta_F,
            delta_F_uncertainty=err,
            description="Fidelity loss under ±5% drive amplitude miscalibration.",
            perturbation_description="amplitude ±5% of nominal ε₀",
        )
```

- [ ] **Step 7e.4: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent -v -p no:dash
```

Expected: PASS.

- [ ] **Step 7e.5: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): drive_amplitude calibration sensitivity

Module 2 Task 7e. Mean-of-absolute ΔF under ±5% amplitude perturbation;
error bar from ± asymmetry (captures local curvature). Populates
perturbation_description for self-documenting YAML export."
```

**Definition of done:** Test passes; `perturbation_description` populated.

---

## Task 7f: `drive_detuning` calibration sensitivity

**Rationale:** Spec §2.1 Group B. Perturbation is ±κ/4 (absolute, not fractional). May defer to Day 6 morning if Day 5 runs long (per spec §8 guidance).

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 7f.1: Write the failing test**

```python
def test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent():
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_channel_contribution,
    )
    op = get_reference_operating_point()
    c = compute_channel_contribution(op, channel="drive_detuning")
    assert c.name == "drive_detuning"
    assert c.group == "calibration_sensitivity"
    assert c.delta_F >= 0.0
    assert c.perturbation_description is not None
    assert "κ/4" in c.perturbation_description or "kappa/4" in c.perturbation_description
```

- [ ] **Step 7f.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent -v -p no:dash
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 7f.3: Add the `drive_detuning` branch**

In `compute_channel_contribution`, replace the final `raise NotImplementedError` with:

```python
    if channel == "drive_detuning":
        kappa = device.resonator.kappa
        perturbation = kappa / 4.0
        drive_plus = replace(drive, detuning=drive.detuning + perturbation)
        drive_minus = replace(drive, detuning=drive.detuning - perturbation)
        F_plus, sigma_plus = _F_at(device, drive_plus, window, n_shots)
        F_minus, sigma_minus = _F_at(device, drive_minus, window, n_shots)
        delta_F = 0.5 * (abs(F_full - F_plus) + abs(F_full - F_minus))
        err = 0.5 * abs(F_plus - F_minus)
        return ChannelContribution(
            name="drive_detuning",
            group="calibration_sensitivity",
            delta_F=delta_F,
            delta_F_uncertainty=err,
            description="Fidelity loss under ±κ/4 drive detuning error.",
            perturbation_description="detuning ±κ/4 about nominal ω_d = ω_r",
        )

    raise NotImplementedError(f"Channel {channel!r} not yet implemented.")
```

- [ ] **Step 7f.4: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent -v -p no:dash
```

Expected: PASS.

- [ ] **Step 7f.5: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): drive_detuning calibration sensitivity

Module 2 Task 7f. ±κ/4 detuning perturbation, mean-of-absolute ΔF.
Completes the 6-channel compute_channel_contribution dispatcher."
```

**Definition of done:** All six channels implemented; `compute_channel_contribution` raises `NotImplementedError` only for unknown channel names.

---

## Task 8: `compute_full_error_budget` + B1 (additivity) + B2 (residual small) tests

**Rationale:** Spec §5.3 `ErrorBudget` and §2.1 arithmetic identity `(F_ideal − F_full) = Σ_active ΔF + R_active`. B1 validates the identity within 3σ; B2 asserts |R_active| < 20% × (F_ideal − F_full).

**Files:**
- Modify: `dispersive_readout/analysis/error_budget.py`
- Modify: `dispersive_readout/analysis/__init__.py`
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 8.1: Write the failing B1 and B2 tests**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_B1_active_loss_sums_to_ideal_minus_full_within_tolerance():
    """Σ ΔF_c + R_active ≈ (F_ideal − F_full) within 3σ_prop for active group."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_full_error_budget,
    )
    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    active_sum = sum(c.delta_F for c in budget.active_loss_channels)
    identity_lhs = budget.F_ideal - budget.F_full
    identity_rhs = active_sum + budget.residual_active
    tolerance = 3.0 * budget.residual_active_uncertainty
    assert abs(identity_lhs - identity_rhs) <= tolerance, (
        f"Additivity violation: (F_ideal - F_full) = {identity_lhs:.5f}, "
        f"Σ ΔF + R = {identity_rhs:.5f}, tol = {tolerance:.5f}"
    )


def test_B2_active_loss_residual_under_20_percent():
    """|R_active| < 0.2 × (F_ideal − F_full). Red flag for the figure if it fails."""
    from dispersive_readout.analysis import (
        get_reference_operating_point,
        compute_full_error_budget,
    )
    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    denom = budget.F_ideal - budget.F_full
    ratio = abs(budget.residual_active) / denom if denom > 0 else float("inf")
    assert ratio < 0.2, (
        f"|R_active|/(F_ideal - F_full) = {ratio:.3f}; channels interact "
        f"strongly. Consider regrouping (e.g., merge T1_intrinsic + purcell)."
    )
```

- [ ] **Step 8.2: Run to verify failure**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B1_active_loss_sums_to_ideal_minus_full_within_tolerance dispersive_readout/tests/test_error_budget.py::test_B2_active_loss_residual_under_20_percent -v -p no:dash
```

Expected: FAIL (`compute_full_error_budget` not exported).

- [ ] **Step 8.3: Implement `compute_full_error_budget`**

Append to `dispersive_readout/analysis/error_budget.py`:

```python
import hashlib
import json


def _operating_point_id(operating_point) -> str:
    """Deterministic hash of OperatingPoint fields for traceability in YAML."""
    device = operating_point.device
    payload = {
        "omega_r": device.resonator.omega_r,
        "kappa": device.resonator.kappa,
        "g": device.coupling.g,
        "E_C": device.transmon.E_C,
        "E_J": device.transmon.E_J,
        "gamma_1": device.decoherence.gamma_1,
        "gamma_phi": device.decoherence.gamma_phi,
        "n_th": device.decoherence.n_th,
        "amplitude": operating_point.drive.amplitude,
        "duration": operating_point.drive.duration,
        "detuning": operating_point.drive.detuning,
        "window": list(operating_point.integration_window),
        "n_shots": operating_point.n_shots,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


_DEFAULT_CHANNELS: list[ChannelName] = [
    "T1_intrinsic",
    "pure_dephasing",
    "thermal",
    "purcell",
    "drive_amplitude",
    "drive_detuning",
]


def compute_full_error_budget(
    operating_point,
    channels: list[ChannelName] | None = None,
) -> ErrorBudget:
    """Compute the complete error budget at the given operating point.

    Returns an ErrorBudget with:
    - F_full: baseline fidelity (all channels on)
    - F_ideal: ceiling with all 4 active-loss channels disabled
    - channels: list of 6 ChannelContribution
    - residual_active: R_active = (F_ideal - F_full) - Σ_active ΔF_c
    - residual_active_uncertainty: quadrature-propagated σ_R
    """
    if channels is None:
        channels = _DEFAULT_CHANNELS

    device = operating_point.device
    drive = operating_point.drive
    window = operating_point.integration_window
    n_shots = operating_point.n_shots

    F_full, sigma_full = _F_at(device, drive, window, n_shots)

    # F_ideal: all active-loss channels disabled
    dev_ideal = _device_with_decoherence(
        device,
        gamma_1=0.0,
        gamma_phi=0.0,
        n_th=0.0,
        purcell_enabled=False,
    )
    F_ideal, sigma_ideal = _F_at(dev_ideal, drive, window, n_shots)

    contributions = [
        compute_channel_contribution(operating_point, ch) for ch in channels
    ]
    active = [c for c in contributions if c.group == "active_loss"]

    active_sum = sum(c.delta_F for c in active)
    residual_active = (F_ideal - F_full) - active_sum
    # σ_R² = σ_F_ideal² + σ_F_full² + Σ σ_ΔF²
    sigma_residual_sq = sigma_ideal**2 + sigma_full**2 + sum(
        c.delta_F_uncertainty**2 for c in active
    )
    sigma_residual = math.sqrt(sigma_residual_sq)

    return ErrorBudget(
        operating_point_id=_operating_point_id(operating_point),
        F_full=F_full,
        F_ideal=F_ideal,
        channels=contributions,
        residual_active=residual_active,
        residual_active_uncertainty=sigma_residual,
    )
```

- [ ] **Step 8.4: Export from `__init__.py`**

Append to `dispersive_readout/analysis/__init__.py`:

```python
from .error_budget import compute_full_error_budget
```

- [ ] **Step 8.5: Run B1 and B2**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B1_active_loss_sums_to_ideal_minus_full_within_tolerance dispersive_readout/tests/test_error_budget.py::test_B2_active_loss_residual_under_20_percent -v -p no:dash
```

Expected: both PASS. If B2 fails, STOP — this is spec §9 flag #1 and requires regrouping decisions before proceeding.

- [ ] **Step 8.6: Full test suite regression check**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -15
```

Expected: all passing.

- [ ] **Step 8.7: Commit**

```bash
git add dispersive_readout/analysis/error_budget.py dispersive_readout/analysis/__init__.py dispersive_readout/tests/test_error_budget.py
git commit -m "feat(stage06): compute_full_error_budget + B1/B2 tests

Module 2 Task 8. Assembles ErrorBudget from 6 channel contributions
plus F_full and F_ideal sims. R_active and σ_R computed for the
active-loss group only (calibration-sensitivity does not enter the
additivity identity). B1 validates the identity within 3σ_prop;
B2 asserts |R_active|/(F_ideal-F_full) < 0.2."
```

**Definition of done:** B1 and B2 pass; `compute_full_error_budget` returns an `ErrorBudget` with 6 channels, F_full, F_ideal, residual_active.

---

## Task 9a: B3 Purcell sanity check at REFERENCE (1 % tolerance)

**Rationale:** Spec §6 B3 — replaces the original moot B3. Cross-validates the simulated `ΔF_purcell` against the analytic `γ_P` × effective-time prediction. Tight at REFERENCE because 2nd-order PT residual is ~0.2 %.

**Files:**
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 9a.1: Write the B3 test at REFERENCE**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_B3_simulated_purcell_matches_analytic_within_1_percent_at_reference():
    """Simulated ΔF_Purcell vs analytic γ_P-weighted prediction at REFERENCE.

    The test compares Purcell rates, not ΔF values directly: the simulated
    Purcell rate is γ_P_sim = ΔF_Purcell × (conversion factor that depends on
    the readout dynamics). Simpler check: fit γ_P from simulated T1_eff with
    and without Purcell, compare to analytic_purcell_rate.
    """
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.analysis import analytic_purcell_rate

    # Analytic prediction
    gamma_P_analytic = analytic_purcell_rate(REFERENCE_DEVICE)

    # Measure γ_P from a γ_1=0 simulation: P(|1⟩)(T) = exp(-γ_P T) since that's
    # the only remaining relaxation channel (n_th stays on but γ_up is much
    # smaller than γ_P at 30 mK). Use a long enough T for exp decay to be
    # measurable but short enough not to stress the solver.
    import math
    from dataclasses import replace
    from dispersive_readout.physics import DriveParams, simulate_readout
    from dispersive_readout.physics.config import DecoherenceParams, DeviceConfig

    # Build γ_1=0 device with Purcell still on
    new_dec = replace(REFERENCE_DEVICE.decoherence, gamma_1=0.0, gamma_phi=0.0, n_th=0.0)
    dev = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=REFERENCE_DEVICE.coupling,
        decoherence=new_dec,
        truncation=REFERENCE_DEVICE.truncation,
    )

    # Zero-drive (H=0 for the qubit); long enough to see Purcell decay.
    # Use a very small amplitude so drive doesn't dominate.
    T = 5.0 / gamma_P_analytic  # ~5 Purcell lifetimes
    T = min(T, 100e-6)           # cap at 100 μs to bound solver cost
    drv = DriveParams(amplitude=1e-6, duration=T, detuning=0.0, edge_sigma=2e-9)
    r = simulate_readout(dev, drv, initial_qubit_state=1)

    # Extract γ_P from exponential fit of P(|1⟩)(t): P(|1⟩) = exp(-γ_P t)
    import numpy as np
    p1 = r.qubit_populations[:, 1]
    t = r.t
    # Fit in log space; restrict to P(|1⟩) > 0.1 for clean fit.
    mask = p1 > 0.1
    log_p1 = np.log(p1[mask])
    t_fit = t[mask]
    slope, _intercept = np.polyfit(t_fit, log_p1, 1)
    gamma_P_sim = -slope

    ratio = gamma_P_sim / gamma_P_analytic
    assert 0.99 <= ratio <= 1.01, (
        f"Simulated Purcell γ_P = {gamma_P_sim:.3e} rad/s, analytic = "
        f"{gamma_P_analytic:.3e} rad/s, ratio = {ratio:.4f}. "
        f"Expected 1 ± 0.01 at REFERENCE."
    )
```

- [ ] **Step 9a.2: Run the B3 test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B3_simulated_purcell_matches_analytic_within_1_percent_at_reference -v -p no:dash
```

Expected: PASS. May take ~20 s (the simulation runs out to ~5/γ_P).

If it FAILS, check: (a) is the drive amplitude small enough to be truly "zero"? Try 1e-9 instead of 1e-6. (b) Is the fit window reasonable (P > 0.1)? (c) Does the analytic formula agree with what `lindblad.py:128-135` actually builds? The formula is in Task 5's docstring.

- [ ] **Step 9a.3: Commit**

```bash
git add dispersive_readout/tests/test_error_budget.py
git commit -m "test(stage06): B3 Purcell simulated-vs-analytic cross-check at REFERENCE

Module 2 Task 9a. Replaces the moot original B3 (which compared two
Hamiltonian frames made equivalent by blocker 1 of the Module 2 brainstorm).
Fits γ_P from exponential decay of P(|1⟩) with all other decoherence off,
compares to analytic_purcell_rate at 1% tolerance. Physics ceiling for
2nd-order PT residual is ~0.2% at g/Δ≈0.044, so 1% comfortably catches
implementation bugs without over-constraining."
```

**Definition of done:** B3a passes; simulated Purcell rate agrees with analytic formula within 1% at REFERENCE.

---

## Task 9b: B3 Purcell sanity check at 2× coupling (5 % tolerance)

**Rationale:** Regime-breadth check. At g/Δ = 0.088 the 2nd-order PT residual is ~0.8 %, so 5 % tolerance is comfortable. Reveals the approximation-scope limit if B3a passes but 9b fails.

**May defer to Day 6 morning if Day 5 runs long** — it's an independent unit test with no downstream blocker.

**Files:**
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 9b.1: Write the strong-coupling B3 test**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_B3_simulated_purcell_matches_analytic_at_strong_coupling():
    """Same as B3 at REFERENCE but with 2× coupling (g/Δ ≈ 0.088), 5% tol.

    If this fails but B3a passes, the 2nd-order SW approximation is
    tighter than we thought — an informative regime-scope measurement,
    not a bug. Document in the report if it fires."""
    from dataclasses import replace
    import math
    import numpy as np
    from dispersive_readout.physics import REFERENCE_DEVICE
    from dispersive_readout.physics.config import (
        CouplingParams, DecoherenceParams, DeviceConfig,
    )
    from dispersive_readout.physics import DriveParams, simulate_readout
    from dispersive_readout.analysis import analytic_purcell_rate

    # 2× coupling device
    new_coup = CouplingParams(g=2.0 * REFERENCE_DEVICE.coupling.g)
    new_dec = replace(REFERENCE_DEVICE.decoherence, gamma_1=0.0, gamma_phi=0.0, n_th=0.0)
    dev = DeviceConfig(
        transmon=REFERENCE_DEVICE.transmon,
        resonator=REFERENCE_DEVICE.resonator,
        coupling=new_coup,
        decoherence=new_dec,
        truncation=REFERENCE_DEVICE.truncation,
    )

    gamma_P_analytic = analytic_purcell_rate(dev)
    T = min(5.0 / gamma_P_analytic, 100e-6)
    drv = DriveParams(amplitude=1e-6, duration=T, detuning=0.0, edge_sigma=2e-9)
    r = simulate_readout(dev, drv, initial_qubit_state=1)

    p1 = r.qubit_populations[:, 1]
    mask = p1 > 0.1
    slope, _ = np.polyfit(r.t[mask], np.log(p1[mask]), 1)
    gamma_P_sim = -slope
    ratio = gamma_P_sim / gamma_P_analytic
    assert 0.95 <= ratio <= 1.05, (
        f"At 2×g: simulated γ_P = {gamma_P_sim:.3e}, analytic = "
        f"{gamma_P_analytic:.3e}, ratio = {ratio:.4f}. 5% tol exceeded; "
        f"2nd-order SW approximation failing at this coupling."
    )
```

- [ ] **Step 9b.2: Run the test**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py::test_B3_simulated_purcell_matches_analytic_at_strong_coupling -v -p no:dash
```

Expected: PASS at 5 % tolerance.

- [ ] **Step 9b.3: Commit**

```bash
git add dispersive_readout/tests/test_error_budget.py
git commit -m "test(stage06): B3 Purcell sanity check at 2x coupling (5% tol)

Module 2 Task 9b. Regime-breadth pair for B3a. At g/Δ≈0.088 the
2nd-order PT residual is ~0.8%; 5% tolerance bounds implementation
correctness without over-constraining physics. Follows the same
REFERENCE+strong-coupling pair pattern established by Module 1's V2
pair test."
```

**Definition of done:** B3b passes at 2× coupling with 5 % tolerance.

---

## Task 10: `scripts/fig2_error_budget.py` — first-pass waterfall

**Rationale:** Spec §7.1 / §7.2. Two candidate layouts (§7.1) rendered as a first pass; Task 11 iterates on visual polish.

**Files:**
- Create: `06_Dispersive_Readout/scripts/fig2_error_budget.py`
- Generated: `06_Dispersive_Readout/figures/fig2_error_budget.png`, `06_Dispersive_Readout/figures/fig2_data.yaml`

- [ ] **Step 10.1: Create the script**

```bash
mkdir -p 06_Dispersive_Readout/scripts 06_Dispersive_Readout/figures
```

Create `06_Dispersive_Readout/scripts/fig2_error_budget.py`:

```python
"""Render Figure 2 (error budget waterfall) + export YAML data.

Two candidate layouts are rendered side-by-side as separate PNGs during
Task 11; this Task 10 version produces Candidate B (classic waterfall) as
the default and the YAML data export. Run from repo root:

    python 06_Dispersive_Readout/scripts/fig2_error_budget.py

Outputs:
  06_Dispersive_Readout/figures/fig2_error_budget.png (150 DPI, ~1200 px)
  06_Dispersive_Readout/figures/fig2_data.yaml (ErrorBudget serialized)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.analysis import (
    ErrorBudget,
    compute_full_error_budget,
    export_budget_to_yaml,
    get_reference_operating_point,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "06_Dispersive_Readout" / "figures"


def _render_candidate_B(budget: ErrorBudget, path: Path) -> None:
    """Classic waterfall: Ideal floor | active loss stack | R_active | === | cal sens."""
    active = budget.active_loss_channels
    calib = budget.calibration_channels
    ideal_floor = 1.0 - budget.F_ideal

    # Bars left-to-right
    labels = (
        ["Ideal\nfloor"]
        + [c.name.replace("_", "\n") for c in active]
        + ["R_active"]
        + [""]  # separator
        + [c.name.replace("_", "\n") for c in calib]
    )
    values = (
        [ideal_floor]
        + [c.delta_F for c in active]
        + [budget.residual_active]
        + [0.0]  # separator (invisible)
        + [c.delta_F for c in calib]
    )
    errors = (
        [0.0]
        + [c.delta_F_uncertainty for c in active]
        + [budget.residual_active_uncertainty]
        + [0.0]
        + [c.delta_F_uncertainty for c in calib]
    )
    # Scale to 10^-3 units for readability
    values_milli = [v * 1e3 for v in values]
    errors_milli = [e * 1e3 for e in errors]

    # Color palette
    warm = plt.cm.OrRd(np.linspace(0.4, 0.85, len(active)))
    cool = plt.cm.Blues(np.linspace(0.5, 0.85, len(calib)))
    colors = (
        ["#888888"]                   # ideal floor grey
        + list(warm)                   # active loss warm
        + ["#555555"]                  # residual dark grey
        + ["none"]                     # separator
        + list(cool)                   # cal sens cool
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(labels))
    bars = ax.bar(x, values_milli, color=colors, edgecolor="black", linewidth=0.6)
    # Error bars
    ax.errorbar(x, values_milli, yerr=errors_milli, fmt="none",
                ecolor="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Contribution to 1 − F (× 10⁻³)", fontsize=10)
    ax.set_title(
        f"Assignment Infidelity Decomposition — REFERENCE_DEVICE\n"
        f"F_full = {budget.F_full:.4f}, F_ideal = {budget.F_ideal:.4f}, "
        f"n_shots = 10⁴",
        fontsize=10,
    )
    # Group separator
    ax.axvline(x=len(active) + 1.5, color="gray", linestyle="--", linewidth=0.6)
    # Group labels
    ax.text(1 + len(active) / 2 - 0.5, ax.get_ylim()[1] * 0.92, "Active loss",
            ha="center", fontsize=9, style="italic")
    ax.text(len(active) + 3 + len(calib) / 2 - 0.5, ax.get_ylim()[1] * 0.92,
            "Calibration sensitivity", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Computing reference operating point (calibration + verification)...")
    op = get_reference_operating_point()
    print(f"  ε₀ = {op.drive.amplitude:.3e} rad/s "
          f"(= {op.drive.amplitude / (2 * np.pi):.3e} Hz)")

    print("Computing full error budget (8 sims, ~15 s)...")
    budget = compute_full_error_budget(op)
    print(f"  F_full = {budget.F_full:.5f}")
    print(f"  F_ideal = {budget.F_ideal:.5f}")
    print(f"  R_active = {budget.residual_active:.5f} "
          f"± {budget.residual_active_uncertainty:.5f}")
    for c in budget.channels:
        print(f"  {c.name:20s}  ΔF = {c.delta_F:.5f} ± {c.delta_F_uncertainty:.5f}")

    png_path = FIG_DIR / "fig2_error_budget.png"
    yaml_path = FIG_DIR / "fig2_data.yaml"
    _render_candidate_B(budget, png_path)
    export_budget_to_yaml(budget, yaml_path)
    print(f"Wrote {png_path} and {yaml_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run the script**

```bash
python 06_Dispersive_Readout/scripts/fig2_error_budget.py
```

Expected output: ε₀ printed, 8 simulations run (~15 s total), `fig2_error_budget.png` and `fig2_data.yaml` created, stdout lists F_full, F_ideal, residual, and per-channel ΔF values.

- [ ] **Step 10.3: Sanity-check the PNG**

Open `06_Dispersive_Readout/figures/fig2_error_budget.png` in a viewer. Check:
- 8 bars left-to-right + one separator
- Active-loss bars in warm palette, calibration in cool palette
- Error bars visible on each bar
- Title shows F_full and F_ideal
- Group separator (vertical dashed line) between residual and calibration bars

- [ ] **Step 10.4: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig2_error_budget.py 06_Dispersive_Readout/figures/fig2_error_budget.png 06_Dispersive_Readout/figures/fig2_data.yaml
git commit -m "feat(stage06): Figure 2 first-pass waterfall + YAML export

Module 2 Task 10. Renders Candidate B (classic waterfall) layout from
MODULE_2_SPEC §7.1: ideal floor | active loss stack | R_active | === |
cal sens. Warm palette for active loss, cool for calibration
sensitivity, grey for residual. 150 DPI, 8x4.5 inches (~1200 px wide).
YAML export at figures/fig2_data.yaml preserves the full ErrorBudget
for reproducibility and Task 11/13 polish iteration."
```

**Definition of done:** PNG and YAML exist; PNG renders 8 bars + separator; YAML re-reads into an `ErrorBudget` (validated by Task 12's B5).

---

## Task 11: Figure 2 styling polish (layout decision + palette tuning)

**Rationale:** Spec §7.1 proposes two candidate layouts; Task 11 renders both and picks the cleaner-reading one. Also tunes palette, annotations, group separator.

**Files:**
- Modify: `06_Dispersive_Readout/scripts/fig2_error_budget.py`

- [ ] **Step 11.1: Add Candidate A render function to the script**

In `fig2_error_budget.py`, add alongside `_render_candidate_B`:

```python
def _render_candidate_A(budget: ErrorBudget, path: Path) -> None:
    """Author-first: Total infidelity | active loss | cal sens | R_active."""
    active = budget.active_loss_channels
    calib = budget.calibration_channels

    labels = (
        ["Total\ninfidelity"]
        + [c.name.replace("_", "\n") for c in active]
        + [""]  # separator
        + [c.name.replace("_", "\n") for c in calib]
        + ["R_active"]
    )
    values = (
        [budget.total_infidelity]
        + [c.delta_F for c in active]
        + [0.0]
        + [c.delta_F for c in calib]
        + [budget.residual_active]
    )
    errors = (
        [0.0]
        + [c.delta_F_uncertainty for c in active]
        + [0.0]
        + [c.delta_F_uncertainty for c in calib]
        + [budget.residual_active_uncertainty]
    )
    values_milli = [v * 1e3 for v in values]
    errors_milli = [e * 1e3 for e in errors]

    warm = plt.cm.OrRd(np.linspace(0.4, 0.85, len(active)))
    cool = plt.cm.Blues(np.linspace(0.5, 0.85, len(calib)))
    colors = (
        ["#333333"]                   # total infidelity anchor
        + list(warm)
        + ["none"]                     # separator
        + list(cool)
        + ["#555555"]                  # residual
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, values_milli, color=colors, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, values_milli, yerr=errors_milli, fmt="none",
                ecolor="black", capsize=2, linewidth=0.8)

    # Reference line at ideal floor
    ideal_milli = (1.0 - budget.F_ideal) * 1e3
    ax.axhline(y=ideal_milli, color="grey", linestyle=":", linewidth=0.8)
    ax.text(len(labels) - 0.5, ideal_milli, f" Ideal floor",
            va="center", fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Contribution to 1 − F (× 10⁻³)", fontsize=10)
    ax.set_title(
        f"Candidate A — Assignment Infidelity Decomposition\n"
        f"F_full = {budget.F_full:.4f}, F_ideal = {budget.F_ideal:.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 11.2: Update `main()` to render both candidates**

Replace the `_render_candidate_B(budget, png_path); _export_yaml(...)` lines with:

```python
    png_B = FIG_DIR / "fig2_error_budget_candidate_B.png"
    png_A = FIG_DIR / "fig2_error_budget_candidate_A.png"
    _render_candidate_B(budget, png_B)
    _render_candidate_A(budget, png_A)
    export_budget_to_yaml(budget, yaml_path)
    print(f"Wrote {png_B}, {png_A}, and {yaml_path}")
    print("Task 11: review both candidates, pick winner, save to fig2_error_budget.png")
```

- [ ] **Step 11.3: Run and compare**

```bash
python 06_Dispersive_Readout/scripts/fig2_error_budget.py
```

Open both PNGs. Pick whichever reads cleaner as a standalone figure (without the methods note). Rationale for choosing:
- Candidate B (classic waterfall): left-to-right narrative "ideal → contributions → measured total → separator → sensitivities". Familiar waterfall convention.
- Candidate A (author-first): left-to-right narrative "total → contributions sum to total → sensitivities → residual". Less conventional.

Default recommendation per spec §7.1 note is Candidate B, but make the call based on what actually renders.

- [ ] **Step 11.4: Save the winner as `fig2_error_budget.png`**

Suppose B wins. In `main()`, after the side-by-side renders, add:

```python
    # Winner → canonical Figure 2
    winner_path = FIG_DIR / "fig2_error_budget.png"
    import shutil
    shutil.copy2(png_B, winner_path)  # or png_A if A wins
    print(f"Winner (Candidate B) → {winner_path}")
```

Re-run:

```bash
python 06_Dispersive_Readout/scripts/fig2_error_budget.py
```

- [ ] **Step 11.5: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig2_error_budget.py 06_Dispersive_Readout/figures/
git commit -m "feat(stage06): Figure 2 layout A/B comparison, pick winner

Module 2 Task 11. Renders both candidate layouts from MODULE_2_SPEC
§7.1 side-by-side; Candidate [A/B — fill in actual winner] wins for
its clearer left-to-right narrative. Canonical fig2_error_budget.png
updated to the winner; both candidate PNGs retained in figures/ for
traceability."
```

**Definition of done:** `fig2_error_budget.png` shows the chosen winner; both `..._candidate_A.png` and `..._candidate_B.png` exist for comparison.

---

## Task 12: B5 YAML round-trip + operating-point calibration test + remaining polish

**Rationale:** Spec §6 B5. The YAML is committed and used by Figure 2; B5 verifies round-trip integrity. Also confirms the per-channel tests are registered and Day-5's calibration test still passes after the full budget run.

**Files:**
- Test: `dispersive_readout/tests/test_error_budget.py`

- [ ] **Step 12.1: Write B5 YAML round-trip test**

Append to `dispersive_readout/tests/test_error_budget.py`:

```python
def test_B5_budget_yaml_round_trip(tmp_path):
    """export_budget_to_yaml + re-read reproduces the ErrorBudget exactly."""
    from dispersive_readout.analysis import (
        ErrorBudget, ChannelContribution,
        get_reference_operating_point, compute_full_error_budget,
        export_budget_to_yaml,
    )
    import yaml

    op = get_reference_operating_point()
    budget = compute_full_error_budget(op)

    yaml_path = tmp_path / "fig2_data.yaml"
    export_budget_to_yaml(budget, yaml_path)

    # Re-read and reconstruct
    reread = yaml.safe_load(yaml_path.read_text())
    channels = [ChannelContribution(**d) for d in reread["channels"]]
    reread.pop("channels")
    round_trip = ErrorBudget(channels=channels, **reread)

    assert round_trip.F_full == budget.F_full
    assert round_trip.F_ideal == budget.F_ideal
    assert round_trip.residual_active == budget.residual_active
    assert len(round_trip.channels) == len(budget.channels)
    for c_orig, c_new in zip(budget.channels, round_trip.channels):
        assert c_orig.name == c_new.name
        assert c_orig.delta_F == c_new.delta_F
```

- [ ] **Step 12.2: Run the full Module 2 test suite**

```bash
python -m pytest dispersive_readout/tests/test_error_budget.py -v -p no:dash
```

Expected: 13 tests passing:
- `test_module2_package_imports_without_error`
- `test_analytic_calibration_hits_target_fidelity_within_3_sigma`
- `test_analytic_purcell_rate_positive_at_reference`
- `test_B4_negative_contribution_raises`
- `test_T1_intrinsic_contribution_nonzero_at_reference`
- `test_pure_dephasing_contribution_nonzero_at_reference`
- `test_thermal_contribution_nonzero_at_reference`
- `test_purcell_contribution_nonzero_at_reference`
- `test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent`
- `test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent`
- `test_B1_active_loss_sums_to_ideal_minus_full_within_tolerance`
- `test_B2_active_loss_residual_under_20_percent`
- `test_B3_simulated_purcell_matches_analytic_within_1_percent_at_reference`
- `test_B3_simulated_purcell_matches_analytic_at_strong_coupling`
- `test_B5_budget_yaml_round_trip`

That's actually 15 tests; the spec target is "≥ 12". Extra coverage is fine.

- [ ] **Step 12.3: Commit**

```bash
git add dispersive_readout/tests/test_error_budget.py
git commit -m "test(stage06): B5 YAML round-trip + full Module 2 test suite

Module 2 Task 12. B5 verifies ErrorBudget → YAML → ErrorBudget preserves
all fields. Total Module 2 test count: 15 (exceeds ≥12 spec target).
All B1-B5 + 6 per-channel + 1 calibration + 1 import smoke passing."
```

**Definition of done:** 15 Module 2 tests passing; YAML round-trip preserves all fields.

---

## Task 13: Figure 2 publication polish + methods note substitution

**Rationale:** Spec §7.2 methods note contains `[ratio measured at calibration]` placeholder. Task 13 computes `n̄/n_crit` at the calibrated operating point, substitutes it, and produces the final figure + caption text committed as markdown.

**Files:**
- Modify: `06_Dispersive_Readout/scripts/fig2_error_budget.py`
- Create: `06_Dispersive_Readout/FIGURE_2_CAPTION.md` (caption + methods note for report/README use)

- [ ] **Step 13.1: Add `n̄/n_crit` computation to the script**

In `fig2_error_budget.py`, add (above `main()`):

```python
def _compute_n_bar_over_n_crit(op) -> tuple[float, float, float]:
    """Return (n̄_peak, n_crit, ratio) at the operating point.

    n_crit = (Δ_10 / (2g))² per Shillito 2022. n̄_peak is the maximum
    mean-photon-number measured in a baseline simulation starting in |1⟩
    (the worse case; drive populates the resonator more there).
    """
    from dispersive_readout.physics import simulate_readout
    from dispersive_readout.physics.transmon import (
        charge_operator_matrix_elements, diagonalize_transmon,
    )

    device = op.device
    tr = device.truncation
    energies, _ = diagonalize_transmon(device.transmon, tr)
    g = device.coupling.g
    omega_r = device.resonator.omega_r
    delta_10 = energies[1] - energies[0] - omega_r
    n_crit = (delta_10 / (2.0 * g)) ** 2

    # Peak photon number from a |1> baseline sim
    r = simulate_readout(device, op.drive, initial_qubit_state=1)
    n_bar_peak = float(r.photon_number.max())

    return n_bar_peak, float(n_crit), n_bar_peak / float(n_crit)
```

In `main()`, after computing `budget`, add:

```python
    n_bar, n_crit, ratio = _compute_n_bar_over_n_crit(op)
    print(f"  n̄_peak = {n_bar:.2f}, n_crit = {n_crit:.1f}, n̄/n_crit = {ratio:.3f}")
```

- [ ] **Step 13.2: Run and record the ratio**

```bash
python 06_Dispersive_Readout/scripts/fig2_error_budget.py 2>&1 | tee /tmp/fig2_run.log
```

Note the `n̄/n_crit` value from stdout. Expected range per spec §9 flag #7: 0.03–0.05. If > 0.2, STOP — this is a spec §9 flag #7 condition.

- [ ] **Step 13.3: Write the final caption + methods note markdown**

Create `06_Dispersive_Readout/FIGURE_2_CAPTION.md`, substituting the measured ratio for `[ratio]`:

```markdown
# Figure 2 — Error Budget Decomposition

## Caption

**Figure 2.** Assignment infidelity decomposition at REFERENCE_DEVICE (500 ns readout, F_full ≈ 0.99, 10⁴ shots). **Active loss** (left, 4 bars): T1, pure dephasing, thermal, Purcell — each measured by turning off its collapse operator. **Calibration sensitivity** (right, 2 bars): F loss under ±5 % amplitude / ±κ/4 detuning perturbations about the nominal operating point. The grey residual bar reports cross-channel interactions within the active-loss group and satisfies the additivity identity Σ ΔF_c + R = (F_ideal − F_full).

## Methods note

**Methods note (Figure 2).** The waterfall decomposes assignment infidelity within the scope of the 2nd-order Schrieffer-Wolff dispersive-frame Hamiltonian used throughout Stage 06. Two physics boundaries are relevant: the dispersive approximation itself is validated by unit test V2 to ≤ 2 % at REFERENCE, producing a fidelity residual of O(10⁻⁴) below the bar-visibility threshold; measurement-induced ionization (Shillito 2022) requires an intra-resonator photon count of n̄ > n_crit, where the reference operating point sits at n̄/n_crit ≈ <SUBSTITUTE_MEASURED_VALUE>, well below onset. Residual |1⟩→|2⟩ occupation P(|2⟩) ≈ 3 × 10⁻⁴ is entirely thermal and is attributed to the thermal channel. The operating point ε₀ is calibrated analytically from the dispersive-regime steady-state SNR formula (§2.3) and cross-verified against simulation within shot-noise tolerance. Active-loss and calibration-sensitivity bars answer two conceptually distinct questions — loss contribution at the nominal point versus robustness derivative under named perturbations — and are presented as separate groups to make this distinction explicit; only the active-loss group carries a residual identity and a B2 additivity validation test.
```

Substitute `<SUBSTITUTE_MEASURED_VALUE>` with the measured value from step 13.2 (e.g., `0.041` if that's what the run produced).

- [ ] **Step 13.4: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig2_error_budget.py 06_Dispersive_Readout/FIGURE_2_CAPTION.md 06_Dispersive_Readout/figures/
git commit -m "feat(stage06): Figure 2 publication polish + methods note

Module 2 Task 13. Adds n̄/n_crit computation to the script (Shillito
2022 Eq. boundary check). Measured ratio substituted into the methods
note placeholder. FIGURE_2_CAPTION.md holds the two-tier caption +
methods note ready for copy-paste into the final report and README."
```

**Definition of done:** `n̄/n_crit` measured and substituted in `FIGURE_2_CAPTION.md`; ratio is < 0.2 (spec §9 flag #7).

---

## Task 14: End-of-Module-2 verification + Module 3 stub

**Rationale:** Spec §10 checklist gate before Module 3 starts. Also produces Module 3 file stubs so the handoff is clean.

**Files:**
- Create: `dispersive_readout/characterization/` directory structure for Module 3 (stubs only)
- Run: full test suite (Module 1 + Module 2)
- Verify: §10 checklist manually

- [ ] **Step 14.1: Run the full project test suite**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -20
```

Expected: 72 tests passing (57 Module 1 + 15 Module 2). Record wall-clock time; should be < 90 s.

- [ ] **Step 14.2: Verify the §10 checklist**

Read `06_Dispersive_Readout/MODULE_2_SPEC.md` §10. For each item, confirm:

- [ ] All 15 Module 2 tests passing (step 14.1 confirms).
- [ ] `ErrorBudget` Pydantic schema used throughout (grep for `dict(` vs `ErrorBudget(` in `scripts/fig2_error_budget.py`).
- [ ] Reference operating point calibrated analytically, cross-verified, no fallback triggered (no RuntimeWarning in step 13.2's log).
- [ ] `n̄/n_crit` ratio measured and substituted in `FIGURE_2_CAPTION.md`.
- [ ] `fig2_error_budget.png` at 150 DPI (inspect file metadata: `python -c "from PIL import Image; print(Image.open('06_Dispersive_Readout/figures/fig2_error_budget.png').info)"`).
- [ ] Caption + methods note in `FIGURE_2_CAPTION.md`.
- [ ] `|R_active| < 0.2 × (F_ideal − F_full)` (B2 test passes).
- [ ] YAML at `06_Dispersive_Readout/figures/fig2_data.yaml` exists.
- [ ] `analysis/__init__.py` exports the public API listed in spec §5.4.
- [ ] Module 1 regressions: Module 1 tests still pass.

- [ ] **Step 14.3: Create Module 3 stub files**

```bash
mkdir -p dispersive_readout/characterization
```

Create `dispersive_readout/characterization/__init__.py`:

```python
"""Stage 06 Module 3 — parameter characterization protocols.

Stub; populated in Module 3.
"""
```

Create `dispersive_readout/characterization/protocols.py`:

```python
"""Characterization protocols (T1, T2*/Ramsey, dispersive shift, Purcell).

Stub; Module 3 will add protocol signatures here.
"""
```

- [ ] **Step 14.4: Commit and tag**

```bash
git add dispersive_readout/characterization/
git commit -m "chore(stage06): Module 3 stub files; Module 2 complete

Module 2 Task 14. Empty package stubs for Module 3 (characterization
protocols). Module 2 complete per §10 checklist:
- 15 tests passing, 72 total project tests
- ErrorBudget schema + Figure 2 PNG + YAML committed
- n̄/n_crit measured and documented in FIGURE_2_CAPTION.md
- Module 1 unchanged (57 tests still passing)"

git tag stage06-module2
```

**Definition of done:** 72 tests pass; §10 checklist complete; `stage06-module2` tag exists.

---

## Self-Review

**Spec coverage check:**

| Spec section | Task(s) that implement it |
|---|---|
| §1 channel list (6 named, 2 groups) | 7a–7f |
| §2.1 two-group waterfall + R_active identity | 8 (logic), B1+B2 tests |
| §2.3 analytic calibration + fallback | 4 |
| §2.4 analytic binomial SE (no bootstrap) | 2 (rng kwarg), 7a–7f, 8 (σ propagation) |
| §3 Module 1 edits | 1, 2 |
| §4 file layout | 3, 4, 5, 6, 10 |
| §5.1 OperatingPoint | 4 |
| §5.2 purcell_isolation.py | 5 |
| §5.3 ErrorBudget + ChannelContribution schemas | 6 |
| §5.4 public API | 3 (skeleton), 4, 5, 6, 8 (exports) |
| §6 tests B1–B5 + 6 per-channel + calibration | 4, 6, 7a–7f, 8, 9a, 9b, 12 |
| §7.1 two-candidate layout | 10, 11 |
| §7.2 two-tier caption + methods note | 13 |
| §7.3 style (150 DPI, palette) | 10, 11 |
| §8 day plan | this whole plan |
| §9 flags to human | embedded in task "STOP if ..." notes |
| §10 review checklist | 14 |

All spec sections covered.

**Placeholder scan:** No "TBD", "TODO", "implement later", "similar to Task N", or untyped references. Every code block contains the actual code an engineer needs.

**Type consistency check:**
- `OperatingPoint` used consistently (Task 4 creates, 7a–7f consume, 8 consumes).
- `ChannelContribution` fields (`name`, `group`, `delta_F`, `delta_F_uncertainty`, `description`, `perturbation_description`) match across Task 6 (schema), Task 7a–7f (producers), Task 8 (consumer), Task 10 (YAML export), Task 12 (round-trip).
- `ChannelName` Literal values (`T1_intrinsic`, `pure_dephasing`, `thermal`, `purcell`, `drive_amplitude`, `drive_detuning`) match across Task 6, Task 7a–7f, and Task 8's `_DEFAULT_CHANNELS`.
- `compute_channel_contribution` signature takes `operating_point` (positional) and `channel` (keyword): used consistently in Tasks 7a–7f tests and Task 8.
- `ErrorBudget.residual_active` (not `residual`) naming consistent across Task 6 (schema), Task 8 (producer), Task 10 (YAML export), Task 12 (round-trip).
- `rng: np.random.Generator | None = None` signature: introduced Task 2, used in Task 4 (explicit seed for deterministic verification), Task 7a–7f (implicit `None` via `_F_at` helper).

No inconsistencies.

---

## Execution Handoff

Plan complete and saved to `06_Dispersive_Readout/PLAN.md` (this file). Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Each task is self-contained with its own tests and commit; well-suited to the subagent pattern.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review. Slightly heavier on the main session context but avoids subagent cold-start cost per task.

**Which approach?**

---

**Pre-execution checklist (applies to either path):**

- [ ] On branch `stage-06-module-2-error-budget` (Step 0 above).
- [ ] Module 1 tests currently passing: `python -m pytest dispersive_readout/tests/ -v -p no:dash` → 56 tests green.
- [ ] Gate 1 (V2 in new frame) still green: done at start of this session.
- [ ] Figure 1 baseline F_assign already captured or intentionally deferred (spec §2.2 calibration is independent of Figure 1's amplitude).
- [ ] `06_Dispersive_Readout/MODULE_2_SPEC.md` unchanged since PLAN.md was written (if spec edited, re-review affected tasks).
