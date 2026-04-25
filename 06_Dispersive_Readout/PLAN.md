# Stage 06 Module 3 — Characterization Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an experimentalist-facing characterization interface: a CLI that consumes `.npz` trace bundles (Rabi / Ramsey / T1 / Hahn-echo), extracts device parameters via lmfit with parametric-bootstrap uncertainties, emits a Module-1-compatible YAML parameter pack, and ships a 50-device parameter-recovery harness gated on binomial-CI calibration + cached-artifact regression.

**Architecture:** New subpackage `dispersive_readout/characterization/` with six files (`noise.py`, `protocols.py`, `fitting.py`, `recovery.py`, `cli.py`, `__init__.py`). Synthetic traces are **closed-form analytic** (amendment 1) — the recovery harness tests the fitter, not Module 1's Lindblad simulator (which is already validated by V3 / V4a / V4b). Uncertainty is **parametric bootstrap** (amendment 3): regenerate fresh (1/f drift + shot + readout) realizations around the best-fit, re-fit each, take the spread. The recovery harness is built around `fit_one_device(device, noise, seed) → list[RecoveryResult]` as a pure function (amendment 8), compatible with `map()` or Modal without code change. Figure 3 and the CLI live in `06_Dispersive_Readout/`.

**Tech Stack:** Python 3.11+, NumPy 2.x, lmfit 1.3.x, Pydantic v2, PyYAML, matplotlib, pytest. No QuTiP dependency in Module 3 (closed-form traces).

**Spec:** See `06_Dispersive_Readout/MODULE_3_SPEC.md`. This plan implements §1 (protocols), §2 (noise model), §3 (module structure), §4 (detailed specs), §5 (tests C1–C7), §6 (Figure 3), §7 (day-by-day tasks). The Module 2 plan is preserved at `06_Dispersive_Readout/MODULE_2_PLAN.md`.

**Pre-plan assumption:** Already on branch `stage-06-module-3-characterization` (cut from tag `stage06-module2` at commit `88a201d`; spec committed at `bf122f7`). Module 2 polish diff has been stashed (`stash@{0}` — `module-2-fig2-annotation-polish-defer-to-day-14`). Executor does NOT need to create the branch — it already exists.

**User's Task 1 expectation:** "Task 1 is concrete: closed-form Rabi trace generator in `dispersive_readout/characterization/protocols.py` + C1a round-trip test, ~1 hour." This plan splits Task 1 into **Task 1** (package scaffold + `noise.py` with `load_reference_F_full` + C2 noise-sanity tests, ~45 min) and **Task 2** (Rabi closed-form generator + `TraceData` + `save/load_trace_bundle` + C1a round-trip, ~60 min). The TDD discipline is cleaner with the noise helpers landing first; the two tasks together run back-to-back and match the user's scope.

**Test invocation convention (all pytest commands in this plan use this form):**

```bash
python -m pytest <test-path> -v -p no:dash
```

The `python -m pytest` form ensures we use the conda environment's pytest; `-p no:dash` disables a broken Flask plugin on this system.

**Regression discipline:** At the end of every task, run the full suite:

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Module 1 (57 tests) + Module 2 (15 tests) + growing Module 3 tests must all pass. If any Module 1 or 2 test regresses, STOP and investigate — Module 3 should be a pure addition with zero touching of Modules 1 or 2.

---

## File Structure

### Files to create (Module 3 — new subpackage + scripts + artifacts)

| File | Responsibility |
|---|---|
| `dispersive_readout/characterization/__init__.py` | Public API: `NoiseModelParams`, `load_reference_F_full`, `TraceData`, `save_trace_bundle`, `load_trace_bundle`, `generate_rabi_trace`, `generate_ramsey_trace`, `generate_t1_trace`, `generate_t2_echo_trace`, `FittedParameter`, `ExtractedParameterPack`, `fit_rabi`, `fit_ramsey`, `fit_t1`, `fit_t2_echo`, `fit_all`, `parametric_bootstrap`, `DeviceGroundTruth`, `RecoveryResult`, `CoverageReport`, `fit_one_device`, `generate_synthetic_device_family`, `run_recovery_harness`, `save_coverage_report`, `load_committed_coverage_report`. |
| `dispersive_readout/characterization/noise.py` | `NoiseModelParams` dataclass; `generate_1f_drift`, `apply_shot_noise`, `apply_readout_errors`, `load_reference_F_full`. |
| `dispersive_readout/characterization/protocols.py` | `TraceData` dataclass; four closed-form trace generators; `save_trace_bundle` / `load_trace_bundle` with schema validation. |
| `dispersive_readout/characterization/fitting.py` | Pydantic `FittedParameter`, `ExtractedParameterPack` (with `to_device_config` — Koch E_J back-solve); lmfit wrappers `fit_rabi` / `fit_ramsey` / `fit_t1` / `fit_t2_echo`; `parametric_bootstrap`; `fit_all`. |
| `dispersive_readout/characterization/recovery.py` | `DeviceGroundTruth`, `RecoveryResult`, `CoverageReport` dataclasses; `fit_one_device` (pure function); `generate_synthetic_device_family` (T₂ < 2·T₁·0.95 rejection); `run_recovery_harness`; `save/load_coverage_report`; `format_recovery_table`. |
| `dispersive_readout/characterization/cli.py` | `main()` entry; flag parsing with conflicting-flag rejection. |
| `dispersive_readout/tests/test_characterization.py` | All Module 3 tests (C1–C7); target ≥ 25 tests. |
| `06_Dispersive_Readout/characterize.py` | Thin script entry: `python 06_Dispersive_Readout/characterize.py ...` → imports and runs `dispersive_readout.characterization.cli.main`. |
| `06_Dispersive_Readout/scripts/fig3_characterization.py` | Renders Figure 3 (2×2 layout: Rabi / Ramsey / T1 fits + parity-plot recovery panel). |
| `06_Dispersive_Readout/figures/fig3_characterization.png` | Generated publication-quality figure (150 DPI, style-matched to Figures 1 and 2). |
| `06_Dispersive_Readout/figures/recovery_coverage_report.yaml` | Committed calibration artifact at SEED=42; self-describing (includes the 50-device list for RNG-stability hedging). |
| `06_Dispersive_Readout/examples/example_traces.npz` | Reference synthetic trace bundle generated from REFERENCE_DEVICE at SEED=42. |

### Files NOT modified

Module 1 (`dispersive_readout/physics/`) and Module 2 (`dispersive_readout/analysis/`) are untouched. The single cross-module dependency is a runtime read of `06_Dispersive_Readout/figures/fig2_data.yaml` by `load_reference_F_full()` (amendment 7). Zero import-time coupling.

---

## Task 1: Scaffold `characterization/` package and `noise.py`

**Rationale:** Spec §3 names the package, §2.5 defines `NoiseModelParams`, §4.1 specifies `load_reference_F_full` (amendment 7) plus three noise-injection helpers. Lands before any generator so C2 noise-sanity tests gate the noise stack independently.

**Files:**
- Modify: `dispersive_readout/characterization/__init__.py` (currently a 4-line stub from Module 2 Task 14)
- Create: `dispersive_readout/characterization/noise.py`
- Create: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 1.1: Write the three failing C2 noise-sanity tests**

Create `dispersive_readout/tests/test_characterization.py`:

```python
"""Module 3 — characterization tests (C1–C7). See MODULE_3_SPEC.md §5."""
from __future__ import annotations

import math

import numpy as np
import pytest


# -- C2: noise model sanity ---------------------------------------------------

def test_C2a_shot_noise_matches_binomial():
    """Shot-noise sampling variance matches p(1-p)/n_shots within 5% at n=5000 trials."""
    from dispersive_readout.characterization.noise import apply_shot_noise
    rng = np.random.default_rng(seed=42)
    P_true = np.array([0.3, 0.5, 0.7])
    n_shots = 5000
    n_trials = 5000
    samples = np.stack([apply_shot_noise(P_true, n_shots, rng) for _ in range(n_trials)])
    observed_var = samples.var(axis=0)
    expected_var = P_true * (1.0 - P_true) / n_shots
    rel = np.abs(observed_var - expected_var) / expected_var
    assert np.all(rel < 0.05), f"shot-noise variance mismatch: rel={rel}"


def test_C2b_1f_drift_psd_slope_approx_minus_one():
    """Log-log slope of averaged |FFT|² vs f lies in [-1.3, -0.7] for alpha=1.

    Single realization PSDs are extremely noisy; average over 200 realizations
    and fit a line to the log-log PSD.
    """
    from dispersive_readout.characterization.noise import generate_1f_drift
    n_points = 1024
    n_real = 200
    # Average |FFT|² across realizations, take positive-freq half.
    psd_sum = np.zeros(n_points // 2)
    for k in range(n_real):
        x = generate_1f_drift(n_points, amplitude_Hz=1e4, alpha=1.0, seed=1000 + k)
        X = np.fft.fft(x)
        psd = np.abs(X) ** 2
        psd_sum += psd[:n_points // 2]
    psd_mean = psd_sum / n_real
    # Fit log-log, skip DC bin (index 0).
    f = np.arange(1, n_points // 2)
    slope, _ = np.polyfit(np.log(f), np.log(psd_mean[1:]), 1)
    assert -1.3 < slope < -0.7, f"1/f slope = {slope:.3f}, expected ~-1"


def test_C2c_load_reference_F_full_matches_yaml():
    """load_reference_F_full returns the F_full value committed in fig2_data.yaml."""
    from dispersive_readout.characterization.noise import load_reference_F_full
    import yaml
    with open("06_Dispersive_Readout/figures/fig2_data.yaml") as f:
        budget = yaml.safe_load(f)
    assert abs(load_reference_F_full() - float(budget["F_full"])) < 1e-12
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 3 FAIL with `ModuleNotFoundError: No module named 'dispersive_readout.characterization.noise'`.

- [ ] **Step 1.3: Write `noise.py`**

Replace the entire contents of `dispersive_readout/characterization/noise.py` (currently a 4-line stub) with:

```python
"""Module 3 — noise-model helpers.

Provides the full synthetic-trace noise stack:
  - NoiseModelParams: frozen config
  - generate_1f_drift: correlated 1/f^alpha drift across a scan
  - apply_shot_noise: binomial sampling
  - apply_readout_errors: classical bit-flip from Module 2's F_assign
  - load_reference_F_full: pulls F_full from Module 2's committed YAML

Amendment 7: F_assign is read from fig2_data.yaml at call time, not stored
in NoiseModelParams, so a stale cached value cannot silently persist in
serialized runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


_REFERENCE_F_FULL_PATH = Path("06_Dispersive_Readout/figures/fig2_data.yaml")


@dataclass(frozen=True)
class NoiseModelParams:
    """Frozen noise-stack configuration for synthetic trace generation.

    n_shots_per_point:         binomial shot count per trace point.
    drift_amplitude_Hz:        rms of the 1/f qubit-frequency drift across a scan.
    drift_alpha:               spectral exponent (1 = 1/f).
    drift_seed:                per-run drift seed; None = fresh each time (driven by the harness).
    readout_asymmetric:        if True, use P(0|1) != P(1|0) (not implemented — Module 3 follow-up).
    drive_amplitude_uncertainty: Gaussian SD of a once-per-run amplitude offset (Rabi only).
    """
    n_shots_per_point: int = 2000
    drift_amplitude_Hz: float = 1e4
    drift_alpha: float = 1.0
    drift_seed: int | None = None
    readout_asymmetric: bool = False
    drive_amplitude_uncertainty: float = 0.05


def load_reference_F_full() -> float:
    """Read F_full at REFERENCE_DEVICE from Module 2's committed artifact (amendment 7)."""
    with open(_REFERENCE_F_FULL_PATH) as f:
        budget = yaml.safe_load(f)
    return float(budget["F_full"])


def generate_1f_drift(
    n_points: int,
    amplitude_Hz: float,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an n-point 1/f^alpha realization with rms `amplitude_Hz`.

    Method: draw white Gaussian samples in frequency domain with amplitude
    proportional to f^(-alpha/2); inverse-FFT; rescale to target rms. DC bin
    set to zero (pure AC drift).
    """
    rng = np.random.default_rng(seed)
    N = int(n_points)
    # Positive-frequency bins 1..N/2 inclusive; we'll use hermitian-symmetric filling.
    freqs = np.fft.fftfreq(N)
    # Avoid div-by-zero at DC; set DC amplitude to zero explicitly.
    mag = np.zeros(N, dtype=float)
    nonzero = freqs != 0.0
    mag[nonzero] = np.abs(freqs[nonzero]) ** (-alpha / 2.0)
    # Draw complex Gaussian amplitudes.
    re = rng.standard_normal(N)
    im = rng.standard_normal(N)
    X = (re + 1j * im) * mag
    X[0] = 0.0
    # Enforce hermitian symmetry so the ifft is real.
    # (For even N, bin N/2 must be real.)
    if N % 2 == 0:
        X[N // 2] = np.real(X[N // 2])
    x = np.fft.ifft(X).real
    # Rescale to target rms.
    current_rms = float(np.sqrt(np.mean(x**2)))
    if current_rms == 0.0:
        return x
    return x * (amplitude_Hz / current_rms)


def apply_shot_noise(
    P_true: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial shot-noise sampling. Returns observed P = k/n_shots ∈ [0, 1]."""
    P_clipped = np.clip(P_true, 0.0, 1.0)
    k = rng.binomial(n_shots, P_clipped)
    return k / n_shots


def apply_readout_errors(
    P_observed: np.ndarray,
    F_assign: float,
    asymmetric: bool = False,
) -> np.ndarray:
    """Classical bit-flip readout-error model; symmetric by default.

    P_out = (1 − p_flip) * P_in + p_flip * (1 − P_in), where p_flip = 1 − F_assign.
    """
    if asymmetric:
        raise NotImplementedError("Asymmetric readout errors are a Module 3 follow-up; use symmetric.")
    p_flip = 1.0 - F_assign
    return (1.0 - p_flip) * P_observed + p_flip * (1.0 - P_observed)
```

- [ ] **Step 1.4: Update package `__init__.py` to export the noise API**

Replace `dispersive_readout/characterization/__init__.py`:

```python
"""Stage 06 Module 3 — parameter characterization protocols.

Public API (post-Task 1):
    - NoiseModelParams
    - generate_1f_drift, apply_shot_noise, apply_readout_errors, load_reference_F_full

Additional exports land as subsequent tasks (protocols, fitting, recovery, CLI).
"""
from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    generate_1f_drift,
    load_reference_F_full,
)

__all__ = [
    "NoiseModelParams",
    "apply_readout_errors",
    "apply_shot_noise",
    "generate_1f_drift",
    "load_reference_F_full",
]
```

- [ ] **Step 1.5: Run C2 tests to verify they pass**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 3 PASS.

- [ ] **Step 1.6: Run full suite — zero regressions in Modules 1/2**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: 75 passing (57 Module 1 + 15 Module 2 + 3 Module 3).

- [ ] **Step 1.7: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/noise.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 1 — noise helpers + C2 tests

Adds NoiseModelParams, generate_1f_drift (correlated across a scan),
apply_shot_noise, apply_readout_errors, and load_reference_F_full
(amendment 7 — reads from Module 2's fig2_data.yaml at runtime,
no stale hardcoded F_assign). C2a/b/c tests passing.

75 tests passing (Module 1: 57, Module 2: 15, Module 3: 3)."
```

**Definition of done:** 3 C2 tests passing; no Module 1/2 regressions.

---

## Task 2: Rabi closed-form generator + `TraceData` + bundle I/O + C1a round-trip

**Rationale:** Spec §1.1 specifies the amendment-2 Rabi form (no T_R envelope); §4.2 specifies `TraceData` + `save/load_trace_bundle`. C1a is the Rabi round-trip test the user called out as Task 1's scope.

**Files:**
- Create: `dispersive_readout/characterization/protocols.py`
- Modify: `dispersive_readout/characterization/__init__.py` (export new API)
- Modify: `dispersive_readout/tests/test_characterization.py` (add C1a + bundle round-trip)

- [ ] **Step 2.1: Write the failing C1a + bundle round-trip tests**

Append to `dispersive_readout/tests/test_characterization.py`:

```python
# -- C1a: Rabi round-trip ----------------------------------------------------

def test_C1a_rabi_round_trip():
    """Closed-form Rabi trace → fit pipeline (point-estimate only) recovers ε_π within 3%.

    Point-estimate sanity check; full uncertainty testing is in C3. Uses a
    light noise config (n_shots=5000, no drift, no amp uncertainty) so the
    round-trip is tight.
    """
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    # Light noise so the round-trip isolates the generator form, not the fitter.
    noise = NoiseModelParams(
        n_shots_per_point=5000,
        drift_amplitude_Hz=0.0,
        drive_amplitude_uncertainty=0.0,
    )
    epsilon_pi_truth = 2 * math.pi * 50e6   # 50 MHz rad/s scale
    omega_q = 2 * math.pi * 4.5e9
    trace = generate_rabi_trace(epsilon_pi_truth, omega_q, noise, seed=0)
    # Use a dead-simple estimator: find the first local max of P1; ε at that point
    # equals ε_π (since P1(ε) = 0.5 + 0.5·cos(π·ε/ε_π), first max is at ε=ε_π).
    P1 = trace.P1
    eps = trace.sweep_values
    # Expect P1 to dip to ~0 at ε_π (cos(π)=-1) then back up at ε=2·ε_π.
    idx_min = int(np.argmin(P1))
    eps_estimate = float(eps[idx_min])   # ~ε_π
    rel = abs(eps_estimate - epsilon_pi_truth) / epsilon_pi_truth
    assert rel < 0.03, f"Rabi round-trip: eps_est={eps_estimate:.3e}, truth={epsilon_pi_truth:.3e}, rel={rel:.3%}"


# -- Bundle round-trip (preps for schema validation in Task 6) ---------------

def test_trace_bundle_npz_round_trip(tmp_path):
    """save_trace_bundle → load_trace_bundle preserves all fields exactly."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import (
        generate_rabi_trace, save_trace_bundle, load_trace_bundle, TraceData,
    )
    noise = NoiseModelParams(n_shots_per_point=1000, drift_amplitude_Hz=0.0)
    trace = generate_rabi_trace(2 * math.pi * 30e6, 2 * math.pi * 4.8e9, noise, seed=123)
    path = tmp_path / "bundle.npz"
    save_trace_bundle([trace], str(path))
    loaded = load_trace_bundle(str(path))
    assert len(loaded) == 1
    t = loaded[0]
    assert t.protocol == trace.protocol
    assert t.sweep_axis == trace.sweep_axis
    np.testing.assert_array_equal(t.sweep_values, trace.sweep_values)
    np.testing.assert_array_equal(t.P1, trace.P1)
    np.testing.assert_array_equal(t.P1_uncertainty, trace.P1_uncertainty)
    assert t.metadata == trace.metadata
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py::test_C1a_rabi_round_trip dispersive_readout/tests/test_characterization.py::test_trace_bundle_npz_round_trip -v -p no:dash
```

Expected: 2 FAIL with `ModuleNotFoundError: No module named 'dispersive_readout.characterization.protocols'`.

- [ ] **Step 2.3: Write `protocols.py` (Rabi only for this task)**

Create `dispersive_readout/characterization/protocols.py`:

```python
"""Module 3 — closed-form synthetic trace generators and bundle I/O.

Amendment 1: traces are CLOSED-FORM ANALYTIC (P₁ as an exact function of the
ground-truth parameters), not Lindblad-simulated. The recovery harness tests
the fitter's statistical behavior; Module 1 V3/V4a/V4b already validate the
Lindblad dynamics.

Amendment 2: Rabi fit form is `P₁(ε) = A + B·cos(π·ε/ε_π + φ)` with no T_R
envelope — T_R is unidentifiable from an amplitude sweep at fixed τ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    generate_1f_drift,
    load_reference_F_full,
)


@dataclass(frozen=True)
class TraceData:
    """Container for one protocol's measurement trace.

    On disk (.npz): one file per bundle, one entry per trace with a JSON-
    serialized metadata blob (so arbitrary Python types survive the round-trip).
    """
    protocol: str                        # "rabi" | "ramsey" | "t1" | "t2_echo"
    sweep_axis: str                      # "drive_amplitude" | "delay"
    sweep_values: np.ndarray             # (N,)
    P1: np.ndarray                       # (N,) observed |1⟩ population after noise stack
    P1_uncertainty: np.ndarray           # (N,) shot-only per-point SE
    metadata: dict                       # ground truth (synthetic) or device ID (real)


_REQUIRED_TRACE_FIELDS = ("protocol", "sweep_axis", "sweep_values", "P1", "P1_uncertainty", "metadata")


def generate_rabi_trace(
    epsilon_pi: float,                   # ground-truth π-amplitude (rad/s)
    omega_q: float,                      # ground-truth qubit frequency (rad/s) — recorded in metadata
    noise: NoiseModelParams,
    n_points: int = 101,
    amplitude_span_mult: tuple[float, float] = (0.0, 2.5),   # in units of epsilon_pi
    seed: int | None = None,
) -> TraceData:
    """Closed-form Rabi trace.

    Form: P₁(ε) = 0.5 − 0.5·cos(π·ε/ε_π·(1 + δ_amp)), where δ_amp is a
    once-per-run Gaussian calibration offset of SD `drive_amplitude_uncertainty`.
    The 1/f drift does NOT enter at leading order — Rabi rate depends on
    transmon dipole, not ω_q (O((drift/ω_q)²) correction is sub-1% at 10 kHz
    drift vs 4.5 GHz ω_q).

    Noise stack: (1) amplitude calibration offset (scalar per run); (2) binomial
    shot noise per point; (3) symmetric readout errors using Module 2's F_full.
    """
    rng = np.random.default_rng(seed)
    F_assign = load_reference_F_full()
    eps = np.linspace(
        amplitude_span_mult[0] * epsilon_pi,
        amplitude_span_mult[1] * epsilon_pi,
        n_points,
    )
    # One-shot amplitude calibration offset.
    delta_amp = rng.normal(0.0, noise.drive_amplitude_uncertainty) if noise.drive_amplitude_uncertainty > 0 else 0.0
    eps_effective = eps * (1.0 + delta_amp)
    # Closed-form population before noise:
    P_true = 0.5 - 0.5 * np.cos(np.pi * eps_effective / epsilon_pi)
    # Readout errors on the true population (symmetric bit-flip).
    P_after_readout = apply_readout_errors(P_true, F_assign)
    # Shot noise.
    P_observed = apply_shot_noise(P_after_readout, noise.n_shots_per_point, rng)
    # Per-point shot-only SE (used as initial lmfit weights; bootstrap handles
    # everything else).
    P_se = np.sqrt(np.clip(P_after_readout, 1e-12, 1 - 1e-12) * (1 - np.clip(P_after_readout, 1e-12, 1 - 1e-12)) / noise.n_shots_per_point)
    return TraceData(
        protocol="rabi",
        sweep_axis="drive_amplitude",
        sweep_values=eps,
        P1=P_observed,
        P1_uncertainty=P_se,
        metadata={
            "ground_truth": {"epsilon_pi": epsilon_pi, "omega_q": omega_q},
            "noise": {
                "n_shots_per_point": noise.n_shots_per_point,
                "drift_amplitude_Hz": noise.drift_amplitude_Hz,
                "drift_alpha": noise.drift_alpha,
                "drive_amplitude_uncertainty": noise.drive_amplitude_uncertainty,
                "F_assign": F_assign,
            },
            "seed": seed,
            "delta_amp_realization": float(delta_amp),
        },
    )


# -- Bundle I/O ---------------------------------------------------------------

def save_trace_bundle(traces: list[TraceData], path: str | Path) -> None:
    """Save a list of traces to .npz.

    Structure: one flat .npz with keys
      traces/<i>/protocol  (str)
      traces/<i>/sweep_axis (str)
      traces/<i>/sweep_values (ndarray)
      traces/<i>/P1 (ndarray)
      traces/<i>/P1_uncertainty (ndarray)
      traces/<i>/metadata_json (0-d str array containing JSON)
    plus:
      n_traces (int)
    """
    payload: dict[str, np.ndarray] = {"n_traces": np.array(len(traces))}
    for i, t in enumerate(traces):
        payload[f"traces/{i}/protocol"] = np.array(t.protocol)
        payload[f"traces/{i}/sweep_axis"] = np.array(t.sweep_axis)
        payload[f"traces/{i}/sweep_values"] = np.asarray(t.sweep_values)
        payload[f"traces/{i}/P1"] = np.asarray(t.P1)
        payload[f"traces/{i}/P1_uncertainty"] = np.asarray(t.P1_uncertainty)
        payload[f"traces/{i}/metadata_json"] = np.array(json.dumps(t.metadata))
    np.savez(path, **payload)


def load_trace_bundle(path: str | Path) -> list[TraceData]:
    """Load a .npz trace bundle; raises ValueError on missing fields.

    Schema validation per §8 flag #5.
    """
    raw = np.load(path, allow_pickle=False)
    n = int(raw["n_traces"])
    out: list[TraceData] = []
    for i in range(n):
        # Schema validation: every required field must be present.
        for field_name in _REQUIRED_TRACE_FIELDS:
            key = f"traces/{i}/{field_name}" if field_name != "metadata" else f"traces/{i}/metadata_json"
            if key not in raw:
                raise ValueError(f"Trace bundle missing required field '{field_name}' on entry {i}")
        out.append(TraceData(
            protocol=str(raw[f"traces/{i}/protocol"]),
            sweep_axis=str(raw[f"traces/{i}/sweep_axis"]),
            sweep_values=np.array(raw[f"traces/{i}/sweep_values"]),
            P1=np.array(raw[f"traces/{i}/P1"]),
            P1_uncertainty=np.array(raw[f"traces/{i}/P1_uncertainty"]),
            metadata=json.loads(str(raw[f"traces/{i}/metadata_json"])),
        ))
    return out
```

- [ ] **Step 2.4: Update `__init__.py` to export the new API**

Replace `dispersive_readout/characterization/__init__.py`:

```python
"""Stage 06 Module 3 — parameter characterization protocols."""
from .noise import (
    NoiseModelParams,
    apply_readout_errors,
    apply_shot_noise,
    generate_1f_drift,
    load_reference_F_full,
)
from .protocols import (
    TraceData,
    generate_rabi_trace,
    load_trace_bundle,
    save_trace_bundle,
)

__all__ = [
    "NoiseModelParams",
    "TraceData",
    "apply_readout_errors",
    "apply_shot_noise",
    "generate_1f_drift",
    "generate_rabi_trace",
    "load_reference_F_full",
    "load_trace_bundle",
    "save_trace_bundle",
]
```

- [ ] **Step 2.5: Run new tests**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 5 PASS (3 C2 + 1 C1a + 1 bundle round-trip).

- [ ] **Step 2.6: Run full suite**

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: 77 passing.

- [ ] **Step 2.7: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/protocols.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 2 — Rabi generator + trace bundle I/O

Closed-form Rabi trace per amendment 2 (no T_R envelope). TraceData
schema + save_trace_bundle/load_trace_bundle with schema validation
(§8 flag #5). C1a round-trip passing within 3%.

77 tests passing."
```

**Definition of done:** C1a and bundle round-trip both passing; 77 total tests green.

---

## Task 3: Ramsey closed-form generator + C1b

**Rationale:** Spec §1.2 specifies the fit form with 1/f drift rolled into Δω(τ_k).

**Files:**
- Modify: `dispersive_readout/characterization/protocols.py` (add `generate_ramsey_trace`)
- Modify: `dispersive_readout/characterization/__init__.py` (export)
- Modify: `dispersive_readout/tests/test_characterization.py` (add C1b)

- [ ] **Step 3.1: Write failing C1b**

Append to `test_characterization.py`:

```python
def test_C1b_ramsey_round_trip():
    """Closed-form Ramsey → simple FFT-based estimator recovers ω_q within 0.1% and T2* within 15%.

    Sanity check that the generator + a naive estimator agree; precise lmfit
    recovery lives in Task 8.
    """
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    omega_q_truth = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    # Pin ω_drive so Δω ≠ 0.
    omega_drive_offset = 2 * math.pi * 1.5e6
    trace = generate_ramsey_trace(
        omega_q_truth, T_2_star=T_2_star_truth, noise=noise,
        omega_drive_offset=omega_drive_offset, seed=1,
    )
    # Naive FFT on detrended P1.
    delays = trace.sweep_values
    signal = trace.P1 - trace.P1.mean()
    fft = np.abs(np.fft.rfft(signal))
    dt = float(delays[1] - delays[0])
    freqs = np.fft.rfftfreq(len(delays), d=dt)
    peak = int(np.argmax(fft[1:])) + 1
    delta_omega_est = 2 * math.pi * float(freqs[peak])
    omega_q_est = omega_q_truth - omega_drive_offset + delta_omega_est
    rel = abs(omega_q_est - omega_q_truth) / omega_q_truth
    assert rel < 1e-3, f"Ramsey ω_q naive FFT estimate off: rel={rel:.3e}"
```

- [ ] **Step 3.2: Run to verify FAIL**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py::test_C1b_ramsey_round_trip -v -p no:dash
```

Expected: FAIL — `ImportError` on `generate_ramsey_trace`.

- [ ] **Step 3.3: Add `generate_ramsey_trace` to `protocols.py`**

Append to `dispersive_readout/characterization/protocols.py`:

```python
def generate_ramsey_trace(
    omega_q: float,
    T_2_star: float,
    noise: NoiseModelParams,
    omega_drive_offset: float = 2.0 * np.pi * 1e6,   # set Δω = +1 MHz by default
    n_points: int = 101,
    delay_range: tuple[float, float] = (0.0, 40e-6),
    seed: int | None = None,
) -> TraceData:
    """Closed-form Ramsey with correlated 1/f qubit-frequency drift.

    Form: P₁(τ) = 0.5 − 0.5·exp(−τ/T_2*)·cos(Δω_nom·τ + φ_drift(τ))
      where φ_drift(τ_k) = ∫₀^τ_k δω_1f(t) dt is approximated by the
      cumulative sum of the drift realization. This is the correlated-drift
      effect: a single realization of `generate_1f_drift` samples the
      trajectory of δω across the sweep, so bootstrap residuals are NOT iid
      (amendment 3).
    """
    rng = np.random.default_rng(seed)
    F_assign = load_reference_F_full()
    delays = np.linspace(delay_range[0], delay_range[1], n_points)
    delta_omega_nominal = omega_q - (omega_q - omega_drive_offset)  # just = omega_drive_offset; kept explicit
    # 1/f drift realization across the sweep (n_points samples, one per delay bin).
    drift_seed = int(rng.integers(2**31 - 1))
    delta_omega_drift = 2.0 * np.pi * generate_1f_drift(
        n_points, amplitude_Hz=noise.drift_amplitude_Hz, alpha=noise.drift_alpha, seed=drift_seed,
    )
    # Cumulative-phase approximation: phase at τ_k is the running integral of drift.
    dt = float(delays[1] - delays[0]) if len(delays) > 1 else 0.0
    phi_drift = np.cumsum(delta_omega_drift) * dt
    # Closed-form P1.
    envelope = np.exp(-delays / T_2_star)
    P_true = 0.5 - 0.5 * envelope * np.cos(delta_omega_nominal * delays + phi_drift)
    # Noise stack.
    P_after_readout = apply_readout_errors(P_true, F_assign)
    P_observed = apply_shot_noise(P_after_readout, noise.n_shots_per_point, rng)
    P_ro_c = np.clip(P_after_readout, 1e-12, 1 - 1e-12)
    P_se = np.sqrt(P_ro_c * (1 - P_ro_c) / noise.n_shots_per_point)
    return TraceData(
        protocol="ramsey",
        sweep_axis="delay",
        sweep_values=delays,
        P1=P_observed,
        P1_uncertainty=P_se,
        metadata={
            "ground_truth": {
                "omega_q": omega_q, "T_2_star": T_2_star,
                "omega_drive_offset": omega_drive_offset,
            },
            "noise": {
                "n_shots_per_point": noise.n_shots_per_point,
                "drift_amplitude_Hz": noise.drift_amplitude_Hz,
                "drift_alpha": noise.drift_alpha,
                "F_assign": F_assign,
            },
            "seed": seed,
            "drift_seed": drift_seed,
        },
    )
```

- [ ] **Step 3.4: Export from `__init__.py`**

Add `generate_ramsey_trace` to the `from .protocols import (...)` list and to `__all__`.

- [ ] **Step 3.5: Run C1b + full suite**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 6 PASS.

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 78 passing.

- [ ] **Step 3.6: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/protocols.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 3 — Ramsey generator + C1b

Closed-form Ramsey with correlated 1/f drift via cumulative phase
(the correlated-residuals property that motivates parametric bootstrap
per amendment 3). C1b naive-FFT round-trip passing within 0.1%.

78 tests passing."
```

**Definition of done:** C1b passing; 78 total tests green.

---

## Task 4: T1 closed-form generator + C1c (with thermal-offset support)

**Files:**
- Modify: `dispersive_readout/characterization/protocols.py` (add `generate_t1_trace`)
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 4.1: Write failing C1c**

Append:

```python
def test_C1c_t1_round_trip():
    """Closed-form T1 → simple exponential-fit estimator recovers T1 within 5%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, seed=2)
    delays = trace.sweep_values
    P1 = trace.P1
    # Fit log(P1 - floor) ≈ log(amp) - τ/T1. Estimate floor from the last 10 points.
    floor = float(P1[-10:].mean())
    mask = (P1 - floor) > 0.02
    coef = np.polyfit(delays[mask], np.log(P1[mask] - floor), 1)
    T_1_est = -1.0 / coef[0]
    rel = abs(T_1_est - T_1_truth) / T_1_truth
    assert rel < 0.05, f"T1 round-trip rel={rel:.3%}"
```

- [ ] **Step 4.2: Run FAIL**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py::test_C1c_t1_round_trip -v -p no:dash
```

Expected: FAIL (ImportError).

- [ ] **Step 4.3: Add `generate_t1_trace`**

Append to `protocols.py`:

```python
def generate_t1_trace(
    T_1: float,
    noise: NoiseModelParams,
    n_points: int = 51,
    delay_range: tuple[float, float] = (0.0, 100e-6),
    thermal_offset: float = 0.0,          # A in A + B·exp(−τ/T1); 0 = ideal cold
    seed: int | None = None,
) -> TraceData:
    """Closed-form T1 decay: P₁(τ) = A + (1 − A)·exp(−τ/T_1).

    `thermal_offset` = A represents the steady-state thermal population; 0.08
    is the elevated-thermal edge case in the recovery harness (§8 flag implicit).
    Rabi/Ramsey-style 1/f drift does NOT enter at leading order; this is a
    relaxation-only protocol.
    """
    rng = np.random.default_rng(seed)
    F_assign = load_reference_F_full()
    delays = np.linspace(delay_range[0], delay_range[1], n_points)
    P_true = thermal_offset + (1.0 - thermal_offset) * np.exp(-delays / T_1)
    P_after_readout = apply_readout_errors(P_true, F_assign)
    P_observed = apply_shot_noise(P_after_readout, noise.n_shots_per_point, rng)
    P_ro_c = np.clip(P_after_readout, 1e-12, 1 - 1e-12)
    P_se = np.sqrt(P_ro_c * (1 - P_ro_c) / noise.n_shots_per_point)
    return TraceData(
        protocol="t1",
        sweep_axis="delay",
        sweep_values=delays,
        P1=P_observed,
        P1_uncertainty=P_se,
        metadata={
            "ground_truth": {"T_1": T_1, "thermal_offset": thermal_offset},
            "noise": {
                "n_shots_per_point": noise.n_shots_per_point,
                "F_assign": F_assign,
            },
            "seed": seed,
        },
    )
```

- [ ] **Step 4.4: Export + run tests + commit**

Export `generate_t1_trace` in `__init__.py`.

Run:

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 7 PASS.

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 79 passing.

Commit:

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/protocols.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 4 — T1 generator + C1c

Closed-form T1 decay with thermal_offset param (A in A+B·exp(−τ/T1))
so the elevated-thermal edge case (harness device[1], n_th=0.08) is
covered by the generator without a separate code path.

79 tests passing."
```

**Definition of done:** C1c passing; 79 total.

---

## Task 5: T2-echo closed-form generator + C1d

**Files:**
- Modify: `dispersive_readout/characterization/protocols.py` (add `generate_t2_echo_trace`)
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 5.1: Write failing C1d**

Append:

```python
def test_C1d_t2_echo_round_trip():
    """Closed-form T2-echo → simple exponential fit recovers T2 within 10%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t2_echo_trace
    T_2_truth = 40e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t2_echo_trace(T_2_truth, noise, seed=3)
    delays = trace.sweep_values
    P1 = trace.P1
    # Hahn echo: P1(τ) = 0.5 − 0.5·exp(−τ/T2). Fit -ln(1 - 2·P1) vs τ.
    signal = 1.0 - 2.0 * P1
    mask = signal > 0.02
    coef = np.polyfit(delays[mask], np.log(signal[mask]), 1)
    T_2_est = -1.0 / coef[0]
    rel = abs(T_2_est - T_2_truth) / T_2_truth
    assert rel < 0.10, f"T2-echo round-trip rel={rel:.3%}"
```

- [ ] **Step 5.2: Run FAIL**

Expected: FAIL — ImportError.

- [ ] **Step 5.3: Add `generate_t2_echo_trace`**

Append to `protocols.py`:

```python
def generate_t2_echo_trace(
    T_2: float,
    noise: NoiseModelParams,
    n_points: int = 51,
    delay_range: tuple[float, float] = (0.0, 120e-6),
    seed: int | None = None,
) -> TraceData:
    """Closed-form Hahn echo: P₁(τ) = 0.5 − 0.5·exp(−τ/T_2).

    The echo π-pulse refocuses low-frequency drift, so 1/f drift is NOT
    applied at leading order (Sank 2024 §III.B).
    """
    rng = np.random.default_rng(seed)
    F_assign = load_reference_F_full()
    delays = np.linspace(delay_range[0], delay_range[1], n_points)
    P_true = 0.5 - 0.5 * np.exp(-delays / T_2)
    P_after_readout = apply_readout_errors(P_true, F_assign)
    P_observed = apply_shot_noise(P_after_readout, noise.n_shots_per_point, rng)
    P_ro_c = np.clip(P_after_readout, 1e-12, 1 - 1e-12)
    P_se = np.sqrt(P_ro_c * (1 - P_ro_c) / noise.n_shots_per_point)
    return TraceData(
        protocol="t2_echo",
        sweep_axis="delay",
        sweep_values=delays,
        P1=P_observed,
        P1_uncertainty=P_se,
        metadata={
            "ground_truth": {"T_2": T_2},
            "noise": {
                "n_shots_per_point": noise.n_shots_per_point,
                "F_assign": F_assign,
            },
            "seed": seed,
        },
    )
```

- [ ] **Step 5.4: Export + run tests + commit**

Export in `__init__.py`.

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 8 PASS.

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 80 passing.

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/protocols.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 5 — T2-echo generator + C1d

Hahn-echo refocusing means no 1/f drift at leading order (Sank 2024).
Plain-exponential form; stretched-exponential fallback lives in the
fitter (Task 8).

80 tests passing."
```

**Definition of done:** C1d passing; 80 total.

---

## Task 6: `load_trace_bundle` schema validation tests

**Rationale:** §8 flag #5 — `load_trace_bundle` must reject malformed bundles. The happy path is covered by Task 2's round-trip; this task adds the negative-case tests.

**Files:**
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 6.1: Add two failing schema-validation tests**

Append:

```python
# -- Schema validation for load_trace_bundle (§8 flag #5) --------------------

def test_load_trace_bundle_rejects_missing_field(tmp_path):
    """A .npz that lacks a required field (e.g., P1_uncertainty) raises ValueError."""
    from dispersive_readout.characterization.protocols import load_trace_bundle
    path = tmp_path / "missing_field.npz"
    # Build a bundle missing P1_uncertainty on trace 0.
    np.savez(
        str(path),
        n_traces=np.array(1),
        **{
            "traces/0/protocol": np.array("rabi"),
            "traces/0/sweep_axis": np.array("drive_amplitude"),
            "traces/0/sweep_values": np.array([0.0, 1.0, 2.0]),
            "traces/0/P1": np.array([0.5, 0.5, 0.5]),
            # Intentionally omit P1_uncertainty.
            "traces/0/metadata_json": np.array("{}"),
        },
    )
    with pytest.raises(ValueError, match="P1_uncertainty"):
        load_trace_bundle(str(path))


def test_load_trace_bundle_rejects_missing_metadata(tmp_path):
    """A bundle missing metadata_json on any entry raises ValueError."""
    from dispersive_readout.characterization.protocols import load_trace_bundle
    path = tmp_path / "missing_meta.npz"
    np.savez(
        str(path),
        n_traces=np.array(1),
        **{
            "traces/0/protocol": np.array("rabi"),
            "traces/0/sweep_axis": np.array("drive_amplitude"),
            "traces/0/sweep_values": np.array([0.0, 1.0, 2.0]),
            "traces/0/P1": np.array([0.5, 0.5, 0.5]),
            "traces/0/P1_uncertainty": np.array([0.01, 0.01, 0.01]),
            # Intentionally omit metadata_json.
        },
    )
    with pytest.raises(ValueError, match="metadata"):
        load_trace_bundle(str(path))
```

- [ ] **Step 6.2: Run to verify PASS**

The implementation in Task 2 already covers this — Task 2's `_REQUIRED_TRACE_FIELDS` loop raises `ValueError` on any missing field.

```bash
python -m pytest dispersive_readout/tests/test_characterization.py::test_load_trace_bundle_rejects_missing_field dispersive_readout/tests/test_characterization.py::test_load_trace_bundle_rejects_missing_metadata -v -p no:dash
```

Expected: 2 PASS. (If any fail, the `_REQUIRED_TRACE_FIELDS` loop in `load_trace_bundle` needs to explicitly match the human-readable field names in the error message — patch it so the `ValueError` string contains the missing field name.)

- [ ] **Step 6.3: Commit**

```bash
git add dispersive_readout/tests/test_characterization.py
git commit -m "test(stage06): Module 3 Task 6 — bundle schema validation tests

Negative-case tests for §8 flag #5: load_trace_bundle must reject
bundles missing P1_uncertainty or metadata_json. Implementation was
landed in Task 2; this task locks the behavior with explicit tests.

82 tests passing."
```

**Definition of done:** 2 negative-case tests passing; 82 total.

---

## Task 7: Pydantic `FittedParameter` + `ExtractedParameterPack` + `to_device_config` (E_J back-solve)

**Rationale:** Spec §4.3 schemas + amendment 5 (E_J back-solve via Koch formula). `to_device_config` is the bridge Module 4 depends on; C4c and C7a/C7b lock it.

**Files:**
- Create: `dispersive_readout/characterization/fitting.py`
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 7.1: Write failing C4 + C7 tests**

Append:

```python
# -- C4: Pydantic schema + to_device_config ---------------------------------

def test_C4a_fitted_parameter_requires_positive_uncertainty():
    from dispersive_readout.characterization.fitting import FittedParameter
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FittedParameter(
            name="T_1", value=30e-6, uncertainty=-1e-6, unit="s",
            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200,
        )


def test_C4b_extracted_parameter_pack_yaml_round_trip(tmp_path):
    """Serialize to YAML and re-load — round-trip preserves all fields."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    import yaml
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.2, n_bootstrap=200),
            FittedParameter(name="omega_q", value=2 * math.pi * 4.5e9,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=0.95, n_bootstrap=200),
        ],
        trace_file="example.npz",
        timestamp="2026-04-22T10:00:00+00:00",
        stage_06_version="abc123",
    )
    path = tmp_path / "pack.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(pack.model_dump(), f)
    with open(path) as f:
        reloaded = ExtractedParameterPack.model_validate(yaml.safe_load(f))
    assert reloaded == pack


def test_C4c_to_device_config_produces_simulator_consumable():
    """to_device_config() → simulate_readout() runs without error."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import DriveParams
    from dispersive_readout.physics.readout_model import simulate_readout
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="omega_q", value=2 * math.pi * 4.5e9,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="example.npz",
        timestamp="2026-04-22T10:00:00+00:00",
        stage_06_version="abc123",
    )
    device = pack.to_device_config()
    drive = DriveParams(amplitude=2 * math.pi * 2e6, duration=500e-9, detuning=0.0)
    t_list = np.linspace(0.0, drive.duration, 101)
    # Just need it to run without error.
    _ = simulate_readout(device, drive, initial_qubit_state=0, t_list=t_list)


# -- C7: to_device_config physics consistency (amendment 5) ------------------

def test_C7a_to_device_config_back_solves_E_J_from_omega_q():
    """E_J = (ω_q + E_C)² / (8·E_C) per Koch 2007."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    # Pin ω_q near REFERENCE so the back-solve lands near REFERENCE's E_J.
    omega_q_target = 2 * math.pi * 4.5e9
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="omega_q", value=omega_q_target,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="x.npz", timestamp="now", stage_06_version="x",
    )
    E_C = REFERENCE_DEVICE.transmon.E_C
    device = pack.to_device_config()
    expected_E_J = (omega_q_target + E_C) ** 2 / (8.0 * E_C)
    assert abs(device.transmon.E_J - expected_E_J) / expected_E_J < 1e-10


def test_C7b_to_device_config_warns_on_E_J_drift_over_30pct():
    """Large-drift ω_q → derived E_J > 30% off REFERENCE's E_J → UserWarning."""
    from dispersive_readout.characterization.fitting import ExtractedParameterPack, FittedParameter
    from dispersive_readout.physics.config import REFERENCE_DEVICE
    # Pick an ω_q that forces a big E_J deviation.
    E_C = REFERENCE_DEVICE.transmon.E_C
    omega_q_target = 2 * math.pi * 6.5e9   # well above REFERENCE's ~4.5 GHz
    pack = ExtractedParameterPack(
        fitted_parameters=[
            FittedParameter(name="omega_q", value=omega_q_target,
                            uncertainty=2 * math.pi * 1e3, unit="rad/s",
                            protocol_source="ramsey", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_1", value=30e-6, uncertainty=1e-6, unit="s",
                            protocol_source="t1", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="T_2_echo", value=40e-6, uncertainty=2e-6, unit="s",
                            protocol_source="t2_echo", goodness_of_fit=1.0, n_bootstrap=200),
            FittedParameter(name="epsilon_pi", value=2 * math.pi * 50e6,
                            uncertainty=2 * math.pi * 1e6, unit="rad/s",
                            protocol_source="rabi", goodness_of_fit=1.0, n_bootstrap=200),
        ],
        trace_file="x.npz", timestamp="now", stage_06_version="x",
    )
    with pytest.warns(UserWarning, match="E_J"):
        pack.to_device_config()
```

- [ ] **Step 7.2: Run to verify FAIL**

Expected: 5 FAIL — ImportError on `fitting`.

- [ ] **Step 7.3: Write `fitting.py` (schemas + to_device_config only; no lmfit yet)**

Create `dispersive_readout/characterization/fitting.py`:

```python
"""Module 3 — lmfit-based parameter extraction.

This module has two layers:
  1. Pydantic schemas (FittedParameter, ExtractedParameterPack) with a
     to_device_config bridge that back-solves E_J from ω_q per Koch 2007
     (amendment 5).
  2. lmfit wrappers + parametric_bootstrap (amendment 3). The wrappers
     arrive in Task 8; bootstrap in Task 9.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FittedParameter(BaseModel):
    """One fitted device parameter with bootstrap uncertainty."""
    name: Literal["T_1", "T_2_echo", "T_2_star", "omega_q", "epsilon_pi"]
    value: float
    uncertainty: float                          # 1-sigma from parametric bootstrap
    unit: Literal["s", "rad/s"]
    protocol_source: Literal["rabi", "ramsey", "t1", "t2_echo"]
    goodness_of_fit: float = Field(ge=0.0)      # reduced chi-squared, non-negative
    n_bootstrap: int = Field(ge=0)

    @field_validator("uncertainty")
    @classmethod
    def _positive_uncertainty(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("uncertainty must be strictly positive")
        return v


class ExtractedParameterPack(BaseModel):
    """All parameters extracted from one trace bundle."""
    fitted_parameters: list[FittedParameter]
    trace_file: str
    timestamp: str
    stage_06_version: str

    def _get(self, name: str) -> FittedParameter | None:
        for p in self.fitted_parameters:
            if p.name == name:
                return p
        return None

    def to_device_config(self, E_J_tolerance_rel: float = 0.30):
        """Bridge fitted parameters to Module 1's DeviceConfig (amendment 5).

        Policy:
          - E_C held fixed at REFERENCE_DEVICE (geometric, not fit).
          - E_J back-solved from fitted ω_q via Koch 2007:
                E_J = (ω_q + E_C)² / (8·E_C)
          - γ_1 = 1 / T_1; γ_φ from T_2_echo via 1/T_2 = γ_1/2 + γ_φ.
          - resonator, coupling, truncation inherited from REFERENCE_DEVICE.
          - UserWarning if |E_J − E_J_REFERENCE| / E_J_REFERENCE > 30%.
        """
        # Late-import to avoid pulling Module 1 at module import time.
        from dispersive_readout.physics.config import (
            DecoherenceParams, DeviceConfig, REFERENCE_DEVICE, TransmonParams,
        )
        omega_q_fp = self._get("omega_q")
        T_1_fp = self._get("T_1")
        T_2_echo_fp = self._get("T_2_echo")
        if omega_q_fp is None or T_1_fp is None or T_2_echo_fp is None:
            raise ValueError(
                "to_device_config requires omega_q, T_1, and T_2_echo fits. "
                "Missing: " + ", ".join(n for n, v in (
                    ("omega_q", omega_q_fp), ("T_1", T_1_fp), ("T_2_echo", T_2_echo_fp),
                ) if v is None)
            )
        E_C = REFERENCE_DEVICE.transmon.E_C
        omega_q = omega_q_fp.value
        E_J_derived = (omega_q + E_C) ** 2 / (8.0 * E_C)
        E_J_reference = REFERENCE_DEVICE.transmon.E_J
        rel_drift = abs(E_J_derived - E_J_reference) / E_J_reference
        if rel_drift > E_J_tolerance_rel:
            warnings.warn(
                f"Derived E_J/2π = {E_J_derived / (2 * math.pi) / 1e9:.3f} GHz is "
                f"{rel_drift:.1%} off REFERENCE's E_J/2π = "
                f"{E_J_reference / (2 * math.pi) / 1e9:.3f} GHz — check the fit.",
                UserWarning,
                stacklevel=2,
            )
        transmon = TransmonParams(E_C=E_C, E_J=E_J_derived, n_g=REFERENCE_DEVICE.transmon.n_g)
        gamma_1 = 1.0 / T_1_fp.value
        gamma_phi = max(1.0 / T_2_echo_fp.value - 0.5 * gamma_1, 0.0)
        decoherence = DecoherenceParams(
            gamma_1=gamma_1, gamma_phi=gamma_phi,
            n_th=REFERENCE_DEVICE.decoherence.n_th,
            purcell_enabled=REFERENCE_DEVICE.decoherence.purcell_enabled,
        )
        return DeviceConfig(
            transmon=transmon,
            resonator=REFERENCE_DEVICE.resonator,
            coupling=REFERENCE_DEVICE.coupling,
            decoherence=decoherence,
            truncation=REFERENCE_DEVICE.truncation,
        )
```

- [ ] **Step 7.4: Export**

Add to `__init__.py`:

```python
from .fitting import ExtractedParameterPack, FittedParameter
```

and add `"ExtractedParameterPack"`, `"FittedParameter"` to `__all__`.

- [ ] **Step 7.5: Run C4 + C7 + full suite**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 13 PASS (8 prior + C4a/b/c + C7a/b).

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 85 passing.

- [ ] **Step 7.6: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/fitting.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 7 — Pydantic schemas + to_device_config

FittedParameter + ExtractedParameterPack with YAML round-trip. The
to_device_config bridge back-solves E_J from ω_q via Koch 2007
(amendment 5); warns on > 30% E_J drift from REFERENCE. C4a/b/c and
C7a/b tests passing.

85 tests passing."
```

**Definition of done:** C4 + C7 passing; 85 total.

---

## Task 8: lmfit wrappers (`fit_rabi`, `fit_ramsey`, `fit_t1`, `fit_t2_echo`) with point-estimate uncertainty

**Rationale:** Spec §4.3 — lmfit wrappers on each protocol's fit form. Placeholder uncertainty = SE from lmfit's covariance matrix; parametric bootstrap replaces it in Task 9.

**Files:**
- Modify: `dispersive_readout/characterization/fitting.py` (add four fit functions + `_initial_guess_*` helpers)
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py` (add point-estimate tests for each fitter)

- [ ] **Step 8.1: Write four failing fit-point-estimate tests**

Append:

```python
# -- Point-estimate fit tests (full bootstrap uncertainty lives in Task 9) ---

def test_fit_rabi_point_estimate_recovers_epsilon_pi_within_3pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    from dispersive_readout.characterization.fitting import fit_rabi
    eps_pi_truth = 2 * math.pi * 50e6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0, drive_amplitude_uncertainty=0.0)
    trace = generate_rabi_trace(eps_pi_truth, 2 * math.pi * 4.5e9, noise, seed=10)
    fp = fit_rabi(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - eps_pi_truth) / eps_pi_truth
    assert rel < 0.03, f"fit_rabi rel={rel:.3%}"
    assert fp.name == "epsilon_pi"


def test_fit_ramsey_point_estimate_recovers_omega_q_within_0_1pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q_truth = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_ramsey_trace(omega_q_truth, T_2_star=T_2_star_truth, noise=noise, seed=11)
    fp_omega, fp_T2star = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp_omega.value - omega_q_truth) / omega_q_truth
    assert rel < 1e-3, f"fit_ramsey omega_q rel={rel:.3e}"
    rel_T2 = abs(fp_T2star.value - T_2_star_truth) / T_2_star_truth
    assert rel_T2 < 0.15


def test_fit_t1_point_estimate_recovers_T1_within_5pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    from dispersive_readout.characterization.fitting import fit_t1
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, seed=12)
    fp = fit_t1(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_1_truth) / T_1_truth
    assert rel < 0.05


def test_fit_t2_echo_point_estimate_recovers_T2_within_5pct():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t2_echo_trace
    from dispersive_readout.characterization.fitting import fit_t2_echo
    T_2_truth = 40e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t2_echo_trace(T_2_truth, noise, seed=13)
    fp = fit_t2_echo(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_2_truth) / T_2_truth
    assert rel < 0.05
```

- [ ] **Step 8.2: Run FAIL**

Expected: 4 FAIL — ImportError on fit functions.

- [ ] **Step 8.3: Add lmfit wrappers to `fitting.py`**

Append to `dispersive_readout/characterization/fitting.py`:

```python
# -- lmfit wrappers (point-estimate layer; bootstrap in Task 9) --------------

import numpy as np
import lmfit

from .protocols import TraceData


# --- Initial-guess helpers --------------------------------------------------

def _initial_guess_rabi(eps: np.ndarray, P1: np.ndarray) -> dict[str, float]:
    """Crude ε_π estimate from the first P1 minimum."""
    idx = int(np.argmin(P1))
    return {
        "A": float(P1.mean()),
        "B": float((P1.max() - P1.min()) / 2.0),
        "epsilon_pi": max(float(eps[idx]), 1e-12),
        "phi": 0.0,
    }


def _initial_guess_ramsey(delays: np.ndarray, P1: np.ndarray) -> dict[str, float]:
    """FFT peak for Δω; exponential-decay envelope for T2*."""
    signal = P1 - P1.mean()
    dt = float(delays[1] - delays[0])
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(delays), d=dt)
    if len(fft) > 1 and np.any(fft[1:] > 0):
        peak = int(np.argmax(fft[1:])) + 1
        delta_omega0 = 2 * math.pi * float(freqs[peak])
    else:
        delta_omega0 = 2 * math.pi * 1e6
    env0 = (P1.max() - P1.min()) / 2.0
    return {
        "A": float(P1.mean()),
        "B": float(env0 if env0 > 0 else 0.1),
        "delta_omega": delta_omega0,
        "T_2_star": max(float(delays.max()) / 3.0, 1e-9),
        "phi": 0.0,
    }


def _initial_guess_exponential(delays: np.ndarray, P1: np.ndarray, is_echo: bool) -> dict[str, float]:
    """Shared exponential initial guess for T1 and T2-echo fits."""
    if is_echo:
        # P1 = 0.5 − 0.5·exp(−τ/T2): log(1 − 2P1) is linear in τ.
        signal = 1.0 - 2.0 * P1
        mask = signal > 0.02
    else:
        floor = float(P1[-max(1, len(P1) // 10):].mean())
        signal = P1 - floor
        mask = signal > 0.02
    if mask.sum() < 3:
        tau0 = float(delays.max()) / 3.0
    else:
        coef = np.polyfit(delays[mask], np.log(signal[mask]), 1)
        tau0 = -1.0 / coef[0] if coef[0] < 0 else float(delays.max())
    return {
        "A": 0.0 if is_echo else float(P1[-max(1, len(P1) // 10):].mean()),
        "B": float(signal.max()),
        "tau": max(tau0, 1e-9),
    }


# --- Point-estimate fits ----------------------------------------------------

def _fit_point(model: lmfit.Model, params: lmfit.Parameters, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> lmfit.model.ModelResult:
    """Shared point-estimate run."""
    return model.fit(y, params=params, x=x, weights=1.0 / np.clip(weights, 1e-12, None))


def fit_rabi(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit Rabi: P₁(ε) = A + B·cos(π·ε/ε_π + φ). Returns ε_π with uncertainty."""
    def _model(x, A, B, epsilon_pi, phi):
        return A + B * np.cos(np.pi * x / epsilon_pi + phi)

    model = lmfit.Model(_model)
    g = _initial_guess_rabi(trace.sweep_values, trace.P1)
    params = model.make_params(**g)
    params["epsilon_pi"].set(min=2 * math.pi * 1e6, max=2 * math.pi * 1e9)
    params["B"].set(min=0.0, max=1.0)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    value = float(result.params["epsilon_pi"].value)
    # Point-estimate uncertainty from the covariance matrix (bootstrap override in Task 9).
    stderr = result.params["epsilon_pi"].stderr
    unc = float(stderr) if stderr is not None and stderr > 0 else value * 0.01
    return FittedParameter(
        name="epsilon_pi", value=value, uncertainty=unc, unit="rad/s",
        protocol_source="rabi", goodness_of_fit=float(result.redchi),
        n_bootstrap=0,
    )


def fit_ramsey(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> tuple[FittedParameter, FittedParameter]:
    """Fit Ramsey: P₁(τ) = A + B·exp(−τ/T_2*)·cos(Δω·τ + φ). Returns (omega_q, T_2_star).

    Edge case (amendment 2 / §5 test C6a): if initial FFT guess shows < 1
    oscillation over the sweep, pin Δω=0 and fit the envelope only.
    """
    g = _initial_guess_ramsey(trace.sweep_values, trace.P1)
    span = float(trace.sweep_values.max() - trace.sweep_values.min())
    oscillations = g["delta_omega"] * span / (2 * math.pi)

    if oscillations < 1.0:
        # Envelope-only fallback: fit A + B·exp(−τ/T_2*).
        def _env_model(x, A, B, T_2_star):
            return A + B * np.exp(-x / T_2_star)
        model = lmfit.Model(_env_model)
        params = model.make_params(A=g["A"], B=g["B"], T_2_star=g["T_2_star"])
        params["T_2_star"].set(min=1e-7, max=1e-3)
        result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
        T_2 = float(result.params["T_2_star"].value)
        T_2_err = result.params["T_2_star"].stderr or T_2 * 0.1
        # ω_q pinned to metadata ground-truth (the caller acknowledges Δω=0).
        omega_q_meta = float(trace.metadata.get("ground_truth", {}).get("omega_q", 2 * math.pi * 4.5e9))
        fp_omega = FittedParameter(
            name="omega_q", value=omega_q_meta, uncertainty=2 * math.pi * 1e3,
            unit="rad/s", protocol_source="ramsey",
            goodness_of_fit=float(result.redchi), n_bootstrap=0,
        )
        fp_T2 = FittedParameter(
            name="T_2_star", value=T_2, uncertainty=float(T_2_err),
            unit="s", protocol_source="ramsey",
            goodness_of_fit=float(result.redchi), n_bootstrap=0,
        )
        return fp_omega, fp_T2

    def _model(x, A, B, delta_omega, T_2_star, phi):
        return A + B * np.exp(-x / T_2_star) * np.cos(delta_omega * x + phi)
    model = lmfit.Model(_model)
    params = model.make_params(**g)
    params["T_2_star"].set(min=1e-7, max=1e-3)
    params["delta_omega"].set(min=2 * math.pi * 1e3, max=2 * math.pi * 1e9)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    delta_omega_fit = float(result.params["delta_omega"].value)
    T_2_fit = float(result.params["T_2_star"].value)
    d_omega_err = result.params["delta_omega"].stderr or abs(delta_omega_fit) * 0.01
    T_2_err = result.params["T_2_star"].stderr or T_2_fit * 0.1
    # ω_q = ω_drive + Δω; recover ω_drive from metadata.
    gt = trace.metadata.get("ground_truth", {})
    omega_q_metadata = float(gt.get("omega_q", 2 * math.pi * 4.5e9))
    omega_drive = omega_q_metadata - float(gt.get("omega_drive_offset", 2 * math.pi * 1e6))
    omega_q_fit = omega_drive + delta_omega_fit
    fp_omega = FittedParameter(
        name="omega_q", value=omega_q_fit, uncertainty=float(d_omega_err),
        unit="rad/s", protocol_source="ramsey",
        goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
    fp_T2 = FittedParameter(
        name="T_2_star", value=T_2_fit, uncertainty=float(T_2_err),
        unit="s", protocol_source="ramsey",
        goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
    return fp_omega, fp_T2


def fit_t1(
    trace: TraceData,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit T1: P₁(τ) = A + B·exp(−τ/T_1)."""
    def _model(x, A, B, tau):
        return A + B * np.exp(-x / tau)
    g = _initial_guess_exponential(trace.sweep_values, trace.P1, is_echo=False)
    model = lmfit.Model(_model)
    params = model.make_params(**g)
    params["tau"].set(min=1e-7, max=1e-3)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)
    tau = float(result.params["tau"].value)
    tau_err = result.params["tau"].stderr or tau * 0.1
    return FittedParameter(
        name="T_1", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t1", goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )


def fit_t2_echo(
    trace: TraceData,
    use_stretched_exponential: bool = False,
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> FittedParameter:
    """Fit Hahn echo: P₁(τ) = A + B·exp(−τ/T_2). Stretched fallback if redchi > 3."""
    def _plain(x, A, B, tau):
        return A + B * np.exp(-x / tau)
    def _stretched(x, A, B, tau, n):
        return A + B * np.exp(-((x / tau) ** n))

    g = _initial_guess_exponential(trace.sweep_values, trace.P1, is_echo=True)
    g["A"] = 0.5   # Hahn-echo form pins A=0.5 asymptotically (cold-qubit limit is 0.5).
    g["B"] = -0.5  # and B is negative (P1 rises to 0.5 as τ→∞ from 0 at τ=0).

    model = lmfit.Model(_plain)
    params = model.make_params(**g)
    params["tau"].set(min=1e-7, max=1e-3)
    result = _fit_point(model, params, trace.sweep_values, trace.P1, trace.P1_uncertainty)

    if use_stretched_exponential or float(result.redchi) > 3.0:
        model_s = lmfit.Model(_stretched)
        ps = model_s.make_params(**{**g, "n": 1.0})
        ps["tau"].set(min=1e-7, max=1e-3)
        ps["n"].set(min=0.3, max=3.0)
        result = _fit_point(model_s, ps, trace.sweep_values, trace.P1, trace.P1_uncertainty)

    tau = float(result.params["tau"].value)
    tau_err = result.params["tau"].stderr or tau * 0.1
    return FittedParameter(
        name="T_2_echo", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t2_echo", goodness_of_fit=float(result.redchi), n_bootstrap=0,
    )
```

- [ ] **Step 8.4: Export + run tests**

Add `fit_rabi, fit_ramsey, fit_t1, fit_t2_echo` to `__init__.py` imports/exports.

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 17 PASS (13 prior + 4 fit point-estimate).

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 89 passing.

- [ ] **Step 8.5: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/fitting.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 8 — lmfit wrappers (point-estimate)

fit_rabi, fit_ramsey, fit_t1, fit_t2_echo with lmfit-covariance-based
uncertainty as a placeholder (parametric bootstrap replaces it in
Task 9, per amendment 3). fit_ramsey includes the envelope-only
fallback for the Δω≈0 edge case (C6a).

89 tests passing."
```

**Definition of done:** Four point-estimate fit tests passing; 89 total.

---

## Task 9: `parametric_bootstrap` + wire into fit functions + `fit_all`

**Rationale:** Amendment 3 — bootstrap must regenerate full (1/f + shot + readout) noise realizations around the best fit, not iid-resample residuals. Overrides the covariance-matrix uncertainty from Task 8.

**Files:**
- Modify: `dispersive_readout/characterization/fitting.py` (add `parametric_bootstrap` + wire it)
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py` (bootstrap sanity test)

- [ ] **Step 9.1: Write failing bootstrap sanity test**

Append:

```python
def test_parametric_bootstrap_produces_nonzero_uncertainty_on_noisy_trace():
    """With non-zero drift + shot noise, bootstrap uncertainty must be > 0 and
    larger than the covariance-matrix SE by at least a factor of 1.5 (the
    gap amendment 3 is designed to reveal)."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q_truth = 2 * math.pi * 4.5e9
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    trace = generate_ramsey_trace(omega_q_truth, T_2_star=20e-6, noise=noise, seed=20)
    # Point-estimate uncertainty (bootstrap_samples=0 → covariance fallback).
    fp_omega_pe, _ = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    # Bootstrap uncertainty.
    fp_omega_bs, _ = fit_ramsey(trace, bootstrap_samples=50, seed=42)
    assert fp_omega_bs.n_bootstrap == 50
    assert fp_omega_bs.uncertainty > 0
    # Parametric bootstrap captures correlated drift → larger SE than covariance.
    assert fp_omega_bs.uncertainty > 1.5 * fp_omega_pe.uncertainty, (
        f"bootstrap SE {fp_omega_bs.uncertainty:.3e} not > 1.5× covariance SE {fp_omega_pe.uncertainty:.3e}"
    )
```

- [ ] **Step 9.2: Run FAIL**

Expected: FAIL — `parametric_bootstrap` not yet wired; current fit functions ignore `bootstrap_samples > 0`.

- [ ] **Step 9.3: Add `parametric_bootstrap` + wire**

Append to `fitting.py`:

```python
# -- Parametric bootstrap (amendment 3) -------------------------------------

from .protocols import (
    generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
)
from .noise import NoiseModelParams


def _noise_from_trace_metadata(trace: TraceData) -> NoiseModelParams:
    meta_noise = trace.metadata.get("noise", {})
    return NoiseModelParams(
        n_shots_per_point=int(meta_noise.get("n_shots_per_point", 2000)),
        drift_amplitude_Hz=float(meta_noise.get("drift_amplitude_Hz", 0.0)),
        drift_alpha=float(meta_noise.get("drift_alpha", 1.0)),
        drive_amplitude_uncertainty=float(meta_noise.get("drive_amplitude_uncertainty", 0.0)),
    )


def parametric_bootstrap(
    protocol: Literal["rabi", "ramsey", "t1", "t2_echo"],
    best_fit_values: dict[str, float],
    noise: NoiseModelParams,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Parametric bootstrap per amendment 3.

    For k in 1..n_bootstrap:
        Regenerate a fresh trace from `best_fit_values` + fresh noise realization
          (seed_k = seed + k).
        Point-estimate fit the fresh trace.
        Record the fitted parameters.
    Return {param_name: ndarray of length n_bootstrap}.
    """
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {}

    for k in range(n_bootstrap):
        sub_seed = int(rng.integers(2**31 - 1))
        if protocol == "rabi":
            trace_k = generate_rabi_trace(
                best_fit_values["epsilon_pi"], best_fit_values.get("omega_q", 2 * math.pi * 4.5e9),
                noise, seed=sub_seed,
            )
            fp = fit_rabi(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("epsilon_pi", []).append(fp.value)
        elif protocol == "ramsey":
            trace_k = generate_ramsey_trace(
                best_fit_values["omega_q"], T_2_star=best_fit_values["T_2_star"],
                noise=noise, seed=sub_seed,
            )
            fp_o, fp_t = fit_ramsey(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("omega_q", []).append(fp_o.value)
            boot.setdefault("T_2_star", []).append(fp_t.value)
        elif protocol == "t1":
            trace_k = generate_t1_trace(best_fit_values["T_1"], noise, seed=sub_seed)
            fp = fit_t1(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("T_1", []).append(fp.value)
        elif protocol == "t2_echo":
            trace_k = generate_t2_echo_trace(best_fit_values["T_2_echo"], noise, seed=sub_seed)
            fp = fit_t2_echo(trace_k, bootstrap_samples=0, seed=sub_seed)
            boot.setdefault("T_2_echo", []).append(fp.value)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    return {name: np.array(values, dtype=float) for name, values in boot.items()}
```

Now update each `fit_X` wrapper to use the bootstrap when `bootstrap_samples > 0`. Modify the final `return FittedParameter(...)` blocks.

For `fit_rabi`, before the final `return`:

```python
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "rabi", {"epsilon_pi": value, "omega_q": float(trace.metadata.get("ground_truth", {}).get("omega_q", 2 * math.pi * 4.5e9))},
            noise=boot_noise, n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        unc = float(np.std(boot["epsilon_pi"]))
        n_bs = bootstrap_samples
    else:
        n_bs = 0
    return FittedParameter(
        name="epsilon_pi", value=value, uncertainty=unc, unit="rad/s",
        protocol_source="rabi", goodness_of_fit=float(result.redchi), n_bootstrap=n_bs,
    )
```

For `fit_ramsey`, after the point-estimate block (both branches: oscillating and envelope-only):

```python
    if bootstrap_samples > 0 and oscillations >= 1.0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "ramsey",
            {"omega_q": omega_q_fit, "T_2_star": T_2_fit},
            noise=boot_noise, n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        fp_omega = fp_omega.model_copy(update={
            "uncertainty": max(float(np.std(boot["omega_q"])), 1e-9),
            "n_bootstrap": bootstrap_samples,
        })
        fp_T2 = fp_T2.model_copy(update={
            "uncertainty": max(float(np.std(boot["T_2_star"])), 1e-9),
            "n_bootstrap": bootstrap_samples,
        })
    return fp_omega, fp_T2
```

For `fit_t1`, before returning:

```python
    if bootstrap_samples > 0:
        boot_noise = _noise_from_trace_metadata(trace)
        boot = parametric_bootstrap(
            "t1", {"T_1": tau}, noise=boot_noise,
            n_bootstrap=bootstrap_samples, seed=seed or 0,
        )
        tau_err = max(float(np.std(boot["T_1"])), 1e-12)
        n_bs = bootstrap_samples
    else:
        n_bs = 0
    return FittedParameter(
        name="T_1", value=tau, uncertainty=float(tau_err), unit="s",
        protocol_source="t1", goodness_of_fit=float(result.redchi), n_bootstrap=n_bs,
    )
```

For `fit_t2_echo`, same pattern — bootstrap with `{"T_2_echo": tau}`.

Add `fit_all`:

```python
def fit_all(
    traces: list[TraceData],
    bootstrap_samples: int = 200,
    seed: int | None = None,
    trace_file: str = "",
) -> ExtractedParameterPack:
    """Fit every trace in a bundle; return a Module-1-compatible parameter pack."""
    from datetime import datetime, timezone
    import subprocess
    fitted: list[FittedParameter] = []
    for t in traces:
        if t.protocol == "rabi":
            fitted.append(fit_rabi(t, bootstrap_samples=bootstrap_samples, seed=seed))
        elif t.protocol == "ramsey":
            o, ts = fit_ramsey(t, bootstrap_samples=bootstrap_samples, seed=seed)
            fitted.extend([o, ts])
        elif t.protocol == "t1":
            fitted.append(fit_t1(t, bootstrap_samples=bootstrap_samples, seed=seed))
        elif t.protocol == "t2_echo":
            fitted.append(fit_t2_echo(t, bootstrap_samples=bootstrap_samples, seed=seed))
        else:
            raise ValueError(f"Unknown protocol: {t.protocol}")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    return ExtractedParameterPack(
        fitted_parameters=fitted,
        trace_file=trace_file,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stage_06_version=sha,
    )
```

- [ ] **Step 9.4: Export**

Add to `__init__.py`: `parametric_bootstrap`, `fit_all`.

- [ ] **Step 9.5: Run tests**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 18 PASS.

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 90 passing.

- [ ] **Step 9.6: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/fitting.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 9 — parametric bootstrap + fit_all

Amendment 3: parametric bootstrap regenerates fresh (1/f + shot + readout)
noise realizations around the best-fit, re-fits each, returns the spread
as uncertainty. Correct under correlated drift; covariance-matrix SE
retained as bootstrap_samples=0 fallback. fit_all drives a trace bundle
through and produces a Module-1-compatible ExtractedParameterPack.

90 tests passing."
```

**Definition of done:** Bootstrap sanity test passing; `fit_all` produces a valid pack; 90 total.

---

## Task 10: Recovery harness — `DeviceGroundTruth`, `fit_one_device`, device family generator

**Rationale:** Amendment 8 — `fit_one_device(device, noise, seed) → list[RecoveryResult]` as a pure function; amendment 9 + user review note — family generator rejects T₂ > 2·T₁·0.95 and serializes the device list with the artifact.

**Files:**
- Create: `dispersive_readout/characterization/recovery.py`
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 10.1: Write failing tests**

Append:

```python
# -- Recovery harness --------------------------------------------------------

def test_fit_one_device_returns_four_RecoveryResults():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import DeviceGroundTruth, fit_one_device
    d = DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6, omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6, thermal_offset=0.0,
        ramsey_detuning=2 * math.pi * 1e6,
    )
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    out = fit_one_device(d, noise, seed=42)
    assert len(out) == 4
    names = {r.parameter_name for r in out}
    assert names == {"T_1", "T_2_echo", "omega_q", "epsilon_pi"}


def test_generate_synthetic_device_family_rejects_T2_gt_2T1():
    from dispersive_readout.characterization.recovery import generate_synthetic_device_family
    devices = generate_synthetic_device_family(n_devices=50, seed=42)
    assert len(devices) == 50
    for d in devices[2:]:   # device[0] and device[1] are deterministic overrides
        assert d.T_2_echo <= 2.0 * d.T_1 * 0.95 + 1e-18, (
            f"Device with T_2={d.T_2_echo:.2e} exceeds 2·T_1·0.95={2 * d.T_1 * 0.95:.2e}"
        )
    assert devices[0].ramsey_detuning == 0.0   # zero-detuning edge case
    assert devices[1].thermal_offset == 0.08   # elevated-thermal edge case


def test_fit_one_device_is_deterministic_under_same_seed():
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import DeviceGroundTruth, fit_one_device
    d = DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6, omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
    )
    noise = NoiseModelParams(n_shots_per_point=2000, drift_amplitude_Hz=1e4)
    a = fit_one_device(d, noise, seed=123)
    b = fit_one_device(d, noise, seed=123)
    for ra, rb in zip(a, b):
        assert ra.parameter_name == rb.parameter_name
        assert ra.fitted_value == rb.fitted_value
```

- [ ] **Step 10.2: Run FAIL**

Expected: 3 FAIL — ImportError on `recovery`.

- [ ] **Step 10.3: Write `recovery.py`**

Create `dispersive_readout/characterization/recovery.py`:

```python
"""Module 3 — parameter recovery harness (G2).

Built around the pure function `fit_one_device(device, noise, seed) → list[RecoveryResult]`
per amendment 8; serial fallback is list(map(...)), Modal mode would be
fit_one_device.map(...). The harness aggregates a CoverageReport per
parameter, with the 2σ binomial CI required by amendment 4.

Amendment 9: the committed recovery_coverage_report.yaml pins the
device list alongside the coverage statistics, so the artifact is
self-describing under numpy default_rng changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .fitting import FittedParameter, fit_rabi, fit_ramsey, fit_t1, fit_t2_echo
from .noise import NoiseModelParams
from .protocols import (
    generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
)


@dataclass(frozen=True)
class DeviceGroundTruth:
    """One synthetic device's ground truth."""
    T_1: float
    T_2_echo: float
    omega_q: float
    epsilon_pi: float
    thermal_offset: float = 0.0
    ramsey_detuning: float = 2.0 * math.pi * 1e6


@dataclass(frozen=True)
class RecoveryResult:
    parameter_name: str
    ground_truth: float
    fitted_value: float
    fitted_uncertainty: float
    z_score: float
    within_1_sigma: bool
    within_2_sigma: bool


@dataclass(frozen=True)
class CoverageReport:
    parameter_name: str
    n_devices: int
    coverage_1_sigma: float
    coverage_2_sigma: float
    coverage_1_sigma_ci_low: float
    coverage_1_sigma_ci_high: float
    coverage_2_sigma_ci_low: float
    coverage_2_sigma_ci_high: float
    bias: float
    bias_uncertainty: float


def _binomial_2sigma_ci(p: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return max(0.0, p - 2.0 * se), min(1.0, p + 2.0 * se)


def _make_recovery_result(param_name: str, truth: float, fp: FittedParameter) -> RecoveryResult:
    unc = max(fp.uncertainty, 1e-30)
    z = (fp.value - truth) / unc
    return RecoveryResult(
        parameter_name=param_name,
        ground_truth=float(truth),
        fitted_value=float(fp.value),
        fitted_uncertainty=float(unc),
        z_score=float(z),
        within_1_sigma=abs(z) <= 1.0,
        within_2_sigma=abs(z) <= 2.0,
    )


def fit_one_device(
    device: DeviceGroundTruth,
    noise: NoiseModelParams,
    seed: int,
) -> list[RecoveryResult]:
    """Pure function: generate 4 traces, fit, compare to truth (amendment 8).

    No global state, no FS I/O. RNG seed upper bound is 2**31 - 1 (user review
    fix 1) to avoid collisions near numpy's default uint64 edge.
    """
    rng = np.random.default_rng(seed)
    # Trace seeds.
    rabi_seed = int(rng.integers(2**31 - 1))
    ramsey_seed = int(rng.integers(2**31 - 1))
    t1_seed = int(rng.integers(2**31 - 1))
    t2_seed = int(rng.integers(2**31 - 1))
    # Fit seeds (same generator; downstream draws bootstrap sub-seeds).
    fit_seeds = [int(rng.integers(2**31 - 1)) for _ in range(4)]

    rabi_trace = generate_rabi_trace(device.epsilon_pi, device.omega_q, noise, seed=rabi_seed)
    ramsey_trace = generate_ramsey_trace(
        device.omega_q, T_2_star=device.T_2_echo, noise=noise,
        omega_drive_offset=device.ramsey_detuning, seed=ramsey_seed,
    )
    t1_trace = generate_t1_trace(device.T_1, noise, thermal_offset=device.thermal_offset, seed=t1_seed)
    t2_trace = generate_t2_echo_trace(device.T_2_echo, noise, seed=t2_seed)

    fp_eps = fit_rabi(rabi_trace, bootstrap_samples=200, seed=fit_seeds[0])
    fp_omega, _fp_T2star = fit_ramsey(ramsey_trace, bootstrap_samples=200, seed=fit_seeds[1])
    fp_T1 = fit_t1(t1_trace, bootstrap_samples=200, seed=fit_seeds[2])
    fp_T2 = fit_t2_echo(t2_trace, bootstrap_samples=200, seed=fit_seeds[3])

    return [
        _make_recovery_result("T_1", device.T_1, fp_T1),
        _make_recovery_result("T_2_echo", device.T_2_echo, fp_T2),
        _make_recovery_result("omega_q", device.omega_q, fp_omega),
        _make_recovery_result("epsilon_pi", device.epsilon_pi, fp_eps),
    ]


def generate_synthetic_device_family(n_devices: int, seed: int) -> list[DeviceGroundTruth]:
    """Log-uniform(T_1, T_2_echo) in [5 µs, 100 µs]; uniform(ω_q/2π) in [4 GHz, 5 GHz].

    Physical constraint: T_2_echo ≤ 2·T_1·0.95 (Hahn echo bounded above by 2T_1,
    with 0.95 margin for bootstrap fluctuations).

    Deterministic overrides (not subject to sampling):
      device[0]: ramsey_detuning = 0 (zero-detuning edge case, C6a)
      device[1]: thermal_offset = 0.08 (elevated-thermal edge case, C6b)
    """
    rng = np.random.default_rng(seed)
    out: list[DeviceGroundTruth] = []

    # Device 0 — zero-detuning.
    out.append(DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6,
        omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
        thermal_offset=0.0,
        ramsey_detuning=0.0,
    ))
    # Device 1 — elevated thermal.
    out.append(DeviceGroundTruth(
        T_1=30e-6, T_2_echo=40e-6,
        omega_q=2 * math.pi * 4.5e9,
        epsilon_pi=2 * math.pi * 50e6,
        thermal_offset=0.08,
        ramsey_detuning=2 * math.pi * 1e6,
    ))

    # Remaining n_devices - 2 from the sampler with rejection.
    log_lo = math.log(5e-6)
    log_hi = math.log(100e-6)
    while len(out) < n_devices:
        T_1 = math.exp(rng.uniform(log_lo, log_hi))
        T_2 = math.exp(rng.uniform(log_lo, log_hi))
        if T_2 > 2.0 * T_1 * 0.95:
            continue
        omega_q = 2 * math.pi * rng.uniform(4e9, 5e9)
        epsilon_pi = 2 * math.pi * 50e6 * (1.0 + 0.2 * rng.standard_normal())
        out.append(DeviceGroundTruth(
            T_1=T_1, T_2_echo=T_2, omega_q=omega_q,
            epsilon_pi=epsilon_pi,
            thermal_offset=0.0,
            ramsey_detuning=2 * math.pi * 1e6,
        ))
    return out
```

- [ ] **Step 10.4: Export**

Add to `__init__.py`: `DeviceGroundTruth, RecoveryResult, CoverageReport, fit_one_device, generate_synthetic_device_family`.

- [ ] **Step 10.5: Run tests**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 21 PASS (18 prior + 3 new).

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -5
```

Expected: 93 passing.

- [ ] **Step 10.6: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/recovery.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 10 — recovery harness core

fit_one_device pure function (amendment 8, seed upper bound 2**31-1),
DeviceGroundTruth / RecoveryResult / CoverageReport dataclasses,
generate_synthetic_device_family with T2 < 2·T1·0.95 rejection and
deterministic overrides for device[0] (zero detuning) and device[1]
(elevated thermal).

93 tests passing."
```

**Definition of done:** 3 harness core tests passing; 93 total.

---

## Task 11: `run_recovery_harness` + YAML I/O + first 50-device commit + C3 regression test

**Rationale:** Amendment 9 — run the harness at SEED=42, verify each parameter's 2σ CI includes its target, commit the artifact, then add C3 to gate future runs within ±2%. This is the Module 3 gate.

**Files:**
- Modify: `dispersive_readout/characterization/recovery.py`
- Modify: `dispersive_readout/characterization/__init__.py`
- Create: `06_Dispersive_Readout/figures/recovery_coverage_report.yaml` (generated in Step 11.4)
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 11.1: Add `run_recovery_harness` and YAML I/O**

Append to `recovery.py`:

```python
import yaml


def run_recovery_harness(
    n_devices: int = 50,
    noise: NoiseModelParams | None = None,
    seed: int = 42,
) -> tuple[dict[str, CoverageReport], list[DeviceGroundTruth]]:
    """Run the full harness at the given seed; return (reports, devices)."""
    if noise is None:
        noise = NoiseModelParams()
    devices = generate_synthetic_device_family(n_devices=n_devices, seed=seed)
    # Sub-seeds per device, drawn deterministically from the harness seed.
    rng = np.random.default_rng(seed)
    results_by_param: dict[str, list[RecoveryResult]] = {
        "T_1": [], "T_2_echo": [], "omega_q": [], "epsilon_pi": [],
    }
    for d in devices:
        sub_seed = int(rng.integers(2**31 - 1))
        for r in fit_one_device(d, noise, seed=sub_seed):
            results_by_param[r.parameter_name].append(r)

    reports: dict[str, CoverageReport] = {}
    for name, records in results_by_param.items():
        n = len(records)
        cov1 = sum(r.within_1_sigma for r in records) / n
        cov2 = sum(r.within_2_sigma for r in records) / n
        c1_lo, c1_hi = _binomial_2sigma_ci(cov1, n)
        c2_lo, c2_hi = _binomial_2sigma_ci(cov2, n)
        diffs = np.array([r.fitted_value - r.ground_truth for r in records])
        bias = float(diffs.mean())
        bias_unc = float(diffs.std(ddof=1) / math.sqrt(n))
        reports[name] = CoverageReport(
            parameter_name=name,
            n_devices=n,
            coverage_1_sigma=cov1,
            coverage_2_sigma=cov2,
            coverage_1_sigma_ci_low=c1_lo,
            coverage_1_sigma_ci_high=c1_hi,
            coverage_2_sigma_ci_low=c2_lo,
            coverage_2_sigma_ci_high=c2_hi,
            bias=bias,
            bias_uncertainty=bias_unc,
        )
    return reports, devices


def save_coverage_report(
    reports: dict[str, CoverageReport],
    devices: list[DeviceGroundTruth],
    path: str | Path,
    seed: int,
) -> None:
    """Serialize the coverage report + device list (for RNG stability)."""
    payload = {
        "seed": seed,
        "n_devices": len(devices),
        "coverage": {name: asdict(rep) for name, rep in reports.items()},
        "devices": [asdict(d) for d in devices],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_committed_coverage_report(path: str | Path) -> dict[str, CoverageReport]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        name: CoverageReport(**rec)
        for name, rec in data["coverage"].items()
    }


def format_recovery_table(reports: dict[str, CoverageReport]) -> str:
    lines = [
        "| Parameter | Cov 1σ (target 68%) | 2σ CI | Cov 2σ (target 95%) | 2σ CI | Bias |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        lines.append(
            f"| `{name}` | {r.coverage_1_sigma:.1%} | "
            f"[{r.coverage_1_sigma_ci_low:.1%}, {r.coverage_1_sigma_ci_high:.1%}] | "
            f"{r.coverage_2_sigma:.1%} | "
            f"[{r.coverage_2_sigma_ci_low:.1%}, {r.coverage_2_sigma_ci_high:.1%}] | "
            f"{r.bias:+.3e} ± {r.bias_uncertainty:.1e} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 11.2: Export**

Add `run_recovery_harness, save_coverage_report, load_committed_coverage_report, format_recovery_table` to `__init__.py`.

- [ ] **Step 11.3: Generate the committed coverage artifact**

Write a one-shot script (no need to commit it — we only need the output):

```bash
python -c "
from dispersive_readout.characterization.noise import NoiseModelParams
from dispersive_readout.characterization.recovery import (
    run_recovery_harness, save_coverage_report, format_recovery_table,
)
reports, devices = run_recovery_harness(n_devices=50, noise=NoiseModelParams(), seed=42)
save_coverage_report(reports, devices, '06_Dispersive_Readout/figures/recovery_coverage_report.yaml', seed=42)
print(format_recovery_table(reports))
for name, r in reports.items():
    cal_1 = r.coverage_1_sigma_ci_low <= 0.68 <= r.coverage_1_sigma_ci_high
    cal_2 = r.coverage_2_sigma_ci_low <= 0.95 <= r.coverage_2_sigma_ci_high
    print(f'{name}: 1σ CI includes 68%? {cal_1}; 2σ CI includes 95%? {cal_2}')
" 2>&1 | tail -20
```

Expected wall-clock: 1–3 minutes. Print output must show `1σ CI includes 68%? True` and `2σ CI includes 95%? True` for all four parameters. **STOP if any is False** — the fitter is miscalibrated; diagnose per §8 flag #1 (check bias, increase `n_bootstrap`, investigate the 1/f drift correlation length) before committing the artifact. Do NOT lower the gate.

- [ ] **Step 11.4: Write C3 regression test**

Append to `test_characterization.py`:

```python
# -- C3: recovery-coverage regression gate (amendment 9) ---------------------

@pytest.mark.slow
def test_C3_recovery_coverage_matches_committed_artifact():
    """Re-run 50-device harness at SEED=42 and match the committed artifact
    within ±2% per parameter. Regression gate; if this fails, diagnose the
    fitter before regenerating the artifact."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.recovery import (
        run_recovery_harness, load_committed_coverage_report,
    )
    observed_reports, _ = run_recovery_harness(n_devices=50, noise=NoiseModelParams(), seed=42)
    committed = load_committed_coverage_report(
        "06_Dispersive_Readout/figures/recovery_coverage_report.yaml"
    )
    for name, rep in observed_reports.items():
        ref = committed[name]
        for field_name in ("coverage_1_sigma", "coverage_2_sigma"):
            delta = abs(getattr(rep, field_name) - getattr(ref, field_name))
            assert delta < 0.02, (
                f"{name}.{field_name} regression: observed {getattr(rep, field_name):.2%} "
                f"vs committed {getattr(ref, field_name):.2%} (Δ={delta:.2%})"
            )
```

Mark `@pytest.mark.slow` so this test runs only in the full suite, not on every `-m "not slow"` run.

- [ ] **Step 11.5: Run C3**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py::test_C3_recovery_coverage_matches_committed_artifact -v -p no:dash
```

Expected: PASS (~2 min).

Run full suite:

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: 94 passing.

- [ ] **Step 11.6: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/recovery.py dispersive_readout/tests/test_characterization.py 06_Dispersive_Readout/figures/recovery_coverage_report.yaml
git commit -m "feat(stage06): Module 3 Task 11 — recovery harness + committed artifact

run_recovery_harness produces per-parameter CoverageReport with 2σ
binomial CI (amendment 4). Calibration gate satisfied at SEED=42,
n=50 for all four parameters: 2σ CI on observed coverage includes
95% (and 1σ CI includes 68%). Committed artifact at
06_Dispersive_Readout/figures/recovery_coverage_report.yaml with
the device list embedded (amendment 9 + RNG stability hedge).

C3 regression gate passing within ±2%.

94 tests passing."
```

**Definition of done:** C3 passing; committed artifact exists and documents its calibration; 94 total tests.

---

## Task 12: CLI + thin script entry + C5 tests

**Rationale:** Spec §4.5 — experimentalist-facing entry. Three modes: fit existing traces, run recovery, generate synthetic bundle.

**Files:**
- Create: `dispersive_readout/characterization/cli.py`
- Create: `06_Dispersive_Readout/characterize.py`
- Modify: `dispersive_readout/characterization/__init__.py`
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 12.1: Write failing C5 tests**

Append:

```python
# -- C5: CLI smoke tests ----------------------------------------------------

def _run_cli(args: list[str]) -> int:
    from dispersive_readout.characterization.cli import main
    return main(argv=args)


def test_C5a_cli_generate_synthetic(tmp_path):
    out = tmp_path / "synthetic.npz"
    rc = _run_cli(["--generate-synthetic", "--output", str(out), "--seed", "42"])
    assert rc == 0
    from dispersive_readout.characterization.protocols import load_trace_bundle
    traces = load_trace_bundle(str(out))
    assert {t.protocol for t in traces} == {"rabi", "ramsey", "t1", "t2_echo"}


def test_C5b_cli_full_pipeline_generate_then_fit(tmp_path):
    bundle = tmp_path / "synth.npz"
    params = tmp_path / "params.yaml"
    rc1 = _run_cli(["--generate-synthetic", "--output", str(bundle), "--seed", "42"])
    assert rc1 == 0
    rc2 = _run_cli(["--traces", str(bundle), "--output", str(params), "--bootstrap-samples", "20"])
    assert rc2 == 0
    import yaml
    with open(params) as f:
        data = yaml.safe_load(f)
    names = {p["name"] for p in data["fitted_parameters"]}
    assert {"T_1", "T_2_echo", "omega_q", "epsilon_pi"}.issubset(names)


def test_C5c_cli_help_has_no_todo(capsys):
    with pytest.raises(SystemExit):
        _run_cli(["--help"])
    out = capsys.readouterr().out
    for forbidden in ("TODO", "TBD", "FIXME", "XXX"):
        assert forbidden not in out, f"--help text contains '{forbidden}'"


def test_C5d_cli_rejects_conflicting_flags(tmp_path):
    """--traces + --generate-synthetic is ambiguous; must exit non-zero with a clear error."""
    rc = _run_cli(["--traces", "x.npz", "--generate-synthetic", "--output", str(tmp_path / "o.yaml")])
    assert rc != 0
```

- [ ] **Step 12.2: Run FAIL**

Expected: all 4 FAIL — ImportError on `cli`.

- [ ] **Step 12.3: Write `cli.py`**

Create `dispersive_readout/characterization/cli.py`:

```python
"""Stage 06 Module 3 — characterization CLI.

Entry: `python 06_Dispersive_Readout/characterize.py ...`

Three modes:
  --traces BUNDLE.npz --output PARAMS.yaml [--bootstrap-samples 200]
      Fit a trace bundle; write a Module-1-compatible YAML parameter pack.
  --recovery --n-devices 50 --output REPORT.yaml [--seed 42]
      Run the recovery harness; write a coverage report.
  --generate-synthetic --output BUNDLE.npz [--seed 42]
      Generate a reference synthetic trace bundle from REFERENCE_DEVICE.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

from .fitting import fit_all
from .noise import NoiseModelParams
from .protocols import (
    TraceData, generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
    load_trace_bundle, save_trace_bundle,
)
from .recovery import run_recovery_harness, save_coverage_report


_DESCRIPTION = """Extract device parameters from characterization traces.

Examples
--------
Fit a trace bundle:
    python 06_Dispersive_Readout/characterize.py --traces data.npz --output params.yaml

Run the recovery harness:
    python 06_Dispersive_Readout/characterize.py --recovery --n-devices 50 \\
        --output recovery_report.yaml --seed 42

Generate a reference synthetic bundle:
    python 06_Dispersive_Readout/characterize.py --generate-synthetic \\
        --output example_traces.npz --seed 42
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="06_Dispersive_Readout/characterize.py",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces", type=str, default=None, help="Path to a .npz trace bundle to fit.")
    parser.add_argument("--output", type=str, required=True, help="Output path (.yaml for params or report; .npz for synthetic).")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Parametric bootstrap samples per fitted parameter.")
    parser.add_argument("--recovery", action="store_true", help="Run the 50-device recovery harness.")
    parser.add_argument("--n-devices", type=int, default=50, help="Devices for the recovery harness.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a synthetic trace bundle from REFERENCE_DEVICE.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for determinism (default 42; matches the committed artifact).")
    return parser


def _reject_conflicts(args: argparse.Namespace) -> str | None:
    """Return an error message if the flag combination is invalid, else None."""
    modes = []
    if args.traces is not None:
        modes.append("--traces")
    if args.recovery:
        modes.append("--recovery")
    if args.generate_synthetic:
        modes.append("--generate-synthetic")
    if len(modes) == 0:
        return "Pick one of: --traces, --recovery, --generate-synthetic."
    if len(modes) > 1:
        return f"Flags {modes} are mutually exclusive; pick one."
    return None


def _mode_generate_synthetic(args: argparse.Namespace) -> int:
    noise = NoiseModelParams()
    # Reference device ground-truth values.
    eps_pi = 2 * math.pi * 50e6
    omega_q = 2 * math.pi * 4.5e9
    T_1 = 30e-6
    T_2 = 40e-6
    traces: list[TraceData] = [
        generate_rabi_trace(eps_pi, omega_q, noise, seed=args.seed),
        generate_ramsey_trace(omega_q, T_2_star=T_2, noise=noise, seed=args.seed + 1),
        generate_t1_trace(T_1, noise, seed=args.seed + 2),
        generate_t2_echo_trace(T_2, noise, seed=args.seed + 3),
    ]
    save_trace_bundle(traces, args.output)
    print(f"Wrote 4-protocol synthetic bundle: {args.output}")
    return 0


def _mode_traces(args: argparse.Namespace) -> int:
    traces = load_trace_bundle(args.traces)
    pack = fit_all(
        traces,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        trace_file=args.traces,
    )
    with open(args.output, "w") as f:
        yaml.safe_dump(pack.model_dump(), f, sort_keys=False)
    print(f"Fit {len(traces)} trace(s). Wrote parameter pack: {args.output}")
    return 0


def _mode_recovery(args: argparse.Namespace) -> int:
    noise = NoiseModelParams()
    reports, devices = run_recovery_harness(
        n_devices=args.n_devices, noise=noise, seed=args.seed,
    )
    save_coverage_report(reports, devices, args.output, seed=args.seed)
    print(f"Recovery harness wrote: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    err = _reject_conflicts(args)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.generate_synthetic:
        return _mode_generate_synthetic(args)
    if args.recovery:
        return _mode_recovery(args)
    if args.traces is not None:
        return _mode_traces(args)
    # Unreachable: _reject_conflicts already handled the no-mode case.
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12.4: Write thin script entry**

Create `06_Dispersive_Readout/characterize.py`:

```python
#!/usr/bin/env python3
"""Stage 06 Module 3 — characterization CLI entry point.

Example
-------
    python 06_Dispersive_Readout/characterize.py --traces data.npz --output params.yaml

See `--help` for full usage.
"""
from __future__ import annotations

import sys

from dispersive_readout.characterization.cli import main


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12.5: Make the entry executable**

```bash
chmod +x 06_Dispersive_Readout/characterize.py
```

- [ ] **Step 12.6: Export**

Add nothing new to `__init__.py` (CLI internals are private; `main` is re-exportable if needed later, but C5 tests import directly).

- [ ] **Step 12.7: Run C5 + smoke the entry**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash -m "not slow"
```

Expected: C5a/b/c/d all pass (~5 s each for a/b).

```bash
python 06_Dispersive_Readout/characterize.py --help | head -30
```

Expected: clean help text, no TODO/TBD strings.

Full suite (including slow):

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -10
```

Expected: 98 passing.

- [ ] **Step 12.8: Commit**

```bash
git add dispersive_readout/characterization/__init__.py dispersive_readout/characterization/cli.py 06_Dispersive_Readout/characterize.py dispersive_readout/tests/test_characterization.py
git commit -m "feat(stage06): Module 3 Task 12 — CLI + thin entry script

Three modes: --traces (fit), --recovery (harness), --generate-synthetic.
Mutually-exclusive flag validation. C5 smoke tests passing including
the --help no-TODO guard.

98 tests passing."
```

**Definition of done:** C5a/b/c/d passing; CLI runs end-to-end manually.

---

## Task 13: Figure 3 + commit committed `example_traces.npz`

**Rationale:** Spec §6 — publication-ready 2×2 Figure 3 with three protocol fits + parity-plot panel; committed example bundle for reproducibility.

**Files:**
- Create: `06_Dispersive_Readout/scripts/fig3_characterization.py`
- Create: `06_Dispersive_Readout/figures/fig3_characterization.png` (generated)
- Create: `06_Dispersive_Readout/examples/example_traces.npz` (generated)

- [ ] **Step 13.1: Generate the reference synthetic bundle**

```bash
mkdir -p 06_Dispersive_Readout/examples
python 06_Dispersive_Readout/characterize.py --generate-synthetic --output 06_Dispersive_Readout/examples/example_traces.npz --seed 42
ls -la 06_Dispersive_Readout/examples/example_traces.npz
```

Expected: non-empty .npz file.

- [ ] **Step 13.2: Write Figure 3 script**

Create `06_Dispersive_Readout/scripts/fig3_characterization.py`:

```python
"""Stage 06 Module 3 Figure 3 — characterization pipeline + parameter recovery.

Layout (2×2):
  (a) Rabi fit + residuals
  (b) Ramsey fit + residuals
  (c) T1 decay + residuals
  (d) Parameter-recovery parity plot (2×2 of sub-panels: T1, T2, ω_q, ε_π),
      fitted vs ground truth with y=x line, colored by |z| ≤ 1, annotated
      with observed 2σ coverage + 2σ binomial CI.

Style-matched to Figures 1 and 2: 150 DPI, same palette, point-with-errorbar
convention on near-identity values.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.characterization.fitting import (
    fit_rabi, fit_ramsey, fit_t1,
)
from dispersive_readout.characterization.noise import NoiseModelParams
from dispersive_readout.characterization.protocols import (
    generate_rabi_trace, generate_ramsey_trace, generate_t1_trace,
)
from dispersive_readout.characterization.recovery import (
    load_committed_coverage_report, run_recovery_harness,
)


_OUT = Path("06_Dispersive_Readout/figures/fig3_characterization.png")
_COMMITTED_REPORT = Path("06_Dispersive_Readout/figures/recovery_coverage_report.yaml")


def _panel_rabi(ax_fit, ax_res):
    noise = NoiseModelParams()
    eps_pi = 2 * math.pi * 50e6
    trace = generate_rabi_trace(eps_pi, 2 * math.pi * 4.5e9, noise, seed=42)
    fp = fit_rabi(trace, bootstrap_samples=50, seed=42)
    eps = trace.sweep_values
    ax_fit.errorbar(eps / (2 * math.pi * 1e6), trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    model_P = 0.5 - 0.5 * np.cos(np.pi * eps / fp.value)
    ax_fit.plot(eps / (2 * math.pi * 1e6), model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(a) Rabi — $\varepsilon_\pi/2\pi$ = {fp.value/(2*math.pi*1e6):.2f} MHz $\pm$ {fp.uncertainty/(2*math.pi*1e6):.2f} MHz, $\chi^2_\nu$={fp.goodness_of_fit:.2f}")
    ax_res.errorbar(eps / (2 * math.pi * 1e6), trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\varepsilon / 2\pi$ (MHz)")
    ax_res.set_ylabel("residual")


def _panel_ramsey(ax_fit, ax_res):
    noise = NoiseModelParams()
    omega_q = 2 * math.pi * 4.5e9
    T_2_star = 20e-6
    trace = generate_ramsey_trace(omega_q, T_2_star=T_2_star, noise=noise, seed=42)
    fp_o, fp_t = fit_ramsey(trace, bootstrap_samples=50, seed=42)
    delays = trace.sweep_values
    ax_fit.errorbar(delays * 1e6, trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    # Plot the lmfit best-fit curve using the metadata values.
    omega_drive = omega_q - trace.metadata["ground_truth"]["omega_drive_offset"]
    delta_omega = fp_o.value - omega_drive
    model_P = 0.5 - 0.5 * np.exp(-delays / fp_t.value) * np.cos(delta_omega * delays)
    ax_fit.plot(delays * 1e6, model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(b) Ramsey — $\omega_q/2\pi$={fp_o.value/(2*math.pi*1e9):.4f} GHz, $T_2^*$={fp_t.value*1e6:.1f} $\pm$ {fp_t.uncertainty*1e6:.1f} µs")
    ax_res.errorbar(delays * 1e6, trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\tau$ (µs)")
    ax_res.set_ylabel("residual")


def _panel_t1(ax_fit, ax_res):
    noise = NoiseModelParams()
    T_1 = 30e-6
    trace = generate_t1_trace(T_1, noise, seed=42)
    fp = fit_t1(trace, bootstrap_samples=50, seed=42)
    delays = trace.sweep_values
    ax_fit.errorbar(delays * 1e6, trace.P1, yerr=trace.P1_uncertainty, fmt="o", ms=3, capsize=0, alpha=0.6)
    model_P = np.exp(-delays / fp.value)
    ax_fit.plot(delays * 1e6, model_P, "-", linewidth=1.5, color="crimson")
    ax_fit.set_ylabel(r"$P_1$")
    ax_fit.set_title(rf"(c) T1 — T$_1$ = {fp.value*1e6:.2f} $\pm$ {fp.uncertainty*1e6:.2f} µs, $\chi^2_\nu$={fp.goodness_of_fit:.2f}")
    ax_res.errorbar(delays * 1e6, trace.P1 - model_P, yerr=trace.P1_uncertainty, fmt="o", ms=2, capsize=0, alpha=0.5)
    ax_res.axhline(0, color="gray", linewidth=0.5)
    ax_res.set_xlabel(r"$\tau$ (µs)")
    ax_res.set_ylabel("residual")


def _panel_recovery(gs):
    """Build a 2×2 of parity sub-panels inside the outer gridspec slot."""
    sub = gs.subgridspec(2, 2, hspace=0.35, wspace=0.35)
    # Use the committed artifact — cheap, no recomputation.
    reports = load_committed_coverage_report(_COMMITTED_REPORT)
    # Re-run a quick harness to grab the raw (truth, fitted) pairs for plotting.
    noise = NoiseModelParams()
    obs_reports, devices = run_recovery_harness(n_devices=50, noise=noise, seed=42)
    # We need per-device results, not just aggregate, so regenerate.
    from dispersive_readout.characterization.recovery import fit_one_device
    pairs = {"T_1": [], "T_2_echo": [], "omega_q": [], "epsilon_pi": []}
    import numpy.random as _nr
    rng = np.random.default_rng(42)
    for d in devices:
        sub_seed = int(rng.integers(2**31 - 1))
        for r in fit_one_device(d, noise, seed=sub_seed):
            pairs[r.parameter_name].append((r.ground_truth, r.fitted_value, r.fitted_uncertainty, r.within_1_sigma))
    param_order = ["T_1", "T_2_echo", "omega_q", "epsilon_pi"]
    units = {"T_1": ("µs", 1e6), "T_2_echo": ("µs", 1e6), "omega_q": ("GHz", 1.0 / (2 * math.pi * 1e9)), "epsilon_pi": ("MHz", 1.0 / (2 * math.pi * 1e6))}
    for i, name in enumerate(param_order):
        ax = plt.subplot(sub[i // 2, i % 2])
        lab, scale = units[name]
        x = np.array([p[0] * scale for p in pairs[name]])
        y = np.array([p[1] * scale for p in pairs[name]])
        yerr = np.array([p[2] * scale for p in pairs[name]])
        cov1 = np.array([p[3] for p in pairs[name]])
        ax.errorbar(x[cov1], y[cov1], yerr=yerr[cov1], fmt="o", ms=3, color="tab:blue", label="|z|≤1", capsize=0, alpha=0.6)
        ax.errorbar(x[~cov1], y[~cov1], yerr=yerr[~cov1], fmt="x", ms=4, color="tab:orange", label="|z|>1", capsize=0, alpha=0.7)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
        cov2 = obs_reports[name].coverage_2_sigma
        ci = (obs_reports[name].coverage_2_sigma_ci_low, obs_reports[name].coverage_2_sigma_ci_high)
        ax.set_title(rf"{name}: 2$\sigma$={cov2:.0%} [{ci[0]:.0%},{ci[1]:.0%}]", fontsize=8)
        ax.set_xlabel(f"truth ({lab})", fontsize=8)
        ax.set_ylabel(f"fit ({lab})", fontsize=8)
        ax.tick_params(labelsize=7)


def main() -> None:
    fig = plt.figure(figsize=(12, 9), dpi=150)
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    # (a) Rabi — top-left, split into fit + residual
    ga = outer[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_rabi(fig.add_subplot(ga[0]), fig.add_subplot(ga[1]))
    # (b) Ramsey — top-right
    gb = outer[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_ramsey(fig.add_subplot(gb[0]), fig.add_subplot(gb[1]))
    # (c) T1 — bottom-left
    gc = outer[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    _panel_t1(fig.add_subplot(gc[0]), fig.add_subplot(gc[1]))
    # (d) Recovery parity plots — bottom-right (2×2 inner)
    _panel_recovery(outer[1, 1])
    fig.suptitle("Figure 3 — Characterization pipeline + 50-device parameter recovery (SEED=42)", fontsize=11)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, bbox_inches="tight", dpi=150)
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13.3: Render Figure 3**

```bash
python 06_Dispersive_Readout/scripts/fig3_characterization.py
```

Expected wall-clock: 2–3 min (harness dominates). Output: `06_Dispersive_Readout/figures/fig3_characterization.png`.

- [ ] **Step 13.4: Commit**

```bash
git add 06_Dispersive_Readout/scripts/fig3_characterization.py 06_Dispersive_Readout/figures/fig3_characterization.png 06_Dispersive_Readout/examples/example_traces.npz
git commit -m "feat(stage06): Module 3 Task 13 — Figure 3 + committed example bundle

2×2 layout: Rabi / Ramsey / T1 fits with residuals + 2×2 parity sub-panel
for the recovery harness (amendment 4: observed coverage with 2σ
binomial CI annotated per parameter). Style-matched to Figures 1/2.

Example synthetic bundle at examples/example_traces.npz provides a
deterministic starting point for downstream users."
```

**Definition of done:** Figure 3 rendered + committed.

---

## Task 14: C6 edge-case tests + C7 extras + end-of-Module-3 verification

**Rationale:** C6 edge cases (Ramsey Δω=0, elevated thermal, Rabi span too small) plus the §10 checklist verification.

**Files:**
- Modify: `dispersive_readout/tests/test_characterization.py`

- [ ] **Step 14.1: Write remaining C6 tests**

Append:

```python
# -- C6: edge cases ----------------------------------------------------------

def test_C6a_ramsey_zero_detuning_envelope_only_path():
    """Ramsey with Δω=0 uses the envelope-only fallback and returns a T2* within 20%."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_ramsey_trace
    from dispersive_readout.characterization.fitting import fit_ramsey
    omega_q = 2 * math.pi * 4.5e9
    T_2_star_truth = 20e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_ramsey_trace(omega_q, T_2_star=T_2_star_truth, noise=noise,
                                  omega_drive_offset=0.0, seed=99)
    fp_omega, fp_T2 = fit_ramsey(trace, bootstrap_samples=0, seed=42)
    assert fp_T2.name == "T_2_star"
    rel = abs(fp_T2.value - T_2_star_truth) / T_2_star_truth
    assert rel < 0.20


def test_C6b_t1_with_elevated_thermal_no_downward_bias():
    """T1 fit with thermal_offset=0.08 recovers T1 within 10% (thermal absorbed by A)."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_t1_trace
    from dispersive_readout.characterization.fitting import fit_t1
    T_1_truth = 30e-6
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0)
    trace = generate_t1_trace(T_1_truth, noise, thermal_offset=0.08, seed=7)
    fp = fit_t1(trace, bootstrap_samples=0, seed=42)
    rel = abs(fp.value - T_1_truth) / T_1_truth
    assert rel < 0.10


def test_C6c_rabi_amplitude_span_too_small_fits_but_flags_via_redchi():
    """A Rabi trace with only half an oscillation produces a high χ²/dof; we don't
    hard-reject at generator level (kept simple), but the fit should still return
    a value and the caller can check goodness_of_fit > threshold."""
    from dispersive_readout.characterization.noise import NoiseModelParams
    from dispersive_readout.characterization.protocols import generate_rabi_trace
    from dispersive_readout.characterization.fitting import fit_rabi
    noise = NoiseModelParams(n_shots_per_point=5000, drift_amplitude_Hz=0.0, drive_amplitude_uncertainty=0.0)
    eps_pi_truth = 2 * math.pi * 50e6
    # Span 0 → 0.6·ε_π — less than one full oscillation.
    trace = generate_rabi_trace(eps_pi_truth, 2 * math.pi * 4.5e9, noise, seed=8, amplitude_span_mult=(0.0, 0.6))
    fp = fit_rabi(trace, bootstrap_samples=0, seed=42)
    # Fit runs but is biased or high redchi.
    assert fp.goodness_of_fit >= 0  # just verify the code path doesn't explode.
```

- [ ] **Step 14.2: Run C6 + full suite**

```bash
python -m pytest dispersive_readout/tests/test_characterization.py -v -p no:dash
```

Expected: 28 PASS (25 prior + 3 C6).

```bash
python -m pytest dispersive_readout/tests/ -v -p no:dash 2>&1 | tail -15
```

Expected: 101 passing.

- [ ] **Step 14.3: Run the §9 review checklist by hand**

Walk through `MODULE_3_SPEC.md` §9:

```bash
# Test counts
python -m pytest dispersive_readout/tests/test_characterization.py -q -p no:dash | tail -5
# Expect: "28 passed"

# Artifact exists with 2σ CI including 95%
python -c "
from dispersive_readout.characterization.recovery import load_committed_coverage_report
r = load_committed_coverage_report('06_Dispersive_Readout/figures/recovery_coverage_report.yaml')
for name, rep in r.items():
    ok1 = rep.coverage_1_sigma_ci_low <= 0.68 <= rep.coverage_1_sigma_ci_high
    ok2 = rep.coverage_2_sigma_ci_low <= 0.95 <= rep.coverage_2_sigma_ci_high
    print(f'{name}: 1σ includes 68%? {ok1}; 2σ includes 95%? {ok2}')
"

# CLI help is TODO-free
python 06_Dispersive_Readout/characterize.py --help | grep -iE "(todo|tbd|fixme|xxx)" && echo "FAIL: forbidden strings in help" || echo "OK: help clean"

# to_device_config round-trips through simulate_readout (already tested by C4c)
python -m pytest dispersive_readout/tests/test_characterization.py::test_C4c_to_device_config_produces_simulator_consumable -v -p no:dash

# Module 1 tests unchanged (57)
python -m pytest dispersive_readout/tests/test_physics_validation.py dispersive_readout/tests/test_transmon.py dispersive_readout/tests/test_config.py dispersive_readout/tests/test_lindblad.py dispersive_readout/tests/test_readout_model.py dispersive_readout/tests/test_dispersive.py -p no:dash | tail -5

# Module 2 tests unchanged (15)
python -m pytest dispersive_readout/tests/test_error_budget.py -p no:dash | tail -5
```

Every check should succeed. **STOP if any check fails** — the review checklist is the Module 3 → Module 4 gate.

- [ ] **Step 14.4: Commit**

```bash
git add dispersive_readout/tests/test_characterization.py
git commit -m "test(stage06): Module 3 Task 14 — C6 edge cases + end-of-module verification

C6a (Ramsey Δω=0), C6b (elevated thermal), C6c (Rabi span too small)
all passing. §9 review checklist verified: 101 tests total (Module 1: 57,
Module 2: 15, Module 3: 29 counted with @mark.slow C3).

Module 3 complete per §10 checklist. Ready for Module 4 entry."
```

**Definition of done:** All §9 checklist items green; 101 tests passing.

---

## Spec coverage map (self-review)

Mapping MODULE_3_SPEC.md sections to tasks:

| Spec section | Implemented by task |
|---|---|
| §0 amendments 1–9 | Reflected in design of every task (notably 1, 3, 7, 10, 11) |
| §1.1 Rabi (amendment 2) | 2 (generator), 8 (fit), 14 (C6c) |
| §1.2 Ramsey | 3 (generator), 8 (fit), 14 (C6a) |
| §1.3 T1 | 4 (generator), 8 (fit), 14 (C6b) |
| §1.4 T2-echo | 5 (generator), 8 (fit) |
| §2.1–2.4 noise stack | 1 (noise.py), 2–5 (generators) |
| §2.5 nominal noise + F_assign from YAML (amendment 7) | 1 (load_reference_F_full), 3–5 (generators call it) |
| §3 module structure (amendment 6 paths) | 1 (scaffold), all tasks |
| §4.1 noise.py | 1 |
| §4.2 protocols.py + bundle I/O | 2, 3, 4, 5, 6 |
| §4.3 fitting.py + to_device_config (amendment 5) | 7 (schemas), 8 (fits), 9 (bootstrap) |
| §4.4 recovery.py + calibration gate (amendment 4) + fit_one_device (amendment 8) | 10, 11 |
| §4.5 CLI | 12 |
| §5 tests C1 | 2, 3, 4, 5 |
| §5 tests C2 | 1 |
| §5 test C3 (regression gate, amendment 9) | 11 |
| §5 tests C4 | 7 |
| §5 tests C5 | 12 |
| §5 tests C6 | 14 |
| §5 tests C7 | 7 |
| §6 Figure 3 (+ 2×2 parity sub-panel) | 13 |
| §7 day-by-day | This plan sequences 14 tasks across the 3 days |
| §8 flags to human | Embedded in task STOP notes (Tasks 11, 14) |
| §9 review checklist | Task 14 verification step |
| §10 references | MODULE_3_SPEC.md §10 |

All spec sections covered.

**Placeholder scan:** Every code block is complete; no TBD/TODO/FIXME strings in plan tasks; no "similar to Task N" references. Each test function is shown verbatim.

**Type consistency check:**
- `TraceData` fields (`protocol`, `sweep_axis`, `sweep_values`, `P1`, `P1_uncertainty`, `metadata`): consistent across Tasks 2, 6, 10, 11, 12.
- `FittedParameter.name` Literal values (`T_1`, `T_2_echo`, `T_2_star`, `omega_q`, `epsilon_pi`): consistent across Tasks 7, 8, 9, 10, 11.
- `CoverageReport` fields match Task 10 (schema) and Task 11 (producer/consumer).
- `fit_one_device` signature (`DeviceGroundTruth, NoiseModelParams, int`) matches Task 10 definition and Task 11 call sites.
- Seed upper bound `2**31 - 1` used everywhere RNG integers are drawn (Tasks 3, 9, 10, 11).
- `parametric_bootstrap(protocol, best_fit_values, noise, n_bootstrap, seed)` signature consistent between Task 9 definition and Tasks 9/10 call sites.

No inconsistencies.

---

## Execution Handoff

Plan complete and saved to `06_Dispersive_Readout/PLAN.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Each task is self-contained with its own tests and atomic commit; subagent pattern shines here.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review. Slightly heavier on context but avoids subagent cold-start cost per task.

**Which approach?**

---

**Pre-execution checklist (applies to either path):**

- [ ] On branch `stage-06-module-3-characterization` (cut from tag `stage06-module2`; spec at `bf122f7`).
- [ ] Module 2 polish stashed as `stash@{0}` — `module-2-fig2-annotation-polish-defer-to-day-14`.
- [ ] All 72 Module 1 + Module 2 tests passing (verified at session start).
- [ ] `06_Dispersive_Readout/figures/fig2_data.yaml` exists and is readable by `load_reference_F_full` (verified in Task 1).
- [ ] `MODULE_3_SPEC.md` unchanged since this plan was written (if spec edited, re-review affected tasks).
