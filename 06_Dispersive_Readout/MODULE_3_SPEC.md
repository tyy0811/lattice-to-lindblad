# Stage 06 Module 3 — Characterization Interface

**Status:** amended design, 2026-04-20. Supersedes the original Module 3 draft pasted into the brainstorming session on the same date.

**Goal.** A command-line tool an experimentalist could run: consumes synthetic characterization traces (or real device data in the same format), returns fitted device parameters with uncertainties, validates recovery across 50 synthetic devices, and produces a YAML parameter pack compatible with Module 1's `DeviceConfig`. Closes the characterization-data-analysis gap in the JD.

**Budget.** 3 working days (days 7–9 in the plan timeline).

**Prerequisites.** Module 1 simulator working (V1a, V2, V3, V4a/b passing); Module 2 shipped at tag `stage06-module2` with `ErrorBudget` YAML committed at `06_Dispersive_Readout/figures/fig2_data.yaml`; `F_full` at REFERENCE_DEVICE readable from that YAML. Module 1 public API (`dispersive_readout.physics`) exposes `DeviceConfig`, `DriveParams`, `DecoherenceParams`, `REFERENCE_DEVICE`.

**Scope lock.** Physics decisions below are locked post-amendment. No expansion to simulation-based inference (SBI), Bayesian MCMC, conformal calibration, non-Markovian drift models, or multi-qubit extensions. Implementation discoveries that challenge a locked decision are raised as blockers, not silently resolved.

---

## 0. Amendments applied to the original spec

Nine substantive blockers were raised during brainstorming and resolved with spec amendments before implementation. The rest of this document is the *post-amendment* spec; this section records what changed and why, so the delta to the original is traceable.

| # | Amendment | Driver |
|---|---|---|
| 1 | Synthetic-trace generator uses **closed-form analytic traces** parameterized by ground-truth (T₁, T₂, ω_q, ε_π), not Module 1's Lindblad `simulate_readout`. Noise stack (shot + 1/f drift + assignment errors + amplitude uncertainty) is layered on top. | The recovery harness is a test of the FITTING PIPELINE, not the qubit dynamics — V3/V4a/V4b already validate dynamics to machine precision. Lindblad-based trace generation would test the simulator twice at ~1000× runtime cost and could hide fitter bugs if simulator and fitter share a common approximation. Closed-form also makes the harness CI-tractable (~1–2 min end-to-end) with no Modal dependency. |
| 2 | Rabi fit form drops the envelope term: `P₁(ε) = A + B·cos(Ω_R(ε)·τ + φ)` at fixed pulse duration τ. | At fixed τ, `exp(−τ/T_R)` is a constant absorbed into B, so T_R is structurally unidentifiable from an amplitude sweep; lmfit returns huge uncertainty on T_R or silently degenerates. T_R extraction requires a different protocol (time sweep or chevron) and is out of Module 3 scope — T_R does not feed Module 4's closed-loop recommendation. |
| 3 | Uncertainty quantification uses **parametric bootstrap** (regenerate fresh 1/f drift + shot-noise realizations around the best-fit params, re-fit each realization), not iid residual bootstrap. | iid residual bootstrap assumes residuals are iid, but 1/f drift is correlated across a scan — residual bootstrap under-covers by ~(drift correlation length / n_points), landing observed 2σ coverage at ~60–70% rather than 95%. Parametric bootstrap reuses the synthetic generator's closed-form noise model as-is; coverage converges to truth as `n_bootstrap → ∞`. |
| 4 | Coverage gate is **binomial-CI-based**, not a point-estimate threshold: gate fires when 95% falls outside the 2σ binomial CI of observed coverage. | Original gate `≥90%` at n=50 has 2σ binomial SE ≈ 3.1%, so the 2σ CI of a true-95% pipeline runs [89%, 100%]. A correctly-calibrated pipeline fails a point-estimate gate in ~15% of runs from pure sampling noise. Report both the observed coverage and the 2σ CI so a reviewer can see whether a low number is significant or binomial-noise. |
| 5 | `to_device_config()` has explicit consistency policy: **E_C held fixed** at REFERENCE_DEVICE (geometric, not fit by Rabi/Ramsey/T1/T2); **E_J back-solved** from fitted ω_q via `E_J = (ω_q + E_C)² / (8·E_C)` (Koch 2007 deep-transmon approximation); Ramsey ω_q replaces REFERENCE's; T1/T2 from fits; resonator, coupling, and truncation from REFERENCE defaults. Raises a warning if derived E_J differs from REFERENCE's by > 30%. | Fitted ω_q encodes (E_C, E_J) via ω_01 ≈ √(8·E_J·E_C) − E_C. Without an explicit policy, `to_device_config` would either ship inconsistent (E_C, E_J, ω_q) triples to Module 1's simulator, or silently use stale defaults. Module 4's closed-loop device recommendation requires a well-defined bridge. |
| 6 | Package paths corrected: `dispersive_readout/characterization/` (not `stage_06_readout/...`). CLI entry is `python 06_Dispersive_Readout/characterize.py` (matches Module 1/2 script convention). All imports: `from dispersive_readout.physics.config import DeviceConfig`, etc. | Pre-refactor path artifact in the draft. Mechanical. |
| 7 | Noise model's `readout_assignment_fidelity` is read at runtime from Module 2's committed YAML (`06_Dispersive_Readout/figures/fig2_data.yaml` → `F_full`), not hardcoded. | Original §2.5 hardcoded `0.99` contradicted §2.3 ("Use F_assign from Module 2's reference operating point"). Reading from the committed YAML gives cross-module consistency and a free regression detector: if a Module 1 refactor changes `F_full`, Module 3's noise model picks up the change on the next run. |
| 8 | Recovery harness built around the pure function `fit_one_device(device_params, noise, seed) → RecoveryResult`. Serial fallback: `list(map(fit_one_device, ...))`; Modal: `fit_one_device.map(...)`. Zero code change between modes. | Original `fit_all(traces)` is the wrong boundary for map-compatible parallelism. Per-device (generate + fit + compare) is the unit of independent work. The pure-function constraint also forces no-global-state / no-filesystem-side-effects discipline from the start, which is non-trivial to retrofit after the fact. |
| 9 | CI gate is **full 50-device regeneration at fixed `SEED=42`**, compared to a committed `recovery_coverage_report.yaml` within ±2% per parameter. If the fitter legitimately improves, regenerate and re-commit the artifact. | Decoupling claim-in-figure (50 devices) from gate-in-CI (20 devices) would let the fitter silently drift post-ship without CI catching it. Committed-artifact comparison collapses the figure claim and the CI check into a single source of truth. At n=50 with closed-form trace generation the full harness runs in 1–2 min, CI-tractable. The ±2% tolerance is binomial-noise-appropriate at n=50. |

---

## 1. Protocols implemented

Four standard single-qubit characterization protocols. For each: physical observable, drive sequence, closed-form generator, fitting form, parameters extracted. The generator and fitter use the **same** analytic form (per amendment 1) — the recovery harness tests the fitter's statistical behavior under the noise stack in §2, not its ability to invert a different physical model.

### 1.1 Rabi oscillation

**Purpose.** Calibrate the π-pulse amplitude `ε_π`.

**Drive sequence.** Single Gaussian drive pulse on resonance with ω_q, amplitude swept from 0 to a maximum covering at least 1.5 full Rabi oscillations (default span: 0 → 2.5 · ε_π). Fixed pulse duration `τ_rabi = 20 ns`. Dispersive readout afterwards.

**Observable.** Final |1⟩ population `P₁` vs drive amplitude ε.

**Generator / fit form (amendment 2):**

```
P₁(ε) = A + B · cos(π · ε / ε_π + φ)
```

Free parameters: `A`, `B`, `ε_π`, `φ`. Extracted parameter: `ε_π` with uncertainty. The envelope decay `exp(−τ/T_R)` is absorbed into `B` at fixed τ and is not identifiable from an amplitude sweep — it is not a fit parameter. See amendment 2 for rationale.

**Edge case.** If drive amplitudes are too small, only one-half oscillation is visible and the fit degenerates. Reject traces with fewer than 1.5 oscillations visible (detected by peak-counting on a smoothed trace) and flag in the output.

### 1.2 Ramsey interferometry

**Purpose.** Extract qubit frequency ω_q and T2\* (free-induction dephasing time).

**Drive sequence.** π/2 pulse → free evolution for delay τ → π/2 pulse → dispersive readout. Delay τ swept from 0 to several T2\*.

**Observable.** P₁ vs τ.

**Generator / fit form:**

```
P₁(τ) = A + B · exp(−τ / T_2*) · cos(Δω · τ + φ)
```

where `Δω = ω_q − ω_drive`. Free parameters: `A`, `B`, `Δω`, `T_2*`, `φ`. Extracted parameters: `ω_q = ω_drive + Δω` and `T_2*`.

**Edge case.** If `Δω → 0` the cosine degenerates and T2\* can only be extracted from the envelope. Fitter must handle this limit gracefully: if initial-guess oscillation-counting returns < 1 oscillation, switch to envelope-only fit (cos term pinned to 1, Δω fixed at 0). The recovery harness includes one device with Δω = 0 to exercise this path.

### 1.3 T1 relaxation

**Purpose.** Extract T1.

**Drive sequence.** π pulse → free evolution for delay τ → dispersive readout. Delay τ swept from 0 to ≥ 3·T1.

**Observable.** P₁ vs τ.

**Generator / fit form:**

```
P₁(τ) = A + B · exp(−τ / T_1)
```

Free parameters: `A`, `B`, `T_1`. The offset `A` absorbs the steady-state thermal population and readout-assignment offset.

**Edge case.** Elevated thermal population (n̄_th > 0.05) biases a naive exponential fit low in T1. Recovery harness includes one device with n̄_th = 0.08 to exercise this.

### 1.4 Hahn echo (T2-echo)

**Purpose.** Extract T2 (echo-refocused dephasing time, insensitive to low-frequency noise).

**Drive sequence.** π/2 → τ/2 free → π → τ/2 free → π/2 → dispersive readout.

**Observable.** P₁ vs τ.

**Generator / fit form:**

```
P₁(τ) = A + B · exp(−τ / T_2)
```

Default exponential (n=1). Stretched-exponential `exp(−(τ/T_2)^n)` with free n is offered as a fallback if the plain fit has reduced chi-squared χ²/dof > 3. Extracted parameter: T2.

---

## 2. Noise model

Synthetic traces are generated from the closed forms in §1 with the following noise stack applied on top.

### 2.1 Shot noise

Each point is the mean of `n_shots` single-shot measurements. Per-point variance:

```
σ²_shot(P) = P(1 − P) / n_shots
```

Default `n_shots = 2000` per point (Marxer 2508.16437 regime). Recovery harness varies this from 500 to 5000 to test robustness.

### 2.2 1/f qubit-frequency drift

During a full protocol scan (one realization per protocol, not per point), the qubit frequency drifts on a 1/f spectrum. Implementation:

```python
def generate_1f_drift(
    n_points: int,
    amplitude_Hz: float,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return n_points samples of a 1/f^alpha process with rms `amplitude_Hz`.

    Single realization per scan — drift is CORRELATED across scan points.
    """
```

The correlation is what makes iid residual bootstrap wrong (amendment 3). The drift enters the Ramsey generator as `Δω(τ_k) = Δω_nominal + δω_1f[k]`; it enters T1/T2/Rabi generators as a small amplitude perturbation via matrix-element rescaling — see §4.2.

### 2.3 Readout assignment errors

For each shot, the measured outcome is corrupted by a classical bit-flip with probability `1 − F_assign`, where `F_assign` is read at runtime from Module 2's committed `ErrorBudget` YAML (amendment 7):

```python
def load_reference_F_full() -> float:
    """Load F_full at REFERENCE_DEVICE from Module 2's committed artifact."""
    import yaml
    path = "06_Dispersive_Readout/figures/fig2_data.yaml"
    with open(path) as f:
        budget = yaml.safe_load(f)
    return float(budget["F_full"])
```

Asymmetric readout errors with `P(0|1) ≠ P(1|0)` remain in the `NoiseModelParams` API surface but are **V1-scope deferred**: setting `readout_asymmetric=True` raises `NotImplementedError` at trace generation (all four generators propagate the flag through `apply_readout_errors`, which is the gate that rejects it). This was a post-implementation tightening driven by the Codex adversarial review (finding #2, 2026-04-20): the original wording "available as an option" was ambiguous enough that generators could silently produce symmetric traces under an asymmetric config. V1's contract is now explicit — symmetric-only until V2, with an audible failure on misuse. Implementing asymmetric is a small V2 extension (two extra parameters `P_01`, `P_10` on `NoiseModelParams`, plumbed through `apply_readout_errors` + generators + bootstrap) but was not justified against visa-deadline constraints, and does not affect the recovery numbers at the reference operating point.

### 2.4 Amplitude calibration uncertainty

The nominal drive amplitude has a ±5% Gaussian calibration uncertainty, realized once per protocol run (not per point). This shifts the inferred `ε_π` in Rabi fitting but does not affect T1/T2/Ramsey (whose observables are delay-swept, not amplitude-swept).

### 2.5 Nominal noise budget

Default noise-model parameters for the canonical synthetic-trace generator:

```python
# characterization/noise.py

from dataclasses import dataclass

@dataclass(frozen=True)
class NoiseModelParams:
    n_shots_per_point: int = 2000
    drift_amplitude_Hz: float = 1e4
    drift_alpha: float = 1.0
    drift_seed: int | None = None          # fresh per-run; reproducible via harness seed
    # readout_assignment_fidelity is NOT a dataclass field — read from
    # Module 2 YAML at trace-generation time (amendment 7)
    readout_asymmetric: bool = False
    drive_amplitude_uncertainty: float = 0.05
```

`F_assign` lives outside the dataclass so it cannot be accidentally pinned to a stale value in serialized runs.

---

## 3. Module structure

Paths corrected per amendment 6:

```
dispersive_readout/
├── characterization/
│   ├── __init__.py                   # public API
│   ├── noise.py                      # NoiseModelParams + noise-injection helpers
│   ├── protocols.py                  # Closed-form synthetic-trace generators
│   ├── fitting.py                    # lmfit-based parameter extraction
│   ├── recovery.py                   # Parameter recovery harness (G2) — fit_one_device
│   └── cli.py                        # CLI entry (invoked via 06_Dispersive_Readout/characterize.py)
└── tests/
    └── test_characterization.py      # All Module 3 tests (C1–C6)

06_Dispersive_Readout/
├── characterize.py                   # thin script: `python 06_Dispersive_Readout/characterize.py ...`
│                                     # imports and runs dispersive_readout.characterization.cli.main
├── scripts/
│   └── fig3_characterization.py      # Figure 3 driver
└── figures/
    ├── fig3_characterization.png
    └── recovery_coverage_report.yaml  # committed artifact for CI (amendment 9)
```

---

## 4. Detailed specifications

### 4.1 `characterization/noise.py`

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml


_REFERENCE_F_FULL_PATH = Path("06_Dispersive_Readout/figures/fig2_data.yaml")


@dataclass(frozen=True)
class NoiseModelParams:
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
    alpha: float,
    seed: int | None,
) -> np.ndarray:
    """Generate an n-point 1/f^alpha realization with rms `amplitude_Hz`.

    Method: draw white Gaussian samples in the frequency domain with amplitude
    proportional to f^(-alpha/2), inverse-FFT, rescale to target rms.
    """
    ...


def apply_shot_noise(
    P_true: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial shot-noise sampling. Returns observed P = k/n_shots."""
    k = rng.binomial(n_shots, np.clip(P_true, 0.0, 1.0))
    return k / n_shots


def apply_readout_errors(
    P_observed: np.ndarray,
    F_assign: float,
    asymmetric: bool = False,
) -> np.ndarray:
    """Classical bit-flip readout-error model; symmetric by default."""
    if asymmetric:
        raise NotImplementedError("Asymmetric readout errors are a Module 3 follow-up.")
    p_flip = 1.0 - F_assign
    return (1.0 - p_flip) * P_observed + p_flip * (1.0 - P_observed)
```

### 4.2 `characterization/protocols.py`

Closed-form generators (amendment 1). Each takes ground-truth parameters and noise config; returns a `TraceData` with the full noise stack applied.

```python
from dataclasses import dataclass
import numpy as np
from .noise import NoiseModelParams, generate_1f_drift, apply_shot_noise, apply_readout_errors, load_reference_F_full


@dataclass(frozen=True)
class TraceData:
    protocol: str                        # "rabi" | "ramsey" | "t1" | "t2_echo"
    sweep_axis: str                      # "drive_amplitude" | "delay"
    sweep_values: np.ndarray             # (N,)
    P1: np.ndarray                       # (N,) measured |1⟩ population
    P1_uncertainty: np.ndarray           # (N,) per-point standard error (shot-only)
    metadata: dict                       # ground truth + noise config (synthetic) or device ID (real)


def generate_rabi_trace(
    epsilon_pi: float,                   # ground-truth π-amplitude (rad/s)
    omega_q: float,                      # ground-truth qubit frequency (rad/s)
    noise: NoiseModelParams,
    n_points: int = 101,
    amplitude_span: tuple[float, float] = (0.0, 2.5),   # in units of epsilon_pi
    seed: int | None = None,
) -> TraceData:
    """Closed-form Rabi: P₁(ε) = A + B·cos(π·ε/ε_π·(1+δ_amp) + φ), A≈B≈0.5, φ=0.

    Noise stack: (a) 1/f drift perturbs the effective ε_π (small); (b) shot noise
    per point; (c) symmetric readout errors; (d) one-shot amplitude calibration
    offset δ_amp ~ N(0, drive_amplitude_uncertainty).
    """
    ...


def generate_ramsey_trace(
    omega_q: float,                      # ground-truth qubit frequency (rad/s)
    T_2_star: float,                     # ground-truth T2*
    noise: NoiseModelParams,
    omega_drive_offset: float = 2 * np.pi * 1e6,   # deliberate detuning to see oscillation
    n_points: int = 101,
    delay_range: tuple[float, float] = (0.0, 40e-6),
    seed: int | None = None,
) -> TraceData:
    """Closed-form Ramsey with 1/f drift rolled into Δω(τ_k) = Δω_nom + δω_1f[k]."""
    ...


def generate_t1_trace(
    T_1: float,
    noise: NoiseModelParams,
    n_points: int = 51,
    delay_range: tuple[float, float] = (0.0, 100e-6),
    thermal_offset: float = 0.0,         # A in A + B·exp(−τ/T1); 0.0 = ideal cold
    seed: int | None = None,
) -> TraceData: ...


def generate_t2_echo_trace(
    T_2: float,
    noise: NoiseModelParams,
    n_points: int = 51,
    delay_range: tuple[float, float] = (0.0, 120e-6),
    seed: int | None = None,
) -> TraceData: ...


def save_trace_bundle(traces: list[TraceData], path: str) -> None:
    """Save a bundle of traces as .npz (the CLI-consumable format)."""
    ...


def load_trace_bundle(path: str) -> list[TraceData]:
    """Load a .npz trace bundle with schema validation.

    Rejects bundles without the required fields (amendment 8 / §8 flag #5 below).
    """
    ...
```

### 4.3 `characterization/fitting.py`

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from pydantic import BaseModel, field_validator
import lmfit
import numpy as np


class FittedParameter(BaseModel):
    name: str                          # "T_1" | "omega_q" | "epsilon_pi" | "T_2_star" | "T_2"
    value: float
    uncertainty: float                 # 1-sigma, from parametric bootstrap (amendment 3)
    unit: str                          # "s" | "rad/s"
    protocol_source: str               # "rabi" | "ramsey" | "t1" | "t2_echo"
    goodness_of_fit: float             # reduced chi-squared of point-estimate fit
    n_bootstrap: int                   # number of parametric-bootstrap samples used

    @field_validator("uncertainty")
    @classmethod
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("uncertainty must be positive")
        return v


class ExtractedParameterPack(BaseModel):
    fitted_parameters: list[FittedParameter]
    trace_file: str
    timestamp: str                     # ISO8601
    stage_06_version: str              # git SHA at run time

    def to_device_config(self, E_C_default: float | None = None):
        """Bridge fitted parameters to a Module-1-compatible DeviceConfig (amendment 5).

        Policy:
          - E_C: held fixed at REFERENCE_DEVICE (geometric, not fit by this module).
          - E_J: back-solved from fitted ω_q via E_J = (ω_q + E_C)² / (8·E_C)
                 (Koch 2007 deep-transmon approximation).
          - ω_q: from Ramsey fit (ω_drive + Δω).
          - γ_1: 1/T_1 from T1 fit.
          - γ_phi: from T2 fit via 1/T_2 = γ_1/2 + γ_phi.
          - resonator, coupling, truncation: from REFERENCE_DEVICE defaults.

        Raises a warning (not an error) if derived E_J differs from REFERENCE's by > 30%.
        Logs the policy in the output YAML for traceability.
        """
        ...


def fit_rabi(trace: TraceData, bootstrap_samples: int = 200, seed: int | None = None) -> FittedParameter: ...
def fit_ramsey(trace: TraceData, bootstrap_samples: int = 200, seed: int | None = None) -> tuple[FittedParameter, FittedParameter]: ...
def fit_t1(trace: TraceData, bootstrap_samples: int = 200, seed: int | None = None) -> FittedParameter: ...
def fit_t2_echo(trace: TraceData, use_stretched_exponential: bool = False, bootstrap_samples: int = 200, seed: int | None = None) -> FittedParameter: ...


def parametric_bootstrap(
    protocol: str,
    best_fit_params: dict,               # point-estimate params (used as bootstrap ground truth)
    noise: NoiseModelParams,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Parametric bootstrap (amendment 3): regenerate fresh (drift + shot + readout) noise
    realizations around best-fit params, re-fit each, return the spread.

    Correct under correlated 1/f drift, unlike iid residual bootstrap which
    under-covers by ~(drift correlation length / n_points).

    Returns a dict keyed by parameter name with an (n_bootstrap,) array of fits.
    The 1σ uncertainty is reported as the standard deviation of this array.
    """
    ...


def fit_all(
    traces: list[TraceData],
    bootstrap_samples: int = 200,
    seed: int | None = None,
) -> ExtractedParameterPack:
    """Fit all traces in a bundle; return a Module-1-compatible parameter pack."""
    ...
```

**Physics decisions locked (post-amendment):**

- **Parametric bootstrap, not residual bootstrap** (amendment 3). The bootstrap regenerates full noise realizations from the best-fit point estimate; correct under correlated 1/f drift.
- **Fixed parameter bounds.** T1, T2, T2\* ∈ [0.1 µs, 1 ms]. ω_q ∈ [2π · 1 GHz, 2π · 10 GHz]. ε_π ∈ [2π · 1 MHz, 2π · 1 GHz]. Out-of-bound best fits raise with a diagnostic message.
- **Reduced chi-squared reported.** Traces with χ²/dof > 5 are flagged "poor fit" but still return a value with the (possibly inflated) bootstrap uncertainty.

### 4.4 `characterization/recovery.py` — parameter recovery harness (G2)

The most important deliverable in Module 3. Built around the pure function `fit_one_device` (amendment 8) so the harness runs serially or under Modal with zero code change.

```python
from dataclasses import dataclass, asdict
import numpy as np
import yaml

from .fitting import fit_rabi, fit_ramsey, fit_t1, fit_t2_echo
from .noise import NoiseModelParams
from .protocols import generate_rabi_trace, generate_ramsey_trace, generate_t1_trace, generate_t2_echo_trace


@dataclass(frozen=True)
class DeviceGroundTruth:
    """One synthetic device's ground-truth parameters."""
    T_1: float
    T_2_echo: float
    omega_q: float
    epsilon_pi: float
    thermal_offset: float = 0.0
    ramsey_detuning: float = 2 * np.pi * 1e6


@dataclass(frozen=True)
class RecoveryResult:
    """One synthetic-device fit-vs-truth record, one per parameter."""
    parameter_name: str
    ground_truth: float
    fitted_value: float
    fitted_uncertainty: float
    z_score: float                      # (fitted - truth) / uncertainty
    within_1_sigma: bool
    within_2_sigma: bool


@dataclass(frozen=True)
class CoverageReport:
    """Aggregate recovery statistics across n_devices."""
    parameter_name: str
    n_devices: int
    coverage_1_sigma: float
    coverage_2_sigma: float
    coverage_1_sigma_ci_low: float       # 2σ binomial CI lower bound (amendment 4)
    coverage_1_sigma_ci_high: float
    coverage_2_sigma_ci_low: float
    coverage_2_sigma_ci_high: float
    bias: float                          # mean (fitted - truth)
    bias_uncertainty: float              # SE of the bias estimate


def fit_one_device(
    device: DeviceGroundTruth,
    noise: NoiseModelParams,
    seed: int,
) -> list[RecoveryResult]:
    """Pure function: generate 4 traces from ground truth, fit all, compare to truth.

    Modal-compatible (amendment 8): no global state, no filesystem I/O, fully
    deterministic under `seed`. Returns one RecoveryResult per fitted parameter
    (T_1, T_2_echo, omega_q, epsilon_pi).

    Serial:  list(map(lambda d: fit_one_device(d, noise, seed), devices))
    Modal:   fit_one_device.map(devices, [noise]*len(devices), seeds)
    """
    rng = np.random.default_rng(seed)
    # Generate 4 traces
    rabi_trace = generate_rabi_trace(device.epsilon_pi, device.omega_q, noise, seed=rng.integers(2**31 - 1))
    ramsey_trace = generate_ramsey_trace(device.omega_q, T_2_star=device.T_2_echo, noise=noise, seed=rng.integers(2**31 - 1))
    t1_trace = generate_t1_trace(device.T_1, noise, thermal_offset=device.thermal_offset, seed=rng.integers(2**31 - 1))
    t2_trace = generate_t2_echo_trace(device.T_2_echo, noise, seed=rng.integers(2**31 - 1))
    # Fit each
    fp_epsilon = fit_rabi(rabi_trace, seed=int(rng.integers(2**31 - 1)))
    fp_omega, fp_T2star = fit_ramsey(ramsey_trace, seed=int(rng.integers(2**31 - 1)))
    fp_T1 = fit_t1(t1_trace, seed=int(rng.integers(2**31 - 1)))
    fp_T2 = fit_t2_echo(t2_trace, seed=int(rng.integers(2**31 - 1)))
    # Compare to truth
    return [
        _make_recovery_result("T_1", device.T_1, fp_T1),
        _make_recovery_result("T_2_echo", device.T_2_echo, fp_T2),
        _make_recovery_result("omega_q", device.omega_q, fp_omega),
        _make_recovery_result("epsilon_pi", device.epsilon_pi, fp_epsilon),
    ]


def generate_synthetic_device_family(n_devices: int, seed: int) -> list[DeviceGroundTruth]:
    """Log-uniform(T_1, T_2) in [5 µs, 100 µs]; uniform(ω_q/2π) in [4 GHz, 5 GHz];
    ε_π at REFERENCE scale with ±20% jitter; E_C, E_J at REFERENCE (held fixed).

    Physical constraint: Hahn-echo T2 is bounded above by 2·T1. Reject-and-resample
    any device where T_2 > 2·T_1·0.95 (the 0.95 keeps margin for bootstrap
    fluctuations). At log-uniform over a 20× range the rejection rate is ~25%,
    so ~13 oversamples on average to obtain 50 valid devices.

    Two deliberately hard cases included as deterministic overrides (not subject
    to sampling):
      - device[0]: Ramsey at zero detuning (ramsey_detuning = 0)
      - device[1]: elevated thermal (thermal_offset = 0.08)

    RNG stability: numpy occasionally changes default_rng behavior across releases,
    which would perturb the device list at fixed SEED=42 and invalidate the
    committed recovery_coverage_report.yaml (amendment 9). Mitigation: serialize
    the generated device list alongside the coverage report as a YAML field so
    the artifact is self-describing, and pin the numpy minor version in the
    Module 3 environment file.
    """
    ...


def run_recovery_harness(
    n_devices: int = 50,
    noise: NoiseModelParams | None = None,
    seed: int = 42,
) -> dict[str, CoverageReport]:
    """Run the full recovery pipeline across n_devices.

    Returns coverage reports keyed by parameter name. At default args
    (n_devices=50, seed=42) this reproduces the committed artifact
    `recovery_coverage_report.yaml` within ±2% per parameter (amendment 9).
    """
    ...


def save_coverage_report(reports: dict[str, CoverageReport], path: str) -> None:
    """Save coverage reports as YAML (the committed-artifact format for CI)."""
    ...


def load_committed_coverage_report(path: str) -> dict[str, CoverageReport]:
    """Load the committed recovery_coverage_report.yaml for regression comparison."""
    ...


def format_recovery_table(reports: dict[str, CoverageReport]) -> str:
    """Render a markdown table of the recovery statistics for the figure caption / report."""
    ...
```

**Calibration gate (amendment 4), enforced at artifact-commit time.** When generating the committed `recovery_coverage_report.yaml` for the first time (or regenerating it after an intentional fitter improvement), verify for each fitted parameter that the 2σ binomial CI on observed coverage includes the target (68% for 1σ, 95% for 2σ). Equivalently:

```
observed_coverage + 2·sqrt(p·(1−p)/n) ≥ target   AND   observed_coverage − 2·sqrt(p·(1−p)/n) ≤ target
```

At n=50, true-95% coverage → binomial SE ≈ 3.1% → 2σ CI ≈ [89%, 100%]. At true-68% → SE ≈ 6.6% → CI ≈ [55%, 81%]. If either CI fails to include its target, the pipeline is miscalibrated — do not commit the artifact; diagnose instead.

**Regression gate (amendment 9), enforced on every CI run.** Once committed, subsequent CI runs call `run_recovery_harness(n_devices=50, seed=42)` and compare to the committed artifact within ±2% per parameter (test C3 in §5). This catches silent drift in the fitter without re-running the calibration analysis on every commit. Report both the observed coverage and the 2σ CI bounds in every harness output so a reader can distinguish "calibration miss" from "binomial noise" at a glance.

### 4.5 `characterization/cli.py` — experimentalist-facing entry

```
Usage
-----
  python 06_Dispersive_Readout/characterize.py --traces data.npz --output params.yaml
  python 06_Dispersive_Readout/characterize.py --recovery --n-devices 50 --output report.yaml
  python 06_Dispersive_Readout/characterize.py --generate-synthetic --output synthetic.npz
```

```python
import argparse
from .fitting import fit_all
from .protocols import load_trace_bundle, save_trace_bundle, generate_rabi_trace, generate_ramsey_trace, generate_t1_trace, generate_t2_echo_trace
from .recovery import run_recovery_harness, save_coverage_report
from .noise import NoiseModelParams


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="06_Dispersive_Readout/characterize.py",
        description="Extract device parameters from characterization traces.",
    )
    parser.add_argument("--traces", type=str, help="Path to .npz trace bundle")
    parser.add_argument("--output", type=str, required=True, help="Output path (.yaml or .md)")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--recovery", action="store_true", help="Run the 50-device recovery harness")
    parser.add_argument("--n-devices", type=int, default=50)
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate synthetic traces using REFERENCE_DEVICE")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Flag-combination validation: --traces + --generate-synthetic is rejected
    # (§8 flag #5). --recovery overrides both.
    ...
```

**CLI invariants.**
- `--help` output is committed and tested (C5d below) — no TODO strings, all three use modes shown.
- Flag combinations are validated up-front; incompatible combinations reject with a clear error before any computation.
- Exit code 0 on success, non-zero on any fit failure or gate miss.
- `--seed` defaults to 42 (the committed-artifact seed per amendment 9). Explicit override supported for ad hoc exploration.

---

## 5. Validation tests (`dispersive_readout/tests/test_characterization.py`)

### C1 — Round-trip for each protocol

```python
def test_C1a_rabi_round_trip(): ...      # recovered ε_π within 3% at default noise
def test_C1b_ramsey_round_trip(): ...    # recovered ω_q within 0.1%, T2* within 10%
def test_C1c_t1_round_trip(): ...        # recovered T1 within 10%
def test_C1d_t2_echo_round_trip(): ...   # recovered T2 within 10%
```

### C2 — Noise model sanity

```python
def test_C2a_shot_noise_matches_binomial(): ...
def test_C2b_1f_drift_psd_slope_approx_minus_one(): ...
def test_C2c_load_reference_F_full_matches_yaml(): ...
```

### C3 — Recovery coverage gate (amendment 9: committed-artifact regression)

```python
def test_C3_recovery_coverage_matches_committed_artifact():
    """Regression gate (amendment 9): re-run 50-device harness at SEED=42,
    compare to committed recovery_coverage_report.yaml within ±2% per parameter.

    This is the CI-time gate. The separate CALIBRATION gate (amendment 4, §4.4)
    is enforced once at artifact-commit time — when the artifact was first
    committed, its observed coverage had a 2σ binomial CI that included 95%.
    A ±2% match to the committed artifact inherits that calibration property by
    construction (±2% << binomial SE ≈ 3.1% at n=50).

    If this regresses: diagnose the fitter before regenerating the artifact.
    Do NOT just re-commit a new artifact to make the test pass — that's the
    silent-drift failure mode amendment 9 exists to prevent.
    """
    observed = run_recovery_harness(n_devices=50, seed=42)
    committed = load_committed_coverage_report(
        "06_Dispersive_Readout/figures/recovery_coverage_report.yaml"
    )
    for name, report in observed.items():
        for field in ("coverage_1_sigma", "coverage_2_sigma"):
            delta = abs(getattr(report, field) - getattr(committed[name], field))
            assert delta < 0.02, (
                f"Recovery {field} regressed for {name}: "
                f"expected {getattr(committed[name], field):.2%}, "
                f"got {getattr(report, field):.2%} (delta {delta:.2%})"
            )
```

Runtime target: ≤ 2 minutes with closed-form trace generation (amendment 1). If this exceeds 3 minutes at n=50, profile before degrading.

### C4 — Pydantic schema validation

```python
def test_C4a_fitted_parameter_requires_positive_uncertainty(): ...
def test_C4b_extracted_parameter_pack_yaml_round_trip(): ...
def test_C4c_to_device_config_produces_simulator_consumable(): ...
    """Integration: to_device_config() → simulate_readout() runs without error."""
```

### C5 — CLI smoke tests

```python
def test_C5a_cli_generate_synthetic(tmp_path): ...
def test_C5b_cli_full_pipeline(tmp_path): ...     # generate → fit → YAML end-to-end
def test_C5c_cli_help_has_no_todo(): ...          # guards against help-text rot
def test_C5d_cli_rejects_conflicting_flags(tmp_path): ...  # --traces + --generate-synthetic
```

### C6 — Edge cases

```python
def test_C6a_ramsey_zero_detuning_envelope_only_path(): ...
def test_C6b_t1_with_elevated_thermal_no_downward_bias(): ...
def test_C6c_rabi_amplitude_span_too_small_rejects(): ...
```

### C7 — `to_device_config` physics consistency (amendment 5)

```python
def test_C7a_to_device_config_back_solves_E_J_from_omega_q(): ...
def test_C7b_to_device_config_warns_on_E_J_drift_over_30pct(): ...
```

Total Module 3 test count target: **≥25 tests** (C1: 4, C2: 3, C3: 1, C4: 3, C5: 4, C6: 3, C7: 2, plus unit tests for noise helpers and generators).

---

## 6. Figure 3 specification

**File:** `06_Dispersive_Readout/scripts/fig3_characterization.py` → `06_Dispersive_Readout/figures/fig3_characterization.png`.

**Layout:** 2×2 grid, 150 DPI, style-matched to Figures 1 and 2.

- **Panel (a) — Rabi fit + residuals.** Top: measured P₁ vs drive amplitude (points with error bars) overlaid with lmfit curve. Bottom: residuals with ±3σ bands. Annotation: ε_π with uncertainty, reduced χ².
- **Panel (b) — Ramsey fit + residuals.** Same layout. Annotation: ω_q, T2\* with uncertainties.
- **Panel (c) — T1 decay + residuals.** Same layout. Annotation: T1 with uncertainty.
- **Panel (d) — Parameter recovery parity plots.** Four parity sub-panels arranged in a 2×2 grid *within* Panel (d) (one per parameter: T1, T2, ω_q, ε_π). Each sub-panel: ground truth on x-axis, fitted value on y-axis (with per-point error bars from parametric bootstrap), y=x reference line, points colored by whether |z| ≤ 1. Observed 2σ coverage + 2σ CI annotated in each sub-panel. Scatter form with error bars, not a table — parity plots are the standard parameter-recovery visual and align with the external-reader-presentation principle (point-with-errorbar near the reference line is more informative than a numeric table).

**Caption (two-tier, per Module 2 convention).**

Short caption (≤70 words, 5-second read): "**Figure 3.** Synthetic characterization pipeline. (a)–(c): single-instance Rabi / Ramsey / T1 fits at REFERENCE_DEVICE under realistic noise (2000 shots/point, 1/f drift, Module 2 F_full assignment errors). (d): parameter recovery across 50 synthetic devices, SEED=42 — fitted vs ground truth with y=x reference; 2σ CI on coverage reported per parameter (target 95%)."

Adjacent methods note (~140 words): closed-form trace generation (amendment 1), parametric bootstrap under correlated 1/f drift (amendment 3), binomial-CI-based coverage gate (amendment 4), `to_device_config` E_J back-solve policy (amendment 5), cached-artifact CI regression (amendment 9).

---

## 7. Day-by-day breakdown

### Day 7 — Protocols + noise model + fitting

**Morning:** `characterization/noise.py` (with `load_reference_F_full`). `characterization/protocols.py` closed-form generators for all four protocols. C2 (noise sanity) + C1 (round-trip) passing.

**Afternoon:** `characterization/fitting.py` lmfit wrappers. Pydantic schemas. Parametric bootstrap. C4 (schema) passing.

**End-of-day checkpoint:** All four generators + fitters functional. C1, C2, C4 passing. Trace bundles round-trip through `.npz`. (Closed-form generation is lighter than originally scoped — day-7 work expands to include the fitting module that would otherwise have been day-8 morning.)

### Day 8 — Recovery harness

**Morning:** `characterization/recovery.py` with `fit_one_device` as the pure function. `generate_synthetic_device_family` sampler. First full 50-device run at SEED=42.

**Afternoon:** Diagnose coverage misses. Likely: bootstrap still mis-calibrated on the Δω=0 edge case, or thermal-offset T1 device. Iterate. Commit `recovery_coverage_report.yaml` once coverage CI includes 95% for all four parameters.

**End-of-day checkpoint:** C3 passing against the committed artifact. `recovery_coverage_report.yaml` committed. Coverage CI ranges documented in the commit message.

### Day 9 — CLI + Figure 3 + polish

**Morning:** `characterization/cli.py`. `06_Dispersive_Readout/characterize.py` thin entry. C5 (CLI smoke) passing including the help-text guard. C6 (edge cases) + C7 (device-config consistency) passing.

**Afternoon:** `06_Dispersive_Readout/scripts/fig3_characterization.py`. Render figure. Style-match to Figures 1 and 2. Commit figure. Commit message: "Stage 06 Module 3: characterization interface with 50-device parameter recovery, coverage CI includes 95% per parameter at SEED=42".

---

## 8. Flags to the human

1. **If C3 fails at 50 devices.** The fitter is miscalibrated. Options in order: (a) increase `n_bootstrap`; (b) verify the 1/f drift correlation length is well-represented in the bootstrap; (c) diagnose via the bias-vs-coverage split (if bias is large, the fit form is wrong; if coverage is low but bias is zero, the uncertainty is miscalibrated). Do NOT lower the gate.
2. **If lmfit returns unphysical best-fit values** (T1 < 0, negative oscillation amplitude). Parameter bounds are wrong or initial guesses are seeded incorrectly. Stop and debug.
3. **If 1/f drift biases Ramsey ω_q by more than 2σ of the fitter's reported uncertainty on average across 50 devices.** Threshold is relative to the fitter's own uncertainty, not an absolute percentage — a bias of 0.1σ_fitted is noise; a bias of 5σ_fitted is a real problem and means either the drift model is too aggressive for the protocol's delay range, or the fitter's initial ω_q guess is seeded too far off. Investigate; don't just widen uncertainties.
4. **If the Ramsey Δω=0 edge case crashes.** The envelope-only fallback is not wired through. Add the path or reject Δω=0 devices in the harness with a clear error.
5. **If `load_trace_bundle` accepts malformed .npz silently.** Add schema validation at load time — reject bundles missing any required field (`protocol`, `sweep_axis`, `sweep_values`, `P1`, `P1_uncertainty`, `metadata`).
6. **If `to_device_config()` returns a `DeviceConfig` that `simulate_readout` rejects.** The bridge is broken at the contract level. C4c is the guard — if it passes but downstream fails, add a targeted test at the failure site.
7. **If the recovery harness takes more than 3 minutes.** Closed-form generation should be sub-second; total runtime is dominated by `lmfit` + parametric bootstrap. Profile before parallelizing — Modal is a Module 4 enabler, not a Module 3 necessity per amendment 1.

---

## 9. Review checklist before advancing to Module 4

- [ ] All Module 3 tests (C1–C7) passing, ≥ 25 tests total; full suite (Module 1 + 2 + 3) ≥ 97 tests passing.
- [ ] `recovery_coverage_report.yaml` committed at `06_Dispersive_Readout/figures/`; 2σ binomial CI on observed coverage includes 95% for each of (T1, T2, ω_q, ε_π) at n=50, SEED=42.
- [ ] CLI runs end-to-end from `python 06_Dispersive_Readout/characterize.py --help` to `--traces ... --output params.yaml`; `--help` text has zero TODO/TBD strings.
- [ ] `ExtractedParameterPack.to_device_config()` produces a `DeviceConfig` that `simulate_readout` consumes without error (C4c passes).
- [ ] `to_device_config()` back-solves E_J from ω_q via the Koch formula and warns on > 30% E_J drift (C7a, C7b pass).
- [ ] Figure 3 rendered at 150 DPI, style-matched to Figures 1/2; two-tier caption committed.
- [ ] `dispersive_readout.characterization` public API exposes `TraceData`, `NoiseModelParams`, `ExtractedParameterPack`, `fit_all`, `fit_one_device`, `run_recovery_harness`.
- [ ] One example synthetic `.npz` committed at `06_Dispersive_Readout/examples/example_traces.npz` (generated from REFERENCE_DEVICE at SEED=42).
- [ ] Module 1 test suite still at 57/57 (no regressions from Module 3 edits). Module 2 test suite still at 15/15.

If any item is unchecked, Module 4 does not start.

---

## 10. Reference list for Module 3

- **Sank et al., arXiv:2402.00413 (2024)** — System characterization of dispersive readout; noise-model realism reference.
- **Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)** — Review of superconducting qubit characterization protocols; fitting-form conventions.
- **Burnett et al., npj Quantum Inf. 5, 54 (2019)** — Decoherence benchmarking; 1/f drift treatment.
- **Koch et al., Phys. Rev. A 76, 042319 (2007)** — Transmon spectrum; E_J back-solve formula ω_01 ≈ √(8·E_J·E_C) − E_C used by `to_device_config` (amendment 5).
- **Marxer et al., arXiv:2508.16437 (Aug 2025)** — Parameter regime anchor; shot-count convention.

---

## Execution handoff

Plan complete once this spec is approved. Two execution options, matching Module 2:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; review between tasks; fast iteration. Each Module 3 task is self-contained with its own tests and commit.

**2. Inline execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Next step after spec approval:** invoke the `writing-plans` skill to produce `06_Dispersive_Readout/PLAN.md` (Module 3 task breakdown). Task 1 executes from that plan.

---

**Pre-execution checklist (applies to either path):**

- [ ] On branch `stage-06-module-2-error-budget` or a new `stage-06-module-3-characterization` branch.
- [ ] Module 1 tests passing (57 of 72).
- [ ] Module 2 tests passing (15 of 72).
- [ ] `06_Dispersive_Readout/figures/fig2_data.yaml` exists and is readable by `load_reference_F_full`.
- [ ] `MODULE_3_SPEC.md` unchanged since PLAN.md is written (if spec edited, re-review affected tasks).
