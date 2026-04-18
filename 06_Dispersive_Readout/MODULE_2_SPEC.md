# Stage 06 Module 2 — Coherent/Incoherent Error Budget

**Status:** amended design, 2026-04-18. Supersedes the original Module 2 draft pasted into the brainstorming session on the same date.

**Goal.** Decomposition of dispersive readout assignment infidelity at a fixed operating point into named coherent and incoherent contributions, using the Module 1 simulator as the physics engine. Output is a two-group waterfall figure that reads in ≤ 10 s and answers the JD bullet "Model coherent and incoherent error sources affecting device performance."

**Budget.** 2 working days (days 5–6).

**Prerequisites.** All four Module 1 physics-validation tests (V1a, V2, V3, V4a/b) passing; Figure 1 committed; Module 1 public API (`dispersive_readout.physics`) exposes `simulate_readout`, `compute_assignment_fidelity`, `ReadoutResult`, `AssignmentFidelityResult`, `REFERENCE_DEVICE`, `DeviceConfig`, `DriveParams`, `DecoherenceParams`.

**Scope lock.** Physics decisions below are locked. No expansion to non-Markovian effects, correlated noise, or measurement backaction beyond what is listed. Implementation discoveries that challenge a locked decision are raised as blockers, not silently resolved.

---

## 0. Amendments applied to the original spec

Seven substantive blockers were raised during brainstorming and resolved with spec amendments before implementation. The rest of this document is the *post-amendment* spec; this section records what changed and why, so the delta to the original is traceable.

| # | Amendment | Driver |
|---|---|---|
| 1 | Drop the "dispersive approximation breakdown" channel | The post-refactor `build_hamiltonian` has `frame="rotating"` and `frame="dispersive"` as aliases for the same 2nd-order SW object; there is no bare-JC path to diff against. V2 unit test already validates the dispersive approximation at ≤2%. |
| 2 | Split the waterfall into two visually and arithmetically distinct groups: **active loss** (decoherence-like channels with a shared additivity identity) vs **calibration sensitivity** (perturbation-derivative probes). | The original "marginal turn-off" methodology only holds for channels present in the baseline; drive miscal/detuning are turn-on probes that don't share the same arithmetic. Also: add `purcell_enabled: bool = True` to `DecoherenceParams` for the one channel that lacks a natural parameter null. |
| 3 | Drop the "leakage to \|2⟩" channel | Measurement-induced ionization physics (Shillito 2022, Dumas 2024) lives in the non-RWA terms that the SW transformation eliminates; the simulator has no coherent \|1⟩→\|2⟩ pathway. Measured P(\|2⟩) at baseline REFERENCE with \|1⟩ init is 2.9 × 10⁻⁴, entirely thermal, double-counts the thermal channel. |
| 4 | Drop bootstrap; use analytic binomial shot-noise SE. Add `rng: np.random.Generator \| None = None` kwarg to `compute_assignment_fidelity`; default ephemeral RNG gives independent shot draws across successive calls. | `seed=42` hardcoded in `readout_model.py:186` correlates shot draws across simulations, breaking quadrature propagation. Bootstrap with fixed seed gives zero variance (no-op). Closed-form binomial variance is exact for perp-bisector on Gaussian IQ. |
| 5 | Analytic drive-amplitude calibration from steady-state \|Δα\| formula, with simulation-verified fallback to grid search if the measured F deviates from target by > 3σ_shot | Binary search on F(ε₀) assumes monotonicity; F(ε₀) has a single-maximum shape, so binary search on a below-peak target returns the low-ε or high-ε crossing unpredictably. Closed-form calibration is exact within the SW scope already used throughout Stage 06. |
| 6 | Reduce `purcell_isolation.py` to one function (`analytic_purcell_rate`); replace moot B3 tests with `test_simulated_purcell_matches_analytic_within_1_percent` at REFERENCE + `..._at_strong_coupling` at 2× REFERENCE coupling with 5% tolerance | Post-blocker-2, Purcell is a proper collapse-operator channel, so `effective_T1_from_device` and `decomposed_T1` are YAGNI. Original B3 compared Hamiltonian frames made equivalent by blocker 1. |
| 7 | Figure 2 presentation is two-tier: short caption (~70 words, 5-second read) + adjacent methods note (~140 words) carrying the defensive content from blockers 1, 3, 5, 6 | Stacked in one caption the defensive sentences hit ~180 words, defeating the caption's 5-second-read purpose. |

---

## 1. Physical scope

The operating point is a single `(DeviceConfig, DriveParams, integration_window)` triple using `REFERENCE_DEVICE` from Module 1 (Marxer 2508.16437 parameters). Integration window 500 ns (Bengtsson 2024 convention).

### Channels

| # | Name | Group | Turn-off / perturbation semantic |
|---|---|---|---|
| 1 | `T1_intrinsic` | active loss | `DecoherenceParams(gamma_1=0, ...)` |
| 2 | `pure_dephasing` | active loss | `DecoherenceParams(gamma_phi=0, ...)` |
| 3 | `thermal` | active loss | `DecoherenceParams(n_th=0, ...)` |
| 4 | `purcell` | active loss | `DecoherenceParams(purcell_enabled=False, ...)` (requires new field — see §3) |
| 5 | `drive_amplitude` | calibration sensitivity | `DriveParams(amplitude=ε₀ × (1 ± 0.05), ...)`, mean of two `\|F_full − F_±\|` |
| 6 | `drive_detuning` | calibration sensitivity | `DriveParams(detuning=± κ/4, ...)`, mean of two `\|F_full − F_±\|` |

Six named channels. Total waterfall: **6 named + 1 residual + 1 total = 8 bars.**

### Explicitly NOT decomposed

- Correlated 1/f noise — Module 3.
- Beyond-2nd-order dispersive corrections — validated at ≤ 2% by V2, contribution to F_assign is O(10⁻⁴).
- Measurement-induced \|1⟩→\|2⟩ ionization — outside simulator scope (SW transformation eliminates the pathway); at REFERENCE the operating point sits at `n̄/n_crit ≈ [ratio; measured at calibration]`, well below onset.
- Amplifier / TWPA noise — absorbed into the Module 1 Gaussian-IQ noise model.
- Crosstalk — single-qubit scope.

---

## 2. Methodology

### 2.1 Two-group waterfall

The six channels decompose into two arithmetically distinct groups.

**Group A — Active loss (4 channels).** Each measures a loss contribution the baseline already incurs. For channel `c`:

```
ΔF_c = F_c_off − F_full     (non-negative, modulo shot noise)
```

with `F_c_off` computed by zeroing channel `c`'s rate while keeping all other active-loss channels at baseline. The arithmetic identity:

```
F_ideal − F_full = Σ_{c ∈ active} ΔF_c + R_active
```

where `F_ideal` = all four active-loss channels disabled (`DecoherenceParams(gamma_1=0, gamma_phi=0, n_th=0, purcell_enabled=False)`, κ at baseline), and `R_active` is the cross-channel interaction residual. Test B2 validates `|R_active| < 0.2 × (F_ideal − F_full)`.

**Group B — Calibration sensitivity (2 channels).** Each measures F loss under a named perturbation about the nominal calibration:

```
ΔF_c = mean(|F_full − F_+|, |F_full − F_-|)     (always non-negative)
err_c = |F_+ − F_-| / 2                          (asymmetry as error bar)
```

Group-B bars do **not** enter the arithmetic identity and are not summable with Group A. Error bars on calibration bars are `err_c` (asymmetry), not bootstrap uncertainty.

### 2.2 Baseline operating point

```python
REFERENCE_OPERATING_POINT = OperatingPoint(
    device=REFERENCE_DEVICE,
    drive=DriveParams(
        amplitude=ε₀_calibrated,   # solved analytically per §2.3
        duration=500e-9,
        detuning=0.0,
        edge_sigma=2e-9,
    ),
    integration_window=(50e-9, 500e-9),
    n_shots=10_000,
)
```

### 2.3 Drive-amplitude calibration (analytic with verified fallback)

Target: `F_full ≈ 0.99` on the low-amplitude branch of the F(ε₀) curve.

Dispersive-regime steady-state coherent amplitude for qubit in `|j⟩`:

```
α_j(ε₀) = ε₀ / (κ/2 − i(χ_j + ω_r − ω_d))
```

At detuning = 0 and on-resonance with ω_r, the separation is linear in ε₀:

```
|Δα(ε₀)| = ε₀ × |1/(κ/2 − iχ_0) − 1/(κ/2 − iχ_1)|
```

Closed-form calibration: given target SNR from `F_target = 1 − Q(SNR/2)`:

```
SNR_target ≈ 2 × Φ⁻¹(F_target)         # Φ⁻¹(0.99) ≈ 2.33 → SNR_target ≈ 4.66
ε₀_analytic = SNR_target / (2 × sqrt(κ × T_int) × |dα/dε₀|)
```

where `T_int = integration_window[1] − integration_window[0]`.

**Verification step (mandatory).** Simulate at `ε₀_analytic`, measure `F_verified` via `compute_assignment_fidelity`, compare to `F_target`:

```
σ_shot = sqrt(F_target × (1 − F_target) / n_shots)   # ≈ 1e-3 at F=0.99, n_shots=1e4
tolerance = 3 × σ_shot
```

If `|F_verified − F_target| ≤ tolerance`: accept `ε₀_analytic`.

If out of tolerance: fall back to grid search on the low-ε branch (~15 points linear in ε from ε_min where F ≈ 0.5 up to ε_max where `n̄_peak ≈ 0.5 × N_resonator`). Pick lowest ε where F ≥ target. Emit a warning and log the fallback; a fallback triggering pre-submission is a blocker, not a routine outcome.

### 2.4 Uncertainty bookkeeping (analytic, no bootstrap)

For each `compute_assignment_fidelity` call, the analytic binomial SE is:

```
σ_F = sqrt(F × (1 − F) / n_shots)
```

For active-loss channels, with independent shot draws (ensured by default `rng=None` per call):

```
σ_ΔF_c = sqrt(σ_F_c_off² + σ_F_full²)
σ_R_active = sqrt(σ_F_ideal² + σ_F_full² + Σ σ_ΔF_c²)   # variance of linear combination
```

For calibration-sensitivity channels, σ is the asymmetry `err_c = |F_+ − F_-| / 2` (not shot-noise-derived; captures local curvature asymmetry).

**Code change required:** `compute_assignment_fidelity` (in `dispersive_readout/physics/readout_model.py`) gains `rng: np.random.Generator | None = None` kwarg. Default `None` → ephemeral RNG per call → independent draws. Existing Module 1 tests that assert a specific `F_assign` value need `rng=np.random.default_rng(seed=42)` passed explicitly for determinism. Other tests with loose tolerances should tolerate `rng=None` because `abs(F − expected) < tol` with `tol > 3 × √(F(1−F)/n_shots)` remains true.

---

## 3. Code changes to Module 1

Module 2 requires two targeted Module 1 edits:

1. **`dispersive_readout/physics/config.py` — add `purcell_enabled: bool = True` to `DecoherenceParams`.**

2. **`dispersive_readout/physics/lindblad.py:128-135` — gate the Purcell loop on the new field:**

   ```python
   if device.decoherence.purcell_enabled:
       for j in range(1, Nq):
           ...  # existing Purcell construction
   ```

3. **`dispersive_readout/physics/readout_model.py:186` and `compute_assignment_fidelity` signature — add `rng` kwarg, default `None`.**

Tests that previously relied on the hardcoded `seed=42` either pass `rng=np.random.default_rng(seed=42)` explicitly or have their tolerance loosened to ≥ 3σ_shot. Flag if more than two tests flake under `rng=None` — that's a signal tolerances were too tight.

---

## 4. Module 2 structure

```
dispersive_readout/
├── analysis/
│   ├── __init__.py          # exposes OperatingPoint, ErrorBudget, compute_full_error_budget
│   ├── error_budget.py      # decomposition logic + Pydantic schemas
│   ├── operating_point.py   # OperatingPoint + analytic calibration + verified fallback
│   └── purcell_isolation.py # analytic_purcell_rate only (15 lines, single exported fn)
└── tests/
    └── test_error_budget.py # B1-B5 + per-channel + Purcell sanity (§6)
```

Scripts live under `06_Dispersive_Readout/`:

```
06_Dispersive_Readout/
├── scripts/
│   └── fig2_error_budget.py # renders figures/fig2_error_budget.png from a computed ErrorBudget
└── figures/
    ├── fig2_error_budget.png
    └── fig2_data.yaml       # YAML-serialized ErrorBudget for the REFERENCE operating point
```

---

## 5. Detailed module specifications

### 5.1 `analysis/operating_point.py`

```python
@dataclass(frozen=True)
class OperatingPoint:
    device: DeviceConfig
    drive: DriveParams
    integration_window: tuple[float, float]
    n_shots: int


def calibrate_drive_amplitude(
    device: DeviceConfig,
    duration: float,
    integration_window: tuple[float, float],
    target_fidelity: float = 0.99,
    n_shots: int = 10_000,
    sigma_tolerance_factor: float = 3.0,
) -> float:
    """Analytic calibration with simulation-verified fallback per §2.3."""

def get_reference_operating_point() -> OperatingPoint:
    """Return the canonical operating point. No cache — calibration is ~2s."""
```

No caching decorator. Calibration runs at module load / first call and takes < 3 s total (analytic solve + one verification sim × two qubit states).

### 5.2 `analysis/purcell_isolation.py`

```python
def analytic_purcell_rate(device: DeviceConfig) -> float:
    """γ_Purcell for the |1>→|0> transition from (g|<0|n̂|1>|/Δ_{10})² κ.

    Reference: Blais et al. RMP 93, 025005 (2021) §III.E. Used as cross-validator
    against the simulated Purcell contribution (test B3).
    """
```

Single exported function. ~15 lines. `effective_T1_from_device` and `decomposed_T1` from the original spec are not implemented (YAGNI post-blocker-6).

### 5.3 `analysis/error_budget.py`

```python
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
    name: ChannelName
    group: ChannelGroup
    delta_F: float            # non-negative (§2.1)
    delta_F_uncertainty: float  # analytic σ for active_loss, asymmetry for calibration_sensitivity
    description: str

    @field_validator("delta_F")
    @classmethod
    def nonnegative(cls, v: float) -> float:
        if v < -0.005:       # shot-noise floor
            raise ValueError(f"Channel contribution unexpectedly negative: {v}")
        return max(v, 0.0)


class ErrorBudget(BaseModel):
    operating_point_id: str
    F_full: float
    F_ideal: float
    channels: list[ChannelContribution]
    residual_active: float          # R_active (§2.1). Defined only over active_loss group.
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
```

**Physics decisions locked (post-amendment):**

- `F_ideal` = simulation with `DecoherenceParams(gamma_1=0, gamma_phi=0, n_th=0, purcell_enabled=False)`, κ at baseline (κ is the readout itself, not a decoherence channel).
- Contributions stored as absolute fidelity deltas; percentage-of-infidelity conversion happens at figure-rendering time.
- Uncertainties are analytic binomial SE (active loss) or perturbation asymmetry (calibration sensitivity). No bootstrap.

### 5.4 Public API (`analysis/__init__.py`)

```python
from .operating_point import OperatingPoint, calibrate_drive_amplitude, get_reference_operating_point
from .error_budget import (
    ChannelName, ChannelGroup, ChannelContribution, ErrorBudget,
    compute_channel_contribution, compute_full_error_budget, export_budget_to_yaml,
)
from .purcell_isolation import analytic_purcell_rate
```

---

## 6. Tests (`tests/test_error_budget.py`)

Target: **≥ 12 tests**. Module 2 gate for advancing to Module 3 is that all pass.

### B1 — Additivity (active-loss group only)

```python
def test_active_loss_sums_to_ideal_minus_full_within_tolerance():
    """Σ ΔF_c + R_active ≈ (F_ideal − F_full) within 3σ_prop for the active group."""
```

### B2 — Residual is small (active-loss group only)

```python
def test_active_loss_residual_under_20_percent():
    """|R_active| < 0.2 × (F_ideal − F_full). If it fails, channels interact strongly
    and the marginal-attribution methodology is breaking down."""
```

### B3 — Purcell simulated-vs-analytic cross-validation (replaces original moot B3)

```python
def test_simulated_purcell_matches_analytic_within_1_percent_at_reference():
    """ΔF_Purcell from simulation vs analytic (g/Δ)²κ prediction; 1% tolerance at REFERENCE
    (2nd-order PT residual is ~0.2% at g/Δ≈0.044, so 1% catches bugs above physics ceiling)."""

def test_simulated_purcell_matches_analytic_at_strong_coupling():
    """Same check at 2× REFERENCE coupling (g/Δ≈0.088); 5% tolerance.
    If stricter REFERENCE test passes but this fails, reveals the 2nd-order SW boundary."""
```

### B4 — Pydantic validation

```python
def test_negative_contribution_raises():
    """ChannelContribution(delta_F=-0.01) must raise; delta_F=-0.003 floors to 0."""
```

### B5 — YAML round-trip

```python
def test_budget_yaml_round_trip():
    """export_budget_to_yaml + re-read reproduces the ErrorBudget exactly."""
```

### Per-channel tests (6)

```python
def test_T1_intrinsic_contribution_nonzero_at_reference(): ...
def test_pure_dephasing_contribution_nonzero_at_reference(): ...
def test_thermal_contribution_nonzero_at_reference(): ...
def test_purcell_contribution_nonzero_at_reference(): ...
def test_drive_amplitude_sensitivity_matches_first_order_taylor_within_20_percent(): ...
def test_drive_detuning_sensitivity_matches_second_order_taylor_within_20_percent(): ...
```

### Operating-point calibration test

```python
def test_analytic_calibration_hits_target_fidelity_within_3_sigma():
    """Analytic calibration produces F_verified in F_target ± 3σ_shot at REFERENCE_DEVICE.
    If this fails, fallback to grid search is triggered — log a warning."""
```

**Total: 13 tests.** Passes the "≥ 12" gate.

---

## 7. Figure 2

### 7.1 Layout

Single-panel waterfall, 8 bars left-to-right:

```
| Total infidelity | [  T1 | dephasing | thermal | Purcell  ] | [ drive_amp | drive_det ] | R_active |
   anchor                   ── Active loss group ──                ── Cal sens group ──      residual
                             (warm palette)                          (cool palette)            (grey)
```

Horizontal dashed reference line at `1 − F_ideal`, labeled "Ideal-limit floor."

Bar heights in 10⁻³ units on y-axis. Annotations above each bar: percentage of total infidelity to 2 sig figs for active-loss and residual bars; `|ΔF| at ±perturbation` for calibration bars (asymmetry shown as small tick marks above and below bar top, not percentage).

### 7.2 Caption (two-tier)

> **Figure 2.** Assignment infidelity decomposition at REFERENCE_DEVICE (500 ns readout, F_full ≈ 0.99, 10⁴ shots). **Active loss** (left, 4 bars): T1, pure dephasing, thermal, Purcell — each measured by turning off its collapse operator. **Calibration sensitivity** (right, 2 bars): F loss under ±5 % amplitude / ±κ/4 detuning perturbations about the nominal operating point. The grey residual bar reports cross-channel interactions within the active-loss group and satisfies the additivity identity Σ ΔF_c + R = (F_ideal − F_full).

Followed immediately by:

> **Methods note (Figure 2).** The waterfall decomposes assignment infidelity within the scope of the 2nd-order Schrieffer-Wolff dispersive-frame Hamiltonian used throughout Stage 06. Two physics boundaries are relevant: the dispersive approximation itself is validated by unit test V2 to ≤ 2 % at REFERENCE, producing a fidelity residual of O(10⁻⁴) below the bar-visibility threshold; measurement-induced ionization (Shillito 2022) requires an intra-resonator photon count of n̄ > n_crit, where the reference operating point sits at n̄/n_crit ≈ [ratio measured at calibration], well below onset. Residual \|1⟩→\|2⟩ occupation P(\|2⟩) ≈ 3 × 10⁻⁴ is entirely thermal and is attributed to the thermal channel. The operating point ε₀ is calibrated analytically from the dispersive-regime steady-state SNR formula (§2.3) and cross-verified against simulation within shot-noise tolerance. Active-loss and calibration-sensitivity bars answer two conceptually distinct questions — loss contribution at the nominal point versus robustness derivative under named perturbations — and are presented as separate groups to make this distinction explicit; only the active-loss group carries a residual identity and a B2 additivity validation test.

### 7.3 Style

- 150 DPI, ~1200 px wide.
- Font matches Figure 1.
- Warm palette for active loss (single color family, darker→lighter across the four bars); cool palette for calibration sensitivity (two-color family); neutral grey for residual.
- Optional second panel (ship only if day 6 finishes ahead): same decomposition at +10 % drive amplitude to show how contributions shift with operating point.

---

## 8. Day-by-day breakdown

### Day 5 (Mon Apr 20) — Decomposition implementation

**Morning:**
- Task 1: Add `purcell_enabled: bool = True` to `DecoherenceParams`; gate Purcell loop in `lindblad.py`. Run full Module 1 test suite to confirm no regressions.
- Task 2: Add `rng: np.random.Generator | None = None` kwarg to `compute_assignment_fidelity`. Update Module 1 tests that need determinism to pass explicit `rng`. Run full Module 1 suite.
- Task 3: Create `dispersive_readout/analysis/` package skeleton + `__init__.py` public API stub.
- Task 4: `operating_point.py` — `OperatingPoint` dataclass, analytic calibration, verification fallback.
- Task 5: `purcell_isolation.py` — `analytic_purcell_rate` (~15 lines).

**Afternoon:**
- Task 6: `error_budget.py` Pydantic schemas (`ChannelContribution`, `ErrorBudget`) + B4 Pydantic test.
- Task 7: `compute_channel_contribution` for each of the 6 channels.
- Task 8: `compute_full_error_budget` + B1 + B2 tests.
- Task 9: B3 Purcell sanity tests (REFERENCE + strong coupling).

**End-of-day checkpoint:** `ErrorBudget` produced + YAML-serialized for REFERENCE. All 6 channels report nonzero contributions. B1, B2, B3, B4 passing. `n̄/n_crit` ratio measured and recorded.

### Day 6 (Tue Apr 21) — Figure 2 + polish

**Morning:**
- Task 10: `scripts/fig2_error_budget.py` → first-pass waterfall.
- Task 11: Iterate on styling (palette, ordering, annotations, reference line).
- Task 12: B5 YAML round-trip + per-channel unit tests + operating-point calibration test.

**Afternoon:**
- Task 13: Figure 2 committed at publication quality; caption + methods note substituted with measured `n̄/n_crit`.
- Task 14: Verify caption reads in ≤ 10 s; methods note completes the defense without requiring external lookup.
- If ahead of schedule: second-panel comparison at +10 % drive amplitude.
- Begin Module 3: sketch `characterization/protocols.py` signatures.

**End-of-day checkpoint:** Figure 2 committed. Module 2 complete. Module 3 stub files in place.

---

## 9. Flags to human

1. **B2 fails (residual > 20 % of active-loss infidelity).** Marginal attribution is breaking down; channels interact strongly. Stop, investigate, consider regrouping (e.g., merge `T1_intrinsic` and `purcell` into a single "relaxation" channel).
2. **Any active-loss contribution negative beyond shot noise.** Turn-off logic bug — the channel is making fidelity *worse* when disabled, which is non-physical for decoherence.
3. **P(\|2⟩) > 0.05 at baseline.** Drive is entering the ionization regime; the simulator is being pushed outside its validated scope. Informational at current calibration (expected P(\|2⟩) ≈ 3 × 10⁻⁴); flag loudly with a regime-scope caveat if it surfaces.
4. **Calibration fallback triggers pre-submission.** Analytic D disagreed with simulation by > 3σ_shot; either the steady-state approximation fails at REFERENCE (unexpected — physics ceiling is ~1 %) or there's a bug. Treat as a Module 2 blocker.
5. **F_ideal < 0.999.** Something fundamentally limits readout even without decoherence; flag for physics review.
6. **More than 10 bars on the waterfall.** Scope-creep check. Cap is 6 named + 1 residual + 1 total.
7. **`n̄/n_crit > 0.2` at measured calibration.** Reference operating point is uncomfortably close to the ionization boundary; either reduce drive amplitude and re-run, or add a regime-scope caveat to the Methods note. Expected range at REFERENCE is 0.03–0.05.
8. **Purcell cross-check (B3) fails at REFERENCE.** Either a bug in `lindblad.py:128-135` or the 2nd-order PT boundary is tighter than assumed. Not a silent loosen-tolerance case — investigate before shipping.

---

## 10. Review checklist before advancing to Module 3

- [ ] All 13 Module 2 tests passing
- [ ] `ErrorBudget` Pydantic schema used throughout — no raw dicts
- [ ] Reference operating point calibrated analytically, cross-verified, no fallback triggered
- [ ] `n̄/n_crit` ratio measured at calibration and substituted into Figure 2 Methods note
- [ ] Figure 2 rendered at 150 DPI
- [ ] Figure 2 caption reads in ≤ 10 s; Methods note completes the defense
- [ ] `|R_active| < 0.2 × (F_ideal − F_full)`
- [ ] YAML export of reference error budget committed to `06_Dispersive_Readout/figures/fig2_data.yaml`
- [ ] `analysis/__init__.py` exposes the public API listed in §5.4
- [ ] Module 1 regressions: full test suite passes after `purcell_enabled` and `rng` edits

If any item is unchecked, Module 3 does not start.

---

## 11. References

- **Shillito et al., Phys. Rev. Applied 18, 034031 (2022)** — transmon ionization onset. Cited as regime-boundary reference for the `n̄/n_crit` check in the Figure 2 Methods note; not a source for a simulated channel.
- **Dumas et al., Phys. Rev. X 14, 041023 (2024)** — measurement-induced transmon ionization. Same role as Shillito: regime boundary, not simulated channel.
- **Bengtsson et al., Phys. Rev. Lett. 132, 100603 (2024)** — integration-window convention (500 ns).
- **Marxer et al., arXiv:2508.16437 (Aug 2025)** — parameter regime anchor (inherited from Module 1).
- **Blais et al., Rev. Mod. Phys. 93, 025005 (2021)** — dispersive-regime effective Hamiltonian (§III.E); analytic Purcell rate formula for B3 cross-validation.
- **Gambetta et al., Phys. Rev. A 77, 012112 (2008)** — perp-bisector discriminator SNR formula, `F = 1 − Q(SNR/2)`; basis for the analytic calibration in §2.3.
