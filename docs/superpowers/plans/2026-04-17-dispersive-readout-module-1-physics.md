# Stage 06 Module 1 — Dispersive Readout Physics Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a QuTiP-based Jaynes-Cummings + Lindblad simulator of a transmon qubit dispersively coupled to a readout resonator, producing IQ-plane trajectories, SNR vs. integration-time curves, and single-shot assignment fidelity — gated by four analytic validation tests (anharmonicity, dispersive shift, T₁, T₂/Purcell).

**Architecture:** A `dispersive_readout/` package at the repo root (sibling of `l2l/`, `vqe_modular/`) holds all importable code. Module 1 lives in `dispersive_readout/physics/`: frozen-dataclass config → charge-basis transmon diagonalizer → analytic χ formulas → Lindblad collapse operators in the dressed eigenbasis → pulsed-readout simulator returning `ReadoutResult` objects → assignment-fidelity calculator. Tests colocate with the package at `dispersive_readout/tests/` (matching `l2l/tests/`). Figure 1 is produced by one runnable stage-level script, `06_Dispersive_Readout/dispersive_readout_simulation.py`, that bridges to the package via the walk-up-to-sentinel sys.path shim used by stages 01–05.

**Tech Stack:** Python 3.10+, QuTiP 4.7+ (`mesolve`, `Qobj`, `tensor`, `destroy`, `basis`), NumPy, SciPy (`scipy.special.erf`), Matplotlib (Figure 1 only), pytest.

**Two deliberate departures from the as-written Module 1 spec (see conversation preceding this plan):**

1. **Package and stage folder are split.** Spec uses a single `stage_06_readout/` tree; this plan uses `dispersive_readout/` (importable, repo-root sibling) + `06_Dispersive_Readout/` (stage scripts with a sys.path shim) to match the 01–05 layout. Tests live in `dispersive_readout/tests/` rather than under the stage folder, so that if Stage 07 later imports `dispersive_readout`, package tests are not coupled to "the stage that happened to use the package."
2. **Figure script names describe the computation, not the paper ordering.** Spec's `fig1_readout_model.py` → this plan's `dispersive_readout_simulation.py`, output `dispersive_readout_simulation.png`. Figure numbering lives in the report/README, not the filename. Reordering figures during report writing no longer forces a code rename.

All physics decisions from the spec (`REFERENCE_DEVICE` values, truncations, dressed-basis collapse operators, ≤1e-4 tolerance on χ, ≤1% on T₁/T₂, ≤5% on anharmonicity/Purcell) are preserved verbatim. Do not re-debate them during implementation.

**`utils_QOS.py`:** Module 1 does **not** import it. QuTiP's `mesolve` is used directly and collapse operators are built fresh in `dispersive_readout/physics/lindblad.py`. The question of whether later modules reach for `utils_QOS.py` is explicitly out of scope for this plan.

---

## File Structure

**Files created (none modified):**

| Path | Responsibility |
|---|---|
| `dispersive_readout/__init__.py` | Package root; exposes version string only. |
| `dispersive_readout/physics/__init__.py` | Public API: config dataclasses, `REFERENCE_DEVICE`, `simulate_readout`, `compute_assignment_fidelity`, `snr_vs_integration_time`, transmon helpers, χ helpers. |
| `dispersive_readout/physics/config.py` | Frozen dataclasses; `REFERENCE_DEVICE` populated from Marxer arXiv:2508.16437. |
| `dispersive_readout/physics/transmon.py` | Charge-basis Hamiltonian, diagonalization, charge-op matrix elements, summary dict. |
| `dispersive_readout/physics/dispersive.py` | Two-level and multi-level analytic χ; numerical χ from dressed-state spectrum. |
| `dispersive_readout/physics/lindblad.py` | Collapse operators in dressed transmon basis; rotating-frame Hamiltonian + drive spec. |
| `dispersive_readout/physics/readout_model.py` | `ReadoutResult`, `AssignmentFidelityResult`, `simulate_readout`, `compute_assignment_fidelity`, `snr_vs_integration_time`. |
| `dispersive_readout/tests/__init__.py` | Empty. |
| `dispersive_readout/tests/test_config.py` | Frozen-dataclass and unit-conversion checks. |
| `dispersive_readout/tests/test_transmon.py` | Hamiltonian hermiticity, eigenvalue ordering, matrix-element sanity. |
| `dispersive_readout/tests/test_dispersive.py` | Two-level sign, multi-level vs. two-level limit. |
| `dispersive_readout/tests/test_lindblad.py` | Collapse operator and Hamiltonian shapes. |
| `dispersive_readout/tests/test_readout_model.py` | `simulate_readout` smoke test, IQ separation, assignment-fidelity plausibility. |
| `dispersive_readout/tests/test_physics_validation.py` | The four gating tests V1–V4, plus truncation convergence. |
| `06_Dispersive_Readout/dispersive_readout_simulation.py` | Runnable stage script → Figure 1 (3 panels). |
| `06_Dispersive_Readout/figures/.gitkeep` | Tracks empty dir. |
| `06_Dispersive_Readout/README.md` | Stage-level README (Module 1 deliverables only; placeholder sections for Modules 2–4). |

---

## Conventions referenced throughout

**Unit convention (locked from spec).** All rates, frequencies, coupling constants, and energies are stored in angular-frequency units (rad/s) *inside* the package. The only Hz boundaries are: (1) docstring/citation comments, (2) `E_C_Hz` / `E_J_Hz` properties on `TransmonParams`, (3) figure axis labels. `_TWO_PI = 2.0 * math.pi` is the only multiplier used for conversions.

**Sign convention (locked from spec).** Detuning `Δ = ω_q − ω_r`. For the reference device `Δ < 0` (qubit below resonator), so the two-level χ = g²/Δ is **negative**. If a test produces χ > 0 for this regime, it is a sign bug.

**χ convention (locked from spec).** χ ≡ (χ₁ − χ₀)/2. The "numerical" extractor returns this half-splitting, and the analytic-formula consumer computes the same half-splitting from per-level χⱼ — not the full χ₁ − χ₀.

**Commit style.** One commit per task. Conventional-commits prefix (`feat:`, `test:`, `chore:`) and a short body line referencing the task number. Commits should be green — all tests passing before each commit.

**Running tests.** From the repo root: `pytest dispersive_readout/tests/ -v`. A single test: `pytest dispersive_readout/tests/test_X.py::test_Y -v`.

---

## Task 1 — Scaffolding, config dataclasses, and REFERENCE_DEVICE

**Files:**
- Create: `dispersive_readout/__init__.py`
- Create: `dispersive_readout/physics/__init__.py`
- Create: `dispersive_readout/physics/config.py`
- Create: `dispersive_readout/tests/__init__.py`
- Create: `dispersive_readout/tests/test_config.py`

- [ ] **Step 1: Create directories and empty init files.**

Shell:
```bash
mkdir -p dispersive_readout/physics dispersive_readout/tests
```

Then create:

`dispersive_readout/__init__.py`:
```python
"""Dispersive-readout simulator for transmon + resonator systems.

Stage 06 of the Quantum_Simulation repo. Package root; see
`dispersive_readout.physics` for Module 1 (simulator foundation).
"""

__version__ = "0.1.0-module1"
```

`dispersive_readout/physics/__init__.py` (initial placeholder — final exports at Task 20):
```python
"""Physics foundation for the dispersive-readout simulator (Module 1)."""
```

`dispersive_readout/tests/__init__.py`:
```python
```

- [ ] **Step 2: Write the config test file with four failing tests.**

`dispersive_readout/tests/test_config.py`:
```python
"""Config dataclass and REFERENCE_DEVICE tests."""
from __future__ import annotations

import dataclasses
import math

import pytest

from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    CouplingParams,
    DecoherenceParams,
    DeviceConfig,
    DriveParams,
    ResonatorParams,
    TransmonParams,
    TruncationParams,
)

_TWO_PI = 2.0 * math.pi


def test_all_config_dataclasses_are_frozen():
    """Accidental mutation of config inside simulation code must be blocked."""
    for cls in (
        TransmonParams,
        ResonatorParams,
        CouplingParams,
        DecoherenceParams,
        DriveParams,
        TruncationParams,
        DeviceConfig,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} is not frozen"


def test_transmon_unit_conversion():
    """E_C_Hz and E_J_Hz are the rad/s values divided by 2π."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9)
    assert p.E_C_Hz == pytest.approx(210e6, rel=1e-12)
    assert p.E_J_Hz == pytest.approx(15.5e9, rel=1e-12)


def test_reference_device_matches_spec():
    """Reference device encodes Marxer 2508.16437 anchor values.

    Tolerance 1% allows the implementer to derive kappa/gamma from T-times etc.,
    but the top-level numbers must match the spec.
    """
    d = REFERENCE_DEVICE
    assert d.transmon.E_C_Hz == pytest.approx(210e6, rel=0.01)
    assert d.transmon.E_J_Hz == pytest.approx(15.5e9, rel=0.01)
    assert (d.resonator.omega_r / _TWO_PI) == pytest.approx(7.3e9, rel=0.01)
    assert (d.resonator.kappa / _TWO_PI) == pytest.approx(5e6, rel=0.01)
    assert (d.coupling.g / _TWO_PI) == pytest.approx(120e6, rel=0.01)
    # T1 = 30 us → γ1 = 1/T1
    assert d.decoherence.gamma_1 == pytest.approx(1.0 / 30e-6, rel=0.01)
    assert d.decoherence.n_th == pytest.approx(0.01, rel=0.01)


def test_truncation_defaults_match_spec():
    t = TruncationParams()
    assert t.N_charge == 13
    assert t.N_transmon == 5
    assert t.N_resonator == 15
    # N_charge must be odd so the charge ladder is symmetric about zero
    assert t.N_charge % 2 == 1
```

- [ ] **Step 3: Run tests — expect all four to fail with ImportError.**

Run:
```bash
pytest dispersive_readout/tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'dispersive_readout.physics.config'`.

- [ ] **Step 4: Implement `config.py`.**

`dispersive_readout/physics/config.py`:
```python
"""Frozen-dataclass config for the dispersive-readout simulator.

All rates and frequencies are stored in angular-frequency units (rad/s).
Hz values appear only at I/O boundaries (property accessors, display,
docstrings). See conversation plan header for citation trail.

REFERENCE_DEVICE values follow Marxer et al., arXiv:2508.16437 (IQM Munich,
Aug 2025) tunable-coupler + shelving-readout device; Bengtsson et al.,
Phys. Rev. Lett. 132, 100603 (2024) is the secondary cross-check reference.
Where Marxer does not tabulate an exact value (bare g, Δ), we use the
mid-range of IQM published values and mark the derivation in the field
comment. No proprietary data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TransmonParams:
    """Transmon qubit parameters (rad/s for energies)."""
    E_C: float
    E_J: float
    n_g: float = 0.0

    @property
    def E_C_Hz(self) -> float:
        return self.E_C / _TWO_PI

    @property
    def E_J_Hz(self) -> float:
        return self.E_J / _TWO_PI


@dataclass(frozen=True)
class ResonatorParams:
    """Readout resonator parameters."""
    omega_r: float  # resonator frequency, rad/s
    kappa: float    # total linewidth, rad/s


@dataclass(frozen=True)
class CouplingParams:
    """Transmon-resonator bare coupling."""
    g: float  # rad/s


@dataclass(frozen=True)
class DecoherenceParams:
    """Incoherent error channels.

    gamma_1:   qubit relaxation rate (1/s, equivalently rad/s for rates).
    gamma_phi: pure dephasing rate; from T2_echo after subtracting gamma_1/2.
    n_th:      bath thermal population (dimensionless).
    """
    gamma_1: float
    gamma_phi: float
    n_th: float = 0.01


@dataclass(frozen=True)
class DriveParams:
    """Readout drive pulse parameters.

    amplitude:  epsilon_0, rad/s.
    duration:   total pulse length, seconds.
    detuning:   omega_drive - omega_resonator, rad/s (0 = on resonance).
    edge_sigma: Gaussian-edge width for the erf-difference envelope, seconds.
    """
    amplitude: float
    duration: float
    detuning: float = 0.0
    edge_sigma: float = 2e-9


@dataclass(frozen=True)
class TruncationParams:
    """Hilbert-space truncation sizes.

    N_charge:    # charge states in [-N_charge//2, +N_charge//2]; must be odd.
    N_transmon:  transmon levels kept after diagonalization.
    N_resonator: resonator Fock basis size. Runtime-checked against
                 mean photon number during readout (readout_model.py).
    """
    N_charge: int = 13
    N_transmon: int = 5
    N_resonator: int = 15


@dataclass(frozen=True)
class DeviceConfig:
    """Complete device spec. Bundles the five param groups above."""
    transmon: TransmonParams
    resonator: ResonatorParams
    coupling: CouplingParams
    decoherence: DecoherenceParams
    truncation: TruncationParams = field(default_factory=TruncationParams)


# T1 = 30 us → γ1 = 1/T1 (≈ 5.3 kHz in "/2π" display units)
_T1_SEC = 30e-6
_T2_ECHO_SEC = 40e-6
_GAMMA_1 = 1.0 / _T1_SEC
# γφ from T2_echo relation: 1/T2 = γ1/2 + γφ  →  γφ = 1/T2 − γ1/2
_GAMMA_PHI = max(1.0 / _T2_ECHO_SEC - 0.5 * _GAMMA_1, 0.0)


REFERENCE_DEVICE: DeviceConfig = DeviceConfig(
    transmon=TransmonParams(
        E_C=_TWO_PI * 210e6,     # 210 MHz — Marxer 2508.16437 anharmonicity range
        E_J=_TWO_PI * 15.5e9,    # 15.5 GHz — gives E_J/E_C ≈ 74 (deep transmon, Koch 2007)
        n_g=0.0,                 # sweet spot
    ),
    resonator=ResonatorParams(
        omega_r=_TWO_PI * 7.3e9, # 7.3 GHz — within IQM tunable-coupler arch readout band
        kappa=_TWO_PI * 5e6,     # 5 MHz — fast-readout regime
    ),
    coupling=CouplingParams(
        g=_TWO_PI * 120e6,       # 120 MHz — mid-range IQM value (derived from reported χ, κ)
    ),
    decoherence=DecoherenceParams(
        gamma_1=_GAMMA_1,        # from T1 = 30 μs
        gamma_phi=_GAMMA_PHI,    # from T2_echo = 40 μs after γ1/2 subtraction
        n_th=0.01,               # ~30 mK base temperature
    ),
    truncation=TruncationParams(),
)
```

- [ ] **Step 5: Run tests — expect all four passing.**

Run:
```bash
pytest dispersive_readout/tests/test_config.py -v
```
Expected: `4 passed`.

- [ ] **Step 6: Commit.**

```bash
git add dispersive_readout/__init__.py \
        dispersive_readout/physics/__init__.py \
        dispersive_readout/physics/config.py \
        dispersive_readout/tests/__init__.py \
        dispersive_readout/tests/test_config.py
git commit -m "feat(stage06): scaffold dispersive_readout package with frozen config and REFERENCE_DEVICE

Task 1 of Module 1 plan. Dataclasses for transmon, resonator, coupling,
decoherence, drive, truncation; REFERENCE_DEVICE populated from Marxer
arXiv:2508.16437 (Bengtsson 2024 secondary reference). All rates in rad/s
internally; Hz only at I/O boundaries (E_C_Hz / E_J_Hz properties)."
```

---

## Task 2 — Transmon charge-basis Hamiltonian

**Files:**
- Create: `dispersive_readout/physics/transmon.py`
- Create: `dispersive_readout/tests/test_transmon.py`

- [ ] **Step 1: Write failing tests for the charge-basis Hamiltonian.**

`dispersive_readout/tests/test_transmon.py`:
```python
"""Transmon eigenstructure tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, TransmonParams, TruncationParams
from dispersive_readout.physics.transmon import (
    charge_basis_hamiltonian,
    charge_operator_matrix_elements,
    diagonalize_transmon,
    transmon_summary,
)

_TWO_PI = 2.0 * math.pi


# -- charge-basis Hamiltonian --------------------------------------------------

def test_charge_basis_hamiltonian_is_hermitian():
    H = charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert np.allclose(H, H.conj().T, atol=1e-20)


def test_charge_basis_hamiltonian_shape_and_dtype():
    trunc = TruncationParams()
    H = charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, trunc)
    assert H.shape == (trunc.N_charge, trunc.N_charge)
    assert H.dtype == np.float64


def test_charge_basis_hamiltonian_rejects_even_N_charge():
    bad = TruncationParams(N_charge=12, N_transmon=5, N_resonator=15)
    with pytest.raises(ValueError, match="odd"):
        charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, bad)


def test_charge_basis_diagonal_is_charging_energy():
    """Diagonal entries must be 4 E_C (n - n_g)^2."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9, n_g=0.0)
    trunc = TruncationParams(N_charge=13, N_transmon=5, N_resonator=15)
    H = charge_basis_hamiltonian(p, trunc)
    n_values = np.arange(-6, 7)
    expected_diag = 4.0 * p.E_C * n_values ** 2
    assert np.allclose(np.diag(H), expected_diag)


def test_charge_basis_offdiagonal_is_josephson():
    """Adjacent off-diagonals are -E_J/2."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9)
    trunc = TruncationParams()
    H = charge_basis_hamiltonian(p, trunc)
    for i in range(trunc.N_charge - 1):
        assert H[i, i + 1] == pytest.approx(-0.5 * p.E_J)
        assert H[i + 1, i] == pytest.approx(-0.5 * p.E_J)
    # Non-adjacent off-diagonals must be zero
    for i in range(trunc.N_charge):
        for j in range(trunc.N_charge):
            if abs(i - j) > 1:
                assert H[i, j] == 0.0
```

- [ ] **Step 2: Run tests — expect ImportError.**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py -v
```
Expected: `ModuleNotFoundError: No module named 'dispersive_readout.physics.transmon'`.

- [ ] **Step 3: Implement the Hamiltonian (and stub the rest of transmon.py so the file is importable).**

`dispersive_readout/physics/transmon.py`:
```python
"""Charge-basis transmon: Hamiltonian, diagonalization, matrix elements, summary.

Convention: ground-state energy is shifted to 0 after diagonalization.
All energies in rad/s. The transmon eigenbasis ("dressed transmon basis")
is the reference basis used by lindblad.py and readout_model.py.
"""
from __future__ import annotations

import numpy as np

from .config import TransmonParams, TruncationParams


def charge_basis_hamiltonian(
    params: TransmonParams,
    trunc: TruncationParams,
) -> np.ndarray:
    """Transmon Hamiltonian in the charge basis.

    H = 4 E_C (n - n_g)^2 - (E_J / 2) (|n><n+1| + |n+1><n|)

    The charge ladder runs over n = -N//2, ..., +N//2 and must be odd-sized
    so it is symmetric about n = 0. Returns a real symmetric matrix in rad/s.
    """
    N = trunc.N_charge
    if N % 2 == 0:
        raise ValueError(f"N_charge must be odd (got {N}) so the ladder is symmetric about zero.")
    n_values = np.arange(-(N // 2), N // 2 + 1, dtype=float)
    H = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(H, 4.0 * params.E_C * (n_values - params.n_g) ** 2)
    off = -0.5 * params.E_J
    idx = np.arange(N - 1)
    H[idx, idx + 1] = off
    H[idx + 1, idx] = off
    return H


def diagonalize_transmon(
    params: TransmonParams,
    trunc: TruncationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize and return (energies, eigenstates) for the lowest N_transmon levels.

    Ground-state energy is shifted to 0. Eigenstates are columns in the charge basis.
    """
    raise NotImplementedError  # Task 3


def charge_operator_matrix_elements(
    eigenstates: np.ndarray,
    trunc: TruncationParams,
) -> np.ndarray:
    """<j|n_hat|k> in the truncated transmon basis."""
    raise NotImplementedError  # Task 4


def transmon_summary(params: TransmonParams, trunc: TruncationParams) -> dict:
    """Diagnostic dict: omega_01, omega_12, alpha, E_J_over_E_C, charge dispersion, n-matrix elements."""
    raise NotImplementedError  # Task 4
```

- [ ] **Step 4: Run tests — expect 5 Hamiltonian tests passing, 0 failing.**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py -v -k hamiltonian_is_hermitian or hamiltonian_shape or rejects_even or diagonal_is or offdiagonal_is
```
Expected: `5 passed` for the above selection. (Other tests in the file will error on `NotImplementedError` — that is fine, they are driven by later tasks.)

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/transmon.py dispersive_readout/tests/test_transmon.py
git commit -m "feat(stage06): transmon charge-basis Hamiltonian

Task 2 of Module 1 plan. Builds the 4 E_C (n-n_g)^2 + Josephson
ladder in the charge basis; validates hermiticity, shape, and
matrix entries. Diagonalization and matrix elements stubbed as
NotImplementedError for Tasks 3–4."
```

---

## Task 3 — Transmon diagonalization

**Files:**
- Modify: `dispersive_readout/physics/transmon.py` (implement `diagonalize_transmon`)
- Modify: `dispersive_readout/tests/test_transmon.py` (add diagonalization tests)

- [ ] **Step 1: Append failing tests.**

Append to `dispersive_readout/tests/test_transmon.py`:
```python
# -- diagonalization -----------------------------------------------------------

def test_diagonalize_returns_correct_shapes():
    trunc = TruncationParams()
    energies, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, trunc)
    assert energies.shape == (trunc.N_transmon,)
    assert states.shape == (trunc.N_charge, trunc.N_transmon)


def test_diagonalize_energies_sorted_ascending():
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert np.all(np.diff(energies) > 0)


def test_diagonalize_ground_energy_shifted_to_zero():
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert energies[0] == pytest.approx(0.0, abs=1e-20)


def test_diagonalize_eigenstates_orthonormal():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    gram = states.conj().T @ states
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


def test_diagonalize_omega01_in_plausible_range():
    """For the reference device ω_01/2π should be ~4.4–4.8 GHz (Marxer device band)."""
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    omega_01_hz = energies[1] / _TWO_PI
    assert 4.3e9 < omega_01_hz < 4.9e9, f"omega_01/2π = {omega_01_hz/1e9:.3f} GHz outside Marxer band"
```

- [ ] **Step 2: Run — expect 5 failures (NotImplementedError).**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py::test_diagonalize_returns_correct_shapes -v
```
Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `diagonalize_transmon`.**

Replace the `diagonalize_transmon` stub in `dispersive_readout/physics/transmon.py` with:
```python
def diagonalize_transmon(
    params: TransmonParams,
    trunc: TruncationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize and return (energies, eigenstates) for the lowest N_transmon levels.

    Ground-state energy is shifted to 0. Eigenstates are returned as a
    (N_charge, N_transmon) array whose columns are eigenvectors in the charge
    basis.
    """
    H = charge_basis_hamiltonian(params, trunc)
    # np.linalg.eigh returns ascending eigenvalues for Hermitian input.
    eigvals_all, eigvecs_all = np.linalg.eigh(H)
    energies = eigvals_all[: trunc.N_transmon].copy()
    eigenstates = eigvecs_all[:, : trunc.N_transmon].copy()
    energies -= energies[0]  # shift ground state to zero by convention
    return energies, eigenstates
```

- [ ] **Step 4: Run — expect all 5 new tests passing.**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py -v
```
Expected: `10 passed, 0 failed` (5 Hamiltonian + 5 diagonalization); remaining `charge_operator_matrix_elements` + `transmon_summary` tests still raise `NotImplementedError` in Task 4 — they are not yet present in the file, so the run is green.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/transmon.py dispersive_readout/tests/test_transmon.py
git commit -m "feat(stage06): transmon diagonalization with ground-shift convention

Task 3 of Module 1 plan. np.linalg.eigh on the charge-basis Hamiltonian;
returns N_transmon lowest eigenpairs with ground energy shifted to 0.
Validates orthonormality, ordering, and that omega_01 falls in the
Marxer device band."
```

---

## Task 4 — Charge matrix elements and transmon summary

**Files:**
- Modify: `dispersive_readout/physics/transmon.py`
- Modify: `dispersive_readout/tests/test_transmon.py`

- [ ] **Step 1: Append failing tests.**

Append to `dispersive_readout/tests/test_transmon.py`:
```python
# -- matrix elements + summary -------------------------------------------------

def test_charge_matrix_elements_shape():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert n_mat.shape == (REFERENCE_DEVICE.truncation.N_transmon,
                          REFERENCE_DEVICE.truncation.N_transmon)


def test_charge_matrix_is_hermitian():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert np.allclose(n_mat, n_mat.conj().T, atol=1e-10)


def test_charge_matrix_element_01_dominant():
    """|<0|n̂|1>| should be larger than |<0|n̂|2>| (selection rule in deep transmon regime)."""
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert abs(n_mat[0, 1]) > 10.0 * abs(n_mat[0, 2])


def test_transmon_summary_keys():
    summary = transmon_summary(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    required = {
        "omega_01", "omega_12", "alpha", "E_J_over_E_C",
        "charge_dispersion_01", "n_matrix_01", "n_matrix_12",
    }
    assert required.issubset(summary.keys()), f"missing keys: {required - summary.keys()}"


def test_transmon_summary_values_plausible():
    s = transmon_summary(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    # anharmonicity negative (transmon); ~-200 MHz
    alpha_hz = s["alpha"] / _TWO_PI
    assert -260e6 < alpha_hz < -160e6, f"alpha/2π = {alpha_hz/1e6:.1f} MHz outside plausible band"
    # E_J/E_C ≈ 74
    assert 70 < s["E_J_over_E_C"] < 80
```

- [ ] **Step 2: Run — expect 5 failures (NotImplementedError).**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py -v -k "matrix or summary"
```
Expected: 5 errors/failures from `NotImplementedError`.

- [ ] **Step 3: Implement `charge_operator_matrix_elements` and `transmon_summary`.**

Replace the stubs in `dispersive_readout/physics/transmon.py`:
```python
def charge_operator_matrix_elements(
    eigenstates: np.ndarray,
    trunc: TruncationParams,
) -> np.ndarray:
    """<j|n_hat|k> in the truncated transmon eigenbasis.

    The charge operator is diagonal in the charge basis with entries
    n = -N//2, ..., +N//2, so the transformed matrix is
        n_mat[j, k] = sum_q conj(eigenstates[q, j]) * n_q * eigenstates[q, k].
    For the standard real-symmetric charge Hamiltonian, eigenstates can be
    chosen real, so n_mat is real symmetric in practice.
    """
    N = trunc.N_charge
    n_values = np.arange(-(N // 2), N // 2 + 1, dtype=float)
    return eigenstates.conj().T @ (n_values[:, None] * eigenstates)


def transmon_summary(params: TransmonParams, trunc: TruncationParams) -> dict:
    """Summary dict for logging and spot checks.

    Returns a dict with keys (all rad/s unless noted):
      omega_01, omega_12: transition frequencies.
      alpha:              anharmonicity = omega_12 - omega_01.
      E_J_over_E_C:       dimensionless.
      charge_dispersion_01: ω_01(n_g=0.5) − ω_01(n_g=0), in rad/s.
      n_matrix_01, n_matrix_12: |<0|n̂|1>|, |<1|n̂|2>|.
    """
    energies, states = diagonalize_transmon(params, trunc)
    n_mat = charge_operator_matrix_elements(states, trunc)

    omega_01 = energies[1] - energies[0]
    omega_12 = energies[2] - energies[1]

    # Charge dispersion: re-diagonalize at n_g = 0.5 and compare omega_01.
    from dataclasses import replace
    params_half = replace(params, n_g=0.5)
    energies_half, _ = diagonalize_transmon(params_half, trunc)
    omega_01_half = energies_half[1] - energies_half[0]

    return {
        "omega_01": omega_01,
        "omega_12": omega_12,
        "alpha": omega_12 - omega_01,
        "E_J_over_E_C": params.E_J / params.E_C,
        "charge_dispersion_01": abs(omega_01_half - omega_01),
        "n_matrix_01": abs(n_mat[0, 1]),
        "n_matrix_12": abs(n_mat[1, 2]),
    }
```

- [ ] **Step 4: Run — expect all tests passing.**

Run:
```bash
pytest dispersive_readout/tests/test_transmon.py -v
```
Expected: `15 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/transmon.py dispersive_readout/tests/test_transmon.py
git commit -m "feat(stage06): transmon charge matrix elements and summary diagnostics

Task 4 of Module 1 plan. <j|n̂|k> transformed into the eigenbasis;
transmon_summary returns omega_01, omega_12, alpha, E_J/E_C, charge
dispersion of |0>, and dominant matrix elements."
```

---

## Task 5 — Validation V1: anharmonicity and charge dispersion

**Files:**
- Create: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Write V1 tests.**

`dispersive_readout/tests/test_physics_validation.py`:
```python
"""Gating physics-validation tests V1–V4 for Module 1.

If any test in this file fails, Module 1 is not complete. Do not loosen
tolerances; debug the implementation. See Module 1 spec §4 for the
tolerance rationale.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, TruncationParams
from dispersive_readout.physics.transmon import diagonalize_transmon, transmon_summary

_TWO_PI = 2.0 * math.pi


# -- V1: transmon eigenstructure ----------------------------------------------

def test_V1a_transmon_anharmonicity_matches_perturbative():
    """Koch 2007: for E_J/E_C >> 1, α ≈ -E_C to leading order.

    Tolerance 5% (spec §4 V1). Tighter than this is unrealistic because
    of higher-order corrections in E_C/E_J.
    """
    d = REFERENCE_DEVICE
    s = transmon_summary(d.transmon, d.truncation)
    alpha_predicted = -d.transmon.E_C
    alpha_numerical = s["alpha"]
    rel_error = abs(alpha_numerical - alpha_predicted) / abs(alpha_predicted)
    assert rel_error < 0.05, (
        f"V1a FAIL: alpha/2π numerical = {alpha_numerical/_TWO_PI/1e6:.2f} MHz, "
        f"predicted = {alpha_predicted/_TWO_PI/1e6:.2f} MHz, rel err = {rel_error:.3%}"
    )


def test_V1b_transmon_charge_dispersion_below_1kHz():
    """In the deep transmon regime, |ω_01(n_g=0.5) − ω_01(n_g=0)| < 1 kHz.

    Also acts as a N_charge = 13 sufficiency check — if charge dispersion
    is artifactually large, the charge ladder is truncating too tightly.
    """
    d = REFERENCE_DEVICE
    s = transmon_summary(d.transmon, d.truncation)
    charge_dispersion_hz = s["charge_dispersion_01"] / _TWO_PI
    assert charge_dispersion_hz < 1e3, (
        f"V1b FAIL: charge dispersion of |0⟩–|1⟩ transition = "
        f"{charge_dispersion_hz:.1f} Hz, expected < 1000 Hz."
    )
```

- [ ] **Step 2: Run — expect failures? No, V1 passes if Tasks 2–4 implemented correctly.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py -v
```
Expected: `2 passed`. If either fails, stop and flag per spec §8 "What to flag to the human" item 1 — do not loosen tolerances.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): V1 validation — anharmonicity and charge dispersion

Task 5 of Module 1 plan. Anharmonicity matches Koch 2007 leading-order
predictions to 5%; charge dispersion of |0>-|1> transition < 1 kHz at
N_charge = 13, confirming truncation sufficiency."
```

---

## Task 6 — Transmon truncation convergence

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Append convergence tests.**

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
# -- truncation convergence (non-gating but required before moving on) --------

def test_N_charge_convergence_below_1e_6_relative():
    """omega_01 must change by < 1e-6 (relative) when N_charge goes 13 → 21."""
    d = REFERENCE_DEVICE
    trunc_small = TruncationParams(N_charge=13, N_transmon=5, N_resonator=15)
    trunc_large = TruncationParams(N_charge=21, N_transmon=5, N_resonator=15)
    e_small, _ = diagonalize_transmon(d.transmon, trunc_small)
    e_large, _ = diagonalize_transmon(d.transmon, trunc_large)
    omega_01_small = e_small[1] - e_small[0]
    omega_01_large = e_large[1] - e_large[0]
    rel = abs(omega_01_large - omega_01_small) / abs(omega_01_small)
    assert rel < 1e-6, f"N_charge not converged at 13: rel change = {rel:.2e}"


def test_N_transmon_top_level_charge_dispersion_small():
    """The highest kept transmon level's charge dispersion must also be < 10 kHz.

    If this fails, N_transmon is large enough that the top level is charge-sensitive
    and the truncation is including states that are not accurately described.
    """
    from dataclasses import replace
    d = REFERENCE_DEVICE
    energies_0, _ = diagonalize_transmon(d.transmon, d.truncation)
    energies_half, _ = diagonalize_transmon(replace(d.transmon, n_g=0.5), d.truncation)
    top_level = d.truncation.N_transmon - 1
    dispersion_hz = abs((energies_half[top_level] - energies_0[top_level])) / _TWO_PI
    assert dispersion_hz < 10e3, f"top level (j={top_level}) dispersion = {dispersion_hz:.1f} Hz"
```

- [ ] **Step 2: Run — expect both passing.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py -v
```
Expected: `4 passed`.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): transmon truncation convergence at N_charge=13, N_transmon=5

Task 6 of Module 1 plan. omega_01 converged to < 1e-6 relative when
extending the charge ladder to 21 states; top transmon level charge
dispersion below 10 kHz confirms N_transmon = 5 stays in the deep
transmon regime."
```

---

## Task 7 — Analytic dispersive-shift formulas

**Files:**
- Create: `dispersive_readout/physics/dispersive.py`
- Create: `dispersive_readout/tests/test_dispersive.py`

- [ ] **Step 1: Write failing tests.**

`dispersive_readout/tests/test_dispersive.py`:
```python
"""Dispersive-shift formula tests (analytic + numerical)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE
from dispersive_readout.physics.dispersive import (
    dispersive_shift_full,
    dispersive_shift_from_simulation,
    dispersive_shift_two_level,
)
from dispersive_readout.physics.transmon import (
    charge_operator_matrix_elements,
    diagonalize_transmon,
)

_TWO_PI = 2.0 * math.pi


# -- two-level formula ---------------------------------------------------------

def test_two_level_formula_positive_delta():
    chi = dispersive_shift_two_level(g=_TWO_PI * 100e6, Delta=_TWO_PI * 1e9)
    assert chi == pytest.approx((_TWO_PI * 100e6) ** 2 / (_TWO_PI * 1e9))


def test_two_level_formula_negative_delta_gives_negative_chi():
    """Reference device has Δ < 0 (qubit below resonator) → χ < 0."""
    chi = dispersive_shift_two_level(g=_TWO_PI * 120e6, Delta=-_TWO_PI * 2.7e9)
    assert chi < 0


# -- multi-level formula -------------------------------------------------------

def test_dispersive_shift_full_shape():
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    assert chi.shape == (d.truncation.N_transmon,)


def test_dispersive_shift_full_gives_plausible_half_splitting():
    """(χ_1 − χ_0)/2 should be roughly -5 MHz for reference device (spec §1.2)."""
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_j = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    chi_half_hz = (chi_j[1] - chi_j[0]) / 2.0 / _TWO_PI
    assert -10e6 < chi_half_hz < -1e6, (
        f"multi-level χ = {chi_half_hz/1e6:.2f} MHz outside plausible band"
    )


def test_dispersive_shift_full_sign_matches_two_level():
    """Δ < 0 → full formula's χ_1 − χ_0 also < 0."""
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_j = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    assert (chi_j[1] - chi_j[0]) < 0
```

- [ ] **Step 2: Run — ModuleNotFoundError.**

Run:
```bash
pytest dispersive_readout/tests/test_dispersive.py -v
```
Expected: `ModuleNotFoundError: No module named 'dispersive_readout.physics.dispersive'`.

- [ ] **Step 3: Implement analytic χ formulas (stub the numerical extractor for Task 8).**

`dispersive_readout/physics/dispersive.py`:
```python
"""Analytic and numerical dispersive-shift formulas.

χ convention: χ ≡ (χ_1 − χ_0)/2, the half-splitting observable in readout.
dispersive_shift_full returns per-level χ_j; the caller computes the
half-splitting from those as needed.
"""
from __future__ import annotations

import numpy as np

from .config import DeviceConfig


def dispersive_shift_two_level(g: float, Delta: float) -> float:
    """Two-level-limit dispersive shift: χ = g² / Δ.

    Inputs are in rad/s; output in rad/s. For Δ < 0 (qubit below resonator,
    the reference device's regime) this is negative.
    """
    return (g ** 2) / Delta


def dispersive_shift_full(
    energies: np.ndarray,
    n_matrix: np.ndarray,
    g: float,
    omega_r: float,
) -> np.ndarray:
    """Multi-level per-level dispersive shifts χ_j.

    χ_j = sum_{k != j} |g <j|n̂|k>|² [ 1/(ω_j - ω_k - ω_r) - 1/(ω_j - ω_k + ω_r) ]

    The observable readout shift is (χ_1 − χ_0)/2.
    """
    N = len(energies)
    chi = np.zeros(N, dtype=float)
    for j in range(N):
        total = 0.0
        for k in range(N):
            if k == j:
                continue
            coupling_sq = (g * abs(n_matrix[j, k])) ** 2
            delta_jk = energies[j] - energies[k]
            denom_minus = delta_jk - omega_r
            denom_plus = delta_jk + omega_r
            if denom_minus == 0.0 or denom_plus == 0.0:
                raise ValueError(
                    f"Degeneracy in denominators at j={j}, k={k}: "
                    f"delta={delta_jk}, omega_r={omega_r}"
                )
            total += coupling_sq * (1.0 / denom_minus - 1.0 / denom_plus)
        chi[j] = total
    return chi


def dispersive_shift_from_simulation(device: DeviceConfig) -> float:
    """Extract χ ≡ (χ₁ − χ₀)/2 from the dressed Jaynes-Cummings spectrum."""
    raise NotImplementedError  # Task 8
```

- [ ] **Step 4: Run — expect 5 passing, 1 stub-raise for the numerical test (absent, added Task 8).**

Run:
```bash
pytest dispersive_readout/tests/test_dispersive.py -v
```
Expected: `5 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/dispersive.py dispersive_readout/tests/test_dispersive.py
git commit -m "feat(stage06): analytic dispersive-shift formulas (two-level + multi-level)

Task 7 of Module 1 plan. chi = g²/Delta in the 2-level limit; per-level
chi_j from the full perturbative expression using the transmon eigen-
energies and charge matrix elements. Numerical extractor stubbed for
Task 8."
```

---

## Task 8 — Dispersive shift from dressed JC spectrum

**Files:**
- Modify: `dispersive_readout/physics/dispersive.py`
- Modify: `dispersive_readout/tests/test_dispersive.py`

- [ ] **Step 1: Add failing test.**

Append to `dispersive_readout/tests/test_dispersive.py`:
```python
# -- numerical from dressed spectrum -------------------------------------------

def test_dispersive_shift_from_simulation_matches_sign_and_magnitude():
    """Dressed-spectrum χ must have the same sign as the two-level estimate and
    magnitude within a factor of 3 (loose — tight comparison is V2 in Task 9)."""
    d = REFERENCE_DEVICE
    chi_num = dispersive_shift_from_simulation(d)
    # Δ = ω_01 − ω_r for reference device is negative, so χ < 0
    assert chi_num < 0
    # Magnitude: naive two-level estimate is |g²/Δ| ≈ (2π·120e6)² / (2π·2.7e9)
    chi_naive_mag = (d.coupling.g ** 2) / (d.resonator.omega_r - d.transmon.E_J)  # very rough
    # Don't pin magnitude here — use a wide factor-3 band on the naive scale.
    assert 1e5 < abs(chi_num) / _TWO_PI < 3e7, (
        f"chi/2π magnitude = {abs(chi_num)/_TWO_PI/1e6:.2f} MHz outside plausible band"
    )


def test_dispersive_shift_from_simulation_is_real():
    """The dressed spectrum is Hermitian; χ must be real."""
    d = REFERENCE_DEVICE
    chi_num = dispersive_shift_from_simulation(d)
    assert np.imag(chi_num) == pytest.approx(0.0, abs=1e-15)
```

- [ ] **Step 2: Run — NotImplementedError.**

Run:
```bash
pytest dispersive_readout/tests/test_dispersive.py::test_dispersive_shift_from_simulation_matches_sign_and_magnitude -v
```
Expected: `NotImplementedError`.

- [ ] **Step 3: Implement the extractor.**

Replace the stub in `dispersive_readout/physics/dispersive.py`:
```python
def dispersive_shift_from_simulation(device: DeviceConfig) -> float:
    """Extract χ ≡ (χ₁ − χ₀)/2 from the dressed Jaynes-Cummings spectrum.

    Builds the full zero-drive Hamiltonian in the
    (transmon ⊗ resonator) basis, diagonalizes it, identifies the dressed
    states adiabatically connected to the bare product states
    |q,n⟩ for q ∈ {0,1} and n ∈ {0,1} (by overlap), and returns
        ((E(1,1) − E(1,0)) − (E(0,1) − E(0,0))) / 2.
    """
    import qutip as qt

    from .transmon import charge_operator_matrix_elements, diagonalize_transmon

    tr = device.truncation
    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    Nq = tr.N_transmon
    Nr = tr.N_resonator

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    H_q = qt.tensor(qt.Qobj(np.diag(energies)), qt.qeye(Nr))
    H_r = device.resonator.omega_r * a.dag() * a
    n_op_q = qt.tensor(qt.Qobj(n_mat), qt.qeye(Nr))
    H_c = device.coupling.g * n_op_q * (a + a.dag())
    H = H_q + H_r + H_c

    eigvals, eigvecs = H.eigenstates()

    # Identify dressed states by max-overlap with bare product kets.
    bare_energies = {}
    for q in (0, 1):
        for n in (0, 1):
            bare_ket = qt.tensor(qt.basis(Nq, q), qt.basis(Nr, n))
            overlaps = np.array(
                [abs((bare_ket.dag() * v).full().item()) ** 2 for v in eigvecs]
            )
            idx = int(np.argmax(overlaps))
            bare_energies[(q, n)] = float(eigvals[idx])

    return (
        (bare_energies[(1, 1)] - bare_energies[(1, 0)])
        - (bare_energies[(0, 1)] - bare_energies[(0, 0)])
    ) / 2.0
```

- [ ] **Step 4: Run — expect all dispersive tests passing.**

Run:
```bash
pytest dispersive_readout/tests/test_dispersive.py -v
```
Expected: `7 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/dispersive.py dispersive_readout/tests/test_dispersive.py
git commit -m "feat(stage06): numerical chi extractor from dressed JC spectrum

Task 8 of Module 1 plan. Builds zero-drive H in transmon ⊗ resonator,
diagonalizes, matches dressed states to bare product kets by overlap,
returns (chi_1 - chi_0)/2. Sign and magnitude match two-level
expectation."
```

---

## Task 9 — Validation V2: χ analytic vs numerical ≤ 1e-4

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Append V2 test.**

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
from dispersive_readout.physics.dispersive import (
    dispersive_shift_from_simulation,
    dispersive_shift_full,
)
from dispersive_readout.physics.transmon import charge_operator_matrix_elements


# -- V2: dispersive shift numerical vs analytic --------------------------------

def test_V2_chi_analytic_vs_numerical_within_1e_minus_4():
    """Multi-level analytic formula must match the dressed-spectrum number.

    Tolerance 1e-4 relative (spec §4 V2). Failure means either the dressed
    identification is picking the wrong states (degeneracy / truncation bug)
    or the analytic formula has a sign / index error.
    """
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_per_level = dispersive_shift_full(
        energies, n_mat, d.coupling.g, d.resonator.omega_r,
    )
    chi_analytic_half = (chi_per_level[1] - chi_per_level[0]) / 2.0
    chi_numerical_half = dispersive_shift_from_simulation(d)
    rel_error = abs(chi_analytic_half - chi_numerical_half) / abs(chi_analytic_half)
    assert rel_error < 1e-4, (
        f"V2 FAIL: chi analytic/2π = {chi_analytic_half/_TWO_PI/1e6:.4f} MHz, "
        f"numerical/2π = {chi_numerical_half/_TWO_PI/1e6:.4f} MHz, "
        f"rel err = {rel_error:.2e}"
    )
```

- [ ] **Step 2: Run — expect pass.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py::test_V2_chi_analytic_vs_numerical_within_1e_minus_4 -v
```
Expected: `1 passed`. If this fails, do not loosen the tolerance — investigate the overlap-based dressed-state identification first (common failure: wrong `k` index in the analytic sum).

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): V2 validation — chi analytic and numerical agree to 1e-4

Task 9 of Module 1 plan. Multi-level perturbative chi_j vs. half-
splitting extracted from the dressed JC spectrum, both evaluated on
REFERENCE_DEVICE. Gate for the Lindblad dynamics work in Tasks 10+."
```

---

## Task 10 — Lindblad collapse operators

**Files:**
- Create: `dispersive_readout/physics/lindblad.py`
- Create: `dispersive_readout/tests/test_lindblad.py`

- [ ] **Step 1: Write failing tests.**

`dispersive_readout/tests/test_lindblad.py`:
```python
"""Lindblad collapse-operator and Hamiltonian-builder tests."""
from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.lindblad import (
    build_collapse_operators,
    build_hamiltonian,
)

_TWO_PI = 2.0 * math.pi


def test_collapse_operators_returned_as_qobj_list():
    d = REFERENCE_DEVICE
    c_ops = build_collapse_operators(d, d.truncation.N_transmon, d.truncation.N_resonator)
    assert isinstance(c_ops, list)
    for op in c_ops:
        assert isinstance(op, qt.Qobj)


def test_collapse_operator_shapes_match_full_hilbert_space():
    d = REFERENCE_DEVICE
    total_dim = d.truncation.N_transmon * d.truncation.N_resonator
    c_ops = build_collapse_operators(d, d.truncation.N_transmon, d.truncation.N_resonator)
    for op in c_ops:
        assert op.shape == (total_dim, total_dim)


def test_collapse_list_has_expected_channel_count():
    """Reference device (n_th > 0) builds:
       2 resonator ops (decay + heating)
       + 2*(Nq-1) qubit transitions (relaxation + thermal heating)
       + (Nq-1) dephasing ops (one per upper level).

    Catches accidental omission of any channel. Quantitative rate correctness
    is validated end-to-end by the V3 / V4 physics tests.
    """
    d = REFERENCE_DEVICE
    Nq = d.truncation.N_transmon
    c_ops = build_collapse_operators(d, Nq, d.truncation.N_resonator)
    expected = 2 + 2 * (Nq - 1) + (Nq - 1)
    assert len(c_ops) == expected, f"expected {expected} collapse ops, got {len(c_ops)}"


def test_collapse_list_reduces_when_thermal_zero():
    """When n_th = 0 and n_th_r = 0, thermal-excitation operators must be omitted."""
    from dataclasses import replace
    d = REFERENCE_DEVICE
    d_cold = replace(d, decoherence=replace(d.decoherence, n_th=0.0))
    c_cold = build_collapse_operators(
        d_cold, d_cold.truncation.N_transmon, d_cold.truncation.N_resonator
    )
    c_warm = build_collapse_operators(
        d, d.truncation.N_transmon, d.truncation.N_resonator
    )
    assert len(c_cold) < len(c_warm)


def test_build_hamiltonian_returns_drift_and_drive_spec():
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 5e6, duration=500e-9, detuning=0.0)
    H0, drive_spec = build_hamiltonian(d, drv, frame="rotating")
    assert isinstance(H0, qt.Qobj)
    assert H0.isherm
    # QuTiP-compatible H(t) form: [op, callable]
    assert isinstance(drive_spec, list) and len(drive_spec) == 2
    op, func = drive_spec
    assert isinstance(op, qt.Qobj)
    assert callable(func)
    # Drive envelope at t=0 should be ~0 (rising edge not yet reached)
    eps0 = func(0.0, {})
    assert abs(eps0) < drv.amplitude * 0.1


def test_drive_envelope_peaks_near_midpulse():
    d = REFERENCE_DEVICE
    drv = DriveParams(amplitude=_TWO_PI * 5e6, duration=500e-9, detuning=0.0)
    _, drive_spec = build_hamiltonian(d, drv, frame="rotating")
    _, func = drive_spec
    mid = drv.duration / 2.0
    # Midpulse should be within 1% of full amplitude
    assert abs(func(mid, {}) - drv.amplitude) < 0.01 * drv.amplitude
```

- [ ] **Step 2: Run — ModuleNotFoundError.**

Run:
```bash
pytest dispersive_readout/tests/test_lindblad.py -v
```
Expected: `ModuleNotFoundError: No module named 'dispersive_readout.physics.lindblad'`.

- [ ] **Step 3: Implement `build_collapse_operators` (stub `build_hamiltonian` for Task 11).**

`dispersive_readout/physics/lindblad.py`:
```python
"""Collapse operators and Hamiltonian builder for the readout simulation.

Collapse operators are constructed in the dressed transmon eigenbasis, not
the bare charge basis and not a 2-level approximation. This matters for
Module 2 leakage tracking. Pure dephasing in the multi-level transmon uses
the convention (|j><j| − |0><0|) for j > 0 with per-level rate scaling.
See Blais et al. RMP 93, 025005 (2021) §III.E.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import qutip as qt
from scipy.special import erf

from .config import DeviceConfig, DriveParams
from .transmon import charge_operator_matrix_elements, diagonalize_transmon


def build_collapse_operators(
    device: DeviceConfig,
    transmon_basis_dim: int,
    resonator_dim: int,
) -> list[qt.Qobj]:
    """Lindblad collapse operators in the (transmon ⊗ resonator) Hilbert space.

    Channels:
      1. Resonator decay:   sqrt(κ (1 + n_th))  a
      2. Resonator heating: sqrt(κ  n_th)       a†     (only if n_th > 0)
      3. Qubit relaxation:  per-transition amplitudes in dressed transmon basis,
         scaled by |<j|n̂|k>|² relative to |<0|n̂|1>|² for |j+1⟩ → |j⟩ transitions.
      4. Qubit pure dephasing: sqrt(2 γ_φ) (|j><j| − |0><0|) for j = 1, ..., Nq−1.
      5. Qubit thermal heating: reverse of (3) scaled by n_th (only if n_th > 0).
    """
    tr = device.truncation
    Nq = transmon_basis_dim
    Nr = resonator_dim
    kappa = device.resonator.kappa
    gamma_1 = device.decoherence.gamma_1
    gamma_phi = device.decoherence.gamma_phi
    n_th = device.decoherence.n_th

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    c_ops: list[qt.Qobj] = []

    # 1. Resonator decay
    c_ops.append(np.sqrt(kappa * (1.0 + n_th)) * a)

    # 2. Resonator heating (only if bath is warm)
    if n_th > 0:
        c_ops.append(np.sqrt(kappa * n_th) * a.dag())

    # Build charge matrix elements for relaxation scaling in the dressed basis.
    _, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)
    # Normalize so |<0|n̂|1>|² is the reference scale (rate γ_1 applies to |1>→|0>).
    ref_sq = abs(n_mat[0, 1]) ** 2

    # 3. Qubit relaxation: |j+1> -> |j> for j = 0, 1, ..., Nq-2
    for j in range(Nq - 1):
        scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
        rate = gamma_1 * scale * (1.0 + n_th)
        if rate > 0:
            op = qt.basis(Nq, j) * qt.basis(Nq, j + 1).dag()
            c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    # 4. Qubit pure dephasing: rate sqrt(2 γ_φ) for each upper level
    for j in range(1, Nq):
        if gamma_phi > 0:
            proj = (
                qt.basis(Nq, j) * qt.basis(Nq, j).dag()
                - qt.basis(Nq, 0) * qt.basis(Nq, 0).dag()
            )
            c_ops.append(np.sqrt(2.0 * gamma_phi) * qt.tensor(proj, qt.qeye(Nr)))

    # 5. Qubit thermal heating (reverse direction)
    if n_th > 0:
        for j in range(Nq - 1):
            scale = abs(n_mat[j, j + 1]) ** 2 / ref_sq
            rate = gamma_1 * scale * n_th
            if rate > 0:
                op = qt.basis(Nq, j + 1) * qt.basis(Nq, j).dag()
                c_ops.append(np.sqrt(rate) * qt.tensor(op, qt.qeye(Nr)))

    return c_ops


def build_hamiltonian(
    device: DeviceConfig,
    drive_params: DriveParams,
    frame: Literal["rotating", "dispersive"] = "rotating",
) -> tuple[qt.Qobj, list]:
    """Rotating-frame drift + time-dependent drive spec. Implemented in Task 11."""
    raise NotImplementedError  # Task 11
```

- [ ] **Step 4: Run tests not gated on `build_hamiltonian`.**

Run:
```bash
pytest dispersive_readout/tests/test_lindblad.py -v -k "collapse or reduces"
```
Expected: 4 passed, 2 stub-errors for `build_hamiltonian` tests (they fail with NotImplementedError — expected).

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/lindblad.py dispersive_readout/tests/test_lindblad.py
git commit -m "feat(stage06): Lindblad collapse operators in dressed transmon basis

Task 10 of Module 1 plan. Resonator decay + heating, per-transition qubit
relaxation with charge-matrix-element scaling, multi-level pure dephasing
following Blais RMP 2021 §III.E, thermal channels guarded on n_th > 0.
Hamiltonian builder stubbed for Task 11."
```

---

## Task 11 — Rotating-frame Hamiltonian + drive envelope

**Files:**
- Modify: `dispersive_readout/physics/lindblad.py`

- [ ] **Step 1: Implement `build_hamiltonian` (tests already written in Task 10).**

Replace the stub in `dispersive_readout/physics/lindblad.py`:
```python
def build_hamiltonian(
    device: DeviceConfig,
    drive_params: DriveParams,
    frame: Literal["rotating", "dispersive"] = "rotating",
) -> tuple[qt.Qobj, list]:
    """Drift Hamiltonian + QuTiP-compatible drive spec.

    Rotating frame at ω_d = ω_r + detuning:
      H_q  = Σ_j (ω_j − j ω_d) |j><j| ⊗ I_r
      H_r  = (ω_r − ω_d) a†a
      H_c  = g Σ_{jk} <j|n̂|k> |j><k| ⊗ (a + a†)
      H_drive(t) = ε(t) (a + a†)

    ε(t) is an erf-difference flat-top pulse with Gaussian edges of width σ.

    'dispersive' frame is not implemented in Module 1; it is reserved for
    validation-only use. Calling with frame='dispersive' raises
    NotImplementedError. Do not silently return rotating frame instead.
    """
    if frame not in ("rotating",):
        raise NotImplementedError(f"frame '{frame}' not implemented in Module 1")

    tr = device.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    energies, eigenstates = diagonalize_transmon(device.transmon, tr)
    n_mat = charge_operator_matrix_elements(eigenstates, tr)

    # Drive frequency: on resonance with resonator plus optional detuning.
    omega_d = device.resonator.omega_r + drive_params.detuning

    # Transmon term in rotating frame: diag(omega_j - j * omega_d)
    qubit_diag = np.array([energies[j] - j * omega_d for j in range(Nq)])
    H_q = qt.tensor(qt.Qobj(np.diag(qubit_diag)), qt.qeye(Nr))

    # Resonator term: (omega_r - omega_d) a†a
    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    H_r = (device.resonator.omega_r - omega_d) * a.dag() * a

    # Coupling term: g * <j|n̂|k> * |j><k| ⊗ (a + a†)
    # Retain only adjacent selection-rule contributions; full matrix keeps all.
    n_op_q = qt.tensor(qt.Qobj(n_mat), qt.qeye(Nr))
    H_c = device.coupling.g * n_op_q * (a + a.dag())

    H0 = H_q + H_r + H_c

    # Drive operator: ε(t) (a + a†)
    drive_op = a + a.dag()

    # Envelope: erf-difference flat-top with sigma_edge gaussian rise/fall.
    eps_0 = drive_params.amplitude
    t_end = drive_params.duration
    sigma = drive_params.edge_sigma
    t_rise = 3.0 * sigma
    t_fall = t_end - t_rise
    if t_fall <= t_rise + 2.0 * sigma:
        raise ValueError(
            f"Drive duration {t_end*1e9:.1f} ns too short for rise/fall "
            f"width {sigma*1e9:.1f} ns; need t_end > 6*sigma + 2*sigma."
        )

    def envelope(t: float, args: dict) -> float:
        return 0.5 * eps_0 * (erf((t - t_rise) / sigma) - erf((t - t_fall) / sigma))

    return H0, [drive_op, envelope]
```

- [ ] **Step 2: Run all lindblad tests.**

Run:
```bash
pytest dispersive_readout/tests/test_lindblad.py -v
```
Expected: `6 passed`.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/physics/lindblad.py
git commit -m "feat(stage06): rotating-frame Hamiltonian with erf-difference drive envelope

Task 11 of Module 1 plan. Transmon + resonator drift in the frame of
omega_d = omega_r + detuning; coupling term via full charge-matrix
elements in dressed basis; drive is a flat-top with Gaussian edges
(scipy.special.erf). Dispersive-frame option intentionally raises
NotImplementedError — reserved for later validation only."
```

---

## Task 12 — `simulate_readout` and `ReadoutResult`

**Files:**
- Create: `dispersive_readout/physics/readout_model.py`
- Create: `dispersive_readout/tests/test_readout_model.py`

- [ ] **Step 1: Write the first two failing tests (smoke + state-dependent response).**

`dispersive_readout/tests/test_readout_model.py`:
```python
"""Readout-model integration tests (dynamics + IQ separation + assignment fidelity)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, DriveParams
from dispersive_readout.physics.readout_model import (
    AssignmentFidelityResult,
    ReadoutResult,
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)

_TWO_PI = 2.0 * math.pi


def _default_drive() -> DriveParams:
    return DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)


def test_simulate_readout_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    t_list = np.linspace(0.0, 500e-9, 101)
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0, t_list=t_list)
    assert isinstance(res, ReadoutResult)
    assert res.t.shape == (101,)
    assert res.a_expectation.shape == (101,)
    assert res.photon_number.shape == (101,)
    assert res.qubit_populations.shape == (101, d.truncation.N_transmon)
    assert res.drive_envelope.shape == (101,)


def test_simulate_readout_photon_number_is_nonnegative():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=0)
    assert np.all(res.photon_number >= -1e-10)


def test_simulate_readout_populations_sum_to_unity():
    d = REFERENCE_DEVICE
    res = simulate_readout(d, _default_drive(), initial_qubit_state=1)
    totals = res.qubit_populations.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-3)


def test_simulate_readout_iq_trajectories_separate_for_0_and_1():
    """The steady-state ⟨a⟩ for |0> and |1> must differ by a measurable amount."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    res0 = simulate_readout(d, drv, initial_qubit_state=0)
    res1 = simulate_readout(d, drv, initial_qubit_state=1)
    # Compare mean ⟨a⟩ over the last 20% of the window (after the rise transient)
    tail0 = res0.a_expectation[int(0.8 * len(res0.a_expectation)):]
    tail1 = res1.a_expectation[int(0.8 * len(res1.a_expectation)):]
    sep = abs(tail0.mean() - tail1.mean())
    assert sep > 0.05, f"IQ separation {sep:.4f} too small — dispersive regime lost?"
```

- [ ] **Step 2: Run — ModuleNotFoundError.**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `simulate_readout` + `ReadoutResult` (stub fidelity for Task 17).**

`dispersive_readout/physics/readout_model.py`:
```python
"""Pulsed readout simulation, IQ trajectories, assignment fidelity.

simulate_readout integrates the Lindblad master equation with QuTiP mesolve.
The observable is <a>(t) — the homodyne signal. Runtime-checks that the
mean photon number during readout stays below an N_resonator-dependent
ceiling and prints a warning if not.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import qutip as qt

from .config import DeviceConfig, DriveParams
from .lindblad import build_collapse_operators, build_hamiltonian


@dataclass(frozen=True)
class ReadoutResult:
    """Single readout trajectory.

    All arrays share first-axis length T = len(t).
    a_expectation is complex (homodyne-observable resonator coherent amplitude).
    photon_number is real (for truncation monitoring).
    qubit_populations is (T, N_transmon).
    """
    t: np.ndarray
    a_expectation: np.ndarray
    photon_number: np.ndarray
    qubit_populations: np.ndarray
    drive_envelope: np.ndarray
    device: DeviceConfig
    drive_params: DriveParams
    initial_qubit_state: int

    def integrated_iq(self, window: tuple[float, float]) -> complex:
        """Return the integrated complex IQ amplitude over [window[0], window[1]]."""
        t0, t1 = window
        mask = (self.t >= t0) & (self.t <= t1)
        if mask.sum() < 2:
            raise ValueError(f"Window {window} contains fewer than 2 samples")
        return np.trapz(self.a_expectation[mask], self.t[mask])


@dataclass(frozen=True)
class AssignmentFidelityResult:
    F_assign: float
    F_assign_uncertainty: float
    centroid_0: complex
    centroid_1: complex
    snr: float
    separation_distance: float
    integration_window: tuple[float, float]
    n_shots: int
    noise_model: str


_MAX_PHOTON_RATIO = 0.33  # warn if mean photon > 1/3 of N_resonator


def simulate_readout(
    device: DeviceConfig,
    drive_params: DriveParams,
    initial_qubit_state: int,
    initial_resonator_state: str = "vacuum",
    t_list: np.ndarray | None = None,
    solver_options: dict | None = None,
) -> ReadoutResult:
    """Integrate the Lindblad ME for the transmon-resonator system under a pulsed drive.

    initial_qubit_state = 0 or 1 selects the dressed transmon eigenket at t=0.
    initial_resonator_state = 'vacuum' is the only supported option in Module 1.
    """
    if initial_qubit_state not in (0, 1):
        raise ValueError("initial_qubit_state must be 0 or 1.")
    if initial_resonator_state != "vacuum":
        raise NotImplementedError(f"only 'vacuum' supported, got '{initial_resonator_state}'.")

    tr = device.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    H0, drive_spec = build_hamiltonian(device, drive_params, frame="rotating")
    c_ops = build_collapse_operators(device, Nq, Nr)

    psi0 = qt.tensor(qt.basis(Nq, initial_qubit_state), qt.basis(Nr, 0))

    if t_list is None:
        t_list = np.linspace(0.0, drive_params.duration, 501)

    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nr))
    n_photon = a.dag() * a
    # Populations: P(|j⟩) = Tr_r(|j><j| ⊗ I · ρ)
    e_ops_pop = [
        qt.tensor(qt.basis(Nq, j) * qt.basis(Nq, j).dag(), qt.qeye(Nr))
        for j in range(Nq)
    ]

    opts = qt.Options()
    opts.nsteps = 10000
    opts.atol = 1e-10
    opts.rtol = 1e-8
    if solver_options:
        for k, v in solver_options.items():
            setattr(opts, k, v)

    result = qt.mesolve(
        H=[H0, drive_spec],
        rho0=psi0,
        tlist=t_list,
        c_ops=c_ops,
        e_ops=[a, n_photon, *e_ops_pop],
        options=opts,
    )

    a_exp = np.asarray(result.expect[0], dtype=complex)
    n_exp = np.asarray(result.expect[1], dtype=float)
    pops = np.stack([np.asarray(result.expect[2 + j], dtype=float) for j in range(Nq)], axis=1)

    # Runtime check — flag if we're close to Fock truncation.
    max_photon = float(n_exp.max())
    if max_photon > _MAX_PHOTON_RATIO * Nr:
        warnings.warn(
            f"Mean photon number peaked at {max_photon:.2f} with N_resonator={Nr}. "
            f"Truncation may be insufficient — consider N_resonator={Nr + 10}.",
            RuntimeWarning,
        )

    # Record drive envelope (convenience — same callable used in drive_spec)
    _, envelope_fn = drive_spec
    drive_env = np.array([envelope_fn(ti, {}) for ti in t_list], dtype=float)

    return ReadoutResult(
        t=np.asarray(t_list, dtype=float),
        a_expectation=a_exp,
        photon_number=n_exp,
        qubit_populations=pops,
        drive_envelope=drive_env,
        device=device,
        drive_params=drive_params,
        initial_qubit_state=initial_qubit_state,
    )


def compute_assignment_fidelity(
    result_ground: ReadoutResult,
    result_excited: ReadoutResult,
    integration_window: tuple[float, float],
    n_shots: int = 10000,
    noise_model: Literal["ideal", "gaussian"] = "gaussian",
) -> AssignmentFidelityResult:
    raise NotImplementedError  # Task 17


def snr_vs_integration_time(
    device: DeviceConfig,
    drive_params: DriveParams,
    t_integration_values: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError  # Task 18
```

- [ ] **Step 4: Run — expect 4 tests passing (plus NotImplementedError on later ones which are not present yet).**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py -v
```
Expected: `4 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/readout_model.py dispersive_readout/tests/test_readout_model.py
git commit -m "feat(stage06): simulate_readout pulsed dynamics via QuTiP mesolve

Task 12 of Module 1 plan. Integrates Lindblad ME in the rotating frame;
returns ReadoutResult dataclass with IQ trajectory, photon number, per-
level populations, drive envelope. Runtime warning when mean photon
number exceeds 1/3 of N_resonator."
```

---

## Task 13 — Validation V3: T₁ recovery

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Add V3 test.**

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
from dataclasses import replace
from dispersive_readout.physics.config import DriveParams
from dispersive_readout.physics.readout_model import simulate_readout


# -- V3: T1 recovery -----------------------------------------------------------

def test_V3_T1_recovery_from_undriven_decay():
    """|1> population decays at rate γ_1 when drive=0, γ_φ=0, n_th=0.

    Tolerance 1% (spec §4 V3). Failure ≥ 1% means either the dressed-basis
    relaxation operator or the state preparation is off. Loosen only as
    a last resort and only with human sign-off.
    """
    d = REFERENCE_DEVICE
    d_pure = replace(
        d,
        decoherence=replace(d.decoherence, gamma_phi=0.0, n_th=0.0),
    )
    T1 = 1.0 / d_pure.decoherence.gamma_1
    drv_zero = DriveParams(amplitude=0.0, duration=5.0 * T1, detuning=0.0)
    t_list = np.linspace(0.0, 5.0 * T1, 300)

    res = simulate_readout(d_pure, drv_zero, initial_qubit_state=1, t_list=t_list)

    # Fit P1(t) = exp(-γ_fit t). Linear fit in log-space over the unsaturated region.
    p1 = res.qubit_populations[:, 1]
    mask = p1 > 1e-3  # avoid log(0)
    log_p1 = np.log(p1[mask])
    coef = np.polyfit(res.t[mask], log_p1, 1)
    gamma_fit = -coef[0]

    rel_err = abs(gamma_fit - d_pure.decoherence.gamma_1) / d_pure.decoherence.gamma_1
    assert rel_err < 0.01, (
        f"V3 FAIL: γ_fit = {gamma_fit:.3e}, γ_input = {d_pure.decoherence.gamma_1:.3e}, "
        f"rel err = {rel_err:.3%}."
    )
```

- [ ] **Step 2: Run — expect pass.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py::test_V3_T1_recovery_from_undriven_decay -v
```
Expected: `1 passed`. Runtime: ~5–15 s for the 5·T₁ simulation.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): V3 validation — T1 recovery within 1%

Task 13 of Module 1 plan. Undriven decay of |1> fits exp(-γ·t) and
recovers the input γ_1 to under 1%."
```

---

## Task 14 — Validation V4a: T₂/Ramsey recovery

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Add V4a test.**

First, ensure `import qutip as qt` is present at the top of `dispersive_readout/tests/test_physics_validation.py` (add it once at the top with the other imports if not already there — the test below constructs a QuTiP density matrix directly).

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
# -- V4a: T2 from Ramsey-like simulation ---------------------------------------

def test_V4a_T2_recovery_from_pure_dephasing():
    """Off-diagonal |rho_01| decays at rate (γ_1/2 + γ_φ). With γ_1 = 0 only γ_φ remains.

    Procedure: initialize in (|0> + |1>)/sqrt(2), evolve with drive=0, γ_1=0;
    read off the coherence |rho_01(t)| and fit exp(-γ_phi t).
    """
    d = REFERENCE_DEVICE
    d_deph = replace(
        d,
        decoherence=replace(d.decoherence, gamma_1=0.0, n_th=0.0),
    )

    tr = d_deph.truncation
    Nq = tr.N_transmon
    Nr = tr.N_resonator

    # Initial state: (|0>+|1>)/sqrt(2) ⊗ |vacuum>
    psi0 = qt.tensor(
        (qt.basis(Nq, 0) + qt.basis(Nq, 1)).unit(),
        qt.basis(Nr, 0),
    )
    rho0 = psi0 * psi0.dag()

    drv_zero = DriveParams(amplitude=0.0, duration=5.0 / d_deph.decoherence.gamma_phi, detuning=0.0)
    t_list = np.linspace(0.0, drv_zero.duration, 200)

    from dispersive_readout.physics.lindblad import build_collapse_operators, build_hamiltonian
    H0, drive_spec = build_hamiltonian(d_deph, drv_zero, frame="rotating")
    c_ops = build_collapse_operators(d_deph, Nq, Nr)
    # Coherence operator |0><1| ⊗ I_r
    coherence_op = qt.tensor(qt.basis(Nq, 0) * qt.basis(Nq, 1).dag(), qt.qeye(Nr))
    opts = qt.Options(nsteps=10000, atol=1e-10, rtol=1e-8)
    out = qt.mesolve(
        H=[H0, drive_spec], rho0=rho0, tlist=t_list, c_ops=c_ops,
        e_ops=[coherence_op], options=opts,
    )
    rho01 = np.asarray(out.expect[0], dtype=complex)
    coh_mag = np.abs(rho01)

    mask = coh_mag > 1e-3
    log_c = np.log(coh_mag[mask])
    coef = np.polyfit(t_list[mask], log_c, 1)
    gamma_phi_fit = -coef[0]

    rel_err = abs(gamma_phi_fit - d_deph.decoherence.gamma_phi) / d_deph.decoherence.gamma_phi
    assert rel_err < 0.01, (
        f"V4a FAIL: γ_phi_fit = {gamma_phi_fit:.3e}, γ_phi_input = "
        f"{d_deph.decoherence.gamma_phi:.3e}, rel err = {rel_err:.3%}."
    )
```

- [ ] **Step 2: Run — expect pass.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py::test_V4a_T2_recovery_from_pure_dephasing -v
```
Expected: `1 passed`. Runtime ~15–30 s.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): V4a validation — T2/gamma_phi recovery within 1%

Task 14 of Module 1 plan. Pure-dephasing coherence decay of an equal
superposition fits exp(-gamma_phi·t) and recovers the input to < 1%."
```

---

## Task 15 — Validation V4b: Purcell rate

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Add V4b test.**

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
# -- V4b: Purcell decay --------------------------------------------------------

def test_V4b_purcell_rate_matches_analytic():
    """With γ_1 = γ_φ = n_th = 0 and κ > 0, an excited qubit decays at rate
    γ_Purcell = (g/Δ)² κ.

    Tolerance 5% (spec §4 V4). Purcell formula is itself perturbative.
    """
    d = REFERENCE_DEVICE
    d_purcell = replace(
        d,
        decoherence=replace(d.decoherence, gamma_1=0.0, gamma_phi=0.0, n_th=0.0),
    )
    # Δ = omega_01 - omega_r (using bare qubit frequency from summary)
    from dispersive_readout.physics.transmon import transmon_summary
    s = transmon_summary(d_purcell.transmon, d_purcell.truncation)
    omega_q = s["omega_01"]
    Delta = omega_q - d_purcell.resonator.omega_r
    g = d_purcell.coupling.g
    gamma_purcell_analytic = (g / Delta) ** 2 * d_purcell.resonator.kappa

    # Simulate 5 / gamma_purcell worth of free evolution.
    drv_zero = DriveParams(amplitude=0.0, duration=5.0 / gamma_purcell_analytic, detuning=0.0)
    t_list = np.linspace(0.0, drv_zero.duration, 200)

    res = simulate_readout(d_purcell, drv_zero, initial_qubit_state=1, t_list=t_list)
    p1 = res.qubit_populations[:, 1]
    mask = p1 > 1e-3
    log_p1 = np.log(p1[mask])
    coef = np.polyfit(res.t[mask], log_p1, 1)
    gamma_fit = -coef[0]

    rel_err = abs(gamma_fit - gamma_purcell_analytic) / gamma_purcell_analytic
    assert rel_err < 0.05, (
        f"V4b FAIL: γ_Purcell_fit = {gamma_fit:.3e}, analytic = {gamma_purcell_analytic:.3e}, "
        f"rel err = {rel_err:.3%}."
    )
```

- [ ] **Step 2: Run — expect pass.**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py::test_V4b_purcell_rate_matches_analytic -v
```
Expected: `1 passed`.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): V4b validation — Purcell rate matches (g/Δ)²κ to 5%

Task 15 of Module 1 plan. Excited-state decay with γ_1=γ_φ=0, kappa>0
recovers the Purcell rate within the 5% perturbative tolerance."
```

---

## Task 16 — Resonator-truncation convergence (N_resonator)

**Files:**
- Modify: `dispersive_readout/tests/test_physics_validation.py`

- [ ] **Step 1: Add convergence test for the simulator.**

Append to `dispersive_readout/tests/test_physics_validation.py`:
```python
# -- N_resonator convergence ---------------------------------------------------

def test_N_resonator_convergence_during_readout():
    """End-of-pulse <a> must change by < 1e-3 (absolute) when N_resonator: 15 → 20.

    If this fails at the reference amplitude, bump N_resonator to 25 and
    flag to human per spec §8 item 3.
    """
    from dataclasses import replace
    from dispersive_readout.physics.config import TruncationParams
    d_15 = REFERENCE_DEVICE
    d_20 = replace(
        REFERENCE_DEVICE,
        truncation=TruncationParams(N_charge=13, N_transmon=5, N_resonator=20),
    )
    drv = DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)
    t_list = np.linspace(0.0, drv.duration, 151)

    res15 = simulate_readout(d_15, drv, initial_qubit_state=0, t_list=t_list)
    res20 = simulate_readout(d_20, drv, initial_qubit_state=0, t_list=t_list)

    end15 = res15.a_expectation[-20:].mean()
    end20 = res20.a_expectation[-20:].mean()
    diff = abs(end15 - end20)
    assert diff < 1e-3, (
        f"N_resonator NOT CONVERGED at 15: |<a>_15 − <a>_20| = {diff:.2e}. "
        f"Bump to 25 and re-flag."
    )
```

- [ ] **Step 2: Run — expect pass (may take ~30 s to run both simulations).**

Run:
```bash
pytest dispersive_readout/tests/test_physics_validation.py::test_N_resonator_convergence_during_readout -v
```
Expected: `1 passed`.

- [ ] **Step 3: Commit.**

```bash
git add dispersive_readout/tests/test_physics_validation.py
git commit -m "test(stage06): N_resonator=15 convergence at reference drive amplitude

Task 16 of Module 1 plan. End-of-pulse <a> unchanged (<1e-3 absolute)
when bumping N_resonator 15→20; flag to human if this regresses under
larger drive amplitudes."
```

---

## Task 17 — Assignment fidelity

**Files:**
- Modify: `dispersive_readout/physics/readout_model.py`
- Modify: `dispersive_readout/tests/test_readout_model.py`

- [ ] **Step 1: Append failing tests.**

Append to `dispersive_readout/tests/test_readout_model.py`:
```python
def test_assignment_fidelity_returns_dataclass_with_expected_fields():
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    assert isinstance(f, AssignmentFidelityResult)
    assert 0.0 <= f.F_assign <= 1.0
    assert f.separation_distance > 0.0
    assert f.snr > 0.0


def test_assignment_fidelity_ideal_is_at_least_as_large_as_gaussian():
    """With no shot noise, fidelity is bounded above by the 'gaussian' noise case."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (400e-9, 500e-9)
    f_g = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
    f_i = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="ideal")
    assert f_i.F_assign >= f_g.F_assign - 1e-9


def test_assignment_fidelity_sanity_on_reference_device():
    """At default params, the reference device should hit ≥ 95% assignment fidelity."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    r0 = simulate_readout(d, drv, initial_qubit_state=0)
    r1 = simulate_readout(d, drv, initial_qubit_state=1)
    window = (300e-9, 500e-9)
    f = compute_assignment_fidelity(r0, r1, window, n_shots=20000, noise_model="gaussian")
    assert f.F_assign >= 0.95, f"Reference device fidelity {f.F_assign:.4f} below 0.95 — flag to human."
```

- [ ] **Step 2: Run — NotImplementedError.**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py::test_assignment_fidelity_returns_dataclass_with_expected_fields -v
```
Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `compute_assignment_fidelity`.**

Replace the stub in `dispersive_readout/physics/readout_model.py`:
```python
def compute_assignment_fidelity(
    result_ground: ReadoutResult,
    result_excited: ReadoutResult,
    integration_window: tuple[float, float],
    n_shots: int = 10000,
    noise_model: Literal["ideal", "gaussian"] = "gaussian",
) -> AssignmentFidelityResult:
    """Single-shot assignment fidelity from two simulated trajectories.

    Integrates ⟨a⟩(t) over the window for each of |0> and |1> to get
    deterministic centroids; adds per-shot circular Gaussian noise in IQ space
    (when noise_model='gaussian'); classifies shots with the perpendicular-
    bisector discriminator; returns F = 1 - (P(1|0) + P(0|1)) / 2.
    """
    if noise_model not in ("ideal", "gaussian"):
        raise ValueError(f"noise_model must be 'ideal' or 'gaussian', got {noise_model!r}")

    c0 = result_ground.integrated_iq(integration_window)
    c1 = result_excited.integrated_iq(integration_window)
    separation = abs(c1 - c0)
    if separation == 0:
        raise ValueError("IQ centroids coincide — dispersive regime lost or window too short.")

    # Shot-noise standard deviation: κ/4 per √t in coherent-state scaling.
    # Concrete value for circular Gaussian: σ = √(κ/2 · Δt) × (unitless scale).
    # Simpler, standard baseline: σ set so that for a vacuum state the noise
    # equals half the minimal-uncertainty level, giving SNR = separation / σ.
    # Here we derive σ directly from the integrated shot noise for a coherent
    # vacuum over the window — standard homodyne noise variance.
    t0, t1 = integration_window
    window_duration = t1 - t0
    kappa = result_ground.device.resonator.kappa
    # Coherent-state quadrature variance per unit time × window; see Gambetta 2008.
    sigma = np.sqrt(kappa * window_duration / 2.0) if noise_model == "gaussian" else 0.0

    rng = np.random.default_rng(seed=42)

    if sigma == 0.0:
        # Ideal case: all shots fall on the centroid; F = 1 if centroids differ.
        F = 1.0
        F_unc = 0.0
    else:
        draws_0 = c0 + sigma * (
            rng.standard_normal(n_shots) + 1j * rng.standard_normal(n_shots)
        ) / np.sqrt(2.0)
        draws_1 = c1 + sigma * (
            rng.standard_normal(n_shots) + 1j * rng.standard_normal(n_shots)
        ) / np.sqrt(2.0)
        # Perpendicular-bisector discriminator:
        # decision axis = unit vector from c0 to c1; midpoint = (c0+c1)/2.
        axis = (c1 - c0) / separation
        midpoint = 0.5 * (c0 + c1)
        proj_0 = np.real((draws_0 - midpoint) * np.conj(axis))
        proj_1 = np.real((draws_1 - midpoint) * np.conj(axis))
        # Classify: proj > 0 → predicted |1>
        wrong_0 = np.mean(proj_0 > 0)   # P(1|0)
        wrong_1 = np.mean(proj_1 <= 0)  # P(0|1)
        F = 1.0 - 0.5 * (wrong_0 + wrong_1)
        # Bootstrap uncertainty (binomial-standard-error of F)
        F_unc = np.sqrt(F * (1.0 - F) / n_shots)

    return AssignmentFidelityResult(
        F_assign=float(F),
        F_assign_uncertainty=float(F_unc),
        centroid_0=complex(c0),
        centroid_1=complex(c1),
        snr=float(separation / sigma) if sigma > 0 else float("inf"),
        separation_distance=float(separation),
        integration_window=(float(t0), float(t1)),
        n_shots=int(n_shots),
        noise_model=noise_model,
    )
```

- [ ] **Step 4: Run — expect 3 new passing.**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py -v
```
Expected: `7 passed` (4 from Task 12 + 3 here).

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/readout_model.py dispersive_readout/tests/test_readout_model.py
git commit -m "feat(stage06): single-shot assignment fidelity with Gaussian IQ noise

Task 17 of Module 1 plan. Integrates <a> over a window to get IQ
centroids; samples shots with circular Gaussian shot-noise variance
sigma² = kappa·Δt/2; classifies with the perpendicular-bisector
discriminator; returns F_assign with bootstrap uncertainty. Reference
device passes ≥ 95% fidelity at default drive."
```

---

## Task 18 — SNR vs integration time

**Files:**
- Modify: `dispersive_readout/physics/readout_model.py`
- Modify: `dispersive_readout/tests/test_readout_model.py`

- [ ] **Step 1: Append failing test.**

Append to `dispersive_readout/tests/test_readout_model.py`:
```python
def test_snr_vs_integration_time_shape_and_monotone_rise():
    """SNR should rise roughly as sqrt(t) over short integrations and plateau."""
    d = REFERENCE_DEVICE
    drv = _default_drive()
    t_int = np.linspace(50e-9, 450e-9, 9)
    snr = snr_vs_integration_time(d, drv, t_int)
    assert snr.shape == (9,)
    # Monotone rise before plateau: first half must be non-decreasing (tolerating noise)
    early = snr[: len(snr) // 2]
    assert np.all(np.diff(early) >= -0.05), f"SNR not rising: {early}"
    # Final SNR should exceed the first SNR
    assert snr[-1] > snr[0]
```

- [ ] **Step 2: Run — NotImplementedError.**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py::test_snr_vs_integration_time_shape_and_monotone_rise -v
```
Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `snr_vs_integration_time`.**

Replace the stub in `dispersive_readout/physics/readout_model.py`:
```python
def snr_vs_integration_time(
    device: DeviceConfig,
    drive_params: DriveParams,
    t_integration_values: np.ndarray,
) -> np.ndarray:
    """SNR(t_int) = |c_1(t_int) - c_0(t_int)| / sigma(t_int).

    Runs one |0> trajectory and one |1> trajectory out to the maximum
    t_integration value, then computes SNR for each window (0, t_int).
    """
    if np.any(t_integration_values <= 0):
        raise ValueError("t_integration_values must be strictly positive.")

    t_max = float(t_integration_values.max())
    if t_max > drive_params.duration:
        raise ValueError(
            f"t_integration max {t_max*1e9:.1f} ns exceeds drive duration "
            f"{drive_params.duration*1e9:.1f} ns."
        )

    # Use a fine grid so cumulative-trapezoid integration is accurate at all windows.
    t_list = np.linspace(0.0, drive_params.duration, 1001)
    r0 = simulate_readout(device, drive_params, initial_qubit_state=0, t_list=t_list)
    r1 = simulate_readout(device, drive_params, initial_qubit_state=1, t_list=t_list)

    snrs = np.zeros_like(t_integration_values, dtype=float)
    kappa = device.resonator.kappa
    for i, t_int in enumerate(t_integration_values):
        c0 = r0.integrated_iq((0.0, float(t_int)))
        c1 = r1.integrated_iq((0.0, float(t_int)))
        sep = abs(c1 - c0)
        sigma = np.sqrt(kappa * t_int / 2.0)
        snrs[i] = sep / sigma if sigma > 0 else float("inf")
    return snrs
```

- [ ] **Step 4: Run — expect pass.**

Run:
```bash
pytest dispersive_readout/tests/test_readout_model.py -v
```
Expected: `8 passed`.

- [ ] **Step 5: Commit.**

```bash
git add dispersive_readout/physics/readout_model.py dispersive_readout/tests/test_readout_model.py
git commit -m "feat(stage06): snr_vs_integration_time for Figure 1b

Task 18 of Module 1 plan. Runs one |0> and one |1> trajectory to t_max,
then computes SNR(t_int) = |c_1 - c_0| / sqrt(kappa·t_int/2) for each
requested window."
```

---

## Task 19 — Figure 1 stage script (`dispersive_readout_simulation.py`)

**Files:**
- Create: `06_Dispersive_Readout/dispersive_readout_simulation.py`
- Create: `06_Dispersive_Readout/figures/.gitkeep`

- [ ] **Step 1: Create the `figures/` placeholder.**

Shell:
```bash
mkdir -p 06_Dispersive_Readout/figures
touch 06_Dispersive_Readout/figures/.gitkeep
```

- [ ] **Step 2: Create the script.**

`06_Dispersive_Readout/dispersive_readout_simulation.py`:
```python
#!/usr/bin/env python3
"""Stage 06 Figure 1 driver — dispersive-readout simulator demonstration.

Produces `figures/dispersive_readout_simulation.png` with three panels:
  (a) IQ trajectories for initial |0> and |1>
  (b) SNR vs integration time, with short-τ ∝ √τ asymptote overlay
  (c) Assignment fidelity vs κ/|χ|, with vertical marker at κ/|χ| = 2

The numerical χ used in panel (c) is extracted from the dressed JC
spectrum (dispersive.dispersive_shift_from_simulation).

Matches the 01–05 stage-script convention: walks up from the script
location to the repo root (identified by the presence of a sibling
``dispersive_readout`` package or a ``.git`` directory), prepends that
path to sys.path, then imports from the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "dispersive_readout").exists() or (_p / ".git").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from dispersive_readout.physics.config import (
    REFERENCE_DEVICE,
    DriveParams,
    ResonatorParams,
)
from dispersive_readout.physics.dispersive import dispersive_shift_from_simulation
from dispersive_readout.physics.readout_model import (
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)

_TWO_PI = 2.0 * math.pi

OUTPUT = Path(__file__).resolve().parent / "figures" / "dispersive_readout_simulation.png"

# Drive amplitude in rad/s; 2 MHz × 2π is a mid-range readout drive for the
# reference device and keeps the mean photon number under the N_resonator=15
# ceiling.
_DRIVE = DriveParams(amplitude=_TWO_PI * 2e6, duration=500e-9, detuning=0.0)


def _panel_a_iq_trajectories(ax) -> None:
    r0 = simulate_readout(REFERENCE_DEVICE, _DRIVE, initial_qubit_state=0)
    r1 = simulate_readout(REFERENCE_DEVICE, _DRIVE, initial_qubit_state=1)
    i0, q0 = r0.a_expectation.real, r0.a_expectation.imag
    i1, q1 = r1.a_expectation.real, r1.a_expectation.imag
    ax.plot(i0, q0, color="#1f77b4", lw=1.6, label="|0⟩")
    ax.plot(i1, q1, color="#d62728", lw=1.6, label="|1⟩")
    ax.plot(i0[0], q0[0], "o", color="#1f77b4", markersize=5)
    ax.plot(i0[-1], q0[-1], "s", color="#1f77b4", markersize=5)
    ax.plot(i1[0], q1[0], "o", color="#d62728", markersize=5)
    ax.plot(i1[-1], q1[-1], "s", color="#d62728", markersize=5)
    ax.set_xlabel("I  (rad/s)")
    ax.set_ylabel("Q  (rad/s)")
    ax.set_title("(a) IQ trajectories")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)


def _panel_b_snr(ax) -> None:
    t_int = np.linspace(30e-9, 450e-9, 30)
    snr = snr_vs_integration_time(REFERENCE_DEVICE, _DRIVE, t_int)
    ax.loglog(t_int * 1e9, snr, "-", color="black", lw=1.6, label="simulation")
    # Short-τ asymptote: SNR ∝ √τ (with prefactor fit to first few points).
    prefactor = snr[0] / np.sqrt(t_int[0])
    asym = prefactor * np.sqrt(t_int)
    ax.loglog(t_int * 1e9, asym, "--", color="#999999", lw=1.2, label=r"$\propto\sqrt{\tau}$")
    ax.set_xlabel("integration time  τ (ns)")
    ax.set_ylabel("SNR")
    ax.set_title("(b) SNR vs integration time")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, which="both")


def _panel_c_fidelity_vs_kappa_over_chi(ax) -> None:
    chi = dispersive_shift_from_simulation(REFERENCE_DEVICE)
    chi_abs = abs(chi)
    ratios = np.logspace(-1.0, 1.0, 11)  # κ/|χ| from 0.1 to 10
    fidelities = np.zeros_like(ratios)
    window = (300e-9, 500e-9)
    for i, ratio in enumerate(ratios):
        new_kappa = ratio * chi_abs
        dev = replace(
            REFERENCE_DEVICE,
            resonator=ResonatorParams(omega_r=REFERENCE_DEVICE.resonator.omega_r, kappa=new_kappa),
        )
        r0 = simulate_readout(dev, _DRIVE, initial_qubit_state=0)
        r1 = simulate_readout(dev, _DRIVE, initial_qubit_state=1)
        f = compute_assignment_fidelity(r0, r1, window, n_shots=5000, noise_model="gaussian")
        fidelities[i] = f.F_assign
    ax.semilogx(ratios, fidelities, "-o", color="black", lw=1.4, markersize=4)
    ax.axvline(2.0, color="#ca0020", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"$\kappa / |\chi|$")
    ax.set_ylabel(r"$F_{\mathrm{assign}}$")
    ax.set_title("(c) Assignment fidelity vs κ/|χ|")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.2, which="both")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    _panel_a_iq_trajectories(axes[0])
    _panel_b_snr(axes[1])
    _panel_c_fidelity_vs_kappa_over_chi(axes[2])
    fig.suptitle(
        "Dispersive-readout simulation — reference device (Marxer arXiv:2508.16437)",
        fontsize=12,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script and verify the PNG exists.**

Shell:
```bash
python 06_Dispersive_Readout/dispersive_readout_simulation.py
ls -la 06_Dispersive_Readout/figures/dispersive_readout_simulation.png
```
Expected: `Wrote .../dispersive_readout_simulation.png` printed, followed by a line showing the PNG has non-zero size. Total runtime ~60–180 s because panel (c) sweeps 11 fidelity curves.

- [ ] **Step 4: Visually inspect the figure.**

Open `06_Dispersive_Readout/figures/dispersive_readout_simulation.png` and confirm:
- Panel (a) shows two clearly separated curves; each has a dot at the origin and a square at the tail.
- Panel (b) rises roughly as √τ on the log-log plot; the grey dashed line traces the asymptote.
- Panel (c) has high fidelity near κ/|χ| = 2 and rolls off to both sides; vertical red dashed line at 2.

If any of these are visibly wrong, stop and debug rather than committing.

- [ ] **Step 5: Commit.**

```bash
git add 06_Dispersive_Readout/dispersive_readout_simulation.py \
        06_Dispersive_Readout/figures/.gitkeep \
        06_Dispersive_Readout/figures/dispersive_readout_simulation.png
git commit -m "feat(stage06): Figure 1 — dispersive_readout_simulation.py (3-panel)

Task 19 of Module 1 plan. (a) IQ trajectories for |0>/|1> at reference
device + drive; (b) SNR vs integration time with short-τ √τ asymptote;
(c) assignment fidelity vs κ/|χ| sweep. Output at
06_Dispersive_Readout/figures/dispersive_readout_simulation.png."
```

---

## Task 20 — Public API surface in `dispersive_readout.physics.__init__`

**Files:**
- Modify: `dispersive_readout/physics/__init__.py`

- [ ] **Step 1: Expose the Module 1 public API.**

Replace the contents of `dispersive_readout/physics/__init__.py`:
```python
"""Public API for the dispersive-readout physics foundation (Module 1).

Stable entry points for scripts and downstream modules:
    - Config dataclasses and REFERENCE_DEVICE
    - simulate_readout, ReadoutResult
    - compute_assignment_fidelity, AssignmentFidelityResult
    - snr_vs_integration_time
    - transmon_summary (for quick device sanity checks)
    - dispersive_shift_{two_level, full, from_simulation} (for validation)
"""
from .config import (
    REFERENCE_DEVICE,
    CouplingParams,
    DecoherenceParams,
    DeviceConfig,
    DriveParams,
    ResonatorParams,
    TransmonParams,
    TruncationParams,
)
from .dispersive import (
    dispersive_shift_from_simulation,
    dispersive_shift_full,
    dispersive_shift_two_level,
)
from .readout_model import (
    AssignmentFidelityResult,
    ReadoutResult,
    compute_assignment_fidelity,
    simulate_readout,
    snr_vs_integration_time,
)
from .transmon import (
    charge_basis_hamiltonian,
    charge_operator_matrix_elements,
    diagonalize_transmon,
    transmon_summary,
)

__all__ = [
    # config
    "REFERENCE_DEVICE",
    "CouplingParams",
    "DecoherenceParams",
    "DeviceConfig",
    "DriveParams",
    "ResonatorParams",
    "TransmonParams",
    "TruncationParams",
    # transmon
    "charge_basis_hamiltonian",
    "charge_operator_matrix_elements",
    "diagonalize_transmon",
    "transmon_summary",
    # dispersive
    "dispersive_shift_from_simulation",
    "dispersive_shift_full",
    "dispersive_shift_two_level",
    # readout model
    "AssignmentFidelityResult",
    "ReadoutResult",
    "compute_assignment_fidelity",
    "simulate_readout",
    "snr_vs_integration_time",
]
```

- [ ] **Step 2: Smoke-import from the package root.**

Shell:
```bash
python -c "from dispersive_readout.physics import REFERENCE_DEVICE, simulate_readout, compute_assignment_fidelity; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Run the full test suite one more time.**

Run:
```bash
pytest dispersive_readout/tests/ -v
```
Expected: all tests green (~30 tests passing).

- [ ] **Step 4: Commit.**

```bash
git add dispersive_readout/physics/__init__.py
git commit -m "feat(stage06): public API for dispersive_readout.physics (Module 1)

Task 20 of Module 1 plan. Re-exports config dataclasses, REFERENCE_DEVICE,
transmon helpers, dispersive-shift formulas, simulate_readout, assignment
fidelity, and SNR helper with explicit __all__."
```

---

## Task 21 — Stage README

**Files:**
- Create: `06_Dispersive_Readout/README.md`

- [ ] **Step 1: Write the stage README.**

`06_Dispersive_Readout/README.md`:
```markdown
# 06 — Dispersive Readout

Stage 06 deliverables for the Quantum_Simulation repo. All importable code lives in the repo-root `dispersive_readout/` package; this folder holds runnable stage scripts and their generated figures.

## Module 1 — Physics foundation (this commit)

**What's here:** A QuTiP-backed Jaynes-Cummings + Lindblad simulator for a transmon qubit dispersively coupled to a readout resonator. The reference device follows Marxer et al., arXiv:2508.16437 (IQM Munich, Aug 2025), with Bengtsson et al., Phys. Rev. Lett. 132, 100603 (2024) as the secondary reference.

**Outputs in this folder:**
- `dispersive_readout_simulation.py` — Figure 1 driver (IQ trajectories, SNR vs integration time, assignment fidelity vs κ/|χ|).
- `figures/dispersive_readout_simulation.png` — the rendered figure.

**Package entry points (import from `dispersive_readout.physics`):**
- `REFERENCE_DEVICE`, `DeviceConfig`, `TransmonParams`, `ResonatorParams`, `CouplingParams`, `DecoherenceParams`, `DriveParams`, `TruncationParams`
- `simulate_readout`, `ReadoutResult`
- `compute_assignment_fidelity`, `AssignmentFidelityResult`
- `snr_vs_integration_time`
- `transmon_summary`, `diagonalize_transmon`, `charge_operator_matrix_elements`, `charge_basis_hamiltonian`
- `dispersive_shift_two_level`, `dispersive_shift_full`, `dispersive_shift_from_simulation`

**Gating validation tests passed (`dispersive_readout/tests/test_physics_validation.py`):**
- V1a anharmonicity vs Koch 2007 leading-order, within 5%.
- V1b charge dispersion of |0⟩–|1⟩ transition < 1 kHz at N_charge = 13.
- V2 χ analytic vs numerical (multi-level) within 1e-4.
- V3 T₁ recovery from undriven decay within 1%.
- V4a T₂/γ_φ recovery from pure-dephasing decay within 1%.
- V4b Purcell rate γ = (g/Δ)²κ within 5%.
- Truncation convergence: N_charge 13→21 < 1e-6 rel; N_resonator 15→20 < 1e-3 abs on end-of-pulse ⟨a⟩.

## How to run

```bash
# From repo root:
pytest dispersive_readout/tests/ -v            # package tests (all physics validation)
python 06_Dispersive_Readout/dispersive_readout_simulation.py   # regenerates Figure 1
```

## Modules 2–4 (pending)

Stage 06 will add:
- Module 2 — error-budget decomposition (script: `error_budget_decomposition.py`).
- Module 3 — characterization recovery (script + CLI: `characterize.py` wrapping `dispersive_readout.characterization.cli`).
- Module 4 — sensitivity / Pareto analysis (script: `sensitivity_pareto_analysis.py`).

Scripts are runnable with `python 06_Dispersive_Readout/<name>.py`; tests for each module are colocated with their package code under `dispersive_readout/<module>/` — see `dispersive_readout/tests/`.
```

- [ ] **Step 2: Commit.**

```bash
git add 06_Dispersive_Readout/README.md
git commit -m "docs(stage06): stage README summarizing Module 1 deliverables

Task 21 of Module 1 plan. Lists entry points, validation tests passed,
and where Modules 2–4 will land."
```

---

## Task 22 — Module 1 completion checkpoint

**Files:** None modified.

- [ ] **Step 1: Run the full test suite.**

Run:
```bash
pytest dispersive_readout/tests/ -v
```
Expected: every test passing. Count must be ≥ 20 (spec §7 gate).

- [ ] **Step 2: Confirm the figure is up to date.**

Shell:
```bash
python 06_Dispersive_Readout/dispersive_readout_simulation.py
```
Expected: `Wrote .../dispersive_readout_simulation.png`. Re-inspect the PNG if anything about the package changed.

- [ ] **Step 3: Run through the spec §7 review checklist.**

Verify each item:

- [ ] All four validation tests (V1, V2, V3, V4) passing.
- [ ] Test count ≥ 20 (confirm with `pytest dispersive_readout/tests/ --collect-only -q | wc -l`).
- [ ] `N_charge`, `N_transmon`, `N_resonator` truncations verified (covered by convergence tests in Tasks 6, 16).
- [ ] `REFERENCE_DEVICE` parameters cite Marxer arXiv:2508.16437 in config docstring.
- [ ] All physics code in rad/s internally; Hz only at I/O boundary.
- [ ] No magic numbers outside `config.py` (grep spot-check: `grep -rn "2.7e9\|7.3e9\|5e6\|120e6" dispersive_readout/physics/` should return only `config.py`).
- [ ] Figure 1 renders at 150 DPI.
- [ ] `dispersive_readout/physics/__init__.py` exposes a clean public API (Task 20).
- [ ] `06_Dispersive_Readout/README.md` references Module 1 deliverables.
- [ ] Commit history is clean: `git log --oneline` shows one commit per task, with `feat(stage06):`, `test(stage06):`, or `docs(stage06):` prefixes.

- [ ] **Step 4: If all checks pass, tag the Module 1 commit.**

```bash
git tag -a stage06-module1 -m "Stage 06 Module 1 — dispersive-readout physics foundation complete"
```
(Do not push the tag without explicit user confirmation.)

- [ ] **Step 5: Stop.** Module 2 does not start until this plan's review-checklist items are all green and the human has signed off.

---

## Notes for the executing engineer

- **QuTiP version.** `pip show qutip` should report 4.7+ (4.7.5 or 5.x both fine). If QuTiP is not installed, `pip install qutip scipy numpy matplotlib`.
- **Runtime budget.** Full test suite runtime on a modern laptop: ~60–120 s; the Figure 1 sweep adds ~60–180 s. V3/V4a run the longest integrations (~10–30 s each).
- **Debugging philosophy (from the preceding CLAUDE.md).** If a validation test fails, do not loosen tolerance. Inspect:
  1. Sign of Δ, χ — convention bugs look like off-by-2 or off-by-sign.
  2. Dressed-state identification in `dispersive_shift_from_simulation` — wrong overlap ordering is the most common cause of V2 failure at large coupling.
  3. Collapse-operator scaling in `lindblad.py` — if V3 is off by √2 or 2, check the dephasing-operator prefactor.
  4. QuTiP `mesolve` stability — bump `nsteps` or tighten `rtol` before blaming the physics.
- **When to flag to the human (spec §8).** Tolerances missed, wrong χ sign, mean photon > 5 at N=15, unphysical fidelities, `mesolve` instability, or a temptation to pre-optimize. These are all stop-and-surface conditions.
