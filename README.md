# Real-Time Quantum Simulation of Gauge Theories and Open Quantum Systems

This repository contains code + validation artifacts for a two-workstream physics simulation portfolio:

- **Gauge simulation (Schwinger model, 1+1D QED)** using a gauge-eliminated spin-chain Hamiltonian, exact diagonalization (ED), symmetry-preserving VQE benchmarking, and real-time non-equilibrium dynamics (string breaking via electric-field quench).
- **Open quantum systems (pNRQCD-motivated quarkonium in medium)** using a singlet–octet Lindblad model (1 ⊕ 8), validated against analytic limits, and used to study sequential suppression and time-dependent (Bjorken) cooling.

The repository is organized as a progressive validation chain: each later deliverable builds on earlier baselines.

---

## Repository layout

- `docs/`  
  Project overview and theory background PDFs.

- `01_Validation-Baseline/`  
  Baseline correctness checks:
  - Pure-gauge U(1) Monte Carlo vs exact Wilson-loop area law
  - Gauge-eliminated Schwinger Hamiltonian consistency checks
  - Lindblad/OQS baseline dynamics vs analytic solution(s)

- `02_Static Benchmarks/`  
  Static benchmarking for the Schwinger Hamiltonian:
  - Sector-projected VQE benchmarks (N=4)
  - Extension to larger systems (e.g., N=8) with staged optimization + Trotter de-risking

- `03_Non-Equilibrium Gauge Dynamics/`  
  Real-time gauge dynamics:
  - Electric-field quench protocol and diagnostics for string breaking (heavy vs light mass regimes)

- `04_Continuum Physics Results/`  
  Physics-grade “packaging” results:
  - Continuum extrapolation of the massless Schwinger-model mass gap
  - Sequential suppression (1S vs 2S-like) from the Lindblad model
  - Optional time-dependent temperature evolution (Bjorken cooling)

---

## Quickstart

### 1) Environment
Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the main scripts (examples)

Validation Baseline

```bash
python 01_Validation-Baseline/code/u1_pure_gauge_mc.py
python 01_Validation-Baseline/code/schwinger-hamiltonian-check.py
python 01_Validation-Baseline/code/OQS_9D_Hilbert_space.py
```

Static Benchmarks

```bash
python "02_Static Benchmarks/code/vqe_optimizer(N=4).py"
python "02_Static Benchmarks/code/vqe_ptimizer(N=8)_trotter_derisk.py"
```

Non-Equilibrium Gauge Dynamics

```bash
python "03_Non-Equilibrium Gauge Dynamics/code/field_quench_gauge.py"
```

Continuum Physics Results

```bash
python "04_Continuum Physics Results/code/schwinger_continuum_massgap.py" --help
python "04_Continuum Physics Results/code/OQS_continuum.py"
```


Outputs (plots) are typically written to the working directory and/or the corresponding figures/ folder.


## Reproducibility tips

- Run scripts from the repo root so relative imports work consistently.

- The OQS scripts rely on shared helpers (e.g., utils_QOS.py). Keep these centralized under src/ (or convert to a small installable package) to avoid path issues.

- For long ED/VQE runs, consider pinning dependency versions and recording your platform in release notes.
