# Lattice-to-Lindblad: Real-Time Gauge Dynamics & Open Quantum Systems

A Python implementation and validation suite spanning **lattice gauge theory (Schwinger model, 1+1D QED)** and **open quantum systems (pNRQCD-motivated quarkonium in medium)**, with results documented in the included *Results_and_Validation* PDFs. Shared OQS utilities live in `utils_QOS.py`.

Open-system (pNRQCD/Lindblad) results are motivated by:

> Brambilla et al., *Phys. Rev. D* **96**, 034021 (2017). arXiv:1612.07248  
> Brambilla et al., *Phys. Rev. D* **109**, 114016 (2024). arXiv:2403.15545  
> Brambilla et al., arXiv:2508.11743 (2025)

## Overview

This repository contains four stages of code + validation artifacts, organized as a **progressive validation chain**:

| Folder | Domain | Core Method(s) | Output |
|--------|--------|----------------|--------|
| `01_Validation-Baseline/` | Gauge + OQS | U(1) MC checks, Schwinger Hamiltonian checks, Lindblad baseline singlet–octet (1 ⊕ 1) evolution | Baseline validation PDF + scripts |
| `02_Static Benchmarks/` | Gauge + OQS| ED cross-checks, sector-preserving VQE benchmarks (N=4, N=8) | Lindblad baseline singlet–octet (1 ⊕ 8) evolution Benchmark PDF + VQE scripts |
| `03_Non-Equilibrium Gauge Dynamics/` | Gauge | Real-time evolution under electric-field quench; string breaking diagnostics | Dynamics PDF + quench script |
| `04_Continuum Physics Results/` | Gauge + OQS | Continuum-facing mass-gap analysis; continuum OQS studies (e.g., sequential suppression) | Continuum PDF + analysis scripts |

**Project documents:**
- `docs/Theoretical_Framework.pdf` — modeling assumptions, derivations, conventions
- `docs/research_highlight.pdf` — high-level summary of goals, methods, outcomes

---

## 01 — Validation Baseline

Establishes correctness of the building blocks used throughout the repo.

**Gauge (U(1) & Schwinger model)**
- Pure-gauge U(1) Monte Carlo with Wilson-loop diagnostics (area law cross-checks).
- Gauge-eliminated Schwinger Hamiltonian sanity checks (construction/consistency).

**Open quantum systems (Lindblad)**
- Compact Hilbert-space Lindblad evolution in the singlet–octet (1 ⊕ 1) structure.
- Baseline comparisons against controlled/analytic limits (as documented in the validation PDF).

**Code (examples):**
```bash
python 01_Validation-Baseline/code/u1_pure_gauge_mc.py
python 01_Validation-Baseline/code/schwinger-hamiltonian-check.py
python 01_Validation-Baseline/code/OQS_2D_Hilbert_space.py
python 01_Validation-Baseline/code/OQS_9D_Hilbert_space.py
````

**Results:** `01_Validation-Baseline/results/Validation_Baseline_Results_and_Validation.pdf`

---

## 02 — Static Benchmarks

Benchmarks static observables for the Schwinger Hamiltonian using **ED cross-checks** and **symmetry/sector-preserving VQE** workflows.

* Sector-projected VQE (N=4) for controlled benchmarking.
* Extension toward larger sizes (e.g., N=8) with staged optimization and Trotter de-risking.
* Compact Hilbert-space Lindblad evolution in the singlet–octet (1 ⊕ 8) structure.

**Code (examples):**

```bash
python "02_Static Benchmarks/code/vqe_optimizer(N=4).py"
python "02_Static Benchmarks/code/vqe_ptimizer(N=8)_trotter_derisk.py"
```

**Results:** `02_Static Benchmarks/results/Static Benchmarks_Results_and_Validation.pdf`

---

## 03 — Non-Equilibrium Gauge Dynamics

Real-time dynamics for the Schwinger model under an **electric-field quench**, with diagnostics targeting **string breaking** behavior across regimes (e.g., heavy vs light mass).

**Code (example):**

```bash
python "03_Non-Equilibrium Gauge Dynamics/code/field_quench_gauge.py"
```

**Results:** `03_Non-Equilibrium Gauge Dynamics/results/Non_Equilibrium_Gauge_Dynamics_Results_and_Validation.pdf`

---

## 04 — Continuum Physics Results

“Packaging” stage: continuum-facing and physics-grade analyses that build on validated baselines.

* **Schwinger-model mass-gap** analysis (continuum-oriented study/extrapolation as documented).
* **Continuum OQS/Lindblad** analyses (e.g., sequential suppression; optional time-dependent medium evolution such as Bjorken-like cooling).

**Code (examples):**

```bash
python "04_Continuum Physics Results/code/schwinger_continuum_massgap.py" --help
python "04_Continuum Physics Results/code/OQS_continuum.py"
```

**Results:** `04_Continuum Physics Results/results/Continuum_Physics_Results_and_Validation.pdf`

---

## Repository Structure

```
  utils_QOS.py                                  # Shared Lindblad/OQS helpers used by OQS scripts
  Theoretical_Framework.pdf
  research_highlight.pdf

01_Validation-Baseline/

    u1_pure_gauge_mc.py
    schwinger-hamiltonian-check.py
    OQS_2D_Hilbert_space.py
    Validation_Baseline_Results_and_Validation.pdf

02_Static Benchmarks/

    vqe_optimizer(N=4).py
    vqe_ptimizer(N=8)_trotter_derisk.py
    OQS_9D_Hilbert_space.py
    Static Benchmarks_Results_and_Validation.pdf

03_Non-Equilibrium Gauge Dynamics/

    field_quench_gauge.py
    Non_Equilibrium_Gauge_Dynamics_Results_and_Validation.pdf

04_Continuum Physics Results/

    schwinger_continuum_massgap.py
    OQS_continuum.py
    Continuum_Physics_Results_and_Validation.pdf
```

---

## Getting Started

Recommended: **Python 3.10+**

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pandas
```

> Note: some folders include spaces; wrap paths in quotes (as shown above).

---

## Key Concepts

**Progressive validation chain:**
Each stage is designed to reduce ambiguity in later physics claims: (i) baseline correctness → (ii) static benchmarks → (iii) real-time dynamics → (iv) continuum-facing results.

**Symmetry/constraint preservation (gauge & VQE):**
Where applicable, workflows aim to respect physical constraints (e.g., Gauss-law/gauge structure via gauge elimination; symmetry/sector projection in VQE) to avoid exploring unphysical sectors and to improve optimization stability.

**Open-system modeling (Lindblad):**
The quarkonium-in-medium component uses a singlet–octet open-system framework and studies survival/suppression under medium effects, with controlled baseline checks documented in the validation PDFs.

Shared OQS utilities:
`utils_QOS.py` centralizes common Lindblad/OQS routines so baseline, continuum, and validation scripts reuse the same operator/propagation/diagnostic logic.

---

## References

**Open quantum systems / pNRQCD (quarkonium in medium):**

* [1] Brambilla, Escobedo, Soto, Vairo, *Phys. Rev. D* **96**, 034021 (2017); arXiv:1612.07248.
* [2] Brambilla, Magorsch, Strickland, Vairo, Vander Griend, *Phys. Rev. D* **109**, 114016 (2024); arXiv:2403.15545.
* [3] Brambilla, Magorsch, Vairo, arXiv:2508.11743 (2025).

---




