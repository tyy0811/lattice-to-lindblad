# Lattice-to-Lindblad: Real-Time Gauge Dynamics, Entanglement Structure & Open Quantum Systems

A Python implementation and validation suite spanning **lattice gauge theory (Schwinger model, 1+1D QED)**, **entanglement-structure diagnostics for tensor-network / QI packaging**, and **open quantum systems (pNRQCD-motivated quarkonium in medium)**, with results documented in the included *Results_and_Validation* reports. Shared OQS utilities live in `utils_QOS.py`.


## Overview

This repository contains five stages of code + validation artifacts, organized as a **progressive validation chain**:

| Folder                               | Domain                | Core Method(s)                                                                                                                | Output                                  |
| ------------------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `01_Validation-Baseline/`            | Gauge + OQS           | U(1) MC checks, Schwinger Hamiltonian checks, Lindblad baseline singlet–octet (1 ⊕ 1) evolution                               | Baseline validation PDF + scripts       |
| `02_Static Benchmarks/`              | Gauge + OQS           | ED cross-checks, sector-preserving VQE benchmarks (N=4, N=8)                                                                  | Static benchmark PDF + VQE scripts      |
| `03_Non-Equilibrium Gauge Dynamics/` | Gauge                 | Real-time evolution under electric-field quench; string breaking diagnostics                                                  | Dynamics PDF + quench script            |
| `04_Continuum Physics Results/`      | Gauge + OQS           | Continuum-facing mass-gap analysis with **ED validation (N≤20)** + **DMRG extension (TeNPy, N=30,40)**; continuum OQS studies | Continuum PDF + analysis scripts        |
| `05_Entanglement_Structure_QI/`      | Gauge + QI/OQS bridge | Entropy profiles, entanglement spectra, Schmidt decay, symmetry-resolved entanglement, weakly open entanglement dynamics      | Entanglement-structure report + scripts |

**Project documents:**

* `docs/Theoretical_Framework.pdf` — modeling assumptions, derivations, conventions
* `docs/research_highlight.pdf` — high-level summary of goals, methods, outcomes

---

## 01 — Validation Baseline

Establishes correctness of the building blocks used throughout the repo.

**Gauge (U(1) & Schwinger model)**

* Pure-gauge U(1) Monte Carlo with Wilson-loop diagnostics (area law cross-checks).
* Gauge-eliminated Schwinger Hamiltonian sanity checks (construction/consistency).

**Open quantum systems (Lindblad)**

* Compact Hilbert-space Lindblad evolution in the singlet–octet (1 ⊕ 1) structure.
* Baseline comparisons against controlled/analytic limits (as documented in the validation PDF).

**Code (examples):**

```bash
python 01_Validation-Baseline/code/u1_pure_gauge_mc.py
python 01_Validation-Baseline/code/schwinger-hamiltonian-check.py
python 01_Validation-Baseline/code/OQS_2D_Hilbert_space.py
python 01_Validation-Baseline/code/OQS_9D_Hilbert_space.py
```

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

**Results:** `02_Static Benchmarks/results/Static_Benchmarks_Results_and_Validation.pdf`

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

Continuum-facing and physics-grade analyses that build on validated baselines.

* **Schwinger-model mass-gap** analysis, using ED validation for `N ≤ 20` and DMRG extension to `N = 30, 40`.
* **Continuum OQS/Lindblad** analyses, including suppression studies and optional time-dependent medium evolution.

**Code (examples):**

```bash
python "04_Continuum Physics Results/code/schwinger_continuum_massgap.py" --help
python "04_Continuum Physics Results/code/schwinger_dmrg.py" --help
python "04_Continuum Physics Results/code/OQS_continuum.py"
```

**Results:** `04_Continuum Physics Results/results/Continuum_Physics_Results_and_Validation.pdf`

---

## 05 — Entanglement Structure / QI Packaging

This stage packages the Schwinger-model results from a **quantum information / tensor-network** perspective. It extends the static and dynamical gauge analyses with diagnostics that quantify not just *how much* entanglement is present, but *how it is organized*, *how compressible it is*, and *how it changes under weak openness*.

### Included scripts

* `schwinger_entanglement_entropy.py`
  Computes full bipartite von Neumann entropy profiles across all MPS cuts for a Schwinger-model ground state or benchmark configuration.

* `schwinger_entanglement_spectrum.py`
  Extracts Schmidt values and entanglement levels $[\xi_i = -\log(\lambda_i^2)]$ at representative cuts, with optional comparison to reference models such as TFIM.

* `schmidt_decay_analysis.py`
  Quantifies Schmidt-value decay, cumulative retained weight, and effective low-rank compressibility relevant for tensor-network representations.

* `schwinger_symmetry_resolved_entanglement.py`
  Decomposes bipartite entanglement into constrained charge-like sectors on a bond, separating inter-sector Shannon structure from intrasector entropy.

* `open_schwinger_entanglement_dynamics.py`
  Extends the closed Schwinger entanglement analysis to **weakly open dynamics**, benchmarking subsystem entropy growth, reduced-state spectrum broadening, and compressibility changes under dephasing against the closed quench.

### Scientific scope

This stage is organized around five linked questions:

1. **Entropy profile:** where is entanglement concentrated across the lattice?
2. **Spectrum structure:** how is Schmidt weight distributed across entanglement levels?
3. **Compressibility:** how many Schmidt components retain most of the state weight?
4. **Symmetry resolution:** which constrained sectors actually carry the entanglement?
5. **Open-system extension:** how does weak dissipation reshape subsystem entropy and effective tensor-network compressibility?

### Main results packaged in this stage

* **Structured entropy profiles** across Schwinger-model cuts, with edge-enhanced structure rather than a trivial flat profile.
* **Distinct entanglement spectra** relative to a simple TFIM reference at matched representative cuts.
* **Strong Schmidt-space compressibility**, with low rank capturing nearly all retained weight in closed-state benchmarks.
* **Symmetry-resolved entanglement organization**, showing that most weight is carried by a small number of constrained sectors.
* **Weakly open dynamics benchmark**, showing that charge dephasing can substantially increase subsystem entropy and broaden the reduced-state spectrum while only modestly perturbing a simple electric-field observable.

### Code (examples)

```bash
python "05_Entanglement_Structure_QI/code/schwinger_entanglement_entropy.py" --help
python "05_Entanglement_Structure_QI/code/schwinger_entanglement_spectrum.py" --help
python "05_Entanglement_Structure_QI/code/schmidt_decay_analysis.py" --help
python "05_Entanglement_Structure_QI/code/schwinger_symmetry_resolved_entanglement.py" --help

python "05_Entanglement_Structure_QI/code/open_schwinger_entanglement_dynamics.py" \
  --N 24 --mass 0.125 --coupling 4.0 --chi 64 \
  --cut 11 --tmax 6.0 --nt 61 \
  --gamma 0.0 --gamma-ref 0.02 \
  --channel dephasing \
  --outdir "05_Entanglement_Structure_QI/results/open_dynamics_test"
```

### Outputs

Typical outputs from this stage include:

* entropy-profile CSV/PNG bundles
* entanglement-spectrum and Schmidt-decay comparison figures
* symmetry-resolved sector-weight tables and bridge summaries
* open-dynamics CSV + PNG bundles
* report-style validation summaries in Markdown/PDF form

**Results:**
`05_Entanglement_Structure_QI/results/Entanglement_Structure_Results_and_Validation.pdf`

---

## Repository Structure

```text
utils_QOS.py                                  # Shared Lindblad/OQS helpers used by OQS scripts
docs/
  Theoretical_Framework.pdf
  research_highlight.pdf

01_Validation-Baseline/
  code/
    u1_pure_gauge_mc.py
    schwinger-hamiltonian-check.py
    OQS_2D_Hilbert_space.py
    OQS_9D_Hilbert_space.py
  results/
    Validation_Baseline_Results_and_Validation.pdf

02_Static Benchmarks/
  code/
    vqe_optimizer(N=4).py
    vqe_ptimizer(N=8)_trotter_derisk.py
  results/
    Static_Benchmarks_Results_and_Validation.pdf

03_Non-Equilibrium Gauge Dynamics/
  code/
    field_quench_gauge.py
  results/
    Non_Equilibrium_Gauge_Dynamics_Results_and_Validation.pdf

04_Continuum Physics Results/
  code/
    schwinger_continuum_massgap.py
    schwinger_dmrg.py
    OQS_continuum.py
  results/
    Continuum_Physics_Results_and_Validation.pdf

05_Entanglement_Structure_QI/
  code/
    schwinger_entanglement_entropy.py
    schwinger_entanglement_spectrum.py
    schmidt_decay_analysis.py
    schwinger_symmetry_resolved_entanglement.py
    open_schwinger_entanglement_dynamics.py
  results/
    Entanglement_Structure_Results_and_Validation.pdf
```

---

## Getting Started

Recommended: **Python 3.10+**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TeNPy is included in `requirements.txt` as `physics-tenpy`.

> Note: some folders include spaces; wrap paths in quotes where needed.

---

## Key Concepts

### Progressive validation chain

Each stage is designed to reduce ambiguity in later physics claims:

1. baseline correctness
2. static benchmarks
3. real-time gauge dynamics
4. continuum-facing extrapolation / OQS packaging
5. entanglement-structure and compressibility analysis

### Symmetry / constraint preservation

Where applicable, workflows aim to respect physical constraints:

* Gauss-law / gauge structure through gauge elimination
* symmetry / sector projection in VQE
* constrained-sector decomposition in symmetry-resolved entanglement analysis

### Tensor-network perspective

The entanglement-structure stage makes the tensor-network story explicit by tracking:

* entropy profiles across cuts
* entanglement spectra
* Schmidt decay and retained weight
* reduced-spectrum broadening under weak openness

This makes it possible to discuss **effective compressibility**, not just raw observable values.

### Tensor-network extension (DMRG)

For continuum-facing Schwinger mass-gap results, ED is used for `N ≤ 20` validation, and TeNPy DMRG extends to `N = 30, 40` with symmetry conservation (`conserve="Sz"`) and `χ = 100`. The long-range electric term is implemented in a compact running-sum MPO to avoid quadratic MPO growth.

### Open-system modeling (Lindblad)

The quarkonium-in-medium component uses a singlet–octet open-system framework and studies survival/suppression under medium effects, with controlled baseline checks documented in the validation reports.

The Schwinger open-dynamics extension complements this by asking a different question: how weak dissipation changes **many-body entanglement structure**, subsystem entropy growth, and reduced-state spectrum compressibility in a lattice gauge theory quench.

### Shared OQS utilities

`utils_QOS.py` centralizes common Lindblad/OQS routines so baseline, continuum, and validation scripts reuse the same operator, propagation, and diagnostic logic.

---

## References

**Open quantum systems / pNRQCD (quarkonium in medium):**

* [1] Brambilla, Escobedo, Soto, Vairo, *Phys. Rev. D* **96**, 034021 (2017); arXiv:1612.07248.
* [2] Brambilla, Magorsch, Strickland, Vairo, Vander Griend, *Phys. Rev. D* **109**, 114016 (2024); arXiv:2403.15545.
* [3] Brambilla, Magorsch, Vairo, arXiv:2508.11743 (2025).
  
 **Quantum information / tensor-structure context:**
 
* [4] Acuaviva, Makam, Nieuwboer, Pérez-García, Sittner, Walter, Witteveen, *The minimal canonical form of a tensor network*, 2022; arXiv:2209.14358.
* [5] van den Berg, Christandl, Lysikov, Nieuwboer, Walter, Zuiddam, *Computing moment polytopes of tensors, with applications in algebraic complexity and quantum information*, STOC 2025. doi:10.1145/3717823.3718221.

**Gauge / tensor-network / entanglement context:**

* Schwinger-model results in this repository use gauge-eliminated Hamiltonian workflows, ED cross-checks, and TeNPy-based tensor-network extensions where applicable.
* The entanglement-structure stage is intended as a quantitative bridge between lattice gauge dynamics, tensor-network compressibility, and weakly open many-body evolution.
* For broader mathematical context on tensor-network canonical structure and representability questions, see Refs. [4] and [5].
