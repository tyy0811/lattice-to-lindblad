# Lattice-to-Lindblad: Dispersive Readout for Transmon Qubits, Lattice Gauge Theory, and Open Quantum Systems

A Python implementation, validation, and optimization suite for **dispersive readout of superconducting transmon qubits** (Stage 06), developed alongside open-quantum-system and tensor-network infrastructure from earlier stages on a **lattice gauge theory** testbed (Schwinger model, 1+1D QED). Earlier stages cover baseline OQS validation, sector-projected VQE on noisy hardware, real-time gauge dynamics, continuum-extrapolated mass-gap analysis, and entanglement-structure diagnostics.

## Repository overview

| Stage | Domain | Status |
|---|---|---|
| 01 — Validation Baseline | Gauge + OQS solver baselines | Closed-form analytic agreement at machine precision |
| 02 — Static Benchmarks | ED + VQE + noisy hardware (Aer + Quantum Inspire) | ZNE+MEM reduces Aer noisy energy error from 24.5% to 0.9% |
| 03 — Non-Equilibrium Dynamics | Real-time gauge dynamics, string breaking | Confined vs string-breaking regimes cleanly distinguished |
| 04 — Continuum Physics | DMRG-extended mass gap, 1+8 quarkonium suppression | DMRG-extended mass-gap extrapolation with bootstrap uncertainty |
| 05 — Entanglement Structure | Tensor-network bipartite entropy + symmetry-resolved sectors | Top-2 charge sectors carry > 99.3% of entanglement weight |
| **06 — Dispersive Readout** | Superconducting-qubit modeling | Validated 4-module pipeline with 4 shipped figures |

**Project documents:**
- `docs/Theoretical_Framework.pdf` — modeling assumptions, derivations, conventions
- `docs/research_highlight.pdf` — high-level summary of goals, methods, outcomes

---

## Featured hardware-facing extension — Stage 06 Dispersive Readout

Stage 06 is the superconducting-qubit readout extension of this repository. It models dispersive readout of a transmon coupled to a readout resonator, validates the simulator against analytic limits, decomposes assignment infidelity into named coherent and incoherent channels, fits synthetic characterization traces, and uses the recovered parameters in a readout-optimization layer.

The stage is organized into four modules:

1. validated transmon–resonator Lindblad simulation,
2. readout error-budget decomposition,
3. synthetic Rabi/Ramsey/T₁/T₂ characterization and parameter recovery,
4. sensitivity analysis, regime-map diagnostics, and speed–fidelity Pareto optimization.

See `06_Dispersive_Readout/SUMMARY.md` for a one-page overview, `06_Dispersive_Readout/README.md` for design notes, and `dispersive_readout/` for the importable package.

---

## Earlier stages — Lattice gauge theory and entanglement structure

Stages 01–05 develop the open-quantum-system, tensor-network, and noisy-hardware infrastructure on a Schwinger-model (1+1D U(1) gauge theory) testbed. They also document broader scientific scope: continuum-facing extrapolation, real-time dynamics, entanglement-structure diagnostics, and quarkonium-in-medium suppression.

### 01 — Validation Baseline

Pure-gauge U(1) Monte Carlo (Wilson-loop area law cross-checks), gauge-eliminated Schwinger-Hamiltonian sanity checks, and 1⊕1 Lindblad evolution validated against closed-form analytic survival curves at three temperatures. Establishes the building-block correctness reused throughout the repo.

![Stage 01 — 2-level OQS baseline: QuTiP vs analytic](figure/2level_dynamics_with_analytic.png)

Singlet survival P_s(t) for the minimal 1⊕1 Lindblad model at T = 200, 300, 450 MeV. QuTiP numerical evolution (solid) overlaps the closed-form analytic solution (dashed) to machine precision, validating the solver, unit conversion, and detailed-balance construction.

- **Code:** `01_Validation-Baseline/code/` — `u1_pure_gauge_mc.py`, `schwinger-hamiltonian-check.py`, `OQS_2D_Hilbert_space.py`, `OQS_9D_Hilbert_space.py`
- **Report:** `01_Validation-Baseline/results/Validation_Baseline_Results_and_Validation.pdf`

### 02 — Static Benchmarks

ED + sector-projected VQE on the Schwinger Hamiltonian (N=4, N=8 with Trotter de-risking), and the `vqe_modular/` package for noisy-simulator (Aer) and real-hardware (Quantum Inspire, Tuna-5) execution with zero-noise extrapolation and measurement-error mitigation. ZNE + MEM reduces Aer noisy error from 24.5% to 0.9% on N=4 Schwinger; on Tuna-5 hardware, gate errors dominate and require richer mitigation than MEM alone.

![Stage 02 — energy benchmark: ED vs Ideal vs Aer vs QI](summary_vqe_gap.png)

Energy estimates for the N=4 Schwinger model with bootstrap error bars (top) and absolute error on a log scale (bottom). Aer + ZNE + MEM recovers to within 7×10⁻² of exact — a 27× improvement over raw Aer. On Tuna-5 hardware, MEM alone is insufficient; gate errors dominate.

- **Code:** `02_Static Benchmarks/code/`, `vqe_modular/vqe_runner.py`
- **Report:** `02_Static Benchmarks/results/Static_Benchmarks_Results_and_Validation.pdf`

### 03 — Non-Equilibrium Gauge Dynamics

Real-time evolution of the Schwinger model under an electric-field quench. Six-panel diagnostic separates confined (heavy mass m/g = 2.5) and string-breaking (light mass m/g = 0.1) regimes via charge density, electric field, excitation count, and Loschmidt echo.

![Stage 03 — string breaking: heavy vs light mass quench dynamics](<03_Non-Equilibrium Gauge Dynamics/gauge_string_breaking.png>)

Heavy (m/g = 2.5, confined) vs light (m/g = 0.1, string-breaking) dynamics under an E₀ = 0 quench. Top: charge density heatmaps. Middle: electric-field heatmaps showing lattice-scale oscillations (confined) vs propagating wavefront (string breaking). Bottom: field diagnostics, excitation count, and Loschmidt echo.

- **Code:** `03_Non-Equilibrium Gauge Dynamics/code/field_quench_gauge.py`
- **Report:** `03_Non-Equilibrium Gauge Dynamics/results/Non_Equilibrium_Gauge_Dynamics_Results_and_Validation.pdf`

### 04 — Continuum Physics

Mass-gap continuum extrapolation: ED for N ≤ 20 and TeNPy DMRG for N up to 80, with a joint 2D fit in (1/N, (ag)²) and bootstrap error bands. Headline result M_gap/g = 0.50 ± 0.09 is consistent with the exact Schwinger value 1/√π ≈ 0.5642 at 0.7σ. Also includes 1S/2S quarkonium sequential-suppression dynamics in 1⊕8 Lindblad form.

![Stage 04 — DMRG mass gap: ED validated, DMRG extended](figure/dmrg_massgap_plot.png)

Left: DMRG-only large-N finite-size convergence at N = 30, 40, 60, 80 across eight lattice spacings. Right: continuum extrapolation in (ag)² = 1/x using the large-N DMRG sequence. Curves visibly approach the exact 1/√π ≈ 0.5642 (dashed); small-N ED/DMRG agreement is documented separately in the validation report.

- **Code:** `04_Continuum Physics Results/`
- **Report:** `04_Continuum Physics Results/Continuum_Physics_Results_and_Validation.md`

### 05 — Entanglement Structure / QI Packaging

Tensor-network packaging of Schwinger states: bipartite entropy profiles, entanglement spectra (vs TFIM reference), Schmidt-value decay, symmetry-resolved (charge-sector) entanglement, and weak-dephasing open dynamics. Quantifies how entanglement is *organized* and *compressed*, not just how much there is. Top-2 charge sectors carry > 99.3% of the bipartite entanglement weight at the benchmark point; weak charge dephasing increases peak S_vN by ~1.6× and the rank for 95% reduced-state weight by 5×.

![Stage 05 — mass sweep entropy profiles](05_Entanglement_Structure_QI/application_breadth/mass_sweep/mass_sweep_entropy_comparison.png)

Bipartite von Neumann entropy profiles across all MPS cuts for four masses (m/g = 0.05, 0.08, 0.125, 0.20) at N = 20, χ = 64, x = 4.0. The oscillatory edge-structured profile shifts monotonically downward with increasing mass — controlled parameter sensitivity at fixed truncation.

- **Code:** `05_Entanglement_Structure_QI/code/`
- **Report:** `05_Entanglement_Structure_QI/Entanglement_Structure_Results_and_Val.md`

---

## 06 — Dispersive Readout for Superconducting Qubits

Stage 06 is the main superconducting-hardware-facing artifact in this repository. It models dispersive readout of a transmon coupled to a readout resonator, validates the simulator against analytic limits, decomposes assignment infidelity into named error channels, fits synthetic characterization traces, and uses the recovered parameters in a readout-optimization layer.

The implementation lives in `dispersive_readout/`; runnable scripts and generated figures live in `06_Dispersive_Readout/`.

### Module 1 — Validated readout model

Open-system simulation of a transmon–resonator system in the second-order Schrieffer–Wolff dispersive frame, with Lindblad channels for transmon relaxation/dephasing, resonator decay, and Purcell decay. The validation suite checks anharmonicity, charge dispersion, dispersive shift, T₁/T₂ recovery, Purcell decay, and Hilbert-space truncation convergence.

![Stage 06 — Figure 1: validated readout model](06_Dispersive_Readout/figures/dispersive_readout_simulation.png)

IQ trajectories, SNR vs integration time, and assignment fidelity vs κ/|χ| at the reference device.

### Module 2 — Error-budget decomposition

Coherent and incoherent contributions to readout infidelity, decomposed into nine named Lindblad channels (Purcell promoted from a coupling effect into its own collapse operator with its own turn-off semantic).

![Stage 06 — Figure 2: error budget](06_Dispersive_Readout/figures/fig2_error_budget.png)

Channel-by-channel infidelity attribution at the reference device, identifying the dominant decoherence pathways at the operating point.

### Module 3 — Characterization and parameter recovery

Synthetic Rabi / Ramsey / T₁ / T₂* characterization protocols producing fitted device parameters consumed by the optimization layer. The recovery-coverage report quantifies how well fitted parameters match injected ground-truth values across a synthetic device population.

![Stage 06 — Figure 3: characterization recovery](06_Dispersive_Readout/figures/fig3_characterization.png)

Synthetic traces, fitted parameters, and recovery coverage across the four protocols.

### Module 4 — Sensitivity, regime map, and Pareto optimization

Three-panel composite: (a) local sensitivity of assignment fidelity to readout-relevant parameters, (b) regime-map diagnostics over \(\chi/\kappa\) and \(\gamma_1\tau_{\rm readout}\) with Purcell, χ-phase-accumulation, and resonator-response boundaries, and (c) speed–fidelity Pareto frontiers with a closed-loop recommendation marker.

![Stage 06 — Figure 4: sensitivity, regime map, Pareto](06_Dispersive_Readout/figures/fig4_optimization.png)

The Pareto argmax is shared across the fitted-device harness; per-device argmax requires resonator spectroscopy and AC-Stark calibration in the characterization layer (deferred extension; see Stage 06 README for details).

### How to run

```bash
pytest dispersive_readout/tests/ -v                                # full suite (~40 s)
pytest dispersive_readout/tests/ -v -m "not slow"                  # fast TDD suite (~5 s)
python 06_Dispersive_Readout/dispersive_readout_simulation.py      # Figure 1
python 06_Dispersive_Readout/scripts/fig2_error_budget.py          # Figure 2
python 06_Dispersive_Readout/characterize.py --help                # Module 3 CLI
python 06_Dispersive_Readout/scripts/fig4_optimization.py          # Figure 4
```

### More
- One-page reviewer summary: `06_Dispersive_Readout/SUMMARY.md`
- Full design notes (validations, silent-failure findings, design decisions): `06_Dispersive_Readout/README.md`
- Importable package: `dispersive_readout/`
- Test suite: `dispersive_readout/tests/`

---

## Repository structure

```text
utils_QOS.py                                  # Shared Lindblad/OQS helpers used by OQS scripts
docs/
  Theoretical_Framework.pdf
  research_highlight.pdf
  results_both.json                           # Aer + QI benchmark results (ZNE + MEM)
  results_qi.json                             # QI-only benchmark results

06_Dispersive_Readout/                        # Stage 06: stage scripts, figures, summary
  README.md                                   # Full design notes (validations, design decisions, silent-failure findings)
  SUMMARY.md                                  # One-page reviewer summary
  FIGURE_2_CAPTION.md                         # Stage 06 Figure 2 caption
  dispersive_readout_simulation.py            # Figure 1 driver
  characterize.py                             # Module 3 CLI (Rabi / Ramsey / T₁ / T₂*)
  scripts/                                    # Per-figure render scripts
  figures/                                    # Rendered PNGs and supporting YAML

dispersive_readout/                           # Importable Python package
  physics/                                    # Transmon, resonator, dispersive-frame Hamiltonian
  analysis/                                   # Error-budget decomposition
  characterization/                           # Protocol fitters and recovery
  optimization/                               # Sensitivity, regime map, Pareto, recommendation
  tests/                                      # Pytest suite (~40 s full, ~5 s fast)

01_Validation-Baseline/
  code/                                       # u1_pure_gauge_mc.py, schwinger-hamiltonian-check.py, OQS_*.py
  results/Validation_Baseline_Results_and_Validation.pdf

02_Static Benchmarks/
  code/                                       # vqe_optimizer(N=4).py, vqe_optimizer(N=8)_trotter_derisk.py, noisy_vqe_zne.py
  results/Static_Benchmarks_Results_and_Validation.pdf

vqe_modular/                                  # Modular VQE (Aer + Quantum Inspire)
  vqe_runner.py                               # CLI entry point
  models/                                     # Schwinger, TFIM, XXZ, custom .npy
  backends/                                   # Aer noisy simulator, QI provider
  mitigation/                                 # MEM (assignment matrix), ZNE (polynomial)
  core/                                       # Ansatz, Pauli decomposition, shot-based evaluation
  analysis/, plotting/

03_Non-Equilibrium Gauge Dynamics/
  code/field_quench_gauge.py
  results/Non_Equilibrium_Gauge_Dynamics_Results_and_Validation.pdf

04_Continuum Physics Results/
  schwinger_continuum_massgap.py, schwinger_dmrg.py, schwinger_joint_extrapolation.py, OQS_continuum.py
  Continuum_Physics_Results_and_Validation.md
  results/                                    # Joint extrapolation figures and CSVs

05_Entanglement_Structure_QI/
  code/                                       # entanglement_entropy / spectrum / schmidt_decay / symmetry-resolved / open dynamics
  Entanglement_Structure_Results_and_Val.md
  results/                                    # Generated figures and CSVs

figure/                                       # Cross-stage figures (e.g. dmrg_massgap_plot.png)
```

---

## Getting started

Recommended: **Python 3.10+**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TeNPy is included in `requirements.txt` as `physics-tenpy`.

> Some folder names contain spaces (e.g. `02_Static Benchmarks/`); wrap paths in quotes where needed.

---

## Key concepts

**Progressive validation chain.** Each earlier stage reduces ambiguity in later physics claims: baseline correctness → static benchmarks → real-time dynamics → continuum extrapolation → entanglement-structure diagnostics. Stage 06 builds on the validated OQS solver and Lindblad machinery developed in stages 01 and 04.

**Symmetry / constraint preservation.** Where applicable, workflows respect physical structure: Gauss-law / gauge structure via gauge elimination; symmetry / sector projection in VQE; constrained-sector decomposition in symmetry-resolved entanglement analysis.

**Tensor-network perspective.** Stage 05 makes the tensor-network story explicit: entropy profiles across cuts, entanglement spectra, Schmidt decay, and reduced-spectrum broadening under weak openness — letting one discuss *effective compressibility*, not just raw observable values.

**Tensor-network extension (DMRG).** Stage 04 mass-gap analysis uses ED for N ≤ 20 validation and TeNPy DMRG for N up to 80 with `Sz` conservation and χ = 100; the long-range electric term is implemented in a compact running-sum MPO.

**Open-system modeling (Lindblad).** Stages 01 and 04 use the singlet–octet (1⊕1, 1⊕8) open-system framework; Stage 06 reuses the same Lindblad machinery for a transmon ↔ resonator system in the dispersive frame.

**Superconducting readout modeling.** Stage 06 applies the same validation-first OQS approach to dispersive transmon readout: analytic-limit checks before optimization, explicit coherent/incoherent error attribution, synthetic characterization, and Pareto analysis of readout controls.

**Shared OQS utilities.** `utils_QOS.py` centralizes Lindblad routines so baseline, continuum, and Stage 06 scripts share the same operator, propagation, and diagnostic logic.

---

## References

**Superconducting-qubit dispersive readout (Stage 06):**
- Marxer et al., arXiv:2508.16437 (2025) — primary reference device.
- Bengtsson et al., *Phys. Rev. Lett.* **132**, 100603 (2024) — secondary reference.

**Open quantum systems / pNRQCD (quarkonium in medium):**
- Brambilla, Escobedo, Soto, Vairo, *Phys. Rev. D* **96**, 034021 (2017); arXiv:1612.07248.
- Brambilla, Magorsch, Strickland, Vairo, Vander Griend, *Phys. Rev. D* **109**, 114016 (2024); arXiv:2403.15545.
- Brambilla, Magorsch, Vairo, arXiv:2508.11743 (2025).

**Quantum information / tensor-structure context:**
- Acuaviva, Makam, Nieuwboer, Pérez-García, Sittner, Walter, Witteveen, *The minimal canonical form of a tensor network*, 2022; arXiv:2209.14358.
- van den Berg, Christandl, Lysikov, Nieuwboer, Walter, Zuiddam, *Computing moment polytopes of tensors, with applications in algebraic complexity and quantum information*, STOC 2025. doi:10.1145/3717823.3718221.

**Gauge / tensor-network / entanglement context:**
- Schwinger-model results in this repository use gauge-eliminated Hamiltonian workflows, ED cross-checks, and TeNPy-based tensor-network extensions.
- The entanglement-structure stage (Stage 05) is intended as a quantitative bridge between lattice gauge dynamics, tensor-network compressibility, and weakly open many-body evolution.
- For broader mathematical context on tensor-network canonical structure, see the Acuaviva et al. and van den Berg et al. references above.
