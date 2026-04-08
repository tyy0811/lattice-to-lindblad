# Modular VQE Benchmark (Aer + Quantum Inspire)

Qiskit-based VQE benchmarking framework for lattice gauge theory Hamiltonians, with noisy simulation on Qiskit Aer, optional execution on Quantum Inspire hardware, and built-in error mitigation (MEM + ZNE).

## Project layout

```
vqe_runner.py          CLI entry point
models/
  schwinger.py         Gauge-eliminated Schwinger model (Tagliacozzo mapping)
  tfim.py              Transverse-field Ising model
  xxz.py               Heisenberg XXZ chain
  npy.py               Load any Hamiltonian from a .npy file
backends/
  aer_backend.py       Qiskit Aer with configurable depolarizing + readout noise
  qi_backend.py        Quantum Inspire provider (Qiskit-QuantumInspire)
mitigation/
  mem.py               Measurement Error Mitigation (full assignment matrix)
  zne.py               Zero-Noise Extrapolation (polynomial fit to scale=0)
core/
  ansatz.py            RY-CX-RZ hardware-efficient ansatz with optional Neel init
  hamiltonian.py       Pauli decomposition of arbitrary Hamiltonians
  ideal_vqe.py         Statevector VQE optimizer (COBYLA, multi-restart)
  measurement.py       Basis grouping, shot-based energy evaluation, bootstrap SE
analysis/
  error_sources.py     Gate-vs-readout error ablation
plotting/
  summary.py           Two-panel summary figure (energy bars + log-scale error)
```

## Supported models

| Model | Flag | Key parameters | Description |
|---|---|---|---|
| Schwinger | `--model schwinger` | `--N`, `--x`, `--m_over_g` | Gauge-eliminated 1+1D QED on a lattice |
| TFIM | `--model tfim` | `--N`, `--J`, `--h`, `--pbc` | Transverse-field Ising model |
| XXZ | `--model xxz` | `--N`, `--Jxy`, `--Jz`, `--pbc` | Heisenberg XXZ spin chain |
| Custom | `--model npy` | `--ham_npy PATH` | Any $2^N \times 2^N$ Hamiltonian from a `.npy` file |

## Error mitigation

**Measurement Error Mitigation (MEM)** (`--do_mem`): Constructs the full $2^N \times 2^N$ assignment matrix $A$ by preparing and measuring all $2^N$ computational basis states. Noisy probability vectors are corrected via $\vec{p}_{\rm mit} = A^{-1} \vec{p}_{\rm raw}$ with non-negativity clipping and renormalization.

**Zero-Noise Extrapolation (ZNE)** (`--do_zne`, Aer only): Scales the depolarizing gate-error rates by factors $\lambda \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ (configurable via `--zne_scales`), holds readout errors fixed (separately corrected by MEM at each scale), fits a degree-2 polynomial to $E(\lambda)$, and extrapolates to $\lambda = 0$.

**Error-source ablation** (`--error_analysis`): Runs three separate Aer evaluations (gate-only noise, readout-only noise, both) to decompose the total error budget.

## Install

```bash
pip install numpy scipy matplotlib qiskit qiskit-aer
pip install qiskit-quantuminspire   # only if you want --backend qi/both
qi login                            # once
```

## Run examples

### Aer noisy simulation + MEM + ZNE + error analysis
```bash
python vqe_runner.py --backend aer --model schwinger --N 4 --x 4 --m_over_g 0 \
  --layers 4 --do_mem --do_zne --error_analysis --shots 4000 --shots_cal 8192 \
  --save_json results_aer.json --save_plot
```

### Quantum Inspire backend (Tuna-5) + MEM
```bash
python vqe_runner.py --backend qi --qi_backend "Tuna-5" --model schwinger --N 4 --x 4 \
  --layers 4 --do_mem --shots 2000 --shots_cal 2048 --save_json results_qi.json --save_plot
```

### Full story figure: ED vs Ideal vs Aer vs QI
```bash
python vqe_runner.py --backend both --qi_backend "Tuna-5" --model schwinger --N 4 --x 4 \
  --layers 4 --do_mem --do_zne --shots 2000 --shots_cal 2048 --save_json results_both.json --save_plot
```

## Results: Schwinger model N=4

Benchmark on the gauge-eliminated Schwinger Hamiltonian ($N=4$, $x=4$, $m/g=0$) with a 4-layer RY-CX-RZ ansatz. Aer noise model: $p_{1q}=10^{-3}$, $p_{2q}=10^{-2}$, $p_{0\to1}=p_{1\to0}=2\times10^{-2}$. QI backend: Tuna-5 (2000 shots).

| Method | Energy | |dE| | Error % |
|---|---|---|---|
| ED (exact) | -7.9550 | -- | -- |
| Ideal VQE | -7.9308 | 2.42e-02 | 0.30% |
| Aer noisy (raw) | -6.0085 | 1.946 | 24.5% |
| **Aer ZNE + MEM** | **-7.8834** | **7.16e-02** | **0.90%** |
| QI Tuna-5 (raw) | -4.1415 | 3.813 | 47.9% |
| QI Tuna-5 + MEM | -4.5461 | 3.409 | 42.9% |

**Key finding:** ZNE + MEM reduces the Aer noisy error from 24.5% to 0.9%, recovering 99.1% of the exact energy. On QI Tuna-5 hardware, MEM alone is insufficient -- gate errors dominate, and gate-error mitigation (ZNE or equivalent) is needed for quantitative accuracy.

### Energy comparison: ED vs Ideal vs Aer vs QI

![ED vs Ideal vs Aer vs QI](../summary_vqe_gap.png)

Upper panel: energy estimates across all methods with bootstrap error bars and $\Delta E$ annotations. The dashed line marks the ED exact value. Lower panel: absolute error $|E - E_{\rm ED}|$ on a log scale -- Aer+mit (ZNE+MEM) recovers to within $7 \times 10^{-2}$ of exact, a 27x improvement over raw Aer.

### Zero-Noise Extrapolation curve

![ZNE extrapolation](../noisy_vqe_zne.png)

Energy vs noise scale factor $\lambda$. The five red points are Aer evaluations at increasing depolarizing rates. The degree-2 polynomial fit extrapolates to $\lambda=0$ (blue square), landing close to the ED exact (dashed black) and ideal VQE (dotted green) reference lines.

### Quantum Inspire hardware: QI raw vs QI + MEM

![QI hardware](../qi_vqe_mem.png)

Bar chart from the standalone `noisy_vqe_zne.py` script on the Tuna-5 backend. MEM provides only a modest improvement, confirming that gate errors -- not readout errors -- are the dominant noise source on this hardware.

## Outputs

- **Console:** summary table with ED, Ideal, noisy, and mitigated energies
- **`--save_json PATH`:** full results payload (energies, noise parameters, bootstrap standard errors)
- **`--save_plot`:** two-panel PNG figure (energy bars + log-scale error)

Saved results: [`results_both.json`](../results_both.json), [`results_qi.json`](../results_qi.json)
