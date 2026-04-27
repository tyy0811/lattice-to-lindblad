# Stage 06 Summary — Dispersive Transmon Readout

## What this stage demonstrates

Stage 06 models dispersive readout of a transmon coupled to a readout resonator, validates the simulator against analytic limits, decomposes readout infidelity into named coherent/incoherent channels, fits synthetic characterization traces, and uses the fitted parameters in a sensitivity/Pareto optimization layer.

## Why it matters

This stage connects four tasks relevant to superconducting-processor modeling:

1. open-system simulation of a transmon–resonator system,
2. quantitative error-budget attribution,
3. characterization-style parameter recovery,
4. optimization of readout controls under speed–fidelity tradeoffs.

## Main artifacts

- Figure 1: validated readout model with IQ trajectories, SNR, and assignment fidelity.
- Figure 2: coherent/incoherent readout error budget.
- Figure 3: synthetic characterization traces with fitted parameters and recovery coverage.
- Figure 4: sensitivity analysis, regime map, Pareto frontier, and closed-loop recommendation marker.
- Figure 5a (Module 5a): DRAG-corrected X-gate trajectories + speed-leakage tradeoff with leakage-vs-fidelity trade-off characterization inset.

## Module 5a — Single-qubit gate modeling (DRAG)

A sin²-windowed-Gaussian π-pulse with calibrated DRAG-1 quadrature correction on the transmon (Duffing approximation) delivers a working X gate with bit-flip error **ε_X^ref(T_gate = 20 ns) = 1.09 × 10⁻³** under full REFERENCE_DEVICE Lindblad (T₁ = 30 μs, T₂_echo = 40 μs, Purcell on, n_th = 0.01) at fidelity-optimal β_opt ≈ 0.5. Calibration uses a gate-error objective `argmin_β (1 − F_transfer)` on the perturbative β grid `[0, 1.2]`. Eight validations pass (V1 trajectory, V2a fidelity threshold, V3 truncation convergence, V4 decoherence-free ceiling, V5a α-scaling, V5b envelope-dependent slope diagnostic, V6 sign convention, V7 endpoint smoothness). The implementation surfaced a leakage-vs-fidelity trade-off (V2b) characterized as published curves: the β values minimizing gate fidelity, final leakage, and peak leakage diverge across the perturbative β grid. Curves over T_gate ∈ [5, 50] ns are exported in `figures/fig5a_drag_leakage_data.yaml` and consumed as data by the eventual Module 5b active-reset spec.

See `figures/fig5a_drag_leakage.png` and `diagnostics/drag_leakage_suppression.md`.

## Scope

The closed-loop demo varies fitted \(T_1,T_2,\omega_q\) over fixed REFERENCE resonator/coupling parameters. Full resonator spectroscopy and AC-Stark calibration are not included; they are the natural next extension needed for per-device default-to-optimized gain estimates.

## How to run

```bash
pytest dispersive_readout/tests/ -v
python 06_Dispersive_Readout/dispersive_readout_simulation.py
python 06_Dispersive_Readout/scripts/fig2_error_budget.py
python 06_Dispersive_Readout/characterize.py --help
python 06_Dispersive_Readout/scripts/fig4_optimization.py
```
