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
