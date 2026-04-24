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
- V1a anharmonicity matches next-order formula α ≈ −E_C · (1 + √(E_C/E_J)) within 1% (plus 15% leading-order sanity band against −E_C).
- V1b charge dispersion of |0⟩–|1⟩ transition < 1 kHz at default `N_charge = 31`.
- V2 χ analytic vs numerical at REFERENCE_DEVICE (2%) and weak coupling (1e-4).
- V3 T₁ recovery from undriven decay within 1%.
- V4a T₂/γ_φ recovery from pure-dephasing decay within 1%.
- V4b Purcell rate γ_P = (g|n₀₁|/Δ)² κ vs full-JC dressed-state overlap within 5%.
- Truncation convergence: N_charge 31 → 51 < 1e-6 rel; N_resonator 15 → 20 < 1e-3 abs on end-of-pulse ⟨a⟩.

## How to run

```bash
# From repo root:
pytest dispersive_readout/tests/ -v            # full suite (~40 s)
pytest dispersive_readout/tests/ -v -m "not slow"  # fast TDD suite (~5 s)
python 06_Dispersive_Readout/dispersive_readout_simulation.py   # regenerates Figure 1
```

## Key implementation decisions

### Dispersive-frame effective Hamiltonian (not bare Jaynes-Cummings)

`build_hamiltonian` returns the 2nd-order Schrieffer-Wolff dispersive-regime Hamiltonian in the fully-rotating frame. Each transmon level rotates at its bare ω_j; the resonator rotates at ω_d. The transverse coupling is transformed out and replaced by the Lamb shift Δω_j and dispersive pull χ_j a†a:

```
H_eff = Σ_j Δω_j |j⟩⟨j| + Σ_j χ_j |j⟩⟨j| a†a + (ω_r − ω_d) a†a + ε(t)(a + a†)
```

Per-level χ_j are computed by `dispersive_shift_full` (includes both JC and Bloch-Siegert contributions). This architecture makes Module 2's Purcell-vs-T₁ error-budget decomposition clean: Purcell is a separately-controllable collapse channel with its own turn-off semantic.

### Explicit Purcell collapse channel

Because the transverse coupling is no longer explicit, Purcell decay is added as a dedicated Lindblad channel with rate γ_P_{j→j-1} = (g|⟨j-1|n̂|j⟩|/Δ_{j,j-1})² κ (1+n_th). V4b validates this formula against the dressed-state resonator-component overlap of the full Jaynes-Cummings Hamiltonian (0.17% agreement at REFERENCE_DEVICE — consistent with the expected O((g/Δ)⁴) 4th-order residual).

### Silent-failure mode caught in the old rotating-frame formulation

The plan originally specified a rotating frame at ω_d = ω_r with the full transverse coupling retained. In that frame the transmon diagonal has GHz-scale entries (level j at ω_j − j·ω_d), so the Lindblad solver needs ~2.5 ps timesteps to integrate correctly. At the default QuTiP `nsteps=10000` limit the integrator was silently returning non-converged trajectories that disagreed with analytic steady states by factors of 4–60 in photon number and 10× in |⟨a⟩|, without raising. The dispersive-frame refactor in commit `9139241` both sped the simulation by ~30× and corrected this latent silent-failure mode. This is a diagnostic finding worth preserving — the kind of issue a tight validation suite is specifically designed to surface.

## Modules 2–4 (shipped)

- **Module 2 — error-budget decomposition.** Figure 2 with nine named Lindblad channels (Purcell promoted to its own operator, plan originally specified eight). Script: `scripts/fig2_error_budget.py`.
- **Module 3 — characterization recovery.** 4-protocol CLI wrapping `dispersive_readout.characterization.cli`. Script: `characterize.py`.
- **Module 4 — sensitivity + regime map + Pareto + closed-loop recommendation.** Figure 4 composite (3 panels): `scripts/fig4_optimization.py`.

Scripts are runnable with `python 06_Dispersive_Readout/<name>.py`; tests for each module are colocated under `dispersive_readout/tests/`.

### Module 4 / Figure 4 — closed-loop rendering note

In the scoped closed-loop harness, fitted devices vary (*T*₁, *T*₂, ω_q) over fixed REFERENCE resonator and coupling. The Pareto argmax is shared across the harness, so the closed-loop demonstration is rendered as a recommended operating-point marker rather than a default-to-optimized arrow. Full resonator spectroscopy and AC-Stark calibration — which would break the shared-argmax regime and enable a default→optimized ΔF gain — are out of Module 3's characterization scope and scheduled as post-submission extensions. See `MODULE_4_SPEC.md` §0.5 Amendment #18 for the full rationale.
