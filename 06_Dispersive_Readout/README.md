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

In the scoped closed-loop harness, fitted devices vary (*T*₁, *T*₂, ω_q) over fixed REFERENCE resonator and coupling. The Pareto argmax is shared across the harness, so the closed-loop demonstration is rendered as a recommended operating-point marker rather than a default-to-optimized arrow. Enabling a per-device argmax would require resonator spectroscopy and AC-Stark calibration in the characterization layer; both are tractable extensions, but they expand Module 3 beyond the scoped Rabi/Ramsey/T₁/T₂ protocol surface and are deferred to keep the submission artifact bounded.

## Module 5a — Single-qubit gate (DRAG-corrected X)

**What's here:** A sin²-windowed-Gaussian π-pulse with calibrated DRAG-1 quadrature correction on the transmon (Duffing approximation), plus an eight-validation suite (V1–V7) and a published `ε_X(T_gate)` curve over `T_gate ∈ [5, 50] ns`. Headline number: **`ε_X^ref(T_gate = 20 ns) = 8.12 × 10⁻⁴`** (where `ε_X = 1 − F_avg` over the Pauli set `{|0⟩, |1⟩, |+⟩, |+i⟩}`, post-N12) under full Lindblad on REFERENCE_DEVICE at fidelity-optimal `β_opt ≈ 0.50`. Module 5b (active reset) consumes the YAML cache as data.

**Outputs in this folder:**
- `scripts/fig5a_drag_leakage.py` — Figure 5a driver (panel a: trajectories at `T_gate = 20 ns` for no-DRAG / β=1 / β_opt; panel b: speed-leakage tradeoff over the sweep + `ε_X(T_gate)` inset + V2b leakage-vs-fidelity trade-off inset).
- `figures/fig5a_drag_leakage.png` — the rendered figure.
- `figures/fig5a_drag_leakage_data.yaml` — published curves (β minimizers, leakage suppressions, `ε_X(T_gate)`).
- `diagnostics/drag_leakage_suppression.md` — diagnostic report with V1–V7 status table and the round-8 / round-9 measurement-before-amendment methodology trail.

**Package entry points (new in `dispersive_readout.control` and `dispersive_readout.analysis`):**
- `sin2_windowed_gaussian`, `sin2_windowed_gaussian_derivative`, `calibrate_pi_pulse_amplitude`, `drag_correction`
- `simulate_x_gate`, `GateResult`
- `calibrate_drag_beta`, `DragCalibrationResult`
- `transfer_fidelity_0_to_1`, `leakage_population`, `leakage_peak`, `epsilon_x_from_transfer`

**Validations passed (`dispersive_readout/tests/`):**
- V1 two-level analytic Rabi trajectory match across the full pulse window (< 10⁻⁴).
- V2a gate error `1 − F_transfer < 10⁻⁴` at headline; empirical 7.3 × 10⁻⁵, plus regime sweep at `T_gate ∈ {10, 15, 20, 30} ns`.
- V2b leakage-vs-fidelity trade-off characterization (the three β minimizers diverge on the perturbative grid; published as panel (b) inset 2 + YAML schema).
- V3 truncation convergence at `T_gate = 20 ns` headline: `n=4` vs `n=5` spread ≈ 3.3 × 10⁻⁸.
- V4 decoherence-free fidelity ceiling diagnostic.
- V5a fitted log-log slope of no-DRAG leakage vs |α| is negative (slope = −10.08); V5b reports the steeper-than-textbook-α⁻² slope as an envelope-dependent finding (slope = −14.5).
- V6 DRAG sign convention check (`β = −1` more leaky than `β = +1`).
- V7 endpoint smoothness: `Ω_x(0) = Ω_x(T) = Ω̇_x(0) = Ω̇_x(T) = 0` to machine precision.

### Calibration objective (post-N11 + N12) and the V2b trade-off

DRAG calibration minimizes average X-gate fidelity error over the Pauli set: `β_opt = argmin_β (1 − F_avg(β))` where `F_avg = mean(F(|ψ_in⟩ → X|ψ_in⟩))` for `ψ_in ∈ {|0⟩, |1⟩, |+⟩, |+i⟩}`, on the perturbative β grid `β ∈ [0, 1.2]`. **Three guards apply jointly** (post-N12) — fidelity objective + Pauli-set averaging + perturbative β grid; custom β grids that exceed `[0, 1.2]` require explicit `allow_nonperturbative=True` opt-in and are flagged with `perturbative_safe=False` in the result. `transfer_fidelity_0_to_1` is retained as an explicit one-way diagnostic. Final and peak leakage are recorded as **diagnostic curves**, not as calibration targets.

The implementation surfaced and characterized a **leakage-vs-fidelity trade-off**: across the panel-(b) `T_gate` sweep, the β values minimizing gate fidelity, final leakage, and peak leakage diverge on the perturbative grid. At `T_gate = 20 ns`: `β_opt_fidelity ≈ 0.50`, `β_min_final_leak ≈ 0.90`, `β_min_peak_leak ≈ 1.20`. The triplet is published as panel (b) inset 2 plus YAML keys `beta_opt_fidelity[]`, `beta_opt_final_leak[]`, `beta_opt_peak_leak[]`. See `diagnostics/drag_leakage_suppression.md` for the methodology trail (round-8 peak-suppression-saturation finding under leakage-objective calibration → round-9 calibration-objective correction; both findings are scientifically valid under their respective objectives).
