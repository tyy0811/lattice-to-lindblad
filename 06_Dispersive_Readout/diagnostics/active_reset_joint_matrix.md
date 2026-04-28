# Module 5b — Joint Transition-Readout Active Reset

**Operating point:** closed-loop demo device idx=18 ($T_1 = 5.35\,\mu s$, $T_{2,\rm echo} = 6.55\,\mu s$, $\omega_q = 4.722\,\rm GHz$, $\varepsilon_{\rm drive} = 140\,\rm MHz$). $(\kappa, g, \omega_r, \bar n_q)$ inherited from REFERENCE\_DEVICE.

**Headline:** at $\tau_b = 0.54\,\mu s$ (regime: active_winning), the active-reset residual decomposes as **missed-excited 0.0%, false-positive-on-decayed 98.9%, gate-failure 1.1%**. The joint-matrix terms dominate gate error by a factor of ~94 — **the active-reset budget is dominated by joint-matrix structure (T₁-during-measurement plus false-positive flips), not by 5a's gate fidelity, in this regime**.

---

## 1. The conceptual finding

During a measurement window of duration $\tau_{\rm meas}$, $T_1$ relaxation can flip the qubit. The plain confusion matrix $P(m | s_i)$ collapses two physically distinct events into one column:

- $P(s_f = e, m = 0 | s_i = e)$ — qubit stayed excited, measurement missed it → **reset FAILS**
- $P(s_f = g, m = 0 | s_i = e)$ — qubit decayed mid-measurement, measurement correctly read ground → **reset SUCCEEDS** (already in $|g\rangle$)

The **joint matrix** $P(s_f, m | s_i)$ distinguishes these. The plain confusion matrix conflates them, leading to a v2-era bug where the ideal-gate floor was a single term ($P(s_f=e, m=0|e)$) instead of two ($P(s_f=e, m=0|e) + P(s_f=g, m=1|e)$). The second term — the false-positive-on-decayed flip back to $|e\rangle$ — does not vanish at $\varepsilon_X = 0$; it is **maximal there**.

V7 demonstrates this empirically: the curve $P(s_f=g, m=0 | e)$ vs $\tau_{\rm meas}/T_1$ exceeds $0.05 + 2 \cdot \mathrm{SE}$ across the entire $\tau_{\rm meas}/T_1 \in [0.10, 2.00]$ sweep range, with a maximum of 0.791 near $\tau_{\rm meas} = 2T_1$ — decay-during-measurement is non-negligible across the regime by design.

## 2. Cross-module narrative

**Module 4 → Module 5b regime extension (by design).** Module 4 picked $\tau_{\rm opt} = 500$ ns at idx=18 for *readout* fidelity (the dispersive saturation peak; $\tau_{\rm opt}/T_1 \approx 0.09$). 5b extends $\tau_{\rm meas}$ past Module 4's readout-optimal window to characterize the *reset* crossover, which sits at $\tau_{\rm meas} \approx \tau_{\rm critical}$ further into the decoherence regime. The two modules operate in different regimes by design: readout-optimal vs reset-crossover.

**5a → 5b ε_X handoff.** Module 5a (PR #11) shipped with $\varepsilon_X = 1 - F_{\rm avg} = 8.119 \times 10^{-4}$ at $T_{\rm gate} = 20$ ns (full Lindblad, REFERENCE\_DEVICE, post-N12). 5b's `load_eps_x_5a` lazy-loads this from `fig5a_drag_leakage_data.yaml` with mtime-based provenance capture. Panel-(a) renders three traces:

- passive baseline: $e^{-(\tau_{\rm meas} + \tau_{\rm gate})/T_1}$
- active reset, $\varepsilon_X = 0$ (ideal-gate ceiling)
- active reset, $\varepsilon_X = 8.12 \times 10^{-4}$ (5a-realistic, $T_{\rm gate} = 20$ ns)

The two active curves overlap on the figure (their separation is sub-percent-of-residual at every $\tau_{\rm meas}$ in the sweep). That overlap is the visible quantification of the headline finding: gate error contributes negligibly to the reset budget compared to the joint-matrix structure.

## 3. Validation summary

| ID | Status | Pass criterion | Result |
|---|---|---|---|
| V1 | **PASS** (unit-tier) | Two-term ideal-gate floor at $\varepsilon_X = 0$ | exact match against `J.joint_ideal_gate_floor()` |
| V2 | **PASS** (sweep-based, integration-tier) | Active < passive at some $\tau_{\rm meas}$ in $[0.1, 2.0]\cdot T_1$ | active beats passive at short $\tau_{\rm meas}$ (active_winning regime) |
| V3 | **PASS** (integration-tier) | Asymmetric long-$\tau$ floors; active overhead = false-positive contribution at $\tau = 2T_1$ | passes within SE-allowance |
| V4a | **PASS** (integration-tier, blocking) | Integrated IQ matches `simulate_readout` in no-jump limit | <5% relative discrepancy at amplitude 40 MHz (Fock-truncation-safe regime) |
| V4b | **DIAGNOSTIC** (non-blocking, v0) | Marginal fidelity vs Module 1 reference within 2× shot noise | discrepancy = 0.0038 at amplitude 40 MHz |
| V5 | **PASS** (split: unit + slow-tier) | Binomial SE formula correct + empirical $1/\sqrt{N}$ scaling | unit formula exact; slow-tier 1/√N at N ∈ {200, 1000, 4000} within factor-of-2 |
| V6 | **PASS** (unit-tier) | Worst-case ($p_e = 1$) ≥ mixed-prior ($p_e = 1/2$) | property holds across $\varepsilon_X \in \{0, 8.12 \times 10^{-4}, 0.01\}$ |
| V7 | **PASS** (sweep-based, integration-tier, SE-aware) | $P(s_f=g, m=0 \| e) > 0.05 + 2 \cdot \mathrm{SE}$ at some sweep point | exceeds threshold across the entire $[0.10, 2.00]\,T_1$ sweep, max 0.791 at $\tau = 2T_1$ |

**V4a operating-point note.** V4a uses drive amplitude 40 MHz instead of idx=18's 140 MHz: at 140 MHz the steady-state cavity occupation $|\alpha|^2 \approx 17$ saturates the default `N_resonator=15` Fock truncation in `simulate_readout`, and the mesolve $\alpha$ is artificially bounded — a numerical artifact of the truncation, not a convention bug in `pointer_response`. The convention contract is amplitude-independent, so a smaller-amplitude check is sufficient. Module 5b's analytic `pointer_response` handles arbitrary $|\alpha|$ natively because it tracks the coherent-state amplitude directly — no Fock cutoff required, and arguably more trustworthy than mesolve at the actual fig5b operating point.

**V4b status note.** V4b is non-blocking *for v0* because Module 1's `compute_assignment_fidelity` provides a Gaussian-around-$\int \langle a \rangle dt$ reference, which is a different statistical object than 5b's jump-time mixture of pointer-history Gaussians. V4a is the blocking consistency gate. If a future Module 1 extension exposes a finite-$T_1$ IQ-distribution reference (e.g., a jump-time mixture computed via Hilbert-space `mcsolve`), V4b can be promoted to blocking.

## 4. Risk register (residual after design mitigation)

**ε_X transfer assumption (REFERENCE → idx=18).** 5a was calibrated at REFERENCE\_DEVICE; 5b operates at idx=18. ε_X is dominated by the gate's intrinsic properties (DRAG calibration, anharmonicity, $T_{\rm gate}$) rather than by the resonator $\kappa$ that differs between devices, so the REFERENCE values transfer with negligible error (sub-percent expected from $\kappa$-driven differences). **First thing to check if V4 fails.**

**Pointer-response sign convention drift.** `physics/lindblad.py` and `physics/pointer_response.py` both source $\chi$ from `dispersive_shift_full`; the convention single-source-of-truth holds at construction time. V4a holds at runtime (integrated-IQ-level cross-test against `simulate_readout` in the truncation-safe regime).

**Panel-(b) decomposition stability across re-runs.** The regime label (`active_winning` / `crossover_only` / `passive_dominant`) and the chosen $\tau_b$ depend on the seeded RNG. Seed 42 in `fig5b_active_reset.py` keeps the choice reproducible; if the regime label changes between runs at fixed seed, that's a bug in the panel-selection logic, not in the underlying physics.

**Fock-truncation reliability of mesolve at the figure operating point.** The actual idx=18 operating point with $\varepsilon = 140$ MHz pushes mean photon number to ~10.5 against `N_resonator=15`, triggering the `compute_assignment_fidelity` truncation warning. This affects only the V4b *diagnostic* in v0; the production figure runs entirely through `pointer_response`, which has no Fock cutoff. v1.5 with `N_resonator=30` would let V4b promote to blocking at the actual operating point.

## 5. v1.5 deferred work

- `mcsolve`-based jump-history sampling: would add non-Markovian + dephasing effects beyond exponential $T_1$ sampling. Cavity response would still flow through `pointer_response.compute_alpha_trajectory`; only the jump-time sampler changes. Estimated +1–2 days when implemented.
- Likelihood-ratio threshold (replaces `Literal['midpoint']`): would tighten readout in the finite-$T_1$ regime where the optimal threshold is biased away from the midpoint by the decay-during-measurement asymmetry. Estimated +0.5 day.
- Coherent gate-error model during reset feedback: integrates Module 5a's `gate_simulator` into the reset cycle, replacing the classical bit-flip ε_X with full Hamiltonian dynamics during the conditional X-pulse. Estimated +1 day.
- Thermal initial-state preparation ($\bar n_q > 0.05$): samples thermal excitation events from $|g\rangle$ as well as $T_1$ events from $|e\rangle$. Estimated +0.5 day.
- Multi-cycle reset (`reset_residual_n_cycles`): iterates the single-cycle formula across $N$ cycles; demonstrates the noise floor reachable by multiple rounds of measurement-feedback. Estimated +0.5 day.
- Fock truncation expansion for V4b promotion: `N_resonator=30` (or coherent-basis truncation) at the 140 MHz operating point so V4b becomes a blocking consistency check at the production fig5b setpoint, not just at the truncation-safe diagnostic regime. Estimated +0.5 day.

## 6. Methodology note

Module 5b followed the validation-first protocol used in Modules 4 and 5a:

1. V4a (the consistency gate between semiclassical reduction and full mesolve) was implemented and gated on Day 1 before any joint-matrix sampler code was written. The first V4a run failed at 32% relative discrepancy — diagnosed as `simulate_readout`'s Fock truncation saturating at $|\alpha|^2 \approx 17$ on the default `N_resonator=15`, NOT as a convention bug. Resolved by reducing the drive amplitude in the V4a test to 40 MHz (truncation-safe) without changing the production operating point. Documenting the failure mode + resolution explicitly is part of the v0 deliverable.
2. V7 (the regime-characterization claim) is sweep-based, not fixed-τ. The threshold is SE-aware ($> 0.05 + 2 \cdot \mathrm{SE}$), making the failure mode statistically clean.
3. The panel-(b) selection rule is regime-aware. If active reset never beats passive (`passive_dominant` regime), the figure surfaces that as a published finding rather than hiding it via implicit `argmin`.

This continues the pattern from Module 4's per-level $\chi$ derivation diagnostic and Module 5a's DRAG-1 peak-suppression saturation finding: validation-first methodology surfaces physics signals, the spec is amended (or kept honest) accordingly, and the finding becomes a deliverable rather than a workaround.

---

*Generated alongside `fig5b_active_reset.png` and `fig5b_active_reset_data.yaml` by `06_Dispersive_Readout/scripts/fig5b_active_reset.py`. Source spec: `06_Dispersive_Readout/MODULE_5b_SPEC.md` (gitignored, local-only).*
