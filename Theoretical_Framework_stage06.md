# Theoretical Framework — `lattice-to-lindblad` Stage 06

**Dispersive Readout of Transmon Qubits: From Cooper-Pair Box to Pareto Frontier**

Companion document to `IMPLEMENTATION_PLAN.md` and `MODULE_{1..4}_SPEC.md`.
Repository path: `lattice-to-lindblad/stage_06_readout/`.
Status: derive-first. No code exists at the time this document is written; every
script reference below is a specification of what *will* implement the derivation.


## 0. How to read this document

This framework derives the physics of dispersive transmon readout from first principles
and maps every equation to the specific Python module, function, and validation test
that implements it. The ordering mirrors the flow of data through the five-module stack:

```
Module 1:  transmon H → resonator coupling → Lindblad → readout observable      §1-§7
Module 2:  error-budget decomposition of the fidelity obtained from Module 1    §8
Module 3:  synthetic characterization traces → lmfit → parameter recovery       §9
Module 4:  sensitivity + Pareto optimization over the Module 1 simulator        §10
Module 5a: DRAG-corrected single-qubit X gates on the Module 1 transmon         §11
Module 5b: semiclassical active reset using Module 1's pointer + Module 5a's X  §12
```

Modules 5a and 5b extend the Module 1 simulator from *measurement* of a
dispersively-coupled transmon to *control* and *measurement-feedback*. They
reuse the §3-§7 Hamiltonian, RWA, dispersive transform, Lindblad, and IQ
discriminator unchanged; the new physics is (§11) the weakly-anharmonic
Duffing-oscillator driven dynamics on a 4-level ladder, and (§12) the
heterodyne stochastic master equation reduced to a controlled
direct-jump + Gaussian IQ model.

Each derivation step carries a confidence tag:

- **[Exact]** — algebraically exact, no approximations.
- **[Exact within model]** — exact after the upstream modeling assumptions
  (e.g. single-mode resonator, capacitive coupling only, RWA on the drive)
  have already been applied; not "exact at the level of the full physical
  Hamiltonian with all distributed modes and amplifier-chain dynamics."
- **[Approximation]** — an approximation; the regime of validity is stated and cited.
- **[Assumption]** — an assumption not proven here; source or conditions given.
- **[Unverified]** — believed correct, but not independently cross-checked in this
  document. Items tagged **[Unverified]** must be checked by hand or against a
  reference during implementation.

Convention warnings (sign/normalization/ordering differences between this document
and common references) are boxed and prominent because they are the most common
source of silent bugs in quantum simulation code.


## 1. Motivation and Physical Setup

### 1.1 The physical system

We model a single **transmon qubit capacitively coupled to a linear readout
resonator**, driven through the resonator port, embedded in a dilution-refrigerator
environment with standard Lindblad decoherence channels. Schematically:

```
    drive ε(t)           resonator a             transmon (E_C, E_J)
      ─────▶        ┌──────────────────┐     ┌────────────────────────┐
                    │  ω_r, κ          │  g  │  φ̂, n̂; levels |0⟩,|1⟩,|2⟩… │
                    │  (bath, n̄_th=0)   │ ◀──▶│  (bath: γ₁, γ_φ, n̄_th)    │
                    └──────────────────┘     └────────────────────────┘
```

The device consists of a single Josephson junction (energies $E_C$, $E_J$) shunted
by a large capacitance such that $E_J / E_C \gg 1$ (**transmon regime**,
Koch et al. 2007), coupled capacitively at rate $g$ to a coplanar-waveguide or
3D-cavity resonator of frequency $\omega_r$ and linewidth $\kappa$. The qubit
transition frequency is $\omega_{01}$, and the qubit-resonator detuning
$\Delta \equiv \omega_{01} - \omega_r$ is taken **negative** (qubit below
resonator) to match the parameter regime of the **contextual benchmark
device** (not a regression target — see §1.4 framing) used throughout
this framework, Marxer et al. (arXiv:2508.16437, 2025).

### 1.2 Why this system matters

Dispersive readout is the workhorse measurement primitive of every superconducting
quantum processor in operation today. Its assignment fidelity — the probability
that a single-shot measurement correctly identifies the qubit state — is a
first-order determinant of any circuit's effective depth, since measurement
errors compound across syndrome extraction rounds in error-corrected circuits.
Improving assignment fidelity meaningfully reduces the physical-qubit count
required for a given logical error rate in surface-code architectures; the
exact factor is resource-estimation-dependent and is not quoted here.

The physics of the process bundles together four core themes of
superconducting-qubit modeling and dynamics:

1. **Open-system modeling** of superconducting processors — the Lindblad master
   equation on $\text{transmon} \otimes \text{resonator}$.
2. **Coherent and incoherent error sources** — dispersive-approximation breakdown,
   drive miscalibration, low-level $|1\rangle \to |2\rangle$ leakage, $T_1$,
   $T_\varphi$, thermal photons, Purcell decay.
3. **Characterization data analysis** — Rabi, Ramsey, $T_1$, Hahn-echo with
   $1/f$-drifted noise.
4. **Device optimization** — sensitivity, regime mapping, Pareto frontier,
   closed-loop recommendation.

### 1.3 Why this computational method

The design decision that sits above every other choice is: **simulate the full
Jaynes-Cummings + Lindblad dynamics using QuTiP, validated against analytic
dispersive-regime formulas, rather than working directly in the dispersive
approximation.** The alternatives and the reason each is rejected:

- *Pure dispersive simulation (diagonal χ_j only, no off-diagonal coupling).*
  Faster, but loses the very thing Module 2 needs to quantify:
  "dispersive-approximation breakdown." If the dispersive simulation were the
  baseline, there would be no well-defined channel against which to compare.
- *Stochastic-master-equation or quantum-trajectory simulation.* Correct for
  modeling continuous-measurement backaction and the stochastic single-shot
  $T_1$-jump structure that the centroid+Gaussian fidelity model misses
  (§6.2, §7.1, §7.3a). Becomes necessary when claiming absolute 99.9 %+
  fidelity at $\tau/T_1 \gtrsim 1\%$, but is overkill for first-order
  parameter sweeps and for the budget-decomposition workflow of Stage 06.
  The trajectory approach is therefore relegated to a **scoped validation
  cross-check** (Module 1 V7 (jump-tail cross-check)) rather than the baseline.
- *Tensor-network / MPS simulation.* Unnecessary: the Hilbert space here is
  $N_\mathrm{transmon} \times N_\mathrm{resonator} = 5 \times 15 = 75$
  dimensional, perfectly tractable for exact Lindblad integration.
- *JAX-native Lindblad solver.* Appealing because it opens gradient-based
  pulse optimization. Deferred to Module 4's contingent autodiff add-on; the
  baseline uses QuTiP for clarity and validation, and the module boundary is
  structured so a JAX replacement is a localized refactor rather than a rewrite.

QuTiP's `qutip.mesolve` is the right tool because the problem is: (a) small
enough for dense vectorization, (b) well-served by adaptive time-stepping at
the $\sim$ ns scale of the readout pulse, (c) extensively validated by the
community against analytic limits for exactly this class of problem. The
default `'adams'` multistep integrator (non-stiff; §5.5) handles our
Lindbladian cleanly at standard tolerances.

**Script connection.** The full simulator derived in §2–§6 lives in
`stage_06_readout/physics/`. The entry point is
`physics/readout_model.py:simulate_readout()`, which consumes a `DeviceConfig`
and `DriveParams` (specified in §7 of this document) and returns the
`ReadoutResult` dataclass whose observable is the homodyne amplitude
$\langle a \rangle(t)$. Every equation in §2–§6 corresponds to one or more
lines in these modules.


### 1.4 Reference-device synthetic seed (used by Module 1–4)

This table documents the **seeded input parameters** for `REFERENCE_DEVICE`.
The seed is a **synthetic Stage 06 default in a Marxer-style transmon-readout
regime, not a faithful extraction of the Marxer et al. device table.**

> **[Seed regime — weak-pull stress-test, not a high-fidelity design point].**
> The synthetic seed gives $|\chi|/\kappa \approx 0.077$ at the multilevel
> $\chi/2\pi \approx -0.385$ MHz and $\kappa/2\pi = 5$ MHz, far below the
> Marxer-style design target $\chi/\kappa \approx 0.5$. This is a
> **deliberate stress-test seed** for Stage 06's framework, not a tuned
> operating point. The $F_\text{assign}$ and $F_\text{ideal}$ "targets"
> in the table below are aspirational — Module 2 reports the actual
> achieved values rather than asserting them.

**`REFERENCE_DEVICE` synthetic seed (used by Module 1–4):**

- **$\omega_{01}/2\pi$** — $4.6\,\text{GHz}$
  - *Status / source:* synthetic Marxer-style (close to but not equal to Marxer's 4.799/4.910 GHz)
- **$\alpha/2\pi$** — $-210\,\text{MHz}$
  - *Status / source:* synthetic ($\approx -E_C$ via Koch 2007 scaling); Marxer reports $-216$ MHz
- **$E_J/E_C$** — $\approx 65.6$
  - *Status / source:* derived from synthetic $(\omega_{01}, E_C)$: $E_J/E_C = ((\omega_{01}/E_C) + 1)^2 / 8 = (4600/210 + 1)^2/8 \approx 65.6$
- **$\epsilon_0, \epsilon_1$ charge dispersion** — $\|\epsilon_0\|, \|\epsilon_1\| < 1\,\text{kHz}$ (Koch asymptotic gives $\|\epsilon_1\| \approx 760\,\text{Hz}$ at $E_J/E_C \approx 65.6$; numerical verification deferred to Module 1 implementation, where `test_transmon_charge_dispersion_negligible` confirms via direct diagonalization. Higher levels: $\|\epsilon_2\|$ exceeds the 1 kHz bound, so the "deep-transmon" charge-dispersion claim applies to $m \le 1$ only.)
  - *Status / source:* Koch Eq. (2.5) at $E_J/E_C \approx 65.6$.
- **$g_{01}/2\pi$** — $120\,\text{MHz}$
  - *Status / source:* **synthetic seed**, no calibration trail to Marxer; the actual Marxer paper reports $g_{qc}/2\pi = 66\,\text{MHz}$ for qubit-coupler coupling, not a 120 MHz readout coupling
- **$\Delta/2\pi$** — $-2700\,\text{MHz}$
  - *Status / source:* inferred from synthetic $(\omega_{01}, \omega_r)$
- **$\omega_r/2\pi$** — $7.3\,\text{GHz}$
  - *Status / source:* synthetic; Marxer reports 6.190/6.350 GHz
- **$\kappa/2\pi$** — $5\,\text{MHz}$
  - *Status / source:* synthetic; Marxer reports 6.1/3.4 MHz
- **$T_1$** — $30\,\mu\text{s}$
  - *Status / source:* synthetic conservative default; Marxer reports 86/102 μs
- **$T_{2,\text{echo}}$** — $40\,\mu\text{s}$
  - *Status / source:* synthetic conservative default; Marxer reports 140/104 μs
- **$\chi/2\pi$ (predicted, multilevel)** — $\approx -0.385\,\text{MHz}$
  - *Status / source:* computed analytically from (4.9) at the seeded $(g_{01}, \Delta, \alpha)$: $\chi = g_{01}^2\alpha / [\Delta(\Delta+\alpha)] = 14400\cdot(-210)/[(-2700)\cdot(-2910)] \approx -0.385$ MHz
- **$\gamma_\text{Purcell}/2\pi$ (predicted at seed)** — $\approx 9.88\,\text{kHz}$ ($\gamma_\text{Purcell} \approx 6.2 \times 10^4\,\text{s}^{-1}$, $T_\text{Purcell} \approx 16.1\,\mu$s)
  - *Status / source:* $(g_{01}/\Delta)^2 \kappa / 2\pi = (120/2700)^2 \cdot 5\,\text{MHz} \approx 9.88$ kHz
- **$\|\chi\|/\kappa$ at seed** — $\approx 0.077$
  - *Status / source:* **weak-pull regime**, far from Marxer-style design target $\chi/\kappa \approx 0.5$
- **$\tau/T_1$ at $\tau = 500\,\text{ns}$** — $\approx 1.67\%$
  - *Status / source:* enters the regime where the centroid+Gaussian model (§7.1) starts to misrepresent assignment-distribution tails (§6.2 warning). One-jump mixture (§7.3a) or trajectory cross-check (V7) recommended for absolute fidelity claims.
- **$F_\text{assign}$ at reference** — target $\gtrsim 0.99$, **to be verified numerically**
  - *Status / source:* reaching $\gtrsim 0.99$ at $\|\chi\|/\kappa \approx 0.08$ requires high $\bar n$ and/or low $\sigma_\parallel$; Module 2 reports the actual achieved value
- **$F_\text{ideal}$ at reference** — target $> 0.999$, **to be verified numerically**
  - *Status / source:* dispersive ceiling at the synthetic seed; achievability depends on the seed's weak-pull regime

**Marxer 2508.16437 actual device parameters (for context, not used as seed):**

> **Source anchor.** Values transcribed from Marxer et al.
> arXiv:2508.16437 v1 (August 22, 2025), Supplementary Appendix A,
> **Table S1: Summary of device parameters** (the arXiv HTML version
> renders the device-parameter table with this exact label). Values
> were transcribed manually on 2025-09 from Table S1 for the two qubits
> (Q1, Q2) used in the high-fidelity demonstration. If the published
> version renumbers the supplementary table, the label here should be
> updated on next revision.

| Quantity | Marxer (Q1) | Marxer (Q2) | Source |
|---|---|---|---|
| $\omega_{01}/2\pi$ | $4.799\,\text{GHz}$ | $4.910\,\text{GHz}$ | Table S1 |
| $\alpha/2\pi$ | $-216\,\text{MHz}$ | $-216\,\text{MHz}$ | Table S1 |
| $\omega_r/2\pi$ | $6.190\,\text{GHz}$ | $6.350\,\text{GHz}$ | Table S1 |
| $\kappa/2\pi$ (readout bandwidth) | $6.1\,\text{MHz}$ | $3.4\,\text{MHz}$ | Table S1 |
| $\chi/2\pi$ (readout dispersive shift) | $2.5\,\text{MHz}$ | $2.6\,\text{MHz}$ | Appendix A; the design target is $\chi/\kappa \approx 0.5$ |
| $T_1$ (idling) | $86\,\mu\text{s}$ | $102\,\mu\text{s}$ | Table S1 |
| $T_{2,\text{echo}}$ | $140\,\mu\text{s}$ | $104\,\mu\text{s}$ | Table S1 |
| $g_{qc}/2\pi$ (qubit-coupler) | $69.6\,\text{MHz}$ | $62.2\,\text{MHz}$ | Table S1 |

> **[Why the seeded $\chi$ disagrees with Marxer's reported $\chi$].**
> The multilevel formula (4.9) at the seeded synthetic $(g_{01}, \Delta, \alpha)
> = (120, -2700, -210)$ MHz predicts $\chi/2\pi \approx -0.385$ MHz. Marxer
> reports $\chi/2\pi \approx 2.5/2.6$ MHz on a different parameter regime
> with $\Delta/2\pi \approx -1.4$ GHz (smaller detuning, no near-cancellation
> suppression). The two systems do not need to predict the same $\chi$ —
> the synthetic seed is *not* trying to reproduce Marxer; it is a
> consistent set of internal default values for Stage 06's reference device.
>
> **Implications for tests and deliverables:**
>
> - V2a (perturbative self-consistency) and V2c (reference-sign check)
>   are real physics gates, unaffected by this issue.
> - V2b is a self-consistency gate on the seeded inputs, not a regression
>   against Marxer's $\chi$. It does not require any specific external value.
> - §1.4 is **not** a regression target; §10, Module 2, and Module 4
>   report results at the synthetic seed.
> - Marxer is used only as **contextual motivation and benchmark framing**
>   (the $\chi/\kappa \approx 0.5$ design target, the Purcell-filter
>   architecture, the shelved-readout strategy). Numerical reproduction
>   of Marxer's actual device would be a separate parameter-extraction
>   task, out of scope for the initial Stage 06 implementation.



## 2. Transmon Hamiltonian: From Cooper-Pair Box to Spectrum

This section derives the transmon Hamiltonian in the charge basis, diagonalizes
it to obtain the eigenenergies $\omega_j$ and the charge matrix elements
$\langle j | \hat n | k \rangle$ that feed everything downstream, and states the
conventions (unit system, ground-state shift, sign of anharmonicity) that the
code must preserve.

### 2.1 The Cooper-pair box Hamiltonian

The starting point is the Cooper-pair box (CPB) circuit: a Josephson junction
shunted by a capacitance, with node charge $\hat Q$ and node flux $\hat\Phi$
as the canonical variables. Writing $\hat n = \hat Q / 2e$ (number of Cooper
pairs) and $\hat\varphi = 2\pi \hat\Phi / \Phi_0$ (gauge-invariant phase drop
across the junction), we have the canonical commutator

$$[\hat\varphi, \hat n] = i \qquad \text{[Exact]} \tag{2.1}$$

so $\hat n = -i \,\partial/\partial \hat\varphi$ in the phase representation.
The CPB Hamiltonian is (Koch et al. 2007, Eq. 2.1):

$$H_\mathrm{CPB} = 4 E_C (\hat n - n_g)^2 - E_J \cos \hat\varphi \qquad \text{[Exact]} \tag{2.2}$$

where $E_C = e^2 / (2 C_\Sigma)$ is the single-electron charging energy
(with $C_\Sigma$ the total capacitance to ground), $E_J = I_c \Phi_0 / (2\pi)$
is the Josephson energy, and $n_g$ is an external dimensionless offset charge
induced on the island by the electrostatic environment.

**Assumption.** $H_\mathrm{CPB}$ models a single isolated junction with no
flux bias; for tunable-coupler architectures (Marxer 2508.16437) the coupler
is assumed held at a fixed flux bias point and therefore enters only through
effective $g$ and $\Delta$.

### 2.2 Charge basis representation

The eigenstates $|n\rangle$ of $\hat n$ with integer eigenvalue
$n \in \mathbb{Z}$ form the natural basis because $\hat n$ is already diagonal
and $\cos\hat\varphi$ has a simple tridiagonal action. Using Euler's formula

$$\cos\hat\varphi = \tfrac{1}{2}(e^{i\hat\varphi} + e^{-i\hat\varphi}) \qquad \text{[Exact]} \tag{2.3}$$

and the charge-shift property
$e^{\pm i\hat\varphi} |n\rangle = |n \pm 1\rangle$ [Exact; this is the canonical
adjoint action of $e^{i\hat\varphi}$ on number eigenstates, following from
$[\hat\varphi, \hat n] = i$], we obtain the matrix representation:

$$\langle n' | H_\mathrm{CPB} | n \rangle = 4 E_C (n - n_g)^2 \,\delta_{n',n} - \tfrac{E_J}{2} (\delta_{n', n+1} + \delta_{n', n-1}) \qquad \text{[Exact]} \tag{2.4}$$

In code, this is a tridiagonal Hermitian matrix of dimension $N_\mathrm{charge}$
(default $13$, i.e. $n \in \{-6, \ldots, +6\}$). The truncation to finite
$N_\mathrm{charge}$ is the first approximation of the calculation:

> **[Approximation].** Truncating the charge basis at $|n| \le N_\mathrm{charge}/2$
> is justified when the low-lying eigenstates have negligible support on large
> $|n|$. For $E_J / E_C \gg 1$ the ground-state RMS charge spread is
> approximately $\sigma_n = (E_J/8E_C)^{1/4}/\sqrt{2}$. At $E_J/E_C \approx 65.6$
> (our reference device, §1.4), this gives $\sigma_n \approx 1.20$. So
> $N_\mathrm{charge} = 13$ covers approximately $\pm 5\sigma_n$, more than
> enough. The truncation error is verified empirically by
> `test_transmon_charge_truncation_converged()` (Module 1 spec §4), which
> requires that eigenenergies change by less than $10^{-4}$ (relative) when
> $N_\mathrm{charge}$ is increased by $2$.

### 2.3 Deep-transmon limit and eigenstructure

In the transmon regime $E_J / E_C \gg 1$, the cosine potential is approximately
harmonic near its minimum. Expanding $\cos\hat\varphi \approx 1 -
\hat\varphi^2/2 + \hat\varphi^4/24 + \ldots$ and quantizing the resulting
anharmonic oscillator (Koch et al. 2007 §II):

$$H_\mathrm{transmon}^{(0)} = \sqrt{8 E_J E_C}\, \hat b^\dagger \hat b -
\tfrac{E_C}{12} (\hat b + \hat b^\dagger)^4 + \ldots \qquad \text{[Approximation]} \tag{2.5}$$

To leading order in $(E_C/E_J)^{1/2}$, the eigenenergies are (Koch et al. 2007,
Eq. 2.11; Blais et al. RMP 2021, §III.A):

$$\omega_j \approx \sqrt{8 E_J E_C}\left(j + \tfrac{1}{2}\right) - \frac{E_C}{12}(6 j^2 + 6 j + 3) - \sqrt{8 E_J E_C}/2 \qquad \text{[Approximation]} \tag{2.6}$$

> **[Note on the additive constant].** The trailing $-\sqrt{8 E_J E_C}/2$
> subtracts the harmonic zero-point but **not** the small quartic ground
> correction, so $\omega_0$ from (2.6) is not exactly zero — it is
> approximately $-3 E_C / 12 = -E_C / 4$. Equation (2.6) is therefore
> correct only up to an irrelevant additive constant; only differences
> $\omega_{j+1} - \omega_j$ enter physical observables. The §2.4
> ground-state shift convention (Convention 2) absorbs this constant
> at diagonalization time, after which $\omega_0 = 0$ exactly.

From which the qubit frequency and **anharmonicity** are:

$$\omega_{01} \equiv \omega_1 - \omega_0 \approx \sqrt{8 E_J E_C} - E_C \qquad \text{[Approximation]} \tag{2.7}$$

$$\alpha \equiv \omega_{12} - \omega_{01} \approx -E_C \qquad \text{[Approximation]} \tag{2.8}$$

**Physical interpretation.** The anharmonicity $\alpha \approx -E_C$ is what
*makes* the transmon a qubit: it separates the $|0\rangle \to |1\rangle$
transition from $|1\rangle \to |2\rangle$ in frequency space by an amount large
enough that a narrow-bandwidth drive can address one without the other.
For our reference device, $E_C / h = 210\,\text{MHz}$, so
$\alpha / 2\pi \approx -210\,\text{MHz}$ (Marxer 2508.16437-consistent;
matches the parameter sheet in Module 1 spec §1.2).

> **[Assumption].** The analytic form (2.7)-(2.8) is used only for
> order-of-magnitude sanity checks. The simulation uses the *numerically*
> diagonalized transmon (equation 2.4 in the charge basis, solved by
> `numpy.linalg.eigh`), which captures higher-order corrections automatically.
> The ≤ 5 % tolerance in test V1 of Module 1 is calibrated to the accuracy
> of the leading-order expansion at $E_J/E_C \approx 65.6$.

### 2.4 Conventions: ground-state shift, units

Two conventions are fixed here and must not be changed downstream:

- **Ground-state energy shift.** After diagonalization we shift all eigenvalues
  by $-\omega_0$ so that $\omega_0 = 0$. This is a convention (a global phase)
  that does not affect any observable; it simplifies rotating-frame transformations
  later. The code documents this in every function's docstring in `physics/transmon.py`.
- **Units.** **Hamiltonian frequencies** ($\omega_j$, $\omega_r$, $g_{jk}$,
  $\chi$, $\varepsilon_0$) are stored in **rad/s**. **Dissipative rates**
  ($\gamma_1$, $\gamma_\varphi$, $\kappa$, Purcell) are stored as inverse
  times in **s$^{-1}$**. The numerical-value system is shared (radians
  being dimensionless), and the paper-reporting convention $\kappa/2\pi$
  is handled at I/O only. See Convention 1 (units) for the full rationale and
  worked examples. Mixing rad/s and Hz internally is the second most
  common bug class in this kind of code (after sign errors on $\chi$);
  making the internal unit system uniform prevents it.

### 2.5 Charge matrix elements

The charge operator in the charge basis is diagonal:
$\hat n |n\rangle = n |n\rangle$. After we transform to the transmon eigenbasis
$|j\rangle = \sum_n c_n^{(j)} |n\rangle$, the charge operator acquires matrix
elements

$$\langle j | \hat n | k \rangle = \sum_n n\, c_n^{(j)*} c_n^{(k)} \qquad \text{[Exact]} \tag{2.9}$$

Two facts about these matrix elements:

1. **Selection rule.** In the harmonic approximation, $\hat n$ is proportional to
   $(\hat b - \hat b^\dagger)/i$, so $\langle j | \hat n | k\rangle$ is nonzero
   only for $|j - k| = 1$ to leading order. Higher-order corrections from the
   quartic term in (2.5) introduce small but nonzero $|j - k| \ge 3$ matrix
   elements; they are included automatically by the numerical diagonalization.
2. **Scale.** $|\langle 0 | \hat n | 1 \rangle| \sim (E_J / 8 E_C)^{1/4} / \sqrt{2}$
   [Approximation, leading order]. This number is the quantity that sets the
   bare transmon–resonator coupling to leading order (see §3.2).

> **[Parity selection rule — why $|j-k|=2$ is absent at $n_g=0$].** At
> $n_g = 0$, the CPB Hamiltonian (2.2) commutes with the parity operator
> $P: |n\rangle \to |-n\rangle$, and the eigenstates $|j\rangle$ have
> definite parity $(-1)^j$. The charge operator $\hat n$ is parity-odd
> ($P\hat n P^{-1} = -\hat n$), so $\langle j|\hat n|k\rangle$ vanishes
> whenever $j$ and $k$ have the same parity — i.e. all even-$|j-k|$ matrix
> elements, including $|j-k|=2$, are zero up to numerical precision. The
> leading non-adjacent corrections from the quartic term in (2.5) are
> therefore $|j-k|=3$, not $|j-k|=2$. (Away from $n_g = 0$ the parity is
> broken and $|j-k|=2$ elements become nonzero but exponentially small in
> $\sqrt{E_J/E_C}$ via the same charge-dispersion mechanism of §2.6.)

> **[Approximation].** The selection rule $|j - k| = 1$ is exact only in the
> harmonic approximation. For the deep-transmon regime, matrix elements with
> $|j - k| \ge 3$ (the leading non-adjacent contributions, by the parity
> argument above) are suppressed by factors of $(E_C / E_J)^{1/2}$ per
> additional hop, and are numerically negligible for the
> $N_\mathrm{transmon} = 5$ levels we keep. They are **monitored** as
> diagnostics for beyond-baseline physics (§4.4 raises a warning if any
> exceeds $10^{-4}$ of the 01 value), but they are **not summed into the
> Stage 06 $\chi$**, because the simulator's JC/RWA Hamiltonian (3.4)
> retains only adjacent one-photon transitions. The distinction between
> two-level and multilevel $\chi$ in Stage 06 comes from the adjacent
> $|1\rangle\leftrightarrow|2\rangle$, $|2\rangle\leftrightarrow|3\rangle$,
> etc. contributions (each with its own detuning $\Delta_j$), not from
> non-adjacent matrix elements. If a future extension keeps beyond-RWA or
> multi-photon physics, §3.4 and §4.4 would both need to be revised.

### 2.6 Charge dispersion and the sweet spot

The offset charge $n_g$ is a source of slow dephasing if the eigenenergies
depend on it. Define the **charge dispersion** of level $j$:

$$\epsilon_j \equiv \max_{n_g} \omega_j(n_g) - \min_{n_g} \omega_j(n_g) \qquad \text{[Exact definition]} \tag{2.10}$$

For a transmon, $\epsilon_j$ decreases exponentially with $\sqrt{E_J / E_C}$
(Koch et al. 2007, Eq. 2.5 — *scaling form only*; the full Koch
expression has a level-dependent prefactor $\propto (E_C/2)\cdot (-1)^j\,2^{4j+5}\,(E_J/E_C)^{j/2+3/4}/(j!\sqrt{2\pi})$
that we do not retain here):

$$\epsilon_j \sim E_C \left(\frac{E_J}{2 E_C}\right)^{(2j+1)/4} e^{-\sqrt{8 E_J / E_C}} \qquad \text{[Approximation — scaling only; numerical validation uses direct CPB diagonalization]} \tag{2.11}$$

For our reference device ($E_J / E_C \approx 65.6$), this gives $\epsilon_{0,1}$
well below $1\,\text{kHz}$ — the "deep transmon" regime where charge noise is
negligible.

> **Physical interpretation.** Equation (2.11) is why the transmon exists:
> making $E_J / E_C$ large exponentially suppresses sensitivity to offset-charge
> noise, at the cost of a linearly-smaller anharmonicity $\alpha \approx -E_C$.
> The transmon regime $E_J / E_C \in [40, 100]$ is the standard
> operating window.
>
> **Note on "sweet spots" and $n_g = 0$.** For the Cooper-pair box (the shallow-$E_J$
> regime), the charge-dispersion curve $\omega_j(n_g)$ has a sweet spot — a
> zero-slope point with vanishing linear sensitivity to charge noise — at
> **$n_g = 1/2$** (half-integer), as discussed in Koch 2007 §II.B citing
> the quantronium work of Vion et al. The transmon's defining advantage is
> that equation (2.11) exponentially *suppresses* the entire charge-dispersion
> curve, so no sweet-spot biasing is needed. Both $n_g = 0$ and $n_g = 1/2$
> are extrema of the (now tiny) periodic dispersion, and at $E_J/E_C \approx 65.6$
> the residual slope at either choice is irrelevant ($\ll 1\,\text{kHz}$).
> We pin $n_g = 0$ in simulation as a **convention** (it is an extremum and
> produces a clean reference spectrum); this is not a sweet-spot appeal.

> **Implementation warning — charge dispersion vs ground-state shifting.**
> §2.4 fixes the convention $\omega_0 = 0$ by shifting all eigenvalues
> by $-\omega_0$ after diagonalization. **This shift must not be applied
> when computing $\epsilon_j$.** If the implementation computes $\epsilon_j$
> from already-shifted spectra at every $n_g$, then $\epsilon_0$ is
> identically zero by construction and the test becomes trivially
> true rather than measuring band dispersion. Two correct implementation
> paths: (i) compute $\epsilon_j$ from the **unshifted** CPB eigenenergies
> $E_j(n_g)$ as functions of $n_g$, then apply the shift only for
> downstream usage; or (ii) compute the dispersion of transition
> frequencies, e.g. $\omega_{01}(n_g) = E_1(n_g) - E_0(n_g)$, which is
> shift-invariant. `test_transmon_charge_dispersion_negligible` must
> use one of these two paths and must explicitly assert that
> $|\epsilon_0| > 0$ at the precision floor (sanity check that the
> shift convention has not been applied prematurely).

### 2.7 Script connection for §2

| Equation / concept | Script | Function |
|---|---|---|
| (2.4) charge-basis matrix | `physics/transmon.py` | `charge_basis_hamiltonian()` |
| Diagonalization + shift | `physics/transmon.py` | `diagonalize_transmon()` |
| (2.9) charge matrix elements | `physics/transmon.py` | `charge_operator_matrix_elements()` |
| (2.7), (2.8), (2.10) summaries | `physics/transmon.py` | `transmon_summary()` |
| (2.8) sanity check | `tests/test_physics_validation.py::test_transmon_anharmonicity_matches_perturbative` | V1, ≤ 5 % |
| (2.11) charge dispersion | `tests/test_physics_validation.py::test_transmon_charge_dispersion_negligible` | V1b, < 1 kHz |
| `N_charge = 13` truncation | `tests/test_physics_validation.py::test_transmon_charge_truncation_converged` | < $10^{-4}$ |


## 3. Resonator, Coupling, and the Jaynes-Cummings Model

With the transmon eigenstructure $\{\omega_j, |j\rangle\}$ in hand, we build
the full driven-coupled Hamiltonian and transform to the rotating frame used
for simulation.

### 3.1 The driven harmonic resonator

The readout resonator is a linear LC mode of frequency $\omega_r$ coupled to
a transmission-line bath that gives it a total linewidth $\kappa$
(internal + external; the external part is what the drive and measurement
pipeline couples to). Setting $\hbar = 1$:

$$H_\mathrm{res} = \omega_r\, a^\dagger a \qquad \text{[Exact within model: single-mode idealization]} \tag{3.1}$$

**[Assumption].** We model a single mode of the readout line. Multi-mode
Purcell filters, reflection from the TWPA, and higher longitudinal
modes are absorbed into the effective $\kappa$ and into the additive
Gaussian noise of the measurement chain. This is the same single-mode
idealization used throughout Blais et al. RMP 2021 and Bengtsson 2024 PRL.

### 3.2 Transmon-resonator coupling

The capacitive coupling between the transmon island and the resonator center
conductor is mediated by a coupling capacitance $C_g$. The Hamiltonian of the
coupled circuit has the form (Blais et al. RMP 2021, §III.B):

$$H_\mathrm{coup} = g\, (a + a^\dagger)\, \hat n \qquad \text{[Exact within model: single-mode linearized circuit]} \tag{3.2}$$

where $g$ is the bare coupling rate, proportional to $C_g$ and to the
zero-point voltage fluctuations of the resonator. In the transmon eigenbasis
$|j\rangle$, using $\hat n = \sum_{j,k} \langle j|\hat n|k\rangle |j\rangle\langle k|$:

$$H_\mathrm{coup} = g \sum_{j,k} \langle j|\hat n|k\rangle (a + a^\dagger)\, |j\rangle\langle k| \qquad \text{[Exact within model]} \tag{3.3}$$

> **[Scope of "Exact within model"].** (3.2)–(3.3) are exact only after the
> upstream circuit reductions: single-mode resonator (no transmission-line
> distributed modes or higher harmonics), linear LC resonator (no
> resonator-internal nonlinearity), capacitive coupling only (no
> inductive-coupling contributions), and $C_g \ll C_\Sigma$ (so that
> renormalization of $E_C$ by $C_g$ is captured in the dressed transmon
> $E_C^\ast$ rather than as a $\hat n^2$ cross-term). Within those
> assumptions, (3.2) is the exact coupling Hamiltonian; the tag means
> "exact within the single-mode linearized circuit model," not "exact at
> the level of the full Hamiltonian of two capacitively-coupled circuits
> with all distributed modes, filter dynamics, and amplifier-chain
> nonlinearities."

**[Approximation] — rotating-wave approximation (RWA) with adjacent-transition
truncation.** Two approximations are applied together at this step:

1. **RWA.** Counter-rotating terms of the form $a\, |k\rangle\langle j|$
   (with $\omega_k < \omega_j$) and $a^\dagger\, |j\rangle\langle k|$
   oscillate at frequency $\omega_j - \omega_k + \omega_r \sim 2 \omega_q$,
   much larger than the couplings they mediate. They are dropped.
2. **Adjacent-transition truncation.** Of the co-rotating terms, only
   **adjacent** transitions $|j+1\rangle\langle j|$ (with $|j - (j+1)| = 1$)
   exchange one resonator photon with a one-level transmon transition
   resonantly. Non-adjacent co-rotating terms $a\,|j+k\rangle\langle j|$
   with $k \ge 2$ oscillate in the rotating frame at residual frequencies
   $(k-1)\omega_d + O(\alpha)$ and are dropped. Their matrix elements are
   small anyway (proportional to higher-order anharmonic corrections,
   $\langle j+k|\hat n|j\rangle = O((E_C/E_J)^{k/2})$) and would
   reintroduce beyond-RWA physics out of scope for Stage 06.

Applying both approximations gives the **adjacent-only JC coupling**:

$$H_\mathrm{coup}^\mathrm{RWA} = \sum_{j=0}^{N_\text{transmon}-2} g_{j,j+1}\, (a\, |j+1\rangle\langle j| + a^\dagger\, |j\rangle\langle j+1|) \qquad \text{[Approximation: RWA + adjacent-only]} \tag{3.4}$$

where $g_{j,j+1} \equiv g\, \langle j | \hat n | j+1 \rangle$ (we take the
phase convention that makes $g_{j,j+1}$ real for our standard charge basis;
[Unverified for complex-valued eigenvectors produced by some LAPACK routines —
in practice `numpy.linalg.eigh` returns real eigenvectors for a real symmetric
input]).

> **Regime of validity (RWA is a deliberate model-scope choice).** The
> RWA drops counter-rotating terms whose resonant denominator is the sum
> frequency $\omega_q + \omega_r$ (rather than the difference
> $\omega_q - \omega_r = \Delta$ that gives the dispersive shift). Two
> distinct quantities have to be distinguished here:
>
> - **Dimensionless admixture amplitude** $g/(\omega_q + \omega_r) \sim 120\,\text{MHz}/11.9\,\text{GHz} \approx 0.010$,
>   so the squared admixture is $\sim 10^{-4}$ — small enough that the
>   eigenstructure is well-approximated by the JC/RWA Hamiltonian.
> - **Static Bloch–Siegert frequency shift** $\delta_\text{BS} \sim g^2/(\omega_q + \omega_r) \approx (120\,\text{MHz})^2/(11.9\,\text{GHz}) \approx 1.2\,\text{MHz}$.
>   This is **not** kHz-scale; it is comparable to (and at the seed
>   actually a few times *larger* than) the multilevel
>   $|\chi|/2\pi \approx 0.385\,\text{MHz}$ at the synthetic seed.
>
> So the RWA is **a deliberate model-scope choice for Stage 06, not
> a "below truncation error" approximation**. V2 (the χ analytic-vs-numerical test) validates
> internal consistency of the JC/RWA model (analytic Schrieffer–Wolff
> $\chi$ vs numerical $\chi$ within the RWA Hamiltonian); it does **not**
> validate agreement with a full quantum-Rabi / cosine-Hamiltonian
> simulator that would include counter-rotating Bloch–Siegert
> corrections. v1.5 / v2 forward pointers extend Stage 06 to the
> beyond-RWA regime if device frequency calibration to MHz precision is
> needed.

### 3.3 Drive term

A classical microwave drive applied to the resonator port couples to the
charge of the resonator:

$$H_\mathrm{drive}(t) = \varepsilon(t) \left(a\, e^{+i\omega_d t} + a^\dagger\, e^{-i\omega_d t}\right) \qquad \text{[Exact within model: classical drive after RWA on the drive]} \tag{3.5}$$

where $\varepsilon(t)$ is the (real) slowly-varying envelope and $\omega_d$ is
the drive carrier frequency.

The envelope we use (locked in Module 1 spec §1.3) is the **erf-square pulse**:

$$\varepsilon(t) = \varepsilon_0\, w(t), \qquad
w(t) = \tfrac{1}{2}\left[\mathrm{erf}\!\left(\tfrac{t - t_\mathrm{rise}}{\sigma_\mathrm{edge}}\right) - \mathrm{erf}\!\left(\tfrac{t - t_\mathrm{fall}}{\sigma_\mathrm{edge}}\right)\right] \qquad \text{[Exact, by construction]} \tag{3.6}$$

with $t_\mathrm{fall} = t_\mathrm{end} - t_\mathrm{rise}$ and the constraint
$t_\mathrm{fall} > t_\mathrm{rise} + 2\sigma_\mathrm{edge}$ so the flat-top is
well-defined. This shape has a smooth (erf-profile) rising and falling edge
of characteristic width $\sigma_\mathrm{edge}$ and a flat top between them.
The reasons for *this* shape rather than a plain square or a pure Gaussian:

- A plain square pulse has discontinuous envelope derivatives, which excites
  the resonator with a broad spectrum and causes oscillatory ringing in
  $\langle a \rangle(t)$.
- A pure Gaussian has no flat top and therefore no "integration plateau" during
  which the IQ amplitudes can be integrated for discrimination.
- The erf-square interpolates: smooth edges $\Rightarrow$ clean spectral content,
  flat top $\Rightarrow$ well-defined integration window. Stage 06 uses
  the erf-square as a **smooth approximation to a rectangular readout
  pulse**; Marxer 2508.16437 reports a rectangular readout pulse (240 ns,
  shelved-readout configuration), not an erf-square envelope. Stage 06's
  erf-square edges are a numerical regularization that suppresses spurious
  cavity ringing in the simulation; they do not change the readout physics
  qualitatively.

### 3.4 Full lab-frame Hamiltonian

Combining §2 and §3:

$$H(t) = \underbrace{\sum_j \omega_j\, |j\rangle\langle j|}_{\text{transmon}} +
\underbrace{\omega_r\, a^\dagger a}_{\text{resonator}} +
\underbrace{\sum_{j=0}^{N_\text{transmon}-2} g_{j,j+1}\, (a\, |j+1\rangle\langle j| + a^\dagger\, |j\rangle\langle j+1|)}_{\text{coupling (adjacent-only)}} +
\underbrace{\varepsilon(t)(a\, e^{+i\omega_d t} + a^\dagger\, e^{-i\omega_d t})}_{\text{drive}} \qquad \text{[Eq. 3.7]} \tag{3.7}$$

This is the lab-frame Hamiltonian. It is **not** what we integrate directly —
the fast carrier oscillation at $\omega_d$ forces unnecessarily small time
steps on a sub-nanosecond scale. We transform to the frame rotating at
$\omega_d$, where the rapid carrier is absorbed into the transformation and
the integrator only needs to resolve the slower envelope dynamics.

### 3.5 Rotating-frame transformation

Apply the unitary

$$U(t) = \exp\!\left[i\omega_d\, t\, \left(a^\dagger a + \sum_j j\, |j\rangle\langle j|\right)\right] \qquad \text{[Exact]} \tag{3.8}$$

The transformed Hamiltonian $H_\mathrm{rot} = U H U^\dagger - i U \partial_t U^\dagger$
is time-independent (the drive term loses its oscillation):

$$H_\mathrm{rot} = \sum_j (\omega_j - j\omega_d)\, |j\rangle\langle j| +
(\omega_r - \omega_d)\, a^\dagger a +
\sum_{j=0}^{N_\text{transmon}-2} g_{j,j+1}\, (a\, |j+1\rangle\langle j| + a^\dagger\, |j\rangle\langle j+1|) +
\varepsilon(t)(a + a^\dagger) \qquad \text{[Exact within model: given (3.4) and RWA on drive]} \tag{3.9}$$

**Derivation of (3.9).** With $U(t) = \exp[+i\omega_d t\,(a^\dagger a + \sum_j j|j\rangle\langle j|)]$,
the Heisenberg-picture transformations are:

- $U\, a\, U^\dagger = a\, e^{-i\omega_d t}$ (from $[a^\dagger a, a] = -a$),
- $U\, a^\dagger\, U^\dagger = a^\dagger\, e^{+i\omega_d t}$,
- $U\, |j\rangle\langle k|\, U^\dagger = e^{+i(j-k)\omega_d t}\, |j\rangle\langle k|$
  (eigenvalues of $\sum_j j|j\rangle\langle j|$ on $|j\rangle$ is $j$, on $\langle k|$ is $-k$).

For the RWA coupling term $a\,|k+1\rangle\langle k|$ (photon destroyed + qubit
raised, both at $\omega_d$ cost to leading order), the combined phase factor is
$e^{-i\omega_d t} \cdot e^{+i((k+1)-k)\omega_d t} = e^{-i\omega_d t} \cdot e^{+i\omega_d t} = 1$.
So the coupling is **time-independent** in the rotating frame [Exact]. For
the drive, $\varepsilon(t)\, a\, e^{+i\omega_d t} \to \varepsilon(t)\, a\, e^{+i\omega_d t} \cdot e^{-i\omega_d t} = \varepsilon(t)\, a$
[Exact], and similarly for $a^\dagger$. For the transmon diagonal,
$|j\rangle\langle j|$ transforms to itself (since $j - j = 0$), leaving the
$-j\omega_d$ shift from the $-iU\partial_t U^\dagger$ term, which gives the
$(\omega_j - j\omega_d)$ diagonal entries in (3.9).

**[Convention warning].** The sign of the detuning terms depends on whether
we rotate at $+\omega_d$ or $-\omega_d$. Our convention is $U = \exp[+i\omega_d\, t\, \ldots]$,
giving $(\omega_j - j\omega_d)$ diagonal entries. Some references (e.g.
Krantz et al. 2019, Appl. Phys. Rev.) rotate the other way and write
$(j\omega_d - \omega_j)$. The physical content is identical; the absolute
sign of all frequencies relative to the frame matters only when comparing
to those references.

### 3.6 Why we keep `build_hamiltonian(frame="rotating")` as the simulation default

The rotating-frame Hamiltonian (3.9) is what `simulate_readout()` integrates.
The alternative, the "dispersive frame" (§4.3 below), is used *only* in
validation tests. The reason is twofold:

1. The rotating-frame Hamiltonian (3.9) retains the full off-diagonal coupling
   $g_{jk}$, so **low-level non-dispersive leakage within the chosen truncated
   Hilbert space** (5-level transmon, 15-photon resonator) — principally
   $|1\rangle \leftrightarrow |2\rangle$ transitions driven by the coupling —
   is captured in the dynamics. **True transmon ionization** (escape into
   highly excited transmon states via multiphoton resonances at specific
   photon populations; Shillito 2022, Dumas 2024) requires both higher
   $N_\text{transmon}$ and a Hamiltonian that preserves the full cosine
   potential (Dumas), and is **out of scope** for Stage 06. The simulator
   detects and attributes *leakage into the kept subspace*; it does not
   claim to reproduce ionization thresholds.
2. The dispersive frame is cheaper but requires trusting the
   adiabatic-elimination hierarchy, which breaks down precisely in the regimes
   (high power, small $\chi/\kappa$) where Module 2 needs to measure the
   deviation. Using the dispersive Hamiltonian as the simulation baseline
   would obscure exactly what Module 2 is trying to quantify.

### 3.7 Script connection for §3

| Equation | Script | Function |
|---|---|---|
| (3.4) RWA + adjacent-only coupling | `physics/lindblad.py` | `build_hamiltonian()` — coupling block, adjacent transitions only |
| (3.6) erf-square envelope | `physics/readout_model.py` | drive-spec callable inside `build_hamiltonian()` |
| (3.9) rotating-frame $H$ | `physics/lindblad.py` | `build_hamiltonian(frame="rotating")` |
| Dispersive option | `physics/lindblad.py` | `build_hamiltonian(frame="dispersive")` — for V2 only |


## 4. Dispersive Transform and the Shift $\chi$

This section derives the dispersive shift $\chi$ in two forms — the two-level
$\chi = g^2 / \Delta$ and the full multi-level Koch formula — and states our
**sign convention** prominently because getting it wrong is the single most
common bug in this kind of simulation.

### 4.1 Regime of validity and scope assumption

> **[Assumption — model hierarchy].** This section derives the dispersive shift
> **within the same Jaynes-Cummings (RWA) Hamiltonian** that §3 defines as the
> simulator baseline. All derivations and the analytic formulas that V2 tests
> against keep only the co-rotating coupling of (3.4). Bloch-Siegert /
> beyond-RWA effects are **outside the V2 comparison by construction**:
> V2 compares the analytic second-order Schrieffer-Wolff dispersive
> formula against the numerical spectrum of the **same JC/RWA
> Hamiltonian**, so beyond-RWA corrections cancel out of both sides. The
> dimensionless counter-rotating admixture is small,
> $g/(\omega_q + \omega_r) \sim 10^{-2}$, but the associated *static
> Bloch-Siegert frequency shift* can be MHz-scale (§3.2,
> $g^2/(\omega_q + \omega_r) \approx 1.2\,\text{MHz}$ at the seed) — so
> beyond-RWA effects must **not** be described as "below the V2 gate
> tolerance" or "below the physical frequency-calibration tolerance".
> They are out of scope by deliberate model-scope choice. If future
> extensions require them, §3 and §4 must both be revised; keeping the
> full dipole coupling in the analytic χ while using RWA in the
> simulator would make V2 systematically undersell agreement for a
> reason that is not a bug.

The dispersive transform then assumes, as the perturbative small parameter,

$$\lambda_{jk} \equiv \frac{|g_{jk}|}{|\Delta_{jk}|} \ll 1, \qquad \Delta_{jk} \equiv \omega_j - \omega_k - \omega_r$$

for all $|j-k|=1$ transitions retained in the JC coupling. For our reference
device,

$$\lambda_{01} \equiv \frac{|g_{01}|}{|\Delta|} \approx \frac{120\,\text{MHz}}{2700\,\text{MHz}} \approx 0.044 \qquad \text{[At seeded $g_{01} = 120$ MHz under \texttt{coupling\_convention="matrix\_element\_01"}; no further matrix-element rescaling]}$$

so the expansion parameter is $\sim 4.4\%$, and second-order dispersive theory
retains terms through $O(\lambda^2)$ and neglects $O(\lambda^3)$ and higher.
This is the standard dispersive regime (Blais et al. RMP 2021, §III.C;
multi-level shift given explicitly in their Eq. 41).

### 4.2 Schrieffer-Wolff transformation

We seek a unitary $U_\mathrm{SW} = \exp(\eta)$ with $\eta^\dagger = -\eta$
(antihermitian) and $\eta = O(\lambda)$ such that $U_\mathrm{SW} H
U_\mathrm{SW}^\dagger$ is block-diagonal in photon number to order $\lambda^2$.
For the JC form of the coupling, the generator takes the standard shape
(see Blais et al. RMP 2021 §III.C and Appendix B.1 for the full multi-level
derivation; for the two-level JC case the analogous Bogoliubov form is their
Eq. 35):

$$\eta = \sum_{|j-k|=1,\, j > k} \frac{g_{jk}}{\omega_j - \omega_k - \omega_r}\, (a\, |j\rangle\langle k| - a^\dagger\, |k\rangle\langle j|) \qquad \text{[Leading-order Schrieffer-Wolff generator]} \tag{4.1}$$

Applying Baker-Campbell-Hausdorff:

$$U_\mathrm{SW} H U_\mathrm{SW}^\dagger = H + [\eta, H] + \tfrac{1}{2}[\eta, [\eta, H]] + \ldots \qquad \text{[Exact formal expansion]} \tag{4.2}$$

By construction $[\eta, H_0] = -H_\mathrm{coup}$ (this fixes $\eta$), so the
linear term cancels $H_\mathrm{coup}$, and the $\tfrac{1}{2}[\eta, [\eta, H_0]] =
\tfrac{1}{2}[\eta, H_\mathrm{coup}]$ is the leading dispersive correction:

$$H_\mathrm{disp} = H_0 + \tfrac{1}{2}[\eta, H_\mathrm{coup}] + \text{higher-order corrections} \qquad \text{[Approximation]} \tag{4.3}$$

> **[Higher-order corrections — what is actually neglected].** Schematically
> the next-order Schrieffer-Wolff term is at $O(\lambda^3)$ in the
> generator expansion, but for the diagonal cavity pull (the quantity
> that becomes $\chi$) the leading neglected correction is **fourth
> order in $g/\Delta$**, i.e. $O(g^4/\Delta^3)$, equivalently
> $O(\lambda^2)$ relative to the second-order pull. This is the
> quantitative residual that V2a's tolerance ($5 \times 10^{-4}$ at
> $\lambda = 0.01$ uncancelled point, §4.5) measures. The earlier
> "$O(\lambda^3)\omega_q$" tag was a generator-level scaling and did
> not match the observed numerical residual, which is why §4.5
> separately discusses the fourth-order amplification by
> $1/|\alpha/(\Delta+\alpha)|$.

Evaluating the commutator requires tracking **oriented** JC transitions: level
$|j,n\rangle$ couples upward to $|j+1,n-1\rangle$ with matrix element
$g_{j,j+1}\sqrt n$ and downward to $|j-1,n+1\rangle$ with matrix element
$g_{j-1,j}\sqrt{n+1}$. The two have **opposite-sign** energy denominators
because the photon-number change is opposite. Introducing

$$\Delta_j \equiv \omega_{j+1} - \omega_j - \omega_r \qquad \text{[upward-transition detuning from resonator]} \tag{4.5a}$$

the photon-number-dependent part of the second-order energy shift of $|j,n\rangle$
is (Blais et al. RMP 2021 Eq. 41; derivation in their Appendix B.1):

$$\chi_j = \frac{|g_{j-1,j}|^2}{\Delta_{j-1}} - \frac{|g_{j,j+1}|^2}{\Delta_j} \qquad \text{[Exact to order $\lambda^2$ within JC/RWA]} \tag{4.5}$$

with the convention that edge terms are zero: $g_{-1,0} = 0$ and
$g_{N_\text{transmon}-1, N_\text{transmon}} = 0$ (since the code truncates at
$N_\text{transmon}$ levels). The full dispersive Hamiltonian is then

$$H_\mathrm{disp} = \sum_j (\omega_j - j\omega_d)\, |j\rangle\langle j| + \left[(\omega_r - \omega_d) + \sum_j \chi_j\, |j\rangle\langle j|\right] a^\dagger a + \varepsilon(t)(a + a^\dagger) + \text{higher-order corrections} \qquad \text{[Approximation; see (4.3) note]} \tag{4.5b}$$

**Physical interpretation.** Equation (4.5b) says that in the dispersive frame,
the resonator sees a qubit-state-dependent frequency shift $\chi_j$: the
cavity frequency becomes $\omega_r + \chi_j$ when the qubit is in $|j\rangle$.
This is the entire mechanism of dispersive readout — a drive at
$\omega_d \approx \omega_r$ reflects with a qubit-state-dependent phase,
and that phase is what homodyne detection measures.

### 4.3 Two-level limit

For a pure two-level qubit, only the $g_{01}$ transition exists (levels
$|j\ge 2\rangle$ are projected out, so $g_{12} = 0$ and
$g_{-1,0} = 0$ by convention). Equation (4.5) collapses to:

$$\chi_1 = \frac{|g_{01}|^2}{\Delta_0} - 0 = \frac{|g_{01}|^2}{\Delta}, \qquad \chi_0 = 0 - \frac{|g_{01}|^2}{\Delta_0} = -\frac{|g_{01}|^2}{\Delta} \qquad \text{[Approximation, JC 2-level]} \tag{4.6}$$

with $\Delta \equiv \omega_{01} - \omega_r = \Delta_0$. The net splitting:

$$\chi_1 - \chi_0 = \frac{2|g_{01}|^2}{\Delta} \qquad \text{[Approximation, JC 2-level]} \tag{4.6a}$$

> **[Notation warning — $g$ vs $g_{01}$].** The Module 1 spec populates
> `REFERENCE_DEVICE` with $g/2\pi = 120\,\text{MHz}$ described in Marxer
> 2508.16437 as "coupling." In our formalism, $g$ is the circuit-level
> prefactor of $(a + a^\dagger)\hat n$, and $g_{jk} = g\,\langle j|\hat n|k\rangle$
> is the effective matrix-element-weighted coupling. These are **not** the
> same in general: for a harmonic transmon
> $|\langle 0|\hat n|1\rangle| = (E_J/8E_C)^{1/4}/\sqrt 2 \approx 1.20$ at
> $E_J/E_C \approx 65.6$, so $g_{01} \approx 1.20\, g$. Marxer's paper does not state
> which convention its "120 MHz" uses, and no independent calibration of
> this parameter is available from the paper.
>
> **Enforceable config decision.** `physics/config.py` carries an
> explicit `coupling_convention: Literal["matrix_element_01",
> "charge_prefactor"]` field on `CouplingParams`. At load time,
> `REFERENCE_DEVICE` declares `coupling_convention = "matrix_element_01"`
> with $g_{01}/2\pi = 120$ MHz as the **synthetic seed value** — see §1.4
> for why this is a synthetic Marxer-style default rather than a Marxer extraction.
>
> **Implementation rule for `matrix_element_01` (mandatory).** If
> `coupling_convention = "matrix_element_01"` and the config value is
> $g_{01}$, the code must compute the higher-transition couplings as
>
> $$g_{j,j+1} = g_{01} \cdot \frac{\langle j|\hat n|j+1\rangle}{\langle 0|\hat n|1\rangle}$$
>
> **not** as $g_{j,j+1} = g_{01} \cdot \langle j|\hat n|j+1\rangle$.
> The latter would silently include a factor of $\langle 0|\hat n|1\rangle$
> (which is $\approx 1.20$ at $E_J/E_C \approx 65.6$, not 1), giving
> an incorrect $g_{01}$ and propagating into $\chi$, Purcell, and every
> downstream rate. The normalization by $\langle 0|\hat n|1\rangle$ is
> what makes the config value $g_{01}$ literally equal the 0→1 coupling.
> This is enforced by a unit test:
>
> ```python
> test_matrix_element_01_convention_normalizes_higher_transition_couplings()
> ```
>
> which constructs a `CouplingParams(g_01=120e6, coupling_convention="matrix_element_01")`,
> requests $g_{0,1}$ from the resulting Hamiltonian builder, and asserts
> that it equals $g_{01}$ exactly (not $g_{01} \cdot \langle 0|\hat n|1\rangle$).
> A second assertion checks $g_{1,2} / g_{0,1} = \langle 1|\hat n|2\rangle / \langle 0|\hat n|1\rangle$,
> which is $\sqrt 2$ in the harmonic limit and is the actual physical
> matrix-element ratio.

> **[Implementation note — eigenvector phase gauge].** `numpy.linalg.eigh`
> returns eigenvectors with **arbitrary global phases** (specifically:
> the largest-magnitude component's sign is implementation-dependent
> and platform-dependent). This means raw $\langle j|\hat n|k\rangle$
> matrix elements can flip sign between runs, breaking the V5 unit test
> and any downstream code that uses signed matrix elements. The fix is
> a deterministic phase-canonicalization step applied immediately after
> diagonalization. Stage 06's choice: **make $\langle 0|\hat n|1\rangle$
> real and positive** by phase-rotating $|1\rangle \to e^{i\theta_1}|1\rangle$
> with $e^{i\theta_1} = \langle 0|\hat n|1\rangle / |\langle 0|\hat n|1\rangle|$
> conjugated appropriately; then propagate through the higher levels by
> requiring $\langle j|\hat n|j+1\rangle$ real and positive for each
> $j \ge 1$. Equivalently, every adjacent matrix element is forced
> real positive. This is documented in `physics/transmon.py` and
> enforced by `test_eigenvector_phase_gauge_is_canonical`. With this
> gauge, the matrix-element ratios in (5.5), (5.8), and the V5
> normalization rule are sign-stable across runs and platforms. The
> $\chi$ formula (4.7) involves $|g_{01}|^2$, so it is gauge-invariant
> regardless; the gauge matters for any code path that reads signed
> matrix elements directly, which includes the V5 ratio assertion above.

The observable dispersive shift (reported to the user) is:

$$\chi \equiv \frac{\chi_1 - \chi_0}{2} = \frac{|g_{01}|^2}{\Delta} \qquad \text{[Convention: standard in Blais RMP 2021]} \tag{4.7}$$

so the resonator frequency shifts by $+\chi$ when the qubit is in $|1\rangle$
and by $-\chi$ when the qubit is in $|0\rangle$. For $\Delta < 0$ (our regime),
$\chi < 0$ in the 2-level limit.

### 4.4 Multi-level formula (adjacent transitions only, consistent with §3.9)

**Scope.** The Hamiltonian (3.9) keeps only $|j-k| = 1$ coupling terms after
the RWA — all non-adjacent terms $a\,|j\rangle\langle k|$ with $|j-k| \ne 1$
oscillate at non-zero residual frequencies and are dropped. The Schrieffer-Wolff
generator η (4.1) is therefore built only from the adjacent-transition
coupling, and the second-order shift $\tfrac{1}{2}[\eta, H_\text{coup}]$
contains only adjacent² contributions. **The dispersive formula must
therefore also be adjacent-only**:

$$\chi_j = \frac{|g_{j-1,j}|^2}{\omega_j - \omega_{j-1} - \omega_r} - \frac{|g_{j,j+1}|^2}{\omega_{j+1} - \omega_j - \omega_r} \qquad \text{[Second order in the JC/RWA Hamiltonian (3.9); adjacent transitions only]} \tag{4.8}$$

with the convention that edge terms are zero: $g_{-1,0} = 0$ and
$g_{N_\text{transmon}-1, N_\text{transmon}} = 0$.

> **[Scope discipline — why adjacent-only].** A naive reading of (4.5)
> would sum over *all* numerically-nonzero matrix elements
> $\langle j|\hat n|k\rangle$ including $|j-k| \ge 3$. That formula is a
> valid second-order perturbative expression *only* for a Hamiltonian
> that retains those non-adjacent coupling terms. The simulator in §3
> retains only $|j-k|=1$ terms (RWA drops the rest), so including
> non-adjacent transitions in the analytic χ while the simulator
> excludes them would make V2 systematically misagree for a reason
> that is not a bug — it would be testing two different models against
> each other. Stage 06 sums (4.5) over adjacent transitions only,
> consistent with the simulator's RWA Hamiltonian.
>
> **What §2.5 does and does not say.** §2.5 notes that non-adjacent
> charge matrix elements $\langle j|\hat n|k\rangle$ with $|j-k| \ge 3$
> become numerically nonzero due to anharmonic corrections in the
> transmon eigenfunctions. (At $n_g = 0$, parity makes $|j-k|=2$ vanish
> identically; see §2.5 parity note.) This affects the off-diagonal
> structure of $\hat n$ in the transmon eigenbasis, but only adjacent
> terms survive the RWA when coupled to a single resonator photon
> exchange. If a future extension keeps beyond-RWA physics
> (Bloch-Siegert, multi-photon resonances à la Shillito/Dumas), §3 and
> §4 would both need to include non-adjacent terms. That is out of scope.
>
> **Validation: non-adjacent matrix elements are *monitored*, not
> *summed*.** `dispersive_shift_full()` reports
> $|\langle j|\hat n|k\rangle|^2$ for $|j-k| \ge 3$ in a diagnostic log,
> and raises a warning if any exceeds $10^{-4}$ of $|\langle 0|\hat n|1\rangle|^2$
> (a threshold below which their potential Bloch-Siegert-scale contribution
> is negligible). They are **not** added to χ.

**Harmonic-limit cross-check.** In the harmonic-matrix-element approximation,
$|g_{j,j+1}|^2 \approx (j+1)|g_{01}|^2$ (so $|g_{12}|^2 \approx 2|g_{01}|^2$,
$|g_{23}|^2 \approx 3|g_{01}|^2$, etc.). Writing $\Delta_0 = \Delta$ and
$\Delta_1 = \Delta + \alpha$:

$$\chi_1 - \chi_0 \approx \frac{2|g_{01}|^2}{\Delta} - \frac{2|g_{01}|^2}{\Delta + \alpha} = \frac{2|g_{01}|^2 \alpha}{\Delta(\Delta + \alpha)} \qquad \text{[Approximation, harmonic-matrix-element limit]} \tag{4.9}$$

This is the compact form quoted in Blais et al. RMP 2021 (cf. Eq. 42).
For $\Delta < 0$ and $\alpha < 0$ (our regime), $\chi_1 - \chi_0 < 0$.
The synthetic seed predicts $\chi/2\pi \approx -0.385$ MHz under the
multilevel formula at $(g_{01}, \Delta, \alpha) = (120, -2700, -210)$ MHz
(verified algebraically below in §4.5); this is a property of the seed,
not a regression against any external device. (Marxer reports $\chi/2\pi
= 2.5/2.6$ MHz on a different parameter regime with smaller detuning;
the two are not directly comparable. See §1.4 for the synthetic-seed
framing.)

**What the code implements.** `dispersive_shift_full()` in
`physics/dispersive.py` computes (4.8) with numerically-computed adjacent
matrix elements $\langle j-1|\hat n|j\rangle$ from §2.5. Equation (4.9)
is provided as a pen-and-paper sanity check; it is what test V2's
harmonic-limit branch compares against.

### 4.5 Convention on the sign of $\chi$

> **Convention warning (critical).** With $\Delta < 0$ (qubit below resonator,
> the Marxer/Bengtsson regime), equation (4.6) gives $\chi \sim g^2/\Delta < 0$
> in the two-level limit. For the multi-level transmon formula (4.8) with
> typical parameters, the sign of the *observable* $\chi = (\chi_1 - \chi_0)/2$
> can be positive or negative depending on the relative magnitudes of the
> $|1\rangle \leftrightarrow |2\rangle$ and $|0\rangle \leftrightarrow |1\rangle$
> contributions. For our reference (synthetic) seed, the multilevel
> formula (4.9) at $(g_{01}, \Delta, \alpha) = (120, -2700, -210)$ MHz
> gives $\chi/2\pi \approx -0.385\,\text{MHz}$. The sign must agree
> between simulation, test V2c, and the analytic formula. If the sign
> disagrees, the convention has been silently changed somewhere and
> `test_chi_reference_sign_is_negative` (added to `test_dispersive.py`)
> is the gate.

**Expected values for V2 at the synthetic reference seed (analytic-scaling estimate).**

At the seeded reference point, $\lambda = g_{01}/|\Delta| \approx 0.044$,
so $\lambda^2 \approx 2 \times 10^{-3}$. The two-level pull would be
$g_{01}^2/\Delta \approx -5.3\,\text{MHz}$, but the multilevel harmonic
formula (4.9) suppresses this by $\alpha/(\Delta+\alpha) \approx 0.072$,
giving $\chi/2\pi \approx -0.385\,\text{MHz}$ under the committed
multilevel model. **The fourth-order correction is amplified relative to
the suppressed multilevel χ.** Bare 4th-order corrections scale as
$O(\lambda^2)$ relative to the 2-level shift ($\sim 0.011$ MHz absolute);
expressed as a fraction of the multilevel $\chi$, this is
$\sim 0.011/0.385 \approx 0.03$, i.e. $\sim 3\%$ relative. A $10^{-4}$
gate is therefore inappropriate at any point that uses the same
$(\Delta, \alpha)$ ratio, because the suppression factor
$\alpha/(\Delta+\alpha) \approx 0.07$ is independent of $\lambda$ and
amplifies the relative error to $\sim \lambda^2 / |\alpha/(\Delta+\alpha)| \sim 10^{-3}$
at $\lambda = 0.01$. **V2a is therefore defined at a synthetic
"uncancelled" anharmonicity ratio with $|\alpha/(\Delta + \alpha)| = O(1)$**
— specifically $\Delta/2\pi = -1000$ MHz, $\alpha/2\pi = -500$ MHz
($\alpha/(\Delta+\alpha) = 1/3$, no near-cancellation). At this point
the multilevel $\chi$ is $1/3$ of the 2-level $\chi$ — they are the
**same order of magnitude but do not agree**; 4th-order corrections
are unmasked rather than amplified, making the gate physically
meaningful.

> **[Status of the residual estimate].** The figure
> "$\sim 3 \times 10^{-4}$ relative residual at $\lambda = 0.01$ and
> $\alpha/(\Delta+\alpha) = 1/3$" is an **analytic-scaling estimate**
> from $\lambda^2/|\alpha/(\Delta+\alpha)|$, **not** a numerical
> Python-verified result. No code exists at the time this framework
> is written. The estimate is what motivates the V2a tolerance of
> $5 \times 10^{-4}$ as a cushion above the predicted residual. Module 1
> implementation will produce a verification script
> (`scripts/verify_v2a_residuals.py`) that computes the actual numerical
> residual at the V2a operating points; if the empirical residual differs
> materially from the estimate, the V2a tolerance and the surrounding
> prose in §4.5 are updated to reflect the empirical value (this is a
> physics finding, not a relaxation of the gate).

V2b at the seeded reference uses $\sim 5 \times 10^{-2}$ tolerance; the
gate should be empirically verified during Module 1 implementation
(tighten if observed residual is smaller; flag the multilevel formula
as inaccurate at the seed if larger — a physics finding in its own right).

**V2 is split into three tests**:

| Test | Coupling | Hamiltonian point | Formula compared | Tolerance |
|---|---|---|---|---|
| **V2a: perturbative self-consistency** | $g_{01}/\|\Delta\| = 0.01$ | synthetic uncancelled point: $\Delta/2\pi = -1000$ MHz, $\alpha/2\pi = -500$ MHz | analytic 2nd-order (4.8) vs numerical (4.10) | $\le 5 \times 10^{-4}$ relative (analytic-scaling estimate $\sim 3 \times 10^{-4}$ at this point; verified empirically during implementation) |
| **V2a-strict (optional)** | $g_{01}/\|\Delta\| = 0.003$ at the same uncancelled $(\Delta, \alpha)$ | as V2a | as V2a | $\le 10^{-4}$ relative |
| **V2a-fallback (if uncancelled point cannot be used)** | $g_{01}/\|\Delta\| = 0.01$ at the seeded $(\Delta, \alpha)$ | reference-regime $(\Delta, \alpha)$ with $\lambda = 0.01$ | as V2a | $\le 10^{-3}$ relative (relaxed for suppression) |
| **V2b: synthetic-seed self-consistency** (synthetic seed values; see §1.4; *not* Marxer's actual device) | $g_{01}/\Delta \approx 0.044$ (synthetic Marxer-style seed) | synthetic-seed $(\Delta, \alpha)$ | analytic (4.8) vs numerical (4.10) at the *same* seeded inputs | $\le 5 \times 10^{-2}$ relative (model-calibrated) |
| **V2c: reference-sign check** | reference regime ($\Delta < 0, \alpha < 0$) | reference-device $(\Delta, \alpha)$ | numerical $\chi$ has the expected negative sign for the configured $(\Delta, \alpha)$ | exact (`test_chi_reference_sign_is_negative`) |

V2a validates that the **formula itself** is implemented correctly
(perturbative agreement in a regime where higher orders are negligible).
V2b validates **internal simulator consistency** at the reference seed —
analytic (4.8) and numerical (4.10) computed from the same JC Hamiltonian
with the same numerically-computed matrix elements should agree. V2c is
a **named sign check at the reference regime**, not a general "sign of
$\chi$ vs sign of $\Delta$" rule: the sign of $\chi$ under the multilevel
formula $\chi \sim g_{01}^2 \alpha / [\Delta(\Delta+\alpha)]$ depends on
the combined sign structure of $(\Delta, \alpha, \Delta+\alpha)$, not on
$\Delta$ alone. For the reference regime this evaluates to $\chi < 0$
and V2c checks that.

> **[Synthetic seed, not joint Marxer calibration].** The reference tuple
> $\{g_{01}/2\pi = 120\,\text{MHz},\ \Delta/2\pi = -2700\,\text{MHz},\ \alpha/2\pi = -210\,\text{MHz}\}$
> is a **synthetic Marxer-style seed** chosen for internal consistency of
> Stage 06's reference scenario. It is **not** an extraction of Marxer's
> 2508.16437 device-table parameters. Marxer reports
> $\omega_{01}/2\pi = 4.799/4.910\,\text{GHz}$, $\omega_r/2\pi = 6.190/6.350\,\text{GHz}$,
> $\chi/2\pi = 2.5/2.6\,\text{MHz}$, $\kappa/2\pi = 6.1/3.4\,\text{MHz}$ —
> none of which match the synthetic seed. Marxer is contextual motivation,
> not a regression target. See §1.4 for the side-by-side table.
>
> **Consequence for V2.** V2a and V2c are physics-validation gates that
> pass or fail on their own terms. V2b is a **self-consistency gate on
> the synthetic-seeded inputs**, not a regression against any external
> $\chi$ value. §1.4 is therefore not a regression target; it is a
> seeding convention that makes the numbers reproducible across runs.

The numerical $\chi$ is extracted from the diagonalization of the full
time-independent Hamiltonian (3.9) at zero drive, by identifying the states
adiabatically connected to the unperturbed
$|\text{transmon}=q\rangle \otimes |\text{photon}=n\rangle$ states and
computing:

$$(\chi_1 - \chi_0)_\text{num} \equiv \bigl[E(|1,1\rangle) - E(|1,0\rangle)\bigr] - \bigl[E(|0,1\rangle) - E(|0,0\rangle)\bigr] \qquad \text{[Exact, by construction; the *full* cavity pull, before the convention division by 2]} \tag{4.10}$$

> **[Naming convention — `cavity_pull_num` vs `chi`].** Equation (4.10)
> defines the **full cavity pull** $\chi_1 - \chi_0$ — twice the
> observable $\chi$ defined by (4.7). To prevent the most common
> Stage 06 bug (silent factor-of-2 between the analytic and numerical
> code paths), the Python API names follow this discipline:
> - `cavity_pull_num` returns the full pull (4.10) — i.e. $\chi_1 - \chi_0$.
> - `chi_num` returns $(\chi_1 - \chi_0)/2$ — i.e. the observable $\chi$
>   matching (4.7) and the analytic `dispersive_shift_full()`.
> - `dispersive_shift_from_simulation()` returns `chi_num` by default
>   (the post-division observable), with the underlying full-pull
>   value accessible via the `.cavity_pull` attribute on the result
>   object for debugging and validation.
>
> The division by 2 happens in exactly **one place** in
> `physics/dispersive.py` (the `_full_pull_to_chi` helper); every
> downstream code path goes through that helper, and it is unit-tested
> by `test_dispersive_pull_to_chi_division_is_factor_of_2_exact`.

with no factor of $1/2$ in (4.10) itself (we report the full pull
$\chi_1 - \chi_0$ at this level and divide by 2 at the API level,
following the convention of (4.7)). This convention-split is the most
error-prone part of the module; the division happens in exactly one
place in `dispersive.py` and is tested.

### 4.6 Script connection for §4

| Equation | Script | Function | Test |
|---|---|---|---|
| (4.6) two-level $\chi$ | `physics/dispersive.py` | `dispersive_shift_two_level()` | unit test |
| (4.8) full multilevel | `physics/dispersive.py` | `dispersive_shift_full()` | unit test |
| (4.10) numerical $\chi$ | `physics/dispersive.py` | `dispersive_shift_from_simulation()` | V2 |
| Reference-regime sign of $\chi$ | `physics/dispersive.py` | — | added `test_chi_reference_sign_is_negative` |

### 4.7 Edge-truncation note for the highest retained level

> **[Truncation artifact at the edge].** The convention $g_{N-1, N} = 0$
> (no coupling out of the highest retained transmon level $N - 1$) is a
> **truncation artifact**, not a physical statement. Stage 06 uses
> $\chi_0$ and $\chi_1$ as physical readout quantities (the only ones
> that enter the synthetic-seed multilevel $\chi = (\chi_1 - \chi_0)/2$
> definition); $\chi_{N-1}$ at the edge is biased by the truncation and
> **should not be interpreted physically** unless convergence in
> $N_\text{transmon}$ is checked. The $N_\text{transmon}$ convergence
> check is part of V1a (truncation convergence) and the truncation convergence sweep
> in `tests/test_physics_validation.py::test_truncation_convergence`.


## 5. Open-System Dynamics: the Lindblad Master Equation

Real transmon qubits are open systems: coupled to finite-temperature electromagnetic
baths, phonons, quasi-particles, flux noise, charge noise. The Born-Markov
approximation reduces this to a Lindblad master equation on the reduced
system density matrix $\rho$, parameterized by a small set of phenomenological
rates.

### 5.1 From microscopic bath to GKSL form

The Born-Markov-secular derivation (standard; see Breuer & Petruccione 2002,
§3.3; Blais et al. RMP 2021, §V) starts from a total Hamiltonian

$$H_\text{tot} = H_\text{sys} + H_\text{bath} + H_\text{sys-bath} \qquad \text{[Exact]} \tag{5.1}$$

and makes three approximations:

- **[Approximation, Born].** The system-bath coupling is weak relative to both
  $H_\text{sys}$ and the bath internal dynamics, so the full state factorizes
  as $\rho_\text{tot}(t) \approx \rho(t) \otimes \rho_\text{bath}(0)$ to
  leading order.
- **[Approximation, Markov].** The bath correlation times are much shorter
  than the system time scales of interest, so the memory kernel reduces to a
  delta function and the dynamics is time-local.
- **[Approximation, Secular].** Fast-oscillating terms at frequencies much
  larger than the decay rates are averaged out.

The result is the GKSL (Gorini-Kossakowski-Sudarshan-Lindblad) form:

$$\dot\rho = -i [H, \rho] + \sum_\mu \mathcal{D}[L_\mu] \rho, \qquad
\mathcal{D}[L] \rho \equiv L \rho L^\dagger - \tfrac{1}{2}\{L^\dagger L, \rho\} \qquad \text{[Exact, given 5.1 approximations]} \tag{5.2}$$

where $H$ is the rotating-frame Hamiltonian (3.9) and $\{L_\mu\}$ is the set
of **collapse operators**, each carrying the structure of one decoherence
channel.

**[Convention warning].** Some references (e.g. Gardiner & Zoller, *Quantum Noise*)
write the Lindblad dissipator in the expanded half-prefactor form
$\mathcal{D}[L] = \tfrac{1}{2}(2 L\rho L^\dagger - L^\dagger L\rho - \rho L^\dagger L)$.
This is **algebraically identical** to (5.2) — $\frac{1}{2}(2L\rho L^\dagger) = L\rho L^\dagger$
and $-\frac{1}{2}(L^\dagger L \rho + \rho L^\dagger L) = -\frac{1}{2}\{L^\dagger L, \rho\}$.
The rate parameter inside $L = \sqrt\gamma A$ is the same in both forms; no
conversion factor is needed to go between them.

> **Do not convert rates solely from the visual form of the dissipator.**
> A factor-of-two adjustment is only warranted if the paper places the rate
> *outside* the dissipator (e.g. writes $\gamma \mathcal{D}[A]\rho$ with $A$
> a bare operator, rather than absorbing the rate into $L = \sqrt\gamma A$)
> *and* the normalization of that external rate differs from ours. When
> transcribing a rate from a reference, check: (a) is the rate parameter
> outside the dissipator or inside $L$? (b) if outside, does the paper
> define $T_1 = 1/\gamma$ or $T_1 = 2/\gamma$? Only after both checks is a
> unit conversion determined. Our rates $\gamma_1, \gamma_\varphi, \kappa$
> are defined such that (5.2) with $L = \sqrt\gamma A$ reproduces the
> standard $T_1 = 1/\gamma_1$ and $\kappa = \text{FWHM in angular units}$.
> QuTiP uses the same convention by default.

### 5.2 Collapse operators for the transmon-resonator system

The five channels we include (Module 1 spec §3.4) and their collapse operators:

1. **Resonator decay** ($\kappa$, photon leakage to the measurement line):
$$L_\kappa = \sqrt{\kappa (1 + \bar n_\text{th,r})}\, a \qquad \text{[Standard]} \tag{5.3}$$

2. **Resonator thermal heating** (if the line temperature is above $\hbar\omega_r/k_B$,
   usually negligible at $\omega_r/2\pi = 7.3\,\text{GHz}$ and $T \sim 30\,\text{mK}$
   since $\hbar\omega_r/k_B \approx 350\,\text{mK} \gg T$, so
   $\bar n_\text{th,r} \approx e^{-\hbar\omega_r/k_B T} \approx 10^{-5}$):
$$L_{\kappa,\text{th}} = \sqrt{\kappa\, \bar n_\text{th,r}}\, a^\dagger \qquad \text{[Included only when $\bar n_\text{th,r} > 0$]} \tag{5.4}$$

3. **Qubit relaxation** ($T_1$ decay) — **list of secular collapse operators
   at distinct Bohr frequencies**, one per adjacent transition:

$$\{L_{\gamma_1}^{(j)}\}_{j=1}^{N_\text{transmon}-1}, \qquad L_{\gamma_1}^{(j)} = \sqrt{\gamma_{j \to j-1}\,(1 + \bar n_\text{th,q})}\; |j-1\rangle\langle j|_\text{transmon}, \qquad \gamma_{j \to j-1} = \gamma_1 \cdot \frac{|\langle j-1 | \hat n | j \rangle|^2}{|\langle 0 | \hat n | 1 \rangle|^2} \qquad \text{[Standard multilevel secular $T_1$]} \tag{5.5}$$

   in the **transmon eigenbasis**. Each $|j\rangle \to |j-1\rangle$
   transition is a *separate* collapse operator passed as a distinct
   element of `c_ops` to `qutip.mesolve`, not summed into a single
   operator. This is the correct **secular multilevel GKSL form**
   (§5.1, secular approximation): the secular condition treats distinct
   Bohr frequencies $\omega_{j,j-1}$ independently when
   $|\omega_{j,j-1} - \omega_{k,k-1}| \sim |\alpha|$ is large compared
   with the decay rates and the inverse bath-correlation /
   coarse-graining timescale. For Stage 06, $|\alpha|/2\pi \sim 210\,\text{MHz}$
   while $\gamma_1, \gamma_\varphi, \kappa$ are all MHz or below, so the
   per-transition collapse-operator list is the documented default
   (Breuer & Petruccione §3.3).
   [Unverified: this assumes the transition splittings are resolved on
   the relevant coarse-graining timescale. A nonsecular local-oscillator
   dissipator with a single summed lowering operator could be explored
   as an alternative model if this condition fails; it would give the
   same $T_1$ in the 2-level truncation and differ only weakly at the
   multilevel level. Stage 06 adopts the secular per-transition list
   as the default.]

   > **[Bath spectral-density assumption].** The matrix-element-scaled
   > rates $\gamma_{j\to j-1} = \gamma_1\,|\langle j-1|\hat n|j\rangle|^2 / |\langle 0|\hat n|1\rangle|^2$
   > implicitly assume the relevant environmental spectral density
   > $J(\omega)$ is approximately constant across the retained adjacent
   > transition frequencies $\omega_{j,j-1}$. For typical transmon
   > parameters this is acceptable because $\omega_{j,j-1}$ varies by
   > $|\alpha| \sim 210$ MHz across $j$, which is small compared with
   > the GHz-scale qubit frequency, and the dominant Purcell-like and
   > dielectric-loss spectral densities are smooth on that scale.
   > **Stricter form:** if $J(\omega)$ has structure on the scale of
   > $|\alpha|$, the rates should include the bath-spectral-density
   > ratio $J(\omega_{j,j-1})/J(\omega_{01})$ as an additional factor.
   > Stage 06's default omits this factor and is documented as such;
   > Module 1's `DecoherenceParams` could be extended with a
   > `bath_spectral_density: Callable[[float], float] | None = None`
   > field if a future device requires explicit frequency dependence.

4. **Pure dephasing** ($T_\varphi$) — 2-level form:
$$L_{\gamma_\varphi}^\text{2-level} = \sqrt{2\gamma_\varphi}\, |1\rangle\langle 1|_\text{transmon} \qquad \text{[2-level form]} \tag{5.6}$$
   in the transmon eigenbasis. The factor of $2$ is such that the
   Bloch-equation $T_2$ coherence decays at rate $\gamma_\varphi + \gamma_1/2$:
   $$\frac{1}{T_2} = \frac{1}{2 T_1} + \frac{1}{T_\varphi} \qquad \text{[Exact for 2-level, Bloch equation]} \tag{5.7}$$
   The multi-level generalization (5.10) is derived in §5.4 — a single
   calibrated diagonal collapse operator (appropriate because it commutes
   with $H_0$, so the secular concern does not apply to dephasing).

5. **Qubit thermal heating** ($\bar n_\text{th} > 0$) — **list of secular
   upward collapse operators**, one per adjacent transition, with
   detailed-balance factors:

$$\{L_{\gamma_1,\text{th}}^{(j)}\}_{j=0}^{N_\text{transmon}-2}, \qquad L_{\gamma_1,\text{th}}^{(j)} = \sqrt{\gamma_{j \to j+1}\,\bar n_\text{th,q}}\; |j+1\rangle\langle j|_\text{transmon}, \qquad \gamma_{j \to j+1} = \gamma_1 \cdot \frac{|\langle j+1 | \hat n | j \rangle|^2}{|\langle 0 | \hat n | 1 \rangle|^2} \qquad \text{[Included only when $\bar n_\text{th,q} > 0$]} \tag{5.8}$$

   Each upward transition is a *separate* element of `c_ops`, matching
   the structure of (5.5). For $\bar n_\text{th,q} \ll 1$ (standard
   dilution-fridge regime), only the $0 \to 1$ term is numerically
   important.

> **[Approximation — common $\bar n_\text{th,q}$ across transitions].**
> Equation (5.8) uses a single thermal occupation $\bar n_\text{th,q}$
> for every adjacent transmon transition. Strictly, the bath occupation
> at transition frequency $\omega_{j,j+1}$ is $\bar n_\text{th}(\omega_{j,j+1})$,
> which depends on the transition through the Bose-Einstein factor at
> finite temperature. The relevant point at dilution-fridge temperatures
> ($T \sim 30\,\text{mK}$, $\hbar\omega/k_B T \sim 7$ for $\omega/2\pi \sim 4.5$ GHz)
> is that the **absolute thermal occupation is exponentially small**
> ($\bar n_\text{th} \sim e^{-\hbar\omega/k_B T} \sim 10^{-3}$ across the
> entire $j = 0, 1, 2, 3$ ladder); the dynamical effect of the
> approximation is therefore negligible at the precision Stage 06 cares
> about, regardless of how the per-transition Bose factors compare to
> each other. Stage 06 documents this as a **scoped simplification**, not
> a derivation; for studies at higher temperatures (several hundred mK
> or above) where $\bar n_\text{th}$ is no longer negligible, the
> per-transition $\bar n_\text{th}(\omega_{j,j+1})$ must be evaluated
> explicitly because both the absolute size *and* the inter-transition
> variation start to matter.

> **[Terminology note].** In circuit QED, "dressed" commonly means the
> eigenbasis of the *coupled* transmon+resonator Hamiltonian, whereas
> "transmon eigenbasis" means the bare-transmon diagonalization before
> any photon-hybridization mixing. Our collapse operators (5.3)-(5.8) and
> (5.10) act in the **transmon eigenbasis** (bare-transmon diagonalized,
> no photon admixture). The Purcell physics (§5.3) involves the *coupled*
> dressed basis implicitly: the transmon-eigenbasis $|1\rangle$ has a
> $(g_{01}/\Delta)$ admixture of resonator photon when the full coupled
> Hamiltonian is diagonalized, and that admixture is what acquires the
> $\kappa$ decay rate. The difference matters for high-coupling regimes
> but not for our dispersive limit; still, the document should say
> "transmon eigenbasis" where it means that (above) and "coupled dressed
> states" where it means the full diagonalization (§5.3).

### 5.3 Why Purcell is not a separate collapse operator

The Purcell effect — the enhanced $T_1$ decay induced by the resonator bath —
is **not** a separate collapse operator in our formulation. It emerges
automatically from the combination of the transmon-resonator coupling (3.4)
and the resonator decay (5.3): a qubit excitation can hybridize with a resonator
photon via $g_{01}$, and the hybridized photonic component decays at rate $\kappa$,
giving the qubit an effective decay rate

$$\gamma_\text{Purcell} \approx \left(\frac{g_{01}}{\Delta}\right)^2 \kappa \qquad \text{[Approximation, two-level + dispersive]} \tag{5.9}$$

**Derivation of (5.9).** In the dispersive regime, the dressed $|1\rangle$
state has the form $|1\rangle_\text{dressed} \approx |1, 0\rangle -
(g_{01}/\Delta)\, |0, 1\rangle + O(g_{01}^2/\Delta^2)$, a small admixture of
one-photon qubit-ground. The photonic component then decays at rate $\kappa$,
and by first-order perturbation theory the dressed-qubit state inherits
$|g_{01}/\Delta|^2 \cdot \kappa$ as an additional decay rate.

In the code, this means two distinct things:

(a) **V4b: Purcell-emergence physics test.** Setting $\gamma_1 = \gamma_\varphi = 0$
and $\bar n_\text{th} = 0$ in `DecoherenceParams` (so no explicit qubit
collapse operators), integrating the full JC + $\kappa$ dynamics, and
checking that the observed $|1\rangle$-population decay rate matches
analytic (5.9) to $\le 5\%$. This validates that Purcell emerges from
the JC-coupling + cavity-decay combination as expected — a standalone
physics gate independent of any error-budget framework.
[Tolerance is looser than $1\%$ because (5.9) is itself only a
leading-order expression.]
**Extraction method (mandatory):** the rate is extracted **either** (i)
from the Liouvillian eigenvalue adiabatically connected to the qubit
excitation, **or** (ii) from a single-exponential fit to
$\rho_{11,\text{transmon}}(t)$ restricted to $t > 5/|\Delta|$, **after**
the initial JC hybridization transient has decayed. Naive global
single-exponential fitting from $t=0$ is forbidden — it is contaminated
by $O(g_{01}/\Delta)$ small-amplitude coherent exchange and gives a
biased rate.

For method (i), the eigenmode identification is precise:

> Diagonalize the Liouvillian $\mathcal{L}$. For each right eigenmode
> $\rho^{(k)}$ with eigenvalue $\lambda_k$, compute its Hilbert–Schmidt
> overlap with the traceless population observable
> $\hat O = |1\rangle\langle 1|_\text{transmon} - |0\rangle\langle 0|_\text{transmon}$
> after tracing out the resonator. Define
> $\rho^{(k)}_\text{transmon} \equiv \mathrm{Tr}_\text{resonator}\,\rho^{(k)}$
> (the reduced operator on the transmon Hilbert space; it is generally
> *not* a state, since $\rho^{(k)}$ is a non-equilibrium right eigenmode
> of $\mathcal{L}$, not a density matrix). The overlap is then a
> Hilbert–Schmidt inner product **on the transmon Hilbert space**:
>
> $$o_k = \frac{\bigl|\,\mathrm{Tr}_\text{transmon}\!\bigl[\hat O^\dagger\, \rho^{(k)}_\text{transmon}\bigr]\,\bigr|}{\bigl\|\rho^{(k)}_\text{transmon}\bigr\|_\text{HS} \cdot \bigl\|\hat O\bigr\|_\text{HS}}$$
>
> where $\|A\|_\text{HS} = \sqrt{\mathrm{Tr}_\text{transmon}(A^\dagger A)}$
> is the transmon-Hilbert-space Hilbert–Schmidt norm (not the norm on
> the full transmon⊗resonator space). The Liouvillian mode identified
> as the "Purcell rate" is then selected via a deterministic three-stage
> filter:
>
> 1. **Restrict to decaying modes:** $\text{Re}(\lambda_k) < 0$ (steady
>    states and growing modes excluded).
> 2. **Restrict to the rate window:** keep only modes whose
>    $|\text{Re}(\lambda_k)|$ is within one decade of the perturbative
>    Purcell rate $\gamma_\text{Purcell} = (g_{01}/\Delta)^2 \kappa$,
>    i.e. $0.1\,\gamma_\text{Purcell} \le |\text{Re}(\lambda_k)| \le 10\,\gamma_\text{Purcell}$.
> 3. **Select by overlap, then by closeness:** among the modes passing
>    (1)–(2), pick the one with the largest $o_k$. If two modes are
>    within $10^{-3}$ in $o_k$ (degenerate-overlap tie), break the tie
>    by smallest $||\text{Re}(\lambda_k)| - \gamma_\text{Purcell}|$.
>
> **Numerical guard.** If
> $\bigl\|\rho^{(k)}_\text{transmon}\bigr\|_\text{HS} < 10^{-12}$ (e.g.
> the eigenmode lives almost entirely in resonator coherences and traces
> to zero on the transmon subspace), set $o_k = 0$ and skip the mode
> rather than divide by a near-zero norm.
>
> The selected rate is $|\text{Re}(\lambda_k)|$. This three-stage filter
> is well-defined and tie-breaking is deterministic. Pure
> "$|1\rangle\langle 1|$ support" is not used because Liouvillian
> eigenmodes are not pure projectors.

Method (i) is preferred because it is unambiguous; method (ii) is
acceptable as a quick sanity check. Implemented in
`tests/test_physics_validation.py::test_purcell_emergence_from_full_JC`.

> **[V4b operating-point note].** At the synthetic seed (§1.4),
> $\gamma_\text{Purcell}/2\pi \approx 9.88\,\text{kHz}$
> (i.e., $\gamma_\text{Purcell} \approx 6.2 \times 10^4\,\text{s}^{-1}$,
> $T_\text{Purcell} = 1/\gamma_\text{Purcell} \approx 16.1\,\mu$s — note
> that "kHz" here is the $\gamma/2\pi$ angular-to-linear conversion, not
> the inverse lifetime in s$^{-1}$, to avoid the common confusion of
> reading "9.88 kHz" as $T \approx 100\,\mu$s), which is small
> enough that the Liouvillian-eigenvalue method (i) is the reliable
> default — a time-domain fit at this rate would need integration
> windows comparable to $1/\gamma_\text{Purcell} \sim 16\,\mu\text{s}$
> to resolve cleanly, much longer than the readout window. For sanity
> checks of the time-domain method (ii), V4b can additionally be run at
> a **Purcell-validation point** with deliberately larger $g_{01}/|\Delta|$
> (e.g. $g_{01}/2\pi = 200\,\text{MHz}$, $\Delta/2\pi = -1500\,\text{MHz}$,
> giving $\gamma_\text{Purcell}/2\pi \sim 90\,\text{kHz}$), still in the
> dispersive regime ($\lambda \sim 0.13$, $\lambda^2 \sim 0.018$), where
> the decay is observable on a $\sim 1\,\mu\text{s}$ timeline; both
> extraction methods then agree on the same rate. The Liouvillian method
> remains the default; the Purcell-validation point is a
> method-cross-check.

(b) **Module 2's $\Delta F_\text{Purcell}^\text{diag}$ analytic attribution.**
A different quantity from V4b: it asks "of the $T_1$-mediated infidelity
cost at the operating point, how much is mediated by the Purcell
channel?" This uses the analytic rate (5.9) combined with the
$\partial \mathcal{I}/\partial(\gamma_1\tau)$ slope from a single
$\gamma_1$-perturbation simulation (§8.5, eq. 8.7). It is reported
as a Group A′ diagnostic, **excluded** from the B1 closure, because
Purcell is already implicit in $F_\text{ideal}$ (§8.1).

> **Physical interpretation.** Equation (5.9) is why low-$\kappa$ resonators
> are used for long-$T_1$ qubits, but a *Purcell filter* is then needed to
> maintain fast readout — the filter makes $\kappa$ effectively large at
> $\omega_r$ while keeping it effectively small at $\omega_q$, breaking the
> $(g_{01}/\Delta)^2 \kappa$ tradeoff. Stage 06 uses a synthetic Marxer-style
> parameter scale (§1.4) but **does not model the Purcell filter or
> shelving readout** used in the actual Marxer device — the simulator
> treats $\kappa$ as a single frequency-independent rate. The simulated
> Purcell contribution (5.9) is therefore an **unfiltered-model estimate**:
> it is the Purcell rate one would expect without a filter. Whether it
> bounds the real device's Purcell cost depends on the actual filter
> transfer function (how much $\kappa$ at $\omega_q$ is suppressed),
> which we do not attempt to quantify. Marxer's reported $T_1$ is
> achieved partly by the filter,
> which our model cannot reproduce.

### 5.4 Multi-level dephasing (number-operator default — phenomenological)

For a strict 2-level system, (5.6) is the full story. For a transmon where
we keep $|j\rangle$ with $j \ge 2$ in the simulation, pure dephasing
generalizes to a **single calibrated diagonal collapse operator**.

> **[Status — phenomenological, not derived from a microscopic noise spectrum].**
> The multilevel dephasing operator below is **phenomenological**: it
> calibrates the diagonal spectrum $\lambda_j$ to give a chosen pairwise
> rate structure, but it does not derive that structure from an
> underlying noise spectral density. A microscopically-derived
> multilevel dephasing operator depends on derivatives of level energies
> with respect to the noisy parameter $x$ (flux, charge, critical
> current, photon number, etc.):
> $$\lambda_j \propto \frac{\partial E_j}{\partial x}\bigg|_{x=x_0}$$
> Different fluctuating parameters give different multilevel structures.
> The number-operator default below corresponds to the phenomenological
> case where $\partial E_j/\partial x \propto j$ (i.e. the noise couples
> linearly to level number), which is reasonable for common-mode
> qubit-frequency fluctuations but is **not universally derived**. The
> C2d test (§5.4) validates the *implementation* of the chosen model,
> not the *physical correctness* of the model — the latter requires
> external justification (noise-spectroscopy data) or a fit to a
> specific device's $T_2$-vs-level data.

**Rate calibration.** For any diagonal operator
$L = \sum_j \lambda_j |j\rangle\langle j|$, the GKSL dissipator damps
coherence $\rho_{mn}$ at rate

$$\Gamma_{\varphi,mn} = \tfrac{1}{2}(\lambda_m - \lambda_n)^2 \qquad \text{[Exact, from GKSL action on off-diagonal elements]} \tag{5.9a}$$

This framework supports three physically-motivated choices for the
diagonal spectrum $\lambda_j$, exposed via a `dephasing_model` config option:

**(i) Number-operator model (default).** Phenomenological model for a
fluctuating qubit transition frequency that acts on the system via a
linear coupling to level number: $\delta\omega(t) \cdot \sum_j j|j\rangle\langle j|$.
This gives $\lambda_j = j\sqrt{2\gamma_\varphi}$ and:

$$L_{\gamma_\varphi}^\text{number} = \sqrt{2\gamma_\varphi}\, \sum_{j \ge 0} j\, |j\rangle\langle j| \qquad \text{[Phenomenological frequency-noise model — see status note above]} \tag{5.10}$$

Pairwise coherence decay rates from (5.9a):

$$\Gamma_{\varphi, mj}^\text{number} = \gamma_\varphi (j - m)^2$$

So $\Gamma_{01} = \gamma_\varphi$, $\Gamma_{12} = \gamma_\varphi$,
$\Gamma_{02} = 4\gamma_\varphi$. This is **a phenomenological model for
common-mode qubit-frequency fluctuations** with a level-number-quadratic
dephasing structure consistent with the phenomenology of transmon $T_2$
measurements; it is **not** a universally-derived multilevel dephasing
model. The microscopic origin can be flux-noise on tunable elements
(SQUID/coupler), drive-phase noise, two-level-system fluctuators, or
thermal-photon population in a nearby resonator; Stage 06 does not commit
to a microscopic mechanism, only to the level-number-quadratic dephasing
structure under the "linear-in-$j$" phenomenological assumption. It
reduces to Convention 8 at $N_\text{transmon} = 2$ with
$\Gamma_{01} = \gamma_\varphi$ matching the 2-level rate exactly.

**(ii) $\sqrt j$ model (alternative, calibrated to $\Gamma_{0j} = j\gamma_\varphi$).**
$\lambda_j = \sqrt{2 j \gamma_\varphi}$, giving
$\Gamma_{\varphi, mj} = \gamma_\varphi (\sqrt j - \sqrt m)^2$. Usable as a
phenomenological fit if data says the $\Gamma_{0j}$ rates scale *linearly*
in $j$ rather than quadratically, but this is unusual; do not select
without justification.

**(iii) Two-level only (default for $N_\text{transmon} = 2$).**
$L_{\gamma_\varphi} = \sqrt{2\gamma_\varphi}|1\rangle\langle 1|$ per
Convention 8. All three models reduce to this at $N_\text{transmon} = 2$
by construction.

**Config-level enforcement.** `DecoherenceParams` exposes:

```python
dephasing_model: Literal["number_operator", "sqrt_level", "two_level_only"] = "number_operator"
```

Implementation in `build_collapse_operators()` selects the diagonal
spectrum accordingly. Changing the model changes the multilevel dephasing
rates; Module 2's $\gamma_\varphi$ channel attribution therefore depends
on this choice, and the choice should be surfaced in the
`RecommendationReport` metadata.

> **[Why number-operator is the default].** Two alternative multilevel
> dephasing models were considered. The difference-of-projectors form
> $L_j = \sqrt{2j\gamma_\varphi}(|j\rangle\langle j| - |0\rangle\langle 0|)$
> gives $\Gamma_{0j} = 4j\gamma_\varphi$, a factor-of-4 rate inflation
> relative to the intended $j\gamma_\varphi$ per pair. The single-operator
> $\sqrt j$ form $\lambda_j = \sqrt{2j\gamma_\varphi}$ recovers
> $\Gamma_{0j} = j\gamma_\varphi$ but yields
> $\Gamma_{12} \approx 0.17\gamma_\varphi$, which is unusual for
> frequency-noise dephasing. The number-operator model (5.10) gives
> $\Gamma_{01} = \Gamma_{12} = \gamma_\varphi$, which matches the
> phenomenology of common-mode qubit-frequency noise modeled as a
> linearly-coupled bath. The $\sqrt j$ form remains available for users
> who can justify it from data.

> **[Validation test — C2d, "operator calibration"].** `test_pairwise_dephasing_rates_calibrated`:
> prepare $(|m\rangle + |j\rangle)/\sqrt 2$ for all $0 \le m < j \le N_\text{transmon}-1$,
> run free evolution under the diagonal $L_{\gamma_\varphi}$, fit the
> off-diagonal decay rate, and verify it matches the model-specific
> prediction ($\gamma_\varphi(j-m)^2$ for number-operator,
> $\gamma_\varphi(\sqrt j - \sqrt m)^2$ for $\sqrt j$, $\gamma_\varphi$ for
> two-level-only) to $\le 5\%$. This test validates that the **implementation
> matches the chosen model**; it does **not** validate that the chosen
> model is the correct physics — that requires external justification or
> an experimental fit.

### 5.5 The QuTiP implementation

QuTiP's `qutip.mesolve` integrates equation (5.2) via a user-selectable ODE
integrator. The **default method is `'adams'`** — a variable-order
Adams-Bashforth-Moulton multistep method wrapped through SciPy's `zvode`
backend — appropriate for the non-stiff Lindbladian of our dispersive
readout problem. Alternative methods exposed through the `method` option
include `'bdf'` (for stiff problems), `'lsoda'` (automatic stiff/non-stiff
switching), `'dop853'` (8th-order Dormand-Prince), and QuTiP-native Verner
Runge-Kutta integrators `'vern7'` / `'vern9'`. Stage 06 uses the default
`'adams'` method unless a specific convergence issue requires otherwise;
this is a runtime choice, not a physics choice. The signature is

```python
# QuTiP 5.x API
result = qutip.mesolve(
    H=[H0, [H_drive, drive_callable]],  # time-dependent H
    rho0=initial_state,
    tlist=t_grid,
    c_ops=[L_1, L_2, ...],              # collapse operators
    e_ops=[a, a.dag()*a, ...],          # expectation values to record
    options={"nsteps": 5000, "rtol": 1e-8, "atol": 1e-10, "method": "adams"},
)
```

> **[API version note].** The `options=` argument in QuTiP 5.x takes a
> plain dict, not `qutip.Options(...)`. The older `qutip.Options` class is
> **deprecated in QuTiP 5.x; avoid for new code, use `options={...}`**.
> `physics/lindblad.py`
> should pin `qutip >= 5.0` in `pyproject.toml` and raise a clear
> version-mismatch error at import if an older QuTiP is loaded.

Tolerances of $10^{-8}$ relative are the starting point; whether they are
**sufficient** for our validation targets is an empirical question that
must be checked, not assumed.

> **[Validation test — V0, new].** `test_solver_convergence`: run the
> reference simulation at 4 tolerance levels
> $(\text{rtol}, \text{atol}) \in \{(10^{-6},10^{-8}), (10^{-7},10^{-9}), (10^{-8},10^{-10}), (10^{-9},10^{-11})\}$
> and record $F_\text{assign}$ and $\langle a\rangle(\tau_\text{end})$. Also
> sweep `nsteps` and time-grid resolution. Converged values must agree
> within $\le 10^{-5}$ relative across the last two refinements. If they
> don't, tighten tolerances until they do, and document the chosen
> configuration. This is a numerical-analysis gate, distinct from the
> physics gates V1-V4.

### 5.6 Script connection for §5

| Equation | Script | Function | Test |
|---|---|---|---|
| (5.2) GKSL form | `physics/lindblad.py` | implicit in QuTiP `mesolve()` | — |
| (5.3) resonator decay | `physics/lindblad.py` | `build_collapse_operators()` line 1 | — |
| (5.4) thermal heating | `physics/lindblad.py` | `build_collapse_operators()` line 2 | — |
| (5.5) qubit relaxation | `physics/lindblad.py` | `build_collapse_operators()` line 3 | V3 |
| (5.6), (5.10) dephasing | `physics/lindblad.py` | `build_collapse_operators()` line 4 | V4 |
| (5.7) Bloch $T_2$ relation | — | consistency check inside V4 | V4 |
| (5.9) Purcell rate | `physics/lindblad.py` | derived quantity, used by `analysis/purcell_isolation.py` | V4b |



## 6. The Readout Observable and Signal Processing

The readout observable is the complex resonator amplitude $\langle a \rangle(t)$
in the rotating frame — this is the quantity that, after heterodyne/homodyne
downconversion in the measurement chain, becomes the IQ-plane signal the
discriminator sees.

### 6.1 Why $\langle a\rangle$ is the right observable

In a homodyne measurement, the field emitted from the resonator is mixed with
a local oscillator at $\omega_d$ and integrated. The resulting voltage
(in units set by the amplifier gain and mixer conversion loss) is proportional
to $\mathrm{Re}[\langle a\rangle(t)\, e^{i\phi_\text{LO}}]$, where
$\phi_\text{LO}$ is the LO phase. A heterodyne measurement records both
quadratures simultaneously, giving the full complex $\langle a\rangle(t)$.

We track the complex $\langle a\rangle(t)$ throughout (equivalent to perfect
heterodyne), and the "choice of phase" happens at the discrimination step
(§7.2). This is slightly more general than pure homodyne — a homodyne
measurement corresponds to projecting the complex trajectory onto an
appropriately-chosen axis — and matches how post-TWPA amplifier chains
process the signal in modern superconducting-qubit experiments (Krantz et al.
2019, §V.B).

### 6.2 The IQ trajectory

For each initial qubit state $|q\rangle \in \{|0\rangle, |1\rangle\}$, the
simulator produces:

$$\langle a \rangle_q(t) = \mathrm{Tr}\bigl[a\, \rho_q(t)\bigr] \qquad \text{[Exact, expectation value]} \tag{6.1}$$

where $\rho_q(0) = |q\rangle\langle q| \otimes |0\rangle\langle 0|$ (qubit in
$|q\rangle$, resonator in vacuum). The two trajectories
$\langle a \rangle_0(t)$ and $\langle a \rangle_1(t)$ are the curves plotted
in Figure 1a — one pair of curves in the IQ plane, starting at the origin
(vacuum), driven outward by the readout pulse, reaching a steady-state
displacement on a timescale $\sim 1/\kappa$, then decaying back as the pulse
turns off.

> **[Centroid-Gaussian baseline: what it captures and what it misses].**
> The simulator produces the *ensemble-averaged* IQ trajectory
> $\langle a\rangle_q(t)$ from the Lindblad master equation. Sections §6
> and §7 then build assignment fidelity by treating the two
> qubit-state-conditional trajectories as deterministic centroids and
> adding circular Gaussian shot noise (§7.1). This **centroid + Gaussian**
> model captures: (a) the mean signal shaped by all coherent dynamics
> (drive ring-up, dispersive separation, $\langle a\rangle$ damping), and
> (b) ensemble-averaged decoherence as a *centroid displacement* (e.g.
> $\langle a\rangle_1$ moves toward $\langle a\rangle_0$ as $T_1$ decay
> reduces the population in $|1\rangle$ on average). It **does not**
> capture the single-shot non-Gaussianity that arises when a $T_1$ jump
> occurs *during* the integration window: a shot prepared in $|1\rangle$
> that decays at time $t^* \in [0,\tau]$ produces a trajectory whose
> integrated IQ point lies between the pure-$|0\rangle$ and pure-$|1\rangle$
> centroids, and the conditional distribution of those points is
> non-Gaussian (typically skewed toward the $|0\rangle$ centroid with a
> tail). For the synthetic seed, $T_{1,\text{intrinsic}} = 30\,\mu$s and
> $\tau = 500\,\text{ns}$, the bare single-shot decay probability is
> $1 - e^{-\tau/T_{1,\text{intrinsic}}} \approx 1.65\%$. **However, the
> *effective* qubit decay in any model where the cavity has been
> integrated out is $T_{1,\text{eff}} \approx 10.5\,\mu$s** (since
> $\gamma_\text{Purcell} \approx 6.2 \times 10^4\,\text{s}^{-1}$ is
> roughly twice $\gamma_{1,\text{intrinsic}} \approx 3.33 \times 10^4\,\text{s}^{-1}$
> at the seed; Convention 21). So the relevant decay probability for
> reduced-model jump-tail estimates is closer to
> $1 - e^{-\tau/T_{1,\text{eff}}} \approx 4.7\%$, comfortably in the
> regime where 99 %+ assignment fidelity is reported, so the omission
> matters. **Stage 06's baseline assignment model
> (§7) treats decoherence through centroid displacement, not stochastic
> single-shot trajectory branching.** This is adequate for first-order
> parameter sweeps and waterfall-style budget decompositions, but
> **underestimates or misrepresents assignment-distribution tails when
> $\tau/T_1$ is not negligible**. Hazra et al. 2025 (arXiv:2407.10934)
> emphasize that ordinary readout-fidelity reporting can miss
> readout-induced backaction in repeated-measurement settings; the same
> caution applies here. Before claiming absolute 99.9 % assignment
> fidelity at the synthetic seed, one of the following is required:
>
> - a **one-jump $T_1$ mixture model** (§7.3a) — closed-form, treats each
>   shot as either "no jump" (Gaussian around $\langle a\rangle_1$
>   centroid) or "one jump at time $t^*$" with $t^*$ exponentially
>   distributed, and the resulting integrated IQ trajectory pre-computed,
>   then the discriminator outcome is averaged over the jump-time
>   distribution; or
>
> - a **jump-tail stress test** (Module 1 V7 (jump-tail cross-check)) — at one
>   reference operating point, compare the centroid+Gaussian fidelity to
>   either the one-jump $T_1$ mixture (§7.3a, preferred) or to a
>   `qutip.mcsolve` trajectory ensemble interpreted as a jump-tail stress
>   test (not as a unique ground truth for the heterodyne IQ
>   distribution; see V7 definition for the precise framing), and
>   document the relative deviation.
>
> Module 4's optimization layer is permitted to use the centroid+Gaussian
> objective for the smooth, deterministic Pareto sweep (§10.6); the
> trajectory check then quantifies how much that smooth objective biases
> the optimum. The default is centroid+Gaussian *with the warning above
> propagated to the report*; if one-jump mixture or trajectory validation
> is required, it is a Module 2 / Module 4 add-on, not a baseline change
> to §7.

**Semi-analytic approximation (for intuition only).** In the dispersive
approximation with no qubit decoherence (resonator damping $\kappa$ retained), the intracavity amplitude executes a
driven-damped oscillation at effective frequency $\omega_r + \chi_q - \omega_d$:

$$\frac{d\langle a \rangle_q}{dt} = -i(\omega_r + \chi_q - \omega_d) \langle a\rangle_q - \frac{\kappa}{2} \langle a\rangle_q - i\varepsilon(t) \qquad \text{[Approximation, semi-classical dispersive limit]} \tag{6.2}$$

For drive symmetric between $|0\rangle$ and $|1\rangle$ (i.e.
$\omega_d = \omega_r + (\chi_0 + \chi_1)/2$) and constant drive
$\varepsilon(t) = \varepsilon_0$, the **intracavity** steady-state amplitudes
are $\langle a\rangle_{0,1}^\text{ss} = -i\varepsilon_0 / (i(\mp\chi) + \kappa/2)$
with $\chi = (\chi_1 - \chi_0)/2$, and the intracavity separation is:

$$|\Delta\langle a\rangle_\text{ss}| = \frac{2|\chi|\varepsilon_0}{(\kappa/2)^2 + \chi^2} \qquad \text{[Approximation, intracavity field]} \tag{6.3}$$

### 6.2a SNR scaling depends on normalization — choose one and state it

The claim "optimal readout at $\kappa \approx 2|\chi|$" appears in many
references but **requires specifying what is held fixed**. The three standard
choices give different optima:

**(i) Fixed drive amplitude $\varepsilon_0$ (the naive sweep).**
From (6.3), $|\Delta\langle a\rangle_\text{ss}|$ is maximized at
$\kappa \to 0$ with value $2\varepsilon_0/|\chi|$. There is **no finite
optimum** in $\kappa$ — smaller $\kappa$ always gives more intracavity
pointer separation. This is not useful as a design rule.

**(ii) Fixed mean photon number $\bar n$ (the standard design rule,
Gambetta et al. / Blais RMP §VI).** For a symmetric dispersive drive,
the steady-state photon number in either pointer state is
$\bar n = \varepsilon_0^2 / [(\kappa/2)^2 + \chi^2]$, so holding $\bar n$
fixed requires rescaling

$$\varepsilon_0 = \sqrt{\bar n}\,\sqrt{(\kappa/2)^2 + \chi^2} \qquad \text{[Exact, symmetric-drive steady state]} \tag{6.3a}$$

With this rescaling, the measurement rate (or equivalently the
output-field-matched-filter SNR² per unit time) becomes

$$\Gamma_\text{meas} \propto \frac{8\chi^2 \bar n \kappa}{\kappa^2 + 4\chi^2} \qquad \text{[Approximation, output-field matched filter]} \tag{6.4}$$

which is maximized at $\kappa = 2|\chi|$. This is the correct origin of
the "typical optimum" design rule. The $\sqrt{\bar n}\,(\kappa/2)$
rescaling that would hold only for an *empty* cavity ($\chi \to 0$) is
wrong at precisely the $\kappa \sim 2|\chi|$ regime where the optimum
sits; (6.3a) is the correct form.

**(iii) Fixed drive power out of the fridge.** Adds cable/amp
considerations; produces yet a third optimum. Out of scope.

> **[Framework convention].** Stage 06 uses the **fixed-mean-photon-number**
> convention for stating design rules and drawing regime-map overlays.
> The numerical SNR reported by `compute_assignment_fidelity()` is computed
> from the *actual* simulation (drive $\varepsilon_0$ held at whatever the
> config says), so the numerical optimum may shift from $\kappa \approx 2|\chi|$
> depending on how $\varepsilon_0$ is chosen relative to ionization-threshold
> photon-number limits. The dashed line at $\kappa/|\chi| \approx 2$ in
> Figure 1c is the design-rule reference, not a prediction of the simulation.

### 6.3 Output-field SNR and short-time asymptote

The measured quantity is the output field. Standard input-output theory
gives $a_\text{out} = a_\text{in} + \sqrt{\kappa_\text{ext}}\, a$, where
$a_\text{in}$ is the coherent drive tone arriving at the resonator's
external port and $\kappa_\text{ext} \le \kappa$ is the external
coupling rate. In Stage 06, the drive tone $a_\text{in}$ is treated as
a **common-mode baseline** that is subtracted from the measured complex
amplitude before centroid discrimination — physically by the LO and
mixer chain in a real heterodyne setup, computationally by removing the
qubit-state-independent reference IQ point $I_\text{ref} + iQ_\text{ref}$
from each pointer. The $a_\text{in}$ contribution is therefore absent
from $I_q - I_\text{ref}$, and the relevant qubit-state-discriminating
quantity reduces to $\sqrt{\kappa_\text{ext}}\, \langle a\rangle_q$.
Integrating this baseline-subtracted output amplitude over a window
$[t_\text{start}, t_\text{end}]$ of length $\tau$ gives the integrated
homodyne/heterodyne IQ point

$$I_q + iQ_q = \sqrt{\kappa_\text{ext}} \int_{t_\text{start}}^{t_\text{end}} \langle a\rangle_q(t)\, dt \qquad \text{[Definition; output-field normalization, baseline-subtracted]} \tag{6.5}$$

> **[Convention on baseline subtraction].** `ReadoutResult.integrated_iq()`
> returns the baseline-subtracted IQ defined by (6.5). The $a_\text{in}$
> contribution is *not* present in this output; only the resonator
> response $\sqrt{\kappa_\text{ext}}\, \langle a\rangle_q$ remains.
> If a future extension reports raw (unsubtracted) IQ, an additional
> field `iq_baseline_subtracted: bool = True` should be added; the
> default and current behavior is `True`.

The integrated IQ noise is a zero-mean Gaussian with **variance proportional
to $\tau$** at the vacuum-noise floor (half a photon per unit bandwidth,
integrated over the bandwidth $\sim 1/\tau$). So $\sigma_\parallel \propto \sqrt\tau$,
*not* $1/\sqrt\tau$; the signal grows as $\tau$ and the noise as $\sqrt\tau$,
giving an SNR that grows as $\sqrt\tau$ in the $\tau \ll \min(1/\kappa, T_1)$ window.
At fixed $\bar n$ and on-resonance drive:

$$\mathrm{SNR}^2 \approx \eta\, \Gamma_\text{meas} \cdot \tau \qquad \text{[Approximation, short-$\tau$ output-field]} \tag{6.6}$$

with $\Gamma_\text{meas}$ from (6.4) and $\eta \in [0, 1]$ the total
measurement efficiency (amplifier noise temperature, cable losses,
mixer/ADC nonidealities, finite $\kappa_\text{ext}/\kappa$).

### 6.3a Integration-window-dependent IQ noise — `sigma_parallel(τ)`

Because integrated IQ noise scales as $\sqrt\tau$ (§6.3), the parameter
$\sigma_\parallel$ that appears in the fidelity formula (§7.1) **is not a
fixed scalar when the integration window varies**. Module 4's Pareto
optimization (§10.6) sweeps $\tau$, so the noise must be calibrated as
a function of $\tau$ rather than as a fixed number, or the optimizer
will mis-attribute the SNR scaling and bias the frontier toward longer
integration windows.

**Convention (mandatory).** `IQNoiseParams` carries either:

- a **noise density** $\sigma_{\parallel,1/\sqrt{\text{s}}}$ (per
  $\sqrt{\text{s}}$), exposed via `sigma_parallel_per_sqrt_second: float`,
  with
  $$\sigma_\parallel(\tau_\text{integration}) = \sigma_{\parallel,1/\sqrt{\text{s}}}\, \sqrt{\tau_\text{integration}} \qquad \text{[Square-root scaling of integrated noise]} \tag{6.6a}$$
  This is the **single source of truth**; or, equivalently,
- a calibration pair $(\sigma_{\parallel,\text{ref}},\, \tau_\text{ref})$
  exposed via the dataclass fields `sigma_parallel_ref: float` and
  `tau_ref: float`, with
  $\sigma_\parallel(\tau_\text{integration}) = \sigma_{\parallel,\text{ref}}\, \sqrt{\tau_\text{integration} / \tau_\text{ref}}$,
  related to the density by
  $\sigma_{\parallel,1/\sqrt{\text{s}}} = \sigma_{\parallel,\text{ref}}/\sqrt{\tau_\text{ref}}$.

The dataclass exposes both as derived properties; the calibration pair
is an experimentalist-friendly entry point (calibrate noise at a
specific window, the dataclass converts internally).

**Implementation rule.** `compute_assignment_fidelity()` must call
`IQNoiseParams.sigma_for_integration_window(tau_integration)` to get the
noise level **at the integration window length actually used in this
fidelity evaluation**, *not* read a stale `sigma_parallel` scalar. A unit
test (`test_integrated_noise_scales_as_sqrt_tau`) prepares two
`IQNoiseParams` instances with the same density, evaluates fidelity at
$\tau$ and $4\tau$, and asserts that the implied $\sigma_\parallel$ ratio
is $\sqrt 4 = 2$ within float precision. A second test
(`test_pareto_uses_tau_dependent_noise`) constructs a Pareto sweep across
$\tau \in \{200, 500, 1000\}\,\text{ns}$ and asserts that the optimizer's
internal $\sigma_\parallel$ varies between calls (and matches (6.6a)),
catching the failure mode where the optimizer holds a fixed scalar while
sweeping $\tau$.

> **[Convention on readout efficiency].** Unless otherwise stated, Stage 06
> absorbs all readout-chain inefficiency ($\eta < 1$, finite
> $\kappa_\text{ext}/\kappa$, amp-chain noise above the vacuum floor) into
> the **noise density $\sigma_{\parallel,1/\sqrt{\text{s}}}$** (or equivalently
> the calibrated $\sigma_\parallel(\tau)$ above). Analytic design-rule
> plots (Figure 1c dashed line, Figure 4a regime-map Purcell overlay, etc.)
> assume the idealized limit $\eta = 1$, $\kappa_\text{ext} = \kappa$. The
> simulation itself (`simulate_readout()`) does not carry an explicit
> $\eta$ parameter; instead, `IQNoiseParams.sigma_parallel_per_sqrt_second`
> is calibrated upstream (either from experimental data or from a target
> amplifier-chain specification) to reflect the full realistic noise floor.
> Any future extension that needs an explicit $\eta$-parameterization
> should add it to `IQNoiseParams` as `efficiency: float = 1.0` and either
> (a) **multiply** the signal centroid separation by $\sqrt\eta$, or
> equivalently (b) **inflate** $\sigma_\parallel$ by $1/\sqrt\eta$. For
> $\eta < 1$ (imperfect readout chain) the effective signal is
> attenuated; the net effect on SNR is SNR $\to \sqrt\eta\,$SNR.

At longer $\tau$, $T_1$ decay reduces the $|1\rangle$ signal and the SNR
saturates or falls. The optimal $\tau$ is where the short-$\tau$ gain
balances decoherence loss. This is **Panel 1b** in Figure 1.

> **Physical interpretation.** Equation (6.6) says the SNR² grows as
> $\eta \times$ measurement-rate $\times$ time, with measurement rate maximized
> at $\kappa = 2|\chi|$ under fixed-$\bar n$ normalization. Doubling $\chi$
> at fixed $\bar n$ roughly doubles $\Gamma_\text{meas}$ (at $\kappa \approx 2|\chi|$),
> so doubles SNR² — a $\sqrt 2$ gain in SNR, not a $2\times$ gain.

### 6.4 Script connection for §6

| Equation | Script | Function |
|---|---|---|
| (6.1) $\langle a\rangle(t)$ | `physics/readout_model.py` | `simulate_readout()`, field `a_expectation` |
| (6.3) steady-state analytic | — | pen-and-paper sanity check |
| (6.5) integrated IQ | `physics/readout_model.py` | `ReadoutResult.integrated_iq()` |
| (6.6) short-$\tau$ SNR | — | dashed line in Figure 1b |
| (6.6a) $\sigma_\parallel(\tau)$ scaling | `physics/readout_model.py` | `IQNoiseParams.sigma_for_integration_window()` |
| Panel 1a, 1b, 1c | `scripts/fig1_readout_model.py` | top-level driver |

> **[Output-field vs intracavity normalization — enforced convention].**
> Stage 06 commits to the **output-field normalization** of (6.5):
> `ReadoutResult.integrated_iq()` returns
> $\sqrt{\kappa_\text{ext}} \int \langle a\rangle_q(t)\, dt$, with
> $\kappa_\text{ext} = \kappa$ in the default (single-port-extracted)
> idealization. `IQNoiseParams.sigma_parallel_per_sqrt_second` must be
> calibrated in the *same* output-field units; mixing intracavity centroids
> ($\int \langle a\rangle_q\, dt$ with no $\sqrt{\kappa_\text{ext}}$ factor)
> with $\sigma_\parallel$ defined in output-field units would introduce a
> silent $\sqrt{\kappa}$ scaling bug. The `ReadoutResult` dataclass
> declares its convention explicitly via the field
> `iq_normalization: Literal["output_field", "intracavity"] = "output_field"`;
> downstream scoring routines assert the convention matches before
> computing centroid separation. If a future extension needs intracavity
> normalization (e.g., for direct comparison against a transmission-line
> simulation), the `iq_normalization` flag flips to `"intracavity"` and
> $\sigma_\parallel$ must be recalibrated in the same units.


## 7. Assignment Fidelity: From Trajectory to Number

This section turns the continuous IQ trajectory into the single number the
entire project optimizes: the assignment fidelity $F_\text{assign}$, the
probability that a single-shot measurement correctly identifies the qubit
state.

### 7.1 Single-shot measurement model

A single shot consists of: (i) the qubit being prepared in $|0\rangle$ or
$|1\rangle$; (ii) the readout pulse applied, producing a trajectory
$\langle a\rangle_q(t)$ in simulation; (iii) a random IQ outcome drawn from
the noisy IQ distribution around the trajectory; (iv) a discriminator applied
to that outcome to produce a $\{0, 1\}$ estimate.

We model the IQ noise as additive **circular Gaussian** in the integrated-IQ
plane:

$$\text{IQ}_q^{(n)} = \bigl(I_q + iQ_q\bigr) + \xi^{(n)}, \qquad \xi^{(n)} = \xi_I^{(n)} + i\xi_Q^{(n)}, \quad \xi_I, \xi_Q \sim \mathcal{N}\!\bigl(0,\; \sigma_\parallel(\tau_\text{integration})^2\bigr) \text{ i.i.d.} \qquad \text{[Assumption]} \tag{7.1}$$

> **[Convention — 1D projected standard deviation, evaluated at the actual integration window].**
> We define $\sigma_\parallel$ as the **one-dimensional** (per-quadrature)
> standard deviation of the integrated IQ noise: Var$(\xi_I) =$ Var$(\xi_Q) = \sigma_\parallel^2$,
> so the total complex variance is $\mathbb{E}|\xi|^2 = 2\sigma_\parallel^2$.
> All fidelity formulas below use
> $\sigma_\parallel = \sigma_\parallel(\tau_\text{integration})$ from (6.6a)
> — the standard deviation **along the discriminator axis**
> (centroid-to-centroid direction) at the *actual* integration window
> length. Two consequences:
> (a) using "total complex variance" where $\sigma_\parallel$ is expected
> introduces a $\sqrt 2$ error in fidelity;
> (b) using a fixed $\sigma_\parallel$ across a Pareto sweep over $\tau$
> makes the SNR scale as $\tau$ rather than $\sqrt\tau$ (§6.3a) and
> biases the frontier toward longer windows.
> `compute_assignment_fidelity()` documents this convention and tests it
> against an analytic Gaussian-overlap ground truth.

where the variance $\sigma_\parallel^2 = \sigma_\parallel(\tau_\text{integration})^2$
captures: (a) quantum vacuum noise of the resonator output field (fundamental
lower bound, set by the zero-point fluctuations of the bath), (b) amplifier
noise (TWPA + HEMT gain chain; typically dominant but treated as a single
Gaussian in our model), (c) any additional Gaussian noise in the mixer/ADC
chain.

> **[Assumption].** The circular-Gaussian model is accurate for an
> amplification chain dominated by phase-insensitive gain (HEMT, standard
> TWPA). Phase-sensitive amplification (squeezed light) would break the
> circular symmetry and require a non-circular covariance matrix. This is
> out of scope for Stage 06 — Bengtsson 2024 and Marxer 2508.16437 both use
> standard phase-insensitive chains.

> **[Approximation — centroid model under $T_1$ jumps; see §6.2 warning].**
> The centroid + circular-Gaussian model treats decoherence as a
> *displacement* of the conditional centroid $(I_q, Q_q)$ rather than as
> stochastic single-shot trajectory branching. For $\tau/T_1 \gtrsim 1\%$
> this can underestimate or misrepresent the assignment-distribution
> tails. Stage 06's baseline uses the centroid model with the §6.2
> warning propagated; if absolute 99.9 % fidelity is being claimed at the
> reference operating point, either the one-jump mixture model (§7.3a)
> or the trajectory cross-check (Module 1 V7 (jump-tail cross-check)) is required as a
> validation step.

### 7.2 The linear discriminator

Given the IQ outcomes from many shots, the two distributions for
$|0\rangle$-prepared and $|1\rangle$-prepared shots cluster around the
centroids $(I_0, Q_0)$ and $(I_1, Q_1)$. The optimal Bayes-risk discriminator
for equal priors and equal-variance circular Gaussians is the
**perpendicular bisector** of the line segment connecting the two centroids:

$$\text{estimate } |1\rangle \iff |\text{IQ}^{(n)} - (I_1 + iQ_1)| < |\text{IQ}^{(n)} - (I_0 + iQ_0)| \qquad \text{[Exact for equal-variance symmetric Gaussians]} \tag{7.2}$$

The probability of misassignment given that the true state is $|0\rangle$ is
(using the 1D projected $\sigma_\parallel$ defined in (7.1)):

$$P(1|0) = \Phi\!\left(-\frac{s}{2\sigma_\parallel}\right) = \tfrac{1}{2} \mathrm{erfc}\!\left(\frac{s}{2\sqrt{2}\,\sigma_\parallel}\right) \qquad \text{[Exact, given 7.1]} \tag{7.3}$$

and similarly for $P(0|1) = \Phi(-s/(2\sigma_\parallel))$, where $\Phi$ is
the standard normal CDF. The assignment fidelity is

$$F_\text{assign} \equiv 1 - \tfrac{1}{2}[P(1|0) + P(0|1)] \qquad \text{[Definition]} \tag{7.4}$$

which at the analytic limit (7.3) gives

$$F_\text{assign}^\text{analytic} = 1 - \Phi\!\left(-\tfrac{\mathrm{SNR}}{2}\right) \qquad \text{[Approximation, equal-variance Gaussian limit]} \tag{7.5}$$

with $\mathrm{SNR} = s/\sigma_\parallel$, consistent with the definition in (6.6).

### 7.3 Why perpendicular-bisector and not Fisher/LDA

For *equal-variance* circular Gaussians, the perpendicular bisector is the
optimal (Bayes-risk-minimizing) linear discriminator; adding a weighting from
Fisher LDA or quadratic discriminant analysis (QDA) would not help. In
practice, experimental IQ distributions have slightly *unequal* variances
(for instance because one state decays during integration, broadening its
distribution), and a Fisher-weighted discriminator can squeeze out an extra
$\sim 0.1\%$ of fidelity. This extra optimization is out of scope for
Stage 06 — we use the perpendicular bisector because (a) 90 % of readout
papers use it, including Bengtsson 2024 and Marxer 2508.16437, so our
numbers are directly comparable, and (b) the Module 2 error-budget
decomposition is cleaner when the discriminator is held fixed.

### 7.3a One-jump $T_1$ mixture model (optional refinement, scoped add-on)

When $\tau/T_1 \gtrsim 1\%$, the centroid+Gaussian model of §7.1 misses the
non-Gaussian structure introduced by stochastic $T_1$ jumps during
integration. A scoped refinement that does **not** require quantum
trajectories: model each $|1\rangle$-prepared shot as a mixture of two
sub-populations:

$$p_\text{IQ|1}(z) = e^{-\tau/T_1}\, \mathcal{N}(z;\, c_1,\, \sigma_\parallel^2 \mathbb{I}) + \int_0^\tau \frac{dt^*}{T_1}\, e^{-t^*/T_1}\, \mathcal{N}(z;\, c_{1,\text{jump}}(t^*),\, \sigma_\parallel^2 \mathbb{I}) \qquad \text{[One-jump mixture; assumes one-jump-per-shot dominates for $\tau/T_1 \ll 1$]} \tag{7.5a}$$

where $c_1$ is the no-jump centroid (i.e. the trajectory if the qubit
stayed in $|1\rangle$ for the whole window) and $c_{1,\text{jump}}(t^*)$
is the integrated IQ point of a trajectory that started in $|1\rangle$ and
jumped to $|0\rangle$ at time $t^*$.

> **[Cavity memory at the jump time — what the post-jump segment is, and
> what it is NOT].** After the qubit jumps from $|1\rangle$ to $|0\rangle$
> at time $t^*$, the cavity field is **already displaced near the
> $|1\rangle$ pointer state** (it has been ringing up under the
> $|1\rangle$-conditioned dispersive dynamics for time $t^*$). The
> post-jump cavity evolution is therefore the $|0\rangle$-conditioned
> dispersive dynamics with **initial condition equal to the cavity
> amplitude present at the jump time**, *not* the unconditional
> $|0\rangle$ trajectory restarted from vacuum. Letting
> $\alpha_1^{\text{no-jump}}(t) \equiv \langle a\rangle_1(t)$ denote the
> deterministic-ME no-jump cavity trajectory, the correct construction is
>
> $$c_{1,\text{jump}}(t^*) = \sqrt{\kappa_\text{ext}}\,\Bigl[\int_0^{t^*}\alpha_1^{\text{no-jump}}(t)\,dt + \int_{t^*}^\tau \alpha_{1\to 0}(t;\, t^*)\,dt\Bigr] \qquad \text{[7.5b — corrected post-jump construction]}$$
>
> where $\alpha_{1\to 0}(t;\, t^*)$ is the post-jump cavity amplitude
> evolved under the $|0\rangle$-conditioned dispersive dynamics with
> initial condition
>
> $$\alpha_{1\to 0}(t^*;\, t^*) = \alpha_1^{\text{no-jump}}(t^*).$$
>
> In the dispersive linear-cavity approximation $\alpha_{1\to 0}(t;\, t^*)$
> can be obtained from the analytic Green's function of the $|0\rangle$
> cavity equation (6.2), or by solving the scalar cavity ODE numerically
> for each $t^*$ on the jump-time grid. In the full JC model it requires
> reinitializing the *conditional* density matrix at the jump time and
> evolving forward; the one-jump mixture is therefore a **controlled
> approximation, not an exact replacement for quantum trajectories**.
> The earlier shorthand "$\int_{t^*}^\tau \langle a\rangle_0(t-t^*)\,dt$"
> would erase the cavity memory at $t^*$ and silently bias
> $c_{1,\text{jump}}$ toward the unconditional-$|0\rangle$ pointer; that
> shorthand is wrong and is replaced by (7.5b) above.

Both $c_1$ and $c_{1,\text{jump}}(t^*)$ are precomputable from a *small*
set of deterministic-ME runs per device: one $|1\rangle$-prepared run to
get $\alpha_1^{\text{no-jump}}(t)$, plus one $|0\rangle$-conditioned
"restart" simulation per $t^*$ grid point (or one analytic-Green's-function
evaluation per $t^*$ in the dispersive linear-cavity limit). Discriminator
outcomes are then averaged over the jump-time distribution. This
refinement captures the dominant non-Gaussian tail at $\tau/T_1 \le 5\%$
without quantum trajectories, with the cavity-memory correction (7.5b)
preserving the displaced initial condition that the naive form would
have erased.

**Status.** §7.3a is an **optional refinement** to the §7.1 baseline,
implemented in `physics/readout_model.py::compute_fidelity_one_jump_mixture()`
and gated on the config field `IQNoiseParams.use_one_jump_mixture: bool = False`.
The default is `False` (centroid+Gaussian, §7.1); turning it on is
recommended whenever a fidelity claim is being made at $\tau/T_1 \ge 1\%$.
The mixture form (7.5a) assumes one jump per shot dominates; for higher
$\tau/T_1$, multi-jump terms or a trajectory simulator (Module 1 V7)
should be used instead.

### 7.4 Bootstrapping shot-noise uncertainty

The fidelity estimate from finite $n_\text{shots}$ has binomial shot-noise
uncertainty

$$\sigma_{F,\text{shot}} \approx \sqrt{\frac{F(1-F)}{n_\text{shots}}} \qquad \text{[Approximation, binomial]} \tag{7.6}$$

> **[Two-binomial estimator for balanced shots].** For balanced shots
> $n_0 = n_1 = n_\text{shots}/2$ over two preparations, the more
> explicit estimator of $\text{Var}(F)$ is the two-binomial form:
> $$\text{Var}(F) = \frac{1}{4}\left[\frac{p_{0,\text{corr}}(1-p_{0,\text{corr}})}{n_0} + \frac{p_{1,\text{corr}}(1-p_{1,\text{corr}})}{n_1}\right],$$
> where $p_{s,\text{corr}}$ is the per-preparation correct-assignment
> probability. Equation (7.6) is used only as a **compact scaling
> estimate**; the two-binomial form should be used when reporting
> uncertainties for actual fidelity estimates.

We additionally bootstrap over independent samples of the noise process
to capture the non-binomial component (e.g. uncertainty in the discriminator
position due to finite-sample drift in the centroids). The default for
**final/reportable runs** is `n_bootstrap = 500` (sufficient for stable
confidence intervals on the order of 1% of $F$). For fast CI smoke tests
where bootstrap noise is dominated by the binomial term anyway,
`n_bootstrap = 20` is acceptable and is the default for `pytest -m fast`.
Module 2's spec must distinguish these two modes via a config flag
(e.g. `BootstrapConfig(mode: Literal["fast_ci", "final"])`). This is the
uncertainty reported in the `AssignmentFidelityResult.F_assign_uncertainty`
field.

### 7.5 Script connection for §7

| Equation | Script | Function |
|---|---|---|
| (7.1) noise model with $\sigma_\parallel(\tau)$ | `physics/readout_model.py` | `compute_assignment_fidelity()` — calls `IQNoiseParams.sigma_for_integration_window()` |
| (7.2) perpendicular-bisector | `physics/readout_model.py` | discriminator inside `compute_assignment_fidelity()` |
| (7.4) $F_\text{assign}$ | `physics/readout_model.py` | return value of `compute_assignment_fidelity()` |
| (7.5) analytic limit | — | cross-check on Figure 1c |
| (7.5a) one-jump mixture + (7.5b) post-jump cavity (optional) | `physics/readout_model.py` | `compute_fidelity_one_jump_mixture()` |
| (7.6) shot-noise bootstrap | `physics/readout_model.py` | `F_assign_uncertainty` field |


## 8. Coherent and Incoherent Error Budget (Module 2)

This section derives the decomposition methodology for Module 2: given a full
simulation with all decoherence channels active, how do we attribute the
total infidelity to named physical contributions?

### 8.1 The decomposition problem — core vs diagnostics

Total assignment infidelity at a fixed operating point (device, drive,
integration window) is

$$\mathcal{I}_\text{total} \equiv 1 - F_\text{full} \qquad \text{[Definition]} \tag{8.1}$$

where $F_\text{full}$ is the simulated fidelity with all channels active.
The decomposition is split into **two conceptually distinct parts**:

**(A) Additive core budget.** Using **Group A only by default** (true
removable Lindblad channels: $T_1$, $\gamma_\varphi$, thermal). Group B
(dispersive-approximation model comparison) is **a signed
model-comparison diagnostic** by default and is **excluded** from the
core-budget closure unless the caller explicitly opts in via
`include_model_comparison_in_core=True` on the `ErrorBudget` constructor.
Purcell is **not** in Group A: it is already implicit in $F_\text{ideal}$
as a coherent leakage of qubit excitation into the cavity-decay channel,
and is reported separately as a Group A′ analytic diagnostic (§8.3, §8.5)
outside the closure sum.

$$\mathcal{I}_\text{total} \approx (1 - F_\text{ideal}) + \sum_{c \in \text{Group A}} \Delta F_c + R_\text{core} \qquad \text{[Additive decomposition over removable channels, default]} \tag{8.2}$$

When `include_model_comparison_in_core=True` is set explicitly, the sum
extends to $c \in \text{Group A} \cup \text{B}$; this is an opt-in
behavior, not the default.

where $F_\text{ideal}$ is the ceiling fidelity with all Group A removable
Lindblad channels turned off while retaining the **full rotating-frame JC
Hamiltonian** (no dispersive substitution; $g$ and $\kappa$ still active).
$\Delta F_c$ is the marginal contribution of channel $c$ (§8.2), and
$R_\text{core}$ is the residual — everything in $\mathcal{I}_\text{total}$
not captured by the ideal-limit floor plus the marginal Group-A
contributions. If `include_model_comparison_in_core=True`, the
Group-B model-comparison term is included separately with its signed
value; this opt-in does **not** change the default definition of
$F_\text{ideal}$. An implementation must not silently switch the ideal
run to the dispersive-frame Hamiltonian, even when the Group-B opt-in
is active — the dispersive-vs-JC comparison is a Group-B diagnostic,
not a redefinition of the ideal floor.

> **[What $(1-F_\text{ideal})$ contains, and what $R_\text{core}$ contains].**
> $(1-F_\text{ideal})$ is computed from the full JC simulator with all
> Group A collapse operators set to zero (no $T_1$, no $\gamma_\varphi$,
> no $\bar n_\text{th}$) but with $g$ and $\kappa$ still active. It
> therefore already contains **all non-removable effects left active in
> the ideal run**, including: (a) JC leakage out of the computational
> $\{|0\rangle, |1\rangle\}$ subspace during the dynamics; (b) Purcell
> loss arising from $g$-$\kappa$ hybridization (since $g$ and $\kappa$
> are still on); (c) finite measurement-overlap $F_\text{discrim}^\text{ideal}$
> at the operating point; (d) any other model floor of the ideal
> simulator.
>
> $R_\text{core}$ is the residual *after* subtracting both the ideal
> floor and the marginal Group A contributions:
>
> $$R_\text{core} = \mathcal{I}_\text{total} - (1-F_\text{ideal}) - \sum_{c \in \text{Group A}} \Delta F_c.$$
>
> It primarily measures the **nonadditivity** between Group A channels
> (cross-channel interactions: how $T_1$, $\gamma_\varphi$, and thermal
> contributions interfere when activated together vs. their sum of
> marginal contributions when activated individually). It will also
> absorb any *interaction* between Group A channels and effects
> implicit in $F_\text{ideal}$. The bare Purcell loss itself is in
> $(1-F_\text{ideal})$, not in $R_\text{core}$. The Group D leakage
> diagnostic ($P_\text{leak}$, §8.4) is a separate scalar and
> not part of this fidelity-scale decomposition.

**(B) Diagnostic panel (not additive).** Group C (calibration stressors:
drive amplitude, detuning) and Group D (post-processing diagnostic: leakage)
produce **informational bars** plotted beside the waterfall but **excluded
from (8.2)**. They do not close the additivity check, because they are
not removable dynamical channels:

$$\{\Delta F_\text{amp-cal},\; \Delta F_\text{det-cal},\; P_\text{leak}\} \qquad \text{[Diagnostic bars — not summed into (8.2); $P_\text{leak}$ is unitless probability, not fidelity-scale]} \tag{8.2a}$$

Equation (8.2) is an **ansatz**, not a theorem. It will not hold exactly when
Group A channels interact nonlinearly (constructive or destructive
interference between $T_1$ and $\gamma_\varphi$, for example), and the
whole point of reporting $R_\text{core}$ explicitly is to quantify how
well the ansatz captures reality within Group A.

### 8.2 Marginal attribution

The marginal contribution of channel $c$ is defined as

$$\Delta F_c \equiv F_{c \text{ off}} - F_\text{full} \qquad \text{[Definition]} \tag{8.3}$$

where $F_{c \text{ off}}$ is the fidelity with channel $c$ turned off while
*all other channels remain active*. This is the "if I fix channel $c$ and
leave everything else broken, how much fidelity do I gain" quantity.

**Why marginal and not sequential.** The alternative is *sequential* turn-off:
$\Delta F_c^\text{seq} = F(\text{channels } 1..c \text{ off}) - F(\text{channels } 1..c-1 \text{ off})$,
which produces channel-ordering-dependent attributions and is not robust to
reordering. Marginal attribution is symmetric and order-independent.

> **[Spec reconciliation].** Module 2 spec prose says "canonical waterfall
> construction is sequential turn-off," but the spec's own formal definition
> $\Delta F_c = F_{c,\text{off}} - F_\text{full}$ is the marginal form we use
> here and in this framework. The two are not equivalent. This framework
> resolves the apparent inconsistency by treating the formal definition as
> authoritative: the waterfall visualization in Figure 2 is built from
> *marginal* contributions, even though the bar-chart layout may
> superficially resemble a sequential-drawdown plot.

In the literature this is analogous to a first-order Möbius / inclusion-exclusion
decomposition of the fidelity loss; the exact (all-orders) version would track
every subset of channels (Shapley values, in the game-theory analogy), which
has $2^N$ cost and is overkill for our 8 channels. The marginal + residual
approach is cheap (linear in the number of channels — concretely, 1 full simulation + 1 ideal + $N$ channel-off runs + diagnostic reruns) and reports the deviation from
additivity honestly.

### 8.3 The eight channels, grouped by category

The eight items we decompose infidelity into are not all the same kind of
thing. Grouping them honestly (and dropping the historical ordinal
numbering, which was non-monotonic across groups and visually confusing):

**Group A — Removable Lindblad channels (true turn-offs).** Setting a
collapse-operator rate to zero cleanly eliminates the underlying Lindblad
process from the dynamics.

- **$T_1$ decay.** Turn-off: set $\gamma_1 = 0$ in `DecoherenceParams`.
- **Pure dephasing.** Turn-off: set $\gamma_\varphi = 0$.
- **Thermal photons.** Turn-off: set $\bar n_\text{th} = 0$.

**Group A′ — Analytic dynamical diagnostic (already implicit in $F_\text{ideal}$).**
Purcell decay is **not** a separate Lindblad channel and **not** a Group A
turn-off. Because $F_\text{ideal}$ in (8.2) is computed with $g$ and
$\kappa$ still active, **Purcell is already implicit in $1 - F_\text{ideal}$**.
Adding a $\Delta F_\text{Purcell}$ bar to the additive closure (8.2) would
double-count.

- **Purcell (analytic dynamical diagnostic).** Reported separately
  alongside the waterfall as a sub-attribution of effective $T_1$ from
  $\gamma_\text{Purcell} = (g_{01}/\Delta)^2 \kappa$ (eq. 5.9). **Excluded
  from the B1 additivity closure** (§8.5).

**Group B — Model-comparison channel.** Not a dynamics turn-off; a
different Hamiltonian entirely. Group B uses a **signed** difference
that does not necessarily satisfy the Group A "harmful → positive"
convention.

- **Dispersive-approximation breakdown (signed model-comparison delta).**
   Comparison: swap `build_hamiltonian(frame="rotating")` for
   `build_hamiltonian(frame="dispersive")` at otherwise-identical
   parameters. To match the marginal convention $\Delta F_c = F_{c,\text{off}} - F_\text{full}$
   (where "off" = non-dispersive effects removed = dispersive frame),
   we define:

   $$\Delta F_\text{disp-breakdown} \equiv F(\text{dispersive}) - F(\text{rotating}) \qquad \text{[Signed model-comparison]}$$

   Reported with explicit sign in `diagnostic_bars["dispersive_breakdown"]`.

**Group C — Calibration stressors (perturbation diagnostics, not turn-offs).**

- **Drive amplitude miscalibration.** Stressor: perturb
   $\varepsilon_0 \to \varepsilon_0 (1 \pm 5\%)$ independently; average
   the two fidelity losses.
- **Drive detuning error.** Stressor: perturb
   $\omega_d \to \omega_d \pm \kappa/4$, averaged.

**Group D — Post-processing diagnostic.**

- **Leakage to** $|2\rangle$. **Not a channel turn-off.** Reported in the
   waterfall as a separate bar with explicit "post-processing diagnostic"
   annotation.

> **[Framing note for Figure 2 waterfall].** The waterfall bar chart
> mixes four epistemically different operations in one display. The
> caption should say so explicitly — e.g. "Groups A (blue), B (green),
> C (orange), D (gray) represent turn-offs, model comparison, calibration
> stressors, and post-processing diagnostics, respectively."

### 8.4 The leakage-to-$|2\rangle$ attribution (post-processing heuristic)

Measurement-induced transmon ionization (Shillito et al. 2022, Dumas et al.
2024) is the phenomenon where a high-power readout drive, via multiphoton
resonances at specific resonator photon populations, can drive the transmon
out of the computational subspace.

The leakage "channel" in our budget is **not** a collapse operator and
**not** a parameter turn-off. It is a **diagnostic post-processing
decomposition device** applied to the simulation output.

**Precise definition (8.4 operationalized).** Let $\rho_\text{full}(\tau)$
be the full simulated density matrix on
$\mathcal H_\text{transmon} \otimes \mathcal H_\text{resonator}$ at the end
of the integration window, and let
$\rho_q^\text{transmon}(\tau) = \operatorname{Tr}_\text{resonator} \rho_\text{full}(\tau)$
be its reduced transmon state. Define the computational-subspace projector
$\Pi_{01} = |0\rangle\langle 0| + |1\rangle\langle 1|$ and the leaked-weight
scalar $w_{01,q} = \operatorname{Tr}[\Pi_{01} \rho_q^\text{transmon}(\tau)]$.

The Stage 06 leakage diagnostic is the **scalar leakage probability**:

$$P_\text{leak} \equiv 1 - \tfrac{1}{2}(w_{01,0} + w_{01,1}) \qquad \text{[Pure diagnostic, not a fidelity-scale loss]} \tag{8.4}$$

reported as `diagnostic_bars["leakage_probability"]` in `ErrorBudget`.
This is a Group D post-processing diagnostic and is **explicitly not
fidelity-scale comparable** with $T_1$ or $\gamma_\varphi$ contributions.

> **[Why no conditional-fidelity $F_\text{proj}$ in Stage 06].** A
> conditional fidelity of the form
> $F_\text{proj} = \sum_q p(q) w_{01,q} P(\text{correct}\mid q, \text{not leaked}) / \sum_q p(q) w_{01,q}$
> would be over-specified relative to the available simulator outputs.
> A correct conditional-fidelity calculation would require either: (a)
> explicit per-level centroid simulation $|0\rangle, |1\rangle, |2\rangle$
> with a mixture-model assumption on the discriminator distribution, or
> (b) quantum trajectories with explicit per-shot conditional measurement
> records. Both are out of scope for Stage 06.

### 8.5 The Purcell isolation analytic

The Purcell channel requires care because it is not a separate collapse
operator (§5.3). We isolate it as follows: given the fitted/specified
device parameters $(g, \Delta, \kappa)$, compute the analytic Purcell rate

$$\gamma_\text{Purcell}^\text{analytic} = \left(\frac{g_{01}}{\Delta}\right)^2 \kappa \qquad \text{[Approximation, 2-level leading order]} \tag{8.5}$$

For the multi-level transmon, the correct expression uses the matrix element
$|\langle 0|\hat n|1\rangle|^2$ explicitly:

$$\gamma_\text{Purcell}^\text{multi-level} = \left(\frac{g\, |\langle 0|\hat n|1\rangle|}{\Delta_{01}}\right)^2 \kappa \qquad \text{[Approximation, dispersive, multi-level]} \tag{8.6}$$

The Purcell **contribution to the fidelity budget** is reported as a
**Group A′ analytic dynamical diagnostic** (not added to the B1 closure;
see §8.1). We attribute fidelity cost to Purcell by imagining the qubit
decays at the additional rate $\gamma_\text{Purcell}$ for the readout
duration $\tau$:

$$\Delta F_\text{Purcell}^\text{diag} \approx \gamma_\text{Purcell}\, \tau \cdot \frac{\partial \mathcal{I}}{\partial (\gamma_1 \tau)}\bigg|_\text{ref} \qquad \text{[Approximation, linear attribution]} \tag{8.7}$$

The $\partial \mathcal{I}/\partial(\gamma_1 \tau)$ slope is obtained from
a single $\gamma_1$-perturbation run of the full JC simulator (which does
not add Purcell a second time, because $\gamma_1$ is set to zero in the
"ideal" reference run and set to a perturbed value $\gamma_1^\text{test}$
in the perturbation run — both with Purcell automatically present via
$g$-$\kappa$). Reported by `analytic_purcell_rate()` in
`analysis/purcell_isolation.py`.

> **[Diagnostic, not bar in the closure sum].** $\Delta F_\text{Purcell}^\text{diag}$
> is a sub-attribution of effective $T_1$ at the operating point. It is
> **not summed** with $\Delta F_{T_1}, \Delta F_{\gamma_\varphi}, \Delta F_{\bar n_\text{th}}$
> in (8.2), because Purcell is already implicit in $F_\text{ideal}$ (§8.1)
> and adding it would double-count.

> **[Do not add $\gamma_\text{Purcell}$ to $\gamma_1$ in the full JC
> simulator].** Adding $\gamma_\text{Purcell}$ as a separate collapse
> operator on top of the JC dynamics — i.e. simulating with
> $\gamma_1 \to \gamma_1 + \gamma_\text{Purcell}$ — would double-count
> Purcell.

> **[Why synthetic $\kappa$-split isolation is not used].** It is tempting
> to split $\kappa$ into two collapse operators — one external
> ($\kappa_\text{readout}$, measurable) and one "internal" ($\kappa_\text{qubit}$,
> ostensibly responsible for Purcell) — then compare $F(\kappa_\text{qubit} = \kappa)$
> against $F(\kappa_\text{qubit} = \kappa/100)$. **This does not work in a
> single-mode Lindblad model.** Both collapse operators are proportional to
> $a$, and the GKSL dissipator is quadratic in the operator:
>
> $$\mathcal{D}[\sqrt{\kappa_\text{readout}}\,a] + \mathcal{D}[\sqrt{\kappa_\text{qubit}}\,a] = \mathcal{D}[\sqrt{\kappa_\text{readout} + \kappa_\text{qubit}}\,a]$$
>
> so splitting $\kappa$ into two channels simply renames the total cavity
> linewidth. A genuine Purcell-isolation simulator would require a
> frequency-dependent bath with distinct $\kappa(\omega_r)$ and $\kappa(\omega_q)$
> (i.e., an explicit Purcell-filter transfer function), out of scope for
> Stage 06 (§5.3; see Hazra et al. 2025 for the relevant frequency-dependent
> modeling). Module 2 therefore uses **analytic attribution (8.7) as the
> sole Purcell isolation method**.

### 8.6 Script connection for §8

| Equation / concept | Script | Function |
|---|---|---|
| (8.2) decomposition ansatz | `analysis/error_budget.py` | `compute_full_error_budget()` |
| (8.3) marginal attribution | `analysis/error_budget.py` | `compute_channel_contribution()` |
| Channel definitions | `analysis/error_budget.py` | `ChannelName` literal + turn-off logic |
| (8.4) leakage diagnostic | `analysis/error_budget.py` | `leakage_probability()` |
| (8.5), (8.6) Purcell analytic | `analysis/purcell_isolation.py` | `analytic_purcell_rate()` |
| Residual $R_\text{core}$ (test B2) | `analysis/error_budget.py` | `ErrorBudget.residual_core` field |
| Diagnostic panel (Groups C, D) | `analysis/error_budget.py` | `ErrorBudget.diagnostic_bars: dict[str, float]` + `included_in_core_sum: dict[str, bool]` |
| Validation: $\sum + R \approx \mathcal{I}_\text{total}$ | `tests/test_error_budget.py::test_budget_sums_to_full_infidelity_within_tolerance` | B1 |


## 9. Characterization Theory (Module 3)

This section derives the fitting forms for the four characterization protocols
and states what each protocol's data actually tells you about the underlying
density-matrix dynamics.

### 9.1 Rabi oscillation

The Rabi protocol: prepare $|0\rangle$, drive on resonance with $\omega_{01}$
at amplitude $\varepsilon$ for a fixed duration $\tau_\text{pulse}$, measure
$P_1$, repeat with $\varepsilon$ swept.

**Derivation of the fit form.** In the rotating frame at $\omega_d = \omega_{01}$,
the 2-level qubit Hamiltonian under a Gaussian-enveloped drive is (to leading
order in the RWA, ignoring the resonator):

$$H_\text{q,drive} = \tfrac{1}{2}\Omega_R(\varepsilon)\, \sigma_x \qquad \text{[Approximation, 2-level + RWA]} \tag{9.1}$$

with Rabi frequency $\Omega_R(\varepsilon) \propto \varepsilon$. For a
pulse of duration $\tau_\text{pulse}$, the final state has

$$P_1 = \sin^2\!\left(\tfrac{1}{2}\Omega_R\tau_\text{pulse}\right) = \tfrac{1}{2}\bigl[1 - \cos(\Omega_R\tau_\text{pulse})\bigr] \qquad \text{[Exact, unitary 2-level]} \tag{9.2}$$

In the presence of imperfect-pulse-shape contrast loss and a small initial-state
offset (thermal $P_1^\text{th} > 0$), the observed curve is

$$P_1(\varepsilon) = A + B\,\cos\!\bigl(\Omega_R(\varepsilon)\,\tau_\text{pulse} + \phi\bigr) \qquad \text{[Approximation]} \tag{9.3}$$

with $\Omega_R = g_\text{drive}\,\varepsilon$ linear in the drive amplitude,
$A \approx 1/2$ the offset, $B \approx 1/2$ the contrast absorbing pulse-shape
imperfections at fixed $\tau_\text{pulse}$, and $\phi$ a small phase offset
from calibration drift. The fit extracts $\varepsilon_\pi$, the amplitude at
which $\Omega_R \tau_\text{pulse} = \pi$ — i.e. the calibrated $\pi$-pulse
amplitude.

> **[Identifiability note].** Since the Rabi protocol sweeps **drive amplitude
> $\varepsilon$ at fixed pulse duration $\tau_\text{pulse}$**, any factor of
> the form $e^{-\tau_\text{pulse}/T_R}$ is **constant across the sweep** and
> is absorbed into the contrast $B$. We therefore do **not** include a
> separate $T_R$ envelope parameter in the fit — doing so would produce a
> redundant, unidentifiable parameter and would degrade the coverage of the
> recovery harness (C1a). The `fit_rabi()` implementation documents this
> and fits only $(A, B, g_\text{drive}, \phi)$. Amplitude-sweep Rabi
> experiments identify the $\pi$-pulse amplitude and the linear drive-coupling
> slope, not a coherence time.

### 9.2 Ramsey interferometry

The Ramsey protocol: prepare $|0\rangle$, apply $\pi/2$ pulse around $\hat x$
(takes $|0\rangle \to (|0\rangle + |1\rangle)/\sqrt 2$), wait time $\tau$, apply
another $\pi/2$ pulse, measure $P_1$. Sweep $\tau$.

Under the Markov-exponential assumption:

$$P_1(\tau) = A + B\, e^{-\tau/T_2^*}\, \cos(\Delta\omega\,\tau + \phi) \qquad \text{[Approximation, Markov noise]} \tag{9.4}$$

Under $1/f$-dominated dephasing the decay is Gaussian:

$$P_1(\tau)\bigr|_{1/f} = A + B\, e^{-(\tau/T_2^*)^2}\, \cos(\Delta\omega\,\tau + \phi) \qquad \text{[Approximation, $1/f$-dominated]} \tag{9.5}$$

**For Module 3 we adopt the following defaults.** For Ramsey, Module 3
defaults to the exponential envelope (9.4) and optionally fits the
Gaussian envelope (9.5). For Hahn echo, the fallback fit is the
stretched-exponential form (9.7) below with $n$ free. The difference
between the exponential and Gaussian Ramsey fits is diagnostic of the
noise spectrum: a Hahn-echo experiment that fits well with $n=1$
(exponential) indicates white-noise dephasing; one that requires $n=2$
(Gaussian) indicates $1/f$-dominated dephasing.

### 9.3 $T_1$ decay

Simplest fit form:

$$P_1(\tau) = A + B\, e^{-\tau/T_1} \qquad \text{[Approximation, single-exponential]} \tag{9.6}$$

with $A$ accounting for steady-state thermal population ($\bar n_\text{th}$)
and readout assignment errors, $B \approx 1$ for a good $\pi$-pulse.

**[Approximation, bias when $\bar n_\text{th}$ is large].** A naive fit to
(9.6) with $A$ clamped to zero underestimates $T_1$ by a factor of
$(1 + 2\bar n_\text{th})^{-1}$ when $\bar n_\text{th} > 0$ — the thermal
steady state looks like an apparent offset that biases the exponential.
Module 3 therefore always fits $A$ as a free parameter.

### 9.4 Hahn-echo $T_2$

A single $\pi$ pulse inserted at time $\tau/2$ in the Ramsey sequence refocuses
*low-frequency* dephasing (below the echo bandwidth $\sim 1/\tau$) while
leaving high-frequency dephasing unaffected. This suppresses $1/f$ noise,
typically yielding $T_{2,\text{echo}} \gg T_2^*$.

$$P_1(\tau) = A + B\, e^{-(\tau/T_{2,\text{echo}})^n} \qquad \text{[Approximation, Module 3 default $n=1$]} \tag{9.7}$$

with $n$ fit as a free parameter only as a fallback.

### 9.5 Bootstrap strategy under 1/f-correlated residuals

lmfit returns two uncertainty estimators: (a) the covariance-matrix
uncertainty from the Jacobian at the best-fit point (quick, but assumes
independent Gaussian residuals); (b) a bootstrap-estimated uncertainty.
Under 1/f-drifted noise, the residuals are **temporally correlated** across
the scan, and **ordinary (i.i.d.) residual bootstrap breaks that correlation
by resampling individual residuals independently** — it therefore
undercovers in exactly the situation Module 3 cares about.

**Correct bootstrap approach (Module 3 spec needs updating).** Module 3 uses
one of the following, not i.i.d. residual bootstrap:

- **Block bootstrap (preferred).** Partition the residual sequence into
  contiguous blocks of length $\ell \sim 5$–$10$ samples (chosen so the
  autocorrelation within a block is high and across blocks is low), then
  resample blocks with replacement. This preserves short-range correlation
  structure while still producing variability across resamples.
- **Parametric bootstrap (fallback).** Refit a $1/f$-drift model to the
  residuals, then generate fresh synthetic residual sequences from that
  noise process. Requires a separately-validated drift model; acceptable
  if the block length is hard to calibrate.
- **Full synthetic regeneration.** Most honest but most expensive: for each
  bootstrap iteration, regenerate a full synthetic trace (including the
  $1/f$ drift process) and rerun the fit. Matches the recovery harness
  directly; use this for the final coverage report.

Ordinary residual bootstrap is **not** used. The implementation in
`characterization/fitting.py::_bootstrap_uncertainty()` must document which
strategy it uses; the default is block bootstrap with $\ell = 8$ samples
(to be validated during implementation).

**Why coverage is the gate (G2).** Coverage is the fraction of synthetic
devices for which the true parameter value falls within the reported
$\pm n\sigma$ confidence interval. Perfect coverage is $68\%$ at $1\sigma$,
$95\%$ at $2\sigma$. Module 3 targets $\ge 60\%$ at $1\sigma$ and $\ge 90\%$
at $2\sigma$ — slightly below nominal to allow for finite-sample (50 devices)
statistical fluctuations. Coverage being too low under i.i.d. bootstrap is
one of the symptoms that motivates the block-bootstrap choice above.

### 9.6 Why lmfit and not MCMC / SBI / conformal

The choice is locked in Module 3 spec §0 as "lmfit, not MCMC/SBI". The reason:

- **MCMC** is overkill for 4-parameter fits to well-characterized exponential
  forms. It would provide marginally better uncertainties at $10\times$ the
  wall-clock cost.
- **Simulation-based inference (SBI)** is a valuable post-submission roadmap
  item (the Module 1 simulator is a ready-made generator), but it requires
  a substantial training pipeline that doesn't fit the 22-day scope.
- **Conformal prediction** would wrap existing fits with distribution-free
  coverage guarantees. Another good roadmap item but requires a calibration
  set infrastructure that is premature.

lmfit + bootstrap is the correct scoped choice: fast, interpretable, testable
by the coverage harness. The SBI/conformal extensions are named as
post-submission roadmap in the README.

### 9.7 Script connection for §9

| Equation | Script | Function |
|---|---|---|
| (9.3) Rabi fit | `characterization/fitting.py` | `fit_rabi()` |
| (9.4) Ramsey exponential | `characterization/fitting.py` | `fit_ramsey()` |
| (9.5) Ramsey Gaussian (fallback) | `characterization/fitting.py` | stretched-exponential mode |
| (9.6) $T_1$ fit | `characterization/fitting.py` | `fit_t1()` |
| (9.7) Hahn-echo | `characterization/fitting.py` | `fit_t2_echo()` |
| Bootstrap uncertainty (block) | `characterization/fitting.py` | shared by all fit functions |
| Coverage harness | `characterization/recovery.py` | `run_recovery_harness()` |


## 10. Optimization and Sensitivity (Module 4)

This section derives the sensitivity analysis, regime map, and Pareto
framework of Module 4, and explains why each analytic boundary in the
regime map is where it is.

### 10.1 Normalized logarithmic sensitivity

For a parameter $\theta$ and objective $F_\text{assign}$ close to unity,
the meaningful quantity is the **log-infidelity sensitivity**:

$$S_\theta \equiv \frac{\partial \ln \mathcal I}{\partial \ln \theta}\bigg|_\text{ref}, \qquad \mathcal I \equiv 1 - F_\text{assign} \qquad \text{[Definition]} \tag{10.1}$$

The quantity (10.1) is dimensionless and stays order unity even when
$F_\text{assign} \to 1$, which makes it directly interpretable: a
log-infidelity sensitivity $S_\theta = +1$ means that a 10 % increase in
$\theta$ produces a 10 % increase in infidelity (harmful).

> **[Why not log-fidelity].** A naive log-fidelity sensitivity
> $S_\theta^{(F)} = \partial \ln F / \partial \ln \theta$ is sometimes
> written with an erroneous $(1-F_\text{ref})^{-1}$ amplification. The
> correct expansion is $\ln F = \ln(1 - \mathcal I) \approx -\mathcal I$
> for $\mathcal I \ll 1$, giving
> $S_\theta^{(F)} \approx -\partial \mathcal I / \partial \ln \theta$ —
> no $1/(1-F)$ factor. Moreover, $S_\theta^{(F)}$ is not order-invariant
> near $F \to 1$: it goes to zero while $S_\theta$ (log-infidelity)
> stays finite. Module 4 therefore uses log-infidelity sensitivity
> (10.1) as the primary diagnostic.

### 10.2 Expected sensitivity signs

- **$|\chi|$** — $-$
  - *Rationale:* larger $|\chi|$ $\Rightarrow$ larger IQ separation $\Rightarrow$ lower $\mathcal I$
- **$\kappa$** — $\pm$
  - *Rationale:* competing: larger $\kappa$ $\Rightarrow$ faster response (helps) and larger Purcell (hurts); sign depends on operating point
- **$\gamma_1$** — $+$
  - *Rationale:* more $T_1$ decay $\Rightarrow$ more $|1\rangle$ IQ signal lost $\Rightarrow$ larger $\mathcal I$
- **$\gamma_\varphi$** — $\ge 0$, often $\approx 0$
  - *Rationale:* dispersive readout of energy eigenstates is population-based; pure dephasing damps coherences but often doesn't affect discrimination. For QND dispersive readout on prepared $|0\rangle, |1\rangle$, $S_{\gamma_\varphi}$ is typically small and can be near-zero. Nonzero contributions arise from drive-induced coherences and non-dispersive couplings.
- **$\bar n_\text{th}$ (with floor; see below)** — $+$
  - *Rationale:* thermal population reduces contrast
- **$\varepsilon_0$** — $-$ initially, $\pm$ at high power
  - *Rationale:* trades higher SNR against leakage; typically $-$ until leakage becomes significant
- **$\tau$** — $-$ initially, $\pm$ at large $\tau$
  - *Rationale:* trades shot noise reduction against $T_1$ decay; optimum at some finite $\tau$

> **[Why $|\chi|$ and not $\chi$].** $S_\theta$ is defined via $\ln\theta$,
> which is undefined for $\theta < 0$. At our reference regime
> $\chi/2\pi \approx -0.385$ MHz, so $\ln\chi$ is undefined. The physically
> meaningful sensitivity is to **the magnitude** of $\chi$ (the IQ pointer-state
> separation scale), so we report $S_{|\chi|} = \partial \ln \mathcal{I}/\partial \ln |\chi|$.
> The sign convention $S_{|\chi|} < 0$ holds for both signs of $\chi$.

> **[Convention for $\bar n_\text{th}$].** $\ln \bar n_\text{th}$ is undefined at
> $\bar n_\text{th} = 0$. Module 4 evaluates $S_{\bar n_\text{th}}$ around a
> small finite floor $\bar n_\text{th}^\text{floor} = 10^{-3}$. If a
> user-configured run sets $\bar n_\text{th} = 0$ exactly, the sensitivity
> falls back to a one-sided additive form $\partial \mathcal{I}/\partial \bar n_\text{th}$
> (not log-log), with appropriate units annotation in the report.

> **[Test O1 — sign conditions].** Under the log-infidelity convention,
> O1 requires:
>
> - $S_{|\chi|} < 0$ (strict),
> - $S_{\gamma_1} > 0$ (strict),
> - $S_{\bar n_\text{th}} > 0$ (strict, evaluated at $\bar n_\text{th}^\text{floor}$),
> - $|S_{\gamma_\varphi}| < 0.5$ (near-zero allowed for QND dispersive readout).

### 10.3 Finite-difference computation

The default implementation is central finite differences on the log-infidelity
with relative step $h = 5\%$:

$$S_\theta^\text{FD} = \frac{\ln \mathcal I(\theta_\text{ref}(1+h)) - \ln \mathcal I(\theta_\text{ref}(1-h))}{\ln(1+h) - \ln(1-h)} \approx \frac{\ln \mathcal I(\theta(1+h)) - \ln \mathcal I(\theta(1-h))}{2h} + O(h^2) \qquad \text{[Approximation]} \tag{10.3}$$

The $O(h^2)$ error requires that $\mathcal I(\theta)$ be smooth on the scale of $h$,
which is true at interior points of the parameter space but can fail near
sharp boundaries (high-drive leakage, $\kappa\tau \sim 1$ response crossover).
Test O2 requires step-independence (comparing $h=5\%$ and $h=2.5\%$); if the
two disagree by more than $10\%$, the operating point is near a non-smooth
boundary and the step should be reduced.

### 10.4 Autodiff upgrade path

Autodiff replaces (10.3) with an exact gradient obtained by differentiating
the simulation graph. This requires the Lindblad integrator to be
JAX-traceable; standard QuTiP `mesolve` is not (it uses scipy ODE backends).
The Stage 06 autodiff scope is therefore narrowed to the two pulse-edge
parameters $(\sigma_\text{edge}, t_\text{plateau})$ via a lightweight JAX
re-implementation of the RK integrator limited to those two knobs.
Full-device autodiff is Gautier et al. (2025) territory — cited as the
post-submission extension path.

### 10.5 The regime map: $(|\chi|/\kappa, \gamma_1 \tau)$ coordinates at fixed $\kappa\tau$

> **[Map type — reduced analytic/dispersive design map, not a full-JC simulator sweep].**
> Figure 4a is a **reduced analytic/dispersive design map**, not a sweep
> of the full Module 1 JC simulator across a 2D grid. The map's two axes
> are *phenomenological* rescalings: $|\chi|/\kappa$ varies independently
> of $\kappa$, and $\gamma_1\tau$ varies independently of all other rates.
> Use the map for **design intuition**, not as a substitute for full-JC
> fidelity at a specific device.

Module 4's regime map plots $F_\text{assign}$ as a heatmap over the
dimensionless plane $(|\chi|/\kappa, \gamma_1 \tau)$. Using $|\chi|/\kappa$
(rather than $\chi/\kappa$) handles both signs of $\chi$ uniformly.

> **[Axes of the map do not uniquely determine the fidelity].** The measurement
> rate (6.4) can be rewritten as
>
> $$\Gamma_\text{meas}\cdot\tau \propto \frac{(|\chi|/\kappa)^2}{1 + 4(|\chi|/\kappa)^2}\cdot (\kappa\tau)\cdot \bar n$$
>
> which depends on **three** dimensionless ratios: $|\chi|/\kappa$,
> $\kappa\tau$, and $\gamma_1\tau$. The 2D heatmap over
> $(|\chi|/\kappa, \gamma_1\tau)$ therefore requires a third parameter to
> be held fixed. Stage 06 fixes $\kappa\tau$ at the reference value
> $(\kappa\tau)_\text{ref} = 2\pi \cdot 5\,\text{MHz} \cdot 500\,\text{ns}
> \approx 15.7$ when drawing the heatmap. The figure caption must state
> this fixed value explicitly.

$$F_\text{ideal}(|\chi|/\kappa, \gamma_1\tau;\; \kappa\tau = (\kappa\tau)_\text{ref}) \qquad \text{[Approximation, dimensionless scaling at fixed $\kappa\tau$]} \tag{10.4}$$

> **[Consistency rule for Purcell and ionization overlays].** A point in
> $(|\chi|/\kappa, \gamma_1\tau)$ space at fixed $\kappa\tau$ does **not**
> uniquely determine $g_{01}/\Delta$. Module 4's regime-map sweeps fix a
> **representative $(g_{01}, \Delta, \alpha)$ triple** at Stage 06's
> synthetic seed values (§1.4) and rescale only $\chi$ and $\kappa$
> phenomenologically along the path. The Purcell value at the seed is
> reported as a **scalar caption annotation**, not as a curve on the
> heatmap.

**Distribution of analytic curves and annotations:** The heatmap (fixed
$\kappa\tau$) carries exactly **one analytic curve drawn on the plotted
plane** — the pointer-state separation marker (10.7), $|\chi|\tau = \pi$.
The Purcell scalar and $\kappa\tau = 1$ response-time line live on the
auxiliary $\kappa\tau$-swept plot:

1. **Purcell scalar annotation (caption only, not a heatmap curve).** At
   the reference seed:

$$\gamma_\text{Purcell}\tau \approx (120/2700)^2 \cdot 15.7 \approx 0.031 \qquad \text{[Scalar at reference seed]} \tag{10.5}$$

2. **$\kappa$-too-small curve.** When $\kappa\tau < 1$, the resonator has
   not fully responded:

$$\kappa\tau = 1 \qquad \text{[Approximation, response-time marker]} \tag{10.6}$$

3. **Pointer-state separation marker ($|\chi|\tau \sim \pi$).**

$$|\chi|\tau = \pi \qquad \text{[Heuristic, visualization marker only]} \tag{10.7}$$

> **[Important — what (10.7) is not].** The line $|\chi|\tau = \pi$ is
> **not** a first-principles "dispersive-approximation breakdown"
> boundary. The actual breakdown — measurement-induced transmon
> ionization — is driven by **multiphoton resonances at specific
> resonator photon populations** (Shillito et al. 2022, Dumas et al.
> 2024). Do not label (10.7) a "regime boundary" in presentation slides.

### 10.5a Empirical ionization awareness

Stage 06 does not attempt to predict the ionization threshold from first
principles. Instead, Module 2's leakage channel (§8.4) catches the
*downstream* effect on assignment fidelity when the threshold is crossed,
and `analysis/operating_point.py` will flag warnings when the drive amplitude
pushes $\bar n_r$ into the regime where published studies (Shillito 2022,
Dumas 2024) report multiphoton resonances for comparable devices.

### 10.6 Pareto frontier formulation

For fixed device parameters $\mathbf{d} = (\chi, \kappa, \gamma_1, \gamma_\varphi, \bar n_\text{th})$,
and control parameters $\mathbf{c} = (\varepsilon_0, \tau)$, the Pareto problem is

$$\mathbf{c}^\ast(\tau_\text{max}; \mathbf{d}) \equiv \arg\max_{\mathbf{c}} F_\text{assign}(\mathbf{c}; \mathbf{d}) \quad \text{s.t. } \tau \le \tau_\text{max} \qquad \text{[Constrained optimization]} \tag{10.8}$$

Sweeping $\tau_\text{max}$ from $100\,\text{ns}$ to $2\,\mu\text{s}$ traces out
the frontier $F_\text{assign}^\ast(\tau_\text{max}; \mathbf{d})$.

**Monotonicity property (test O4).** Relaxing the duration constraint cannot
decrease the optimum. This is a constrained-optimization tautology
**for a deterministic objective**.

> **[Optimization objective convention].** Module 4's Pareto optimization
> evaluates $F_\text{assign}$ **analytically** via the Gaussian-overlap
> formula (7.5) using centroids $(I_q, Q_q)$ from the deterministic
> trajectory $\langle a\rangle_q(t)$ and the deterministic
> $\sigma_\parallel(\tau_\text{integration})$ from (6.6a). **Note this is
> $\tau$-dependent**: as $\tau$ varies along the Pareto sweep, the optimizer
> queries `IQNoiseParams.sigma_for_integration_window(tau)` for the
> noise level at each candidate $\tau$, ensuring SNR scales as $\sqrt\tau$
> rather than $\tau$. This gives a smooth, deterministic objective that
> SLSQP can optimize cleanly. **Shot-sampled fidelity** is used only for
> the final bootstrap uncertainty bar on the reported optimum and for
> independent validation of the analytic fidelity formula. It is **not**
> the optimizer's objective.
>
> **The centroid+Gaussian objective inherits the §6.2/§7.1 caveat:** at
> $\tau/T_1 \gtrsim 1\%$ it can bias the frontier toward longer windows
> by missing non-Gaussian $T_1$-jump tails. The §10.7 closed-loop runs
> therefore re-evaluate the recommended optimum either with the one-jump
> mixture (§7.3a) or with a single trajectory cross-check (V7), and
> report the deviation between the smooth-objective optimum and the
> jump-aware fidelity.

**Solver choice (SLSQP).** SLSQP handles: (a) bound constraints on $\mathbf{c}$,
(b) inequality constraints ($\tau \le \tau_\text{max}$), (c) smooth objectives
via sequential quadratic subproblems. Warm-start from a $5\times 5$ grid
prevents SLSQP from sliding into a local maximum.

### 10.7 Closed-loop recommendation

The full closed loop, end-to-end:

```
Module 3 fitted parameters  →  partial parameter pack
       (ω_q, T₁, T₂*, T₂,echo, ε_π)       │
                                           ▼
                         [ fill defaults: E_C, E_J, χ, κ, g, η_readout ]
                                           │
                                           ▼
                                      DeviceConfig
                                           │
                                           ▼
                                Module 4 Pareto optimization
                                           │
                                           ▼
                              Optimal (ε₀, τ) for this device
                                           │
                                           ▼
                     Module 4 sensitivity analysis at the optimum
                                           │
                                           ▼
                       Module 2 error budget at the optimum
                                           │
                                           ▼
                 RecommendationReport (YAML, narrative-templated)
```

> **[Partial parameter pack — not a full DeviceConfig].** Rabi / Ramsey /
> $T_1$ / Hahn-echo protocols determine at most
> $\{\omega_q, T_1, T_2^*, T_{2,\text{echo}}, \varepsilon_\pi\}$. They do
> **not** determine $E_C$, $E_J$, $\chi$, $\kappa$, $g_{01}$, or readout-chain
> noise parameters. The output of `to_device_config()` is therefore a
> **partial parameter pack** where the protocol-unmeasurable fields are
> filled with `REFERENCE_DEVICE` defaults (or user-supplied overrides).
> The YAML output **explicitly tags each field** with `source: fitted`
> or `source: default_filled`.

The narrative generation is **template-based**, not LLM-based: a fixed
string format with numerical placeholders filled from the `RecommendationReport`
dataclass.

### 10.8 Script connection for §10

| Equation | Script | Function | Test |
|---|---|---|---|
| (10.1), (10.3) sensitivities | `optimization/sensitivity.py` | `compute_log_sensitivity()` | O1, O2 |
| (10.5) Purcell scalar annotation (heatmap caption) | `optimization/regime_map.py` | `purcell_scalar_annotation()` | structural |
| (10.5 auxiliary) Purcell threshold curve (auxiliary $\kappa\tau$-swept plot) | `optimization/regime_map.py` | `purcell_threshold_kappa_tau()` | O3a |
| Derivation of the Purcell-threshold curve | from $\gamma_\text{Purcell}\,\tau = (g_{01}/\Delta)^2\,\kappa\tau \le 0.1$ (advisory threshold), solved for $\kappa\tau \le 0.1\,(\Delta/g_{01})^2$ | (closed-form, no script) | (analytic) |
| (10.6) $\kappa\tau$ response marker | `optimization/regime_map.py` | `kappa_tau_response_boundary()` | O3 |
| (10.7) $|\chi|\tau=\pi$ pointer marker (heuristic) | `optimization/regime_map.py` | `chi_tau_pointer_marker()` | — (visualization only) |
| §10.5a ionization awareness | `analysis/operating_point.py` | `ionization_warning_check()` | advisory |
| (10.8) Pareto formulation with $\sigma_\parallel(\tau)$ | `optimization/pareto.py` | `find_pareto_point()` | O4 |
| Closed loop | `optimization/recommend.py` | `recommend_from_fitted_parameters()` | O5 |
| Autodiff add-on | `optimization/autodiff_addon.py` | `autodiff_refine_pulse_edges()` | O7 |


## 11. Module 5a — DRAG-Corrected Single-Qubit X Gates with sin²-Windowed Gaussian Envelope

This chapter and the next one (§12) extend the framework from *measurement* of a
dispersively-coupled transmon to two further hardware-control problems on the
same device: **single-qubit gate calibration** (Module 5a) and
**measurement-feedback active reset** (Module 5b). The dispersive Hamiltonian,
RWA-rotating-frame conventions, and Lindblad collapse-operator structure of
§3-§5 are reused throughout. The new physics that enters here is (§11) the
weakly-anharmonic Duffing-oscillator *driven* dynamics on a non-trivial
$\{|0\rangle, |1\rangle, |2\rangle, |3\rangle\}$ ladder, and (§12) the
heterodyne stochastic master equation in the bad-cavity limit.

### 11.0 Recruiter-readable framing

**Hardware problem.** A transmon is a weakly anharmonic oscillator: the
$|1\rangle \to |2\rangle$ transition lies only $|\alpha|/2\pi \approx 210$ MHz
below the $|0\rangle \to |1\rangle$ transition. Resonant microwave pulses
short enough to be useful for fault-tolerant computing
($T_\text{gate} \sim 10$–$20\,\text{ns} \ll T_1, T_2$) carry spectral weight
at the $|1\rangle \to |2\rangle$ frequency and excite **leakage** out of the
computational subspace. DRAG (Derivative Removal by Adiabatic Gate; Motzoi,
Gambetta, Rebentrost, Wilhelm, *Phys. Rev. Lett.* **103**, 110501, 2009;
Gambetta, Motzoi, Merkel, Wilhelm, *Phys. Rev. A* **83**, 012308, 2011)
supplies a quadrature derivative of the in-phase envelope to cancel this
leakage perturbatively in $\Omega_x/|\alpha|$.

**Approximation made by Module 5a.** The chapter adopts a four-level Duffing
model of the transmon (truncation $|0\rangle, |1\rangle, |2\rangle, |3\rangle$;
the buffer level $|3\rangle$ verifies that next-nearest leakage
$|2\rangle \to |3\rangle$ is small under DRAG) in the rotating frame at
$\omega_d = \omega_q$, with a sin²-windowed Gaussian in-phase envelope and the
leading-order ("DRAG-1" / Motzoi-Wilhelm 2009) derivative quadrature
$\Omega_y(t) = -\beta \cdot \dot\Omega_x(t)/\alpha$ (sign matches the
implementation plan and the original Motzoi-Wilhelm 2009 result; with
$\alpha < 0$, this gives a positive $\Omega_y/\dot\Omega_x$ ratio). The model is *controlled*
in the sense that the omitted physics — counter-rotating contributions
([Approximation]: RWA), charge-basis corrections to the matrix-element ratio
$\lambda_{12}/\lambda_{01} = \sqrt 2$ ([Approximation]: Duffing), and AC-Stark
shift renormalization at higher $\Omega/|\alpha|$ ([Approximation]: not
absorbed into β at leading order) — are explicitly enumerated and are
individually small for the parameter regime where the v0 implementation
operates. Decoherence ($T_1$, $T_\varphi$, thermal, Purcell) **is**
propagated as Lindblad evolution during the pulse on the four-level
truncated Hilbert space; see §11.7 for the collapse-operator definitions
and Convention 21 for the intrinsic-vs-effective $T_1$ bookkeeping.

**Planned validation targets and prototype-notebook results.** The
"validated numerically" claims in this paragraph are
**[Status: prototype notebook result, not yet reproduced in the
v0 repo]** — values come from an exploratory pre-implementation
notebook supplied by the user and have not yet been reproduced by
the in-repo `gate_simulator.py` (which does not exist at the time of
writing per the framework header). The values listed below are
*reported as planned validation targets / expected prototype
outcomes* that the in-repo v0 must reproduce. Three observables are
tracked:

(i) Final-state leakage $P_2(T_\text{gate}) = \langle 2 | \rho(T_\text{gate}) | 2\rangle$
collapses by orders of magnitude when $\beta$ is calibrated. Prototype
suppression ratios $P_2^{\text{no-DRAG}}(T_\text{gate}) / P_2^{\text{DRAG}}(T_\text{gate}; \beta_\text{opt})$
across the prototype-notebook $T_\text{gate}$ sweep are $9.5\times$ at 10 ns,
$26.8\times$ at 12 ns, $92.1\times$ at 15 ns, with a sharp drop to
$1.4\times$ at 20 ns **[Status: prototype notebook, pending in-repo
reproduction]**.

(ii) Peak transient leakage $\max_t P_2(t)$ is suppressed *only modestly*
in the prototype notebook: $3.1\times$ at 10 ns, $1.7\times$ at 15 ns,
$1.3\times$ at 20 ns **[Status: prototype notebook, pending in-repo
reproduction]**. This ceiling is structurally consistent with the
DRAG-1 mechanism (§11.5), not a calibration failure.

(iii) The numerically optimal $\beta$ for a sin²-windowed Gaussian at
$T_\text{gate} = 10$ ns is $\beta_\text{opt} \approx 2.2$ **[Status:
prototype notebook, pending in-repo reproduction]** (in units where
the Motzoi-Wilhelm pure-Gaussian DRAG-1 prediction is $\beta = 1$).
The shift away from 1 is plausible for a shaped envelope and a
metric-dependent calibration objective (§11.4); the v0 in-repo scan
is the authoritative source.

The headline benchmark for the chapter is the probe-set X-gate error
$\varepsilon_X^\text{ref}(T_\text{gate} = 20\,\text{ns})$ — **[Status:
TBD pending v0 calibration runs]**; a prototype-level estimate of
$\sim 10^{-3}$ is expected based on the $T_\text{gate}/T_{1,\text{eff}} \sim 1.9 \times 10^{-3}$
**decay-event scale** (eq. 11.28; Convention 21 — Purcell folded into the qubit-only
gate-simulator decay rate; *not* an X-gate-error floor — the actual
incoherent contribution to $\varepsilon_X$ depends on input state,
metric, and target operation, see §11.7 boxed warning) plus a coherent residual whose magnitude depends on
the calibrated $\beta_\text{opt}$. $\varepsilon_X$ is the mean error
over the probe set $\{|0\rangle, |1\rangle, |+\rangle, |+i\rangle\}$ at
the calibrated $\beta$. The actual $\varepsilon_X^\text{ref}$ value
will be substituted in once `gate_simulator.py` produces it.

---

### 11.1 Microscopic Hamiltonian and rotating frame

**[Exact within model — Duffing approximation].** A capacitively-driven
transmon is modeled as a weakly anharmonic Kerr oscillator (cf. §3.2):

$$H_\text{lab}(t) = \omega_q b^\dagger b + \tfrac{\alpha}{2}\, b^\dagger b\, (b^\dagger b - 1) + \Omega(t)\,\cos\!\bigl(\omega_d t + \varphi(t)\bigr)\,(b + b^\dagger) \qquad \text{(11.1)}$$

with $b$ the Duffing ladder operator obeying $[b, b^\dagger] = 1$ within the
truncation, $\alpha < 0$ the negative anharmonicity ($|\alpha|/2\pi = 210$ MHz
at the synthetic seed), and the drive parameterized by a real envelope
$\Omega(t) \ge 0$ with phase $\varphi(t)$. We split the drive into
*in-phase* and *quadrature* components

$$\Omega(t)\cos(\omega_d t + \varphi(t)) = \Omega_x(t)\cos\omega_d t + \Omega_y(t)\sin\omega_d t \qquad \text{(11.2)}$$

with $\Omega_x(t) = \Omega(t)\cos\varphi(t)$ and
$\Omega_y(t) = -\Omega(t)\sin\varphi(t)$. The DRAG correction will assign
$\Omega_y$ as a derivative of $\Omega_x$.

> **[Convention warning — sign of $\alpha$ and $\beta$].** We adopt
> $\alpha < 0$ throughout (transmon negative anharmonicity), so the
> $|2\rangle$ level sits at energy $\alpha$ below $2\omega_q$ on the
> diagonal of the rotating-frame Hamiltonian. The DRAG quadrature sign
> $\beta > 0$ corresponds to $\Omega_y(t) = -\beta\, \dot\Omega_x(t)/\alpha$
> in this convention. With $\alpha < 0$ this means $\Omega_y$ has the
> *same* sign as $\dot\Omega_x$ scaled by $1/|\alpha|$. A sign error on
> $\beta$ changes the interference condition that cancels the
> leakage-channel Fourier component and generally appears as **increased
> endpoint leakage together with coherent phase/axis error on the
> computational subspace**. The exact form of the residual coherent
> error is envelope-, detuning-, and metric-dependent (no closed-form
> $Z(\theta)$ phase formula is asserted here); the sign is therefore
> diagnosed **operationally** by the V6 sign-flip test of §11.9 rather
> than by a closed-form phase expression. The convention
> matches Gambetta et al. *Phys. Rev. A* **83**, 012308 (2011), Eq. (3);
> some references (Krantz et al. *Appl. Phys. Rev.* **6**, 021318, 2019,
> §III.D) absorb the sign of $\alpha$ into $\beta$.

**Truncation.** We work on the four-level Duffing manifold
$\{|0\rangle, |1\rangle, |2\rangle, |3\rangle\}$. The buffer level
$|3\rangle$ is included for two reasons: (a) it provides a convergence
check on the $|2\rangle$ population (V3, §11.9 — populations of $|3\rangle$
should be $\le 10^{-2}$ of $|2\rangle$ throughout the pulse for our regime),
and (b) higher-order DRAG variants (DRAG-2, FAST DRAG; cf. §11.6) generate
direct $|0\rangle \to |3\rangle$ couplings that the four-level truncation
captures. The harmonic matrix-element ratios in the Duffing limit are
$\langle j+1 | b | j \rangle = \sqrt{j+1}$, so $\sqrt 2$ couples the
$|1\rangle \to |2\rangle$ transition and $\sqrt 3$ couples
$|2\rangle \to |3\rangle$.

> **[Approximation — Duffing vs charge-basis matrix elements].** In the
> *charge basis* of a transmon (Koch et al. 2007, §III), the true
> matrix elements differ from the Duffing $\sqrt{j+1}$ by
> $O((E_C/E_J)^{1/2})$. For $E_J/E_C \approx 65.6$ at the seed, this is
> a $\sim 6\%$ correction on $|\langle 1|\hat n|2\rangle / \langle 0|\hat n|1\rangle|$
> relative to $\sqrt 2$. Khani, Gambetta, Motzoi, Wilhelm, *New J. Phys.*
> **11**, 113006 (2009) [arXiv:0909.4788] show that the correction matters
> for sub-3-ns pulses but is sub-dominant for the $T_\text{gate} \ge 10$ ns
> regime of v0. A charge-basis drive using the exact matrix elements from
> §2.5 is a v1.5 extension; v0 claims should be phrased as
> Duffing-model results.

**[Approximation — RWA].** Going to the frame rotating at $\omega_d = \omega_q$
via the unitary $U_R = \exp(i\omega_d t \sum_j j\,|j\rangle\langle j|)$
and dropping counter-rotating terms (good when $\Omega \ll \omega_q \approx 2\pi \cdot 4.6$ GHz,
certainly true for $\Omega/2\pi \le 100$ MHz used in v0):

$$\begin{aligned}
H_R(t) =\;& \alpha\, |2\rangle\langle 2| + 3\alpha\, |3\rangle\langle 3| \\
& + \tfrac{1}{2}\Omega_x(t)\,(b + b^\dagger) - \tfrac{i}{2}\Omega_y(t)\,(b - b^\dagger)
\end{aligned} \qquad \text{(11.3)}$$

In matrix form on $\{|0\rangle, |1\rangle, |2\rangle, |3\rangle\}$,
identifying $b\,|j\rangle = \sqrt j\,|j-1\rangle$:

$$H_R(t) = \begin{pmatrix} 0 & \tfrac{1}{2}(\Omega_x - i\Omega_y) & 0 & 0 \\ \tfrac{1}{2}(\Omega_x + i\Omega_y) & 0 & \tfrac{\sqrt 2}{2}(\Omega_x - i\Omega_y) & 0 \\ 0 & \tfrac{\sqrt 2}{2}(\Omega_x + i\Omega_y) & \alpha & \tfrac{\sqrt 3}{2}(\Omega_x - i\Omega_y) \\ 0 & 0 & \tfrac{\sqrt 3}{2}(\Omega_x + i\Omega_y) & 3\alpha \end{pmatrix} \qquad \text{(11.4)}$$

The two-level subblock (rows/columns 0-1) is the standard Rabi problem; the
crucial new feature is the off-diagonal coupling $\sqrt 2(\Omega_x + i\Omega_y)/2$
between $|1\rangle$ and $|2\rangle$, with the on-diagonal $\alpha$ providing
the only protection against leakage. **DRAG cancels the leading effect of
this coupling perturbatively in $\Omega_x/|\alpha|$.**

---

### 11.2 Adiabatic-frame derivation of DRAG-1

We now derive the leading-order DRAG correction $\Omega_y(t) = -\dot\Omega_x(t)/\alpha$,
following Gambetta, Motzoi, Merkel, Wilhelm, *Phys. Rev. A* **83**, 012308
(2011) (henceforth GMMW-11), §II-III.

**[Setup].** Define projectors $P_q = |0\rangle\langle 0| + |1\rangle\langle 1|$
(qubit subspace) and $P_L = |2\rangle\langle 2| + |3\rangle\langle 3|$
(leakage manifold). Decompose

$$H_R(t) = \underbrace{P_q H_R P_q + P_L H_R P_L}_{\text{block-diagonal: } H_q \oplus H_L} + \underbrace{P_q H_R P_L + P_L H_R P_q}_{\text{coupling: } V_{qL}} \qquad \text{(11.5)}$$

The leakage block $H_L = \alpha\, |2\rangle\langle 2| + 3\alpha\, |3\rangle\langle 3|$
has gap $|\alpha|$ from the qubit block; the coupling
$V_{qL}$ has matrix-element scale $\sim \tfrac{\sqrt 2}{2}|\Omega(t)|$.
The natural smallness parameter is

$$\varepsilon_\text{DRAG} \equiv \bar\Omega / |\alpha| \qquad \text{(11.6)}$$

with $\bar\Omega = \max_t |\Omega(t)|$ a characteristic envelope amplitude
(for our $\pi$-pulse with $T_\text{gate} = 20$ ns, $\bar\Omega/2\pi \approx 1/(2 T_\text{gate}) \approx 25$ MHz, so
$\varepsilon_\text{DRAG} \approx 0.12$ — comfortably small for a perturbative
expansion).

**[Construction — explicit derivation following user-review request].**
The DRAG-1 cancellation can be derived in five steps from the leakage
block in (11.4). We isolate the $|1\rangle \leftrightarrow |2\rangle$
sector (the dominant leakage channel; $|0\rangle \to |2\rangle$ is
suppressed by $|\lambda_{02}|^2 = 0$ at first order).

*Step 1 — Leakage-block Hamiltonian.* Reading off the (1,2) and (2,1)
matrix elements of (11.4) plus the diagonal $\alpha$ on $|2\rangle$:

$$H_R^{(12)}(t) = \alpha\,|2\rangle\langle 2| + \frac{\sqrt 2}{2}\bigl[\Omega(t)\,|2\rangle\langle 1| + \Omega^*(t)\,|1\rangle\langle 2|\bigr], \qquad \Omega(t) \equiv \Omega_x(t) + i\,\Omega_y(t). \qquad \text{(11.10a)}$$

The first (diagonal) piece is $H_0 = \alpha\,|2\rangle\langle 2|$; the
second is the leakage coupling $V_L(t)$ of order $\Omega/\alpha$.

*Step 2 — Leakage amplitude in the interaction picture.* In the
interaction picture with respect to $H_0$, the $|0\rangle \to |1\rangle \to |2\rangle$
transition amplitude at time $T_\text{gate}$ is, to first order in
$V_L$ and treating the qubit-subspace amplitude $c_1(t)$ as slowly
varying and order unity over the leakage-timescale oscillation
$1/|\alpha|$ (the spectral-cancellation argument depends on this
slow-variation property, not on $|c_1(t)| \approx 1$ throughout the
pulse — for an $X$ pulse starting from $|0\rangle$, $c_1$ ramps from
$0$ to $1$ over the pulse, but its variation is slow compared to
$1/|\alpha|$ in the perturbative regime $\bar\Omega/|\alpha| \ll 1$):

$$c_2(T_\text{gate}) \approx \frac{\sqrt 2}{2 i}\int_0^{T_\text{gate}}\Omega(t)\,c_1(t)\,e^{i\alpha t}\,dt \qquad \text{(11.10b)}$$

(the factor $e^{i\alpha t}$ comes from going to the interaction picture
with respect to $H_0$; the qubit-subspace amplitude $c_1(t)$ is *kept
explicitly* in the integrand because for an $X$ pulse starting from
$|0\rangle$, $c_1(t)$ ramps from $0$ to $1$ over the pulse and is *not*
order unity throughout). Endpoint leakage $P_2^\text{final} = |c_2(T_\text{gate})|^2$
is therefore controlled by the Fourier component of the *windowed*
quantity $\Omega(t)\,c_1(t)$ at frequency $-\alpha$.

> **[Caveat — Fourier-notch picture is intuition, not full proof].** The
> integration-by-parts manipulation in Step 3 below cancels the leading
> contribution to (11.10b) when $c_1(t)$ is treated as slowly varying
> on the leakage-timescale $1/|\alpha|$ (a much weaker assumption than
> $|c_1(t)| \approx 1$). This Fourier-notch picture is an **intuition
> for endpoint leakage**, not a complete proof for a finite-area $X$
> pulse where $c_1$ varies substantially. The textbook DRAG cancellation
> is more rigorously obtained by the **adiabatic-frame /
> Schrieffer-Wolff construction** (Motzoi-Wilhelm 2009 §III; GMMW-11);
> what (11.10b)–(11.11) provide here is an explicit derivation of *why*
> the choice $\Omega_y = -\dot\Omega_x/\alpha$ is the natural one to
> spectrally cancel endpoint leakage, not a from-first-principles
> demonstration that *all* leakage cancels at first order.

*Step 3 — Integration by parts.* If $\Omega_x(0) = \Omega_x(T_\text{gate}) = 0$
(enforced by the sin²-window of §11.4),

$$\int_0^{T_\text{gate}}\Omega_x(t)\,e^{i\alpha t}\,dt = \underbrace{\biggl[\frac{\Omega_x(t)\,e^{i\alpha t}}{i\alpha}\biggr]_0^{T_\text{gate}}}_{=\;0\;\text{by boundary conditions}} - \frac{1}{i\alpha}\int_0^{T_\text{gate}}\dot\Omega_x(t)\,e^{i\alpha t}\,dt = \frac{i}{\alpha}\int_0^{T_\text{gate}}\dot\Omega_x(t)\,e^{i\alpha t}\,dt. \qquad \text{(11.10c)}$$

This is the key trick: the "DC" component of $\Omega_x$ at frequency
$-\alpha$ is converted, by integration by parts, into a $\dot\Omega_x$
component multiplied by $i/\alpha$.

*Step 4 — DRAG-1 cancellation condition.* Substitute (11.10c) into
(11.10b) using $\Omega = \Omega_x + i\,\Omega_y$:

$$c_2(T_\text{gate}) \approx \frac{\sqrt 2}{2 i}\int_0^{T_\text{gate}}\biggl[\frac{i}{\alpha}\dot\Omega_x(t) + i\,\Omega_y(t)\biggr]\,e^{i\alpha t}\,dt = \frac{\sqrt 2}{2}\int_0^{T_\text{gate}}\biggl[\frac{\dot\Omega_x(t)}{\alpha} + \Omega_y(t)\biggr]\,e^{i\alpha t}\,dt. \qquad \text{(11.10d)}$$

For the integrand to vanish identically, choose

$$\boxed{\Omega_y(t) = -\frac{\dot\Omega_x(t)}{\alpha}} \qquad \text{(11.11) — DRAG-1}$$

This is the canonical Motzoi-Wilhelm 2009 result, derived here from
boundary integration by parts on the leakage Fourier integral.

*Step 5 — Sign-convention check under $\alpha < 0$.* For the transmon
with $\alpha < 0$, dividing $\dot\Omega_x$ by $\alpha$ flips the sign:

$$\Omega_y(t) = -\dot\Omega_x(t)/\alpha = +\dot\Omega_x(t)/|\alpha| \quad (\text{when }\alpha < 0).$$

So a *positive* lobe of $\dot\Omega_x$ (envelope rising) generates a
*positive* $\Omega_y$ correction, in phase with the rising envelope.
This is the convention adopted throughout the document and the
implementation plan; it matches Convention 18 (DRAG quadrature sign) and the V6
sign-flip test (§11.9).

*Generalization to a free $\beta$.* Under the convention
$\Omega_y = -\beta\,\dot\Omega_x/\alpha$,
the canonical pure-Gaussian DRAG-1 result corresponds to $\beta = 1$
(Motzoi *et al.* 2009, Eq. 12). For a sin²-windowed Gaussian (§11.4)
the empirically optimal $\beta$ shifts away from 1, as discussed in
§11.4 and §11.5.

The corresponding final-state leakage scales as

$$P_2^\text{final, no-DRAG} \sim \frac{|\lambda_{12}|^2}{4}\biggl|\int_0^{T_\text{gate}} \Omega_x(t)\, e^{i\alpha t}\, dt\biggr|^2 \qquad \text{(11.12)}$$

— that is, the squared modulus of the Fourier component of the envelope
at the leakage frequency $\omega = |\alpha|$, with the dimensionless
matrix-element prefactor $|\lambda_{12}|^2/4$. The integral
$\int \Omega_x(t)\,e^{i\alpha t}\,dt$ is dimensionless (rad/s × s →
dimensionless rotation angle) and is of order $\bar\Omega/|\alpha|$ for
endpoint-zero envelopes by integration by parts, so

$$P_2^\text{final, no-DRAG} = O\!\bigl((\bar\Omega/|\alpha|)^2\bigr) \qquad \text{[scaling consequence of (11.12)]}.$$

DRAG-1 *spectrally cancels* the leading Fourier component to the chosen
order: equivalently, it places a notch in the spectrum of the complex
envelope $\tilde\Omega(t) = \Omega_x + i\Omega_y$ at $\omega = |\alpha|$
(Theis, Motzoi, Machnes, Wilhelm, *EPL* **123**, 60001, 2018 [arXiv:1809.04919],
§III). After cancellation,

$$P_2^\text{final, DRAG-1} \sim O(\varepsilon_\text{DRAG}^4) \qquad \text{(11.13)}$$

a quartic suppression in $\varepsilon_\text{DRAG}$, validated by the V2a numerical
test in §11.9.

> **[Crucial structural remark — DRAG-1 cancels endpoint, not transient,
> leakage].** Equation (11.11) cancels the *endpoint* leakage Fourier
> component (eq. 11.12) under the boundary condition
> $\Omega_x(0) = \Omega_x(T_\text{gate}) = 0$. It does **not** annul the
> off-diagonal qubit-$|2\rangle$ matrix element of the leakage-block
> instantaneously during the pulse; transient leakage populations of
> order $(\Omega(t)/\alpha)^2$ persist mid-pulse. This is the structural
> origin of the saturation in peak-leakage suppression observed
> numerically (§11.5).

**[Approximation — the hierarchy].** The DRAG-1 form (11.11) cancels
$O(\varepsilon_\text{DRAG}^1)$ qubit-leakage coupling at the endpoints. The leading
non-cancelled contribution at $O(\varepsilon_\text{DRAG}^2)$ is an AC-Stark shift on
the qubit transition: virtual occupation of $|2\rangle$ during the pulse
displaces the qubit transition frequency by

$$\delta\omega_\text{Stark}(t) = -\frac{\Omega_x(t)^2}{4\alpha}\,\bigl(|\lambda_{12}|^2 - |\lambda_{01}|^2\bigr) = -\frac{\Omega_x(t)^2}{4\alpha} \qquad \text{(11.14)} \qquad \text{[Unverified sign/factor pending source check against GMMW-11]}$$

(with Duffing matrix elements $|\lambda_{01}|^2 = 1$ and $|\lambda_{12}|^2 = 2$).
Compensating this shift requires a *time-dependent detuning*

$$\delta_d(t) = -\Omega_x(t)^2 / (4\alpha) \qquad \text{(11.15)} \qquad \text{[Unverified sign/factor pending source check against GMMW-11]}$$

added either as a phase ramp on the drive or as a real-time frequency
modulation. v0 of Module 5a omits this AC-Stark detuning at the
implementation level (it is absorbed phenomenologically into the
calibrated $\beta$, see §11.3); v1.5 will add it explicitly.


---

### 11.3 Higher-order DRAG and the family of solutions

The full GMMW-11 (Gambetta-Motzoi-Merkel-Wilhelm 2011) construction extends
Eq. (11.11) to a one-parameter family. Introducing a phase ramp $\varphi(t)$
on the drive, the Hamiltonian (11.3) acquires an additional time-dependent
detuning $\delta_d(t) = -\dot\varphi(t)$. Their canonical analytic family
(GMMW-11 Eqs. 16-20) is parameterized by a single scalar $\beta$:

$$\Omega_y(t) = -\beta\, \dot\Omega_x(t)/\alpha, \qquad \delta_d(t) = -(2\beta - 1)\, \Omega_x(t)^2 / (4\alpha) \qquad \text{(11.16)} \qquad \text{[Stark prefactor (2β−1)/4 unverified; pending line-by-line check against GMMW-11. The DRAG quadrature sign convention is independently verified in §11.2 / Convention 18.]}$$

Different choices of $\beta$ minimize different error metrics:

- **$\beta = 1$ — leakage-minimizing for pure Gaussian envelope.** This is the
  original Motzoi 2009 prediction. The spectral notch at $\omega = |\alpha|$
  is fully cancelled.
- **$\beta \approx 1/2$ — often associated with reduced Stark-phase error
  under the GMMW-style parameterization (the exact optimum is hardware-,
  envelope-, and metric-dependent).** Lucero, Kelly,
  Bialczak et al. *Phys. Rev. A* **82**, 042339 (2010) [arXiv:1007.1690]
  showed that for phase-qubit hardware, the metric of interest is the
  AC-Stark phase error rather than population leakage; in their
  parameterization, the optimal $\beta$ shifts toward $\sim 1/2$ in
  the regime where AC-Stark detuning compensation matters most. The
  $(2\beta - 1)/4$ prefactor in (11.16) is **unverified pending source
  check against GMMW-11**, so the specific value $\beta = 1/2$
  exactly should not be treated as a derived theory prediction here.
- **Average gate fidelity** $1 - \bar F$ — a compromise. For transmons with
  $\sqrt 2$ ladder factor and $\Omega/|\alpha|$ modest, the leakage and
  phase contributions to $1 - \bar F$ are comparable, and the optimum
  $\beta_F^\text{opt}$ lies in $[0.5, 1.0]$ for a *pure Gaussian envelope*.

For the *sin²-windowed Gaussian* envelope of v0, the optimum $\beta_\text{opt}$
shifts substantially because the second derivative $\ddot\Omega_x$ is no
longer a Gaussian — see §11.4.

> **[Practical consequence].** Experimental implementations *always*
> calibrate $\beta$ in situ rather than using a textbook value (Chow,
> DiCarlo, Gambetta et al., *Phys. Rev. A* **82**, 040305(R), 2010
> [arXiv:1005.1279]; Lucero et al. 2010; Werninghaus, Egger, Roy,
> Machnes, Wilhelm, Filipp, *npj QI* **7**, 14, 2021 [arXiv:2003.05952]).
> v0 of Module 5a follows this convention: $\beta_\text{opt}$ is found
> by a 1D scan minimizing the X-gate probe-set error $\varepsilon_X$.

---

### 11.4 sin²-windowed Gaussian envelope: explicit functional form

**[Exact within model — envelope definition].** Define the dimensionless
shape function

$$g(t) = \sin^2\!\left(\frac{\pi t}{T_\text{gate}}\right)\,\exp\!\left[-\frac{(t - T_\text{gate}/2)^2}{2\sigma^2}\right] \qquad t \in [0, T_\text{gate}], \qquad \text{(11.17)}$$

with $\sigma$ a width parameter (default $\sigma = T_\text{gate}/4$). Let

$$I(\sigma, T_\text{gate}) \equiv \int_0^{T_\text{gate}} g(t')\,dt' \quad [\text{units: seconds}] \qquad \text{(11.18a)}$$

be the time integral of the shape (a numerical constant computed once for
each $(\sigma, T_\text{gate})$ pair). The in-phase envelope is then
constructed by area-normalization to give a $\pi$-pulse:

$$\Omega_x(t) = \frac{\Theta_\text{target}}{I(\sigma, T_\text{gate})}\,g(t), \qquad \Theta_\text{target} = \pi \quad (\text{X-gate}) \qquad \text{(11.18)}$$

so that $\int_0^{T_\text{gate}} \Omega_x(t)\,dt = \Theta_\text{target} = \pi$ exactly,
$\Omega_x$ has units of rad/s, and the **peak amplitude is a derived
quantity**:

$$\Omega_\text{pk} \equiv \max_t \Omega_x(t) = \frac{\Theta_\text{target}}{I(\sigma, T_\text{gate})}\,\max_t g(t) \quad [\text{rad/s, derived}]. \qquad \text{(11.18b)}$$

The implementation reads $T_\text{gate}$ and $\sigma$ from `PulseParams`,
computes $I$ numerically, and returns $\Omega_x(t)$ on the time grid; no
free amplitude parameter is exposed at v0.

> **Notation note (per F4.1 review).** v0 does *not* keep both a free
> peak amplitude $A_0$ and a $\pi$-area constraint as independent inputs
> — these would be over-specified. The single user-visible knob is
> $T_\text{gate}$ (with $\sigma$ defaulted to $T_\text{gate}/4$); the
> rotation angle is fixed at $\pi$. If a future v1.5 extension exposes
> $\Omega_\text{pk}$ as an independent input, the rotation angle becomes
> $\Theta = \Omega_\text{pk}\,I/\max_t g(t)$ and is no longer constrained
> to $\pi$.

**[Boundary properties].** The sin²-window enforces:

1. $\Omega_x(0) = \Omega_x(T_\text{gate}) = 0$ — exact endpoint zeros, required
   for DRAG-1 (Eq. 11.11) to cancel the endpoint leakage Fourier component.
2. $\dot\Omega_x(0) = \dot\Omega_x(T_\text{gate}) = 0$ — endpoint-derivative
   zeros, automatic for $\sin^2$ since
   $\partial_t \sin^2(\pi t/T_\text{gate}) = (\pi/T_\text{gate})\sin(2\pi t/T_\text{gate})$
   vanishes at $t = 0, T_\text{gate}$. This makes the DRAG quadrature
   $\Omega_y \propto \dot\Omega_x$ also vanish at the boundaries — eliminating
   spectral leakage from a sudden turn-on of the quadrature drive.
3. $C^\infty$ smoothness on the interior.

The DRAG quadrature obtained from (11.16) is

$$\Omega_y(t) = -\frac{\beta}{\alpha}\,\dot\Omega_x(t) = -\frac{\beta\,\pi}{\alpha\,I(\sigma, T_\text{gate})}\,\biggl\{\frac{2\pi}{T_\text{gate}}\,\sin\!\left(\frac{\pi t}{T_\text{gate}}\right)\!\cos\!\left(\frac{\pi t}{T_\text{gate}}\right) - \sin^2\!\left(\frac{\pi t}{T_\text{gate}}\right)\frac{t - T_\text{gate}/2}{\sigma^2}\biggr\}\, e^{-(t-T_\text{gate}/2)^2/(2\sigma^2)} \qquad \text{(11.19)}$$

This analytic form is what `pulses.py::sin2_windowed_gaussian_derivative` must
return; computing $\dot\Omega_x$ by finite difference would lose precision near
the endpoints and bias DRAG calibration.

**[Approximation — leading-order $\beta_\text{opt}$ shift for sin²-window].**
The spectral-notch analysis (Theis et al. 2018, §III; GMMW-11 Eq. 17) shows
that the optimal $\beta$ for *endpoint* leakage cancellation depends on the
envelope shape through the structure of $\ddot\Omega_x$ at the endpoints. For
a pure Gaussian centered in $[0, T_\text{gate}]$, $\ddot\Omega_x$ at the
boundaries is set entirely by the Gaussian curvature
$\sim \Omega_x/\sigma^2$, giving $\beta_\text{opt} = 1$. For the
sin²-windowed Gaussian, the boundary curvature acquires an additional
contribution from the window:

$$\ddot\Omega_x(0) \approx (\pi/I) \cdot \biggl[\frac{2\pi^2}{T_\text{gate}^2}\, e^{-T_\text{gate}^2/(8\sigma^2)} + \text{near-boundary curvature structure}\biggr] \qquad \text{(11.20) — schematic scaling only}$$

The sin² window changes the boundary curvature and the spectral content
of the envelope, so the leakage-minimizing $\beta$ need not equal the
pure-Gaussian DRAG-1 value $\beta = 1$. Any explicit perturbative
expression in $(\sigma/T_\text{gate})^2$ would have an undetermined
shape-specific O(1) constant whose exact value depends on which error
metric is being minimized. Stage 06 therefore treats $\beta_\text{opt}$
as an **empirical calibration output**, not an analytic prediction.
We do *not* derive a shape-specific β-shift formula here — the precise
envelope-shape correction for the sin²-window with truncated Gaussian
core has not been worked out analytically in the literature for the
four-level Duffing model with AC-Stark and $|3\rangle$-admixture
contributions.

> **[Framing — empirical calibration is authoritative].** v0 of Module 5a
> calibrates $\beta_\text{opt}$ by a numerical 1D scan
> (`drag_calibration.py::calibrate_drag_beta`) and treats the result as
> empirical. The deviation $\beta_\text{opt} \neq 1$ is *plausible* for a
> sin²-windowed envelope and for a metric-dependent calibration objective
> (leakage vs. phase-error vs. average fidelity), but its quantitative
> value is not pinned down by analytic theory at v0. v0 reports
> $\beta_\text{opt}$ from the numerical scan as the headline value, with
> the analytic Gaussian DRAG-1 reference $\beta = 1$ reported alongside
> for context.

---

### 11.5 Peak vs final leakage: structural saturation as a model-limits FINDING

This subsection explains why DRAG-1 cancels *final-state* leakage by orders
of magnitude but only modestly suppresses *peak* mid-pulse leakage. The
explanation is structural and is recognized as a positive theoretical
finding rather than a calibration failure.

**[Setup — instantaneous $|2\rangle$ population].** During a DRAG-1 pulse,
the instantaneous $|2\rangle$ population can be obtained by transforming
back from the adiabatic frame and using that, in the adiabatic frame, the
$|0\rangle \to |2\rangle$ amplitude is small but *non-zero* for
$t \in (0, T_\text{gate})$. Following the leading-order adiabatic-frame
interpretation of GMMW-11 (specific equation number not verified
line-by-line; cf. also Theis et al. 2018, §IV.B), the $|2\rangle$
amplitude in the lab frame factorizes as

$$c_2(t) \approx \frac{|\lambda_{12}|}{2|\alpha|}\,\Omega_x(t)\,c_1(t) + \text{DRAG-corrected residual} \qquad \text{(11.22)}$$

so the instantaneous population is

$$|c_2(t)|^2 \approx \frac{|\lambda_{12}|^2}{4\alpha^2}\,\Omega_x(t)^2 \cdot |c_1(t)|^2 + O(\varepsilon_\text{DRAG}^4) \qquad \text{(11.23)}$$

This is the **AC-Stark virtual population**: it does *not* vanish at
DRAG-1 order. Its peak scales as

$$P_2^\text{peak} \approx \frac{|\lambda_{12}|^2}{4\alpha^2}\,\bar\Omega^2 = \frac{\bar\Omega^2}{2\alpha^2} \qquad \text{(11.24)}$$

with $|\lambda_{12}|^2 = 2$ in the Duffing limit. For a $\pi$-pulse with
$\bar\Omega \sim \pi/T_\text{gate}$ (envelope-shape-dependent prefactor),

$$P_2^\text{peak} \sim \frac{1}{2}\biggl(\frac{\pi}{T_\text{gate}\,|\alpha|}\biggr)^2 \qquad \text{(11.25)}$$

**[Independence from $\beta$ at leading order].** Crucially, $\beta$ in
(11.16) only multiplies the *correction* to $c_2(t)$, not the leading-order
Stark virtual term in (11.23). To leading order in $\varepsilon_\text{DRAG}$, **DRAG-1
cannot suppress $P_2^\text{peak}$**: the suppression ratio
$P_2^\text{peak, no-DRAG}/P_2^\text{peak, DRAG-1}$ saturates at an
envelope-shape-dependent value of order 1 to a few. This is the structural
origin of the prototype-notebook saturation pattern reported in §11.0
($3.1\times$ at 10 ns, $1.7\times$ at 15 ns, $1.3\times$ at 20 ns —
**[Status: prototype, pending in-repo reproduction]** as labeled there).

**[Final-state leakage scaling].** In contrast to the peak, the *final-state*
leakage *does* receive direct DRAG-1 cancellation. From (11.12) and (11.13),
the suppression ratio scales as

$$\frac{P_2^\text{final, no-DRAG}}{P_2^\text{final, DRAG-1}} \sim \frac{O(\varepsilon_\text{DRAG}^2)}{O(\varepsilon_\text{DRAG}^4)} = O(\varepsilon_\text{DRAG}^{-2}) \sim \biggl(\frac{T_\text{gate}\,|\alpha|}{\pi}\biggr)^2 \qquad \text{(11.26)}$$

so the suppression ratio grows roughly *quadratically* with $T_\text{gate}$
in the regime where DRAG-1 is dominant — matching the empirical
$9.5 \to 26.8 \to 92.1$ across $T_\text{gate} = 10 \to 12 \to 15$ ns.

**[Why the suppression collapses at $T_\text{gate} = 20$ ns — open
empirical finding].** The empirical observation that final-leakage
suppression *drops* to $1.4\times$ at 20 ns is **not yet diagnosed**.
The naive perturbative-exit story is *not* obviously correct: the
DRAG-1 perturbative parameter $\varepsilon_\text{DRAG} = \bar\Omega/|\alpha|$
typically *decreases* as $T_\text{gate}$ increases for a fixed-area
$\pi$-pulse (since $\bar\Omega \sim \pi/T_\text{gate}$), so we are
moving deeper into the perturbative regime, not out of it. Plausible
mechanisms for the observed collapse — to be diagnosed before being
asserted — include:

1. **Floor crossover.** The DRAG-1 residual stays $O(\varepsilon_\text{DRAG}^4)$ at
   the boxed eq. (11.13), but the no-DRAG baseline shrinks like
   $1/T_\text{gate}^2$ (eq. 11.12), so the *ratio*
   $P_2^\text{no-DRAG}/P_2^\text{DRAG-1}$ becomes the comparison of two
   small numbers and can collapse to $O(1)$ even without exiting the
   perturbative regime.
2. **Numerical artifacts.** Time-grid resolution $dt$ relative to
   $1/|\alpha|$, ODE solver tolerance, or boundary-condition handling
   in the time integrator could plausibly contribute at the floor.
3. **AC-Stark phase-pull and $|3\rangle$-admixture residuals.** These
   become a larger fraction of the (small) total leakage as
   $T_\text{gate}$ increases.
4. **Spectral interference with the sin²-window edges.** The Fourier
   spectrum of the windowed envelope acquires structure that
   interferes with the leakage cancellation in a $T_\text{gate}$-dependent
   way.
5. **Calibration-objective tradeoff.** $\beta_\text{opt}$ minimizes the
   X-gate error metric, not pure final leakage; at long $T_\text{gate}$
   the AC-Stark phase pull and $T_1$ contributions become a larger
   fraction of $\varepsilon_X$, shifting $\beta_\text{opt}$ away from
   the leakage-only optimum.

The collapse should be diagnosed from absolute leakage curves,
$\beta_\text{opt}(T_\text{gate})$, grid-refinement tests, separated
phase-error vs. leakage metrics, and $|3\rangle$-population sweeps
*before* a mechanism is asserted in writing. v0 reports the empirical
suppression-ratio curve and notes the $T_\text{gate} = 20$ ns
collapse as an **open empirical finding awaiting diagnosis**, not as
an exit-from-perturbative-regime conclusion.

**[Framing — model-limits FINDING, not calibration failure].** None of
this is a bug:

- The peak-leakage saturation cap of $\sim 3\times$ at 10 ns is *qualitatively*
  predicted by (11.23)-(11.25) — DRAG-1 cancels endpoint, not transient,
  leakage.
- The exact numerical value of the saturation cap (3.1×, 1.7×, 1.3×) is
  **not** something the leading-order analytic formula must reproduce to
  within tens of percent; the analytic formula predicts the *structure*
  and the *trend*, not a precise numerical match. This is honest about
  scope: a more careful prediction would require including the AC-Stark
  detuning (11.15) explicitly in the rotating-frame Hamiltonian, the
  Duffing-truncation $|3\rangle$-admixture, and the second-order DRAG
  correction.
- The collapse at $T_\text{gate} = 20$ ns is **compatible with** the
  higher-order DRAG floor / floor-crossover argument (see also the
  candidate-mechanism list above), but **not yet diagnosed**: the
  weaker statement in the **[Why the suppression collapses ...]**
  block is the authoritative framing, not a quantitative prediction.

This framing is summarized in §11.9 as the V2a/V2b validation split: V2a
(blocking, final-leakage) tests the regime where DRAG-1 is dominant; V2b
(diagnostic, peak suppression) reports the structural saturation curve as
a v0 deliverable.


---

### 11.6 Why not DRAG-2? Higher-order variants and scope statement

The DRAG family extends beyond DRAG-1, but the higher-order variants are
out of scope for v0 of Module 5a. We list them here as forward pointers:

- **DRAG-2 (Gambetta et al. 2011, §IV.B).** Adds a third-order correction
  to $\Omega_x$ and a second-order correction to the AC-Stark detuning
  $\delta_d(t)$, in principle suppressing $P_2^\text{final}$ to
  $O(\varepsilon_\text{DRAG}^6)$. Useful in the regime where DRAG-1 has saturated
  (i.e., $T_\text{gate}|\alpha| \gtrsim 4$). Implementation cost: one
  additional analytic envelope and a real-time frequency modulation.
- **Half-DRAG (Lucero et al. 2010).** Heuristic $\beta = 1/2$ pulse used
  when the dominant calibration metric is phase error, not leakage.
  v0's $\beta_\text{opt}$ scan subsumes this case.
- **R²D / multi-derivative DRAG (Li, Calarco, Motzoi, *PRX Quantum* **3**,
  030313, 2022).** Recursive cancellation of multi-photon
  $|0\rangle \leftrightarrow |2\rangle$ resonances; useful below
  $T_\text{gate} \sim 7$ ns where leading-order DRAG breaks.
- **FAST DRAG and HD DRAG (Hyyppä et al., arXiv:2402.17757, 2024).**
  Fourier-ansatz spectrum tuning and higher-derivative corrections;
  achieve leakage $< 3 \times 10^{-5}$ at $T_\text{gate} = 6.25$ ns on
  real hardware. Closed-form analytical envelopes whose Fourier
  spectrum is tuned to vanish at all relevant transition frequencies.
- **GRAPE / Krotov / numerical optimal control (Werninghaus et al.
  2021; Theis et al. 2018, §V).** Numerical pulse shaping when even FAST
  DRAG is insufficient. Bypasses the analytic DRAG construction
  entirely; computational cost scales with $T_\text{gate}/dt$.

> **Scope of Module 5a (v0).** DRAG-1 + sin²-windowed Gaussian envelope +
> calibrated $\beta$ scan. **AC-Stark detuning compensation (eq. 11.15)
> is *not* part of the v0 default**; it is exposed as an optional
> `enable_ac_stark` flag for v1.5 follow-on use, and the present
> framework reports its absence as an explicit Group D / model-limits
> diagnostic rather than as a v0 capability. Higher-order DRAG variants
> and numerical optimal control are explicitly out of scope for v0 and
> are listed as v1.5 / v2 forward pointers.

---

### 11.7 Decoherence during the pulse

**[v0 model — Lindblad during pulse, in line with the implementation
plan].** The primary v0 gate simulator (`gate_simulator.py`) propagates
the four-level Lindblad master equation

$$\dot\rho = -i[H_R(t), \rho] + \sum_\mu \mathcal D[L_\mu]\,\rho \qquad \text{(11.27)}$$

during the pulse, with collapse operators (cf. §5.2):

- **Qubit relaxation** — secular list of resolved adjacent-transition
  collapse operators, in line with §5.2:
  $$L_{j \to j-1} = \sqrt{j\,\gamma_{1,\text{eff}}}\,\lvert j-1\rangle\langle j\rvert, \qquad j = 1, \ldots, N-1,$$
  for the four-level Duffing model ($N = 4$). The single-operator
  shorthand "$\sqrt{\gamma_{1,\text{eff}}}\,\sigma^-$" is *not* used in
  v0 because $\sigma^-$ is ambiguous in the multilevel truncation
  (could mean $\lvert 0\rangle\langle 1\rvert$ alone, the oscillator
  lowering operator $b$, or the secular list above); the explicit list
  is the unambiguous choice and matches §5.2's secular treatment;
- **Pure dephasing** $L_{T_\varphi}$ — multilevel number-operator
  default $L_{T_\varphi}^\text{number} = \sqrt{2\gamma_\varphi}\,\hat n$
  with $\hat n = \sum_j j\,|j\rangle\langle j|$ the Duffing
  level-number operator (matching the §5.4 multilevel definition,
  eq. 5.10 — the |0⟩-|1⟩ coherence then decays at rate $\gamma_\varphi$
  exactly, not $\gamma_\varphi/4$ as the shorthand $\sqrt{\gamma_\varphi/2}\,\hat n$
  would have given);
  the two-level Pauli shorthand $L = \sqrt{\gamma_\varphi/2}\,\sigma_z$
  (with $\sigma_z$ having eigenvalues $\pm 1$) is also exposed as a
  `dephasing_op="sigma_z"` option for cross-check against textbook
  qubit results, since both forms give the same $\gamma_\varphi$ on the
  $|0\rangle$-$|1\rangle$ coherence by direct calculation;
- **Thermal excitation** — secular list of upward transitions
  $L^{(\text{th})}_{j-1 \to j} = \sqrt{j\,\bar n_\text{th}\,\gamma_{1,\text{eff}}}\,\lvert j\rangle\langle j-1\rvert$,
  $j = 1, \ldots, N-1$ (off by default with $\bar n_\text{th} = 0$;
  can be enabled in the reference stress-test configuration with
  $\bar n_\text{th} = 0.05$ for thermal-population sensitivity studies);
- **Purcell** is incorporated by using $\gamma_{1,\text{eff}} = \gamma_{1,\text{intrinsic}} + \gamma_\text{Purcell}$ in the qubit-relaxation operators above, since the qubit-only gate simulator does *not* include the resonator and therefore Purcell does not emerge naturally as it does in the full JC+$\kappa$ simulator of §5.3 (see Convention 21 below).

A diagnostic `decoherence_off` mode is also exposed for unitary-only
debug runs, but Lindblad is the v0 default for all reported metrics.

**[Diagnostic-only — separable $T_{1,\text{eff}}$ decay-event scale].**
The following back-of-envelope is retained as an analytic sanity check,
*not* as the v0 model. Per Convention 21, in the **qubit-only** Module 5a Lindblad
simulator the cavity has been integrated out, so the relevant rate is
$\gamma_{1,\text{eff}} = \gamma_{1,\text{intrinsic}} + \gamma_\text{Purcell}$.
At the synthetic seed, $\gamma_{1,\text{intrinsic}} \approx 3.33 \times 10^4\,\text{s}^{-1}$
($T_{1,\text{intrinsic}} = 30\,\mu$s) and $\gamma_\text{Purcell} \approx 6.2 \times 10^4\,\text{s}^{-1}$
($T_\text{Purcell} \approx 16.1\,\mu$s), giving $T_{1,\text{eff}} \approx 10.5\,\mu$s
and so for $T_\text{gate} \le 20$ ns:

$$T_\text{gate}/T_{1,\text{eff}} \approx 1.9 \times 10^{-3} \quad (T_\text{gate} = 20\,\text{ns}), \qquad T_\text{gate}/T_{1,\text{eff}} \approx 9.5 \times 10^{-4} \quad (T_\text{gate} = 10\,\text{ns}) \qquad \text{(11.28)}$$

so a separable estimate of the $T_{1,\text{eff}}$-induced **decay-event scale** is

$$\varepsilon_X^{(T_{1,\text{eff}}\text{-scale, est.})} \approx 1 - e^{-T_\text{gate}/T_{1,\text{eff}}} \qquad \text{[Diagnostic decay-event scale, not the v0 model]} \qquad \text{(11.29)}$$

> **[Important — this is *not* an X-gate-error floor].** The quantity
> $1 - e^{-T_\text{gate}/T_{1,\text{eff}}}$ is a *decay-event probability
> scale*, not the probe-set X-gate error or the average gate infidelity.
> The actual incoherent contribution to $\varepsilon_X$ depends on the
> input state, the target operation (X), and the chosen error metric
> (probe-set mean vs average gate infidelity vs other), with a state-
> and metric-dependent prefactor that is *not* unity. Eq. (11.29) is
> therefore reported only as an **order-of-magnitude scale estimate**;
> the actual incoherent contribution to $\varepsilon_X$ must be
> extracted from the Lindblad-propagated probe-set metric returned by
> `gate_simulator.py`. v0 reports both numbers in the calibration
> output for transparency.

For comparing against the actual Lindblad-propagated $\varepsilon_X$, this
scale estimate is informative *as a sanity bound*: if the simulated
$\varepsilon_X$ is much larger than (11.29), coherent residuals
dominate; if comparable, incoherent ($T_{1,\text{eff}}$-scale)
contributions cannot be ignored — but the comparison cannot be reduced
to "is the gate $T_1$-limited?" without specifying the input state and
metric (see boxed note above). The cross-terms (e.g., $T_1$ during a
high-amplitude Stark virtual occupation of $|2\rangle$) are not
captured by (11.29) and *are* captured by (11.27); v0 reports both
numbers in the calibration
output for transparency.

> **Convention 21 (DeviceConfig.T_1 is intrinsic; effective T_1 is
> derived).** Throughout this framework, `DeviceConfig.T_1` and the seed
> value $T_1 = 30\,\mu$s refer to the **intrinsic** qubit relaxation
> $1/\gamma_{1,\text{intrinsic}}$ (§5.3). The effective Lindblad-rate
> $\gamma_{1,\text{eff}} = \gamma_{1,\text{intrinsic}} + \gamma_\text{Purcell}$
> is computed *internally* from device parameters as needed and only in
> contexts where the cavity has been integrated out (Module 5a's
> qubit-only Lindblad simulator; Module 5b's polaron-eliminated reset
> model). For the full JC+$\kappa$ Module 1 simulator, the bare
> $\gamma_{1,\text{intrinsic}}$ is used as the $\sigma^-$ collapse rate
> and Purcell emerges naturally — adding $\gamma_\text{Purcell}$ a
> second time would double-count, per the §5.3 boxed warning. If the
> seed value were ever interpreted as a *measured* $T_1$ (which in
> experiment includes Purcell), the convention conversion would be
> $\gamma_{1,\text{intrinsic}} = 1/T_1^\text{measured} - \gamma_\text{Purcell}$
> — but v0 takes $T_1 = 30\,\mu$s as intrinsic by convention, with
> Marxer-reported 86/102 μs cited as a separate calibration target.

---

### 11.8 Diagnostic comparison: theory vs. numerical findings

The user's empirical data are presented as numerical validation that the
simulation is operating in the DRAG-1 regime predicted by Eqs. (11.11)-(11.26),
with the following correspondence:

- **Final-leakage suppression at $T_\text{gate} = 12$-$15$ ns** — $\sim (T_\text{gate} \cdot \|\alpha\|)^2$ in DRAG-1-dominated regime, eq. (11.26)
  - *Prototype-notebook / planned v0 target:* $26.8\times \to 92.1\times$ ($12 \to 15$ ns)
  - *Interpretation:* DRAG-1 working as expected; quadratic-in-$T_\text{gate}$ scaling consistent with (11.26)
- **Final-leakage suppression at $T_\text{gate} = 20$ ns** — Mechanism not yet diagnosed; floor-crossover, AC-Stark, grid-resolution, and calibration-objective tradeoff are all candidates (see §11.5)
  - *Prototype-notebook / planned v0 target:* $1.4\times$
  - *Interpretation:* **Open empirical finding** — to be diagnosed from absolute leakage, $\beta_\text{opt}(T)$, grid-refinement, and phase-vs-leakage metric separation before a mechanism is asserted
- **Peak-leakage suppression cap** — $O(1)-O(\text{few})$ regardless of $\beta$; structural, eq. (11.23)-(11.25)
  - *Prototype-notebook / planned v0 target:* $3.1\times$ (10 ns) $\to 1.7\times$ (15 ns) $\to 1.3\times$ (20 ns)
  - *Interpretation:* Model-limits FINDING: DRAG-1 cancels endpoint, not transient, leakage. Decreasing trend with $T_\text{gate}$ tracks the shrinking bare-leakage baseline.
- **$\beta_\text{opt}$ for sin²-Gaussian, $T_\text{gate} = 10$ ns** — $\neq 1$ (Gaussian DRAG-1 value); shape-dependent shift plausible per the §11.4 prose discussion of envelope-induced β-shift (no closed-form analytic prediction)
  - *Prototype-notebook / planned v0 target:* $\approx 2.2$ (TBD, prototype value pending v0 calibration scan)
  - *Interpretation:* Empirical calibration result; deviation from $\beta = 1$ is plausible given the shaped envelope and the metric-dependent objective, not a predicted value
- **Probe-set X-gate error $\varepsilon_X^\text{ref}(20\,\text{ns})$** — Coherent residual + $T_\text{gate}/T_{1,\text{eff}}$ decay-event scale (eq. 11.29) — *not* a strict floor
  - *Prototype-notebook / planned v0 target:* **TBD** (prototype expected $\sim 10^{-3}$, $T_{1,\text{eff}}$ decay-event scale $\approx 1.9 \times 10^{-3}$ at $T_{1,\text{eff}} \approx 10.5\,\mu$s)
  - *Interpretation:* $T_{1,\text{eff}}$ decay-event scale at 20 ns is the order-of-magnitude bound (Convention 21: cavity integrated out, Purcell folded in); coherent residual to be measured by v0 calibration. **Note (cross-ref §11.7 boxed warning + §12.3 classical-flip-error assumption):** $\varepsilon_X$ here is the probe-set average gate error, which is *not* the same as the classical flip-failure probability that Module 5b consumes via $\varepsilon_X = 1 - F_X$ (§12.3).

The expected comparison structure is qualitative for leakage-shape
trends (saturation, monotonic decrease with $T_\text{gate}$) and
quantitative only for directly computed diagnostic quantities (e.g., the
$T_\text{gate}/T_{1,\text{eff}}$ decay-event scale) once the v0 in-repo
calibration run exists and the prototype-notebook values listed in §11.0
are reproduced under the **[Status: prototype, pending in-repo
reproduction]** label.

---

### 11.9 Validation and convergence tests

**Truncation convergence (V3, §11.9).** The default truncation
$N_\text{transmon} = 4$ levels is verified by re-running the simulation at
$N = 5$ and checking that final populations of $|2\rangle$ and $|3\rangle$
change by less than $10^{-4}$ (relative). Beyond this convergence, the
buffer level $|3\rangle$ should carry $\le 10^{-2}$ of the $|2\rangle$
population at all times — direct $|0\rangle \to |3\rangle$ DRAG-cancelled
coupling is structurally absent in DRAG-1, so any large $|3\rangle$
population is a numerical artifact (most likely insufficient dt).

**Endpoint smoothness (V7, §11.9).** Both $\Omega_x$ and $\Omega_y$ must
vanish at $t = 0$ and $t = T_\text{gate}$. The sin²-window guarantees
$\Omega_x(0) = \Omega_x(T_\text{gate}) = 0$ and
$\dot\Omega_x(0) = \dot\Omega_x(T_\text{gate}) = 0$ analytically — this
should be checked numerically against the implementation's analytic
derivative output.

**$\pi$-pulse normalization.** The integral
$\int_0^{T_\text{gate}} \Omega_x(t)\, dt = \pi$ should hold to
$10^{-6}$ relative tolerance. v0 enforces this via numerical integration
in `pulses.py::sin2_windowed_gaussian_amplitude`.

**Sign-flip test (V6, §11.9).** Replacing $\beta \to -\beta$ must
*increase* the final leakage relative to $+\beta_\text{opt}$ by a
clearly resolved amount (well outside numerical noise / shot-noise
bands) — the actual factor is envelope- and metric-dependent and
should not be predicted in closed form here. A failure of V6 typically
indicates a sign error in either $\alpha$ or $\beta$.

**Anharmonicity scaling (V5a, V5b, §11.9).** Leakage without DRAG should
scale as $|\alpha|^{-2}$ in the perturbative regime $\bar\Omega/|\alpha| \ll 1$ (dimensionless small parameter; spectral resolution requires the separate condition $|\alpha|\,T_\text{gate} \gtrsim 1$ for the leakage band to be distinguishable from the qubit-driving band over the pulse duration),
per (11.12). V5a (blocking) requires the fitted log-log slope to be
*negative* across the swept range; V5b (diagnostic) requires the slope
to be $-2 \pm 0.5$ in the perturbative sub-range only.

---

### 11.10 Convention warnings collected for §11

> **Convention 17 (sign of $\alpha$).** $\alpha < 0$ throughout. The
> sign is fixed by the transmon's negative anharmonicity
> $\omega_{12} - \omega_{01} \approx -E_C < 0$ (cf. §2.4). Code that
> reads $\alpha$ from device parameters must respect this sign.

> **Convention 18 (DRAG quadrature sign).** $\Omega_y(t) = -\beta\, \dot\Omega_x(t)/\alpha$
> with $\alpha < 0$. With this convention, $\beta > 0$ corresponds to
> the standard DRAG (suppressing leakage), and a sign-flip test V6 must
> show *increased* leakage. In some references (e.g., Krantz et al. 2019
> §III.D) the sign of $\alpha$ is absorbed into $\beta$, so a positive
> $\beta$ in their convention may correspond to a negative $\beta$ in
> ours; cross-check carefully when transcribing parameter values.

> **Convention 19 (rotating frame at $\omega_d = \omega_q$).** v0 sets
> the drive carrier frequency exactly at the qubit transition,
> $\omega_d = \omega_q$, with the AC-Stark detuning correction
> (11.15) absorbed phenomenologically into the calibrated $\beta$. v1.5
> will add an explicit time-dependent detuning $\delta_d(t)$ as in
> (11.16).

---

### 11.11 Script connection table for §11

| Equation | Script | Function | Test |
|---|---|---|---|
| (11.4) Rotating-frame $H_R$ | `stage_06_readout/control/gate_simulator.py` | `build_drift_hamiltonian()` | `test_h_r_diagonal_alpha()` |
| (11.17)-(11.18) sin²-Gaussian envelope | `stage_06_readout/control/pulses.py` | `sin2_windowed_gaussian()`, `sin2_windowed_gaussian_amplitude()` | `test_endpoint_zeros`, `test_pi_pulse_normalization` |
| (11.19) DRAG quadrature | `stage_06_readout/control/pulses.py` | `sin2_windowed_gaussian_derivative()` × `(-β/α)` | `test_drag_endpoint_zero`, `test_drag_sign_convention` (V6) |
| (11.15) AC-Stark detuning (optional) | `stage_06_readout/control/pulses.py` | `ac_stark_detuning()` | optional v0 |
| (11.23)-(11.25) Peak-leakage diagnostic | `stage_06_readout/analysis/gate_metrics.py` | `compute_p2_peak()` | (V2b diagnostic) |
| (11.26) Final-leakage suppression | `stage_06_readout/analysis/gate_metrics.py` | `compute_p2_final()`, `leakage_suppression_ratio()` | V2a (blocking) |
| $\beta_\text{opt}$ scan (per §11.4 prose discussion) | `stage_06_readout/control/drag_calibration.py` | `calibrate_drag_beta()` | `test_beta_opt_in_grid_interior` |
| Closed-form $\varepsilon_X$ probe set | `stage_06_readout/analysis/gate_metrics.py` | `xgate_probe_set_error()` | V2a, V4 |
| (11.29) Diagnostic $T_{1,\text{eff}}$ decay-event scale (analytic estimate, not v0 model) | `stage_06_readout/analysis/gate_metrics.py` | `estimate_t1_decay_event_scale()` | informational |

---

### 11.12 References for §11

- **Motzoi, Gambetta, Rebentrost, Wilhelm**, *Phys. Rev. Lett.* **103**, 110501 (2009) [arXiv:0901.0534] — original DRAG-1 derivation.
- **Gambetta, Motzoi, Merkel, Wilhelm**, *Phys. Rev. A* **83**, 012308 (2011) [arXiv:1011.1949] — canonical higher-order treatment, family of DRAG solutions, AC-Stark detuning.
- **Motzoi, Wilhelm**, *Phys. Rev. A* **88**, 062318 (2013) — derivative-based transition suppression and spectral selectivity.
- **Theis, Motzoi, Machnes, Wilhelm**, *EPL* **123**, 60001 (2018) [arXiv:1809.04919] — DRAG ten-year retrospective; spectral-notch viewpoint.
- **Chow, DiCarlo, Gambetta, Motzoi, Frunzio, Girvin, Schoelkopf**, *Phys. Rev. A* **82**, 040305(R) (2010) [arXiv:1005.1279] — first transmon DRAG demonstration.
- **Lucero, Kelly, Bialczak et al.**, *Phys. Rev. A* **82**, 042339 (2010) [arXiv:1007.1690] — half-DRAG, APE pulses, phase-error metric.
- **Werninghaus, Egger, Roy, Machnes, Wilhelm, Filipp**, *npj QI* **7**, 14 (2021) [arXiv:2003.05952] — closed-loop optimization; DRAG limits at 4-ns gates.
- **Khani, Gambetta, Motzoi, Wilhelm**, *New J. Phys.* **11**, 113006 (2009) [arXiv:0909.4788] — Fock-state generation; charge-basis vs Duffing scaling.
- **Hyyppä et al.**, arXiv:2402.17757 (2024) — FAST DRAG / HD DRAG analytical envelopes; sub-7-ns gates.
- **Li, Calarco, Motzoi**, *PRX Quantum* **3**, 030313 (2022) — non-perturbative analytical diagonalization for multi-photon resonances.
- **Krantz, Kjaergaard, Yan, Orlando, Gustavsson, Oliver**, *Appl. Phys. Rev.* **6**, 021318 (2019) [arXiv:1904.06560] — engineering review.
- **Setiawan, Groszkowski, Motzoi et al.**, *Phys. Rev. X* **7**, 011021 (2017) — Magnus-based approach for non-adiabatic errors.

## 12. Module 5b — Semiclassical Active Reset (SME → Bad-Cavity Elimination → Direct Jump)

### 12.0 Recruiter-readable framing

**Hardware problem.** Active reset — using a measurement to project a qubit
into a known state, followed (optionally) by a conditional $\pi$-pulse — is
the standard technique for fast initialization of superconducting qubits,
replacing slow passive thermalization (Riste, van Leeuwen, Ku, Lehnert,
DiCarlo, *Phys. Rev. Lett.* **109**, 050507, 2012). For algorithmic and
QEC applications, reset must run in $\sim 1\,\mu$s and reach residual
excitation $p_e' \lesssim 1\%$, which is impossible to achieve passively
on devices with $T_1 = 30\,\mu$s. Faithfully simulating active reset would
require propagating the *full circuit-QED stochastic master equation*
(SME) at every time step — qubit, resonator, drive, dispersive coupling
$\chi$, cavity decay $\kappa$, qubit decay $\gamma_1$, heterodyne
measurement record — which is computationally prohibitive in a
benchmarking loop over many shots and many operating points.

**Approximation made by Module 5b.** Starting from the full Wiseman-Milburn
/ Gambetta-Blais-Boissonneault-Houck-Schuster-Girvin SME for heterodyne
dispersive readout (Gambetta et al., *Phys. Rev. A* **77**, 012112, 2008
[arXiv:0709.4264]), we eliminate the cavity adiabatically in the
*bad-cavity limit* $\kappa \gg \chi, \gamma_1, \gamma_\varphi$ (Carmichael,
*Statistical Methods in Quantum Optics 2*, ch. 8 and §12.1.3, Springer 2008)
and replace the resulting effective qubit SME with two independent
stochastic processes:

(i) a **Poisson direct-jump** for $|1\rangle \to |0\rangle$ at rate
$\Gamma_1 \equiv \gamma_{1,\text{eff}} = \gamma_{1,\text{intrinsic}} + \gamma_\text{Purcell}$ (Convention 21; the polaron-eliminated qubit-only model of §12.2 is precisely the regime where Purcell must be added explicitly because the cavity has been integrated out — *not* the regime where Purcell is implicit in `DeviceConfig.T_1`), generating zero
or one jump events per measurement window with probability
$1 - e^{-\Gamma_1 \tau_\text{meas}}$; and

(ii) a **Gaussian-noise-augmented IQ measurement record** with
state-dependent mean given by the Module 1 pointer-response helper and
integrated noise $\sigma_\parallel(\tau_\text{meas}) \propto \sqrt{\tau_\text{meas}}$
(cf. §6.3a).

The model is **controlled in scope** (per the framing of §6.2 and §7.3a):
we explicitly enumerate the omitted physics — finite-cavity transient
response, heterodyne back-action induced dephasing on coherences, thermal
photon excitation, coherent reset $\pi$-pulse errors, multi-photon
non-QND processes (measurement-induced state transitions à la
Sank/Khezri/Cohen/Dumas), leakage to $|2\rangle$ during readout — and
identify each with its small parameter. We do *not* claim formal
operator-norm error bounds: such bounds would require a full
Nakajima-Zwanzig analysis with explicit estimates on the irrelevant-coupling
spectral function, which is out of scope of v0. What we *do* claim is
that under the stated regime conditions $O(\chi/\kappa) \ll 1$,
$O(\gamma_1 \tau_\text{meas}) \ll 1$, $O(\bar n / n_\text{crit}) \ll 1$,
and $O(\text{SNR}) \gg 1$ (heterodyne signal-to-noise), the leading
omitted terms are explicit and small.

**Planned validation targets and prototype-notebook expectations.** As
in §11.0, the claims below are **[Status: prototype notebook /
expected behavior, not yet reproduced in the v0 repo]**. They are
*reported as planned validation targets* the in-repo Module 5b must
demonstrate:

(i) The active-reset residual on the short-$T_1$ demo device is
*expected* to be dominated by *joint readout-decay structure* rather
than by conditional-X gate error; gate failure should contribute
only a small fraction (prototype notebook: $\sim 1\%$) of the
residual at the selected operating point **[Status: prototype-notebook
result, pending in-repo reproduction]**.

(ii) The joint matrix $P(s_f, m | s_i)$ should exhibit a non-zero
$P(s_f = g, m = 0 | s_i = e)$ entry in the regime $\tau_\text{meas}/T_{1,\text{eff}} \in [0.1, 2.0]$
— the qubit-decayed-AND-measurement-read-ground case that the plain
confusion matrix conflates with reset failure. This is a **planned
validation target** (V7 of §12.7) and will be reported quantitatively
once the in-repo simulation runs.

(iii) Module 5b's output should reduce to the §7.3a one-jump
$T_1$ mixture model in the **measurement-only, no-feedback limit**
(same readout drive applied, but the conditional X-pulse / reset
feedback disabled, with the same one-jump $T_1$ trajectory
construction). The reduction is **exact within the shared
pointer-response / direct-jump reduced model**, not exact relative to
the full JC + heterodyne SME (which the reduced model itself
approximates by adiabatic elimination). This is a **planned
consistency check on the derivation** (V4a of §12.7); the identity
is structural and is expected to hold to numerical precision within
the reduced-model regime.

---

### 12.1 Starting point: full circuit-QED SME for heterodyne readout

**[Exact within model — full circuit-QED Hamiltonian].** The lab-frame
Hamiltonian for a transmon-resonator system with drive (cf. §3.1-3.4):

$$H_\text{sys} = \omega_r\, a^\dagger a + \tfrac{\omega_q}{2}\,\sigma_z + g\,(a^\dagger \sigma^- + a\,\sigma^+) + \tfrac{1}{2}\bigl[\varepsilon(t)\,e^{-i\omega_d t}\,a^\dagger + \text{h.c.}\bigr] \qquad \text{(12.1)}$$

> **[Drive normalization warning].** The $\varepsilon(t)$ in (12.1)
> uses the drive normalization $\tfrac{1}{2}[\varepsilon(t)\,e^{-i\omega_d t}\,a^\dagger + \text{h.c.}]$,
> which is **not necessarily the same normalization** as the §3.3
> Hamiltonian-frame drive term
> $\varepsilon(t)(a\,e^{i\omega_d t} + a^\dagger\,e^{-i\omega_d t})$
> (which differs by a factor of 2 in the implied $\varepsilon$). Module 5b
> defers the choice of drive normalization to Module 1's
> `pointer_response()` helper (§12.2 deferral box). Downstream §12 prose
> never substitutes a specific factor and uses the symbolic
> $\alpha_g^\text{ss}, \alpha_e^\text{ss}$ outputs of Module 1 instead.

with the four-channel decoherence model of §5.2:

$$\dot\rho = -i[H_\text{sys}, \rho] + \kappa\,\mathcal D[a]\rho + \gamma_1\,\mathcal D[\sigma^-]\rho + \tfrac{\gamma_\varphi}{2}\,\mathcal D[\sigma_z]\rho \qquad \text{(12.2)}$$

In the dispersive regime $|\Delta| \gg g$ ($\Delta = \omega_q - \omega_r$;
cf. §4), the Schrieffer-Wolff transformation yields (§4.4):

$$H_\text{disp} = (\omega_r + \chi\sigma_z)\,a^\dagger a + \tfrac{\omega_q'}{2}\sigma_z + \text{drive} \qquad \text{(12.3)}$$

with the Lamb-shifted $\omega_q'$ and the multilevel $\chi$ that
incorporates the $|2\rangle$ admixture (Koch et al. 2007, eq. 3.4;
§4.4 of this framework). The dispersive Lindblad master equation is then
(12.2) with $H_\text{sys} \to H_\text{disp}$.

> **[Notation].** Throughout §12 we use $\sigma_z = |e\rangle\langle e| - |g\rangle\langle g|$
> with the convention $|g\rangle = |0\rangle$, $|e\rangle = |1\rangle$
> (qubit subspace truncation, ignoring leakage to $|2\rangle$ during
> readout — see omissions table §12.5).

**[Heterodyne unraveling].** Continuous heterodyne detection at efficiency
$\eta$ of the cavity output unravels the cavity-decay term as a
continuous measurement. The polaron-transformed circuit-QED trajectory
framework comes from Gambetta, Blais, Boissonneault, Houck, Schuster,
Girvin, *Phys. Rev. A* **77**, 012112 (2008) [arXiv:0709.4264], who
develop the construction for *homodyne* unraveling of the resonator
output field. The two-Wiener-process *heterodyne* extension used below
follows the standard homodyne/heterodyne unraveling conventions in
Wiseman & Milburn, *Quantum Measurement and Control*, Cambridge UP
(2010), Ch. 4-5. The Itô-form heterodyne SME is therefore:

$$\begin{aligned}
d\rho_c =\;& \mathcal L\rho_c\, dt \\
& + \sqrt{\eta\kappa/2}\,\bigl(a\,\rho_c + \rho_c\,a^\dagger - \langle a + a^\dagger\rangle\,\rho_c\bigr)\,dW_1 \\
& + \sqrt{\eta\kappa/2}\,\bigl(-i a\,\rho_c + i\,\rho_c\,a^\dagger - \langle -ia + ia^\dagger\rangle\,\rho_c\bigr)\,dW_2
\end{aligned} \qquad \text{(12.4)}$$

with $\mathcal L\rho_c$ the master-equation drift (12.2) and two independent
Wiener processes $dW_{1,2}$ corresponding to the I and Q quadratures of
the local-oscillator-mixed photocurrent. The instantaneous measurement
record is

$$\frac{dM_I}{dt} = \sqrt{\eta\kappa/2}\,\langle a + a^\dagger\rangle + \xi_I(t), \quad \langle \xi_I(t)\xi_I(t')\rangle = \delta(t-t') \qquad \text{(12.5)}$$

and analogously for $Q$. The "joint matrix" $P(s_f, m | s_i)$ referenced
in the user's setup is, at this exact level, the conditional probability
density obtained by propagating Eq. (12.4) from $t = 0$ (state $s_i$) to
$t = \tau_\text{meas}$ (final-state projector $s_f$ and integrated
measurement-record $m$).

**[Approximation — leading correction not retained at this level].**
Khezri, Mlinar, Dressel, Korotkov, *Phys. Rev. A* **94**, 012347 (2016)
[arXiv:1606.04204] show that beyond the leading dispersive picture,
ringing-up the resonator produces *dressed coherent / squeezed states* in
the joint-eigenbasis ladder, with $O(\chi/(2\chi + \kappa))$ corrections
to the simple coherent-state ansatz used below. We quote this as the
leading correction *not retained* in v0. Khezri, Dressel, Korotkov,
*Phys. Rev. A* **92**, 052306 (2015) further identifies measurement
errors from coupling to detuned neighbors — an $O(g^2/\Delta_\text{nbr}^2)$
correction also not retained (single-qubit v0).

---

### 12.2 Bad-cavity adiabatic elimination

**[Approximation — bad-cavity limit].** In the limit
$\kappa \gg \chi, \gamma_1, \gamma_\varphi$, drive ramp rates, the cavity
field adiabatically tracks the qubit state on a timescale $1/\kappa$, and
we can integrate it out. The standard treatment is given in Carmichael,
*SMQO 2*, §12.1.3 ("Adiabatic Elimination in the Master Equation") and
Ch. 13 (Cavity QED I), and in Gambetta et al. 2008, §III via a polaron
transformation. We sketch the derivation here.

**[Step (i) — Polaron / displaced-frame transformation].** Apply the
qubit-state-conditional displacement

$$D(\sigma_z) = \exp\!\bigl[\alpha_g(t)\,a^\dagger - \alpha_g^*(t)\,a\bigr]\,P_g + \exp\!\bigl[\alpha_e(t)\,a^\dagger - \alpha_e^*(t)\,a\bigr]\,P_e \qquad \text{(12.6)}$$

where $\alpha_s(t)$ for $s \in \{g, e\}$ is the qubit-state-conditional
coherent amplitude. In the displaced frame, the cavity is in
vacuum if it is following the conditional pointer state exactly. The
$\alpha_s(t)$ obey the classical Lindblad-driven cavity equation,
schematically

$$\dot\alpha_s = -\biggl(\frac{\kappa}{2} + i\,\delta_s\biggr)\,\alpha_s + (\text{drive term}), \qquad \delta_s = s\,\chi \quad \text{for}\ s = \pm 1\ (g, e) \qquad \text{(12.7)}$$

> **Convention deferral — Module 1 is the implementation authority.**
> Equation (12.7) is written schematically with a generic
> "(drive term)". The exact normalization of the drive term — i.e.,
> whether it is $-i\varepsilon(t)$, $-i\varepsilon(t)/2$, or
> $-i\sqrt\kappa\,\varepsilon(t)/2$ in the convention of (12.1)–(12.4) —
> depends on the input/output theory normalization fixed by Module 1's
> pointer-response helper (`physics/pointer_response.py`). **Module 5b
> consumes that helper directly and does not re-derive it locally.**
> Different references differ by factors that do not affect the *physics
> of the readout*, only the relationship between the engineering input
> $\varepsilon(t)$ and the cavity steady-state amplitude. To avoid a
> factor-of-two convention conflict between this section and Module 1,
> downstream formulas in this chapter (eqs. 12.8–12.10) are written
> conditional on Module 1's $\alpha_g^\text{ss}, \alpha_e^\text{ss}$
> outputs rather than on $\varepsilon$ directly.

> **[Authoritative `pointer_response()` API contract].** The Module 1
> helper that all of §12 consumes has the following formal signature:
>
> ```python
> def pointer_response(
>     state: Literal["g", "e"],
>     t_grid: np.ndarray,         # shape (N,), monotonic, units: seconds
>     kappa: float,               # rad/s; cavity linewidth
>     chi: float,                 # rad/s; dispersive shift (signed; cf. Convention 5)
>     drive: Callable[[float], complex],   # ε(t); units rad/s
>     alpha0: complex = 0.0,      # initial cavity amplitude (default vacuum)
>     normalization: Literal["intracavity", "output_field"] = "intracavity",
> ) -> np.ndarray:                # complex amplitude in the requested normalization (see `normalization` arg)
> ```
>
> The implementation-authoritative ODE is (in Module 1's chosen
> normalization, fixed in `physics/pointer_response.py` and **not**
> in §12):
> $$\dot\alpha(t) = -\bigl(\kappa/2 + i\delta_s\bigr)\,\alpha(t) + (\text{drive term per Module 1 normalization}), \qquad \alpha(0) = \alpha_0,$$
> with $\delta_s = +\chi$ for $s = e$ and $\delta_s = -\chi$ for $s = g$
> under Convention 5. The `normalization` parameter controls whether the
> returned amplitude is the *intracavity* field $\alpha(t)$ or the
> *output field* $\alpha_\text{out}(t) = \sqrt\kappa\,\alpha(t)$ (the
> convention used by §6.3 for the IQ measurement record). Module 5b
> calls the helper with `normalization="output_field"` to align with
> §6.3a's IQ-noise convention; the conversion factor $\sqrt\kappa$ does
> not enter §12's downstream algebra because all formulas (eqs. 12.8 →)
> are expressed in terms of $\alpha_g^\text{ss}, \alpha_e^\text{ss}$
> directly. An out-of-band v0 unit-test verifies that Module 1's
> steady-state output of `pointer_response()` matches the analytic
> $-i\varepsilon/(\kappa/2 + i\delta_s)$ form (with whichever drive
> normalization Module 1 has fixed), to numerical precision.

**[Step (ii) — Steady-state cavity tracking; consumes Module 1].** For
constant drive amplitude and $t \gg 1/\kappa$, define the steady-state
amplitudes

$$\alpha_{s,\text{out}}^\text{ss} \equiv \lim_{t \to \infty} \alpha_s(t)\bigg|_{\dot\varepsilon = 0} = \texttt{pointer\_response}(\texttt{state}{=}s,\, \texttt{t\_grid},\, \kappa,\, \chi,\, \texttt{drive}{=}\varepsilon(t),\, \texttt{normalization}{=}\texttt{"output\_field"})\bigg|_{t \to \infty} \qquad \text{(12.8)}$$

returned by Module 1 in the **output-field normalization** (consumed
directly by §12.3 reset IQ sampling). The qubit-state-discriminating
quantity in the IQ
plane is the *separation* of the two pointers,
$|\alpha_{g,\text{out}}^\text{ss} - \alpha_{e,\text{out}}^\text{ss}|$,
also returned by Module 1. The corresponding **intracavity** amplitudes
$\alpha_{s,\text{intra}}^\text{ss} = \alpha_{s,\text{out}}^\text{ss}/\sqrt{\kappa_\text{ext}}$
are obtained by a separate call to `pointer_response(..., normalization="intracavity")`
(consumed by `meas_induced_dephasing()` for $\Gamma_d^\text{meas}$ in
(12.12) and by the intracavity photon number $\bar n$ below).
For the synthetic seed $\chi/2\pi = -0.385$ MHz, $\kappa/2\pi = 5$ MHz,
$|\chi|/\kappa \approx 0.077$ — comfortably in the bad-cavity regime
$|\chi|/\kappa \ll 1$, justifying the leading-order approximation. The
characteristic **intracavity** photon number is
$\bar n \equiv |\alpha_{e,\text{intra}}^\text{ss}|^2$
(intracavity convention, dimensionless photon number; the symbol $\bar n$
also appears in §3.4 for the same physical quantity and in §5.6 / §11.7
for thermal occupation, disambiguated by the §16 notation table) — or
the average $(|\alpha_{g,\text{intra}}^\text{ss}|^2 + |\alpha_{e,\text{intra}}^\text{ss}|^2)/2$;
either choice is fine for the leading-order $\Gamma_d^\text{meas}$
formula below since they agree to $O((\chi/\kappa)^2)$.

> **[Normalization conflict — explicit disambiguation between (12.12)
> and reset IQ sampling].** Eq. (12.12) is the standard
> **intracavity-amplitude** formula: it consumes $\alpha_g^\text{ss}, \alpha_e^\text{ss}$
> in the *intracavity* normalization, so $\bar n = |\alpha_e^\text{ss}|^2$
> directly equals the dimensionless intracavity photon number. The reset
> IQ sampling of §12.3, however, consumes $\alpha_g^\text{ss}, \alpha_e^\text{ss}$
> in the **output-field** normalization (§6.5; matches $z_\text{out}$
> with units of $\sqrt{\text{photons/s}}$ when $\kappa_\text{ext} = \kappa$).
> Since `pointer_response()` exposes a `normalization` parameter, the
> two §12 consumers must request *different* normalizations of the same
> Module 1 helper:
>
> - `meas_induced_dephasing()` (§12.9 mapping) calls
>   `pointer_response(..., normalization="intracavity")` and substitutes
>   $|\alpha_e^\text{ss,intra}|^2$ as $\bar n$ in (12.12).
> - Reset IQ sampling (§12.3 step 3) calls
>   `pointer_response(..., normalization="output_field")` and consumes
>   $|\alpha_g^\text{ss,out} - \alpha_e^\text{ss,out}|\cdot\tau_\text{meas}$
>   as the cluster separation, with `IQNoiseParams` providing the
>   matched output-field noise variance.
>
> The two are related by
> $\alpha_{s,\text{intra}} = \alpha_{s,\text{out}} / \sqrt{\kappa_\text{ext}}$;
> if downstream code receives output-field pointers and needs to evaluate
> (12.12), it must convert by dividing by $\sqrt{\kappa_\text{ext}}$
> first. v0 adds two unit tests for this:
> `test_pointer_response_intracavity_to_output_conversion()` (round-trip)
> and `test_meas_induced_dephasing_uses_intracavity_alpha()` (regression
> guard against a silent normalization mismatch). The §12.9 mapping table
> below tags each call site with the required normalization explicitly.

**[Step (iii) — Polaron-transformed effective qubit master equation].**
Substituting the displaced ansatz into (12.2) and projecting onto cavity
vacuum gives the effective qubit master equation (Gambetta et al. 2008,
eqs. 24-28; Boissonneault, Gambetta, Blais, *Phys. Rev. A* **79**, 013819,
2009 [arXiv:0810.1336]):

$$\dot\rho_q = -i\bigl[H_q^\text{eff}, \rho_q\bigr] + \gamma_{1,\text{eff}}\,\mathcal D[\sigma^-]\,\rho_q + \tfrac{\Gamma_d^\text{meas} + \gamma_\varphi}{2}\,\mathcal D[\sigma_z]\,\rho_q + \text{(diffusive heterodyne unraveling on}\ \sigma_z\text{)} \qquad \text{(12.11)}$$

where the **measurement-induced dephasing rate** is

$$\Gamma_d^\text{meas} = 2\chi\,\text{Im}\bigl[\alpha_g^\text{ss}\,(\alpha_e^\text{ss})^*\bigr] \approx \frac{8\chi^2\,\bar n}{\kappa} \qquad (\text{bad-cavity limit, } |\chi|/\kappa \ll 1) \qquad \text{(12.12)}$$

and the **effective qubit relaxation rate** is

$$\gamma_{1,\text{eff}} = \gamma_{1,\text{intrinsic}} + \gamma_\text{Purcell}, \qquad \gamma_\text{Purcell} = \biggl(\frac{g_{01}}{\Delta}\biggr)^2\,\kappa \qquad \text{(12.13)}$$

— consistent with §5.3, which derived the same expression as the emergent
dressed-state Purcell rate. **Bookkeeping discipline (Convention 21,
.** Throughout v0, $\gamma_{1,\text{intrinsic}} \equiv 1/T_1$ where
$T_1$ is `DeviceConfig.T_1` (the intrinsic value, with seed
$T_1 = 30\,\mu$s). The effective rate $\gamma_{1,\text{eff}}$ is computed
*internally* by Module 5b's reset model (and Module 5a's qubit-only
gate simulator) by adding $\gamma_\text{Purcell}$ derived from the
device parameters $g_{01}$, $\Delta$, $\kappa$. Critically,
$\gamma_\text{Purcell}$ is *not* a separate Lindblad channel in either
model: in the full JC+$\kappa$ Module 1 simulator (§5.3) it emerges from
the dynamics; in the polaron-eliminated effective qubit ME (12.11) the
cavity has been integrated out and $\gamma_\text{Purcell}$ is folded into
$\gamma_{1,\text{eff}}$ exactly once. Adding $\gamma_\text{Purcell}$ to
$\gamma_{1,\text{intrinsic}}$ in the *full JC+$\kappa$* simulator would
double-count, per the §5.3 boxed warning.

**[Step (iv) — Reduced heterodyne measurement record].** The heterodyne
measurement record (12.5), projected onto the optimal axis in the IQ
plane separating $\alpha_g^\text{ss}$ from $\alpha_e^\text{ss}$, separates
into a constant common-mode offset and a state-dependent component:

$$\frac{dM_\parallel}{dt} = \operatorname{Re}\!\biggl[\,\underbrace{\frac{\alpha_g^\text{ss} + \alpha_e^\text{ss}}{2}\,\hat n_\parallel^*}_{\text{common-mode (subtractable)}} \;-\; \underbrace{\frac{\alpha_g^\text{ss} - \alpha_e^\text{ss}}{2}\,\hat n_\parallel^*\,\langle\sigma_z\rangle}_{\text{state-dependent}}\,\biggr] + \xi_\parallel(t) \qquad \text{(12.14)}$$

(the $\operatorname{Re}[\cdot]$ is explicit because $M_\parallel$ is a
real scalar record; the complex factors $\hat n_\parallel^*$ and
$\alpha_s^\text{ss}$ inside are projected onto the real discrimination
axis).

where $\hat n_\parallel = (\alpha_g^\text{ss} - \alpha_e^\text{ss})/|\alpha_g^\text{ss} - \alpha_e^\text{ss}|$
is the unit vector along the discrimination axis (Re-projection convention
implicit) and $\langle\sigma_z\rangle$ uses the $\sigma_z = |e\rangle\langle e| - |g\rangle\langle g|$
convention from §12.1. Substituting the eigenvalues
$\langle\sigma_z\rangle_g = -1$ and $\langle\sigma_z\rangle_e = +1$:

- For trajectories in $|g\rangle$: $dM_\parallel/dt = +\tfrac{1}{2}|\alpha_g^\text{ss} - \alpha_e^\text{ss}|$ (positive offset);
- For trajectories in $|e\rangle$: $dM_\parallel/dt = -\tfrac{1}{2}|\alpha_g^\text{ss} - \alpha_e^\text{ss}|$ (negative offset);
- Cluster separation: $|M_\parallel^{(g)} - M_\parallel^{(e)}| = |\alpha_g^\text{ss} - \alpha_e^\text{ss}|\cdot \tau_\text{meas}$, as expected.

with white Gaussian noise $\xi_\parallel$ of spectral density $(2\eta\kappa)^{-1}$.
Integrating over the measurement window for a piecewise-constant
trajectory $s(t) \in \{g, e\}$:

$$M_\parallel(\tau_\text{meas}) = \operatorname{Re}\!\biggl[-\frac{\alpha_g^\text{ss} - \alpha_e^\text{ss}}{2}\,\hat n_\parallel^*\,\int_0^{\tau_\text{meas}}\!\langle\sigma_z\rangle_{s(t)}\,dt\biggr] + \int_0^{\tau_\text{meas}} \xi_\parallel(t)\,dt + \text{(common-mode constant)} \qquad \text{(12.15)}$$

The variance of the integrated noise is

$$\text{Var}\biggl[\int_0^{\tau_\text{meas}} \xi_\parallel(t)\,dt\biggr] = \frac{\tau_\text{meas}}{2\eta\kappa} \qquad \text{(12.16)}$$

**[Normalization bridge — raw record vs. amplitude record].** Eq. (12.5)
writes the *raw* heterodyne record with whitened noise
$\langle\xi_I(t)\xi_I(t')\rangle = \delta(t-t')$, so its integrated
white-noise variance over $\tau_\text{meas}$ is $\tau_\text{meas}$ — not
$\tau_\text{meas}/(2\eta\kappa)$. The result (12.16) corresponds to the
*amplitude-unit* record obtained by rescaling

$$M_\parallel^\text{amp}(\tau) \equiv \frac{1}{\sqrt{2\eta\kappa}}\,M_\parallel^\text{raw}(\tau), \qquad \text{(12.16a)}$$

so that the rescaled signal $\langle a + a^\dagger\rangle\cdot\tau$
coincides with the cavity-amplitude separation
$|\alpha_g^\text{ss} - \alpha_e^\text{ss}|\cdot\tau$ used by Module 1's
pointer-response helper. Under this rescaling, integrated noise variance
becomes $\tau_\text{meas}/(2\eta\kappa)$ — eq. (12.16) — which is the
form §6.3a (eq. 6.6a) postulated phenomenologically and which
`IQNoiseParams.sigma_for_integration_window()` exposes. The bridge from
raw SME notation to the §6 / Module 1 amplitude convention is therefore
the single rescaling factor $\sqrt{2\eta\kappa}$. Implementations must
not mix the two: the §12 chapter henceforth uses the amplitude-record
convention everywhere downstream.

> **[Output-field vs. intracavity convention reconciliation — supersedes
> the bridging-table efficiency entry below].** Equation (12.16) as
> written gives the integrated-noise variance in the **intracavity**
> amplitude convention: $\sigma_{\parallel,\text{intra}}^2(\tau) = \tau/(2\eta\kappa)$.
> §6 (eq. 6.5) commits to the **output-field** convention,
> $z_\text{out}(\tau) = \sqrt{\kappa_\text{ext}}\,\int_0^\tau \langle a\rangle\,dt$,
> in which both signal and noise variance carry an extra factor of
> $\kappa_\text{ext}$. In the output-field convention,
> $$\sigma_{\parallel,\text{out}}^2(\tau) = \kappa_\text{ext}\,\frac{\tau}{2\eta\kappa} \;\xrightarrow{\;\kappa_\text{ext} = \kappa\;}\; \frac{\tau}{2\eta} \qquad (12.16\text{b})$$
> (assuming $\kappa_\text{ext} = \kappa$, i.e., a fully external decay
> channel and no internal losses). **Module 1's `IQNoiseParams` is the
> implementation authority** for which convention is in force; downstream
> code must never hardcode $\tau/(2\eta\kappa)$ as the noise variance
> unless it has confirmed `iq_normalization == "intracavity"`. v0 sets
> `iq_normalization = "output_field"` to match §6; `IQNoiseParams.sigma_for_integration_window()`
> returns $\sigma_{\parallel,\text{out}}(\tau) = \sqrt{\tau/(2\eta)}$ (with
> $\kappa_\text{ext} = \kappa$, simple-Purcell-filter assumption). The
> bridging table below tabulates *both* conventions; downstream §12 prose
> uses the symbol $\sigma_\parallel^2(\tau)$ generically and refers to
> `IQNoiseParams` rather than substituting a specific factor.

**[Bridging table — three normalizations of the IQ record].**

| Quantity | Raw SME record (12.5) | Intracavity amplitude record (12.16a) | Output-field record (eq. 6.5; v0 default via `IQNoiseParams`) |
|---|---|---|---|
| Signal scale | $\sqrt{\eta\kappa/2}\,\langle a + a^\dagger\rangle$ | $\langle a + a^\dagger\rangle/2$ (= Re of cavity amplitude) | $\sqrt{\kappa_\text{ext}}\,\langle a\rangle$ (Re/Im components → $I, Q$) |
| Integrated noise variance over $\tau$ | $\tau$ | $\tau/(2\eta\kappa)$ | $\kappa_\text{ext}\cdot\tau/(2\eta\kappa) = \tau/(2\eta)$ at $\kappa_\text{ext} = \kappa$ |
| Efficiency $\eta$ handling | explicit prefactor | absorbed into rescaling $\sqrt{2\eta\kappa}$ | absorbed unless `IQNoiseParams.eta` is configured |
| Convention used downstream in §12 | not used | optional intermediate | **v0 adopted** (matches §6 eq. 6.5; consumed via `IQNoiseParams`) |

The §12 prose downstream of (12.16) refers to $\sigma_\parallel^2(\tau)$
*symbolically* and to `IQNoiseParams.sigma_for_integration_window()` for
the numerical value; no specific factor is hardcoded inside §12. This
recovers exactly the $\sigma_\parallel(\tau_\text{integration}) \propto \sqrt{\tau_\text{integration}}$
scaling derived independently in §6.3a (eq. 6.6a). The bad-cavity
adiabatic elimination *derives* the $\sqrt\tau$ scaling, with the
specific normalization (intracavity vs output-field, factor of
$\sqrt{2\eta\kappa}$, etc.) **delegated to `IQNoiseParams`** as the
implementation authority — see the output-field reconciliation box
above. §6 *postulated* the $\sqrt\tau$ scaling phenomenologically. The
two are therefore fully
consistent.


---

### 12.3 Reduction to direct-jump + Gaussian IQ

Equation (12.11) is itself still expensive to simulate at scale (it is a
diffusive SDE on the qubit Bloch vector). The v0 model takes one further
controlled approximation step.

**[Approximation — strong-measurement / projective qubit localization].**
In the regime where the measurement-induced dephasing rate dominates the
intrinsic relaxation,

$$\Gamma_d^\text{meas} \cdot \tau_\text{meas} \gg 1 \qquad \text{(12.17)}$$

the qubit rapidly localizes to $|0\rangle$ or $|1\rangle$ and the
$\sigma_z$ heterodyne diffusion update becomes (modulo Gaussian readout
noise) a *projective* update with Born-rule probabilities. **A direct
consequence** is that by the time the conditional X-pulse is applied
(§12.3 step 5), the measurement has destroyed all qubit coherences in
the $\sigma_x, \sigma_y$ subspace to leading order in $\chi/\kappa$, and
the post-measurement state can be treated as a *classical* mixture over
$|0\rangle, |1\rangle$ with weights given by the joint matrix entries
$P(s_f, m | s_i)$. The reset feedback therefore acts on populations
rather than on a coherent superposition; this is *implicitly* assumed
by the classical-state-history formulation of §12.3 and is made
explicit here. The leading correction is non-projective
back-action on $\sigma_x, \sigma_y$ coherences, of size $O(\chi/\kappa)$
(see omissions table §12.5). For our
parameters: $\Gamma_d^\text{meas}/2\pi \approx 8\chi^2\bar n/(\kappa \cdot 2\pi) \sim 1$ MHz
at $\bar n \sim 5$, so $\Gamma_d^\text{meas} \tau_\text{meas} \sim 3$
at $\tau_\text{meas} = 500$ ns — marginally satisfied. v0 takes (12.17)
as a working assumption.

**[Two equivalent unravellings].** Under (12.17), the conditional
dynamics over the integration window $\tau_\text{meas}$ can be unraveled
in two equivalent ways (Wiseman-Milburn 2010, §3.7; Gambetta et al. 2008,
§IV):

(A) **Diffusive trajectories.** Continuous SDE for $\rho_q$ under
heterodyne; noise is Gaussian; Bloch vector diffuses on a slow timescale
relative to projective collapse. Computationally expensive: requires
full SDE integration per shot.

(B) **Direct-jump (Poisson) for $\sigma^-$ + diffusive $\sigma_z$ update
from heterodyne.** Equivalent stochastic master equation, but the
relaxation channel is treated as a discrete jump rather than a continuous
contribution. Computationally cheap *if* the heterodyne unraveling is
also reduced (next step).

**[v0 reduction (B') — Poisson direct-jump + integrated Gaussian IQ].**
In the strong-measurement regime, the diffusive $\sigma_z$ update in (B)
becomes effectively a single projective update at $t = \tau_\text{meas}$,
with the projector chosen by the integrated heterodyne record (12.15). The
v0 model collapses (B) further to:

(B') A **Poisson direct-jump** for $\sigma^-$ at rate $\Gamma_1 = \gamma_{1,\text{eff}}$
(generating zero or one $|1\rangle \to |0\rangle$ jump events during
$\tau_\text{meas}$), and an **unconditional Gaussian IQ measurement
record** whose mean is determined by the *integrated time spent in each
qubit state*.

Concretely, for each Monte-Carlo shot:

1. *Sample the jump event.* If $s_i = e$, sample whether a $T_1$ jump
   occurs at a random time $t^* \in [0, \tau_\text{meas})$ with probability
   $P(\text{jump}) = 1 - e^{-\Gamma_1\,\tau_\text{meas}}$ (jump time
   exponentially distributed on $[0, \tau_\text{meas})$ conditional on
   occurrence). With complementary probability $P(\text{no jump}) = e^{-\Gamma_1 \tau_\text{meas}}$,
   the qubit stays in $|e\rangle$ throughout.
2. *Compute the trajectory mean IQ.* Construct the piecewise-constant
   qubit-state history $s(t)$:
   - No jump: $s(t) = s_i$ for $t \in [0, \tau_\text{meas}]$.
   - Jump at $t^*$: $s(t) = e$ for $t < t^*$ and $s(t) = g$ for $t \ge t^*$.
   Then propagate the cavity amplitude $\alpha(t)$ via (12.7) with
   $\alpha(0) = 0$ (initial cavity vacuum) and $s$-dependent detuning
   following $s(t)$. *Across a qubit jump at $t = t^*$, the cavity field
   is continuous*; only the conditioning frequency changes, after which
   the cavity relaxes toward the new pointer state with timescale $1/\kappa$.
   This is the **same construction** as the §7.3a one-jump $T_1$ mixture
   model, eq. (7.5b):

   $$c_{1,\text{jump}}(t^*) = \sqrt{\kappa_\text{ext}}\,\biggl[\int_0^{t^*}\alpha_1^\text{no-jump}(t)\,dt + \int_{t^*}^{\tau_\text{meas}}\alpha_{1\to 0}(t; t^*)\,dt\biggr] \qquad \text{(12.18)}$$

   with $\alpha_{1\to 0}(t^*; t^*) = \alpha_1^\text{no-jump}(t^*)$ (the
   continuous-cavity initial condition). v0 must call the Module 1
   pointer-response helper to compute $\alpha_s(t)$, not re-derive the
   cavity equation locally — see §12.6.
3. *Add Gaussian IQ noise.* Add zero-mean Gaussian noise on each axis
   using `IQNoiseParams.sigma_for_integration_window(tau_meas)` from
   Module 1 — **no $\tau$-scaling factor is hardcoded here**. In the
   intracavity convention this helper reduces to $\sqrt{\tau_\text{meas}/(2\eta\kappa)}$
   (eq. 12.16); in the v0 output-field convention with
   $\kappa_\text{ext} = \kappa$, it reduces to $\sqrt{\tau_\text{meas}/(2\eta)}$
   (eq. 12.16b). Module 5b consumes whichever value `IQNoiseParams`
   returns, preserving the §12.2 "Module 1 is the implementation
   authority" rule.
4. *Threshold for measurement outcome.* The discriminator (perpendicular
   bisector of $\alpha_g^\text{ss}, \alpha_e^\text{ss}$ in the IQ plane;
   §7.2) maps each integrated IQ point to a binary outcome
   $m \in \{0, 1\}$.
5. *Conditional X-pulse with finite fidelity.* Conditional on $m = 1$, a
   coherent $\pi$-pulse (Module 5a's calibrated X-gate) is applied with
   classical flip-error probability $\varepsilon_X = 1 - F_X$.

> **[Approximation — classical-flip-error reduction of $F_X$].** Module 5b
> consumes the **probe-set scalar** $\varepsilon_X = 1 - F_X$ from Module
> 5a as a *phenomenological classical, symmetric, state-independent
> flip-failure probability*: with probability $\varepsilon_X$ the
> conditional pulse fails to flip the qubit; with probability
> $1 - \varepsilon_X$ it flips correctly. This is **not** a process-matrix-level
> reset model and **not** generally equivalent to the average gate infidelity $1 - \bar F$. Coherent
> over-rotation, leakage to $|2\rangle$ during the X-pulse, and
> state-dependent error structure (e.g., different errors for
> $|e\rangle \to |g\rangle$ vs $|g\rangle \to |e\rangle$) do not collapse
> to a single symmetric scalar without an additional simplification.
> v0 of Module 5b adopts this simplification as a **phenomenological
> approximation**; the assumption is named explicitly so that v1.5 can
> upgrade Module 5b to consume the full process matrix from Module 5a
> rather than a single-scalar $\varepsilon_X$.

> **[Approximation — late-time jump dependence; multi-jump truncation].**
> The one-jump mixture is *exact* for the no-jump case. For a jump at
> $t^* < \tau_\text{meas}$, the IQ vector has spent time $t^*$ near
> $\alpha_e^\text{ss}$ and $\tau_\text{meas} - t^*$ near $\alpha_g^\text{ss}$;
> the resulting integrated IQ lies on the line segment between them.
> Multi-jump contributions (a $T_1$ jump followed by a thermal excitation
> back, etc.) are $O((\Gamma_1\tau_\text{meas})^2)$ and are dropped at this
> order. At the synthetic seed and using $\Gamma_1 = \gamma_{1,\text{eff}} \approx 9.5 \times 10^4\,\text{s}^{-1}$
> (Convention 21; $T_{1,\text{eff}} \approx 10.5\,\mu$s) for
> $\tau_\text{meas} = 500\,\text{ns}$,
> $\Gamma_1\tau_\text{meas} \approx 4.7\%$, so the dropped multi-jump
> terms are $O(2 \times 10^{-3})$ — small but not as small as the
> 1.67% / $3 \times 10^{-4}$ estimate one would obtain using
> $T_{1,\text{intrinsic}} = 30\,\mu$s alone. v0 reports the
> $T_{1,\text{eff}}$-derived value; the intrinsic-only number is
> available as a diagnostic.

This (B') is the v0 semiclassical reset model. Its consistency with §7.3a
is *exact*: the §7.3a one-jump $T_1$ mixture, originally postulated as a
controlled refinement to the §7.1 centroid-Gaussian baseline, is now
**derived** from the bad-cavity-eliminated SME. This consistency is a
non-trivial cross-check on both derivations.

---

### 12.4 Joint matrix $P(s_f, m | s_i)$ and reset-cycle infidelity

**[Definition].** The full joint matrix of the reset protocol is the
4-tensor

$$P(s_f, m \,|\, s_i), \quad s_i, s_f \in \{g, e\}, \quad m \in \{0, 1\} \qquad \text{(12.19)}$$

For each initial state $s_i$, the joint distribution over
$(s_f, m)$ has 4 entries summing to unity. The semiclassical model of
§12.3 computes these entries via Monte-Carlo sampling over $N$ trajectories
(default $N = 1000$ for exploratory runs, escalating to $N = 4000$ for
final figure quality).

**[Joint matrix elements at the reduced level].** Working through the
trajectory categories (using the discriminator-axis projection of the
trajectory-mean cavity amplitude rather than the $\sigma_z$ formalism,
which avoids any ambiguity in convention):

- $s_i = e$, no jump: probability $e^{-\Gamma_1\tau_\text{meas}}$; final
  state $s_f = e$; integrated IQ centered at the projection of $\alpha_e^\text{ss}$
  onto the discrimination axis times $\tau_\text{meas}$ (i.e., the
  "excited-state cluster" $\bar M_e$); discriminator returns $m = 1$ with
  probability $F_\text{disc}^{(e)}$.
- $s_i = e$, jump at $t^*$: probability density
  $\Gamma_1\,e^{-\Gamma_1 t^*}\,dt^*$; final state $s_f = g$; integrated
  IQ centered at the trajectory-weighted mean of $\alpha_e^\text{ss}$ for
  duration $t^*$ and $\alpha_g^\text{ss}$ for duration $\tau_\text{meas} - t^*$
  (modulo cavity-memory continuity at $t^*$ per Convention 20), located
  on the line segment between $\bar M_e$ and $\bar M_g$ in the IQ plane;
  discriminator probability depends on $t^*$.
- $s_i = g$, no jump (no thermal excitation): probability 1; final state
  $s_f = g$; integrated IQ centered at $\bar M_g$; discriminator returns
  $m = 0$ with probability $F_\text{disc}^{(g)}$.

Under the $\sigma_z = |e\rangle\langle e| - |g\rangle\langle g|$ convention
adopted in §12.1, eq. (12.14) gives the trajectory-mean centroids
$\bar M_e \propto -|\alpha_g^\text{ss} - \alpha_e^\text{ss}|/2$ (negative
offset, "excited cluster") and $\bar M_g \propto +|\alpha_g^\text{ss} - \alpha_e^\text{ss}|/2$
(positive offset, "ground cluster"), matching $\langle\sigma_z\rangle_e = +1$
and $\langle\sigma_z\rangle_g = -1$.

Symbolically, ignoring thermal excitation $\bar n_\text{th}$:

$$\begin{aligned}
P(s_f = e, m = 0\,|\,e) &= e^{-\Gamma_1\tau_\text{meas}}\,(1 - F_\text{disc}^{(e)}) \\
P(s_f = e, m = 1\,|\,e) &= e^{-\Gamma_1\tau_\text{meas}}\,F_\text{disc}^{(e)} \\
P(s_f = g, m = 0\,|\,e) &= \int_0^{\tau_\text{meas}} \Gamma_1 e^{-\Gamma_1 t^*}\,(1 - F_\text{disc}^{(\text{jumped},t^*)})\,dt^* \\
P(s_f = g, m = 1\,|\,e) &= \int_0^{\tau_\text{meas}} \Gamma_1 e^{-\Gamma_1 t^*}\,F_\text{disc}^{(\text{jumped},t^*)}\,dt^*
\end{aligned} \qquad \text{(12.20)}$$

with $F_\text{disc}^{(\text{jumped},t^*)}$ the discriminator probability of
returning $m = 1$ for a trajectory that jumped at time $t^*$. Analogous
expressions hold for $s_i = g$.

**[Why the joint matrix is necessary].** A *plain confusion matrix*
$P(m | s_i)$ marginalizes over $s_f$ and conflates two physically distinct
events:

- $P(s_f = e, m = 0 | e)$: qubit stayed excited, measurement missed it
  → **reset FAILS** (qubit needed flipping; nothing was applied).
- $P(s_f = g, m = 0 | e)$: qubit decayed mid-measurement, measurement
  correctly read ground → **reset SUCCEEDS** (already in $|g\rangle$,
  no flip needed).

The joint matrix distinguishes these. The plain confusion matrix
conflates them and gives a misleadingly pessimistic estimate of
reset success.

**[Reset-cycle infidelity formula].** Per single reset cycle, starting
from $|e\rangle$ (worst case, $p_e = 1$), with conditional X-gate of
classical flip-error probability $\varepsilon_X$:

$$p_e' = P(s_f{=}e, m{=}0 | e) + P(s_f{=}e, m{=}1 | e)\cdot \varepsilon_X + P(s_f{=}g, m{=}1 | e)\cdot (1-\varepsilon_X) \qquad \text{(12.21)}$$

Three terms:

1. **Missed-excited.** Qubit stayed in $|e\rangle$, readout returned 0,
   no flip applied → still excited.
2. **Gate failure on detected-excited.** Qubit stayed in $|e\rangle$,
   readout returned 1, X-gate failed (probability $\varepsilon_X$) → still
   excited.
3. **False-positive on decayed.** Qubit decayed during measurement (final
   state $|g\rangle$), readout returned 1, X-gate succeeded with
   probability $(1 - \varepsilon_X)$ → flipped back to $|e\rangle$.

The fourth combination $P(s_f = g, m = 0 | e)$ does *not* appear: the
qubit decayed, readout correctly said 0, no flip, ends in $|g\rangle$.
This is the case the plain confusion matrix wrongly counts as reset
failure.

**[Ideal-gate floor (corrected two-term form)].** For perfect X-gate
($\varepsilon_X = 0$),

$$p_e'\bigm|_{\varepsilon_X = 0} = P(s_f{=}e, m{=}0 | e) + P(s_f{=}g, m{=}1 | e) \qquad \text{(12.22)}$$

This is **two terms, not one**: missed-excited *plus*
false-positive-on-decayed flip back to $|e\rangle$. The latter is
maximal at $\varepsilon_X = 0$ (the X-gate flips perfectly on the
decayed qubit, undoing the desirable decay-to-$|g\rangle$). This is the
non-trivial structural finding that motivates measuring $p_e$ via the
joint matrix rather than the plain confusion matrix; it matches Magnard,
Kurpiers, Royer, Walter, Besse, Gasparinetti, Pechal, Heinsoo, Storz,
Blais, Wallraff, *Phys. Rev. Lett.* **121**, 060502 (2018) [arXiv:1801.07689]
in form (their $|f,0\rangle \leftrightarrow |g,1\rangle$ Purcell-filter
protocol replaces the conditional $\pi$-pulse with a parametric drive but
has the same overall accounting).


---

### 12.5 Approximation hierarchy and explicit omissions

The derivation §12.1 → §12.2 → §12.3 has the following layered structure;
each layer is a controlled approximation with an identifiable small
parameter:

| Level | Equation / process | Smallness parameter | Status in v0 |
|---|---|---|---|
| 0. Full circuit-QED Hamiltonian | (12.1) | — | analytic starting point |
| 1. Schrieffer-Wolff to dispersive | (12.3) | $(g/\Delta)^2$ | [Approximation]: included via $\chi$ from §4 |
| 2. Lindblad master equation | (12.2) | Markov, secular | [Approximation]: standard |
| 3. Heterodyne SME unraveling | (12.4) | $\eta \in (0, 1]$ | [Exact within model]: structure preserved |
| 4. Polaron / displaced-frame transformation | §12.2 (i)-(iii) | $\kappa \gg \chi, \gamma_1$ | [Approximation]: bad-cavity adiabatic elimination |
| 5. Steady-state cavity tracking | (12.7) → (12.8) | $(\kappa\,\tau_\text{ramp})^{-1}$ | [Approximation]: omits transient cavity response during drive ramp |
| 6. Reduction to $\sigma_z$-only diffusive update | (12.11) | $(\chi/\kappa)^2$ for $\sigma_x, \sigma_y$ back-action | [Approximation]: neglects coherence back-action |
| 7. Strong-measurement projective limit | (12.17) | $\Gamma_d^\text{meas}\,\tau_\text{meas} \gg 1$ | [Approximation]: marginally satisfied at default $\tau_\text{meas}$ |
| 8. One-jump direct-jump unraveling for $\sigma^-$ | §12.3 (B') | $(\gamma_1\,\tau_\text{meas})^2$ for multi-jump | [Approximation]: $\le 1$ jump per measurement |
| 9. Gaussian IQ integration | §6.3a | finite SNR; CLT | [Exact within reduced model] |
| 10. Discriminator (perpendicular bisector) | §7.2 | optimal under Gaussian | [Exact within reduced model] |
| 11. Conditional $\pi$-pulse | §11 (DRAG) | $F_X = 1 - \varepsilon_X$ | [Approximation]: gate error folded in classically |

Explicitly omitted physics, with natural small parameters:

- **Finite-cavity transient response** — $(\kappa\,\tau_\text{ramp})^{-1}$ ≪ 1 if drive rise/fall time ≫ $1/\kappa$. For $\kappa/2\pi = 5$ MHz, $1/\kappa \approx 32$ ns; if the readout pulse rises in $< 100$ ns, transients matter and (12.8) should be replaced by the time-resolved solution of (12.7).
  - *Comment:* v0 assumes square-pulse-like steady-state behavior
- **Heterodyne back-action on $\sigma_x, \sigma_y$ coherences** — $O((\chi/\kappa)^2)$ corrections to the diffusive term in (12.11) at the rate/probability level (the underlying amplitude-level correction is $O(\chi/\kappa)$, but it enters the dephasing-rate expression squared)
  - *Comment:* For our parameters, $\chi/\kappa \approx 0.077$, so $(\chi/\kappa)^2 \approx 0.6\%$ corrections to coherence trajectories — sub-dominant to discriminator error in v0
- **Thermal excitation** — equilibrium Bose factor $\bar n_\text{B}(\omega, T)$ is exponentially small at 20-30 mK for 4-6 GHz qubit frequencies (cf. §5.8); values such as $\bar n_\text{th} = 0.05$ should be interpreted as a **deliberately elevated non-equilibrium stress-test population**, not as a thermal-equilibrium occupation at 20 mK.
  - *Comment:* Optional stress test; default off ($\bar n_\text{th} = 0$).
- **Multi-photon non-QND / measurement-induced state transitions** — $\bar n / n_\text{crit}$, $n_\text{crit} = \Delta^2/(4\,g_{01}^2)$ (under `coupling_convention="matrix_element_01"`; the bare-charge $g$ in alternative conventions would give a numerically different $n_\text{crit}$ unless rescaled)
  - *Comment:* For our parameters, $n_\text{crit} \approx 125$; safe at $\bar n \le 5$. Sank, Chen, Khezri, Kelly et al., *Phys. Rev. Lett.* **117**, 190503 (2016) for the mechanism; not modelled at v0
- **Drive-induced $T_1$ degradation (Slichter mechanism)** — $\bar n \cdot S_\varphi(\omega_q)/\kappa$ — frequency-converted dephasing noise
  - *Comment:* Slichter, Vijay, Weber, Boutin, Boissonneault, Gambetta, Blais, Siddiqi, *Phys. Rev. Lett.* **109**, 153601 (2012); not modelled
- **Leakage to $|2\rangle$ during readout** — $\sim 10^{-3}$ at high power; suppressed in well-Purcell-filtered designs (Hazra et al. arXiv:2407.10934)
  - *Comment:* Out-of-subspace population accumulation; not modelled
- **Coherent X-pulse errors during reset** — Classical $\varepsilon_X$ folded in; coherent over/under-rotation requires v1.5
  - *Comment:* Linear in $1 - F_X$ at leading order
- **Detuned-neighbor measurement error** — $O(g^2/\Delta_\text{nbr}^2)$
  - *Comment:* Khezri-Dressel-Korotkov, *Phys. Rev. A* **92**, 052306 (2015); not modelled (single-qubit v0)
- **Quasiparticle and 1/f flux noise** — Out of model scope
  - *Comment:* Affects $T_1, T_2$ only; absorbed phenomenologically

> **[Framing — controlled-scope statements, not formal error bounds].**
> None of the small parameters above are quoted as formal operator-norm
> error bounds. They are *scope statements*: "the model is controlled in
> the sense that omitted terms have explicit small parameters and become
> quantitatively negligible in the stated regime, beyond which the model
> should be re-derived." This matches the standard practice in
> dispersive-readout theory where formal bounds are typically only
> available for restricted analytic models (e.g., Khezri et al. 2016 give
> $O(\chi/\kappa)$ numerical bounds for the dressed-coherent-state
> approximation specifically). Validity of v0 outside the bad-cavity,
> low-photon, fast-Purcell-filter regime requires re-deriving from
> Eq. (12.4) without the §12.2 elimination.

---

### 12.6 Cross-references and consistency with §3-§7

The Module 5b derivation is consistent with:

- **§3 Jaynes-Cummings + RWA.** Provides Eq. (12.1).
- **§4 Dispersive shift via Schrieffer-Wolff.** Provides $\chi$ in
  Eq. (12.3); transmon-corrected $\chi$ accounting for $|2\rangle$ admixture
  via the multilevel formula (4.9).
- **§5 Lindblad master equation.** Provides Eq. (12.2); the Purcell decay
  $\gamma_\text{Purcell} = (g_{01}/\Delta)^2\,\kappa$ is *part of*
  $\gamma_1^\text{eff}$ in (12.13), *not* a separate channel — same
  convention as §5.3.
- **§6.3a IQ trajectory and noise scaling.** Provides
  $\sigma_\parallel(\tau_\text{integration}) \propto \sqrt\tau$. Bad-cavity
  elimination *derives* this scaling (12.16) where §6 *postulated* it
  phenomenologically.
- **§7 Assignment fidelity, perpendicular-bisector discriminator.**
  Applied directly in step 4 of §12.3.
- **§7.3a one-jump $T_1$ mixture.** Module 5b reduces to the §7.3a model
  in the **measurement-only, no-feedback limit** (same readout drive
  applied, reset feedback disabled, same one-jump $T_1$ trajectory
  construction); this is a non-trivial consistency check on the
  derivation. The post-jump cavity amplitude (12.18) uses the §7.3a form
  (7.5b) explicitly, with continuous cavity initial condition.
- **Module 1 pointer-response helper.** Consumed for computing
  $\alpha_g^\text{ss}, \alpha_e^\text{ss}$ and the IQ means $m_g, m_e$ per
  shot. v0 must reuse this helper rather than re-derive (12.7) locally.

---

### 12.7 Validation strategy

Validation tests for Module 5b, in the order they are typically run:

- **V1 — corrected two-term ideal-gate floor.** With $\varepsilon_X = 0$ and $p_e = 1$, the residual $p_e'$ matches Eq. (12.22) — i.e., $P(s_f=e, m=0|e) + P(s_f=g, m=1|e)$ — to $< 10^{-6}$. Both terms are non-negligible in the $T_1$-during-measurement regime; missing either term (the single-term plain-confusion-matrix "floor" $P(m=0|e)$) is a structural bug.

- **V2 — active beats passive at short $\tau_\text{meas}$.** Sweep $\Gamma_1\,\tau_\text{meas} \in [0.05, 1.0]$ (Convention 21: $\Gamma_1 = \gamma_{1,\text{eff}}$ in Module 5b). Require that at *some* operating point, the active-reset residual from (12.21) lies below the matched-duration passive baseline $e^{-\Gamma_1 \tau_\text{meas}}$. Sweep-based (not fixed-point) for robustness to operating-point choice.

- **V3 — long-$\tau$ asymmetric floor.** Passive reset approaches the thermal floor ($\bar n_q$ or 0). Active reset approaches a *different* floor: thermal + readout false-positives (the $P(s_f=g, m=1|g)$ term that triggers an unnecessary X-gate) + gate error. The gap is the active-reset overhead at long $\tau$, not a bug.

- **V4a — no-jump deterministic limit (blocking).** With `ResetParams(force_no_jumps=True)` (equivalently `gamma1_eff_override=0.0`), the IQ-record stochastic relaxation is disabled while pointer-response parameters $(\chi, \kappa, \varepsilon(t))$ are unchanged. The marginal $\frac{1}{2}(P(m=0|g_\text{init}) + P(m=1|e_\text{init}))$ must equal Module 1's $F_\text{assign}$ to numerical precision. Failure indicates a wrong IQ-record / pointer / threshold convention.

- **V4b — finite-$T_1$ marginal consistency (diagnostic).** With $\Gamma_1 > 0$, the same marginal must agree with a Module 1 reference *that includes finite-$T_1$ at the IQ-distribution level* (jump-time mixture, not Gaussian around $\langle\alpha(t)\rangle$) within $2\times$ shot noise + solver tolerance. If Module 1 doesn't expose such a reference, V4b downgrades to a diagnostic and V4a is the blocking gate.

- **V5 — trajectory count convergence.** Standard error on $p_e'$ scales as $1/\sqrt{N_\text{trajectories}}$; binomial standard errors on joint-matrix entries are reported.

- **V6 — worst-case dominates lower-$p_e$ prior.** Compare $p_e = 1$ (worst case) against $p_e = 1/2$ (mixed prior); the worst-case residual must dominate.

- **V7 — joint matrix reveals $T_1$-during-measurement effect.** At $\varepsilon_X = 0$, sweep $\Gamma_1\,\tau_\text{meas} \in [0.1, 2.0]$ and identify a regime where $P(s_f=g, m=0|e)$ — the qubit-decayed-AND-measurement-read-ground case — exceeds 0.05. This demonstrates the conceptual point that the joint matrix is needed.

### 12.8 Convention warning collected for §12

> **Convention 20 (cavity-amplitude initial condition across a $T_1$
> jump).** When sampling a trajectory in the v0 direct-jump model and a
> $T_1$ jump occurs at $t = t^*$, the cavity amplitude is **continuous**
> across the jump: $\alpha_{1\to 0}(t^*; t^*) = \alpha_1^\text{no-jump}(t^*)$,
> per the cavity-memory correction in §7.3a (eq. 7.5b). The naive
> shorthand of restarting the post-jump cavity from vacuum (or from the
> $\alpha_g^\text{ss}$ steady state) erases the cavity displacement
> present at the jump time and biases $c_{1,\text{jump}}$ toward the
> unconditional-$|g\rangle$ pointer. Implementations must explicitly
> propagate the cavity ODE through the jump time using the displaced
> initial condition.

---

### 12.9 Script connection table for §12

- **(12.1)-(12.3) Full circuit-QED Lindblad (theoretical reference)** — — (out of v0; see §12.10)
  - *Function:* —
  - *Test:* (v1.5 forward pointer)
- **(12.7)-(12.8) Pointer-response helper $\alpha_g^\text{ss}, \alpha_e^\text{ss}$** — `stage_06_readout/physics/pointer_response.py` (Module 1)
  - *Function:* `pointer_response()`
  - *Test:* (validated in Module 1)
- **(12.12) Measurement-induced dephasing $\Gamma_d^\text{meas}$ (consumes `pointer_response(normalization="intracavity")`)** — `stage_06_readout/control/reset_protocol.py`
  - *Function:* `meas_induced_dephasing()`
  - *Test:* `test_recovers_zero_at_chi_zero`, `test_meas_induced_dephasing_uses_intracavity_alpha` (B1 regression guard)
- **Pointer-response intracavity-to-output round-trip (Module 1 helper, consumed by §12)** — `stage_06_readout/physics/pointer_response.py` (Module 1)
  - *Function:* `pointer_response()` (with `normalization` parameter)
  - *Test:* `test_pointer_response_intracavity_to_output_conversion` (B1 round-trip guard)
- **(12.13) Effective $T_1$ inclusive of Purcell** — (consumed from §5.3)
  - *Function:* (already validated)
  - *Test:* (V4b in §5)
- **(12.16) Integrated noise variance $\sigma_\parallel^2 \propto \tau_\text{meas}$** — `stage_06_readout/physics/iq_noise.py` (Module 1)
  - *Function:* `IQNoiseParams.sigma_for_integration_window()`
  - *Test:* (V6a in §12.7)
- **(12.18) Post-jump cavity construction** — `stage_06_readout/control/reset_protocol.py`
  - *Function:* `compute_post_jump_iq()`
  - *Test:* `test_cavity_continuous_across_jump`, V4a
- **(12.20) Joint matrix elements** — `stage_06_readout/control/reset_protocol.py`
  - *Function:* `extract_joint_matrix()`
  - *Test:* `test_joint_matrix_normalization`, `test_against_one_jump_mixture` (consistency with §7.3a)
- **(12.21) Reset-cycle infidelity** — `stage_06_readout/analysis/reset_metrics.py`
  - *Function:* `reset_residual_single_cycle()`
  - *Test:* V1 (corrected two-term floor)
- **(12.22) Ideal-gate two-term floor** — `stage_06_readout/analysis/reset_metrics.py`
  - *Function:* `joint_ideal_gate_floor()`
  - *Test:* `test_reset_residual_ideal_gate_two_term_floor`
- **Passive baseline $e^{-\gamma_{1,\text{eff}}\,\tau}$ (Convention 21; uses $\Gamma_1 = \gamma_{1,\text{eff}}$ matching the direct-jump model, *not* $1/T_{1,\text{intrinsic}}$)** — `stage_06_readout/analysis/reset_metrics.py`
  - *Function:* `passive_reset_residual()` (defaults to $\gamma_{1,\text{eff}}$; intrinsic-idle option must be explicitly labeled)
  - *Test:* `test_passive_reset_baseline_formula`
- **One-jump trajectory sampling** — `stage_06_readout/control/reset_protocol.py`
  - *Function:* `sample_t1_jump_time()`
  - *Test:* `test_jump_pdf_exponential_truncated`
- **Conditional X-pulse with $F_X$** — (consumed from Module 5a)
  - *Function:* `apply_x_with_fidelity()`
  - *Test:* (validated in §11)

---

### 12.10 Forward-pointers (out of v0 scope)

- **Magnard et al. 2018 (Purcell-filter-assisted unconditional reset).**
  $|f, 0\rangle \leftrightarrow |g, 1\rangle$ parametric drive with
  Purcell filter. Achieves $p_e' = 0.2\%$ in 280 ns. Requires explicit
  filter-resonator modeling; out of v0 scope.
- **Riste et al. 2012 (original measurement-based reset experiment).**
  Reports $p_e' = 1.2\%$ for the same conceptual protocol modeled here.
  Used as primary qualitative cross-check for v0.
- **Continuous-measurement back-action on coherences.** v1.5 may add a
  scoped diffusive-SME extension via `qutip.smesolve` on the
  qubit-only space, retaining the Module 1 pointer-response helper for
  cavity dynamics. Adds the $O(\chi/\kappa)$ coherence-trajectory
  corrections of (12.11).
- **`mcsolve`-based jump-history sampling.** v1.5 may replace the
  exponential-jump-time sampler with `qutip.mcsolve` on qubit channels
  only (never $\sqrt\kappa\,a$, which would conflate photon-counting
  unraveling with the heterodyne-style IQ observable). Cavity response
  in such a v1.5 extension would still flow through the Module 1
  pointer-response helper.
- **Coherent X-gate errors during reset.** v1.5 may simulate the X-pulse
  Hamiltonian dynamics during the conditional flip (using Module 5a's
  gate simulator) rather than treating it as a classical bit-flip.
- **Optimal-control reset pulses (Govia-Wilhelm 2015; Egger et al. 2018).**
  Driven reset protocols beyond conditional $\pi$-pulses; out of v0 scope.

---

### 12.11 References for §12

- **Wiseman, Milburn**, *Quantum Measurement and Control*, Cambridge UP (2010), Ch. 3-5 — homodyne / heterodyne SME formalism.
- **Carmichael**, *Statistical Methods in Quantum Optics 2: Non-Classical Fields*, Springer (2008), Ch. 8, §12.1.3, Ch. 13 — adiabatic elimination, bad-cavity limit.
- **Gambetta, Blais, Boissonneault, Houck, Schuster, Girvin**, *Phys. Rev. A* **77**, 012112 (2008) [arXiv:0709.4264] — quantum trajectory for circuit-QED with polaron transformation; foundational reference for §12.2 derivation.
- **Boissonneault, Gambetta, Blais**, *Phys. Rev. A* **79**, 013819 (2009) [arXiv:0810.1336] — photon-dependent qubit dephasing/relaxation, dispersive corrections.
- **Tornberg, Johansson**, *Phys. Rev. A* **82**, 012329 (2010) — feedback-assisted parity measurement; structure of dispersive readout back-action.
- **Khezri, Mlinar, Dressel, Korotkov**, *Phys. Rev. A* **94**, 012347 (2016) [arXiv:1606.04204] — dressed coherent / squeezed states; precise structure of the dispersive readout SME beyond the simplest model.
- **Khezri, Dressel, Korotkov**, *Phys. Rev. A* **92**, 052306 (2015) [arXiv:1506.06321] — measurement error from coupling to detuned neighbors.
- **Riste, van Leeuwen, Ku, Lehnert, DiCarlo**, *Phys. Rev. Lett.* **109**, 050507 (2012) — original measurement-based reset experimental demonstration.
- **Magnard, Kurpiers, Royer, Walter, Besse, Gasparinetti, Pechal, Heinsoo, Storz, Blais, Wallraff**, *Phys. Rev. Lett.* **121**, 060502 (2018) [arXiv:1801.07689] — Purcell-filter-assisted unconditional all-microwave reset (forward-pointer).
- **Hazra, Dai, Connolly, Kurilovich, Wang, Frunzio, Devoret**, arXiv:2407.10934 (2024) — readout-induced leakage benchmarking.
- **Sank, Chen, Khezri, Kelly et al.**, *Phys. Rev. Lett.* **117**, 190503 (2016) — measurement-induced state transitions beyond RWA.
- **Slichter, Vijay, Weber, Boutin, Boissonneault, Gambetta, Blais, Siddiqi**, *Phys. Rev. Lett.* **109**, 153601 (2012) — measurement-induced qubit state mixing from up-converted dephasing noise.
- **Walter, Kurpiers, Gasparinetti, Magnard et al.**, *Phys. Rev. Applied* **7**, 054020 (2017) [arXiv:1701.06933] — single-shot dispersive readout SNR in the bad-cavity / Purcell-filter regime.
- **Magesan, Gambetta**, *Phys. Rev. A* **101**, 052308 (2020); *PRX Quantum* **2**, 020324 (2021) — joint-matrix and SNR characterization for transmon readout.
- **Govia, Wilhelm**, *Phys. Rev. Applied* **4**, 054001 (2015) — unitary-feedback-improved qubit initialization in the dispersive regime.
- **Geerlings, Leghtas, Pop, Shankar, Frunzio, Schoelkopf, Mirrahimi, Devoret**, *Phys. Rev. Lett.* **110**, 120501 (2013) — driven reset protocol.
- **Egger, Werninghaus, Ganzhorn, Salis, Fuhrer, Müller, Filipp**, *Phys. Rev. Applied* **10**, 044030 (2018) — pulsed reset for fixed-frequency transmons.

## 13. Scope Notes

Items deliberately *out of scope* for Stage 06 v0, with forward pointers:

- **Full SME / continuous-measurement diffusive trajectories.** Beyond assignment-fidelity scope. Trajectory simulation IS used as a scoped cross-check (Module 1 V7 (jump-tail cross-check)) at one operating point with $\tau/T_1 \gtrsim 1\%$ to bound the centroid+Gaussian baseline bias. Module 5b §12 *derives* its v0 direct-jump model from the SME via bad-cavity adiabatic elimination but does not propagate the diffusive SDE on $\sigma_x, \sigma_y$. → v1.5 `qutip.smesolve`-on-qubit-only with Module 1 pointer-response helper.
- **Higher-order DRAG (DRAG-2, FAST DRAG, optimal control); AC-Stark detuning compensation as default.** Module 5a v0 covers DRAG-1 + sin²-window + calibrated $\beta$ scan only. → Hyyppä 2024, Werninghaus 2021, Gambetta 2011 §IV.B.
- **Charge-basis transmon matrix elements.** $E_J/E_C \approx 65.6$ at the synthetic seed gives ~6% Duffing-vs-charge-basis matrix-element gap; relevant for sub-3-ns gates only. → v1.5 charge-basis drive.
- **Multi-qubit / multiplexed readout.** Single-qubit by design. → crosstalk + correlated-noise extension.
- **Flux-tunable / asymmetric SQUID.** Fixed-frequency transmon assumed. → extension to `TransmonParams` dataclass.
- **Phase-sensitive amplification; correlated / non-Markovian noise.** Standard amp chain + Lindblad Markov assumption. → squeezed-state extension; Redfield / TCL2.
- **Microscopically-derived multilevel dephasing.** v0 uses phenomenological number-operator default (§5.4). → noise-spectroscopy extension.
- **Multi-jump $T_1$ mixture as budget baseline.** One-jump mixture (§7.3a) covers $\tau/T_1 \le 5\%$. → trajectory-aware budget; v1.5 `mcsolve`-based reset jump-history sampler.
- **Purcell-filter-assisted unconditional reset; optimal-control reset pulses.** → Magnard 2018; Govia-Wilhelm 2015; Egger 2018.
- **Coherent over/under-rotation X-gate errors during reset.** Module 5b folds Module 5a's gate error in classically as $\varepsilon_X = 1 - F_X$. → v1.5 simulate the X-pulse Hamiltonian dynamics during the conditional flip.
- **Generative / diffusion-based pulse design; full-device autodiff.** → post-submission roadmap.

## 14. Summary

This framework derives the physics of dispersive transmon readout from first principles and maps every equation to the Python module, function, and validation test that implements it.

**Module 1 (§1–§7)** — the core simulator. Cooper-pair box → transmon spectrum → Jaynes-Cummings → dispersive transform → Lindblad dynamics → IQ readout. The synthetic seed $\chi/2\pi \approx -0.385$ MHz, $\kappa/2\pi = 5$ MHz gives a deliberately weak-pull stress-test regime ($|\chi|/\kappa \approx 0.077$) far from the Marxer-style design optimum — the framework is built to handle the harder regime, not to chase the easier one.

**Module 2 (§8)** — coherent + incoherent error budget via marginal (channel-off) decomposition (Convention 12), with explicit cross-term diagnostics.

**Module 3 (§9)** — characterization theory: synthetic traces → `lmfit` → parameter recovery, with rad/s-internal / SI-output convention (Convention 14).

**Module 4 (§10)** — sensitivity in log-infidelity space (Convention 13) and Pareto optimization over the Module 1 simulator, including the Purcell-threshold curve (§10.5).

**Module 5a (§11)** — DRAG-corrected single-qubit X gates on a 4-level Duffing manifold with sin²-windowed Gaussian envelope. v0 calibrates $\beta_\text{opt}$ empirically; the AC-Stark $(2\beta - 1)/4$ prefactor in (11.16) is unverified pending GMMW-11 source check. Prototype suppression-ratio and $\beta_\text{opt} \approx 2.2$ values are explicitly **[Status: prototype, pending in-repo reproduction]**.

**Module 5b (§12)** — semiclassical active reset derived from the heterodyne SME via bad-cavity adiabatic elimination (Gambetta-Blais-Boissonneault polaron), reduced to a controlled direct-jump + Gaussian IQ model. Convention 21 ($T_{1,\text{eff}} = T_{1,\text{intrinsic}}^{-1} + \gamma_\text{Purcell}$) is enforced uniformly. The $\Gamma_d^\text{meas}$ formula (12.12) consumes intracavity pointers; reset IQ sampling consumes output-field pointers; the two normalizations are reconciled via `pointer_response(normalization=...)`.

**Validation strategy** is split by module: V1–V7 for Module 1, V1–V6 for Modules 2 / 3 / 4, V1–V8 for Module 5a (DRAG), V1–V8 for Module 5b (reset). Every blocking gate has a named test function; every advisory gate has a relative-tolerance threshold and a denominator convention (§12.7 V7 uses the infidelity, not the fidelity, to avoid instability when $F \to 1$).

**Honest scope.** The centroid+Gaussian fidelity is a *baseline*, not ground truth (Convention 16). Module 5 prototype values are pending in-repo reproduction. Beyond-RWA Bloch-Siegert frequency shifts ($\sim 1.2$ MHz at the seed) are deliberately out of model scope, not "below truncation". The synthetic seed is *not* a Marxer parameter extraction. These limitations are documented at every consuming call site.

The framework can be **frozen as theory v0** with the prototype-status labels carried explicitly into the v0 Python implementation; empirical claims are upgraded to v0-confirmed only after in-repo reproduction.
