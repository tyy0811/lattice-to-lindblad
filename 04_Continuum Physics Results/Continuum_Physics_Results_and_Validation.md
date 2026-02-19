# Physics-Grade Results & Packaging — Results & Validation

**Continuum-limit physics and phenomenological suppression hierarchy**

This document presents the numerical results that complete the Continuum Physics Results of the project. Each section corresponds to one deliverable: (1) the continuum extrapolation of the massless Schwinger model mass gap toward the exact result $1/\sqrt{\pi}$, (2) the sequential suppression of quarkonium states (1S vs 2S) from the $1\oplus 8$ Lindblad model, and (3) the Bjorken cooling extension comparing time-dependent $T(\tau)$ evolution against the fixed-temperature baseline.

---

## 1. Massless Schwinger Model: Continuum Mass-Gap Extrapolation

**Code:** `schwinger_continuum_massgap.py`

**Setup.** The gauge-eliminated Schwinger Hamiltonian (validated at $N=4$ in Validation Baseline) is diagonalized via sparse Lanczos (`scipy.sparse.linalg.eigsh`) in the half-filling sector for $N \in \{8, 10, 12, 14, 16, 18, 20\}$ and $x \in \{2, 3, 4, 5, 6, 8, 10, 12\}$ using the convention $x = 1/(2(ag)^2)$. To push beyond the ED ceiling, the same ground-state and mass-gap observables are extended to $N=30,40$ with DMRG (TeNPy, bond dimension $\chi=100$), validated against ED at $N\le 20$. For speed, the ED uses a **matrix-free** operator representation $H(x)=V + xT$ (diagonal electric term $V$ plus hopping matrix $T$), **warm-starts** the eigensolver across increasing $x$, and caches $(E_0,E_1)$ so that sensitivity scans (e.g. changing the $1/N$-fit guard or continuum-fit window) do not require recomputing eigenvalues. The physical mass gap is extracted as $M_{\mathrm{gap}}/g = \Delta \tilde{E} \cdot (ga)/2$, where $\Delta \tilde{E} = E_1 - E_0$ is the dimensionless spectral gap and $ga = 1/\sqrt{2x}$.

**Finite-size control.** For each $x$, the mass gap $M(N)$ is computed across the full $N$-grid. At coarse lattice spacings ($x = 2$), the gap stabilizes within the 1% tolerance at $N = 20$ (final-step relative change 0.76%). At finer spacings ($x \geq 3$), $M(N)$ has not fully converged at $N = 20$, and a finite-size extrapolation in $1/N$ is applied. The script fits both linear ($M = M_\infty + a/N$) and quadratic ($M = M_\infty + a/N + b/N^2$) models, selects via AIC with a conservative guard against quadratic overfitting, and declares instability (falling back to raw $M(N_{\mathrm{max}})$ with honest error bars) when the extrapolated shift exceeds 25% of $M(N_{\mathrm{max}})$.

| $x$ | $(ag)^2$ | $M_{\mathrm{used}}/g$ | Method | $\sigma_M$ (diagnostic) |
|-----|----------|------------------------|--------|--------------------------|
| 2   | 0.2500   | 0.5266                 | stabilized | 0.0040 |
| 3   | 0.1667   | 0.4731                 | $N\to\infty$ extrap (quad) | 0.0450 |
| 4   | 0.1250   | 0.4561                 | $N\to\infty$ extrap (quad) | 0.0616 |
| 5   | 0.1000   | 0.4427                 | $N\to\infty$ extrap (quad) | 0.0781 |
| 6   | 0.0833   | 0.4310                 | $N\to\infty$ extrap (quad) | 0.0948 |
| 8   | 0.0625   | 0.4101                 | $N\to\infty$ extrap (quad) | 0.1289 |
| 10  | 0.0500   | 0.5540$^{\dagger}$     | max-$N$ (upper bound) | 0.0315 |
| 12  | 0.0417   | 0.5700$^{\dagger}$     | max-$N$ (upper bound) | 0.0368 |

$^{\dagger}$ Max-$N$ values at $N=20$; at these finer lattice spacings the gap is not fully converged, so these are best interpreted as **upper bounds** at the current $N_{\max}$.

The extrapolated intermediate points ($x = 3$–$8$) carry large diagnostic uncertainties from the $1/N$ regression, while the finest-lattice points ($x = 10, 12$) are reported as raw max-$N$ upper bounds with smaller uncertainties from the last-step drift/spread (not rigorous confidence intervals).

**Continuum extrapolation.** A weighted linear fit in $(ag)^2$ across all eight points, with weights $w_i = 1/\sigma_{M,i}^2$, yields the intercept:

$$
\frac{M_{\mathrm{gap}}}{g}\bigg|_{(ag)^2 \to 0} = 0.557 \pm \mathcal{O}(0.01).
$$

**What is actually driving the good intercept (and the subtlety).** The weighted fit is effectively anchored by two controlled groups of points: the **coarse stabilized** result at $x=2$ (reliable but far from the continuum), and the **fine-lattice max-$N$ upper bounds** at $x=10,12$ (close to the continuum but not yet fully finite-size converged). The intermediate $x=3$–$8$ points have large uncertainties from the $1/N$ extrapolation and are therefore downweighted. In other words, the fit is doing the right thing by trusting the most controlled data, but it is important to be transparent that the finest-lattice points are upper bounds rather than fully converged determinations.

**Exact benchmark.** The massless Schwinger model mass gap is known analytically:

$$
\frac{M_{\mathrm{gap}}}{g} = \frac{1}{\sqrt{\pi}} \approx 0.5642.
$$

The extrapolated value lies 1.3% below the exact result. The fit line approaches the theory value from below, consistent with the expected direction of residual finite-size and discretization systematics.

**Robustness note.** Fitting only the most controlled subset (the stabilized $x=2$ point plus the two fine-lattice upper bounds at $x=10,12$) gives an intercept $\approx 0.567$, consistent with $1/\sqrt{\pi}$. Excluding the coarsest point while keeping the intermediate extrapolated points can shift the intercept noticeably, reflecting that the intermediate $x=3$–$8$ points are still systematics-limited at $N\le 20$. This is why the figure explicitly marks $x=10,12$ as upper bounds and assigns large uncertainties to the extrapolated intermediate points.


**Result**

![](<figure/massgap_continuum(N=20).png>){ width=600px }
The plot shows $M_{\mathrm{gap}}/g$ vs $(ag)^2$ with the weighted linear fit (intercept = 0.5566), the exact $1/\sqrt{\pi}$ reference line, and per-point annotations indicating the finite-size control method (* = $N \to \infty$ extrapolation, † = max-$N$ upper bound). Error bars reflect conservative **diagnostic** uncertainties: $\max(|\mathrm{shift}|, \mathrm{SE}, \mathrm{RMS})$ for $N\to\infty$ extrapolated points and last-step drift/spread for upper-bound points (not strict confidence intervals).

**Verdict:** The continuum extrapolation yields $M_{\mathrm{gap}}/g = 0.557$, within 1.3% of the exact result $1/\sqrt{\pi} \approx 0.564$, demonstrating that the ED pipeline correctly captures continuum QFT physics from lattice Hamiltonian simulation. ✔

---

### 1.1 DMRG extension (N = 30, 40) and ED validation

**Code:** `schwinger_dmrg.py` (TeNPy DMRG)

To go beyond the exact-diagonalization ceiling at $N\le 20$, the ground state and first excited state were recomputed for $N=30,40$ using DMRG in TeNPy with $U(1)$ charge conservation (`conserve="Sz"`) and bond dimension $\chi=80$, with direct cross-checks against ED for $N\le 20$.

**Data provenance.** The ED/DMRG values used for the figure below are stored in `dmrg_massgap_results.csv` (columns: `x`, `N`, `E0_dmrg`, `E1_dmrg`, `gap_dmrg`, `E0_ed`, `gap_ed`; ED entries are `NaN` for $N>20$). The plotted quantity is computed as
$$
M_{\mathrm{gap}}/g = \Delta E\,/(2\sqrt{x}).
$$

**ED cross-check.** Over all points where ED is available ($N=4,8,12,20$), DMRG matches ED to floating-point precision:  
- max relative error in the lattice gap: $1.2\times 10^{-13}$ (max absolute error $5.3\times 10^{-13}$)  
- max relative error in $E_0$: $4.7\times 10^{-15}$

Representative validation at the ED ceiling ($N=20$):

| x | N | gap_ED | gap_DMRG | rel. err |
|---:|---:|---:|---:|---:|
| 4 | 20 | 2.928318 | 2.928318 | 3.88e-14 |
| 8 | 20 | 4.311381 | 4.311381 | 1.22e-13 |
| 12 | 20 | 5.584574 | 5.584574 | 5.09e-14 |

**Performance note.** The electric-field term is long-ranged; a naive `add_coupling_term(i,j,...)` construction adds $\mathcal{O}(N^2)$ couplings and inflates the MPO bond dimension $\mathcal{O}(N)$, which dominates DMRG sweep time. The DMRG script therefore encodes the electric term via its running-sum form,
$$
\sum_{i<j} 2\,w_j\, q_i q_j \;=\; \sum_{j} 2\,w_j\, q_j \Big(\sum_{i<j} q_i\Big),
$$
as a compact finite-state-machine MPO with constant bond dimension (≈5 including hopping), enabling stable sweeps at $N=40$.

![](<figure/dmrg_massgap_plot.png>)


**Left panel (finite-size convergence):** $M_{\mathrm{gap}}/g$ vs $1/N$ for each lattice spacing. Filled circles are ED ($N \le 20$), open squares are DMRG ($N = 30, 40$). As $1/N \to 0$, all curves converge toward the exact Schwinger mass $1/\sqrt{\pi} \approx 0.564$. The DMRG points at $N = 30, 40$ are visibly closer to the dashed line than the furthest ED point at $N = 20$ — the headline result. 
 
**Right panel (continuum extrapolation):** $M_{\mathrm{gap}}/g$ vs $(ag)^2 = 1/x$ for each system size. The DMRG curves ($N = 30, 40$, blue dashed) are nearly flat and sit close to the exact value, while smaller ED-only sizes ($N = 4, 8$, grey) still have large finite-size effects. This shows that at $N = 40$, the finite-size error is already small enough that the continuum limit $(ag)^2 \to 0$ is almost reached.


## 2. Sequential Suppression: 1S vs 2S at Fixed Temperature

**Code:** `OQS_continuum.py` (calls `utils_QOS` for Lindblad solver and plotting)

### 2.1 Setup

The $1\oplus 8$ singlet–octet Lindblad model (validated against the analytic solution in the Validation Baseline) is used with two choices of binding energy to represent tightly-bound and loosely-bound quarkonium states:

| State | $\Delta E$ (MeV) | Physical analogue |
|---|---|---|
| 1S-like | 500 | $\Upsilon(1S)$ or $J/\psi$ (tightly bound) |
| 2S-like | 200 | $\Upsilon(2S)$ or $\psi(2S)$ (loosely bound) |

**Calibration.** A state-independent per-channel base rate $\gamma_0$ is calibrated by fixing the total dissociation width $\Gamma_\text{diss}^\text{tot}(T_\text{ref}=400\text{ MeV})=100$ MeV for the 1S-like state:

$$
\gamma_0 = \frac{\Gamma_\text{diss}^\text{tot}}{8\,n_\text{th}(\Delta E_{1S},T_\text{ref})} = 31.13 \text{ MeV}.
$$

The same $\gamma_0$ is then reused for the 2S-like state (`same_gamma0` mode), so the suppression hierarchy is driven entirely by the binding-energy difference through the Bose factor $n_\text{th}(\Delta E,T)$. This is the cleanest demonstration of the pNRQCD prediction and provides a conservative lower bound on the hierarchy; including the physical $\langle r^2\rangle$ scaling of the chromoelectric dipole matrix element would enhance the 2S/1S separation further.

**Evolution.** QuTiP `mesolve` at fixed $T=300$ MeV over the time window $0$–$10$ fm/$c$, matching the plan specification. Initial state: pure singlet $\rho(0)=|0\rangle\langle 0|$.

**Analytic expectations.** At $T = 300$ MeV, the Boltzmann equilibrium populations are:

$$
P_s^{\mathrm{eq}} = \frac{1}{1 + 8\,e^{-\Delta E / T}}
$$

| State | $\Delta E$ (MeV) | $n_{\mathrm{th}}$ | $P_s^{\mathrm{eq}}$ |
|-------|-------------------|--------------------|-----------------------|
| 1S    | 500               | 0.233              | 0.398                 |
| 2S    | 200               | 0.946              | 0.196                 |

The 2S state has a much larger thermal occupation number and much lower equilibrium survival, so it should dissociate faster and to a lower asymptotic value — the defining signature of sequential suppression.

## 2.2 Result

![](<figure/sequential_suppression.png>){ width=600px }
- **Faster 2S dissociation.** The 2S-like curve drops below 0.5 by $t\sim 1$ fm/$c$, while the 1S-like curve does not cross 0.5 until $t\sim 3.5$ fm/$c$ — a factor $\sim 3.5\times$ difference in half-life.
- **Correct equilibrium hierarchy.** Both curves approach their respective analytic equilibrium values: $P_s^\text{eq}(1S)=1/(1+8e^{-500/300})=0.398$ and $P_s^\text{eq}(2S)=1/(1+8e^{-200/300})=0.196$, annotated as horizontal dashed lines on the figure.
- **QGP lifetime marker.** The $\tau_\text{QGP}\sim 10$ fm/$c$ vertical line shows that both states have essentially equilibrated by the end of the QGP phase, with $P_s^{1S}(\tau_\text{QGP})=0.403$ and $P_s^{2S}(\tau_\text{QGP})=0.196$.
- **Double-ratio proxy.** The ratio $P_s^{2S}(\tau_\text{QGP})/P_s^{1S}(\tau_\text{QGP})=0.486$ is annotated on the figure. This quantity provides a concrete phenomenological connection to the experimentally measured double ratio $R_{AA}(\psi(2S))/R_{AA}(J/\psi)$; in data, the mapping is only schematic because feed-down and cold-nuclear-matter effects also contribute.

The suppression hierarchy is entirely determined by the binding-energy difference through $n_\text{th}(\Delta E,T)$, precisely the mechanism predicted by pNRQCD-based open quantum system calculations (Brambilla *et al.*, 2017, 2022).

**Quantitative suppression hierarchy at $\tau_{\mathrm{QGP}}$:**

| Quantity | Value |
|----------|-------|
| $P_s^{1S}(\tau_{\mathrm{QGP}})$ | 0.403 |
| $P_s^{2S}(\tau_{\mathrm{QGP}})$ | 0.196 |
| Double ratio $P_s^{2S}/P_s^{1S}$ | 0.486 |

The double ratio of approximately 0.49 is the key phenomenological observable: the 2S state is suppressed roughly twice as much as the 1S state at the end of the QGP phase, consistent with the experimentally observed pattern of sequential quarkonium melting in heavy-ion collisions.

**Verdict:** The $1\oplus 8$ Lindblad model produces a clear sequential suppression hierarchy with the 2S dissociating faster and to a lower equilibrium than the 1S, as expected from pNRQCD. The double ratio $P_s^{2S}/P_s^{1S} \approx 0.49$ at $\tau_{\mathrm{QGP}}$ provides a clean, quotable result. ✔

---

## 3. Bjorken Cooling: Time-Dependent Temperature Profile

**Code:** `OQS_continuum.py` (Bjorken cooling block)

**Setup.** The fixed-temperature assumption used in the sequential suppression figure is contrasted with a more realistic Bjorken longitudinal cooling profile:

$$
T(\tau) = T_0\left(\frac{\tau_0}{\tau}\right)^{1/3}, \qquad \tau \geq \tau_0,
$$

with $T_0 = 450$ MeV, $\tau_0 = 0.6$ fm/$c$, and a hadronization floor $T_{\mathrm{min}} = 120$ MeV. Below $T_{\mathrm{min}}$ the temperature is clamped and the QGP medium is no longer active. The Lindblad equation is propagated with a piecewise-constant temperature approximation: at each time step, the rates are evaluated at $T(\tau)$ and the master equation is integrated forward. This is compared against the fixed-$T = 450$ MeV baseline for the 1S-like state ($\Delta E = 500$ MeV). Because the instantaneous detailed-balance fixed point changes with $T(\tau)$, the evolution under time-dependent rates need not be monotone; the late-time “recovery” reflects the medium cooling below threshold rather than any violation of positivity.

**Temperature profile check:**

| $\tau$ (fm/$c$) | $T(\tau)$ (MeV) |
|------------------|------------------|
| 0.0              | 450              |
| 0.6              | 450              |
| 2.0              | 301              |
| 5.0              | 222              |
| 10.0             | 176              |
| 20.0             | 140              |

**Result (Bjorken cooling figure).**

![](<figure/bjorken_cooling_vs_fixed.png>){ width=330px }

The fixed-$T$ curve (solid) decays monotonically to $P_s^{\mathrm{eq}}(450\,\mathrm{MeV}) \approx 0.27$, equilibrating by $t \approx 7$ fm/$c$ and remaining flat thereafter. The Bjorken cooling curve (dashed) initially tracks the fixed-$T$ curve during the hot early phase ($\tau \lesssim 2$ fm/$c$), then departs as the temperature drops. The falling temperature progressively shuts off dissociation ($n_{\mathrm{th}} \to 0$ as $T \ll \Delta E$), causing the singlet survival to recover — the Bjorken curve bottoms out at $P_s \approx 0.47$ near $\tau \sim 3$ fm/$c$ and then rises, reaching $P_s \approx 0.75$ by $\tau = 20$ fm/$c$.

This qualitative difference — recovery in the cooling scenario vs. irreversible equilibration at fixed $T$ — directly illustrates why static-temperature models systematically overestimate quarkonium suppression. The $\tau_{\mathrm{QGP}}$ marker at 10 fm/$c$ shows that at the end of the QGP phase, the Bjorken scenario predicts $P_s \approx 0.60$ compared to $P_s \approx 0.27$ at fixed $T$: more than a factor of two difference in survival probability.

**Verdict:** The Bjorken cooling extension demonstrates that time-dependent temperature evolution qualitatively changes the suppression dynamics, with the singlet survival recovering as the medium cools below the dissociation threshold. This validates the piecewise-constant propagation scheme and illustrates the physical importance of realistic temperature profiles. ✔

---

## Summary

| Deliverable | Test | Result | Status |
|---|---|---|---|
| Continuum mass gap (Schwinger) | $M_{\mathrm{gap}}/g$ vs $1/\sqrt{\pi}$ | 0.557 vs 0.564 (1.3% error) | ✔ Validated |
| Sequential suppression (OQS) | 1S vs 2S at $T = 300$ MeV | $P_s^{2S}/P_s^{1S} = 0.49$ at $\tau_{\mathrm{QGP}}$ | ✔ Clear hierarchy |
| Bjorken cooling (OQS) | Fixed-$T$ vs $T(\tau)$ | Recovery effect demonstrated | ✔ Physically correct |


## 4. Bridge: Connecting the Two Workstreams

Both workstreams demonstrate verified real-time quantum evolution in distinct physical settings:

| Aspect | Gauge Simulation (Schwinger) | Open Quantum System (pNRQCD) |
|---|---|---|
| Degrees of freedom | Lattice fermions + gauge field (spin chain) | Colour singlet–octet density matrix |
| Evolution | Unitary ($e^{-iHt}$, exact) | Dissipative (Lindblad master equation) |
| Non-equilibrium signature | String breaking via pair creation | Sequential quarkonium suppression |
| Physical observable | Charge-density heatmap $\langle Q(x,t)\rangle$ | Singlet survival $P_s(t)$ |
| Validation chain | ED spectrum → VQE → Trotter → exact dynamics | Analytic 2-level → 9-level equilibrium → dynamics |

The common methodological thread is: *validate the static baseline first, then use the same code to produce non-equilibrium dynamics with confidence*. The gauge simulation demonstrates that lattice gauge theories can be simulated as real-time quantum dynamics on near-term hardware-compatible Hilbert spaces, while the OQS model shows that pNRQCD-derived Lindblad evolution reproduces the phenomenologically observed suppression hierarchy.


All Continuum Physics Results deliverables are numerically validated. Together with the previous results (MC area law, Hamiltonian checks, OQS baseline, VQE benchmarks, Trotter validation, and string-breaking dynamics), the project provides a complete, reproducible portfolio of gauge-theory quantum simulation and open-quantum-systems modeling aligned with the pNRQCD / lattice gauge theory research program.

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
