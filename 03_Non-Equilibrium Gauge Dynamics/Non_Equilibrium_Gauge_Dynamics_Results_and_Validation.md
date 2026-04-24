# Non-Equilibrium Gauge Dynamics — Results & Validation

**Real-time dynamics across both workstreams**

This document presents the numerical results for Non-Equilibrium Gauge Dynamics (Non-Equilibrium Gauge Dynamics) and the Continuum Physics Results sequential suppression deliverable. Each section corresponds to one workstream: (1) string breaking via electric-field quench in the gauge-eliminated Schwinger model, and (2) sequential quarkonium suppression in the $1\oplus 8$ singlet–octet Lindblad model. Both workstreams build directly on the validated baselines from the Validation Baseline and the Static Benchmarks.

---

## 1. String Breaking via Electric-Field Quench (Schwinger Model)

**Code:** `field_quench_gauge.py`

### 1.1 Setup and Quench Protocol

The gauge-eliminated **staggered-fermion** Schwinger Hamiltonian (validated at $N=4$ in the Validation Baseline) is used at $N=14$ sites ($N_\text{cells}=7$ unit cells, $N_\text{links}=13$ internal links) in the global-charge-neutral sector $\sum_n Q_n=0$ (Hilbert space dimension $\binom{14}{7}=3432$). We work in units $\hbar=1$ and $g=1$.

Define the site occupation and (staggered-background) charge operators

$$
n_n = \frac{1+\sigma_n^z}{2},\qquad Q_n = n_n - \frac{1-(-1)^n}{2} = \frac{\sigma_n^z+(-1)^n}{2}.
$$

With open boundaries, Gauss's law fixes the link electric field on each link $\ell$ as

$$
L_\ell = E_0 + \sum_{k=0}^{\ell} Q_k,\qquad \ell=0,...,N-2.
$$

The Hamiltonian (in the gauge-eliminated spin basis) is

$$
H = x\sum_{n=0}^{N-2}\big(\sigma_n^+\sigma_{n+1}^-+\sigma_n^-\sigma_{n+1}^+\big)
+ m\sum_{n=0}^{N-1}(-1)^n n_n
+ \frac{1}{2}\sum_{\ell=0}^{N-2} L_\ell^2,
$$

with $x=1/(ag)^2$ and quench parameters $g=1$, $x=1$ (time in units of $1/g$).

**Quench protocol.** A uniform electric string is prepared in the ground state of $H(E_0=1)$, then at $t=0$ the system evolves under $H(E_0=0)$. Two fermion-mass regimes are compared:

| Regime | $m/g$ | Initial state preparation | Physical expectation |
|---|---|---|---|
| Heavy | 2.5 | GS of $H(E_0=1)$ | Pair creation exponentially suppressed; string metastable |
| Light | 0.1 | Constrained string prep ($\lambda=2$) | Copious pair creation screens the string |

At $m/g=0.1$, the true ground state of $H(E_0=1)$ is already screened (mean $\langle L\rangle < 0.80$), so a penalty term $\lambda\sum_\ell(L_\ell - 1)^2$ is added during state preparation only — at the *same* physical mass — to bias the initial state toward a string-like configuration. This avoids the interpretive ambiguity of a simultaneous mass quench.

**Time evolution.** Exact matrix exponentiation via `scipy.sparse.linalg.expm_multiply` (no Trotter error), evolving from $t=0$ to $t_\text{max}=15/g$ in 301 steps.

### 1.2 Energy-Excess Diagnostic

To verify that the constrained-prep initial state is not unphysically high-energy, we compute the excess energy $\Delta E = \langle\psi_0|H_\text{evolve}|\psi_0\rangle - E_\text{GS}^\text{evolve}$ and compare it to the post-quench Hamiltonian bandwidth:

| Regime | $\langle H_\text{evolve}\rangle$ | $E_\text{GS}^\text{evolve}$ | Excess $\Delta E$ | Excess / Bandwidth |
|---|---|---|---|---|
| Heavy ($m/g=2.5$) | $-19.614$ | $-19.678$ | $0.064$ | $0.001$ |
| Light ($m/g=0.1$) | $-6.205$ | $-7.263$ | $1.059$ | $0.024$ |

The heavy regime is prepared directly from the GS of $H(E_0=1)$ and sits only $0.064$ above the post-quench ground state (0.1% of the bandwidth) — essentially a gentle quench with minimal excess energy. The light regime, which uses constrained string preparation, carries an excess of $1.059$ (2.4% of the bandwidth), reflecting the energy cost of maintaining a string-like configuration at light mass. Both values are small compared to the full spectral bandwidth, indicating that the quench does not populate the Hilbert space broadly; the ensuing dynamics reflect controlled string relaxation rather than high-energy scrambling.

**Initial-state validation.** The pre-quench electric field profile confirms the string preparation:

| Regime | $\langle L\rangle$ mean | $\langle L\rangle$ min | $\langle L\rangle$ max | Prep type |
|---|---|---|---|---|
| Heavy | 0.993 | 0.962 | 1.020 | GS($E_0=1$) |
| Light | 0.957 | 0.863 | 1.044 | Constrained ($\lambda=2$) |

Both initial states carry a near-uniform electric field $\langle L\rangle\approx 1$ across the chain, confirming successful string preparation. The light regime shows slightly more spatial variation ($\pm 0.09$ around the mean) due to the penalty-based preparation, but the mean field remains above the $0.80$ screening threshold.

### 1.3 Result: Charge-Density Heatmaps 

The unit-cell charge density $\langle Q_\text{cell}(x,t)\rangle = \langle Q_{2x}(t)\rangle + \langle Q_{2x+1}(t)\rangle$ is measured at each time step and displayed as a space–time heatmap (Fig. 3a: left = heavy, right = light), where $x$ labels unit cells (each unit cell = two staggered sites). The two regimes use independent color scales (per-regime normalization) to preserve readability across the order-of-magnitude amplitude difference.

**Heavy regime ($m/g=2.5$, scale $\pm 0.036$).** The charge density shows only small-amplitude ($\sim 0.03$) boundary-localized oscillations with a coherent period $\sim 3/g$. The bulk of the chain remains nearly charge-neutral at all times. This is consistent with the expected physics: at large fermion mass, Schwinger pair creation is exponentially suppressed ($\propto e^{-\pi m^2/g^2}$), and the electric string is metastable on the simulated timescale.

**Light regime ($m/g=0.1$, scale $\pm 0.21$).** The charge density develops large-amplitude ($\sim 0.2$) structures that propagate inward from the chain boundaries, creating a characteristic "light-cone" pattern of charge separation. Pairs are created throughout the bulk, progressively screening the initial string. By $t\sim 5/g$ the charge distribution has spread across the full chain, signaling complete string breaking.

**Figure 3a (charge density).** Space–time heatmaps of the unit-cell charge $\langle Q_\mathrm{cell}(x,t)\rangle$ after the $E_0{=}1\to 0$ quench (left: heavy $m/g=2.5$; right: light $m/g=0.1$). Color scales are per-regime (see scale tags in-panel).

![](<../figure/gauge_string_breaking_row1_charge.png>){ width=560px }

**Origin of the patch structure (light regime).** The spatially modulated pattern visible in the light-regime heatmap has three distinct temporal phases, each with a clear physical origin:

*Phase 1 — Boundary-initiated pair creation ($t\approx 0$–$3/g$).* The charge signal appears first at the chain boundaries (cells 0 and 6), not in the bulk. The string "frays" at the open ends first because boundary sites have fewer neighbors constraining them, lowering the effective energy barrier to pair creation. The diagonal edges of the early-time charge pattern trace out an approximate Lieb–Robinson light cone: in any local lattice Hamiltonian, excitation propagation is bounded by a maximum velocity that scales as $O(x)$ (set by the hopping strength). The charge front takes roughly $3/g$ to travel from either boundary (cell 0 or 6) to the center (cell 3), giving an empirical front velocity $v\approx 3\,\text{cells}/(3/g)\approx 1$ cell per $1/g$, consistent with an $O(x)$ Lieb–Robinson scale at $x=1$.

*Phase 2 — Wavefront collision and interference ($t\approx 3$–$7/g$).* When the left-moving and right-moving pair wavefronts meet in the center, they interfere. The resulting checkerboard-like patches are standing-wave patterns formed by counter-propagating charge excitations. The spatial period of the patches ($\sim 2$–$3$ unit cells) is set by the dominant quasi-particle momenta excited by the quench. At $m/g=0.1$ the dispersion relation is nearly relativistic ($\omega(k)\approx\sqrt{m^2+k^2}$ in the continuum limit), so many momentum modes are excited with comparable amplitude, creating a broad interference pattern rather than a clean single-frequency standing wave.

*Phase 3 — Boundary reflections and quasi-equilibration ($t>7/g$).* On a finite chain, the charge excitations reflect off both boundaries and re-interfere, creating the persistent patch pattern visible at late times. The pattern does not fully thermalize on the simulated time window; this is consistent with the small system size and the resulting quasi-recurrences from a discrete spectrum (dimension 3432), which allow the dynamics to revisit similar configurations. This is the same finite-size effect visible as the slow modulation in $N_\text{ex}(t)$ and the absence of a clean steady state in $\overline{|\langle L\rangle|}$.

**Why the heavy regime looks completely different.** At $m/g=2.5$, the Schwinger pair-creation rate scales as $\propto e^{-\pi m^2/g^2}=e^{-\pi(6.25)}\approx 3\times 10^{-9}$, so real pair production is exponentially suppressed. The small boundary oscillations visible in the heavy panel are consistent with *virtual* (coherent) fluctuations of the confined string, without appreciable real pair creation. They stay localized at the boundaries and never propagate inward, which is why the bulk is featureless. The Loschmidt echo staying near unity confirms this: the state never actually leaves the string sector.

The qualitative contrast between regimes — static confinement vs. dynamic string breaking — is the central result of the gauge-simulation workstream.

### 1.4 Electric-Field Heatmaps

The post-quench electric field $\langle L_n(t)\rangle$ (measured in the $E_0=0$ frame) provides a complementary view (Fig. 3b):

**Figure 3b (electric field).** Space–time heatmaps of the post-quench electric field $\langle L_n(t)\rangle$ in the $E_0{=}0$ frame (left: heavy; right: light). Color scales are per-regime.

![](<../figure/gauge_string_breaking_row2_field_heatmap.png>){ width=560px }

**Heavy regime (scale $\pm 0.038$).** A regular standing-wave pattern with period $\sim 2$ link spacings, identified as lattice-scale oscillations in the confined sector. The amplitude is small and the pattern is purely coherent — no irreversible field decay is observed. An annotation on the figure flags these as finite-size/lattice artifacts rather than physical pair dynamics.

**Light regime (scale $\pm 0.42$).** Large-amplitude, spatially structured field variations develop rapidly. The dark spots correspond to locations where created pairs have locally screened the string field (driving $\langle L_n\rangle$ negative in the $E_0=0$ frame), while bright spots indicate constructive field enhancement. The spatial correlation between charge patches and field patches is exactly what Gauss's law requires: $L_{n+1}-L_n=Q_n$. By late times the field distribution has equilibrated to a spatially heterogeneous but statistically stationary pattern, reflecting the same standing-wave interference and finite-size quasi-recurrences described above for the charge density.

### 1.5 Scalar Diagnostics

Three scalar observables corroborate the heatmap interpretation (Fig. 3c–d):

**Figure 3c (field diagnostics).** Time traces of $\overline{|\langle L\rangle|}$ and $\overline{\langle L^2\rangle}$ in the post-quench ($E_0{=}0$) frame.

![](<../figure/gauge_string_breaking_row3_field_diagnostics.png>){ width=600px }

**Figure 3d (excitation & echo).** Excitation proxy $N_\mathrm{ex}(t)$ and Loschmidt echo $|\langle\psi(0)|\psi(t)\rangle|^2$ for heavy vs light regimes.

![](<../figure/gauge_string_breaking_row4_excitation_echo.png>){ width=600px }

**Mean absolute field $\overline{|\langle L\rangle|}(t)$.** The heavy regime maintains a nearly constant, small value ($\sim 0.02$) throughout the evolution. The light regime shows an initial spike as the string begins to break, followed by a slow relaxation, with $\overline{|\langle L\rangle|}$ remaining an order of magnitude larger than in the heavy case.

**Excitation proxy $N_\text{ex}(t)$.** Defined as $N_\text{ex}=\sum_\text{even}\langle n_\text{even}\rangle + \sum_\text{odd}\langle 1-n_\text{odd}\rangle$, this counts particle–hole excitations above the staggered vacuum (which is also the post-quench ground state at $E_0=0$). The heavy regime maintains $N_\text{ex}\approx 0.5$ (residual pair fluctuations from the $E_0=1$ ground state), while the light regime rapidly generates $\sim 5$ excitations and sustains this level with small fluctuations — directly quantifying the pair-creation asymmetry.

**Loschmidt echo $|\langle\psi(0)|\psi(t)\rangle|^2$.** The heavy regime stays near unity with small periodic revivals (period $\sim 7/g$), identified as finite-size recurrences from the $N=14$ chain. The light regime drops to $\sim 0.05$ within $t\sim 1/g$ and remains low, confirming that the post-quench state has irreversibly departed the initial string manifold on the accessible timescale. The finite-size recurrences visible in the heavy regime would shift to longer times and diminish in amplitude at larger $N$.

### 1.6 Verdict

The charge-density heatmaps provide an unambiguous non-equilibrium gauge dynamics figure with clearly distinct behavior in the two $m/g$ regimes, satisfying the Non-Equilibrium Gauge Dynamics deliverable (Figure 3). ✔


**Reproducibility (Gauge quench).**
- Script: `field_quench_gauge.py`
- System size: $N=14$ (half-filling sector, dim $\binom{14}{7}=3432$)
- Quench: prep with $E_0=1$ (heavy: true GS; light: constrained string prep with penalty $\lambda=2$), then evolve with $E_0=0$
- Time evolution: exact sparse propagation via `scipy.sparse.linalg.expm_multiply` from $t=0$ to $t_{\max}=15/g$ in 301 steps
- Figures embedded here: `gauge_string_breaking_row1_charge.png`, `gauge_string_breaking_row2_field_heatmap.png`, `gauge_string_breaking_row3_field_diagnostics.png`, `gauge_string_breaking_row4_excitation_echo.png`

---


## Summary

| Deliverable | Test / Observable | Status |
|---|---|---|
| String-breaking heatmap (Figure 3) | Two $m/g$ regimes with qualitatively distinct dynamics | ✔ Validated |
| Energy-excess diagnostic | Heavy: $\Delta E=0.064$; Light: $\Delta E=1.059$ (not unphysically hot) | ✔ Confirmed |
| Loschmidt echo + $N_\text{ex}$ | Quantitative corroboration of string-breaking contrast | ✔ Consistent |




<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
