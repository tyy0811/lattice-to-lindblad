exit
import numpy as np
import matplotlib.pyplot as plt
try:
    import qutip as qt
except ModuleNotFoundError as e:
    raise SystemExit(
        "QuTiP (qutip) is required. Install with one of:\n"
        "  pip install qutip\n"
        "  conda install -c conda-forge qutip\n"
    ) from e


# =============================================================================
# Run switches (handy during iteration)
# =============================================================================
RUN_2LEVEL = True
RUN_9LEVEL = True
RUN_SEQUENTIAL = True
RUN_BJORKEN_COOLING = True

# =============================================================================
# Plot / figure defaults (Milestone 3 "polish pass")
# =============================================================================
PLOT_DPI = 300
FIGSIZE = (7.5, 5.5)

SEQ_CALIB_MODE = "same_gamma0"   # options: "same_gamma0", "independent_total_width", "r2_ratio"
R2_RATIO_2S_TO_1S = 4.0          # crude placeholder if you choose "r2_ratio"

# --- Units ---
HBAR_C = 197.327  # MeV*fm, so t[MeV^-1] = t[fm]/HBAR_C

# --- Physics knobs: "state" interpretation  ---
DELTA_E_STATES = {
    "1S-like": 500.0,  # MeV
    "2S-like": 200.0,  # MeV
}

# --- Calibration: keep total dissociation width fixed at T_ref  ---
CALIB_T_REF = 400.0     # MeV
CALIB_WIDTH_REF = 100.0 # MeV (target total dissociation width at T_ref)

# --- Default time window for fixed-T figures ---
TIME_MAX_FM = 20.0
NUM_STEPS = 500
TEMPS_DYNAMICS = [200, 300, 450]
TAU_QGP_FM = 10.0
SEQ_TIME_MAX_FM = 10.0  # Milestone 4: use 0–10 fm/c window for sequential suppression

# --- Equilibrium validation  ---
T_RANGE_EQ = np.linspace(150, 600, 20)

# --Bjorken cooling  ---
ENABLE_BJORKEN_COOLING = RUN_BJORKEN_COOLING
BJORKEN_T0 = 450.0   # MeV, temperature at tau0
BJORKEN_TAU0 = 0.6   # fm/c (avoid divergence at tau=0)
BJORKEN_TMIN = 120.0 # MeV, clamp lower bound
BJORKEN_TIME_MAX_FM = 20.0
BJORKEN_NUM_STEPS = 250  # fewer steps since it's per-step mesolve

# --- Outputs ---
OUT_W1_DYN = "2level_dynamics_with_analytic.png"    
OUT_W1_VAL = "2level_analytic_error.png"             
OUT_W2_DYN = "9level_dynamics.png"
OUT_W2_EQ  = "9level_equilibrium_check.png"
OUT_SEQ_PREVIEW = "sequential_suppression_preview.png" 
OUT_BJORKEN = "bjorken_cooling_vs_fixed.png"            


# =============================================================================
# Helpers
# =============================================================================

def n_th(energy, T):
    """Bose-Einstein n_B(E,T)."""
    if T <= 0:
        return 0.0
    x = energy / T
    if x > 100:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)

def calibrate_gamma0(delta_E, T_ref, width_ref, n_octets):
    """
    Calibrate per-channel gamma0 from:
      Gamma_total(T_ref) = n_octets * gamma0 * nB(delta_E, T_ref)

    NOTE: gamma0 is phenomenological here. In pNRQCD it is set by the
    chromoelectric correlator κ(T) (lattice-determined) and dipole matrix elements.
    """
    nth = n_th(delta_E, T_ref)
    if nth == 0.0:
        raise ValueError("Calibration failed: n_th=0 (T_ref too low or delta_E too large).")
    return width_ref / (float(n_octets) * nth)


def gamma_total(delta_E, T, gamma0, n_octets):
    """Total dissociation width Gamma_total(T) = n_octets * gamma0 * n_B(delta_E, T)."""
    return float(n_octets) * gamma0 * n_th(delta_E, T)

def analytic_singlet_eq(T, delta_E, n_octets):
    """P_s^eq(T) = 1 / (1 + n_octets * exp(-delta_E/T))."""
    if T <= 0:
        return 1.0
    return 1.0 / (1.0 + float(n_octets) * np.exp(-delta_E / T))

def build_hamiltonian(delta_E, n_octets):
    """(1 + n_octets)-level Hamiltonian: singlet at 0, each octet at +delta_E."""
    dim = 1 + n_octets
    return sum(delta_E * qt.basis(dim, k) * qt.basis(dim, k).dag() for k in range(1, dim))

def build_lindblad_ops(delta_E, T, gamma0, n_octets):
    """
    Collapse ops implement detailed balance:
      L_diss ~ sqrt(gamma0 * nB) |k><0|
      L_rec  ~ sqrt(gamma0 * (1+nB)) |0><k|
    """
    dim = 1 + n_octets
    nth = n_th(delta_E, T)
    rate_diss = gamma0 * nth
    rate_rec  = gamma0 * (1.0 + nth)

    c_ops = []
    for k in range(1, dim):
        c_ops.append(np.sqrt(rate_diss) * qt.basis(dim, k) * qt.basis(dim, 0).dag())
        c_ops.append(np.sqrt(rate_rec)  * qt.basis(dim, 0) * qt.basis(dim, k).dag())
    return c_ops

def solve_ps_fixed_T(delta_E, gamma0, n_octets, temps, t_fm_array):
    """Return dict: T -> P_s(t) using mesolve at fixed T."""
    dim = 1 + n_octets
    t_MeV = t_fm_array / HBAR_C

    rho0 = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    P_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    H = build_hamiltonian(delta_E, n_octets)

    out = {}
    for T in temps:
        c_ops = build_lindblad_ops(delta_E, T, gamma0, n_octets)
        res = qt.mesolve(H, rho0, t_MeV, c_ops, e_ops=[P_s_op])
        out[T] = np.array(res.expect[0], dtype=float)
    return out

# ---  2-level analytic closed form validation ---
def ps_2level_analytic(t_fm_array, delta_E, T, gamma0):
    """
    Closed-form solution ONLY for the 2-level (1⊕1) model (i.e., n_octets = 1).

    For n_octets=1:
      P_eq = 1 / (1 + exp(-ΔE/T))          # degeneracy 1:1
      Γ_relax = gamma0 * (1 + 2 nB(ΔE,T))
      P(t) = P_eq + (1 - P_eq) * exp(-Γ_relax * t)
    """
    nth = n_th(delta_E, T)
    P_eq = 1.0 / (1.0 + np.exp(-delta_E / T))   # DO NOT reuse for n_octets=8
    Gamma_relax = gamma0 * (1.0 + 2.0 * nth)
    t_MeV = t_fm_array / HBAR_C
    return P_eq + (1.0 - P_eq) * np.exp(-Gamma_relax * t_MeV)

# ---  equilibrium validation (robust across QuTiP versions) ---
def steady_state_ps(delta_E, gamma0, n_octets, T):
    dim = 1 + n_octets
    P_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()

    H = build_hamiltonian(delta_E, n_octets)
    c_ops = build_lindblad_ops(delta_E, T, gamma0, n_octets)

    # Use Liouvillian steady state without rho0 kw (avoids SciPy spsolve rho0 error)
    L = qt.liouvillian(H, c_ops)
    rho_ss = qt.steadystate(L)
    return float(qt.expect(P_s_op, rho_ss))

# --- Bjorken cooling profile + piecewise constant propagation ---
def T_bjorken(t_fm, T0=BJORKEN_T0, tau0=BJORKEN_TAU0, Tmin=BJORKEN_TMIN):
    """
    Bjorken cooling: T(tau)=T0*(tau0/tau)^(1/3), for tau>=tau0.
    For tau<tau0, use T0. Clamp below Tmin.
    """
    tau = max(t_fm, tau0)
    T = T0 * (tau0 / tau) ** (1.0 / 3.0)
    return max(T, Tmin)

def solve_ps_bjorken_piecewise(delta_E, gamma0, n_octets, t_fm_array, T_profile_fn):
    """
    Piecewise-constant mesolve per time step (robust across QuTiP versions).
    We need the final density matrix each step, so we *try* to force state storage.
    """
    dim = 1 + n_octets
    H = build_hamiltonian(delta_E, n_octets)
    P_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()

    rho = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    ps = [1.0]

    # QuTiP >=5 prefers dict options; older versions may ignore/raise on options
    opts = {"store_states": True, "progress_bar": None}

    for i in range(len(t_fm_array) - 1):
        t0 = float(t_fm_array[i])
        t1 = float(t_fm_array[i + 1])
        dt_fm = t1 - t0
        dt_MeV = dt_fm / HBAR_C

        if dt_MeV <= 0.0:
            ps.append(ps[-1])
            continue

        t_mid = 0.5 * (t0 + t1)
        T_mid = float(T_profile_fn(t_mid))
        c_ops = build_lindblad_ops(delta_E, T_mid, gamma0, n_octets)

        # Try the modern call (with options + e_ops)
        try:
            res = qt.mesolve(H, rho, [0.0, dt_MeV], c_ops, e_ops=[P_s_op], options=opts)
            # Some versions still won’t populate states even with store_states unless e_ops is None,
            # so we defensively check:
            if hasattr(res, "states") and len(res.states) > 0:
                rho = res.states[-1]
                ps.append(float(res.expect[0][-1]))
            elif hasattr(res, "final_state"):
                rho = res.final_state
                ps.append(float(qt.expect(P_s_op, rho)))
            else:
                raise RuntimeError("mesolve returned no states/final_state")
        except TypeError:
            # Fallback: request states by omitting e_ops/options, then compute expectation manually
            res = qt.mesolve(H, rho, [0.0, dt_MeV], c_ops)
            if hasattr(res, "states") and len(res.states) > 0:
                rho = res.states[-1]
                ps.append(float(qt.expect(P_s_op, rho)))
            elif hasattr(res, "final_state"):
                rho = res.final_state
                ps.append(float(qt.expect(P_s_op, rho)))
            else:
                raise RuntimeError("Fallback mesolve returned no states/final_state")

    return np.array(ps, dtype=float)




# =============================================================================
# Plotting
# =============================================================================

# --- Global matplotlib style (consistent fonts/line weights across all OQS figures) ---
def init_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 2.5,
        "figure.dpi": 100,
        "savefig.dpi": PLOT_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# --- Deterministic temperature -> color mapping (same T-color across figures) ---
_TEMP_COLOR_MAP = {}

def _canonical_T(T):
    return float(np.round(float(T), 6))

def init_temp_color_map(temps):
    # Build a stable mapping from temperature values to matplotlib default-cycle colors.
    global _TEMP_COLOR_MAP
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) == 0:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    uniq = sorted({_canonical_T(t) for t in temps})
    _TEMP_COLOR_MAP = {T: cycle[i % len(cycle)] for i, T in enumerate(uniq)}

def temp_color(T):
    key = _canonical_T(T)
    # If a new temperature slips in, assign it deterministically after existing ones.
    if key not in _TEMP_COLOR_MAP:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(cycle) == 0:
            cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        _TEMP_COLOR_MAP[key] = cycle[len(_TEMP_COLOR_MAP) % len(cycle)]
    return _TEMP_COLOR_MAP[key]

def annotate_tau_qgp(ax, tau_fm=TAU_QGP_FM):
    # Consistent τ_QGP marker + label (more prominent than a bare vline).
    ax.axvline(tau_fm, color="k", linestyle=":", alpha=0.9, linewidth=2.0)
    ax.text(
        tau_fm, 0.98, r"$\tau_{\mathrm{QGP}}$",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )






def _value_at_time(t_fm, y, t0_fm):
    """Linear interpolation y(t0) for monotone t_fm."""
    return float(np.interp(float(t0_fm), np.asarray(t_fm, dtype=float), np.asarray(y, dtype=float)))
def plot_dynamics_with_optional_analytic(t_fm, ps_dict, title, outfile, analytic_dict=None):
    """
    Plot fixed-temperature singlet survival curves.

    - QuTiP curves: colored by temperature.
    - Analytic overlays: neutral black dashed (so the overlay is visible even when it
      lies nearly on top of the numerical curve).
    """
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    # ~15 markers per curve (helps show overlap without clutter)
    markevery = max(1, int(len(t_fm) / 15))

    for T in sorted(ps_dict.keys()):
        ps = np.asarray(ps_dict[T], dtype=float)
        c = temp_color(T)

        ax.plot(
            t_fm, ps,
            color=c, linestyle="-",
            linewidth=2.2, alpha=0.95,
            marker="o", markevery=markevery, markersize=3.2,
            markerfacecolor="none", markeredgewidth=1.0,
            label=rf"QuTiP  $T={T}$ MeV",
            zorder=2,
        )

        if analytic_dict is not None and T in analytic_dict:
            ana = np.asarray(analytic_dict[T], dtype=float)
            ax.plot(
                t_fm, ana,
                color="k", linestyle="--",
                dashes=(6, 3),
                linewidth=2.8, alpha=0.75,
                label=rf"Analytic  $T={T}$ MeV",
                zorder=3,
            )

    annotate_tau_qgp(ax, TAU_QGP_FM)
    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"Singlet survival $P_s(t)$")
    ax.set_xlim(float(t_fm[0]), float(t_fm[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=1, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")

def plot_analytic_error(t_fm, err_dict, title, outfile):
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    for T in sorted(err_dict.keys()):
        err = err_dict[T]
        c = temp_color(T)
        ax.semilogy(t_fm, np.clip(err, 1e-16, None), color=c, label=rf"$T={T}$ MeV")

    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"$|P_s^{\mathrm{QuTiP}}(t) - P_s^{\mathrm{analytic}}(t)|$")
    ax.set_xlim(float(t_fm[0]), float(t_fm[-1]))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_equilibrium(Ts, ps_ana, ps_sim, title, outfile, delta_E, n_octets):
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    ax.plot(
        Ts, ps_ana,
        "k-", linewidth=2.5,
        label=rf"Analytic: $P_s^{{eq}}=(1+{n_octets}e^{{-\Delta E/T}})^{{-1}}$"
    )
    ax.plot(Ts, ps_sim, "o", markersize=6, alpha=0.9, label="QuTiP steady state")

    ax.set_xlabel("Temperature [MeV]")
    ax.set_ylabel(r"Equilibrium singlet population $P_s^{eq}$")
    ax.set_title(title + f"\nΔE={delta_E} MeV, octet degeneracy={n_octets}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")



def plot_sequential_preview(
    t_fm,
    curves_by_state,
    title,
    outfile,
    peq_by_state=None,
    ratio_text=None,
):
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    for state_label, ps in curves_by_state.items():
        ax.plot(t_fm, ps, label=state_label)

    annotate_tau_qgp(ax, TAU_QGP_FM)

    # Annotate equilibrium plateaus (thin horizontal dashed lines)
    if peq_by_state:
        x_text = 0.98 * float(t_fm[-1])
        for short_label, peq in peq_by_state.items():
            peq = float(peq)
            ax.axhline(peq, color="0.25", linestyle="--", linewidth=1.6, alpha=0.75, zorder=1)
            ax.text(
                x_text, peq,
                rf"{short_label}  $P_{{eq}}={peq:.2f}$",
                ha="right", va="bottom",
                fontsize=11, color="0.15",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.75),
                zorder=4,
            )

    # Optional phenomenology-style ratio annotation at τ_QGP
    if ratio_text:
        ax.text(
            0.02, 0.04,
            ratio_text,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )

    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"Singlet survival $P_s(t)$")
    ax.set_xlim(float(t_fm[0]), float(t_fm[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def make_sequential_suppression_fixed_T(
    T_seq=300.0,
    t_fm=None,
    n_octets=8,
    seq_calib_mode=SEQ_CALIB_MODE,
    outfile=OUT_SEQ_PREVIEW,
):
    """
    Overlay P_s(t) at fixed T for multiple ΔE values (e.g. 1S vs 2S).
    Uses QuTiP mesolve via solve_ps_fixed_T.

    seq_calib_mode:
      - "same_gamma0": use gamma0 calibrated on 1S-like and reuse it for all states (clean hierarchy from ΔE)
      - "independent_total_width": calibrate each state separately to same Γ_total at T_ref (less clean hierarchy)
      - "r2_ratio": apply crude <r^2> scaling to gamma0 (optional)
    """
    if t_fm is None:
        t_fm = np.linspace(0, SEQ_TIME_MAX_FM, NUM_STEPS)

    # Clean labels for the final PDF figure
    pretty = {"1S-like": "1S", "2S-like": "2S"}

    curves_for_plot = {}
    peq_for_plot = {}
    ps_at_tau = {}

    # Reference gamma0 from 1S-like (so ΔE drives hierarchy when mode="same_gamma0")
    dE_ref = DELTA_E_STATES["1S-like"]
    gamma0_ref = calibrate_gamma0(dE_ref, CALIB_T_REF, CALIB_WIDTH_REF, n_octets)

    for state_label, dE in DELTA_E_STATES.items():
        if seq_calib_mode == "same_gamma0":
            g0 = gamma0_ref
        elif seq_calib_mode == "independent_total_width":
            g0 = calibrate_gamma0(dE, CALIB_T_REF, CALIB_WIDTH_REF, n_octets)
        elif seq_calib_mode == "r2_ratio":
            g0 = gamma0_ref * (R2_RATIO_2S_TO_1S if state_label == "2S-like" else 1.0)
        else:
            raise ValueError(f"Unknown seq_calib_mode={seq_calib_mode}")

        ps = solve_ps_fixed_T(dE, g0, n_octets, [T_seq], t_fm)[T_seq]

        short = pretty.get(state_label, state_label)
        curves_for_plot[f"{short} (ΔE={dE:.0f} MeV)"] = ps

        peq = analytic_singlet_eq(T_seq, dE, n_octets)
        peq_for_plot[short] = peq

        ps_at_tau[short] = _value_at_time(t_fm, ps, TAU_QGP_FM)

        print(f"  {short}: gamma0={g0:.6f} MeV, P_eq={peq:.3f}, P(τ_QGP)={ps_at_tau[short]:.3f}")

    # Development metadata belongs in the caption/text, not the plot title:
    print(f"  Sequential mode: {seq_calib_mode} (document this choice in the PDF)")

    ratio_text = None
    if "1S" in ps_at_tau and "2S" in ps_at_tau and ps_at_tau["1S"] > 0:
        ratio = ps_at_tau["2S"] / ps_at_tau["1S"]
        ratio_text = f"$P_s^{{2S}}(\\tau_{{QGP}})/P_s^{{1S}}(\\tau_{{QGP}})={ratio:.2f}$"
        print(f"  Double ratio proxy at τ_QGP: {ratio:.3f}")

    title = rf"Sequential quarkonium suppression at $T={T_seq:.0f}$ MeV (1⊕{n_octets} Lindblad)"

    plot_sequential_preview(
        t_fm,
        curves_for_plot,
        title=title,
        outfile=outfile,
        peq_by_state=peq_for_plot,
        ratio_text=ratio_text,
    )
    return curves_for_plot

def plot_bjorken_vs_fixed(t_fm, ps_fixed, t_fm_cool, ps_cool, title, outfile):
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    # Use the same temperature color for the fixed-T baseline (T0) everywhere.
    c0 = temp_color(BJORKEN_T0)
    ax.plot(t_fm, ps_fixed, color=c0, label=rf"Fixed $T={BJORKEN_T0:.0f}$ MeV")
    ax.plot(t_fm_cool, ps_cool, "--", color=c0, alpha=0.9, label=r"Bjorken cooling $T(\tau)$")

    annotate_tau_qgp(ax, TAU_QGP_FM)
    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"Singlet survival $P_s(t)$")
    ax.set_xlim(float(min(t_fm[0], t_fm_cool[0])), float(max(t_fm[-1], t_fm_cool[-1])))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")

def main():
    print("--- OQS ---")

    # Milestone 3: consistent plotting style + stable temperature colors
    init_plot_style()
    init_temp_color_map(list(TEMPS_DYNAMICS) + [BJORKEN_T0, CALIB_T_REF, 300.0])

    # Common time axis for fixed-T figures
    t_fm = np.linspace(0, TIME_MAX_FM, NUM_STEPS)

    # -------------------------------------------------------------------------
    #  Bridge text printed for write-up reuse (no computation)
    # -------------------------------------------------------------------------
    print("\n[Bridge note for final PDF]")
    print("  models open-system real-time dynamics via Lindblad evolution (singlet↔octet).")
    print("  models real-time gauge dynamics via Hamiltonian evolution (string breaking).")
    print("  Both emphasize verified time evolution (exact/analytic checks) and reproducible figures.\n")

    # Choose which ΔE is the primary baseline for the main figures
    baseline_state = "1S-like"
    delta_E_base = DELTA_E_STATES[baseline_state]

    generated = []

    # -------------------------------------------------------------------------
    #  Minimal 2-level (1⊕1) + analytic de-risking
    # -------------------------------------------------------------------------
    if RUN_2LEVEL:
        n_oct_w1 = 1
        gamma0_w1 = calibrate_gamma0(delta_E_base, CALIB_T_REF, CALIB_WIDTH_REF, n_oct_w1)
        print(f"2-level baseline (1⊕1), ΔE={delta_E_base} MeV ({baseline_state})")
        print(f"  Calibrate to Γ_total={CALIB_WIDTH_REF} MeV at T_ref={CALIB_T_REF} MeV")
        print(f"  gamma0 (per-channel) = {gamma0_w1:.6f} MeV")

        print("  Width summary (Γ_total(T) in MeV):")
        for T in TEMPS_DYNAMICS:
            print(f"    T={T}: Γ_total={gamma_total(delta_E_base, T, gamma0_w1, n_oct_w1):.3f}")

        ps_w1 = solve_ps_fixed_T(delta_E_base, gamma0_w1, n_oct_w1, TEMPS_DYNAMICS, t_fm)

        # Analytic overlay + error plot
        analytic_w1 = {T: ps_2level_analytic(t_fm, delta_E_base, T, gamma0_w1) for T in TEMPS_DYNAMICS}
        err_w1 = {T: np.abs(ps_w1[T] - analytic_w1[T]) for T in TEMPS_DYNAMICS}

        for T in TEMPS_DYNAMICS:
            print(f"  Analytic check T={T}: max|Δ|={err_w1[T].max():.3e}, mean|Δ|={err_w1[T].mean():.3e}")

        plot_dynamics_with_optional_analytic(
            t_fm,
            ps_w1,
            title=rf"Minimal: 2-level OQS (1⊕1), ΔE={delta_E_base} MeV",
            outfile=OUT_W1_DYN,
            analytic_dict=analytic_w1
        )

        plot_analytic_error(
            t_fm,
            err_w1,
            title="2-level solver de-risking (QuTiP vs analytic)",
            outfile=OUT_W1_VAL
        )
        generated += [OUT_W1_DYN, OUT_W1_VAL]

    # -------------------------------------------------------------------------
    #  9-level (1⊕8) dynamics + equilibrium validation
    # -------------------------------------------------------------------------
    gamma0_w2 = None
    n_oct_w2 = 8

    if RUN_9LEVEL:
        gamma0_w2 = calibrate_gamma0(delta_E_base, CALIB_T_REF, CALIB_WIDTH_REF, n_oct_w2)
        print(f"\n9-level model (1⊕8), ΔE={delta_E_base} MeV ({baseline_state})")
        print(f"  Calibrate to Γ_total={CALIB_WIDTH_REF} MeV at T_ref={CALIB_T_REF} MeV")
        print(f"  gamma0 (per-channel) = {gamma0_w2:.6f} MeV  (note: ~1/8 of 2-level to keep Γ_total fixed)")

        print("  Width summary (Γ_total(T) in MeV):")
        for T in TEMPS_DYNAMICS:
            print(f"    T={T}: Γ_total={gamma_total(delta_E_base, T, gamma0_w2, n_oct_w2):.3f}")

        ps_w2 = solve_ps_fixed_T(delta_E_base, gamma0_w2, n_oct_w2, TEMPS_DYNAMICS, t_fm)

        plot_dynamics_with_optional_analytic(
            t_fm,
            ps_w2,
            title=rf"9-level OQS (1⊕8), ΔE={delta_E_base} MeV",
            outfile=OUT_W2_DYN,
            analytic_dict=None
        )

        # Equilibrium validation
        print("\nEquilibrium validation P_s^eq(T): QuTiP vs analytic")
        ps_ana = [analytic_singlet_eq(T, delta_E_base, n_oct_w2) for T in T_RANGE_EQ]
        ps_sim = [steady_state_ps(delta_E_base, gamma0_w2, n_oct_w2, float(T)) for T in T_RANGE_EQ]

        plot_equilibrium(
            T_RANGE_EQ,
            ps_ana,
            ps_sim,
            title="Equilibrium validation",
            outfile=OUT_W2_EQ,
            delta_E=delta_E_base,
            n_octets=n_oct_w2
        )
        generated += [OUT_W2_DYN, OUT_W2_EQ]

    # -------------------------------------------------------------------------
    # Sequential suppression preview (1S-like vs 2S-like) at fixed T
    # -------------------------------------------------------------------------
    if RUN_SEQUENTIAL:
        T_seq = 300.0
        t_fm_seq = np.linspace(0, SEQ_TIME_MAX_FM, NUM_STEPS)  # Milestone 4: 0–10 fm/c window
        print(f"\n[Preview] Sequential suppression at fixed T={T_seq} MeV (1⊕8)")
        make_sequential_suppression_fixed_T(
            T_seq=T_seq,
            t_fm=t_fm_seq,
            n_octets=n_oct_w2,
            seq_calib_mode=SEQ_CALIB_MODE,
            outfile=OUT_SEQ_PREVIEW
        )
        generated += [OUT_SEQ_PREVIEW]

    # -------------------------------------------------------------------------
    # Time-dependent temperature profile (Bjorken cooling) optional
    # -------------------------------------------------------------------------
    if ENABLE_BJORKEN_COOLING:
        # Bjorken cooling can run independently of the 9-level static figure generation.
        # If the 1⊕8 calibration wasn't computed above, do it here.
        if gamma0_w2 is None:
            gamma0_w2 = calibrate_gamma0(delta_E_base, CALIB_T_REF, CALIB_WIDTH_REF, n_oct_w2)

        print("\n[Bjorken cooling] Piecewise-constant propagation (1⊕8) vs fixed-T baseline")
        t_fm_cool = np.linspace(0, BJORKEN_TIME_MAX_FM, BJORKEN_NUM_STEPS)

        # Compare at fixed-T = T0 vs cooling from T0
        ps_fixed_T0 = solve_ps_fixed_T(delta_E_base, gamma0_w2, n_oct_w2, [BJORKEN_T0], t_fm_cool)[BJORKEN_T0]
        ps_cool = solve_ps_bjorken_piecewise(
            delta_E_base, gamma0_w2, n_oct_w2, t_fm_cool,
            T_profile_fn=lambda tfm: T_bjorken(tfm, T0=BJORKEN_T0, tau0=BJORKEN_TAU0, Tmin=BJORKEN_TMIN)
        )

        plot_bjorken_vs_fixed(
            t_fm_cool,
            ps_fixed_T0,
            t_fm_cool,
            ps_cool,
            title=rf"Time-dependent $T(\tau)$ vs fixed $T$ (1⊕8), ΔE={delta_E_base} MeV; $T_0$={BJORKEN_T0:.0f} MeV",
            outfile=OUT_BJORKEN
        )
        generated += [OUT_BJORKEN]

        # Also print a few temperatures for sanity
        for tfm in [0.0, 0.6, 2.0, 5.0, 10.0, 20.0]:
            print(f"  T({tfm:.1f} fm) = {T_bjorken(tfm, T0=BJORKEN_T0, tau0=BJORKEN_TAU0, Tmin=BJORKEN_TMIN):.1f} MeV")

    # -------------------------------------------------------------------------
    # Final output list
    # -------------------------------------------------------------------------
    print("\nDone. Generated figures:")
    for f in generated:
        print(f"  {f}")

if __name__ == "__main__":
    main()
