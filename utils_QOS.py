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


HBAR_C = 197.327  # MeV*fm, so t[MeV^-1] = t[fm]/HBAR_C

_TEMP_COLOR_MAP = {}


def n_th(energy, temp):
    """Bose-Einstein n_B(E, T)."""
    if temp <= 0:
        return 0.0
    x = energy / temp
    if x > 100:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def calibrate_gamma0(delta_e, t_ref, width_ref, n_octets):
    """Calibrate per-channel gamma0 from Gamma_total(T_ref)=n_octets*gamma0*n_B."""
    nth = n_th(delta_e, t_ref)
    if nth == 0.0:
        raise ValueError("Calibration failed: n_th=0 (T_ref too low or delta_E too large).")
    return width_ref / (float(n_octets) * nth)


def gamma_total(delta_e, temp, gamma0, n_octets):
    """Total dissociation width Gamma_total(T)."""
    return float(n_octets) * gamma0 * n_th(delta_e, temp)


def analytic_singlet_eq(temp, delta_e, n_octets):
    """P_s^eq(T) = 1 / (1 + n_octets * exp(-delta_E/T))."""
    if temp <= 0:
        return 1.0
    return 1.0 / (1.0 + float(n_octets) * np.exp(-delta_e / temp))


def build_hamiltonian(delta_e, n_octets):
    """(1+n_octets)-level Hamiltonian with octets at +delta_E."""
    dim = 1 + n_octets
    return sum(delta_e * qt.basis(dim, k) * qt.basis(dim, k).dag() for k in range(1, dim))


def build_lindblad_ops(delta_e, temp, gamma0, n_octets):
    """Collapse ops implementing detailed balance for singlet-octet transfer."""
    dim = 1 + n_octets
    nth = n_th(delta_e, temp)
    rate_diss = gamma0 * nth
    rate_rec = gamma0 * (1.0 + nth)

    c_ops = []
    for k in range(1, dim):
        c_ops.append(np.sqrt(rate_diss) * qt.basis(dim, k) * qt.basis(dim, 0).dag())
        c_ops.append(np.sqrt(rate_rec) * qt.basis(dim, 0) * qt.basis(dim, k).dag())
    return c_ops


def solve_ps_fixed_t(delta_e, gamma0, n_octets, temps, t_fm_array):
    """Return dict temp->P_s(t) using mesolve at fixed T."""
    dim = 1 + n_octets
    t_mev = np.asarray(t_fm_array, dtype=float) / HBAR_C

    rho0 = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    p_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    h = build_hamiltonian(delta_e, n_octets)

    out = {}
    for temp in temps:
        c_ops = build_lindblad_ops(delta_e, temp, gamma0, n_octets)
        res = qt.mesolve(h, rho0, t_mev, c_ops, e_ops=[p_s_op])
        out[temp] = np.array(res.expect[0], dtype=float)
    return out


def ps_2level_analytic(t_fm_array, delta_e, temp, gamma0):
    """Closed form for 2-level (1⊕1) model."""
    nth = n_th(delta_e, temp)
    p_eq = 1.0 / (1.0 + np.exp(-delta_e / temp))
    gamma_relax = gamma0 * (1.0 + 2.0 * nth)
    t_mev = np.asarray(t_fm_array, dtype=float) / HBAR_C
    return p_eq + (1.0 - p_eq) * np.exp(-gamma_relax * t_mev)


def steady_state_ps(delta_e, gamma0, n_octets, temp):
    """Compute steady-state singlet population from Liouvillian steady state."""
    dim = 1 + n_octets
    p_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    h = build_hamiltonian(delta_e, n_octets)
    c_ops = build_lindblad_ops(delta_e, temp, gamma0, n_octets)
    l_op = qt.liouvillian(h, c_ops)
    rho_ss = qt.steadystate(l_op)
    return float(qt.expect(p_s_op, rho_ss))


def t_bjorken(t_fm, t0=450.0, tau0=0.6, tmin=120.0):
    """Bjorken cooling profile T(tau)=T0*(tau0/tau)^(1/3), clamped at Tmin."""
    tau = max(float(t_fm), float(tau0))
    temp = float(t0) * (float(tau0) / tau) ** (1.0 / 3.0)
    return max(temp, float(tmin))


def solve_ps_bjorken_piecewise(delta_e, gamma0, n_octets, t_fm_array, t_profile_fn):
    """Piecewise-constant mesolve propagation under time-dependent temperature."""
    dim = 1 + n_octets
    h = build_hamiltonian(delta_e, n_octets)
    p_s_op = qt.basis(dim, 0) * qt.basis(dim, 0).dag()

    rho = qt.basis(dim, 0) * qt.basis(dim, 0).dag()
    ps = [1.0]
    opts = {"store_states": True, "progress_bar": None}

    for i in range(len(t_fm_array) - 1):
        t0 = float(t_fm_array[i])
        t1 = float(t_fm_array[i + 1])
        dt_mev = (t1 - t0) / HBAR_C

        if dt_mev <= 0.0:
            ps.append(ps[-1])
            continue

        t_mid = 0.5 * (t0 + t1)
        temp_mid = float(t_profile_fn(t_mid))
        c_ops = build_lindblad_ops(delta_e, temp_mid, gamma0, n_octets)

        try:
            res = qt.mesolve(h, rho, [0.0, dt_mev], c_ops, e_ops=[p_s_op], options=opts)
            if hasattr(res, "states") and len(res.states) > 0:
                rho = res.states[-1]
                ps.append(float(res.expect[0][-1]))
            elif hasattr(res, "final_state"):
                rho = res.final_state
                ps.append(float(qt.expect(p_s_op, rho)))
            else:
                raise RuntimeError("mesolve returned no states/final_state")
        except TypeError:
            res = qt.mesolve(h, rho, [0.0, dt_mev], c_ops)
            if hasattr(res, "states") and len(res.states) > 0:
                rho = res.states[-1]
                ps.append(float(qt.expect(p_s_op, rho)))
            elif hasattr(res, "final_state"):
                rho = res.final_state
                ps.append(float(qt.expect(p_s_op, rho)))
            else:
                raise RuntimeError("Fallback mesolve returned no states/final_state")

    return np.array(ps, dtype=float)


def init_plot_style(plot_dpi=300):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.5,
            "figure.dpi": 100,
            "savefig.dpi": plot_dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _canonical_temp(temp):
    return float(np.round(float(temp), 6))


def init_temp_color_map(temps):
    """Build deterministic temperature->color mapping."""
    global _TEMP_COLOR_MAP
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    uniq = sorted({_canonical_temp(t) for t in temps})
    _TEMP_COLOR_MAP = {t: cycle[i % len(cycle)] for i, t in enumerate(uniq)}


def temp_color(temp):
    key = _canonical_temp(temp)
    if key not in _TEMP_COLOR_MAP:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not cycle:
            cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        _TEMP_COLOR_MAP[key] = cycle[len(_TEMP_COLOR_MAP) % len(cycle)]
    return _TEMP_COLOR_MAP[key]


def annotate_tau_qgp(ax, tau_qgp_fm=10.0):
    ax.axvline(tau_qgp_fm, color="k", linestyle=":", alpha=0.9, linewidth=2.0)
    ax.text(
        tau_qgp_fm,
        0.98,
        r"$\tau_{\mathrm{QGP}}$",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )


def value_at_time(t_fm, y, t0_fm):
    """Linear interpolation y(t0) for monotone t."""
    return float(np.interp(float(t0_fm), np.asarray(t_fm, dtype=float), np.asarray(y, dtype=float)))


def plot_dynamics_with_optional_analytic(
    t_fm,
    ps_dict,
    title,
    outfile,
    tau_qgp_fm=10.0,
    figsize=(7.5, 5.5),
    plot_dpi=300,
    analytic_dict=None,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    markevery = max(1, int(len(t_fm) / 15))

    for temp in sorted(ps_dict.keys()):
        ps = np.asarray(ps_dict[temp], dtype=float)
        c = temp_color(temp)
        ax.plot(
            t_fm,
            ps,
            color=c,
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
            marker="o",
            markevery=markevery,
            markersize=3.2,
            markerfacecolor="none",
            markeredgewidth=1.0,
            label=rf"QuTiP  $T={temp}$ MeV",
            zorder=2,
        )
        if analytic_dict is not None and temp in analytic_dict:
            ana = np.asarray(analytic_dict[temp], dtype=float)
            ax.plot(
                t_fm,
                ana,
                color="k",
                linestyle="--",
                dashes=(6, 3),
                linewidth=2.8,
                alpha=0.75,
                label=rf"Analytic  $T={temp}$ MeV",
                zorder=3,
            )

    annotate_tau_qgp(ax, tau_qgp_fm)
    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"Singlet survival $P_s(t)$")
    ax.set_xlim(float(t_fm[0]), float(t_fm[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_analytic_error(t_fm, err_dict, title, outfile, figsize=(7.5, 5.5), plot_dpi=300):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for temp in sorted(err_dict.keys()):
        err = np.asarray(err_dict[temp], dtype=float)
        c = temp_color(temp)
        ax.semilogy(t_fm, np.clip(err, 1e-16, None), color=c, label=rf"$T={temp}$ MeV")

    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"$|P_s^{\mathrm{QuTiP}}(t)-P_s^{\mathrm{analytic}}(t)|$")
    ax.set_xlim(float(t_fm[0]), float(t_fm[-1]))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_equilibrium(
    temps,
    ps_analytic,
    ps_sim,
    title,
    outfile,
    delta_e,
    n_octets,
    figsize=(7.5, 5.5),
    plot_dpi=300,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot(
        temps,
        ps_analytic,
        "k-",
        linewidth=2.5,
        label=rf"Analytic: $P_s^{{eq}}=(1+{n_octets}e^{{-\Delta E/T}})^{{-1}}$",
    )
    ax.plot(temps, ps_sim, "o", markersize=6, alpha=0.9, label="QuTiP steady state")
    ax.set_xlabel("Temperature [MeV]")
    ax.set_ylabel(r"Equilibrium singlet population $P_s^{eq}$")
    ax.set_title(title + f"\nΔE={delta_e} MeV, octet degeneracy={n_octets}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_sequential_preview(
    t_fm,
    curves_by_state,
    title,
    outfile,
    tau_qgp_fm=10.0,
    peq_by_state=None,
    ratio_text=None,
    figsize=(7.5, 5.5),
    plot_dpi=300,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for state_label, ps in curves_by_state.items():
        ax.plot(t_fm, np.asarray(ps, dtype=float), label=state_label)

    annotate_tau_qgp(ax, tau_qgp_fm)

    if peq_by_state:
        x_text = 0.98 * float(t_fm[-1])
        for short_label, peq in peq_by_state.items():
            peq = float(peq)
            ax.axhline(peq, color="0.25", linestyle="--", linewidth=1.6, alpha=0.75, zorder=1)
            ax.text(
                x_text,
                peq,
                rf"{short_label}  $P_{{eq}}={peq:.2f}$",
                ha="right",
                va="bottom",
                fontsize=11,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.75),
                zorder=4,
            )

    if ratio_text:
        ax.text(
            0.02,
            0.04,
            ratio_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
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
    plt.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_bjorken_vs_fixed(
    t_fm,
    ps_fixed,
    t_fm_cool,
    ps_cool,
    title,
    outfile,
    fixed_temp=None,
    tau_qgp_fm=10.0,
    figsize=(7.5, 5.5),
    plot_dpi=300,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    if fixed_temp is None:
        c0 = "C0"
        fixed_label = "Fixed T"
    else:
        c0 = temp_color(fixed_temp)
        fixed_label = rf"Fixed $T={fixed_temp:.0f}$ MeV"

    ax.plot(t_fm, ps_fixed, color=c0, label=fixed_label)
    ax.plot(t_fm_cool, ps_cool, "--", color=c0, alpha=0.9, label=r"Bjorken cooling $T(\tau)$")

    annotate_tau_qgp(ax, tau_qgp_fm)
    ax.set_xlabel("Time [fm/c]")
    ax.set_ylabel(r"Singlet survival $P_s(t)$")
    ax.set_xlim(float(min(t_fm[0], t_fm_cool[0])), float(max(t_fm[-1], t_fm_cool[-1])))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def make_sequential_suppression_fixed_t(
    state_delta_e,
    t_seq,
    t_fm,
    n_octets,
    calib_t_ref,
    calib_width_ref,
    seq_calib_mode="same_gamma0",
    r2_ratio_2s_to_1s=4.0,
    tau_qgp_fm=10.0,
    title=None,
    outfile="sequential_suppression_preview.png",
    figsize=(7.5, 5.5),
    plot_dpi=300,
):
    """
    Build and plot fixed-T sequential suppression for multiple states.

    seq_calib_mode:
      - same_gamma0
      - independent_total_width
      - r2_ratio
    """
    curves_for_plot = {}
    peq_for_plot = {}
    ps_at_tau = {}

    if "1S-like" not in state_delta_e:
        raise ValueError('state_delta_e must include key "1S-like" for reference calibration.')

    d_e_ref = state_delta_e["1S-like"]
    gamma0_ref = calibrate_gamma0(d_e_ref, calib_t_ref, calib_width_ref, n_octets)
    pretty = {"1S-like": "1S", "2S-like": "2S"}

    for state_label, d_e in state_delta_e.items():
        if seq_calib_mode == "same_gamma0":
            g0 = gamma0_ref
        elif seq_calib_mode == "independent_total_width":
            g0 = calibrate_gamma0(d_e, calib_t_ref, calib_width_ref, n_octets)
        elif seq_calib_mode == "r2_ratio":
            g0 = gamma0_ref * (r2_ratio_2s_to_1s if state_label == "2S-like" else 1.0)
        else:
            raise ValueError(f"Unknown seq_calib_mode={seq_calib_mode}")

        ps = solve_ps_fixed_t(d_e, g0, n_octets, [t_seq], t_fm)[t_seq]
        short = pretty.get(state_label, state_label)
        curves_for_plot[f"{short} (ΔE={d_e:.0f} MeV)"] = ps

        peq = analytic_singlet_eq(t_seq, d_e, n_octets)
        peq_for_plot[short] = peq
        ps_at_tau[short] = value_at_time(t_fm, ps, tau_qgp_fm)

        print(f"  {short}: gamma0={g0:.6f} MeV, P_eq={peq:.3f}, P(τ_QGP)={ps_at_tau[short]:.3f}")

    ratio_text = None
    if "1S" in ps_at_tau and "2S" in ps_at_tau and ps_at_tau["1S"] > 0:
        ratio = ps_at_tau["2S"] / ps_at_tau["1S"]
        ratio_text = f"$P_s^{{2S}}(\\tau_{{QGP}})/P_s^{{1S}}(\\tau_{{QGP}})={ratio:.2f}$"
        print(f"  Double ratio proxy at τ_QGP: {ratio:.3f}")

    if title is None:
        title = rf"Sequential quarkonium suppression at $T={t_seq:.0f}$ MeV (1⊕{n_octets} Lindblad)"

    plot_sequential_preview(
        t_fm=t_fm,
        curves_by_state=curves_for_plot,
        title=title,
        outfile=outfile,
        tau_qgp_fm=tau_qgp_fm,
        peq_by_state=peq_for_plot,
        ratio_text=ratio_text,
        figsize=figsize,
        plot_dpi=plot_dpi,
    )
    return curves_for_plot
