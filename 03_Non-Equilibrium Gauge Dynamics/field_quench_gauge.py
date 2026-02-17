import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools


# =============================================================================
# 1. PARAMETERS & CONFIGURATION
# =============================================================================

# Lattice Parameters
N = 14
assert N % 2 == 0, "Even N recommended so the staggered vacuum sits in the ΣQ=0 sector."
N_CELLS = N // 2
N_LINKS = N - 1

# Physics Parameters (dimensionless demo units)
G_COUP = 1.0
X_COUP = 1.0
DT = 0.05
T_MAX = 15.0
STEPS = int(T_MAX / DT) + 1
TIMES = np.linspace(0.0, T_MAX, STEPS)

# Background-field quench protocol (Milestone 3)
E0_INIT = 1.0
E0_EVOLVE = 0.0

# Work in the physically relevant global-charge sector:
# ΣQ_n = 0  <=>  Σ n_n = N/2 for staggered background (no net charge).
N_UP = N // 2

# If GS(H(E0=1)) is screened (common at light m), prepare a string-like *initial state*
# at the SAME mass using a penalty term that disfavors screening charges.
SCREENING_MEAN_E_THRESHOLD = 0.80 * E0_INIT
STRING_PENALTY_LAMBDA = 2.0  # λ in λ * Σ_links (L - E0_INIT)^2  (tune 1–5)

# Regimes to compare
M_HEAVY = 2.5
M_LIGHT = 0.1

# Plot controls
HEATMAP_SCALE_MODE = "per_column"  # "per_column" (recommended) or "shared"
PLOT_L2 = True                      # set False to drop the <L^2> twin-axis curve
# Bandwidth estimate (extra eigsh call); fine for N=14 sector ED, expensive at larger N.
COMPUTE_E_MAX = True              # set True to estimate bandwidth via max eigenvalue

# Output filename
OUTFILE = "week3_track1_string_breaking_v7.png"

# Line colors (explicit so twin-axis curves don't accidentally look identical)
COL_HEAVY = "tab:blue"
COL_LIGHT = "tab:orange"

# =============================================================================
# 2. SECTOR BASIS
# =============================================================================

def sector_basis(N, n_up):
    """All bitstrings with exactly n_up spins up (Hamming weight n_up)."""
    basis = []
    for comb in itertools.combinations(range(N), n_up):
        s = 0
        for i in comb:
            s |= (1 << i)
        basis.append(s)
    return np.array(basis, dtype=np.uint32)

BASIS = sector_basis(N, N_UP)
DIM = len(BASIS)
STATE_INDEX = {int(s): i for i, s in enumerate(BASIS)}

VAC_SPIN = np.array([1.0 if (n % 2 == 1) else 0.0 for n in range(N)], dtype=np.float64)
EVEN = np.array([n for n in range(N) if n % 2 == 0], dtype=int)
ODD = np.array([n for n in range(N) if n % 2 == 1], dtype=int)

# =============================================================================
# 3. PRECOMPUTE OCCUPANCIES / FLUX
# =============================================================================

def precompute_occ_and_Llinks(N, basis, E0):
    """Return occ[n,i] and L_links[ell,i] for ell=0..N-2 (internal links)."""
    dim = len(basis)
    occ = np.zeros((N, dim), dtype=np.float64)
    for n in range(N):
        occ[n, :] = ((basis >> n) & 1).astype(np.float64)

    background = np.array([n % 2 for n in range(N)], dtype=np.float64)[:, None]
    Q = occ - background
    L = np.cumsum(Q, axis=0) + E0
    L_links = L[:N - 1, :]
    return occ, L_links

OCC_INIT, L_INIT = precompute_occ_and_Llinks(N, BASIS, E0_INIT)
OCC_EV,   L_EV   = precompute_occ_and_Llinks(N, BASIS, E0_EVOLVE)

# =============================================================================
# 4. HAMILTONIAN (in fixed sector)
# =============================================================================

def build_schwinger_hamiltonian_sector(
    N, m, g, E0, x, basis, state_index, occ, L_links,
    penalty_lambda=0.0, L_links_for_penalty=None, E0_penalty=None
):
    """
    Internal demo-normalization:
      H = x Σ (swap 01↔10)
        + m Σ (-1)^n n_n
        + (g^2/2) Σ_links L_link^2

    Optional penalty for string-like initial-state preparation:
      + penalty_lambda * Σ_links (L_link - E0_penalty)^2

    where the L_link used in the penalty can be provided separately (typically L_INIT).
    """
    dim = len(basis)

    # kinetic term (off-diagonal): swap adjacent 01 <-> 10
    rows, cols, data = [], [], []
    for i in range(N - 1):
        mask = (1 << i) | (1 << (i + 1))
        for col, state in enumerate(basis):
            seg = (int(state) & mask) >> i
            if seg == 1 or seg == 2:
                new_state = int(state) ^ mask
                row = state_index[new_state]
                rows.append(row)
                cols.append(col)
                data.append(x)
    H_hop = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()

    # diagonal mass term
    signs = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(N)], dtype=np.float64)[:, None]
    diag_m = (m * signs * occ).sum(axis=0)

    # diagonal electric term
    J = (g**2) / 2.0
    diag_el = J * np.sum(L_links**2, axis=0)

    diag = diag_m + diag_el

    if penalty_lambda > 0.0:
        if L_links_for_penalty is None or E0_penalty is None:
            raise ValueError("Penalty requested but L_links_for_penalty/E0_penalty not provided")
        pen = np.sum((L_links_for_penalty - E0_penalty)**2, axis=0)
        diag = diag + penalty_lambda * pen

    return (H_hop + sp.diags(diag, 0, shape=(dim, dim), dtype=np.float64)).tocsr()

# =============================================================================
# 5. MEASUREMENTS
# =============================================================================

def measure_Q_sites(psi, occ):
    """Site-resolved staggered charge Q_n = <n_n> - (n mod 2)."""
    probs = np.abs(psi)**2
    exp_occ = occ @ probs
    return exp_occ - VAC_SPIN


def sites_to_unit_cells(Q_sites):
    """Q_cell[x] = Q_{2x} + Q_{2x+1}."""
    return Q_sites[0::2] + Q_sites[1::2]


def measure_L_links(psi, L_links):
    """<L_link(ell)> on internal links, with the provided E0 convention."""
    probs = np.abs(psi)**2
    return L_links @ probs


def measure_L2_links_mean(psi, L_links):
    """Mean over links of <L^2> (electric-energy proxy)."""
    probs = np.abs(psi)**2
    L2 = (L_links**2) @ probs
    return float(np.mean(L2))


def measure_excitation_number(psi, occ):
    """
    Excitation (pair) proxy relative to the staggered vacuum:
      N_ex = Σ_even <n_even> + Σ_odd <1 - n_odd>

    In this project, the post-quench GS of H(E0=0) is (approximately) the staggered vacuum,
    so N_ex is a physically meaningful “excitations above the post-quench vacuum” scalar.
    """
    probs = np.abs(psi)**2
    exp_occ = occ @ probs
    return float(exp_occ[EVEN].sum() + (1.0 - exp_occ[ODD]).sum())

# =============================================================================
# 6. INITIAL STATE PREPARATION
# =============================================================================

def ground_state(H):
    evals, evecs = spla.eigsh(H, k=1, which="SA")
    return float(evals[0]), evecs[:, 0].astype(np.complex128)


def prepare_initial_state(m_val):
    """
    Try GS of H(m, E0=1). If it's screened (mean <L_init> too small),
    prepare a string-like initial state at the SAME mass by adding
    a penalty term λ Σ (L_init - 1)^2 during state preparation only.

    Returns:
      psi0, info_dict
    """
    H_init = build_schwinger_hamiltonian_sector(
        N=N, m=m_val, g=G_COUP, E0=E0_INIT, x=X_COUP,
        basis=BASIS, state_index=STATE_INDEX, occ=OCC_INIT, L_links=L_INIT
    )
    E_init, psi0 = ground_state(H_init)

    L0 = measure_L_links(psi0, L_INIT)
    mean_L0 = float(L0.mean())

    info = {
        "prep_type": "GS(E0=1)",
        "prep_energy": E_init,
        "mean_L_pre": mean_L0,
        "min_L_pre": float(L0.min()),
        "max_L_pre": float(L0.max()),
    }

    if mean_L0 < SCREENING_MEAN_E_THRESHOLD:
        # Constrained prep: SAME mass, add penalty to favor L_init ≈ 1.
        H_prep = build_schwinger_hamiltonian_sector(
            N=N, m=m_val, g=G_COUP, E0=E0_INIT, x=X_COUP,
            basis=BASIS, state_index=STATE_INDEX, occ=OCC_INIT, L_links=L_INIT,
            penalty_lambda=STRING_PENALTY_LAMBDA,
            L_links_for_penalty=L_INIT,
            E0_penalty=E0_INIT,
        )
        E_prep, psi_p = ground_state(H_prep)
        Lp = measure_L_links(psi_p, L_INIT)

        info.update({
            "prep_type": f"Constrained string prep (λ={STRING_PENALTY_LAMBDA:g})",
            "prep_energy": E_prep,
            "mean_L_pre": float(Lp.mean()),
            "min_L_pre": float(Lp.min()),
            "max_L_pre": float(Lp.max()),
        })
        psi0 = psi_p

    return psi0, info

# =============================================================================
# 7. QUENCH SIMULATION
# =============================================================================

def expect_energy(psi, H):
    """Return <psi|H|psi> / <psi|psi> as a real scalar."""
    nrm = np.vdot(psi, psi).real
    if nrm <= 0:
        return np.nan
    return float(np.vdot(psi, H.dot(psi)).real / nrm)


def run_quench(m_val, label):
    print(f"--- Simulating Regime: {label} (m/g = {m_val / G_COUP}) ---")

    print("   Preparing initial state...")
    psi0, info = prepare_initial_state(m_val)
    print(
        "   t=0 check (pre-quench operator, E0=1): "
        f"<L> mean={info['mean_L_pre']:.3f}, min={info['min_L_pre']:.3f}, max={info['max_L_pre']:.3f} "
        f"[{info['prep_type']}]"
    )

    print("   Building evolution Hamiltonian (post-quench H with E0=0)...")
    H_evolve = build_schwinger_hamiltonian_sector(
        N=N, m=m_val, g=G_COUP, E0=E0_EVOLVE, x=X_COUP,
        basis=BASIS, state_index=STATE_INDEX, occ=OCC_EV, L_links=L_EV
    )

    # --- Energy-excess diagnostic (validates constrained prep) ---
    E_gs_evolve, _ = ground_state(H_evolve)
    E0_in_evolve = expect_energy(psi0, H_evolve)
    E_excess = E0_in_evolve - E_gs_evolve

    bandwidth = None
    excess_frac = None

    if COMPUTE_E_MAX:
        Emax, _ = spla.eigsh(H_evolve, k=1, which="LA")
        bandwidth = float(Emax[0] - E_gs_evolve)
        excess_frac = E_excess / bandwidth if bandwidth != 0 else np.nan
        print(
            f"   Energy diagnostic: <H_evolve>={E0_in_evolve:.6f}, E_GS={E_gs_evolve:.6f}, "
            f"excess={E_excess:.6f} (excess/bandwidth≈{excess_frac:.3f})"
        )
    else:
        print(
            f"   Energy diagnostic: <H_evolve>={E0_in_evolve:.6f}, E_GS={E_gs_evolve:.6f}, excess={E_excess:.6f}"
        )

    print("   Evolving under H(E0=0) using expm_multiply...")
    psi_t = spla.expm_multiply(
        -1j * H_evolve, psi0,
        start=0.0, stop=T_MAX, num=STEPS, endpoint=True
    )

    print("   Measuring observables (Q_cell, post-quench <L>, diagnostics)...")

    Qcell_map = np.zeros((STEPS, N_CELLS), dtype=np.float64)
    L_map = np.zeros((STEPS, N_LINKS), dtype=np.float64)      # post-quench field operator (E0=0)

    Eabs_mean = np.zeros(STEPS, dtype=np.float64)             # mean |<L>| (post-quench)
    E2_mean = np.zeros(STEPS, dtype=np.float64)               # mean <L^2> (electric energy proxy)
    N_ex = np.zeros(STEPS, dtype=np.float64)                  # pair-excitation proxy
    loschmidt = np.zeros(STEPS, dtype=np.float64)             # |<psi0|psi(t)>|^2

    psi0_norm = psi0 / np.linalg.norm(psi0)

    for t_idx in range(STEPS):
        psi = psi_t[t_idx]

        # unit-cell charge
        Q_sites = measure_Q_sites(psi, OCC_EV)
        Qcell_map[t_idx, :] = sites_to_unit_cells(Q_sites)

        # post-quench electric field operator (E0=0 convention)
        L_links = measure_L_links(psi, L_EV)
        L_map[t_idx, :] = L_links

        Eabs_mean[t_idx] = np.mean(np.abs(L_links))
        E2_mean[t_idx] = measure_L2_links_mean(psi, L_EV)

        # excitation proxy relative to staggered vacuum
        N_ex[t_idx] = measure_excitation_number(psi, OCC_EV)

        # Loschmidt echo
        amp = np.vdot(psi0_norm, psi / np.linalg.norm(psi))
        loschmidt[t_idx] = float(np.abs(amp)**2)

    return {
        "label": label,
        "m": m_val,
        "prep": info,
        "E_gs_evolve": E_gs_evolve,
        "E0_in_evolve": E0_in_evolve,
        "E_excess": E_excess,
        "bandwidth": bandwidth,
        "excess_frac": excess_frac,
        "Qcell": Qcell_map,
        "L": L_map,
        "Eabs": Eabs_mean,
        "E2": E2_mean,
        "Nex": N_ex,
        "echo": loschmidt,
    }

# =============================================================================
# 8. PLOTTING HELPERS
# =============================================================================

def symmetric_vlim(data, percentile=99.5, floor=0.02):
    v = float(np.percentile(np.abs(data), percentile))
    v = max(v, floor)
    return -v, v


def add_scale_note(ax, vmax, loc=(0.01, 0.98), prefix="scale"):
    ax.text(loc[0], loc[1], f"{prefix}: ±{vmax:.2g}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))


def add_inset_colorbar(ax, im, label):
    """Slim in-axis colorbar so heatmaps don't get squeezed by external colorbars."""
    cax = inset_axes(ax, width="2.8%", height="78%", loc="center right", borderpad=1.1)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb

# =============================================================================
# 9. MAIN & PLOT
# =============================================================================

if __name__ == "__main__":
    heavy = run_quench(M_HEAVY, "Heavy")
    light = run_quench(M_LIGHT, "Light")

    # Heatmap scales
    if HEATMAP_SCALE_MODE == "shared":
        qmin, qmax = symmetric_vlim(np.vstack([heavy["Qcell"], light["Qcell"]]), floor=0.01)
        lmin, lmax = symmetric_vlim(np.vstack([heavy["L"], light["L"]]), floor=0.01)
        qlims = {"heavy": (qmin, qmax), "light": (qmin, qmax)}
        llims = {"heavy": (lmin, lmax), "light": (lmin, lmax)}
    else:
        qlims = {
            "heavy": symmetric_vlim(heavy["Qcell"], floor=0.005),
            "light": symmetric_vlim(light["Qcell"], floor=0.02),
        }
        llims = {
            "heavy": symmetric_vlim(heavy["L"], floor=0.005),
            "light": symmetric_vlim(light["L"], floor=0.02),
        }

    extent_cells = [-0.5, N_CELLS - 0.5, 0.0, T_MAX]
    extent_links = [-0.5, N_LINKS - 0.5, 0.0, T_MAX]

    # Titles with prep transparency
    t_heavy = rf"Heavy ($m/g={heavy['m']:.2g}$): {heavy['prep']['prep_type']}"
    t_light = rf"Light ($m/g={light['m']:.2g}$): {light['prep']['prep_type']}"

    # Row 1: unit-cell charge
    qmin_h, qmax_h = qlims["heavy"]
    qmin_l, qmax_l = qlims["light"]
    fig_q, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(14.8, 4.2), constrained_layout=True)
    im_q1 = ax_q1.imshow(heavy["Qcell"], aspect="auto", origin="lower",
                         extent=extent_cells, cmap="RdBu_r", vmin=qmin_h, vmax=qmax_h)
    ax_q1.set_title(t_heavy)
    ax_q1.set_xlabel("Unit cell x")
    ax_q1.set_ylabel(r"Time ($1/g$)")
    add_inset_colorbar(ax_q1, im_q1, r"$\langle Q_{\mathrm{cell}}(x,t)\rangle$")
    add_scale_note(ax_q1, qmax_h)

    im_q2 = ax_q2.imshow(light["Qcell"], aspect="auto", origin="lower",
                         extent=extent_cells, cmap="RdBu_r", vmin=qmin_l, vmax=qmax_l)
    ax_q2.set_title(t_light)
    ax_q2.set_xlabel("Unit cell x")
    ax_q2.set_yticks([])
    add_inset_colorbar(ax_q2, im_q2, r"$\langle Q_{\mathrm{cell}}(x,t)\rangle$")
    add_scale_note(ax_q2, qmax_l)
    if HEATMAP_SCALE_MODE == "per_column":
        fig_q.suptitle("Note: heatmap color scales are per-regime (see scale tags in each panel).", fontsize=11)
    fig_q.savefig("gauge_string_breaking_row1_charge.png", dpi=240)
    plt.close(fig_q)

    # Row 2: post-quench electric field operator (E0=0 convention)
    lmin_h, lmax_h = llims["heavy"]
    lmin_l, lmax_l = llims["light"]
    fig_l, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(14.8, 4.2), constrained_layout=True)
    im_l1 = ax_l1.imshow(heavy["L"], aspect="auto", origin="lower",
                         extent=extent_links, cmap="RdBu_r", vmin=lmin_h, vmax=lmax_h)
    ax_l1.set_title(r"Post-quench electric field: $\langle L_n(t)\rangle$ (E0=0)")
    ax_l1.set_xlabel("Link index n")
    ax_l1.set_ylabel(r"Time ($1/g$)")
    add_inset_colorbar(ax_l1, im_l1, r"$\langle L_n(t)\rangle$")
    add_scale_note(ax_l1, lmax_h)
    ax_l1.text(
        0.02, 0.08,
        "Lattice-scale oscillations\n(finite-size/confined sector)",
        transform=ax_l1.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none")
    )

    im_l2 = ax_l2.imshow(light["L"], aspect="auto", origin="lower",
                         extent=extent_links, cmap="RdBu_r", vmin=lmin_l, vmax=lmax_l)
    ax_l2.set_title(r"Post-quench electric field: $\langle L_n(t)\rangle$ (E0=0)")
    ax_l2.set_xlabel("Link index n")
    ax_l2.set_yticks([])
    add_inset_colorbar(ax_l2, im_l2, r"$\langle L_n(t)\rangle$")
    add_scale_note(ax_l2, lmax_l)
    if HEATMAP_SCALE_MODE == "per_column":
        fig_l.suptitle("Note: heatmap color scales are per-regime (see scale tags in each panel).", fontsize=11)
    fig_l.savefig("gauge_string_breaking_row2_field_heatmap.png", dpi=240)
    plt.close(fig_l)

    # Row 3: field diagnostics
    # Use explicit colors so the twin-axis dashed curves don't get confused with the solid ones.
    fig_field, ax_field = plt.subplots(1, 1, figsize=(14.8, 3.0), constrained_layout=True)
    ax_field.plot(
        TIMES, heavy["Eabs"], color=COL_HEAVY, linewidth=2,
        label=r"Heavy (solid): $\overline{|\langle L\rangle|}$"
    )
    ax_field.plot(
        TIMES, light["Eabs"], color=COL_LIGHT, linewidth=2,
        label=r"Light (solid): $\overline{|\langle L\rangle|}$"
    )
    ax_field.set_title("Field diagnostics (post-quench frame E0=0)")
    ax_field.set_xlabel(r"Time ($1/g$)")
    ax_field.set_ylabel(r"$\overline{|\langle L\rangle|}$")
    ax_field.grid(True, alpha=0.25)

    if PLOT_L2:
        ax_field2 = ax_field.twinx()
        ax_field2.plot(
            TIMES, heavy["E2"], color=COL_HEAVY, linestyle="--", linewidth=1.8, alpha=0.55,
            label=r"Heavy (dashed): $\overline{\langle L^2\rangle}$"
        )
        ax_field2.plot(
            TIMES, light["E2"], color=COL_LIGHT, linestyle="--", linewidth=1.8, alpha=0.55,
            label=r"Light (dashed): $\overline{\langle L^2\rangle}$"
        )
        ax_field2.set_ylabel(r"$\overline{\langle L^2\rangle}$")

        # Two legends (one per axis)
        h1, l1 = ax_field.get_legend_handles_labels()
        h2, l2 = ax_field2.get_legend_handles_labels()
        ax_field.legend(h1, l1, frameon=False, loc="upper left")
        ax_field2.legend(h2, l2, frameon=False, loc="upper right")
    else:
        ax_field.legend(frameon=False, loc="upper left")
    fig_field.savefig("gauge_string_breaking_row3_field_diagnostics.png", dpi=240)
    plt.close(fig_field)

    # Row 4 left: excitation proxy
    fig_bottom, (ax_n, ax_echo) = plt.subplots(1, 2, figsize=(14.8, 4.0), constrained_layout=True)
    ax_n.plot(TIMES, heavy["Nex"], label=rf"Heavy $m/g={heavy['m']:.2g}$", linewidth=2)
    ax_n.plot(TIMES, light["Nex"], label=rf"Light $m/g={light['m']:.2g}$", linewidth=2)
    ax_n.set_title(r"Excitation proxy above post-quench vacuum: $N_{ex}$")
    ax_n.set_xlabel(r"Time ($1/g$)")
    ax_n.set_ylabel(r"$N_{ex}(t)$")
    ax_n.grid(True, alpha=0.25)
    ax_n.legend(frameon=False)

    # Row 4 right: Loschmidt echo
    ax_echo.plot(TIMES, heavy["echo"], label=rf"Heavy $m/g={heavy['m']:.2g}$", linewidth=2)
    ax_echo.plot(TIMES, light["echo"], label=rf"Light $m/g={light['m']:.2g}$", linewidth=2)
    ax_echo.set_title(r"Loschmidt echo: $|\langle\psi(0)|\psi(t)\rangle|^2$")
    ax_echo.set_xlabel(r"Time ($1/g$)")
    ax_echo.set_ylabel("Echo")
    ax_echo.set_ylim(-0.02, 1.02)
    ax_echo.grid(True, alpha=0.25)
    ax_echo.legend(frameon=False)
    ax_echo.text(0.02, 0.08, "Finite-size recurrences (N=14)", transform=ax_echo.transAxes,
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))
    fig_bottom.savefig("gauge_string_breaking_row4_excitation_echo.png", dpi=240)
    plt.close(fig_bottom)
  
