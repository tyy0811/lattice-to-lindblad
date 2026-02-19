"""
DMRG mass gap for the gauge-eliminated Schwinger model (TeNPy).

Matches the ED conventions in schwinger_continuum_massgap.py exactly:
  H = x * Sum_n (hopping)
    + 0.5 * mu * Sum_n stag_n * z_n      [= mu * Sum stag * Sz]
    + Sum_{l=0}^{N-2} L_l^2              [NO ga^2/2 prefactor]

  where L_l = E0 + Sum_{i<=l} q_i,  q_i = Sz_i + stag_i/2.

Drop into: 04_Continuum Physics Results/
Requires: pip install "physics-tenpy>=1.0"
Usage:    python schwinger_dmrg.py
"""
import math, csv
import numpy as np
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.algorithms import dmrg


# ========================== MPO construction ===============================
#
# MPO bond states (D=5):
#   0: IdL      left vacuum (identity propagation)
#   1: S_run    running sum S_l = Sum_{i<=l} q_i
#   2: Sp_hop   hopping: Sp waiting for Sm on next site
#   3: Sm_hop   hopping: Sm waiting for Sp on next site
#   4: IdR      right vacuum (accumulated Hamiltonian)
#
# Electric cross terms: Sum_{l} (E0+S_l)^2 where S_l = Sum_{i<=l} q_i.
# Expand: Sum_l S_l^2 = Sum_j w_j q_j^2 + Sum_j 2 w_j q_j (Sum_{i<j} q_i)
# The second term is (local operator at j) x (running sum from left).
# This gives bond dim O(1) for the electric term.


def build_schwinger_mpo(lat, x, mu, E0=0.0):
    """Build bond-dim-5 MPO matching ED conventions exactly."""
    L = lat.N_sites
    sites = [lat.site(i) for i in range(L)]

    stag = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(L)])
    b = 0.5 * stag                                           # q_i = Sz + stag/2
    w = np.maximum(0, (L - 1) - np.arange(L)).astype(float)  # links per site

    def linop(c_id=0.0, c_sz=0.0, c_sp=0.0, c_sm=0.0):
        """Represent onsite linear combinations in MPO.from_grids format."""
        terms = []
        if c_id != 0.0:
            terms.append(("Id", float(c_id)))
        if c_sz != 0.0:
            terms.append(("Sz", float(c_sz)))
        if c_sp != 0.0:
            terms.append(("Sp", float(c_sp)))
        if c_sm != 0.0:
            terms.append(("Sm", float(c_sm)))
        return terms if terms else None

    D = 5
    grids = []

    for n in range(L):
        W = [[None]*D for _ in range(D)]

        # identity propagation
        W[0][0] = "Id"

        # running sum: S_n = S_{n-1} + q_n
        W[0][1] = linop(c_id=b[n], c_sz=1.0)  # q_n fed from identity stream
        W[1][1] = "Id"                         # propagate existing sum

        # accumulated H propagation
        W[4][4] = "Id"

        # hopping: x * (Sp_n Sm_{n+1} + h.c.)
        if n < L - 1:
            W[0][2] = linop(c_sp=x)
            W[0][3] = linop(c_sm=x)
        if n > 0:
            W[2][4] = "Sm"
            W[3][4] = "Sp"

        # electric cross terms: running_sum * 2 w_n q_n  (no prefactor)
        if w[n] > 0:
            W[1][4] = linop(c_id=2.0 * w[n] * b[n], c_sz=2.0 * w[n])

        # onsite terms collected into 0 -> 4
        c_id = 0.0
        c_sz = mu * stag[n]                      # mass: mu * stag * Sz
        if w[n] > 0:
            # q_n^2 = (Sz + b_n Id)^2 = (1/4 + b_n^2) Id + 2 b_n Sz
            c_id += w[n] * (0.25 + b[n]**2)      # diagonal electric: w_n * q_n^2
            c_sz += 2.0 * w[n] * b[n]
            c_id += 2.0 * E0 * w[n] * b[n]       # E0 linear: 2 E0 w_n q_n
            c_sz += 2.0 * E0 * w[n]
        if n == 0:
            c_id += (L - 1) * E0**2              # E0^2 constant
        W[0][4] = linop(c_id=c_id, c_sz=c_sz)

        grids.append(W)

    return MPO.from_grids(
        sites, grids, bc='finite', IdL=0, IdR=D-1, mps_unit_cell_width=lat.mps_unit_cell_width
    )


def make_model(L, x, m_over_g, E0=0.0, x_def="tagliacozzo"):
    """Build TeNPy model matching ED conventions."""
    ga = 1.0 / math.sqrt(x) if x_def == "tagliacozzo" else 1.0 / math.sqrt(2*x)
    mu = 2.0 * m_over_g / ga
    site = SpinHalfSite(conserve="Sz")
    lat = Chain(L, site, bc="open", bc_MPS="finite")
    H = build_schwinger_mpo(lat, x=x, mu=mu, E0=E0)
    return MPOModel(lat, H)


def _map_x_def_for_ed(x_def):
    if x_def in ("tagliacozzo", "1_over_ag2"):
        return "1_over_ag2"
    if x_def in ("banuls", "1_over_2ag2"):
        return "1_over_2ag2"
    raise ValueError(f"Unsupported x_def for ED bridge: {x_def}")


# ========================== DMRG ==========================================

def ground_state(model, chi=80, sweeps=20):
    L = model.lat.N_sites
    psi = MPS.from_product_state(
        model.lat.mps_sites(),
        ["up" if i % 2 == 0 else "down" for i in range(L)],
        bc="finite",
        unit_cell_width=model.lat.mps_unit_cell_width,
    )
    info = dmrg.run(psi, model, {
        "mixer": True,
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "max_E_err": 1e-12, "max_sweeps": sweeps, "combine": True,
    })
    return info["E"], psi


def first_excited(model, psi_gs, chi=80, sweeps=20):
    psi = psi_gs.copy()
    psi.perturb(
        randomize_params={"N_steps": 10, "trunc_params": {"chi_max": chi}},
        close_1=True,
    )
    info = dmrg.run(psi, model, {
        "mixer": True,
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "max_E_err": 1e-12, "max_sweeps": sweeps, "combine": True,
    }, orthogonal_to=[psi_gs])
    return info["E"], psi

# ========================== plotting ======================================

def plot_results(csv_path, x_def="tagliacozzo"):
    """
    Two-panel figure:
      Left:  M_gap/g  vs  1/N   for each x  (finite-size convergence)
      Right: M_gap/g  vs  (ag)^2  for each N  (continuum extrapolation)
    ED points = filled circles, DMRG-only = open squares.
    """
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            x = float(r["x"]); N = int(r["N"]); gap = float(r["gap_dmrg"])
            ga = 1.0/math.sqrt(x) if x_def == "tagliacozzo" else 1.0/math.sqrt(2*x)
            mg = gap * ga / 2.0
            has_ed = (r.get("E0_ed", "") != "")
            rows.append(dict(x=x, N=N, ga=ga, ag2=ga**2, mg=mg, has_ed=has_ed))

    xs = sorted(set(r["x"] for r in rows))
    Ns = sorted(set(r["N"] for r in rows))
    exact = 1.0 / math.sqrt(math.pi)

    colors = {4: "#2166ac", 8: "#b2182b", 12: "#1b7837"}
    N_colors = {4: "#bdbdbd", 8: "#969696", 12: "#737373",
                20: "#525252", 30: "#2171b5", 40: "#08519c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left panel: M_gap/g vs 1/N for each x ----
    for x in xs:
        pts = sorted([r for r in rows if r["x"] == x], key=lambda r: r["N"])
        inv_N = [1.0/r["N"] for r in pts]
        mg    = [r["mg"] for r in pts]
        ed_mask   = [r["has_ed"] for r in pts]
        dmrg_mask = [not r["has_ed"] for r in pts]
        c = colors.get(int(x), "#333333")

        # ED points (filled)
        ax1.plot([v for v, m in zip(inv_N, ed_mask) if m],
                 [v for v, m in zip(mg, ed_mask) if m],
                 'o', color=c, ms=7, label=f'x={int(x)} (ED)')
        # DMRG-only points (open squares)
        ax1.plot([v for v, m in zip(inv_N, dmrg_mask) if m],
                 [v for v, m in zip(mg, dmrg_mask) if m],
                 's', color=c, ms=8, mfc='none', mew=1.5, label=f'x={int(x)} (DMRG)')
        # connecting line
        ax1.plot(inv_N, mg, '-', color=c, alpha=0.4, lw=1)

    ax1.axhline(exact, color='black', ls='--', lw=1, alpha=0.7, label=f'Exact $1/\\sqrt{{\\pi}}$')
    ax1.set_xlabel('$1/N$', fontsize=12)
    ax1.set_ylabel('$M_{\\rm gap}/g$', fontsize=12)
    ax1.set_title('Finite-size convergence', fontsize=13)
    ax1.set_xlim(left=0)
    ax1.legend(fontsize=8, ncol=2)

    # ---- Right panel: M_gap/g vs (ag)^2 for each N ----
    for N in Ns:
        pts = sorted([r for r in rows if r["N"] == N], key=lambda r: r["ag2"])
        ag2 = [r["ag2"] for r in pts]
        mg  = [r["mg"] for r in pts]
        has_ed = pts[0]["has_ed"] if pts else True
        c = N_colors.get(N, "#333333")

        if has_ed:
            ax2.plot(ag2, mg, 'o-', color=c, ms=6, lw=1, label=f'N={N} (ED)')
        else:
            ax2.plot(ag2, mg, 's--', color=c, ms=7, mfc='none', mew=1.5, lw=1,
                     label=f'N={N} (DMRG)')

    ax2.axhline(exact, color='black', ls='--', lw=1, alpha=0.7, label=f'Exact $1/\\sqrt{{\\pi}}$')
    ax2.set_xlabel('$(ag)^2 = 1/x$', fontsize=12)
    ax2.set_ylabel('$M_{\\rm gap}/g$', fontsize=12)
    ax2.set_title('Continuum extrapolation', fontsize=13)
    ax2.set_xlim(left=0)
    ax2.legend(fontsize=9)

    fig.suptitle('Schwinger model mass gap: ED validated, DMRG extended ($m/g=0$)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig('dmrg_massgap_plot.png', dpi=200, bbox_inches='tight')
    print("Saved dmrg_massgap_plot.png")
    plt.show()

# ========================== main ==========================================

if __name__ == "__main__":
    try:
        import schwinger_continuum_massgap as ed
        HAS_ED = True
    except ImportError:
        HAS_ED = False
        print("ED module not found -- DMRG-only mode.\n")

    # ---- parameters (edit these) -----------------------------------------
    m_over_g = 0.0
    x_def    = "tagliacozzo"
    E0       = 0.0
    chi      = 80

    runs = [
        (4,  [4, 8, 12, 20, 30, 40]),
        (8,  [4, 8, 12, 20, 30, 40]),
        (12, [4, 8, 12, 20, 30, 40]),
    ]
    # ----------------------------------------------------------------------

    print("=" * 85)
    print(f"Schwinger DMRG  |  m/g={m_over_g}  x_def={x_def}  chi={chi}  D_MPO=5")
    print("=" * 85)

    results = []

    for x, Ns in runs:
        print(f"\n--- x = {x} ---")
        if HAS_ED:
            print(f"{'N':>4}  {'E0_ED':>14}  {'E0_DMRG':>14}  {'|dE0|':>9}"
                  f"  {'gap_ED':>12}  {'gap_DMRG':>12}  {'|dgap|':>9}")
        else:
            print(f"{'N':>4}  {'E0_DMRG':>14}  {'E1_DMRG':>14}  {'gap':>12}")

        for N in Ns:
            model = make_model(N, x, m_over_g, E0, x_def)

            E0d, psi0 = ground_state(model, chi=chi)
            E1d, _    = first_excited(model, psi0, chi=chi)
            gap_d = E1d - E0d

            row = dict(x=x, N=N, E0_dmrg=E0d, E1_dmrg=E1d, gap_dmrg=gap_d)

            if HAS_ED and N <= 20:
                if hasattr(ed, "lowest_energies_sector"):
                    Eed = ed.lowest_energies_sector(
                        x=x, m_over_g=m_over_g, L=N, x_def=x_def, E0=E0, N_ev=2
                    )
                else:
                    ed_x_def = _map_x_def_for_ed(x_def)
                    p = ed.SchwingerParams(
                        N=N, x=x, x_def=ed_x_def, m_over_g=m_over_g, E0=E0
                    )
                    H, _ = ed.build_hamiltonian_sector(p)
                    Eed = ed.lowest_energies(H, k=2)
                gap_e = float(Eed[1] - Eed[0])
                row.update(E0_ed=float(Eed[0]), gap_ed=gap_e)
                print(f"{N:4d}  {Eed[0]:14.10f}  {E0d:14.10f}  {abs(E0d-Eed[0]):9.2e}"
                      f"  {gap_e:12.8f}  {gap_d:12.8f}  {abs(gap_d-gap_e):9.2e}")
            else:
                print(f"{N:4d}  {E0d:14.10f}  {E1d:14.10f}  {gap_d:12.8f}")

            results.append(row)

    out_csv = "dmrg_massgap_results.csv"
    fields = ["x", "N", "E0_dmrg", "E1_dmrg", "gap_dmrg"]
    if HAS_ED:
        fields += ["E0_ed", "gap_ed"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows to {out_csv}")

    # ---- plot results ----
    plot_results(out_csv, x_def=x_def)
















