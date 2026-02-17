import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_QOS import (
    analytic_singlet_eq,
    calibrate_gamma0,
    gamma_total,
    init_plot_style,
    init_temp_color_map,
    plot_dynamics_with_optional_analytic,
    plot_equilibrium,
    solve_ps_fixed_t,
    steady_state_ps,
)


PLOT_DPI = 300
FIGSIZE = (7.5, 5.5)

DELTA_E_BASE = 500.0      # MeV (1S-like baseline)
N_OCTETS = 8              # 1⊕8 Hilbert space
CALIB_T_REF = 400.0       # MeV
CALIB_WIDTH_REF = 100.0   # MeV

TEMPS_DYNAMICS = [200, 300, 450]  # MeV
T_RANGE_EQ = np.linspace(150, 600, 20)

TIME_MAX_FM = 20.0
NUM_STEPS = 500
TAU_QGP_FM = 10.0

OUT_W2_DYN = "9level_dynamics.png"
OUT_W2_EQ = "9level_equilibrium_check.png"


def main():
    print("--- OQS Milestone 2 (+Milestone 3 polish) ---")

    init_plot_style(PLOT_DPI)
    init_temp_color_map(list(TEMPS_DYNAMICS) + [CALIB_T_REF])

    t_fm = np.linspace(0.0, TIME_MAX_FM, NUM_STEPS)
    gamma0 = calibrate_gamma0(DELTA_E_BASE, CALIB_T_REF, CALIB_WIDTH_REF, N_OCTETS)

    print(f"9-level model (1⊕{N_OCTETS}), ΔE={DELTA_E_BASE:.0f} MeV")
    print(f"Calibrate to Γ_total={CALIB_WIDTH_REF} MeV at T_ref={CALIB_T_REF} MeV")
    print(f"gamma0 (per channel) = {gamma0:.6f} MeV")
    for temp in TEMPS_DYNAMICS:
        print(f"  T={temp} MeV -> Γ_total={gamma_total(DELTA_E_BASE, temp, gamma0, N_OCTETS):.3f} MeV")

    ps_dyn = solve_ps_fixed_t(DELTA_E_BASE, gamma0, N_OCTETS, TEMPS_DYNAMICS, t_fm)
    plot_dynamics_with_optional_analytic(
        t_fm=t_fm,
        ps_dict=ps_dyn,
        title=rf"9-level OQS (1⊕{N_OCTETS}), $\Delta E={DELTA_E_BASE:.0f}$ MeV",
        outfile=OUT_W2_DYN,
        tau_qgp_fm=TAU_QGP_FM,
        figsize=FIGSIZE,
        plot_dpi=PLOT_DPI,
        analytic_dict=None,
    )

    print("Equilibrium validation: QuTiP steady state vs analytic curve")
    ps_ana = [analytic_singlet_eq(temp, DELTA_E_BASE, N_OCTETS) for temp in T_RANGE_EQ]
    ps_sim = [steady_state_ps(DELTA_E_BASE, gamma0, N_OCTETS, float(temp)) for temp in T_RANGE_EQ]
    plot_equilibrium(
        temps=T_RANGE_EQ,
        ps_analytic=ps_ana,
        ps_sim=ps_sim,
        title="Equilibrium validation",
        outfile=OUT_W2_EQ,
        delta_e=DELTA_E_BASE,
        n_octets=N_OCTETS,
        figsize=FIGSIZE,
        plot_dpi=PLOT_DPI,
    )

    print("Generated:")
    print(f"  {OUT_W2_DYN}")
    print(f"  {OUT_W2_EQ}")


if __name__ == "__main__":
    main()
