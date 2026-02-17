import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_QOS import (
    calibrate_gamma0,
    gamma_total,
    init_plot_style,
    init_temp_color_map,
    plot_analytic_error,
    plot_dynamics_with_optional_analytic,
    ps_2level_analytic,
    solve_ps_fixed_t,
)


PLOT_DPI = 300
FIGSIZE = (7.5, 5.5)

DELTA_E_BASE = 500.0  # MeV
CALIB_T_REF = 400.0   # MeV
CALIB_WIDTH_REF = 100.0  # MeV
TEMPS_DYNAMICS = [200, 300, 450]  # MeV

TIME_MAX_FM = 20.0
NUM_STEPS = 500
TAU_QGP_FM = 10.0

OUT_M1_DYN = "m1_2level_survival_vs_time.png"
OUT_M1_ERR = "m1_2level_analytic_error.png"


def main():
    print("--- OQS Milestone 1 (2-level baseline) ---")

    init_plot_style(PLOT_DPI)
    init_temp_color_map(TEMPS_DYNAMICS)

    t_fm = np.linspace(0.0, TIME_MAX_FM, NUM_STEPS)
    n_octets = 1

    gamma0 = calibrate_gamma0(DELTA_E_BASE, CALIB_T_REF, CALIB_WIDTH_REF, n_octets)
    print(f"Calibrated gamma0={gamma0:.6f} MeV for n_octets={n_octets}")
    for temp in TEMPS_DYNAMICS:
        print(f"  T={temp} MeV -> Gamma_total={gamma_total(DELTA_E_BASE, temp, gamma0, n_octets):.3f} MeV")

    ps_num = solve_ps_fixed_t(DELTA_E_BASE, gamma0, n_octets, TEMPS_DYNAMICS, t_fm)
    ps_ana = {temp: ps_2level_analytic(t_fm, DELTA_E_BASE, temp, gamma0) for temp in TEMPS_DYNAMICS}
    err = {temp: np.abs(ps_num[temp] - ps_ana[temp]) for temp in TEMPS_DYNAMICS}

    plot_dynamics_with_optional_analytic(
        t_fm=t_fm,
        ps_dict=ps_num,
        title=rf"Milestone 1: 2-level OQS (1⊕1), $\Delta E={DELTA_E_BASE:.0f}$ MeV",
        outfile=OUT_M1_DYN,
        tau_qgp_fm=TAU_QGP_FM,
        figsize=FIGSIZE,
        plot_dpi=PLOT_DPI,
        analytic_dict=ps_ana,
    )

    plot_analytic_error(
        t_fm=t_fm,
        err_dict=err,
        title="2-level de-risking: QuTiP vs analytic",
        outfile=OUT_M1_ERR,
        figsize=FIGSIZE,
        plot_dpi=PLOT_DPI,
    )

    print("Generated:")
    print(f"  {OUT_M1_DYN}")
    print(f"  {OUT_M1_ERR}")


if __name__ == "__main__":
    main()
