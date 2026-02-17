import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_QOS import (
    calibrate_gamma0,
    init_plot_style,
    init_temp_color_map,
    make_sequential_suppression_fixed_t,
    plot_bjorken_vs_fixed,
    solve_ps_bjorken_piecewise,
    solve_ps_fixed_t,
    t_bjorken,
)


PLOT_DPI = 300
FIGSIZE = (7.5, 5.5)

DELTA_E_STATES = {
    "1S-like": 500.0,  # MeV
    "2S-like": 200.0,  # MeV
}

N_OCTETS = 8
T_SEQ = 300.0  # MeV
TAU_QGP_FM = 10.0
SEQ_TIME_MAX_FM = 20.0
NUM_STEPS = 500

CALIB_T_REF = 400.0      # MeV
CALIB_WIDTH_REF = 100.0  # MeV
SEQ_CALIB_MODE = "same_gamma0"  # same_gamma0, independent_total_width, r2_ratio
R2_RATIO_2S_TO_1S = 4.0

# Optional: time-dependent temperature profile (Bjorken cooling)
ENABLE_BJORKEN_COOLING = True
BJORKEN_T0 = 450.0         # MeV
BJORKEN_TAU0 = 0.6         # fm/c
BJORKEN_TMIN = 120.0       # MeV
BJORKEN_TIME_MAX_FM = 20.0
BJORKEN_NUM_STEPS = 250

OUT_M4 = "sequential_suppression.png"
OUT_BJORKEN = "bjorken_cooling_vs_fixed.png"


def main():
    print("--- OQS Milestone 4 (Sequential suppression) ---")
    init_plot_style(PLOT_DPI)
    init_temp_color_map([T_SEQ, BJORKEN_T0, CALIB_T_REF])

    generated = []
    t_fm = np.linspace(0.0, SEQ_TIME_MAX_FM, NUM_STEPS)
    title = rf"Sequential suppression at $T={T_SEQ:.0f}$ MeV (1⊕{N_OCTETS} Lindblad)"
    print(f"Mode={SEQ_CALIB_MODE}, T={T_SEQ} MeV, window=[0, {SEQ_TIME_MAX_FM}] fm/c")

    make_sequential_suppression_fixed_t(
        state_delta_e=DELTA_E_STATES,
        t_seq=T_SEQ,
        t_fm=t_fm,
        n_octets=N_OCTETS,
        calib_t_ref=CALIB_T_REF,
        calib_width_ref=CALIB_WIDTH_REF,
        seq_calib_mode=SEQ_CALIB_MODE,
        r2_ratio_2s_to_1s=R2_RATIO_2S_TO_1S,
        tau_qgp_fm=TAU_QGP_FM,
        title=title,
        outfile=OUT_M4,
        figsize=FIGSIZE,
        plot_dpi=PLOT_DPI,
    )
    generated.append(OUT_M4)

    if ENABLE_BJORKEN_COOLING:
        delta_e_base = DELTA_E_STATES["1S-like"]
        gamma0_w2 = calibrate_gamma0(delta_e_base, CALIB_T_REF, CALIB_WIDTH_REF, N_OCTETS)

        print("\n[Bjorken cooling] Piecewise-constant propagation (1⊕8) vs fixed-T baseline")
        t_fm_cool = np.linspace(0.0, BJORKEN_TIME_MAX_FM, BJORKEN_NUM_STEPS)

        ps_fixed_t0 = solve_ps_fixed_t(
            delta_e_base, gamma0_w2, N_OCTETS, [BJORKEN_T0], t_fm_cool
        )[BJORKEN_T0]
        ps_cool = solve_ps_bjorken_piecewise(
            delta_e_base,
            gamma0_w2,
            N_OCTETS,
            t_fm_cool,
            t_profile_fn=lambda tfm: t_bjorken(
                tfm,
                t0=BJORKEN_T0,
                tau0=BJORKEN_TAU0,
                tmin=BJORKEN_TMIN,
            ),
        )

        plot_bjorken_vs_fixed(
            t_fm=t_fm_cool,
            ps_fixed=ps_fixed_t0,
            t_fm_cool=t_fm_cool,
            ps_cool=ps_cool,
            title=rf"Time-dependent $T(\tau)$ vs fixed $T$ (1⊕8), $\Delta E$={delta_e_base:.0f} MeV; $T_0$={BJORKEN_T0:.0f} MeV",
            outfile=OUT_BJORKEN,
            fixed_temp=BJORKEN_T0,
            tau_qgp_fm=TAU_QGP_FM,
            figsize=FIGSIZE,
            plot_dpi=PLOT_DPI,
        )
        generated.append(OUT_BJORKEN)

        for tfm in [0.0, 0.6, 2.0, 5.0, 10.0, 20.0]:
            temp = t_bjorken(tfm, t0=BJORKEN_T0, tau0=BJORKEN_TAU0, tmin=BJORKEN_TMIN)
            print(f"  T({tfm:.1f} fm) = {temp:.1f} MeV")

    print("Generated:")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
