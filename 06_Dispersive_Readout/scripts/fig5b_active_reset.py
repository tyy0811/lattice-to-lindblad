#!/usr/bin/env python
"""fig5b — Joint transition-readout active reset on closed-loop demo idx=18.

Two-panel figure with three traces in panel (a) and a regime-aware
decomposition in panel (b). Single-driver script, no CLI args. Writes
fig5b_active_reset.png and fig5b_active_reset_data.yaml to
06_Dispersive_Readout/figures/.

Operating point: idx=18 from Module 4's closed-loop harness (T₁=5.35 µs).
ε_X handoff from 5a's full-Lindblad headline at T_gate=20 ns.
Sweep range: τ_meas/T₁ ∈ [0.1, 2.0], 16 points.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `dispersive_readout` resolves regardless
# of the working directory used to launch this script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dispersive_readout.control.reset_protocol import (
    closed_loop_demo_drive_params,
    device_idx18,
    extract_joint_matrix,
    load_eps_x_5a,
    passive_reset_residual,
    reset_residual_single_cycle,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
OUTPUT_PNG = FIGURES_DIR / "fig5b_active_reset.png"
OUTPUT_YAML = FIGURES_DIR / "fig5b_active_reset_data.yaml"

T_GATE = 20e-9  # 5a's headline gate duration
N_TRAJECTORIES = 1000
N_TAU_POINTS = 16


def identify_panel_b_tau(
    tau_meas_grid: np.ndarray,
    p_active_realistic: np.ndarray,
    p_passive: np.ndarray,
) -> tuple[float, str, int]:
    """Regime-aware τ_b selection.

    Returns (tau_b, regime_label, idx) with regime ∈
      {'active_winning', 'crossover_only', 'passive_dominant'}.
    """
    active_winning_mask = p_active_realistic < p_passive
    if active_winning_mask.any():
        candidates = np.where(active_winning_mask)[0]
        idx = int(candidates[np.argmin(p_active_realistic[candidates])])
        return float(tau_meas_grid[idx]), 'active_winning', idx

    margin = p_active_realistic - p_passive
    if margin.min() < 0.05 * float(p_passive.min()):
        idx = int(np.argmin(margin))
        return float(tau_meas_grid[idx]), 'crossover_only', idx

    idx = int(np.argmin(p_active_realistic))
    return float(tau_meas_grid[idx]), 'passive_dominant', idx


def main() -> None:
    # 1. Setup device + ε_X handoff
    device = device_idx18()
    T1 = 1.0 / device.decoherence.gamma_1
    eps_X_5a, eps_X_provenance = load_eps_x_5a(t_gate=T_GATE)

    # 2. Sweep grid + per-τ RNG substreams
    tau_meas_grid = T1 * np.linspace(0.1, 2.0, N_TAU_POINTS)
    master_rng = np.random.default_rng(seed=42)
    per_tau_rngs = master_rng.spawn(len(tau_meas_grid))

    # 3. Extract joint matrices
    joints = []
    for tau, rng in zip(tau_meas_grid, per_tau_rngs):
        drive = closed_loop_demo_drive_params(duration=tau)
        joints.append(extract_joint_matrix(
            device, drive, n_trajectories=N_TRAJECTORIES, rng=rng,
        ))

    # 4. Three traces for panel (a)
    p_active_ideal = np.array([
        reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=0.0)
        for J in joints
    ])
    p_active_realistic = np.array([
        reset_residual_single_cycle(p_e=1.0, joint=J, gate_error=eps_X_5a)
        for J in joints
    ])
    p_passive = np.array([
        passive_reset_residual(T1, tau + T_GATE)
        for tau in tau_meas_grid
    ])

    # SE band on realistic (formula error propagation: linear in joint entries)
    p_active_realistic_se = np.array([
        np.sqrt(
            J.binomial_se[(1, 1, 0)] ** 2
            + (J.binomial_se[(1, 1, 1)] * eps_X_5a) ** 2
            + (J.binomial_se[(1, 0, 1)] * (1.0 - eps_X_5a)) ** 2
        )
        for J in joints
    ])

    # 5. Panel-(b) regime-aware τ selection
    tau_b, regime_label, idx_b = identify_panel_b_tau(
        tau_meas_grid, p_active_realistic, p_passive,
    )
    J_b = joints[idx_b]

    contributions = {
        'missed_excited': J_b.probabilities[(1, 1, 0)],
        'false_positive_decayed': J_b.probabilities[(1, 0, 1)] * (1.0 - eps_X_5a),
        'gate_failure_excited': J_b.probabilities[(1, 1, 1)] * eps_X_5a,
    }
    contribution_se = {
        'missed_excited': J_b.binomial_se[(1, 1, 0)],
        'false_positive_decayed': J_b.binomial_se[(1, 0, 1)] * (1.0 - eps_X_5a),
        'gate_failure_excited': J_b.binomial_se[(1, 1, 1)] * eps_X_5a,
    }
    total_at_b = sum(contributions.values())
    relative_pct = {k: 100.0 * v / total_at_b for k, v in contributions.items()}

    # V7 reporting curve
    p_decay_missed = np.array([J.probabilities[(1, 0, 0)] for J in joints])
    p_decay_missed_se = np.array([J.binomial_se[(1, 0, 0)] for J in joints])

    # 6. Render
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel (a): three traces
    tau_us = tau_meas_grid * 1e6
    ax_a.plot(tau_us, p_passive, 'k--', label='passive (T₁ decay only)')
    ax_a.plot(tau_us, p_active_ideal, 'C0-',
              label=r'active reset, ideal gate ($\varepsilon_X = 0$)')
    ax_a.plot(tau_us, p_active_realistic, 'C1-',
              label=fr'active reset, Module-5a gate ($\varepsilon_X = {eps_X_5a:.2e}$)')
    ax_a.fill_between(
        tau_us,
        p_active_realistic - p_active_realistic_se,
        p_active_realistic + p_active_realistic_se,
        color='C1', alpha=0.3,
    )
    ax_a.axvline(tau_b * 1e6, color='gray', linestyle=':', alpha=0.5,
                 label='operating point shown in (b)')
    ax_a.set_xlabel(r'measurement window $\tau_{\rm meas}$ ($\mu$s)')
    ax_a.set_ylabel(r"excited-state residual $p_e'$ after one cycle")
    ax_a.set_yscale('log')
    ax_a.set_title(
        r'Short-window active reset beats passive $T_1$ decay' '\n'
        r'($T_1 = 5.35\,\mu s$, $\varepsilon_{\rm drive} = 140$ MHz)',
        fontsize=11,
    )
    ax_a.legend(fontsize=8, loc='lower right')
    ax_a.grid(alpha=0.3)

    # Panel (b): 100%-stacked horizontal bar (largest-to-smallest)
    ordered = sorted(contributions.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in ordered]
    fractions_pct = [100.0 * v / total_at_b for v in (contributions[k] for k in labels)]
    colors = {
        'missed_excited': '#7a7a7a',
        'false_positive_decayed': '#c0392b',
        'gate_failure_excited': '#2980b9',
    }
    pretty = {
        'missed_excited': 'missed excited state',
        'false_positive_decayed': 'false-positive after decay',
        'gate_failure_excited': 'gate failure (Module 5a)',
    }
    cumulative = 0.0
    for label, pct in zip(labels, fractions_pct):
        ax_b.barh(
            0, pct, left=cumulative,
            color=colors[label],
            edgecolor='white', linewidth=1.2,
            label=f'{pretty[label]} — {pct:.1f}%',
        )
        # Annotate inside the segment if it's wide enough to be readable
        if pct > 4.0:
            ax_b.text(
                cumulative + pct / 2.0, 0,
                f'{pct:.1f}%',
                ha='center', va='center',
                color='white', fontsize=11, fontweight='bold',
            )
        cumulative += pct

    ax_b.set_xlim(0, 100)
    ax_b.set_yticks([])
    ax_b.set_xlabel(
        f"% of total residual  "
        fr"($p_e' = {total_at_b:.4f}$ at $\tau_{{\rm meas}} = {tau_b * 1e6:.2f}\,\mu s$)"
    )
    ax_b.set_title(
        'Residual error budget at selected active-reset point',
        fontsize=11,
    )
    ax_b.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    ax_b.grid(alpha=0.3, axis='x')
    for spine in ('top', 'right', 'left'):
        ax_b.spines[spine].set_visible(False)

    # Bottom caption explaining the dominant-residual mechanism in plain language
    fig.text(
        0.5, -0.02,
        'Dominant residual: qubit decays during measurement, but thresholded IQ '
        'still triggers an unnecessary X-flip.',
        ha='center', va='top',
        fontsize=10, style='italic', color='#444',
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Wrote {OUTPUT_PNG}")

    # 7. YAML serialization
    yaml_data = {
        'tau_meas_grid_us': [float(x * 1e6) for x in tau_meas_grid],
        'p_passive': [float(x) for x in p_passive],
        'p_active_ideal': [float(x) for x in p_active_ideal],
        'p_active_realistic': [float(x) for x in p_active_realistic],
        'p_active_realistic_se': [float(x) for x in p_active_realistic_se],
        'p_decay_missed': [float(x) for x in p_decay_missed],
        'p_decay_missed_se': [float(x) for x in p_decay_missed_se],
        'tau_b_us': float(tau_b * 1e6),
        'regime_label': regime_label,
        'panel_b_contributions': {k: float(v) for k, v in contributions.items()},
        'panel_b_contribution_se': {k: float(v) for k, v in contribution_se.items()},
        'panel_b_relative_pct': {k: float(v) for k, v in relative_pct.items()},
        'panel_b_joint_matrix': {
            f"{s_i}{s_f}{m}": float(J_b.probabilities[(s_i, s_f, m)])
            for s_i in (0, 1) for s_f in (0, 1) for m in (0, 1)
        },
        'epsilon_x_5a': float(eps_X_5a),
        'epsilon_x_5a_provenance': eps_X_provenance,
        'operating_point': J_b.operating_point,
        'device_yaml_path': str(
            SCRIPT_DIR.parent / 'figures' / 'closed_loop_demo_device.yaml'
        ),
        'n_trajectories': N_TRAJECTORIES,
        'T1_us': float(T1 * 1e6),
        'T_gate_ns': float(T_GATE * 1e9),
    }
    OUTPUT_YAML.write_text(yaml.safe_dump(yaml_data, default_flow_style=False))
    print(f"Wrote {OUTPUT_YAML}")


if __name__ == '__main__':
    main()
