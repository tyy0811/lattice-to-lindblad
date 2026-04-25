"""H1 cross-device verification: F(eps) 1D scan at tau=500ns on 2 extreme-T_1
devices. Verifies whether peak #1 vs peak #2 relative ordering is stable
across the 50-device recovery harness.

Same scan resolution as the REFERENCE scan: 50 eps points linear
[5e7, 5e8], tau=500ns, noise_model='analytic' via _F_analytic_at.

Output: prints per-device shape + saves
figures/diagnostic_peak_ordering.yaml
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from dispersive_readout.optimization.modal_pareto import app, F_analytic_at_point
from dispersive_readout.physics.config import REFERENCE_DEVICE


_TAU = 500e-9


def _build_synthetic(entry: dict):
    new_dec = replace(
        REFERENCE_DEVICE.decoherence,
        gamma_1=1.0 / entry["T_1"],
        gamma_phi=max(
            1.0 / entry["T_2_echo"] - 0.5 / entry["T_1"], 0.0,
        ),
        n_th=max(
            float(entry.get("thermal_offset", 0.0)),
            REFERENCE_DEVICE.decoherence.n_th,
        ),
    )
    return replace(REFERENCE_DEVICE, decoherence=new_dec)


def _analyze_shape(label: str, eps_axis: np.ndarray, Fs: np.ndarray) -> dict:
    # Find all local extrema in interior
    local_max_idx = []
    local_min_idx = []
    for i in range(1, len(Fs) - 1):
        if Fs[i] > Fs[i-1] and Fs[i] > Fs[i+1]:
            local_max_idx.append(i)
        elif Fs[i] < Fs[i-1] and Fs[i] < Fs[i+1]:
            local_min_idx.append(i)

    print(f"\n=== {label} ===")
    print(f"Local maxima: {len(local_max_idx)}")
    for i in local_max_idx:
        print(f"  idx={i:2d} eps={eps_axis[i]:.4e} F={Fs[i]:.5f}")
    print(f"Local minima: {len(local_min_idx)}")
    for i in local_min_idx:
        print(f"  idx={i:2d} eps={eps_axis[i]:.4e} F={Fs[i]:.5f}")
    global_argmax_idx = int(np.argmax(Fs))
    print(f"Global argmax: idx={global_argmax_idx} eps={eps_axis[global_argmax_idx]:.4e} "
          f"F={Fs[global_argmax_idx]:.5f}")

    return {
        "label": label,
        "n_local_maxima": len(local_max_idx),
        "local_maxima": [
            {"idx": int(i), "eps": float(eps_axis[i]), "F": float(Fs[i])}
            for i in local_max_idx
        ],
        "n_local_minima": len(local_min_idx),
        "local_minima": [
            {"idx": int(i), "eps": float(eps_axis[i]), "F": float(Fs[i])}
            for i in local_min_idx
        ],
        "global_argmax": {
            "idx": global_argmax_idx,
            "eps": float(eps_axis[global_argmax_idx]),
            "F": float(Fs[global_argmax_idx]),
        },
    }


def main() -> None:
    payload = yaml.safe_load(
        Path("06_Dispersive_Readout/figures/recovery_coverage_report.yaml").read_text()
    )
    devs = payload["devices"]
    by_T1 = sorted(range(len(devs)), key=lambda i: devs[i]["T_1"])
    idx_low, idx_high = by_T1[0], by_T1[-1]
    dev_low = _build_synthetic(devs[idx_low])
    dev_high = _build_synthetic(devs[idx_high])

    eps_axis = np.linspace(5e7, 5e8, 50)
    labels = [
        (f"low_T1 (idx={idx_low}, T_1={devs[idx_low]['T_1']*1e6:.1f}us)", dev_low),
        (f"high_T1 (idx={idx_high}, T_1={devs[idx_high]['T_1']*1e6:.1f}us)", dev_high),
    ]

    # Pack 2 devices x 50 eps values = 100 F evals into one Modal batch.
    all_devices = []
    all_eps = []
    all_tau = []
    for _, d in labels:
        all_devices.extend([d] * len(eps_axis))
        all_eps.extend([float(e) for e in eps_axis])
        all_tau.extend([_TAU] * len(eps_axis))

    print(f"Dispatching {len(all_eps)} F evals on Modal "
          f"(2 devices x 50 eps points at tau={_TAU*1e9:.0f}ns)")
    with app.run():
        F_values = list(F_analytic_at_point.map(all_devices, all_eps, all_tau))

    shapes = []
    for i, (label, _) in enumerate(labels):
        Fs = np.array(F_values[i*50:(i+1)*50])
        shape = _analyze_shape(label, eps_axis, Fs)
        shape["F_values"] = [float(v) for v in Fs]
        shapes.append(shape)

    # Compare peak #1 vs peak #2 ordering across devices
    print("\n=== Peak ordering comparison ===")
    ref_global = {"eps": 1.5102e8, "F": 0.99421}
    print(f"REFERENCE global argmax (from prior scan): eps={ref_global['eps']:.3e}, F={ref_global['F']:.5f}")
    for shape in shapes:
        ga = shape["global_argmax"]
        eps_delta_decade = np.log10(ga["eps"] / ref_global["eps"])
        in_peak2_basin = abs(eps_delta_decade) < 0.15  # within 40% eps
        print(f"  {shape['label']}: global argmax eps={ga['eps']:.3e} F={ga['F']:.5f}  "
              f"(log10 delta = {eps_delta_decade:+.3f} dec; in peak #2 basin? {in_peak2_basin})")

    out = Path("06_Dispersive_Readout/figures/diagnostic_peak_ordering.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(
            {
                "eps_axis": [float(e) for e in eps_axis],
                "tau_ns": _TAU * 1e9,
                "reference_scan_summary": {
                    "source": "fig4_optimization /tmp/F_scan_50pts_tau500_reference.npy",
                    "peak1": {"eps": 7.7551e7, "F": 0.99324},
                    "valley": {"eps": 1.051e8, "F": 0.90853},
                    "peak2_global": {"eps": 1.5102e8, "F": 0.99421},
                },
                "device_scans": shapes,
            },
            f, sort_keys=False,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
