"""Stage 06 Module 3 — characterization CLI.

Entry: `python 06_Dispersive_Readout/characterize.py ...`

Three modes:
  --traces BUNDLE.npz --output PARAMS.yaml [--bootstrap-samples 200]
      Fit a trace bundle; write a Module-1-compatible YAML parameter pack.
  --recovery --n-devices 50 --output REPORT.yaml [--seed 42]
      Run the recovery harness; write a coverage report.
  --generate-synthetic --output BUNDLE.npz [--seed 42]
      Generate a reference synthetic trace bundle from REFERENCE_DEVICE.
"""
from __future__ import annotations

import argparse
import math
import sys

import yaml

from .fitting import fit_all
from .noise import NoiseModelParams
from .protocols import (
    TraceData, generate_rabi_trace, generate_ramsey_trace,
    generate_t1_trace, generate_t2_echo_trace,
    load_trace_bundle, save_trace_bundle,
)
from .recovery import run_recovery_harness, save_coverage_report


_DESCRIPTION = """Extract device parameters from characterization traces.

Examples
--------
Fit a trace bundle:
    python 06_Dispersive_Readout/characterize.py --traces data.npz --output params.yaml

Run the recovery harness:
    python 06_Dispersive_Readout/characterize.py --recovery --n-devices 50 \\
        --output recovery_report.yaml --seed 42

Generate a reference synthetic bundle:
    python 06_Dispersive_Readout/characterize.py --generate-synthetic \\
        --output example_traces.npz --seed 42
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="06_Dispersive_Readout/characterize.py",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traces", type=str, default=None, help="Path to a .npz trace bundle to fit.")
    parser.add_argument("--output", type=str, required=True, help="Output path (.yaml for params or report; .npz for synthetic).")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Parametric bootstrap samples per fitted parameter.")
    parser.add_argument("--recovery", action="store_true", help="Run the 50-device recovery harness.")
    parser.add_argument("--n-devices", type=int, default=50, help="Devices for the recovery harness.")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a synthetic trace bundle from REFERENCE_DEVICE.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for determinism (default 42; matches the committed artifact).")
    return parser


def _reject_conflicts(args: argparse.Namespace) -> str | None:
    """Return an error message if the flag combination is invalid, else None."""
    modes = []
    if args.traces is not None:
        modes.append("--traces")
    if args.recovery:
        modes.append("--recovery")
    if args.generate_synthetic:
        modes.append("--generate-synthetic")
    if len(modes) == 0:
        return "Pick one of: --traces, --recovery, --generate-synthetic."
    if len(modes) > 1:
        return f"Flags {modes} are mutually exclusive; pick one."
    return None


def _mode_generate_synthetic(args: argparse.Namespace) -> int:
    noise = NoiseModelParams()
    eps_pi = 2 * math.pi * 50e6
    omega_q = 2 * math.pi * 4.5e9
    T_1 = 30e-6
    T_2 = 40e-6
    traces: list[TraceData] = [
        generate_rabi_trace(eps_pi, omega_q, noise, seed=args.seed),
        generate_ramsey_trace(omega_q, T_2_star=T_2, noise=noise, seed=args.seed + 1),
        generate_t1_trace(T_1, noise, seed=args.seed + 2),
        generate_t2_echo_trace(T_2, noise, seed=args.seed + 3),
    ]
    save_trace_bundle(traces, args.output)
    print(f"Wrote 4-protocol synthetic bundle: {args.output}")
    return 0


def _mode_traces(args: argparse.Namespace) -> int:
    traces = load_trace_bundle(args.traces)
    pack = fit_all(
        traces,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        trace_file=args.traces,
    )
    with open(args.output, "w") as f:
        yaml.safe_dump(pack.model_dump(), f, sort_keys=False)
    print(f"Fit {len(traces)} trace(s). Wrote parameter pack: {args.output}")
    return 0


def _mode_recovery(args: argparse.Namespace) -> int:
    noise = NoiseModelParams()
    reports, devices = run_recovery_harness(
        n_devices=args.n_devices, noise=noise, seed=args.seed,
    )
    save_coverage_report(reports, devices, args.output, seed=args.seed)
    print(f"Recovery harness wrote: {args.output}")
    rej = sum(r.n_rejected for r in reports.values())
    if rej:
        print(f"Rejections (spec §1.1 reject_flag set): total {rej} across all parameters")
        for name, r in reports.items():
            if r.n_rejected:
                print(f"  {name}: {r.n_rejected}/{r.n_devices} rejected — "
                      f"accepted 2σ cov {r.coverage_2_sigma_on_accepted:.1%}")
    else:
        print("Rejections: 0 (no devices triggered the 1.5-oscillation flag)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    err = _reject_conflicts(args)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.generate_synthetic:
        return _mode_generate_synthetic(args)
    if args.recovery:
        return _mode_recovery(args)
    if args.traces is not None:
        return _mode_traces(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
