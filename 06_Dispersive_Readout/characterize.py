#!/usr/bin/env python3
"""Stage 06 Module 3 — characterization CLI entry point.

Example
-------
    python 06_Dispersive_Readout/characterize.py --traces data.npz --output params.yaml

See `--help` for full usage.
"""
from __future__ import annotations

import sys

from dispersive_readout.characterization.cli import main


if __name__ == "__main__":
    sys.exit(main())
