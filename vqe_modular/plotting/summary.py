from __future__ import annotations

from typing import Optional
import numpy as np

def make_summary_figure(
    out_path: str,
    title: str,
    E_ed: float,
    E_ideal: float,
    E_aer_raw: Optional[float],
    E_aer_mit: Optional[float],
    E_qi_raw: Optional[float],
    E_qi_mem: Optional[float],
    se_aer_raw: Optional[float] = None,
    se_aer_mit: Optional[float] = None,
    se_qi_raw: Optional[float] = None,
    se_qi_mem: Optional[float] = None,
) -> None:
    """Single persuasive summary figure:

    Panel A: energy bars (ED vs Ideal vs Aer vs QI), with error bars and ΔE labels.
    Panel B: |E − E_ED| on a log y-scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["ED", "Ideal", "Aer noisy", "Aer+mit", "QI raw", "QI+MEM"]
    vals = [
        E_ed,
        E_ideal,
        E_aer_raw if E_aer_raw is not None else np.nan,
        E_aer_mit if E_aer_mit is not None else np.nan,
        E_qi_raw if E_qi_raw is not None else np.nan,
        E_qi_mem if E_qi_mem is not None else np.nan,
    ]
    errs = [
        0.0,
        0.0,
        float(se_aer_raw or 0.0),
        float(se_aer_mit or 0.0),
        float(se_qi_raw or 0.0),
        float(se_qi_mem or 0.0),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(labels))

    ax1.bar(x, vals, yerr=errs, capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Energy")
    ax1.set_title(title)
    ax1.axhline(E_ed, linestyle="-", linewidth=1)
    ax1.axhline(E_ideal, linestyle="--", linewidth=1)
    ax1.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        ax1.text(i, v, f"ΔE={v - E_ed:+.3f}", ha="center", va="bottom", fontsize=8)

    # absolute error (log)
    names2 = []
    err2 = []
    for name, v in zip(labels[1:], vals[1:]):
        if np.isnan(v):
            continue
        names2.append(name)
        err2.append(abs(v - E_ed))

    ax2.semilogy(np.arange(len(names2)), err2, marker="o")
    ax2.set_xticks(np.arange(len(names2)))
    ax2.set_xticklabels(names2, rotation=20, ha="right")
    ax2.set_ylabel("|E − E_ED|")
    ax2.grid(True, which="both", axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
