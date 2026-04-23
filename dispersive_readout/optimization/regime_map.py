"""Closed-form analytic regime-map surface and boundary functions.

See MODULE_4_SPEC.md §3.2. The map is an analytic evaluation of the
dispersive-readout SNR formula (Bengtsson 2024 PRL §II, cross-checked
against Blais RMP 2021 §V.B §V.B), not a Lindblad grid — the 100× chi/kappa
range would otherwise extrapolate the 2nd-order SW dispersive PT well
outside its validity envelope (Q3 lock).
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm


ArrayLike = Union[float, np.ndarray]


def f_analytic_dispersive(
    chi_over_kappa: ArrayLike,
    gamma_1_tau: ArrayLike,
    n_phot: float,
) -> ArrayLike:
    """Closed-form F_assign per Bengtsson 2024 PRL §II.

    Parameters
    ----------
    chi_over_kappa : float or ndarray
        χ/κ at each evaluation point. Non-broadcastable axes must match.
    gamma_1_tau : float or ndarray
        γ_1 · τ_readout (dimensionless decoherence budget).
    n_phot : float
        Steady-state resonator photon number (scalar; held fixed across
        the grid and quoted on Figure 4 Panel (b)'s subtitle).

    Returns
    -------
    F : float or ndarray
        Assignment fidelity Φ(SNR_eff / 2).

    Notes
    -----
    Decoherence envelope is linearized: (1 − γ_1·τ/2)^(1/2). Within 1%
    of exp(−γ_1·τ/4) over the spec's y-axis range [1e-4, 1e-1]; a
    unit test asserts this.
    """
    chi_k = np.asarray(chi_over_kappa, dtype=float)
    g_t = np.asarray(gamma_1_tau, dtype=float)
    snr_steady = 4.0 * chi_k * np.sqrt(n_phot) / (1.0 + (2.0 * chi_k) ** 2)
    envelope = np.sqrt(np.clip(1.0 - g_t / 2.0, 0.0, 1.0))
    snr_eff = snr_steady * envelope
    return norm.cdf(snr_eff / 2.0)
