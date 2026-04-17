"""Dispersive-shift formula tests (analytic + numerical)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE
from dispersive_readout.physics.dispersive import (
    dispersive_shift_full,
    dispersive_shift_from_simulation,
    dispersive_shift_two_level,
)
from dispersive_readout.physics.transmon import (
    charge_operator_matrix_elements,
    diagonalize_transmon,
)

_TWO_PI = 2.0 * math.pi


# -- two-level formula ---------------------------------------------------------

def test_two_level_formula_positive_delta():
    chi = dispersive_shift_two_level(g=_TWO_PI * 100e6, Delta=_TWO_PI * 1e9)
    assert chi == pytest.approx((_TWO_PI * 100e6) ** 2 / (_TWO_PI * 1e9))


def test_two_level_formula_negative_delta_gives_negative_chi():
    """Reference device has Δ < 0 (qubit below resonator) → χ < 0."""
    chi = dispersive_shift_two_level(g=_TWO_PI * 120e6, Delta=-_TWO_PI * 2.7e9)
    assert chi < 0


# -- multi-level formula -------------------------------------------------------

def test_dispersive_shift_full_shape():
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    assert chi.shape == (d.truncation.N_transmon,)


def test_dispersive_shift_full_gives_plausible_half_splitting():
    """(χ_1 − χ_0)/2 should be roughly -5 MHz for reference device (spec §1.2)."""
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_j = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    chi_half_hz = (chi_j[1] - chi_j[0]) / 2.0 / _TWO_PI
    assert -10e6 < chi_half_hz < -1e6, (
        f"multi-level χ = {chi_half_hz/1e6:.2f} MHz outside plausible band"
    )


def test_dispersive_shift_full_sign_matches_two_level():
    """Δ < 0 → full formula's χ_1 − χ_0 also < 0."""
    d = REFERENCE_DEVICE
    energies, states = diagonalize_transmon(d.transmon, d.truncation)
    n_mat = charge_operator_matrix_elements(states, d.truncation)
    chi_j = dispersive_shift_full(energies, n_mat, d.coupling.g, d.resonator.omega_r)
    assert (chi_j[1] - chi_j[0]) < 0
