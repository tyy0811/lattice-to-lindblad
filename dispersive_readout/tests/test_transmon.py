"""Transmon eigenstructure tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dispersive_readout.physics.config import REFERENCE_DEVICE, TransmonParams, TruncationParams
from dispersive_readout.physics.transmon import (
    charge_basis_hamiltonian,
    charge_operator_matrix_elements,
    diagonalize_transmon,
    transmon_summary,
)

_TWO_PI = 2.0 * math.pi


# -- charge-basis Hamiltonian --------------------------------------------------

def test_charge_basis_hamiltonian_is_hermitian():
    H = charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert np.allclose(H, H.conj().T, atol=1e-20)


def test_charge_basis_hamiltonian_shape_and_dtype():
    trunc = TruncationParams()
    H = charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, trunc)
    assert H.shape == (trunc.N_charge, trunc.N_charge)
    assert H.dtype == np.float64


def test_charge_basis_hamiltonian_rejects_even_N_charge():
    bad = TruncationParams(N_charge=12, N_transmon=5, N_resonator=15)
    with pytest.raises(ValueError, match="odd"):
        charge_basis_hamiltonian(REFERENCE_DEVICE.transmon, bad)


def test_charge_basis_diagonal_is_charging_energy():
    """Diagonal entries must be 4 E_C (n - n_g)^2."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9, n_g=0.0)
    trunc = TruncationParams(N_charge=13, N_transmon=5, N_resonator=15)
    H = charge_basis_hamiltonian(p, trunc)
    n_values = np.arange(-6, 7)
    expected_diag = 4.0 * p.E_C * n_values ** 2
    assert np.allclose(np.diag(H), expected_diag)


def test_charge_basis_offdiagonal_is_josephson():
    """Adjacent off-diagonals are -E_J/2."""
    p = TransmonParams(E_C=_TWO_PI * 210e6, E_J=_TWO_PI * 15.5e9)
    trunc = TruncationParams()
    H = charge_basis_hamiltonian(p, trunc)
    for i in range(trunc.N_charge - 1):
        assert H[i, i + 1] == pytest.approx(-0.5 * p.E_J)
        assert H[i + 1, i] == pytest.approx(-0.5 * p.E_J)
    # Non-adjacent off-diagonals must be zero
    for i in range(trunc.N_charge):
        for j in range(trunc.N_charge):
            if abs(i - j) > 1:
                assert H[i, j] == 0.0
