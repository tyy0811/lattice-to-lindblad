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


# -- diagonalization -----------------------------------------------------------

def test_diagonalize_returns_correct_shapes():
    trunc = TruncationParams()
    energies, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, trunc)
    assert energies.shape == (trunc.N_transmon,)
    assert states.shape == (trunc.N_charge, trunc.N_transmon)


def test_diagonalize_energies_sorted_ascending():
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert np.all(np.diff(energies) > 0)


def test_diagonalize_ground_energy_shifted_to_zero():
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert energies[0] == pytest.approx(0.0, abs=1e-20)


def test_diagonalize_eigenstates_orthonormal():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    gram = states.conj().T @ states
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


def test_diagonalize_omega01_in_plausible_range():
    """For the reference device ω_01/2π should be ~4.4–4.8 GHz (Marxer device band)."""
    energies, _ = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    omega_01_hz = energies[1] / _TWO_PI
    assert 4.3e9 < omega_01_hz < 4.9e9, f"omega_01/2π = {omega_01_hz/1e9:.3f} GHz outside Marxer band"


# -- matrix elements + summary -------------------------------------------------

def test_charge_matrix_elements_shape():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert n_mat.shape == (REFERENCE_DEVICE.truncation.N_transmon,
                          REFERENCE_DEVICE.truncation.N_transmon)


def test_charge_matrix_is_hermitian():
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert np.allclose(n_mat, n_mat.conj().T, atol=1e-10)


def test_charge_matrix_element_01_dominant():
    """|<0|n̂|1>| should be larger than |<0|n̂|2>| (selection rule in deep transmon regime)."""
    _, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    n_mat = charge_operator_matrix_elements(states, REFERENCE_DEVICE.truncation)
    assert abs(n_mat[0, 1]) > 10.0 * abs(n_mat[0, 2])


def test_transmon_summary_keys():
    summary = transmon_summary(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    required = {
        "omega_01", "omega_12", "alpha", "E_J_over_E_C",
        "charge_dispersion_01", "n_matrix_01", "n_matrix_12",
    }
    assert required.issubset(summary.keys()), f"missing keys: {required - summary.keys()}"


def test_transmon_summary_values_plausible():
    s = transmon_summary(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    # anharmonicity negative (transmon); ~-200 MHz
    alpha_hz = s["alpha"] / _TWO_PI
    assert -260e6 < alpha_hz < -160e6, f"alpha/2π = {alpha_hz/1e6:.1f} MHz outside plausible band"
    # E_J/E_C ≈ 74
    assert 70 < s["E_J_over_E_C"] < 80


def test_transmon_summary_requires_N_transmon_ge_3():
    """Two-level truncation cannot supply omega_12 / |<1|n̂|2>|."""
    tr = TruncationParams(N_charge=5, N_transmon=2, N_resonator=15)
    with pytest.raises(ValueError, match="N_transmon"):
        transmon_summary(REFERENCE_DEVICE.transmon, tr)


# -- diagonalize_transmon cache -------------------------------------------------

def test_diagonalize_transmon_returns_readonly_cached_arrays():
    """Cached eigenpairs must not be mutable — mutation would corrupt
    subsequent cache hits."""
    energies, states = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert not energies.flags.writeable
    assert not states.flags.writeable

    # Repeated calls with the same frozen-dataclass inputs return the same array identities.
    e2, s2 = diagonalize_transmon(REFERENCE_DEVICE.transmon, REFERENCE_DEVICE.truncation)
    assert e2 is energies, "cache miss on identical inputs"
    assert s2 is states


def test_diagonalize_transmon_cache_distinguishes_params():
    """Different n_g must miss the cache and produce a different eigendecomposition."""
    from dataclasses import replace
    p0 = REFERENCE_DEVICE.transmon
    p_half = replace(p0, n_g=0.5)
    e0, _ = diagonalize_transmon(p0, REFERENCE_DEVICE.truncation)
    eh, _ = diagonalize_transmon(p_half, REFERENCE_DEVICE.truncation)
    # Ground-shifted energies differ at n_g=0 vs 0.5 (charge dispersion)
    assert not np.allclose(e0, eh, atol=0.0, rtol=0.0)
