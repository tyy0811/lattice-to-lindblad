"""Tests for l2l/entanglement.py utilities."""
from __future__ import annotations

import numpy as np
import pytest


# --- Pure unit tests (fast, no DMRG) ---

def test_entanglement_levels_basic():
    """entanglement_levels should compute xi = -log(lambda^2)."""
    from l2l.entanglement import entanglement_levels

    lambdas = np.array([0.5, 0.5, 0.5, 0.5])
    xi = entanglement_levels(lambdas)

    expected = -np.log(0.25)
    assert np.allclose(xi, expected)


def test_entanglement_levels_handles_small_values():
    """entanglement_levels should clip tiny values to avoid inf."""
    from l2l.entanglement import entanglement_levels

    lambdas = np.array([0.99, 0.1, 1e-20, 0.0])
    xi = entanglement_levels(lambdas, cutoff=1e-15)

    assert np.all(np.isfinite(xi))


def test_cumulative_weight_basic():
    """cumulative_weight should compute running sum of lambda^2."""
    from l2l.entanglement import cumulative_weight

    lambdas = np.array([0.6, 0.5, 0.4, 0.3, 0.2])
    cum = cumulative_weight(lambdas)

    expected = np.cumsum(lambdas**2)
    assert np.allclose(cum, expected)


def test_compute_entropy_profile_validates_finite_mps():
    """compute_entropy_profile should raise for non-finite MPS."""
    from l2l.entanglement import compute_entropy_profile

    class FakeMPS:
        L = 4
        bc = "infinite"

    with pytest.raises(ValueError, match="finite"):
        compute_entropy_profile(FakeMPS())


def test_compute_entropy_profile_validates_min_length():
    """compute_entropy_profile should raise for L < 2."""
    from l2l.entanglement import compute_entropy_profile

    class FakeMPS:
        L = 1
        bc = "finite"

    with pytest.raises(ValueError, match="at least 2"):
        compute_entropy_profile(FakeMPS())


# --- Integration tests (slow, require DMRG) ---

@pytest.mark.slow
def test_compute_entropy_profile_returns_cuts_and_entropies():
    """compute_entropy_profile should return (cuts, entropies) arrays."""
    from l2l.entanglement import compute_entropy_profile
    from l2l.tfim_adapter import tfim_ground_state

    N = 6
    E0, psi = tfim_ground_state(N=N, J=1.0, g=1.0, chi=16)

    cuts, entropies = compute_entropy_profile(psi)

    assert isinstance(cuts, np.ndarray)
    assert isinstance(entropies, np.ndarray)
    assert len(cuts) == N - 1
    assert len(entropies) == N - 1
    assert np.all(cuts == np.arange(N - 1))
    assert np.all(entropies >= 0)


@pytest.mark.slow
def test_extract_schmidt_values_returns_array():
    """extract_schmidt_values should return normalized Schmidt values."""
    from l2l.entanglement import extract_schmidt_values
    from l2l.tfim_adapter import tfim_ground_state

    N = 6
    E0, psi = tfim_ground_state(N=N, J=1.0, g=1.0, chi=16)
    cut = N // 2 - 1

    lambdas = extract_schmidt_values(psi, cut)

    assert isinstance(lambdas, np.ndarray)
    assert len(lambdas) > 0
    assert np.abs(np.sum(lambdas**2) - 1.0) < 1e-10


@pytest.mark.slow
def test_extract_schmidt_values_validates_cut_range():
    """extract_schmidt_values should raise for out-of-range cut."""
    from l2l.entanglement import extract_schmidt_values
    from l2l.tfim_adapter import tfim_ground_state

    N = 6
    E0, psi = tfim_ground_state(N=N, J=1.0, g=1.0, chi=16)

    with pytest.raises(ValueError, match="cut"):
        extract_schmidt_values(psi, -1)

    with pytest.raises(ValueError, match="cut"):
        extract_schmidt_values(psi, N - 1)


# --- Tests for extract_schmidt_values_by_sector ---

@pytest.mark.slow
def test_extract_schmidt_values_by_sector_returns_dict():
    """extract_schmidt_values_by_sector returns dict with string keys."""
    pytest.importorskip("tenpy")

    from l2l.entanglement import extract_schmidt_values_by_sector
    from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter

    adapter = SchwingerMassGapAdapter(m_over_g=0.05, E0=0.0)
    result = adapter.dmrg_solve_point(6, {"x": 4.0}, chi=16, return_mps=True)
    psi = result["psi0"]

    sector_to_lambdas = extract_schmidt_values_by_sector(psi, cut=2)

    assert isinstance(sector_to_lambdas, dict)
    assert len(sector_to_lambdas) > 0

    for key in sector_to_lambdas:
        assert isinstance(key, str)
        assert key.startswith("q")

    for lambdas in sector_to_lambdas.values():
        assert isinstance(lambdas, np.ndarray)
        assert len(lambdas) > 0
        assert np.all(lambdas[:-1] >= lambdas[1:])


@pytest.mark.slow
def test_extract_schmidt_values_by_sector_weights_sum_to_one():
    """Sector weights should sum to approximately 1."""
    pytest.importorskip("tenpy")

    from l2l.entanglement import extract_schmidt_values_by_sector
    from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter

    adapter = SchwingerMassGapAdapter(m_over_g=0.05, E0=0.0)
    result = adapter.dmrg_solve_point(6, {"x": 4.0}, chi=16, return_mps=True)
    psi = result["psi0"]

    sector_to_lambdas = extract_schmidt_values_by_sector(psi, cut=2)

    total_weight = sum(np.sum(lambdas**2) for lambdas in sector_to_lambdas.values())

    assert np.isclose(total_weight, 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_extract_schmidt_values_by_sector_invalid_cut():
    """Invalid cut should raise ValueError."""
    pytest.importorskip("tenpy")

    from l2l.entanglement import extract_schmidt_values_by_sector
    from l2l.schwinger_massgap_adapter import SchwingerMassGapAdapter

    adapter = SchwingerMassGapAdapter(m_over_g=0.05, E0=0.0)
    result = adapter.dmrg_solve_point(6, {"x": 4.0}, chi=16, return_mps=True)
    psi = result["psi0"]

    with pytest.raises(ValueError, match="cut must be in"):
        extract_schmidt_values_by_sector(psi, cut=10)

    with pytest.raises(ValueError, match="cut must be in"):
        extract_schmidt_values_by_sector(psi, cut=-1)


# --- Tests for compute_sector_weights ---

def test_compute_sector_weights_returns_dict():
    """compute_sector_weights returns dict with float values."""
    from l2l.entanglement import compute_sector_weights

    sector_to_lambdas = {
        "q0": np.array([0.7, 0.3]),
        "q1": np.array([0.5, 0.1]),
    }

    weights = compute_sector_weights(sector_to_lambdas)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == {"q0", "q1"}
    # q0: 0.7^2 + 0.3^2 = 0.49 + 0.09 = 0.58
    assert np.isclose(weights["q0"], 0.58)
    # q1: 0.5^2 + 0.1^2 = 0.25 + 0.01 = 0.26
    assert np.isclose(weights["q1"], 0.26)


def test_compute_sector_weights_empty_sector():
    """Empty sector should have weight 0."""
    from l2l.entanglement import compute_sector_weights

    sector_to_lambdas = {
        "q0": np.array([0.5]),
        "q1": np.array([]),
    }

    weights = compute_sector_weights(sector_to_lambdas)

    assert weights["q0"] == 0.25
    assert weights["q1"] == 0.0
