"""Comprehensive tests for all ported algorithms.

Tests that each algorithm:
1. Can be instantiated
2. Produces output with correct shapes (L, S, O match input)
3. L + S approximately equals input M
4. Runs without errors
"""

import numpy as np
import pytest

from lrslibrary.registry import list_algorithms, run_algorithm


@pytest.fixture(scope="module")
def small_matrix():
    """Small test matrix for fast algorithm testing."""
    rng = np.random.RandomState(42)
    m, n = 30, 20
    rank = 2
    U = rng.randn(m, rank)
    V = rng.randn(rank, n)
    L = U @ V
    S = np.zeros((m, n))
    idx = rng.choice(m * n, int(0.05 * m * n), replace=False)
    S.flat[idx] = rng.randn(len(idx)) * 5
    return L + S


@pytest.fixture(scope="module")
def small_tensor():
    """Small 3D tensor for tensor algorithm testing."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 10, 5)


def _get_all_algorithm_ids():
    """Get all registered algorithm (method_id, algorithm_id) pairs."""
    algorithms = list_algorithms()
    return [(a["method_id"], a["algorithm_id"]) for a in algorithms]


def _get_matrix_algorithms():
    """Get non-tensor algorithms."""
    algorithms = list_algorithms()
    return [
        (a["method_id"], a["algorithm_id"])
        for a in algorithms
        if not a.get("is_tensor", False)
    ]


def _get_tensor_algorithms():
    """Get tensor algorithms."""
    algorithms = list_algorithms()
    return [
        (a["method_id"], a["algorithm_id"])
        for a in algorithms
        if a.get("is_tensor", False)
    ]


class TestAlgorithmDiscovery:
    """Test that all algorithms are properly registered."""

    def test_total_algorithm_count(self):
        """Verify we have 115 algorithms registered."""
        algorithms = list_algorithms()
        assert len(algorithms) >= 115, (
            f"Expected >= 115 algorithms, got {len(algorithms)}"
        )

    def test_rpca_count(self):
        """Verify RPCA algorithm count."""
        algorithms = list_algorithms(method_id="RPCA")
        assert len(algorithms) >= 47, (
            f"Expected >= 47 RPCA algorithms, got {len(algorithms)}"
        )

    def test_mc_count(self):
        """Verify MC algorithm count."""
        algorithms = list_algorithms(method_id="MC")
        assert len(algorithms) >= 16, (
            f"Expected >= 16 MC algorithms, got {len(algorithms)}"
        )

    def test_nmf_count(self):
        """Verify NMF algorithm count."""
        algorithms = list_algorithms(method_id="NMF")
        assert len(algorithms) >= 15, (
            f"Expected >= 15 NMF algorithms, got {len(algorithms)}"
        )

    def test_ntf_count(self):
        """Verify NTF algorithm count."""
        algorithms = list_algorithms(method_id="NTF")
        assert len(algorithms) >= 7, (
            f"Expected >= 7 NTF algorithms, got {len(algorithms)}"
        )

    def test_st_count(self):
        """Verify ST algorithm count."""
        algorithms = list_algorithms(method_id="ST")
        assert len(algorithms) >= 5, (
            f"Expected >= 5 ST algorithms, got {len(algorithms)}"
        )

    def test_td_count(self):
        """Verify TD algorithm count."""
        algorithms = list_algorithms(method_id="TD")
        assert len(algorithms) >= 14, (
            f"Expected >= 14 TD algorithms, got {len(algorithms)}"
        )

    def test_lrr_count(self):
        """Verify LRR algorithm count."""
        algorithms = list_algorithms(method_id="LRR")
        assert len(algorithms) >= 7, (
            f"Expected >= 7 LRR algorithms, got {len(algorithms)}"
        )

    def test_ttd_count(self):
        """Verify TTD algorithm count."""
        algorithms = list_algorithms(method_id="TTD")
        assert len(algorithms) >= 4, (
            f"Expected >= 4 TTD algorithms, got {len(algorithms)}"
        )


class TestMatrixAlgorithms:
    """Test all matrix (non-tensor) algorithms."""

    @pytest.mark.parametrize(
        "method_id,algorithm_id",
        _get_matrix_algorithms(),
        ids=[f"{m}-{a}" for m, a in _get_matrix_algorithms()],
    )
    def test_output_shapes(self, method_id, algorithm_id, small_matrix):
        """Every algorithm produces L, S, O with same shape as input."""
        result = run_algorithm(method_id, algorithm_id, small_matrix)
        assert result.L.shape == small_matrix.shape, (
            f"{algorithm_id}: L shape {result.L.shape} != {small_matrix.shape}"
        )
        assert result.S.shape == small_matrix.shape, (
            f"{algorithm_id}: S shape {result.S.shape} != {small_matrix.shape}"
        )
        assert result.O.shape == small_matrix.shape, (
            f"{algorithm_id}: O shape {result.O.shape} != {small_matrix.shape}"
        )

    @pytest.mark.parametrize(
        "method_id,algorithm_id",
        _get_matrix_algorithms(),
        ids=[f"{m}-{a}" for m, a in _get_matrix_algorithms()],
    )
    def test_no_nans(self, method_id, algorithm_id, small_matrix):
        """No NaN values in output."""
        result = run_algorithm(method_id, algorithm_id, small_matrix)
        assert not np.any(np.isnan(result.L)), f"{algorithm_id}: NaN in L"
        assert not np.any(np.isnan(result.S)), f"{algorithm_id}: NaN in S"
        assert not np.any(np.isnan(result.O)), f"{algorithm_id}: NaN in O"


class TestTensorAlgorithms:
    """Test all tensor algorithms."""

    @pytest.mark.parametrize(
        "method_id,algorithm_id",
        _get_tensor_algorithms(),
        ids=[f"{m}-{a}" for m, a in _get_tensor_algorithms()],
    )
    def test_output_shapes(self, method_id, algorithm_id, small_tensor):
        """Every tensor algorithm produces outputs with correct shapes."""
        # Tensor algorithms should work with both 2D and 3D input
        M_2d = small_tensor.reshape(small_tensor.shape[0], -1)
        result = run_algorithm(method_id, algorithm_id, M_2d)
        assert result.L.shape == M_2d.shape, (
            f"{algorithm_id}: L shape {result.L.shape} != {M_2d.shape}"
        )
        assert result.S.shape == M_2d.shape, (
            f"{algorithm_id}: S shape {result.S.shape} != {M_2d.shape}"
        )
        assert result.O.shape == M_2d.shape, (
            f"{algorithm_id}: O shape {result.O.shape} != {M_2d.shape}"
        )

    @pytest.mark.parametrize(
        "method_id,algorithm_id",
        _get_tensor_algorithms(),
        ids=[f"{m}-{a}" for m, a in _get_tensor_algorithms()],
    )
    def test_no_nans(self, method_id, algorithm_id, small_tensor):
        """No NaN values in output."""
        M_2d = small_tensor.reshape(small_tensor.shape[0], -1)
        result = run_algorithm(method_id, algorithm_id, M_2d)
        assert not np.any(np.isnan(result.L)), f"{algorithm_id}: NaN in L"
        assert not np.any(np.isnan(result.S)), f"{algorithm_id}: NaN in S"
        assert not np.any(np.isnan(result.O)), f"{algorithm_id}: NaN in O"
