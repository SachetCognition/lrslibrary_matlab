"""Tests for the algorithm registry system."""

import numpy as np
import pytest

from lrslibrary.algorithms.base import Decomposer, DecompositionResult
from lrslibrary.registry import (
    _registry,
    list_algorithms,
    list_methods,
    register,
    run_algorithm,
)


def test_register_and_lookup():
    """Register a dummy algorithm, verify it can be found."""

    @register("TEST", "DUMMY", "Dummy Test Algorithm", speed_class=2)
    class DummyDecomposer(Decomposer):
        def decompose(self, data, params=None):
            return DecompositionResult(
                L=np.zeros_like(data),
                S=np.zeros_like(data),
                O=np.zeros_like(data),
            )

    assert ("TEST", "DUMMY") in _registry
    assert _registry[("TEST", "DUMMY")] is DummyDecomposer

    # Clean up
    del _registry[("TEST", "DUMMY")]


def test_list_algorithms_all():
    """Verify at least FPCP and IALM are listed."""
    algos = list_algorithms()
    algo_ids = {a["algorithm_id"] for a in algos}
    assert "FPCP" in algo_ids
    assert "IALM" in algo_ids


def test_list_algorithms_filtered():
    """Filter by RPCA returns only RPCA algorithms."""
    algos = list_algorithms(method_id="RPCA")
    for algo in algos:
        assert algo["method_id"] == "RPCA"
    algo_ids = {a["algorithm_id"] for a in algos}
    assert "FPCP" in algo_ids
    assert "IALM" in algo_ids


def test_unknown_algorithm_raises():
    """Verify ValueError for bad method/algorithm combo."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_algorithm("INVALID", "NONEXISTENT", np.zeros((10, 10)))


def test_decomposition_result_shapes():
    """Verify L, S, O have same shape as input."""
    rng = np.random.RandomState(42)
    data = rng.randn(50, 30)
    result = run_algorithm("RPCA", "FPCP", data)
    assert result.L.shape == data.shape
    assert result.S.shape == data.shape
    assert result.O.shape == data.shape


def test_list_methods():
    """Verify list_methods returns RPCA."""
    methods = list_methods()
    assert "RPCA" in methods
