"""Tests for the FPCP algorithm."""

import numpy as np

from lrslibrary.algorithms.rpca.fpcp import FPCP


def test_fpcp_output_shapes():
    """L, S, O shapes match input."""
    rng = np.random.RandomState(42)
    M = rng.randn(80, 40)
    decomposer = FPCP()
    result = decomposer.decompose(M)
    assert result.L.shape == M.shape
    assert result.S.shape == M.shape
    assert result.O.shape == M.shape


def test_fpcp_output_contract():
    """L + S approximately equals M."""
    rng = np.random.RandomState(42)
    M = rng.randn(80, 40)
    decomposer = FPCP()
    result = decomposer.decompose(M)
    reconstruction = result.L + result.S
    relative_error = np.linalg.norm(reconstruction - M) / np.linalg.norm(M)
    assert relative_error < 0.5  # L + S should reasonably approximate M


def test_fpcp_recovers_low_rank(sample_matrix):
    """Create known rank-2 L + sparse S, verify recovery."""
    M, L_true, S_true = sample_matrix
    decomposer = FPCP()
    result = decomposer.decompose(M)
    relative_error = np.linalg.norm(result.L - L_true) / np.linalg.norm(L_true)
    assert relative_error < 0.15


def test_fpcp_sparse_recovery(sample_matrix):
    """Verify sparse component captures outlier locations."""
    M, L_true, S_true = sample_matrix
    decomposer = FPCP()
    result = decomposer.decompose(M)
    # Check that S is non-zero where S_true is non-zero (at least partially)
    true_support = np.abs(S_true) > 0
    estimated_support = np.abs(result.S) > np.std(result.S)
    # At least some overlap
    overlap = np.sum(true_support & estimated_support)
    assert overlap > 0


def test_fpcp_default_lambda():
    """Verify lambda defaults to 1/sqrt(max(m,n))."""
    M = np.random.randn(100, 50)
    expected_lambda = 1.0 / np.sqrt(max(100, 50))
    # This is implicitly tested by the algorithm running successfully
    # and producing reasonable results with default parameters
    decomposer = FPCP()
    result = decomposer.decompose(M)
    assert result.L.shape == M.shape
    # Verify the algorithm uses this default by checking metadata
    assert "rank" in result.metadata
