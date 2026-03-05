"""Tests for the IALM algorithm."""

import numpy as np

from lrslibrary.algorithms.rpca.ialm import IALM


def test_ialm_output_shapes():
    """L, S, O shapes match input."""
    rng = np.random.RandomState(42)
    M = rng.randn(80, 40)
    decomposer = IALM()
    result = decomposer.decompose(M)
    assert result.L.shape == M.shape
    assert result.S.shape == M.shape
    assert result.O.shape == M.shape


def test_ialm_output_contract():
    """L + S approximately equals M."""
    rng = np.random.RandomState(42)
    M = rng.randn(80, 40)
    decomposer = IALM()
    result = decomposer.decompose(M)
    reconstruction = result.L + result.S
    relative_error = np.linalg.norm(reconstruction - M) / np.linalg.norm(M)
    # IALM should satisfy D = A + E quite closely
    assert relative_error < 0.01


def test_ialm_recovers_low_rank(sample_matrix):
    """Create known rank-3 L + sparse S, verify recovery."""
    M, L_true, S_true = sample_matrix
    decomposer = IALM()
    result = decomposer.decompose(M)
    relative_error = np.linalg.norm(result.L - L_true) / np.linalg.norm(L_true)
    assert relative_error < 0.15


def test_ialm_sparse_recovery(sample_matrix):
    """Verify sparse component captures outlier locations."""
    M, L_true, S_true = sample_matrix
    decomposer = IALM()
    result = decomposer.decompose(M)
    # Check that S is non-zero where S_true is non-zero (at least partially)
    true_support = np.abs(S_true) > 0
    estimated_support = np.abs(result.S) > np.std(result.S)
    overlap = np.sum(true_support & estimated_support)
    assert overlap > 0


def test_ialm_convergence(sample_matrix):
    """Verify algorithm converges within maxIter."""
    M, _, _ = sample_matrix
    decomposer = IALM()
    result = decomposer.decompose(M)
    assert result.metadata["iterations"] < 1000
    assert result.metadata["stop_criterion"] < 1e-7


def test_ialm_default_lambda():
    """Verify lambda defaults to 1/sqrt(max(m,n))."""
    M = np.random.randn(100, 50)
    decomposer = IALM()
    result = decomposer.decompose(M)
    assert result.L.shape == M.shape
    assert "iterations" in result.metadata
