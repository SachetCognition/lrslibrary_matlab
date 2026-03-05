"""Tests for the video exporter."""

import numpy as np

from lrslibrary.algorithms.base import hard_threshold
from lrslibrary.video.exporter import mat2gray, matrix_results_to_frames


def test_matrix_results_to_frames():
    """Verify output frame shapes and dtypes."""
    height, width, nframes = 32, 48, 5
    rng = np.random.RandomState(42)

    L = rng.randn(height * width, nframes)
    S = rng.randn(height * width, nframes)
    O = (rng.rand(height * width, nframes) > 0.9).astype(np.float64)

    result = matrix_results_to_frames(L, S, O, height, width)

    assert len(result["L"]) == nframes
    assert len(result["S"]) == nframes
    assert len(result["O"]) == nframes

    for component in ("L", "S", "O"):
        for frame in result[component]:
            assert frame.shape == (height, width)
            assert frame.dtype == np.uint8


def test_hard_threshold():
    """Known input produces expected binary output."""
    # Create a sparse matrix with known values
    S = np.array([[0.0, 0.0, 5.0], [0.0, 3.0, 0.0], [10.0, 0.0, 0.0]])

    O = hard_threshold(S)

    # O should be binary
    assert set(np.unique(O)).issubset({0.0, 1.0})

    # Large values in S should produce 1.0 in O
    # beta = 0.5 * std(S)^2
    # For this S, std ~ 3.5, beta ~ 6.1
    # O[i,j] = 1 if 0.5 * S[i,j]^2 > beta
    # 0.5 * 5^2 = 12.5 > beta => 1
    # 0.5 * 3^2 = 4.5 < beta => 0
    # 0.5 * 10^2 = 50 > beta => 1
    assert O[0, 2] == 1.0  # S = 5
    assert O[2, 0] == 1.0  # S = 10
    assert O[0, 0] == 0.0  # S = 0


def test_mat2gray():
    """Verify mat2gray normalization."""
    M = np.array([[0.0, 0.5], [1.0, 2.0]])
    result = mat2gray(M)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_mat2gray_constant():
    """Verify mat2gray with constant matrix returns zeros."""
    M = np.ones((3, 3)) * 5.0
    result = mat2gray(M)
    assert np.all(result == 0)
