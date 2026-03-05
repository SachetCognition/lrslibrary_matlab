"""Pytest fixtures for LRSLibrary tests."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lrslibrary.api.app import app


@pytest.fixture
def sample_matrix():
    """Create a 100x50 matrix = rank-3 L + 5% sparse S.

    Returns:
        Tuple of (M, L_true, S_true) where M = L_true + S_true.
    """
    rng = np.random.RandomState(42)
    m, n = 100, 50
    rank = 3

    # Low-rank component
    U = rng.randn(m, rank)
    V = rng.randn(rank, n)
    L_true = U @ V

    # Sparse component (5% non-zero entries)
    S_true = np.zeros((m, n))
    num_sparse = int(0.05 * m * n)
    sparse_indices = rng.choice(m * n, num_sparse, replace=False)
    S_true.flat[sparse_indices] = rng.randn(num_sparse) * 10

    M = L_true + S_true
    return M, L_true, S_true


@pytest.fixture
def sample_video_path():
    """Generate a small 10-frame 64x64 synthetic test video as .avi.

    Returns:
        Path to the temporary .avi file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.avi"
        height, width, nframes = 64, 64, 10
        fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height), isColor=False)

        rng = np.random.RandomState(123)
        for i in range(nframes):
            # Create a frame with a moving bright rectangle on dark background
            frame = np.zeros((height, width), dtype=np.uint8)
            x = (i * 5) % (width - 20)
            y = (i * 3) % (height - 20)
            frame[y : y + 20, x : x + 20] = 200
            # Add some noise
            noise = rng.randint(0, 30, (height, width), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            writer.write(frame)

        writer.release()
        yield str(video_path)


@pytest.fixture
def api_client():
    """FastAPI TestClient instance."""
    return TestClient(app)
