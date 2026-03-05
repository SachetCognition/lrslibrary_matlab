"""Tests for the video converter."""

import numpy as np

from lrslibrary.video.converter import video_to_2d, video_to_3d


def _make_test_video():
    """Create a simple test video dict."""
    rng = np.random.RandomState(42)
    nframes, height, width = 5, 32, 48
    frames = rng.randint(0, 256, (nframes, height, width), dtype=np.uint8)
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "nframes": nframes,
        "fps": 25.0,
    }


def test_video_to_2d_shape():
    """Result shape should be (height*width, nframes)."""
    video = _make_test_video()
    matrix = video_to_2d(video)
    assert matrix.shape == (32 * 48, 5)


def test_video_to_2d_normalized():
    """Values should be in [0, 1]."""
    video = _make_test_video()
    matrix = video_to_2d(video)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0
    assert matrix.dtype == np.float64


def test_video_to_3d_shape():
    """Result shape should be (height, width, nframes)."""
    video = _make_test_video()
    tensor = video_to_3d(video)
    assert tensor.shape == (32, 48, 5)


def test_roundtrip_2d():
    """Convert to 2d and reshape back, verify consistency."""
    video = _make_test_video()
    matrix = video_to_2d(video)

    # Reconstruct frames from matrix columns
    height, width = video["height"], video["width"]
    for i in range(video["nframes"]):
        frame_reconstructed = (matrix[:, i].reshape(height, width) * 255).astype(np.uint8)
        frame_original = video["frames"][i]
        np.testing.assert_array_equal(frame_reconstructed, frame_original)
