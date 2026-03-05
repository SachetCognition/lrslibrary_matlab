"""Tests for the video loader."""

import pytest

from lrslibrary.video.loader import load_video


def test_load_video_avi(sample_video_path):
    """Load synthetic .avi, verify shape and metadata."""
    video = load_video(sample_video_path)
    assert "frames" in video
    assert "width" in video
    assert "height" in video
    assert "nframes" in video
    assert "fps" in video
    assert video["nframes"] == 10
    assert video["frames"].shape == (10, 64, 64)


def test_load_video_dimensions(sample_video_path):
    """Verify width, height, nframes match."""
    video = load_video(sample_video_path)
    assert video["width"] == 64
    assert video["height"] == 64
    assert video["nframes"] == 10
    assert video["fps"] > 0


def test_load_video_file_not_found():
    """Verify FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_video("/nonexistent/path/video.avi")
