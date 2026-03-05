"""Video-to-matrix/tensor conversion utilities.

Ports convert_video_to_2d.m and convert_video_to_3d.m.
"""

import numpy as np


def video_to_2d(video: dict) -> np.ndarray:
    """Convert video frames to a 2D matrix.

    Ports convert_video_to_2d.m: flatten each grayscale frame to a column vector.
    Result shape = (height*width, nframes), normalized to float64 [0, 1].

    Args:
        video: Dict with 'frames' key containing np.ndarray of shape
               (nframes, height, width).

    Returns:
        2D matrix of shape (height*width, nframes), dtype float64, values in [0, 1].
    """
    frames = video["frames"]  # (nframes, height, width)
    nframes, height, width = frames.shape
    # Reshape each frame to a column vector
    matrix = frames.reshape(nframes, height * width).T  # (height*width, nframes)
    # Normalize to [0, 1] float64 (mirrors im2double)
    return matrix.astype(np.float64) / 255.0


def video_to_3d(video: dict) -> np.ndarray:
    """Convert video frames to a 3D tensor.

    Ports convert_video_to_3d.m: stack frames as (height, width, nframes).
    Normalized to float64 [0, 1].

    Args:
        video: Dict with 'frames' key containing np.ndarray of shape
               (nframes, height, width).

    Returns:
        3D tensor of shape (height, width, nframes), dtype float64, values in [0, 1].
    """
    frames = video["frames"]  # (nframes, height, width)
    # Transpose to (height, width, nframes)
    tensor = np.transpose(frames, (1, 2, 0))
    # Normalize to [0, 1] float64
    return tensor.astype(np.float64) / 255.0
