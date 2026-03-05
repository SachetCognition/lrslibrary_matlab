"""Video file loading using OpenCV.

Ports utils/load_video_file.m (lines 4-29).
"""

from pathlib import Path

import cv2
import numpy as np


def load_video(file_path: str | Path) -> dict:
    """Load a video file and return frames with metadata.

    Supports .avi and .mp4 formats. Converts frames to grayscale.

    Args:
        file_path: Path to the video file.

    Returns:
        Dict with keys:
            - frames: np.ndarray of shape (nframes, height, width), dtype uint8
            - width: int
            - height: int
            - nframes: int
            - fps: float

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file cannot be opened or has no frames.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # default fallback

    frames_list: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if needed (mirrors MATLAB rgb2gray)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_list.append(frame)

    cap.release()

    if len(frames_list) == 0:
        raise ValueError(f"No frames found in video: {file_path}")

    frames = np.stack(frames_list, axis=0)  # (nframes, height, width)

    return {
        "frames": frames,
        "width": width,
        "height": height,
        "nframes": len(frames_list),
        "fps": fps,
    }
