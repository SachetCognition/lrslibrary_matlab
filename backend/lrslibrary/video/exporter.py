"""Result export utilities.

Ports convert_2dresults2mov.m (lines 7-61) and save_results.m (lines 3-40).
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt2d


def mat2gray(M: np.ndarray) -> np.ndarray:
    """Normalize matrix to [0, 255] uint8, matching MATLAB mat2gray + im2uint8.

    Args:
        M: Input matrix.

    Returns:
        uint8 matrix scaled to [0, 255].
    """
    min_val = M.min()
    max_val = M.max()
    if max_val - min_val == 0:
        return np.zeros_like(M, dtype=np.uint8)
    normalized = (M - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def matrix_results_to_frames(
    L: np.ndarray,
    S: np.ndarray,
    O: np.ndarray,
    height: int,
    width: int,
) -> dict[str, list[np.ndarray]]:
    """Convert decomposition result matrices back to frame sequences.

    Ports convert_2dresults2mov.m (lines 19-61). Reshapes each column
    back to (height, width), normalizes with mat2gray, applies median
    filter to O.

    Args:
        L: Low-rank matrix (height*width, nframes).
        S: Sparse matrix (height*width, nframes).
        O: Outlier matrix (height*width, nframes).
        height: Original frame height.
        width: Original frame width.

    Returns:
        Dict with keys "L", "S", "O", each a list of uint8 frames.
    """
    nframes = L.shape[1]
    result: dict[str, list[np.ndarray]] = {"L": [], "S": [], "O": []}

    for i in range(nframes):
        # Low-rank
        low_rank = L[:, i].reshape(height, width)
        result["L"].append(mat2gray(low_rank))

        # Sparse
        sparse = S[:, i].reshape(height, width)
        result["S"].append(mat2gray(sparse))

        # Outlier with median filter (matches medfilt2(Outlier, [5 5]))
        outlier = O[:, i].reshape(height, width)
        outlier_uint8 = mat2gray(outlier)
        outlier_filtered = medfilt2d(outlier_uint8.astype(np.float64), kernel_size=5)
        result["O"].append(outlier_filtered.astype(np.uint8))

    return result


def save_result_videos(
    frames_dict: dict[str, list[np.ndarray]],
    output_dir: str | Path,
    fps: float = 25.0,
) -> dict[str, str]:
    """Write L, S, O as separate .avi files.

    Ports save_results.m (lines 10-39).

    Args:
        frames_dict: Dict with "L", "S", "O" keys, each a list of uint8 frames.
        output_dir: Directory to write output files.
        fps: Frames per second for output videos.

    Returns:
        Dict mapping component name to output file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for component, frames in frames_dict.items():
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        out_path = output_dir / f"{component}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=False)
        for frame in frames:
            writer.write(frame)
        writer.release()
        paths[component] = str(out_path)

    return paths


def frames_to_png(
    frames: list[np.ndarray],
    output_dir: str | Path,
    prefix: str = "frame",
) -> list[str]:
    """Save individual frames as PNGs for frontend display.

    Args:
        frames: List of uint8 frames.
        output_dir: Directory to write PNG files.
        prefix: Filename prefix.

    Returns:
        List of output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i, frame in enumerate(frames):
        out_path = output_dir / f"{prefix}_{i:04d}.png"
        cv2.imwrite(str(out_path), frame)
        paths.append(str(out_path))

    return paths
