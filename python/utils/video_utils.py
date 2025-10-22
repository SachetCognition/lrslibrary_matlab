import numpy as np
import cv2
from pathlib import Path


def video_to_matrix(video_path):
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    
    cap.release()
    
    frames_array = np.array(frames)
    m, n, p = height, width, len(frames)
    
    M = frames_array.transpose(1, 2, 0).reshape(m * n, p)
    M = M.astype(np.float64) / 255.0
    
    return M, m, n, p, fps


def matrix_to_video(M, m, n, output_path, fps=30.0, normalize=True):
    p = M.shape[1]
    
    M_reshaped = M.reshape(m, n, p)
    
    if normalize:
        M_min = M_reshaped.min()
        M_max = M_reshaped.max()
        if M_max > M_min:
            M_reshaped = (M_reshaped - M_min) / (M_max - M_min)
        M_reshaped = (M_reshaped * 255).astype(np.uint8)
    else:
        M_reshaped = (np.clip(M_reshaped, 0, 1) * 255).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (n, m), isColor=False)
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video file: {output_path}")
    
    for i in range(p):
        frame = M_reshaped[:, :, i]
        out.write(frame)
    
    out.release()


def process_video_with_algorithm(video_path, algorithm_func, params=None, output_dir=None):
    if output_dir is None:
        output_dir = Path('output')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    M, m, n, p, fps = video_to_matrix(video_path)
    
    results = algorithm_func(M, params)
    
    L = results['L']
    S = results['S']
    
    video_name = Path(video_path).stem
    
    input_path = output_dir / f"{video_name}_input.mp4"
    low_rank_path = output_dir / f"{video_name}_low_rank.mp4"
    sparse_path = output_dir / f"{video_name}_sparse.mp4"
    output_path = output_dir / f"{video_name}_output.mp4"
    
    matrix_to_video(M, m, n, input_path, fps, normalize=False)
    matrix_to_video(L, m, n, low_rank_path, fps, normalize=True)
    matrix_to_video(S, m, n, sparse_path, fps, normalize=True)
    matrix_to_video(L, m, n, output_path, fps, normalize=True)
    
    return {
        'input': str(input_path),
        'output': str(output_path),
        'L': str(low_rank_path),
        'S': str(sparse_path),
        'results': results
    }
