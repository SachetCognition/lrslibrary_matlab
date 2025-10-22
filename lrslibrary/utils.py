import numpy as np
from typing import Tuple

def subsampling(M: np.ndarray, observation_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    total_entries = M.size
    num_observed = int(total_entries * observation_ratio)
    
    Idx = np.random.choice(total_entries, size=num_observed, replace=False)
    Idx = np.sort(Idx)
    
    Omega = np.zeros_like(M)
    Omega.ravel()[Idx] = 1
    
    return Idx, Omega

def convert_video3d_to_2d(V: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    if V.ndim != 3:
        raise ValueError("Input must be a 3D array (height x width x frames)")
    
    m, n, p = V.shape
    M = V.reshape(m * n, p, order='F')
    
    return M, m, n, p
