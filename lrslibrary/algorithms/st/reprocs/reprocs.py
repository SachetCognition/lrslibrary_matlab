import numpy as np
from typing import Tuple, Dict
from scipy.linalg import svd

def reprocs(M: np.ndarray, L_init: np.ndarray, mu: np.ndarray, 
            ev_thresh: float, alpha: float, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n, t = M.shape
    r = L_init.shape[1]
    
    BG = np.zeros_like(M)
    FG = np.zeros_like(M)
    L_hat = L_init.copy()
    P = L_hat @ L_hat.T
    
    T_hat = np.zeros(t, dtype=int)
    t_hat = 0
    S_hat = np.zeros_like(M)
    
    lambda_s = alpha / np.sqrt(n)
    
    for frame_idx in range(t):
        y = M[:, frame_idx]
        
        y_centered = y - mu
        
        y_proj = L_hat @ (L_hat.T @ y_centered)
        s = y_centered - y_proj
        
        s_flat = s.ravel()
        threshold_idx = np.argsort(np.abs(s_flat))[::-1]
        
        s_sparse = np.zeros_like(s)
        s_sparse.ravel()[threshold_idx[:K]] = s.ravel()[threshold_idx[:K]]
        
        l_hat = y - s_sparse
        
        y_centered_new = l_hat - mu
        
        ev = np.linalg.norm(y_centered_new - L_hat @ (L_hat.T @ y_centered_new))**2
        
        if ev > ev_thresh:
            T_hat[frame_idx] = 1
            t_hat += 1
            
            if t_hat >= 2 * r:
                recent_frames = np.where(T_hat[:frame_idx+1] == 1)[0]
                if len(recent_frames) >= r:
                    train_data = M[:, recent_frames[-2*r:]] - mu[:, np.newaxis]
                    U_temp, S_temp, _ = svd(train_data / np.sqrt(len(recent_frames[-2*r:])), full_matrices=False)
                    L_hat = U_temp[:, :r]
                    P = L_hat @ L_hat.T
        else:
            y_centered_for_update = l_hat - mu
            proj_y = L_hat @ (L_hat.T @ y_centered_for_update)
            residual = y_centered_for_update - proj_y
            
            if np.linalg.norm(residual) > 1e-10:
                L_hat_new = L_hat + (residual[:, np.newaxis] @ (residual[:, np.newaxis].T @ L_hat)) / (np.linalg.norm(residual)**2 + 1e-10)
                L_hat, _ = np.linalg.qr(L_hat_new)
                P = L_hat @ L_hat.T
        
        BG[:, frame_idx] = l_hat
        FG[:, frame_idx] = s_sparse
        S_hat[:, frame_idx] = s_sparse
    
    return BG, FG, L_hat, S_hat, T_hat, t_hat
