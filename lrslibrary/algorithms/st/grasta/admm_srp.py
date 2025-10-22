import numpy as np
from typing import Tuple

def shrinkage(a: np.ndarray, kappa: float) -> np.ndarray:
    """Soft-thresholding operator for L1 minimization."""
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)

def admm_srp(U: np.ndarray, v: np.ndarray, OPTS: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve sparse residual pursuit (SRP) via ADMM.
    
    Solves: minimize ||s||_1 subject to Uw + s - v = 0
    
    Args:
        U: Observed subspace matrix (m x n)
        v: Observed data vector (m,)
        OPTS: Options dict with RHO, TOL, MAX_ITER
    
    Returns:
        s: Sparse component
        w: Weights
        y: Dual variable (Lagrange multiplier)
    """
    rho = OPTS.get('RHO', 1.8)
    TOL = OPTS.get('TOL', 1e-7)
    MAX_ITER = OPTS.get('MAX_ITER', 50)
    DEBUG = OPTS.get('DEBUG', False)
    
    m, n = U.shape
    
    w = np.zeros(n)
    s = np.zeros(m)
    y = np.zeros(m)
    mu = 1.25 / np.linalg.norm(v)
    
    try:
        UtU_inv = np.linalg.inv(U.T @ U)
        P = UtU_inv @ U.T
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(U)
    
    converge = False
    iter_count = 0
    initial_residual = None
    
    while not converge and iter_count < MAX_ITER:
        iter_count += 1
        
        w = P @ (v - s - y / mu)
        
        Uw = U @ w
        s = shrinkage(v - Uw - y / mu, 1.0 / mu)
        
        h = Uw + s - v
        residual_norm = np.linalg.norm(h)
        
        if initial_residual is None:
            initial_residual = residual_norm
        
        y = y + mu * h
        
        mu = rho * mu
        
        if residual_norm < TOL:
            converge = True
    
    w_norm = np.linalg.norm(w)
    s_norm = np.linalg.norm(s)
    if DEBUG:
        ratio = w_norm / s_norm if s_norm > 0 else 0
        print(f"  ADMM: iters={iter_count}/{MAX_ITER}, converged={converge}, " +
              f"residual={residual_norm:.2e}, ||w||={w_norm:.4f}, " +
              f"||s||={s_norm:.4f}, w/s={ratio:.4f}, sparsity={np.sum(np.abs(s) > 1e-6) / len(s):.2%}")
    
    return s, w, y
