import numpy as np
from typing import Tuple

def ncrpca(X: np.ndarray, rank: int, maxiter: int = 100, tol: float = 1e-5) -> np.ndarray:
    
    m, n = X.shape
    
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    
    lambda_s = 1.0 / np.sqrt(max(m, n))
    
    for iteration in range(maxiter):
        L_old = L.copy()
        
        U, Sigma, Vt = np.linalg.svd(X - S, full_matrices=False)
        
        if len(Sigma) > rank:
            U = U[:, :rank]
            Sigma = Sigma[:rank]
            Vt = Vt[:rank, :]
        
        L = U @ np.diag(Sigma) @ Vt
        
        T = X - L
        T_flat = T.ravel()
        threshold_idx = np.argsort(np.abs(T_flat))[::-1]
        
        S = np.zeros_like(X)
        num_sparse = int(lambda_s * m * n)
        S.ravel()[threshold_idx[:num_sparse]] = T.ravel()[threshold_idx[:num_sparse]]
        
        rel_change = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10)
        if rel_change < tol:
            break
    
    return L
