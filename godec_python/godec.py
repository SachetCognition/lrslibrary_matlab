import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, List


def godec(X: np.ndarray, rank: int, card: int, power: int = 0) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
    """
    GoDec (Go Decomposition) Algorithm for Robust PCA
    
    Decomposes a matrix X into low-rank (L) and sparse (S) components using
    randomized low-rank approximation and hard thresholding.
    
    Parameters
    ----------
    X : np.ndarray
        n x p data matrix with n samples and p features
    rank : int
        Constraint on rank(L) <= rank
    card : int
        Cardinality constraint: card(S) <= card (number of non-zero elements in S)
    power : int, optional
        Power scheme modification (>= 0). Higher values lead to better accuracy
        but more computation time. Default is 0.
    
    Returns
    -------
    L : np.ndarray
        Low-rank component matrix
    S : np.ndarray
        Sparse component matrix
    RMSE : List[float]
        List of error values per iteration
    error : float
        Final reconstruction error: ||X-L-S|| / ||X||
    
    References
    ----------
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse Matrix
    Decomposition in Noisy Case", ICML 2011
    
    Notes
    -----
    - When m < n (more features than samples), the algorithm transposes X
      internally and transposes the results back before returning
    - Uses randomized power iteration for low-rank approximation
    - Uses hard thresholding for sparse component extraction
    """
    iter_max = 100
    error_bound = 1e-3
    iter_count = 1
    RMSE = []
    
    m, n = X.shape
    transposed = False
    
    if m < n:
        X = X.T
        m, n = X.shape
        transposed = True
    
    L = X.copy()
    S = np.zeros_like(X)
    
    while True:
        Y2 = np.random.randn(n, rank)
        
        for _ in range(power + 1):
            Y1 = L @ Y2
            Y2 = L.T @ Y1
        
        Q, R = np.linalg.qr(Y2, mode='reduced')
        L_new = (L @ Q) @ Q.T
        
        T = L - L_new + S
        L = L_new
        
        T_flat = T.flatten()
        idx = np.argsort(np.abs(T_flat))[::-1]
        
        S = np.zeros_like(X)
        S.flat[idx[:card]] = T_flat[idx[:card]]
        
        T_flat[idx[:card]] = 0
        T_residual = T_flat.reshape(T.shape)
        rmse_val = np.linalg.norm(T_residual)
        RMSE.append(rmse_val)
        
        if rmse_val < error_bound or iter_count > iter_max:
            break
        else:
            L = L + T_residual
        
        iter_count += 1
    
    LS = L + S
    error = np.linalg.norm(LS - X) / np.linalg.norm(X)
    
    if transposed:
        L = L.T
        S = S.T
    
    return L, S, RMSE, error
