import numpy as np
from typing import Tuple, List

def godec(X: np.ndarray, rank: int, card: int, power: int = 0) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
    iter_max = 100
    error_bound = 1e-3
    iter_count = 1
    RMSE = []
    
    m, n = X.shape
    transpose_flag = False
    if m < n:
        X = X.T
        m, n = n, m
        transpose_flag = True
    
    L = X.copy()
    S = np.zeros_like(X)
    
    while True:
        Y2 = np.random.randn(n, rank)
        for i in range(power + 1):
            Y1 = L @ Y2
            Y2 = L.T @ Y1
        
        Q, R = np.linalg.qr(Y2, mode='reduced')
        L_new = (L @ Q) @ Q.T
        
        T = L - L_new + S
        L = L_new
        
        T_flat = T.ravel()
        idx = np.argsort(np.abs(T_flat))[::-1]
        
        S = np.zeros_like(X)
        S.ravel()[idx[:card]] = T.ravel()[idx[:card]]
        
        T.ravel()[idx[:card]] = 0
        rmse = np.linalg.norm(T)
        RMSE.append(rmse)
        
        if rmse < error_bound or iter_count > iter_max:
            break
        else:
            L = L + T
        
        iter_count += 1
    
    LS = L + S
    error = np.linalg.norm(LS - X) / np.linalg.norm(X)
    
    if transpose_flag:
        L = L.T
        S = S.T
    
    return L, S, RMSE, error
