import numpy as np
from typing import Tuple, List

def godec(X: np.ndarray, rank: int, card: int, power: int = 0, 
          iter_max: int = 100, error_bound: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
    """
    Go Decomposition (GoDec) for robust PCA using randomized low-rank approximation.
    
    Args:
        X: Input data matrix (m x n)
        rank: Desired rank of low-rank component
        card: Cardinality (number of non-zero elements in sparse component)
        power: Power scheme parameter (>= 0, default 0). Higher values improve accuracy but increase cost.
        iter_max: Maximum number of iterations (default 100)
        error_bound: Convergence tolerance (default 1e-3)
        
    Returns:
        L: Low-rank component
        S: Sparse component
        RMSE: List of RMSE values per iteration
        error: Final relative error ||L+S-X||/||X||
        
    Raises:
        ValueError: If input parameters are invalid
        
    Example:
        >>> import numpy as np
        >>> X = np.random.randn(100, 50) 
        >>> L, S, RMSE, error = godec(X, rank=5, card=100, power=1)
    """
    from ....utils import validate_input_matrix, validate_rank, validate_positive_param
    
    validate_input_matrix(X, "X")
    validate_rank(rank, X, "rank")
    
    if not isinstance(card, int) or card < 0:
        raise ValueError(f"card must be a non-negative integer, got {card}")
    
    if not isinstance(power, int) or power < 0:
        raise ValueError(f"power must be a non-negative integer, got {power}")
    
    validate_positive_param(iter_max, "iter_max")
    validate_positive_param(error_bound, "error_bound")
    
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
