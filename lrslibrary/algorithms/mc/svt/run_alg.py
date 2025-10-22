import numpy as np
from typing import Dict, Any
from .svt import svt

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    Idx = params.get('Idx')
    if Idx is None:
        raise ValueError("Matrix completion requires 'Idx' parameter")
    
    MIdx = M.ravel()[Idx]
    
    maxiter = params.get('maxiter', 500)
    tol = params.get('tol', 1e-4)
    tau = params.get('tau', 2 * 5 * np.sqrt(M.size))
    delta = params.get('delta', 1.2)
    
    U, S, V, numiter, out = svt(M.shape, Idx, MIdx, tau, delta, maxiter, tol)
    
    L = U @ S @ V.T
    S_sparse = M - L
    
    return {
        'L': L,
        'S': S_sparse,
        'O': S_sparse,
        'numiter': numiter,
        'out': out
    }
