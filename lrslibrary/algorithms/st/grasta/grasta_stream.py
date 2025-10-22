import numpy as np
from typing import Dict, Any, Tuple
from scipy.linalg import qr

def grasta_stream(I_Omega: np.ndarray, idx: np.ndarray, U_hat: np.ndarray, 
                   status: Dict, OPTIONS: Dict, OPTS: Dict) -> Tuple[np.ndarray, Dict, Dict]:
    
    DIM_M = OPTIONS['DIM_M']
    RANK = OPTIONS['RANK']
    rho = OPTIONS['rho']
    MAX_MU = OPTIONS['MAX_MU']
    MIN_MU = OPTIONS['MIN_MU']
    ITER_MAX = OPTIONS['ITER_MAX']
    USE_MEX = OPTIONS.get('USE_MEX', 0)
    CONSTANT_STEP = OPTIONS.get('CONSTANT_STEP', 0)
    
    if status['init'] == 0:
        U_hat = np.random.randn(DIM_M, RANK)
        U_hat, _ = qr(U_hat, mode='economic')
        status['init'] = 1
        status['count'] = 0
        status['w'] = np.zeros(RANK)
        status['SCALE'] = 1.0
        OPTS['step_size'] = 0.1
    
    Z = len(idx)
    U_Omega = U_hat[idx, :]
    
    y = I_Omega / status['SCALE']
    
    w = status['w']
    
    for iter_admm in range(ITER_MAX):
        w_old = w.copy()
        
        grad = U_Omega.T @ (U_Omega @ w - y)
        
        if CONSTANT_STEP:
            step_size = CONSTANT_STEP
        else:
            step_size = OPTS.get('step_size', 0.1)
        
        w = w - step_size * grad
        
        if np.linalg.norm(w - w_old) < 1e-5:
            break
    
    s = y - U_Omega @ w
    
    grad_U = np.zeros((DIM_M, RANK))
    grad_U[idx, :] = -s[:, np.newaxis] @ w[np.newaxis, :]
    
    if CONSTANT_STEP:
        step_size_U = CONSTANT_STEP
    else:
        step_size_U = 0.01
    
    U_hat = U_hat - step_size_U * grad_U
    
    U_hat, _ = qr(U_hat, mode='economic')
    
    status['w'] = w
    status['count'] += 1
    
    return U_hat, status, OPTS
