import numpy as np
from typing import Dict, Any, Tuple
from scipy.linalg import qr
from .admm_srp import admm_srp

def grasta_stream(I_Omega: np.ndarray, idx: np.ndarray, U_hat: np.ndarray, 
                   status: Dict, OPTIONS: Dict, OPTS: Dict) -> Tuple[np.ndarray, Dict, Dict]:
    """
    GRASTA streaming update for online subspace tracking.
    
    Args:
        I_Omega: Observed data at sampled indices
        idx: Indices of observed entries
        U_hat: Current subspace estimate
        status: Algorithm status dictionary
        OPTIONS: Algorithm options
        OPTS: ADMM solver options
        
    Returns:
        U_hat: Updated subspace estimate
        status: Updated status dictionary
        OPTS: Updated solver options
        
    Example:
        >>> import numpy as np
        >>> I_Omega = np.random.randn(500)
        >>> idx = np.random.choice(1000, 500, replace=False)
        >>> U_hat = np.random.randn(1000, 5)
        >>> U_hat, _ = np.linalg.qr(U_hat)
        >>> status = {'init': 0}
        >>> OPTIONS = {'DIM_M': 1000, 'RANK': 5, 'rho': 1.8, 'MAX_MU': 10000, 'MIN_MU': 1, 'ITER_MAX': 20}
        >>> OPTS = {'DEBUG': False}
        >>> U_hat, status, OPTS = grasta_stream(I_Omega, idx, U_hat, status, OPTIONS, OPTS)
    """
    DIM_M = OPTIONS['DIM_M']
    RANK = OPTIONS['RANK']
    rho = OPTIONS['rho']
    MAX_MU = OPTIONS['MAX_MU']
    MIN_MU = OPTIONS['MIN_MU']
    ITER_MAX = OPTIONS['ITER_MAX']
    USE_MEX = OPTIONS.get('USE_MEX', 0)
    CONSTANT_STEP = OPTIONS.get('CONSTANT_STEP', 0)
    
    if status['init'] == 0:
        if 'DIM_M' not in OPTIONS or 'RANK' not in OPTIONS:
            raise ValueError("OPTIONS must contain 'DIM_M' and 'RANK'")
        if 'U_init' in OPTIONS and OPTIONS['U_init'] is not None:
            U_hat = OPTIONS['U_init']
            U_hat, _ = qr(U_hat, mode='economic')
        else:
            U_hat = np.random.randn(DIM_M, RANK)
            U_hat, _ = qr(U_hat, mode='economic')
        status['init'] = 1
        status['count'] = 0
        status['w'] = np.zeros(RANK)
        status['SCALE'] = 1.0
        status['last_mu'] = MIN_MU
        status['level'] = 0
        status['step_scale'] = 0
        OPTS['TOL'] = 1e-7
        OPTS['RHO'] = rho
    
    U_Omega = U_hat[idx, :]
    
    y = I_Omega / status['SCALE']
    
    s_t, w, ldual = admm_srp(U_Omega, y, OPTS)
    
    UtDual_omega = U_Omega.T @ ldual
    gamma_2 = U_hat @ UtDual_omega
    gamma = np.zeros(DIM_M)
    gamma[idx] = ldual
    gamma = gamma - gamma_2
    
    gamma_norm = np.linalg.norm(gamma)
    w_norm = np.linalg.norm(w)
    sG = gamma_norm * w_norm
    
    if not status['step_scale'] and sG > 0:
        status['step_scale'] = 0.5 * np.pi * (1 + MIN_MU) / sG
    
    if CONSTANT_STEP > 0:
        t = CONSTANT_STEP
    else:
        if status['step_scale'] > 0 and sG > 0:
            t = status['step_scale'] * sG / (1 + status['last_mu'])
            t = min(t, np.pi / 3)
        else:
            t = 0.01
    
    if w_norm > 0 and gamma_norm > 0:
        alpha = w / w_norm
        beta = gamma / gamma_norm
        
        step = (np.cos(t) - 1) * U_hat @ (alpha[:, np.newaxis] @ alpha[np.newaxis, :])
        step = step - np.sin(t) * (beta[:, np.newaxis] @ alpha[np.newaxis, :])
        
        step_norm = np.linalg.norm(step)
        U_hat = U_hat + step
        
        U_hat, _ = qr(U_hat, mode='economic')
        
        if OPTS.get('DEBUG', False):
            if 'debug_history' not in status:
                status['debug_history'] = []
            status['debug_history'].append({
                't': t,
                'step_norm': step_norm,
                'U_norm': np.linalg.norm(U_hat)
            })
    
    status['w'] = w
    status['s_t'] = s_t
    status['ldual'] = ldual
    status['count'] += 1
    
    return U_hat, status, OPTS
