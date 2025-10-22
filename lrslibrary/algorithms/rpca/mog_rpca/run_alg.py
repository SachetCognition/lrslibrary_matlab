import numpy as np
from typing import Dict, Any
from .mog_rpca import mog_rpca

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    r = params.get('r', 1)
    
    param = {
        'mog_k': params.get('mog_k', 3),
        'lr_init': params.get('lr_init', 'SVD'),
        'maxiter': params.get('maxiter', 100),
        'initial_rank': params.get('initial_rank', 2 * r),
        'tol': params.get('tol', 1e-3)
    }
    
    lr_prior = {
        'a0': params.get('lr_a0', 1e-6),
        'b0': params.get('lr_b0', 1e-6)
    }
    
    mog_prior = {
        'mu0': params.get('mog_mu0', 0),
        'c0': params.get('mog_c0', 1e-3),
        'd0': params.get('mog_d0', 1e-3),
        'alpha0': params.get('mog_alpha0', 1e-3),
        'beta0': params.get('mog_beta0', 1e-3)
    }
    
    lr_model, mog_model, final_rank = mog_rpca(M, param, lr_prior, mog_prior)
    
    L = lr_model['U'] @ lr_model['V'].T
    S = M - L
    
    return {
        'L': L,
        'S': S,
        'O': S,
        'rank': final_rank,
        'lr_model': lr_model,
        'mog_model': mog_model
    }
