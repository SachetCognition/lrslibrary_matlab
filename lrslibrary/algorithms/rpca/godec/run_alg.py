import numpy as np
import time
from typing import Dict, Any
from .godec import godec

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    rank = params.get('rank', 1)
    card = params.get('card', M.size)
    power = params.get('power', 0)
    iter_max = params.get('iter_max', 100)
    error_bound = params.get('error_bound', 1e-3)
    
    start_time = time.time()
    L, S, RMSE, error = godec(M, rank, card, power, iter_max, error_bound)
    cputime = time.time() - start_time
    
    return {
        'L': L,
        'S': S,
        'O': S,
        'RMSE': RMSE,
        'error': error,
        'cputime': cputime
    }
