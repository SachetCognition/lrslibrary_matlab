import numpy as np
from typing import Dict, Any
from .godec import godec

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    rank = params.get('rank', 1)
    card = params.get('card', M.size)
    power = params.get('power', 0)
    
    L, S, RMSE, error = godec(M, rank, card, power)
    
    return {
        'L': L,
        'S': S,
        'O': S,
        'RMSE': RMSE,
        'error': error
    }
