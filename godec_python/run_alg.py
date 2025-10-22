import numpy as np
from typing import Dict, Any
from godec import godec


def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standard interface for GoDec algorithm compatible with LRSLibrary structure.
    
    This function provides a standardized interface that takes a data matrix M
    and returns a dictionary with keys 'L', 'S', and 'O' (other outputs).
    
    Parameters
    ----------
    M : np.ndarray
        Input data matrix to decompose
    params : Dict[str, Any], optional
        Dictionary of parameters with keys:
        - 'rank' : int, optional
            Constraint on rank(L). Default is 1.
        - 'card' : int, optional
            Cardinality constraint on S (number of non-zero elements).
            Default is total number of elements in M.
        - 'power' : int, optional
            Power scheme modification (>= 0). Default is 0.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary with keys:
        - 'L' : np.ndarray - Low-rank component
        - 'S' : np.ndarray - Sparse component
        - 'O' : Dict[str, Any] - Other outputs containing:
            - 'RMSE' : List[float] - Error values per iteration
            - 'error' : float - Final reconstruction error
    
    Examples
    --------
    >>> import numpy as np
    >>> M = np.random.randn(100, 50)
    >>> result = run_alg(M)
    >>> L = result['L']
    >>> S = result['S']
    >>> rmse = result['O']['RMSE']
    >>> error = result['O']['error']
    
    >>> result = run_alg(M, params={'rank': 5, 'card': 500, 'power': 1})
    """
    if params is None:
        params = {}
    
    rank = params.get('rank', 1)
    card = params.get('card', M.size)
    power = params.get('power', 0)
    
    L, S, RMSE, error = godec(M, rank=rank, card=card, power=power)
    
    result = {
        'L': L,
        'S': S,
        'O': {
            'RMSE': RMSE,
            'error': error
        }
    }
    
    return result
