import time
import numpy as np
from typing import Dict, Optional, Any
from .config import get_config

def run_algorithm(
    category: str,
    algorithm_id: str,
    data: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    config = get_config()
    
    category = category.strip().upper()
    algorithm_id = algorithm_id.strip()
    
    category_map = {
        'RPCA': config.rpca_path,
        'ST': config.st_path,
        'MC': config.mc_path,
        'LRR': config.lrr_path,
        'TTD': config.ttd_path,
        'NMF': config.nmf_path,
        'NTF': config.ntf_path,
        'TD': config.td_path
    }
    
    if category not in category_map:
        raise ValueError(f"Undefined category: {category}")
    
    params['data'] = data
    params['M'] = data
    
    if 'rows' not in params and 'cols' not in params:
        params['rows'] = data.shape[0]
        params['cols'] = data.shape[1]
    
    if category == 'MC' and 'Idx' not in params and 'Omega' not in params:
        from .utils import subsampling
        params['Idx'], params['Omega'] = subsampling(data, 0.5)
    
    if 'Idx' in params and 'Omega' not in params:
        Omega = np.zeros_like(data)
        Omega.ravel()[params['Idx']] = 1
        params['Omega'] = Omega
    
    if 'Omega' in params and 'Idx' not in params:
        params['Idx'] = np.where(params['Omega'].ravel())[0]
    
    start_time = time.time()
    
    algorithm_lower = algorithm_id.lower()
    if category == 'RPCA':
        if algorithm_lower == 'godec':
            from .algorithms.rpca.godec.run_alg import run_alg
        elif algorithm_lower in ['mog-rpca', 'mog_rpca', 'mogrpca']:
            from .algorithms.rpca.mog_rpca.run_alg import run_alg
        else:
            raise ValueError(f"Algorithm {algorithm_id} not implemented yet")
    elif category == 'MC':
        if algorithm_lower == 'svt':
            from .algorithms.mc.svt.run_alg import run_alg
        else:
            raise ValueError(f"Algorithm {algorithm_id} not implemented yet")
    elif category == 'ST':
        if algorithm_lower == 'grasta':
            from .algorithms.st.grasta.run_alg import run_alg
        elif algorithm_lower in ['reprocs', 'medrop']:
            from .algorithms.st.reprocs.run_alg import run_alg
        else:
            raise ValueError(f"Algorithm {algorithm_id} not implemented yet")
    else:
        raise ValueError(f"Category {category} not fully implemented yet")
    
    result = run_alg(data, params)
    
    cputime = time.time() - start_time
    
    if 'L' not in result:
        result['L'] = np.zeros_like(data)
    if 'S' not in result:
        result['S'] = np.zeros_like(data)
    if 'O' not in result:
        result['O'] = hard_threshold(result['S'])
    
    result['cputime'] = cputime
    if 'Omega' in params:
        result['Omega'] = params['Omega']
    if 'Idx' in params:
        result['Idx'] = params['Idx']
    
    return result

def hard_threshold(S: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.where(np.abs(S) > threshold, S, 0)
