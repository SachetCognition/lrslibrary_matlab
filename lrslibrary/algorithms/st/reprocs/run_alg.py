import numpy as np
from typing import Dict, Any
from .ncrpca import ncrpca
from .reprocs import reprocs

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    t_train = params.get('t_train', 10)
    rank_init = params.get('rank_init', 5)
    alpha = params.get('alpha', 60)
    K = params.get('K', 3)
    
    TrainData = M[:, :t_train]
    
    L_hat_init = ncrpca(TrainData, rank_init)
    
    mu = np.mean(L_hat_init, axis=1)
    
    U_temp, S_temp, _ = np.linalg.svd((L_hat_init - mu[:, np.newaxis]) / np.sqrt(t_train), full_matrices=False)
    ss1 = S_temp
    L_init = U_temp[:, :rank_init]
    
    theta_thresh = 20 * np.pi / 180
    ev_thresh = 0.1 * ss1[rank_init - 1] * np.sin(theta_thresh)**2
    
    BG, FG, L_hat, S_hat, T_hat, t_hat = reprocs(
        M[:, t_train:], L_init, mu, ev_thresh, alpha, K
    )
    
    return {
        'L': BG,
        'S': FG,
        'O': FG
    }
