import numpy as np
from typing import Dict, Any
from .grasta_stream import grasta_stream

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    DIM = M.shape[0]
    nframes = M.shape[1]
    
    subsampling = params.get('subsampling', 1.0)
    rank = params.get('rank', 1)
    debug_admm = params.get('debug_admm', False)
    
    OPTIONS = {
        'RANK': rank,
        'rho': 1.8,
        'MAX_MU': 10000,
        'MIN_MU': 1,
        'ITER_MAX': 20,
        'DIM_M': DIM,
        'USE_MEX': 0,
        'CONSTANT_STEP': 0
    }
    
    OPTS = {'DEBUG': debug_admm}
    status = {'init': 0, 'frame_count': 0}
    U_hat = np.zeros((1,))
    
    max_cycles = 200
    training_frames = min(20, nframes)
    
    for outiter in range(max_cycles):
        frame_order = np.random.permutation(training_frames)
        for i in range(training_frames):
            I = M[:, frame_order[i]]
            
            Z = int(np.round(subsampling * DIM))
            rp = np.random.permutation(DIM)
            idx = rp[:Z]
            I_Omega = I[idx]
            
            if debug_admm and status['frame_count'] < 5:
                print(f"Training frame {status['frame_count']} (cycle {outiter}, frame {i}):")
                OPTS['DEBUG'] = True
            else:
                OPTS['DEBUG'] = False
            
            U_hat, status, OPTS = grasta_stream(I_Omega, idx, U_hat, status, OPTIONS, OPTS)
            status['frame_count'] += 1
    
    OPTIONS['CONSTANT_STEP'] = 1e-2
    
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    
    for i in range(nframes):
        I = M[:, i]
        
        Z = int(np.round(subsampling * DIM))
        rp = np.random.permutation(DIM)
        idx = rp[:Z]
        I_Omega = I[idx]
        
        U_hat, status, OPTS = grasta_stream(I_Omega, idx, U_hat, status, OPTIONS, OPTS)
        
        L_hat = U_hat @ status['w'] * status['SCALE']
        S_hat = I - L_hat
        
        L[:, i] = L_hat
        S[:, i] = S_hat
    
    return {
        'L': L,
        'S': S,
        'O': S
    }
