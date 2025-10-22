import numpy as np
from typing import Dict, Any
from .grasta_stream import grasta_stream

def compute_svd_initialization(M: np.ndarray, rank: int, num_frames: int = 10) -> np.ndarray:
    """
    Compute SVD-based initial subspace from first batch of frames.
    
    Args:
        M: Data matrix (DIM x nframes)
        rank: Desired rank of subspace
        num_frames: Number of frames to use for initialization
    
    Returns:
        U_init: Initial subspace (DIM x rank) with orthonormal columns
    """
    num_frames = min(num_frames, M.shape[1])
    
    M_batch = M[:, :num_frames]
    
    from scipy.sparse.linalg import svds
    if M_batch.shape[0] > 1000 or M_batch.shape[1] > 1000:
        U, s, Vt = svds(M_batch, k=rank)
        U = U[:, ::-1]
    else:
        U, s, Vt = np.linalg.svd(M_batch, full_matrices=False)
        U = U[:, :rank]
    
    return U

def run_alg(M: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    
    DIM = M.shape[0]
    nframes = M.shape[1]
    
    subsampling = params.get('subsampling', 1.0)
    rank = params.get('rank', 1)
    debug_admm = params.get('debug_admm', False)
    use_svd_init = params.get('use_svd_init', True)
    svd_init_frames = params.get('svd_init_frames', 10)
    
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
    
    if use_svd_init:
        U_init = compute_svd_initialization(M, rank, svd_init_frames)
        OPTIONS['U_init'] = U_init
    else:
        OPTIONS['U_init'] = None
    
    OPTS = {'DEBUG': debug_admm}
    status = {'init': 0, 'frame_count': 0}
    U_hat = np.zeros((1,))
    
    max_cycles = 30
    training_frames = min(10, nframes)
    
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
