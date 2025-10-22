import numpy as np
from scipy.sparse.linalg import svds
from typing import Tuple, Dict, List

def svt(n: Tuple[int, int], Omega: np.ndarray, b: np.ndarray, tau: float, 
        delta: float, maxiter: int = 500, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict]:
    
    n1, n2 = n
    m = len(Omega)
    
    normb = np.linalg.norm(b)
    
    i_coords = Omega // n2
    j_coords = Omega % n2
    
    from scipy.sparse import csr_matrix
    Y = csr_matrix((b, (i_coords, j_coords)), shape=(n1, n2))
    
    if n1 * n2 < 100 * 100:
        SMALLSCALE = True
    else:
        SMALLSCALE = False
    
    normProjM = np.linalg.norm(Y.toarray(), 'fro')
    k0 = int(np.ceil(tau / (delta * normProjM)))
    y = k0 * delta * b
    
    Y = csr_matrix((y, (i_coords, j_coords)), shape=(n1, n2))
    
    r = 0
    
    out = {
        'residual': [],
        'rank': [],
        'time': [],
        'nuclearNorm': []
    }
    
    for k in range(maxiter):
        s = r + 1
        rInc = 4
        s = min(r + rInc, n1, n2)
        
        if SMALLSCALE:
            U, Sigma, Vt = np.linalg.svd(Y.toarray(), full_matrices=False)
            sigma = Sigma
        else:
            try:
                if s < min(n1, n2):
                    U, sigma, Vt = svds(Y, k=s, which='LM')
                    sort_idx_svd = np.argsort(sigma)[::-1]
                    U = U[:, sort_idx_svd]
                    sigma = sigma[sort_idx_svd]
                    Vt = Vt[sort_idx_svd, :]
                else:
                    U, sigma, Vt = np.linalg.svd(Y.toarray(), full_matrices=False)
            except:
                U, sigma, Vt = np.linalg.svd(Y.toarray(), full_matrices=False)
        
        r = np.sum(sigma > tau)
        
        if r == 0:
            r = 1
        
        U = U[:, :r]
        V = Vt[:r, :].T
        sigma = sigma[:r] - tau
        sigma = np.maximum(sigma, 0)
        
        x = (U * sigma) @ V.T
        x = x.ravel()[Omega]
        
        relRes = np.linalg.norm(x - b) / normb
        out['residual'].append(relRes)
        out['rank'].append(r)
        out['nuclearNorm'].append(np.sum(sigma))
        
        if relRes < tol:
            break
        
        if relRes > 1e5:
            break
        
        y = y + delta * (b - x)
        Y = csr_matrix((y, (i_coords, j_coords)), shape=(n1, n2))
    
    numiter = k + 1
    
    U_final = U
    Sigma_final = np.diag(sigma)
    V_final = V
    
    return U_final, Sigma_final, V_final, numiter, out
