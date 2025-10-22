import numpy as np
from scipy.sparse.linalg import svds
from typing import Tuple, Dict, List

def svt(n: Tuple[int, int], Omega: np.ndarray, b: np.ndarray, tau: float, 
        delta: float, maxiter: int = 500, tol: float = 1e-4,
        smallscale_threshold: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict]:
    """
    Singular Value Thresholding (SVT) for matrix completion.
    
    Args:
        n: Matrix dimensions (n1, n2)
        Omega: Linear indices of observed entries
        b: Values at observed entries
        tau: Singular value threshold
        delta: Step size
        maxiter: Maximum iterations (default 500)
        tol: Convergence tolerance (default 1e-4)
        smallscale_threshold: Threshold for using full SVD vs sparse SVD (default 10000)
        
    Returns:
        U_final: Left singular vectors
        Sigma_final: Diagonal matrix of singular values
        V_final: Right singular vectors
        numiter: Number of iterations performed
        out: Dictionary with convergence history
        
    Raises:
        ValueError: If input parameters are invalid
        
    Example:
        >>> import numpy as np
        >>> n = (100, 50)
        >>> Omega = np.random.choice(100*50, size=100*50//2, replace=False)
        >>> b = np.random.randn(len(Omega))
        >>> U, Sigma, V, iters, out = svt(n, Omega, b, tau=10.0, delta=1.2)
    """
    from ....utils import validate_positive_param
    
    n1, n2 = n
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Matrix dimensions must be positive, got {n}")
    
    m = len(Omega)
    if m == 0:
        raise ValueError("Omega (observed indices) cannot be empty")
    
    if len(b) != m:
        raise ValueError(f"Length of b ({len(b)}) must match length of Omega ({m})")
    
    validate_positive_param(tau, "tau")
    validate_positive_param(delta, "delta")
    validate_positive_param(tol, "tol")
    validate_positive_param(maxiter, "maxiter")
    
    normb = np.linalg.norm(b)
    if normb < 1e-10:
        raise ValueError("Observed values b have near-zero norm")
    
    i_coords = Omega // n2
    j_coords = Omega % n2
    
    from scipy.sparse import csr_matrix
    Y = csr_matrix((b, (i_coords, j_coords)), shape=(n1, n2))
    
    SMALLSCALE = (n1 * n2 < smallscale_threshold)
    
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
            if s < min(n1, n2):
                try:
                    U, sigma, Vt = svds(Y, k=s, which='LM')
                    sort_idx_svd = np.argsort(sigma)[::-1]
                    U = U[:, sort_idx_svd]
                    sigma = sigma[sort_idx_svd]
                    Vt = Vt[sort_idx_svd, :]
                except (np.linalg.LinAlgError, ValueError) as e:
                    U, sigma, Vt = np.linalg.svd(Y.toarray(), full_matrices=False)
            else:
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
