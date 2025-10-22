import numpy as np
from scipy.special import digamma as psi
from typing import Dict, Tuple

def mog_rpca(Y: np.ndarray, param: Dict, lr_prior: Dict, mog_prior: Dict) -> Tuple[Dict, Dict, int]:
    
    m, n = Y.shape
    mn = m * n
    
    maxiter = param.get('maxiter', 100)
    tol = param.get('tol', 1e-4)
    mog_k = param.get('mog_k', 3)
    lr_init = param.get('lr_init', 'SVD')
    initial_rank = param.get('initial_rank', min(m, n))
    
    k = mog_k
    
    a0 = lr_prior.get('a0', 1e-6)
    b0 = lr_prior.get('b0', 1e-6)
    
    mu0 = mog_prior.get('mu0', 0)
    c0 = mog_prior.get('c0', 1e-3)
    d0 = mog_prior.get('d0', 1e-3)
    alpha0 = mog_prior.get('alpha0', 1e-3)
    beta0 = mog_prior.get('beta0', 1e-3)
    
    Y2sum = np.sum(Y**2)
    scale2 = Y2sum / mn
    scale = np.sqrt(scale2)
    
    if lr_init == 'SVD':
        U_full, s_full, Vt_full = np.linalg.svd(Y, full_matrices=False)
        r = initial_rank
        U = U_full[:, :r] * np.sqrt(s_full[:r])
        V = (Vt_full[:r, :].T) * np.sqrt(s_full[:r])
        Sigma_U = np.tile(scale * np.eye(r), (m, 1, 1))
        Sigma_V = np.tile(scale * np.eye(r), (n, 1, 1))
        gammas = scale * np.ones(r)
    elif lr_init == 'rand':
        r = initial_rank
        U = np.random.randn(m, r) * np.sqrt(scale)
        V = np.random.randn(n, r) * np.sqrt(scale)
        Sigma_U = np.tile(scale * np.eye(r), (m, 1, 1))
        Sigma_V = np.tile(scale * np.eye(r), (n, 1, 1))
        gammas = scale * np.ones(r)
    
    lr_model = {
        'U': U,
        'V': V,
        'Sigma_U': Sigma_U,
        'Sigma_V': Sigma_V,
        'gammas': gammas
    }
    L = U @ V.T
    
    E = Y - L
    R = R_initialization(E.ravel(), k)
    nk = np.sum(R, axis=0)
    nxbar = E.ravel() @ R
    
    mog_model = {
        'c': c0 + nk / 2,
        'beta': beta0 + nk,
        'R': R.reshape(m, n, k)
    }
    mog_model['d'] = d0 + 0.5 * ((E.ravel()**2 @ R) + beta0 * mu0**2 - 
                                   (1 / mog_model['beta']) * (nxbar + beta0 * mu0)**2)
    mog_model['mu'] = (1 / mog_model['beta']) * (beta0 * mu0 + nxbar)
    
    for iter_num in range(maxiter):
        L_old = L.copy()
        
        lr_model, r, E_YminusUV, E_YminusUV_2 = lr_update(Y, lr_model, mog_model, r, lr_prior)
        L = lr_model['U'] @ lr_model['V'].T
        
        mog_model = mog_vmax(mog_model, mog_prior, E_YminusUV, E_YminusUV_2)
        mog_model = mog_vexp(mog_model, E_YminusUV, E_YminusUV_2)
        
        if np.linalg.norm(L - L_old, 'fro') / np.linalg.norm(L_old, 'fro') < tol:
            break
    
    label = np.argmax(mog_model['R'], axis=2)
    mog_model['label'] = label
    
    return lr_model, mog_model, r

def lr_update(Y, lr_model, mog_model, r, lr_prior):
    m, n = Y.shape
    
    a0 = lr_prior.get('a0', 1e-6)
    b0 = lr_prior.get('b0', 1e-6)
    
    U = lr_model['U']
    V = lr_model['V']
    Sigma_U = lr_model['Sigma_U']
    Sigma_V = lr_model['Sigma_V']
    gammas = lr_model['gammas']
    
    R = mog_model['R']
    c = mog_model['c']
    d = mog_model['d']
    mu = mog_model['mu']
    
    k = len(mu)
    
    tau = c / d
    Gam = np.diag(gammas)
    Rtau = np.einsum('ijk,k->ij', R, tau)
    Rtaumu = np.einsum('ijk,k->ij', R, tau * mu)
    RtauYmu = Rtau * Y - Rtaumu
    
    re_Sigma_V = Sigma_V.reshape(r * r, n)
    diagsU = np.zeros(r)
    temp_U = np.zeros((r, r, m))
    for i in range(m):
        Sigma_U[i] = np.linalg.inv(
            (re_Sigma_V @ Rtau[i, :]).reshape(r, r) +
            (V.T * Rtau[i, :]) @ V + Gam
        )
        U[i, :] = (RtauYmu[i, :] @ V) @ Sigma_U[i]
        diagsU += np.diag(Sigma_U[i])
        temp_U[:, :, i] = Sigma_U[i] + np.outer(U[i, :], U[i, :])
    
    re_Sigma_U = Sigma_U.reshape(r * r, m)
    diagsV = np.zeros(r)
    temp_V = np.zeros((r, r, n))
    for j in range(n):
        Sigma_V[j] = np.linalg.inv(
            (re_Sigma_U @ Rtau[:, j]).reshape(r, r) +
            (U.T * Rtau[:, j]) @ U + Gam
        )
        V[j, :] = (RtauYmu[:, j].T @ U) @ Sigma_V[j]
        diagsV += np.diag(Sigma_V[j])
        temp_V[:, :, j] = Sigma_V[j] + np.outer(V[j, :], V[j, :])
    
    gammas = (2 * a0 + m + n) / (2 * b0 + np.sum(U**2, axis=0) + diagsU + np.sum(V**2, axis=0) + diagsV)
    
    dim_thr = 1e2
    max_gamma = np.min(gammas) * dim_thr
    indices = gammas <= max_gamma
    
    if np.sum(indices) > 0:
        U = U[:, indices]
        V = V[:, indices]
        gammas = gammas[indices]
        Sigma_U = Sigma_U[:, indices][:, :, indices]
        Sigma_V = Sigma_V[:, indices][:, :, indices]
        temp_U = temp_U[indices][:, indices]
        temp_V = temp_V[indices][:, indices]
        r = U.shape[1]
    
    lr_model['U'] = U
    lr_model['V'] = V
    lr_model['Sigma_U'] = Sigma_U
    lr_model['Sigma_V'] = Sigma_V
    lr_model['gammas'] = gammas
    
    E_YminusUV = Y - U @ V.T
    E_YminusUV_2 = Y**2 - 2 * Y * (U @ V.T) + np.einsum('ijk,jlk->il', temp_U.transpose(2, 0, 1), temp_V.transpose(2, 0, 1))
    
    return lr_model, r, E_YminusUV, E_YminusUV_2

def mog_vmax(mog_model, mog_prior, E_YminusUV, E_YminusUV_2):
    alpha0 = mog_prior.get('alpha0', 1e-3)
    beta0 = mog_prior.get('beta0', 1e-3)
    mu0 = mog_prior.get('mu0', 0)
    c0 = mog_prior.get('c0', 1e-3)
    d0 = mog_prior.get('d0', 1e-3)
    R = mog_model['R']
    
    m, n, k = R.shape
    
    nxbar = E_YminusUV.ravel() @ R.reshape(m * n, k)
    nk = np.sum(R.reshape(m * n, k), axis=0)
    
    alpha = alpha0 + nk
    beta = beta0 + nk
    c = c0 + nk / 2
    mu = (beta0 * mu0 + nxbar) / beta
    d = d0 + 0.5 * (E_YminusUV_2.ravel() @ R.reshape(m * n, k) + beta0 * mu0**2 - 
                    (1 / beta) * (nxbar + beta0 * mu0)**2)
    
    mog_model['alpha'] = alpha
    mog_model['beta'] = beta
    mog_model['mu'] = mu
    mog_model['c'] = c
    mog_model['d'] = d
    
    return mog_model

def mog_vexp(mog_model, E_YminusUV, E_YminusUV_2):
    alpha = mog_model['alpha']
    beta = mog_model['beta']
    mu = mog_model['mu']
    c = mog_model['c']
    d = mog_model['d']
    
    m, n = E_YminusUV.shape
    k = len(mu)
    Ex = E_YminusUV.ravel()
    Ex2 = E_YminusUV_2.ravel()
    
    tau = c / d
    EQ = np.zeros((m * n, k))
    for i in range(k):
        EQ[:, i] = 1 / beta[i] + tau[i] * mu[i]**2 + tau[i] * Ex2 - 2 * tau[i] * mu[i] * Ex
    
    Elogtau = psi(c) - np.log(d)
    Elogpi = psi(alpha) - psi(np.sum(alpha))
    
    logRho = (EQ - 2 * Elogpi - Elogtau + np.log(2 * np.pi)) / (-2)
    logR = logRho - logsumexp(logRho, axis=1, keepdims=True)
    R = np.exp(logR)
    
    mog_model['logR'] = logR.reshape(m, n, k)
    mog_model['R'] = R.reshape(m, n, k)
    
    return mog_model

def R_initialization(X, k):
    n = X.shape[0]
    idx = np.random.choice(n, k, replace=False)
    m = X[idx]
    
    distances = -2 * np.outer(m, X) + (m**2)[:, np.newaxis]
    label = np.argmax(distances, axis=0)
    
    u = np.unique(label)
    while len(u) != k:
        idx = np.random.choice(n, k, replace=False)
        m = X[idx]
        distances = -2 * np.outer(m, X) + (m**2)[:, np.newaxis]
        label = np.argmax(distances, axis=0)
        u = np.unique(label)
    
    R = np.zeros((n, k))
    R[np.arange(n), label] = 1
    
    return R

def logsumexp(X, axis=None, keepdims=False):
    X_max = np.max(X, axis=axis, keepdims=True)
    result = X_max + np.log(np.sum(np.exp(X - X_max), axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result
