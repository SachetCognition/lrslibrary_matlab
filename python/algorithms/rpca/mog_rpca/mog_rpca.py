import numpy as np
from scipy.special import digamma
from scipy.linalg import svd as scipy_svd


def logsumexp(x, axis=None, keepdims=False):
    y = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - y
    result = y + np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
    
    if not keepdims:
        result = result.squeeze(axis=axis)
        y_squeezed = y.squeeze(axis=axis)
    else:
        y_squeezed = y
    
    non_finite = ~np.isfinite(y_squeezed)
    if np.any(non_finite):
        result[non_finite] = y_squeezed[non_finite]
    
    return result


def r_initialization(X, k):
    n = X.shape[1]
    idx = np.random.choice(n, k, replace=False)
    m = X[:, idx]
    
    distances = m.T @ X - 0.5 * np.sum(m * m, axis=0, keepdims=True).T
    label = np.argmax(distances, axis=0)
    
    u, _, label_inv = np.unique(label, return_index=True, return_inverse=True)
    
    while k != len(u):
        idx = np.random.choice(n, k, replace=False)
        m = X[:, idx]
        distances = m.T @ X - 0.5 * np.sum(m * m, axis=0, keepdims=True).T
        label = np.argmax(distances, axis=0)
        u, _, label_inv = np.unique(label, return_index=True, return_inverse=True)
    
    R = np.zeros((n, k))
    R[np.arange(n), label_inv] = 1
    
    return R


def mog_vmax(mog_model, mog_prior, E_YminusUV, E_YminusUV_2):
    alpha0 = mog_prior['alpha0']
    beta0 = mog_prior['beta0']
    mu0 = mog_prior['mu0']
    c0 = mog_prior['c0']
    d0 = mog_prior['d0']
    R = mog_model['R']
    
    m, n, k = R.shape
    
    E_flat = E_YminusUV.reshape(1, m * n)
    R_flat = R.reshape(m * n, k)
    
    nxbar = (E_flat @ R_flat).flatten()
    nk = np.sum(R_flat, axis=0)
    
    alpha = alpha0 + nk
    beta = beta0 + nk
    c = c0 + nk / 2
    mu = (beta0 * mu0 + nxbar) / beta
    
    E2_flat = E_YminusUV_2.reshape(1, m * n)
    d = d0 + 0.5 * ((E2_flat @ R_flat).flatten() + beta0 * mu0**2 - (nxbar + beta0 * mu0)**2 / beta)
    
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
    
    Ex = E_YminusUV.reshape(m * n, 1)
    Ex2 = E_YminusUV_2.reshape(m * n, 1)
    
    tau = c / d
    EQ = np.zeros((m * n, k))
    
    for i in range(k):
        EQ[:, i] = 1 / beta[i] + tau[i] * mu[i]**2 + tau[i] * Ex2.flatten() - 2 * tau[i] * mu[i] * Ex.flatten()
    
    Elogtau = digamma(c) - np.log(d)
    Elogpi = digamma(alpha) - digamma(np.sum(alpha))
    
    logRho = (EQ - 2 * Elogpi - Elogtau + np.log(2 * np.pi)) / (-2)
    logR = logRho - logsumexp(logRho, axis=1, keepdims=True)
    R = np.exp(logR)
    
    mog_model['logR'] = logR.reshape(m, n, k)
    mog_model['R'] = R.reshape(m, n, k)
    
    return mog_model


def lr_update(Y, lr_model, mog_model, r, lr_prior):
    m, n = Y.shape
    
    a0 = lr_prior['a0']
    b0 = lr_prior['b0']
    
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
    R_flat = R.reshape(m * n, k)
    Rtau = (R_flat @ tau.T).reshape(m, n)
    Rtaumu = (R_flat @ (tau * mu).T).reshape(m, n)
    RtauYmu = Rtau * Y - Rtaumu
    
    Gam = np.diag(gammas)
    
    re_Sigma_V = Sigma_V.reshape(r * r, n)
    diagsU = np.zeros(r)
    temp_U = np.zeros((r, r, m))
    
    for i in range(m):
        Sigma_U_inv = (re_Sigma_V @ Rtau[i, :]).reshape(r, r) + (V.T * Rtau[i, :]) @ V + Gam
        Sigma_U[:, :, i] = np.linalg.inv(Sigma_U_inv)
        U[i, :] = (RtauYmu[i, :] @ V) @ Sigma_U[:, :, i]
        diagsU += np.diag(Sigma_U[:, :, i])
        temp_U[:, :, i] = Sigma_U[:, :, i] + np.outer(U[i, :], U[i, :])
    
    re_Sigma_U = Sigma_U.reshape(r * r, m)
    diagsV = np.zeros(r)
    temp_V = np.zeros((r, r, n))
    
    for j in range(n):
        Sigma_V_inv = (re_Sigma_U @ Rtau[:, j]).reshape(r, r) + (U.T * Rtau[:, j]) @ U + Gam
        Sigma_V[:, :, j] = np.linalg.inv(Sigma_V_inv)
        V[j, :] = (RtauYmu[:, j].T @ U) @ Sigma_V[:, :, j]
        diagsV += np.diag(Sigma_V[:, :, j])
        temp_V[:, :, j] = Sigma_V[:, :, j] + np.outer(V[j, :], V[j, :])
    
    gammas = (2 * a0 + m + n) / (2 * b0 + np.sum(U * U, axis=0) + diagsU + np.sum(V * V, axis=0) + diagsV)
    
    dim_thr = 1e2
    max_gamma = np.min(gammas) * dim_thr
    
    if np.sum(gammas > max_gamma) > 0:
        indices = gammas <= max_gamma
        U = U[:, indices]
        V = V[:, indices]
        gammas = gammas[indices]
        Sigma_U = Sigma_U[np.ix_(indices, indices, range(m))]
        Sigma_V = Sigma_V[np.ix_(indices, indices, range(n))]
        temp_U = temp_U[np.ix_(indices, indices, range(m))]
        temp_V = temp_V[np.ix_(indices, indices, range(n))]
        r = U.shape[1]
    
    lr_model['U'] = U
    lr_model['V'] = V
    lr_model['Sigma_U'] = Sigma_U
    lr_model['Sigma_V'] = Sigma_V
    lr_model['gammas'] = gammas
    
    E_YminusUV = Y - U @ V.T
    
    temp_U_reshaped = temp_U.reshape(r * r, m).T
    temp_V_reshaped = temp_V.reshape(r * r, n)
    E_YminusUV_2 = Y**2 - 2 * Y * (U @ V.T) + (temp_U_reshaped @ temp_V_reshaped).reshape(m, n)
    
    return lr_model, r, E_YminusUV, E_YminusUV_2


def mog_rpca(Y, param=None, lr_prior=None, mog_prior=None):
    m, n = Y.shape
    mn = m * n
    
    if param is None:
        param = {}
    
    maxiter = param.get('maxiter', 100)
    tol = param.get('tol', 1e-4)
    mog_k = param.get('mog_k', 3)
    lr_init = param.get('lr_init', 'SVD')
    initial_rank = param.get('initial_rank', min(m, n))
    
    k = mog_k
    
    if lr_prior is None:
        lr_prior = {
            'a0': 1e-6,
            'b0': 1e-6
        }
    
    if mog_prior is None:
        mog_prior = {
            'mu0': 0,
            'c0': 1e-6,
            'd0': 1e-6,
            'alpha0': 1e-6,
            'beta0': 1e-6
        }
    
    Y2sum = np.sum(Y**2)
    scale2 = Y2sum / mn
    scale = np.sqrt(scale2)
    
    if lr_init == 'SVD':
        u, s, vt = scipy_svd(Y, full_matrices=False)
        r = min(initial_rank, len(s))
        U = u[:, :r] @ np.diag(np.sqrt(s[:r]))
        V = (np.diag(np.sqrt(s[:r])) @ vt[:r, :]).T
        Sigma_U = np.tile(scale * np.eye(r), (1, 1, m)).reshape(r, r, m)
        Sigma_V = np.tile(scale * np.eye(r), (1, 1, n)).reshape(r, r, n)
        gammas = scale * np.ones(r)
    elif lr_init == 'rand':
        r = initial_rank
        U = np.random.randn(m, r) * np.sqrt(scale)
        V = np.random.randn(n, r) * np.sqrt(scale)
        Sigma_U = np.tile(scale * np.eye(r), (1, 1, m)).reshape(r, r, m)
        Sigma_V = np.tile(scale * np.eye(r), (1, 1, n)).reshape(r, r, n)
        gammas = scale * np.ones(r)
    else:
        raise ValueError(f"Unknown lr_init method: {lr_init}")
    
    lr_model = {
        'U': U,
        'V': V,
        'Sigma_U': Sigma_U,
        'Sigma_V': Sigma_V,
        'gammas': gammas
    }
    
    L = U @ V.T
    
    E = Y - L
    E_flat = E.flatten().reshape(1, -1)
    R = r_initialization(E_flat, k)
    nk = np.sum(R, axis=0)
    nxbar = (E_flat @ R).flatten()
    
    mog_model = {
        'c': mog_prior['c0'] + nk / 2,
        'beta': mog_prior['beta0'] + nk,
        'mu': (mog_prior['beta0'] * mog_prior['mu0'] + nxbar) / (mog_prior['beta0'] + nk)
    }
    
    E_flat_sq = E_flat**2
    mog_model['d'] = (mog_prior['d0'] + 0.5 * 
                      ((E_flat_sq @ R).flatten() + mog_prior['beta0'] * mog_prior['mu0']**2 - 
                       (nxbar + mog_prior['beta0'] * mog_prior['mu0'])**2 / mog_model['beta']))
    
    mog_model['R'] = R.reshape(m, n, k)
    
    for iter_num in range(maxiter):
        L_old = L.copy()
        
        lr_model, r, E_YminusUV, E_YminusUV_2 = lr_update(Y, lr_model, mog_model, r, lr_prior)
        L = lr_model['U'] @ lr_model['V'].T
        
        mog_model = mog_vmax(mog_model, mog_prior, E_YminusUV, E_YminusUV_2)
        mog_model = mog_vexp(mog_model, E_YminusUV, E_YminusUV_2)
        
        rel_change = np.linalg.norm(L - L_old, 'fro') / np.linalg.norm(L_old, 'fro')
        
        if rel_change < tol:
            break
    
    label = np.argmax(mog_model['R'], axis=2)
    mog_model['label'] = label
    
    return lr_model, mog_model, r
