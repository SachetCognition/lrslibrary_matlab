"""LSADM algorithm for Robust PCA.

Ports algorithms/rpca/LSADM/run_alg.m.
Reference: Goldfarb et al. 2010.
"""

import numpy as np
from scipy.linalg import svd

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("RPCA", "LSADM", "LSADM (Goldfarb et al. 2010)", speed_class=3)
class LSADM(Decomposer):
    """Linearized and Smoothed ADM for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rho_param = 1.0 / np.sqrt(m)
        mu = np.linalg.norm(M, 2) / 1.25
        sigma = 1e-6
        max_iter = 500
        tol = 1e-7
        eta_mu = 2.0 / 3.0
        eta_sigma = 2.0 / 3.0
        mu_f = 1e-6
        sigma_f = 1e-6

        X = M.copy()
        Y_var = M.copy()
        Lambda = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            # X-subproblem (SVT)
            temp = M - Y_var + Lambda / mu
            U, s, Vt = svd(temp, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                X = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                X = np.zeros_like(M)

            # Y-subproblem (soft thresholding)
            temp = M - X + Lambda / mu
            Y_var = np.sign(temp) * np.maximum(np.abs(temp) - rho_param / mu, 0)

            # Dual update
            residual = M - X - Y_var
            Lambda = Lambda + mu * residual

            # Update parameters
            mu = max(mu * eta_mu, mu_f)
            sigma = max(sigma * eta_sigma, sigma_f)

            err = np.linalg.norm(residual, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

        L = X
        S = Y_var
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
