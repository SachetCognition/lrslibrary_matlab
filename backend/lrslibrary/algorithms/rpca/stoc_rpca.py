"""STOC-RPCA: Online Robust PCA via Stochastic Optimization.

Ports algorithms/rpca/STOC-RPCA/run_alg.m.
Reference: Feng et al. 2013.
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


@register("RPCA", "STOC-RPCA", "STOC-RPCA (Feng et al. 2013)", speed_class=2)
class STOCRPCA(Decomposer):
    """Online Robust PCA via Stochastic Optimization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lambda1 = 1.0 / np.sqrt(max(m, n))
        lambda2 = lambda1
        tol = 1e-7
        max_iter = 500

        Y = M.copy()
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lambda1
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A_hat = np.zeros((m, n), dtype=np.float64)
        E_hat = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            temp_T = M - A_hat + (1.0 / mu) * Y
            E_hat = np.maximum(temp_T - lambda1 / mu, 0) + np.minimum(temp_T + lambda1 / mu, 0)

            U, s, Vt = svd(M - E_hat + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > lambda2 / mu))
            if svp > 0:
                A_hat = U[:, :svp] @ np.diag(s[:svp] - lambda2 / mu) @ Vt[:svp, :]
            else:
                A_hat = np.zeros_like(M)

            Z = M - A_hat - E_hat
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        O = hard_threshold(E_hat)
        return DecompositionResult(L=A_hat, S=E_hat, O=O)
