"""OP-RPCA algorithm for Matrix Completion.

Ports algorithms/mc/OP-RPCA/run_alg.m.
Reference: Zhou and Tao, 2011.
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


@register("MC", "OP-RPCA", "OP-RPCA (Zhou and Tao, 2011)", speed_class=3)
class OPRPCA(Decomposer):
    """Outlier Pursuit for Robust PCA in MC setting."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-6
        max_iter = 500

        Y = M.copy()
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A = np.zeros((m, n), dtype=np.float64)
        E = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            temp_T = M - A + (1.0 / mu) * Y
            E = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)

            U, s, Vt = svd(M - E + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                A = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                A = np.zeros_like(M)

            Z = M - A - E
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        L = A
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
