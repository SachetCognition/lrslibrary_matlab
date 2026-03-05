"""AS-RPCA: Active Subspace RPCA.

Ports algorithms/rpca/AS-RPCA/run_alg.m.
Reference: Liu and Yan, 2012.
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


@register("RPCA", "AS-RPCA", "AS-RPCA (Liu and Yan, 2012)", speed_class=2)
class ASRPCA(Decomposer):
    """Active Subspace: Towards Scalable Low-Rank Learning."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(min(m, n))

        tol = 1e-7
        max_iter = 1000
        mu = 1.25 / np.linalg.norm(M, 2)
        mu_bar = mu * 1e7
        rho = 1.5

        Y = M.copy() / max(np.linalg.norm(M, 2), np.linalg.norm(M.ravel(), np.inf) / lam)
        A_hat = np.zeros((m, n), dtype=np.float64)
        E_hat = np.zeros((m, n), dtype=np.float64)
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            temp_T = M - A_hat + (1.0 / mu) * Y
            E_hat = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)

            U, s, Vt = svd(M - E_hat + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                A_hat = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
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
