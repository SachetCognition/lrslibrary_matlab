"""Dual RPCA algorithm.

Ports algorithms/rpca/DUAL/run_alg.m.
Reference: Lin et al. 2009.
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


@register("RPCA", "DUAL", "Dual RPCA (Lin et al. 2009)", speed_class=2)
class Dual(Decomposer):
    """Dual approach to Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))

        tol = 1e-7
        max_iter = 1000
        mu = 1e-2
        rho = 1.5

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            # L-step: SVT
            U, s, Vt = svd(M - S + Y / mu, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            # S-step: soft thresholding
            temp = M - L + Y / mu
            S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

            # Dual update
            residual = M - L - S
            Y = Y + mu * residual
            mu = min(mu * rho, 1e10)

            err = np.linalg.norm(residual, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
