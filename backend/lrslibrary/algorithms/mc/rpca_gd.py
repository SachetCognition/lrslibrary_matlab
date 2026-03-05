"""RPCA-GD algorithm for Matrix Completion.

Ports algorithms/mc/RPCA-GD/run_alg.m.
Reference: Yi et al. 2016.
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


@register("MC", "RPCA-GD", "RPCA-GD (Yi et al. 2016)", speed_class=2)
class RPCAGD(Decomposer):
    """RPCA via Gradient Descent for MC."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        tol = 1e-5
        max_iter = 300
        eta = 0.5

        U, s, Vt = svd(M, full_matrices=False)
        k = min(rank, len(s))
        L_factor = U[:, :k] @ np.diag(np.sqrt(s[:k]))
        R_factor = np.diag(np.sqrt(s[:k])) @ Vt[:k, :]

        for _it in range(max_iter):
            L_prev = L_factor.copy()
            LR = L_factor @ R_factor
            residual = LR - M

            grad_L = residual @ R_factor.T
            grad_R = L_factor.T @ residual

            L_factor = L_factor - eta * grad_L
            R_factor = R_factor - eta * grad_R

            numer = np.linalg.norm(L_factor - L_prev, 'fro')
            change = numer / (np.linalg.norm(L_prev, 'fro') + 1e-16)
            if change < tol:
                break

        L = L_factor @ R_factor
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
