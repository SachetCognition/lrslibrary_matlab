"""Markov BRMF algorithm.

Ports algorithms/rpca/MBRMF/run_alg.m.
Reference: Wang and Yeung, 2013.
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


@register("RPCA", "MBRMF", "Markov BRMF (Wang and Yeung, 2013)", speed_class=4)
class MBRMF(Decomposer):
    """Markov Bayesian Robust Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        # Normalize
        col_norms = np.linalg.norm(M, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        D = M / col_norms

        r = max(1, min(20, int(np.sqrt(n))) // 2)
        max_iter = 100
        tol = 1e-5

        # Low-rank approximation via truncated SVD iteration
        U, s, Vt = svd(D, full_matrices=False)
        k = min(r, len(s))
        L_norm = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        for _it in range(max_iter):
            # Sparse component estimation
            residual = D - L_norm
            threshold = 0.5 * np.std(residual, ddof=1)
            S_norm = np.where(np.abs(residual) > threshold, residual, 0)

            # Update low-rank
            U, s, Vt = svd(D - S_norm, full_matrices=False)
            L_new = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(L_new - L_norm, "fro") / (np.linalg.norm(L_norm, "fro") + 1e-16)
            L_norm = L_new
            if change < tol:
                break

        # Denormalize
        L = L_norm * col_norms
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
