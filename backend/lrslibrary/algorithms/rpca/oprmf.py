"""Online PRMF algorithm.

Ports algorithms/rpca/OPRMF/run_alg.m.
Reference: Wang et al. 2012.
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


@register("RPCA", "OPRMF", "Online PRMF (Wang et al. 2012)", speed_class=3)
class OPRMF(Decomposer):
    """Online Probabilistic Robust Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        # Normalize
        col_norms = np.linalg.norm(M, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        X = M / col_norms

        rk = 2
        tol = 1e-2
        max_iter = 100

        U, s, Vt = svd(X, full_matrices=False)
        k = min(rk, len(s))
        L_norm = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        for _it in range(max_iter):
            residual = X - L_norm
            threshold = np.std(residual, ddof=1)
            S_norm = np.where(np.abs(residual) > threshold, residual, 0)

            U, s, Vt = svd(X - S_norm, full_matrices=False)
            L_new = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(L_new - L_norm, "fro") / (np.linalg.norm(L_norm, "fro") + 1e-16)
            L_norm = L_new
            if change < tol:
                break

        L = L_norm * col_norms
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
