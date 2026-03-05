"""DRMF: Direct Robust Matrix Factorization.

Ports algorithms/nmf/DRMF/run_alg.m.
Reference: Xiong et al. 2011.
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


@register("NMF", "DRMF", "DRMF (Xiong et al. 2011)", speed_class=2)
class DRMF(Decomposer):
    """Direct Robust Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        max_iter = 100
        tol = 1e-5

        U, s, Vt = svd(M, full_matrices=False)
        k = min(rank, len(s))
        L = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        for _it in range(max_iter):
            L_prev = L.copy()
            S = M - L

            # Trim outliers
            threshold = 3.0 * np.std(S, ddof=1)
            mask = np.abs(S) > threshold
            M_trim = M.copy()
            M_trim[mask] = L[mask]

            U, s, Vt = svd(M_trim, full_matrices=False)
            L = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(L - L_prev, "fro") / (np.linalg.norm(L_prev, "fro") + 1e-16)
            if change < tol:
                break

        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
