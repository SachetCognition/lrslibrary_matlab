"""RPCA (De la Torre and Black, 2001).

Ports algorithms/rpca/RPCA/run_alg.m.
Reference: De la Torre and Black, 2001.
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


@register("RPCA", "RPCA", "RPCA (De la Torre and Black, 2001)", speed_class=2)
class RPCAClassic(Decomposer):
    """Classic Robust PCA via iteratively reweighted least squares."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rank = min(2, min(m, n))
        max_iter = 100
        tol = 1e-5

        # Iteratively reweighted low-rank approximation
        W = np.ones((m, n), dtype=np.float64)
        U, s, Vt = svd(M, full_matrices=False)
        L = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

        for _it in range(max_iter):
            residual = M - L
            sigma = np.std(residual, ddof=1) + 1e-10

            # Robust weights (Cauchy-like)
            W = 1.0 / (1.0 + (residual / sigma) ** 2)

            # Weighted SVD
            WM = W * M
            U, s, Vt = svd(WM, full_matrices=False)
            L_new = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

            change = np.linalg.norm(L_new - L, "fro") / (np.linalg.norm(L, "fro") + 1e-16)
            L = L_new
            if change < tol:
                break

        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
