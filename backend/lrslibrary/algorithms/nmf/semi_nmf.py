"""Semi-NMF.

Ports algorithms/nmf/Semi-NMF/run_alg.m.
Reference: Ding et al. 2010.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "Semi-NMF", "Semi-NMF (Ding et al. 2010)", speed_class=2)
class SemiNMF(Decomposer):
    """Semi Non-negative Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        # F can be negative (semi-NMF), G must be non-negative
        F = rng.randn(m, rank)
        G = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            # Update G (non-negative)
            FtM = F.T @ M
            FtF = F.T @ F
            pos_num = np.maximum(FtM, 0) + np.abs(np.minimum(FtF, 0)) @ G
            pos_den = np.maximum(FtF, 0) @ G + np.abs(np.minimum(FtM, 0)) + 1e-10
            G = G * np.sqrt(pos_num / pos_den)

            # Update F (unconstrained)
            F = M @ G.T @ np.linalg.inv(G @ G.T + 1e-8 * np.eye(rank))

            res = np.linalg.norm(M - F @ G, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if res < tol:
                break

        L = F @ G
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
