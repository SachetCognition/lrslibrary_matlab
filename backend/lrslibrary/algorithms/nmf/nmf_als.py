"""NMF-ALS: NMF with Alternating Least Squares.

Ports algorithms/nmf/NMF-ALS/run_alg.m.
Reference: Paatero and Tapper, 1994.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "NMF-ALS", "NMF-ALS (Paatero and Tapper, 1994)", speed_class=1)
class NMFALS(Decomposer):
    """Non-negative Matrix Factorization with Alternating Least Squares."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        M_pos = np.maximum(M, 0)
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1
        H = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            # Update H
            H = np.linalg.solve(W.T @ W + 1e-8 * np.eye(rank), W.T @ M_pos)
            H = np.maximum(H, 0)

            # Update W
            W = (M_pos @ H.T @ np.linalg.inv(H @ H.T + 1e-8 * np.eye(rank)))
            W = np.maximum(W, 0)

            res = np.linalg.norm(M_pos - W @ H, "fro") / (np.linalg.norm(M_pos, "fro") + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
