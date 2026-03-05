"""NMF-PG: NMF with Projected Gradient.

Ports algorithms/nmf/NMF-PG/run_alg.m.
Reference: Lin, 2007.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "NMF-PG", "NMF-PG (Lin, 2007)", speed_class=2)
class NMFPG(Decomposer):
    """Non-negative Matrix Factorization with Projected Gradient."""

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
        alpha = 1e-3

        for _k in range(max_iter):
            # Update H via projected gradient
            grad_H = W.T @ (W @ H - M_pos)
            H = np.maximum(H - alpha * grad_H, 0)

            # Update W via projected gradient
            grad_W = (W @ H - M_pos) @ H.T
            W = np.maximum(W - alpha * grad_W, 0)

            res = np.linalg.norm(M_pos - W @ H, "fro") / (np.linalg.norm(M_pos, "fro") + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
