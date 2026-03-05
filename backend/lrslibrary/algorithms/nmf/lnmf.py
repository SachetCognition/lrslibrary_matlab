"""LNMF: Local NMF.

Ports algorithms/nmf/LNMF/run_alg.m.
Reference: Li et al. 2001.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "LNMF", "LNMF (Li et al. 2001)", speed_class=2)
class LNMF(Decomposer):
    """Local Non-negative Matrix Factorization."""

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
            W @ H
            # Update W with locality constraint
            num_W = M_pos @ H.T
            den_W = np.ones((m, 1)) @ np.sum(H, axis=1, keepdims=True).T + 1e-10
            W = W * (num_W / (W @ (H @ H.T) + den_W))
            W = np.maximum(W, 1e-10)

            # Update H
            num_H = W.T @ M_pos
            den_H = W.T @ (W @ H) + 1e-10
            H = H * (num_H / den_H)
            H = np.maximum(H, 1e-10)

            res = np.linalg.norm(M_pos - W @ H, "fro") / (np.linalg.norm(M_pos, "fro") + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
