"""ManhNMF: Manhattan NMF.

Ports algorithms/nmf/ManhNMF/run_alg.m.
Reference: Guan et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "ManhNMF", "ManhNMF (Guan et al. 2012)", speed_class=2)
class ManhNMF(Decomposer):
    """Manhattan distance NMF (L1-norm based)."""

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
            WH = W @ H + 1e-10
            M_pos / WH

            # L1 weights
            weights = 1.0 / (np.abs(M_pos - WH) + 1e-10)

            # Update H
            num_H = W.T @ (weights * M_pos / WH)
            den_H = W.T @ weights + 1e-10
            H = H * (num_H / den_H)

            WH = W @ H + 1e-10
            # Update W
            num_W = (weights * M_pos / WH) @ H.T
            den_W = weights @ H.T + 1e-10
            W = W * (num_W / den_W)

            res = np.sum(np.abs(M_pos - W @ H)) / (np.sum(np.abs(M_pos)) + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
