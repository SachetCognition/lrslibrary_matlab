"""ENMF: Efficient NMF.

Ports algorithms/nmf/ENMF/run_alg.m.
Reference: Dong et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "ENMF", "ENMF (Dong et al. 2012)", speed_class=1)
class ENMF(Decomposer):
    """Efficient Non-negative Matrix Factorization."""

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

        WtW = W.T @ W
        WtM = W.T @ M_pos

        for _k in range(max_iter):
            # Update H
            H = H * (WtM / (WtW @ H + 1e-10))

            # Update W
            MHt = M_pos @ H.T
            HHt = H @ H.T
            W = W * (MHt / (W @ HHt + 1e-10))

            WtW = W.T @ W
            WtM = W.T @ M_pos

            res = np.linalg.norm(M_pos - W @ H, "fro") / (np.linalg.norm(M_pos, "fro") + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
