"""bcuNTD: Block Coordinate Update for NTD.

Ports algorithms/ntf/bcuNTD/run_alg.m.
Reference: Xu and Yin, 2013.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NTF", "bcuNTD", "bcuNTD (Xu and Yin, 2013)", speed_class=2, is_tensor=True)
class BcuNTD(Decomposer):
    """Block Coordinate Update for Nonnegative Tucker Decomposition."""

    is_tensor = True

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        if M.ndim > 2:
            shape = M.shape
            M_2d = M.reshape(shape[0], -1)
        else:
            shape = M.shape
            M_2d = M

        m, n = M_2d.shape
        M_pos = np.maximum(M_2d, 0)
        rank = min(2, min(m, n))
        max_iter = 200

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1
        H = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            WtW = W.T @ W
            WtM = W.T @ M_pos
            for r in range(rank):
                idx = [j for j in range(rank) if j != r]
                residual_r = WtM[r, :] - sum(WtW[r, j] * H[j, :] for j in idx) if idx else WtM[r, :]
                H[r, :] = np.maximum(residual_r / (WtW[r, r] + 1e-10), 0)

            HHt = H @ H.T
            MHt = M_pos @ H.T
            for r in range(rank):
                idx = [j for j in range(rank) if j != r]
                residual_r = MHt[:, r] - sum(HHt[j, r] * W[:, j] for j in idx) if idx else MHt[:, r]
                W[:, r] = np.maximum(residual_r / (HHt[r, r] + 1e-10), 0)

        L = (W @ H).reshape(shape) if data.ndim > 2 else W @ H
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
