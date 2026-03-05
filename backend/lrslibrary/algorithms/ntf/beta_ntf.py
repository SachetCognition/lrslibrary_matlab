"""betaNTF: Beta-divergence NTF.

Ports algorithms/ntf/betaNTF/run_alg.m.
Reference: Cichocki et al. 2009.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NTF", "betaNTF", "betaNTF (Cichocki et al. 2009)", speed_class=2, is_tensor=True)
class BetaNTF(Decomposer):
    """Beta-divergence Non-negative Tensor Factorization."""

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
        M_pos = np.maximum(M_2d, 1e-10)
        rank = min(2, min(m, n))
        max_iter = 200

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1
        H = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            WH = W @ H + 1e-10
            # KL divergence updates (beta=1)
            num_H = W.T @ (M_pos / WH)
            den_H = np.sum(W, axis=0, keepdims=True).T + 1e-10
            H = H * (num_H / den_H)

            WH = W @ H + 1e-10
            num_W = (M_pos / WH) @ H.T
            den_W = np.sum(H, axis=1, keepdims=True).T + 1e-10
            W = W * (num_W / den_W)

        L = (W @ H).reshape(shape) if data.ndim > 2 else W @ H
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
