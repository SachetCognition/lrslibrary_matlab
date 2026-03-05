"""lraNTD: Low-Rank Approximation NTD.

Ports algorithms/ntf/lraNTD/run_alg.m.
Reference: Cichocki et al. 2009.
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


@register("NTF", "lraNTD", "lraNTD (Cichocki et al. 2009)", speed_class=2, is_tensor=True)
class LraNTD(Decomposer):
    """Low-Rank Approximation based Non-negative Tucker Decomposition."""

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

        # Initial low-rank approximation via SVD
        U, s, Vt = svd(M_pos, full_matrices=False)
        k = min(rank, len(s))
        W = np.abs(U[:, :k]) + 0.1
        H = np.abs(np.diag(s[:k]) @ Vt[:k, :]) + 0.1

        max_iter = 200
        for _it in range(max_iter):
            num_H = W.T @ M_pos
            den_H = W.T @ W @ H + 1e-10
            H = H * (num_H / den_H)

            num_W = M_pos @ H.T
            den_W = W @ H @ H.T + 1e-10
            W = W * (num_W / den_W)

        L = (W @ H).reshape(shape) if data.ndim > 2 else W @ H
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
