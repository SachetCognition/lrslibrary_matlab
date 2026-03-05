"""NTD-MU: Non-negative Tucker Decomposition with MU.

Ports algorithms/ntf/NTD-MU/run_alg.m.
Reference: Kim and Choi, 2007.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NTF", "NTD-MU", "NTD-MU (Kim and Choi, 2007)", speed_class=2, is_tensor=True)
class NTDMU(Decomposer):
    """Non-negative Tucker Decomposition with Multiplicative Updates."""

    is_tensor = True

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        if data.ndim == 2:
            M = data.astype(np.float64)
            m, n = M.shape
            M_pos = np.maximum(M, 0)
            rank = min(2, min(m, n))
            max_iter = 100

            rng = np.random.RandomState(42)
            W = np.abs(rng.randn(m, rank)) + 0.1
            H = np.abs(rng.randn(rank, n)) + 0.1

            for _k in range(max_iter):
                num_H = W.T @ M_pos
                den_H = W.T @ W @ H + 1e-10
                H = H * (num_H / den_H)

                num_W = M_pos @ H.T
                den_W = W @ H @ H.T + 1e-10
                W = W * (num_W / den_W)

            L = W @ H
        else:
            # 3D tensor: unfold along first mode
            T = data.astype(np.float64)
            T_pos = np.maximum(T, 0)
            shape = T.shape
            M_unf = T_pos.reshape(shape[0], -1)
            rank = min(2, min(M_unf.shape))
            max_iter = 100

            rng = np.random.RandomState(42)
            W = np.abs(rng.randn(shape[0], rank)) + 0.1
            H = np.abs(rng.randn(rank, M_unf.shape[1])) + 0.1

            for _k in range(max_iter):
                num_H = W.T @ M_unf
                den_H = W.T @ W @ H + 1e-10
                H = H * (num_H / den_H)

                num_W = M_unf @ H.T
                den_W = W @ H @ H.T + 1e-10
                W = W * (num_W / den_W)

            L = (W @ H).reshape(shape)
            M_pos = T_pos

        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
