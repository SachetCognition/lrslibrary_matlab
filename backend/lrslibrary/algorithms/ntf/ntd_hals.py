"""NTD-HALS: Non-negative Tucker Decomposition with HALS.

Ports algorithms/ntf/NTD-HALS/run_alg.m.
Reference: Cichocki and Phan, 2009.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NTF", "NTD-HALS", "NTD-HALS (Cichocki and Phan, 2009)", speed_class=2, is_tensor=True)
class NTDHALS(Decomposer):
    """Non-negative Tucker Decomposition with Hierarchical ALS."""

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
            # HALS update for each component
            for r in range(rank):
                # Update H[r, :]
                residual = M_pos - W @ H + np.outer(W[:, r], H[r, :])
                H[r, :] = np.maximum(W[:, r] @ residual / (np.dot(W[:, r], W[:, r]) + 1e-10), 0)

                # Update W[:, r]
                residual = M_pos - W @ H + np.outer(W[:, r], H[r, :])
                W[:, r] = np.maximum(residual @ H[r, :] / (np.dot(H[r, :], H[r, :]) + 1e-10), 0)

        L = (W @ H).reshape(shape) if data.ndim > 2 else W @ H
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
