"""NMF-ALS-OBS: NMF-ALS with Optimal Brain Surgeon.

Ports algorithms/nmf/NMF-ALS-OBS/run_alg.m.
Reference: Bucak and Gunsel, 2009.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "NMF-ALS-OBS", "NMF-ALS-OBS (Bucak and Gunsel, 2009)", speed_class=2)
class NMFALSOBS(Decomposer):
    """NMF-ALS with Optimal Brain Surgeon pruning."""

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
            H = np.linalg.solve(W.T @ W + 1e-8 * np.eye(rank), W.T @ M_pos)
            H = np.maximum(H, 0)

            W = M_pos @ H.T @ np.linalg.inv(H @ H.T + 1e-8 * np.eye(rank))
            W = np.maximum(W, 0)

            # OBS pruning: zero out small elements
            threshold = 0.01 * np.max(W)
            W[W < threshold] = 0

            res = np.linalg.norm(M_pos - W @ H, "fro") / (np.linalg.norm(M_pos, "fro") + 1e-16)
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
