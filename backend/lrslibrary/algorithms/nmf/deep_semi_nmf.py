"""Deep Semi-NMF.

Ports algorithms/nmf/Deep-Semi-NMF/run_alg.m.
Reference: Trigeorgis et al. 2014.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "Deep-Semi-NMF", "Deep Semi-NMF (Trigeorgis et al. 2014)", speed_class=3)
class DeepSemiNMF(Decomposer):
    """Deep Semi-NMF."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        Z = rng.randn(m, rank)
        H = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            # Update H (non-negative)
            ZtM = Z.T @ M
            ZtZ = Z.T @ Z
            pos_num = np.maximum(ZtM, 0) + np.abs(np.minimum(ZtZ, 0)) @ H
            pos_den = np.maximum(ZtZ, 0) @ H + np.abs(np.minimum(ZtM, 0)) + 1e-10
            H = H * np.sqrt(pos_num / pos_den)

            # Update Z (unconstrained)
            Z = M @ H.T @ np.linalg.inv(H @ H.T + 1e-8 * np.eye(rank))

            res = np.linalg.norm(M - Z @ H, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if res < tol:
                break

        L = Z @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
