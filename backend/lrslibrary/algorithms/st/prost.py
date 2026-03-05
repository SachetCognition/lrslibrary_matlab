"""pROST: Robust Online Subspace Tracking.

Ports algorithms/st/pROST/run_alg.m.
Reference: Hage and Kleinsteuber, 2014.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("ST", "pROST", "pROST (Hage and Kleinsteuber, 2014)", speed_class=2)
class PROST(Decomposer):
    """Robust Online Subspace Tracking."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))

        rng = np.random.RandomState(42)
        U = np.linalg.qr(rng.randn(m, rank))[0]

        L = np.zeros_like(M)
        S_out = np.zeros_like(M)

        for j in range(n):
            v = M[:, j]
            w = U.T @ v
            l_col = U @ w
            s_col = v - l_col

            # Robust reweighting
            sigma = np.median(np.abs(s_col)) / 0.6745 + 1e-10
            weights = np.exp(-0.5 * (s_col / sigma) ** 2)
            weights = weights / (np.sum(weights) + 1e-10) * m

            l_col = U @ (U.T @ (weights * v))
            s_col = v - l_col

            L[:, j] = l_col
            S_out[:, j] = s_col

            # Subspace update via rank-one modification
            residual = v - l_col
            res_norm = np.linalg.norm(residual)
            w_norm = np.linalg.norm(w)
            if res_norm > 1e-10 and w_norm > 1e-10:
                theta = np.arctan2(res_norm, w_norm)
                alpha = 0.5 * theta
                u_dir = residual / res_norm
                w_dir = w / w_norm
                U = U + (np.cos(alpha) - 1) * (U @ w_dir.reshape(-1, 1)) @ w_dir.reshape(1, -1)
                U = U + np.sin(alpha) * u_dir.reshape(-1, 1) @ w_dir.reshape(1, -1)

        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
