"""GRASTA: Grassmannian Robust Adaptive Subspace Tracking.

Ports algorithms/st/GRASTA/run_alg.m.
Reference: He et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("ST", "GRASTA", "GRASTA (He et al. 2012)", speed_class=1)
class GRASTA(Decomposer):
    """Grassmannian Robust Adaptive Subspace Tracking Algorithm."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        step_size = 1.0

        # Initialize subspace
        rng = np.random.RandomState(42)
        U = np.linalg.qr(rng.randn(m, rank))[0]

        L = np.zeros_like(M)
        S_out = np.zeros_like(M)

        for j in range(n):
            v = M[:, j]
            w = U.T @ v
            p = U @ w
            residual = v - p

            # Robust weighting (L1)
            sigma = np.median(np.abs(residual)) + 1e-10
            weights = 1.0 / (np.abs(residual) / sigma + 1.0)

            # Weighted projection
            w_weighted = U.T @ (weights * v)
            l_col = U @ w_weighted
            s_col = v - l_col

            L[:, j] = l_col
            S_out[:, j] = s_col

            # Grassmann gradient update
            res_norm = np.linalg.norm(residual)
            if res_norm > 1e-10:
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-10:
                    theta = np.arctan(res_norm / w_norm)
                    alpha = step_size * theta
                    cos_term = (np.cos(alpha) - 1) * np.outer(
                        p / (w_norm + 1e-16),
                        w / (w_norm + 1e-16),
                    )
                    U = U + cos_term
                    sin_term = np.sin(alpha) * np.outer(
                        residual / (res_norm + 1e-16),
                        w / (w_norm + 1e-16),
                    )
                    U = U + sin_term

        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
