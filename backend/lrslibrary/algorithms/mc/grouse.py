"""GROUSE algorithm for Matrix Completion.

Ports algorithms/mc/GROUSE/run_alg.m.
Reference: Balzano et al. 2010.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("MC", "GROUSE", "GROUSE (Balzano et al. 2010)", speed_class=1)
class GROUSE(Decomposer):
    """Grassmannian Rank-One Update Subspace Estimation."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rank = min(2, min(m, n))
        step_size = 1.0
        max_cycles = 5

        # Initialize subspace
        rng = np.random.RandomState(42)
        U = np.linalg.qr(rng.randn(m, rank))[0]

        for _cycle in range(max_cycles):
            for j in range(n):
                v = M[:, j]
                # Project onto subspace
                w = U.T @ v
                p = U @ w
                residual = v - p

                sigma = np.linalg.norm(residual)
                if sigma < 1e-12:
                    continue

                p_norm = np.linalg.norm(w)
                if p_norm < 1e-12:
                    continue

                theta = np.arctan(sigma / p_norm)
                alpha = step_size * theta

                # Rank-one update
                w_norm = np.linalg.norm(w)
                cos_term = (np.cos(alpha) - 1) * np.outer(
                    p / (p_norm + 1e-16), w / (w_norm + 1e-16)
                )
                U = U + cos_term
                sin_term = np.sin(alpha) * np.outer(
                    residual / (sigma + 1e-16), w / (w_norm + 1e-16)
                )
                U = U + sin_term

        # Reconstruct
        coeffs = U.T @ M
        L = U @ coeffs
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
