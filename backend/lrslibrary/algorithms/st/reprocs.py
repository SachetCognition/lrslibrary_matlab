"""ReProCS: Online Robust PCA via Recursive Projected Compressive Sensing.

Ports algorithms/st/ReProCS/run_alg.m.
Reference: Guo et al. 2014.
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


@register("ST", "ReProCS", "ReProCS (Guo et al. 2014)", speed_class=3)
class ReProCS(Decomposer):
    """Online Robust PCA via Recursive Projected Compressive Sensing."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))

        n_init = min(30, n)
        U, s, Vt = svd(M[:, :n_init], full_matrices=False)
        P = U[:, :rank]

        L = np.zeros_like(M)
        S_out = np.zeros_like(M)

        for j in range(n):
            y = M[:, j]

            # Project perpendicular to current subspace
            Phi = np.eye(m) - P @ P.T
            y_perp = Phi @ y

            # Sparse recovery via thresholding
            thresh = 2.0 * np.std(y_perp, ddof=1) if np.std(y_perp, ddof=1) > 0 else 0.1
            support = np.abs(y_perp) > thresh

            # Estimate sparse on support
            s_hat = np.zeros(m)
            if np.any(support):
                s_hat[support] = y_perp[support]

            # Estimate low-rank
            l_hat = y - s_hat

            L[:, j] = l_hat
            S_out[:, j] = s_hat

            # Periodically update subspace
            if (j + 1) % max(1, n // 5) == 0 and j >= n_init:
                U, s_vals, Vt = svd(L[:, max(0, j - n_init):j + 1], full_matrices=False)
                P = U[:, :rank]

        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
