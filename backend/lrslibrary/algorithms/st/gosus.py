"""GOSUS: Grassmannian Online Subspace Updates with Structured-sparsity.

Ports algorithms/st/GOSUS/run_alg.m.
Reference: Xu and Bhatt, 2013.
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


@register("ST", "GOSUS", "GOSUS (Xu and Bhatt, 2013)", speed_class=2)
class GOSUS(Decomposer):
    """Grassmannian Online Subspace Updates with Structured-sparsity."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        lam = 1.0 / np.sqrt(m)

        # Initialize subspace from first few columns
        n_init = min(10, n)
        U, s, Vt = svd(M[:, :n_init], full_matrices=False)
        U_sub = U[:, :rank]

        L = np.zeros_like(M)
        S_out = np.zeros_like(M)

        for j in range(n):
            col = M[:, j]
            # Project onto subspace
            coeffs = U_sub.T @ col
            l_col = U_sub @ coeffs
            s_col = col - l_col

            # Soft threshold sparse component
            s_col = np.sign(s_col) * np.maximum(np.abs(s_col) - lam, 0)
            l_col = col - s_col

            L[:, j] = l_col
            S_out[:, j] = s_col

            # Update subspace periodically
            if (j + 1) % max(1, n // 10) == 0:
                U, s_vals, Vt = svd(L[:, :j + 1], full_matrices=False)
                U_sub = U[:, :rank]

        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
