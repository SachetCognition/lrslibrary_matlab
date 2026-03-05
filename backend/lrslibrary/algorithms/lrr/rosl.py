"""ROSL: Robust Orthonormal Subspace Learning.

Ports algorithms/lrr/ROSL/run_alg.m.
Reference: Shu and Bhatt, 2014.
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


@register("LRR", "ROSL", "ROSL (Shu and Bhatt, 2014)", speed_class=1)
class ROSL(Decomposer):
    """Robust Orthonormal Subspace Learning."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(5, min(m, n))
        lam = 1.0 / np.sqrt(max(m, n))
        max_iter = 200
        tol = 1e-5

        # Initialize via SVD
        U, s, Vt = svd(M, full_matrices=False)
        k = min(rank, len(s))
        D = U[:, :k]
        alpha = np.diag(s[:k]) @ Vt[:k, :]

        E = np.zeros((m, n), dtype=np.float64)

        for _it in range(max_iter):
            L_prev = D @ alpha

            # Update E via soft thresholding
            residual = M - D @ alpha
            E = np.sign(residual) * np.maximum(np.abs(residual) - lam, 0)

            # Update D and alpha via SVD of M - E
            U, s, Vt = svd(M - E, full_matrices=False)
            k_new = min(k, int(np.sum(s > s[0] * 0.01))) if len(s) > 0 else k
            k_new = max(1, k_new)
            D = U[:, :k_new]
            alpha = np.diag(s[:k_new]) @ Vt[:k_new, :]

            numer = np.linalg.norm(D @ alpha - L_prev, 'fro')
            change = numer / (np.linalg.norm(L_prev, 'fro') + 1e-16)
            if change < tol:
                break

        L = D @ alpha
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
