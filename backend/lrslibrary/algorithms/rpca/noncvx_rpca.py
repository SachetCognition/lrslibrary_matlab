"""Nonconvex RPCA algorithm.

Ports algorithms/rpca/noncvxRPCA/run_alg.m.
Reference: Kang, 2015 (ICDM).
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


@register("RPCA", "noncvxRPCA", "Nonconvex RPCA (Kang, 2015)", speed_class=3)
class NoncvxRPCA(Decomposer):
    """Robust PCA via Nonconvex Rank Approximation."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lam = 1.0 / np.sqrt(max(m, n))
        mu = 0.9
        rate = 1.1
        gamma = 1e-2
        tol = 1e-6
        max_iter = 500

        S = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)
        L = M.copy()

        for _k in range(max_iter):
            D = M - S - Y / mu

            # DC step: truncated SVD with nonconvex penalty
            U, s, Vt = svd(D, full_matrices=False)
            # Nonconvex shrinkage on singular values
            s_thresh = np.maximum(s - 1.0 / (mu * (s + gamma)), 0)
            L = U @ np.diag(s_thresh) @ Vt

            # Sparse step: soft thresholding
            temp = M - L + Y / mu
            S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

            # Dual update
            Y = Y + mu * (L - M + S)
            mu = mu * rate

            # Convergence
            err = np.linalg.norm(M - S - L, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
