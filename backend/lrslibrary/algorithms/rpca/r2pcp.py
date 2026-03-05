"""R2PCP: Riemannian Robust PCP.

Ports algorithms/rpca/R2PCP/run_alg.m.
Reference: Hintermuller and Wu, 2014.
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


@register("RPCA", "R2PCP", "R2PCP (Hintermuller and Wu, 2014)", speed_class=3)
class R2PCP(Decomposer):
    """Riemannian Robust Principal Component Pursuit."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rank_est = 5
        max_iter = 100
        tol = 1e-4

        # Initialize with truncated SVD
        U, s, Vt = svd(M, full_matrices=False)
        k = min(rank_est, len(s))
        A = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        B = np.zeros_like(M)

        for _it in range(max_iter):
            A_prev = A.copy()

            # Update B (sparse) via thresholding
            residual = M - A
            threshold = 0.12 * np.max(np.abs(residual))
            B = np.where(np.abs(residual) > threshold, residual, 0)

            # Update A (low-rank) via truncated SVD
            U, s, Vt = svd(M - B, full_matrices=False)
            # Trim small singular values
            sv_thresh = 10.0 * s[0] if len(s) > 0 else 1.0
            k_new = int(np.sum(s > s[0] / sv_thresh)) if len(s) > 0 else 1
            k_new = max(1, min(k_new, len(s)))
            A = U[:, :k_new] @ np.diag(s[:k_new]) @ Vt[:k_new, :]

            change = np.linalg.norm(A - A_prev, "fro") / (np.linalg.norm(A_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = A
        S = B
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
