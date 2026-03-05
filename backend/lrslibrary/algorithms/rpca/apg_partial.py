"""APG Partial (Accelerated Proximal Gradient - Partial) for Robust PCA.

Ports algorithms/rpca/APG_PARTIAL/run_alg.m.
Reference: Lin et al. 2009.
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


@register("RPCA", "APG_PARTIAL", "APG Partial (Lin et al. 2009)", speed_class=1)
class APGPartial(Decomposer):
    """Partial Accelerated Proximal Gradient for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))

        tol = 1e-7
        max_iter = 1000
        mu_inv = 0.5 * np.linalg.norm(M, 2)
        rho = 6.0

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            # Singular value thresholding for L
            U, s, Vt = svd(M - S, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu_inv))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu_inv) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            # Soft thresholding for S
            temp = M - L
            S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu_inv, 0)

            # Convergence check
            err = np.linalg.norm(M - L - S, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

            mu_inv = mu_inv * rho

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
