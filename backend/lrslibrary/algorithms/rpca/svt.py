"""SVT (Singular Value Thresholding) for Robust PCA.

Ports algorithms/rpca/SVT/run_alg.m.
Reference: Cai et al. 2008.
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


@register("RPCA", "SVT", "SVT (Cai et al. 2008)", speed_class=2)
class SVT(Decomposer):
    """Singular Value Thresholding for RPCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))

        max_iter = 500
        delta = 0.9
        tau = 1e4

        Y = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            U, s, Vt = svd(Y, full_matrices=False)
            # Nuclear norm proximal (singular value thresholding)
            svp = int(np.sum(s > tau))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - tau) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            # Soft thresholding for sparse
            residual = M - L
            S = np.sign(residual) * np.maximum(np.abs(residual) - lam * tau, 0)

            # Gradient update
            Y = Y + delta * (M - L - S)

            err = np.linalg.norm(M - L - S, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < 1e-5:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
