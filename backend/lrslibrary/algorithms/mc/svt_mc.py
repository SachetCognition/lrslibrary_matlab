"""SVT for Matrix Completion.

Ports algorithms/mc/SVT/run_alg.m.
Reference: Cai et al. 2010.
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


@register("MC", "SVT-MC", "SVT for MC (Cai et al. 2010)", speed_class=2)
class SVTMC(Decomposer):
    """Singular Value Thresholding for Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        tau = 5.0 * np.sqrt(m * n)
        delta = 1.2
        tol = 1e-4
        max_iter = 500

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        Y = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            U, s, Vt = svd(Y, full_matrices=False)
            s_thresh = np.maximum(s - tau, 0)
            svp = int(np.sum(s_thresh > 0))
            if svp > 0:
                X = U[:, :svp] @ np.diag(s_thresh[:svp]) @ Vt[:svp, :]
            else:
                X = np.zeros_like(M)

            Y = Y + delta * (M - X) * Omega

            numer = np.linalg.norm((M - X) * Omega, 'fro')
            res = numer / (np.linalg.norm(M * Omega, 'fro') + 1e-16)
            if res < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
