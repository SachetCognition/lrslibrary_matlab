"""3WD: Three-Way Decomposition.

Ports algorithms/ttd/3WD/run_alg.m.
Reference: Sobral et al. 2015.
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


@register("TTD", "3WD", "3WD (Sobral et al. 2015)", speed_class=2)
class ThreeWD(Decomposer):
    """Three-Way Decomposition: L + S + N (noise)."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))
        mu_n = 0.01  # noise penalty
        tol = 1e-6
        max_iter = 500

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)
        N = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M, 2)
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            # Update L via SVT
            U, s, Vt = svd(M - S - N + Y / mu, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            # Update S via soft thresholding
            temp_S = M - L - N + Y / mu
            S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - lam / mu, 0)

            # Update N (noise - shrinkage)
            N = (mu / (mu + mu_n)) * (M - L - S + Y / mu)

            residual = M - L - S - N
            Y += mu * residual
            mu = min(mu * rho, 1e10)

            err = np.linalg.norm(residual, "fro") / (d_norm + 1e-16)
            if err < tol:
                break

        S_out = M - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
