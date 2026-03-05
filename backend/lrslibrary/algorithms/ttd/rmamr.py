"""RMAMR: Robust Moving Average Matrix Recovery.

Ports algorithms/ttd/RMAMR/run_alg.m.
Reference: Sobral et al. 2016.
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


@register("TTD", "RMAMR", "RMAMR (Sobral et al. 2016)", speed_class=2)
class RMAMR(Decomposer):
    """Robust Moving Average Matrix Recovery for TTD."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-6
        max_iter = 500

        # Robust moving average (median-based)
        window = min(10, n)
        bg = np.zeros_like(M)
        for j in range(n):
            start = max(0, j - window // 2)
            end = min(n, j + window // 2 + 1)
            bg[:, j] = np.median(M[:, start:end], axis=1)

        L = bg.copy()
        S = M - L
        Y = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M, 2)
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            U, s, Vt = svd(M - S + Y / mu, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            temp_S = M - L + Y / mu
            S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - lam / mu, 0)

            residual = M - L - S
            Y += mu * residual
            mu = min(mu * rho, 1e10)

            if np.linalg.norm(residual, "fro") / (d_norm + 1e-16) < tol:
                break

        S_out = M - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
