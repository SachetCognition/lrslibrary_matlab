"""NSA v1 algorithm.

Ports algorithms/rpca/NSA1/run_alg.m.
Reference: Aybat et al. 2011.
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


@register("RPCA", "NSA1", "NSA v1 (Aybat et al. 2011)", speed_class=2)
class NSA1(Decomposer):
    """Noisy and Smooth Approach v1 for RPCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        tol = 5e-6
        max_iter = 500
        stdev = 1.0
        mu = stdev * np.sqrt(2.0 * np.log(m * n))

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)
        beta = 0.25 / np.mean(np.abs(M))

        for _k in range(max_iter):
            # L-step: SVT
            U, s, Vt = svd(M - S + Y / beta, full_matrices=False)
            svp = int(np.sum(s > 1.0 / beta))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / beta) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            # S-step
            temp = M - L + Y / beta
            S = np.sign(temp) * np.maximum(np.abs(temp) - mu / beta, 0)

            residual = M - L - S
            Y = Y + beta * residual

            err = np.linalg.norm(residual, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

        S_out = M - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
