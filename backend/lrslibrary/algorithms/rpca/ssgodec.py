"""Semi-Soft GoDec (SSGoDec) algorithm.

Ports algorithms/rpca/SSGoDec/SSGoDec.m.
Reference: Zhou and Tao, 2011.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("RPCA", "SSGoDec", "Semi-Soft GoDec (Zhou and Tao, 2011)", speed_class=1)
class SSGoDec(Decomposer):
    """Semi-Soft Go Decomposition with soft thresholding on S."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        X = data.astype(np.float64)
        m, n = X.shape
        transposed = False
        if m < n:
            X = X.T
            m, n = X.shape
            transposed = True

        rank = 1
        tau = 8.0
        power = 0
        iter_max = 100
        error_bound = 1e-3

        L = X.copy()
        S = np.zeros_like(X)

        for _it in range(iter_max):
            # Update L via randomized power iteration
            Y2 = np.random.randn(n, rank)
            for _i in range(power + 1):
                Y1 = L @ Y2
                Y2 = L.T @ Y1
            Q, _ = np.linalg.qr(Y2, mode="reduced")
            L_new = (L @ Q) @ Q.T

            # Update S via soft thresholding (wthresh equivalent)
            T = L - L_new + S
            L = L_new
            S = np.sign(T) * np.maximum(np.abs(T) - tau, 0)

            # Error / stopping
            T = T - S
            rmse = np.linalg.norm(T.ravel())
            if rmse < error_bound:
                break
            L = L + T

        if transposed:
            L = L.T
            S = S.T

        S_out = data.astype(np.float64) - L if not transposed else data.astype(np.float64) - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
