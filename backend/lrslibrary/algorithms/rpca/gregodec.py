"""GreGoDec algorithm.

Ports algorithms/rpca/GreGoDec/run_alg.m.
Reference: Zhou and Tao, 2013.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("RPCA", "GreGoDec", "GreGoDec (Zhou and Tao, 2013)", speed_class=1)
class GreGoDec(Decomposer):
    """Greedy Semi-Soft GoDec."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rank = 1
        tau = 7.0
        power = 5
        tol = 1e-3
        max_iter = 100

        X = M.copy()
        transposed = False
        if m < n:
            X = X.T
            m, n = X.shape
            transposed = True

        L = X.copy()
        S = np.zeros_like(X)

        for _it in range(max_iter):
            # Update L via randomized power iteration
            Y2 = np.random.randn(n, rank)
            for _i in range(power + 1):
                Y1 = L @ Y2
                Y2 = L.T @ Y1
            Q, _ = np.linalg.qr(Y2, mode="reduced")
            L_new = (L @ Q) @ Q.T

            # Update S via soft thresholding
            T = L - L_new + S
            L = L_new
            S = np.sign(T) * np.maximum(np.abs(T) - tau, 0)

            T = T - S
            rmse = np.linalg.norm(T.ravel())
            if rmse < tol:
                break
            L = L + T

        if transposed:
            L = L.T

        S_out = data.astype(np.float64) - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
