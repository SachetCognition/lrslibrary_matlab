"""GoDec algorithm.

Ports algorithms/rpca/GoDec/GoDec.m.
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


@register("RPCA", "GoDec", "GoDec (Zhou and Tao, 2011)", speed_class=1)
class GoDec(Decomposer):
    """Go Decomposition: randomized low-rank + sparse decomposition."""

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
        card = X.size
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

            # Update S by keeping largest entries
            T = L - L_new + S
            L = L_new
            idx = np.argsort(np.abs(T).ravel())[::-1]
            S = np.zeros_like(X)
            k = min(card, len(idx))
            S.flat[idx[:k]] = T.flat[idx[:k]]

            # Error / stopping
            T.flat[idx[:k]] = 0.0
            rmse = np.linalg.norm(T.ravel())
            if rmse < error_bound:
                break
            L = L + T

        if transposed:
            L = L.T
            S = S.T

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
