"""MEDRoP: Modified Efficient Dense Row Processing.

Ports algorithms/st/MEDRoP/run_alg.m.
Reference: Vaswani and Narayanamurthy, 2018.
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


@register("ST", "MEDRoP", "MEDRoP (Vaswani and Narayanamurthy, 2018)", speed_class=3)
class MEDRoP(Decomposer):
    """Modified Efficient Dense Row Processing for streaming RPCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        alpha = 50.0

        n_init = min(30, n)
        U, s, Vt = svd(M[:, :n_init], full_matrices=False)
        P = U[:, :rank]

        L = np.zeros_like(M)
        S_out = np.zeros_like(M)

        for j in range(n):
            y = M[:, j]
            x = P.T @ y
            l_hat = P @ x
            s_hat = y - l_hat

            thresh = alpha * np.median(np.abs(s_hat)) / 0.6745
            support = np.abs(s_hat) > thresh
            if np.any(support):
                Ic = ~support
                y_Ic = y.copy()
                y_Ic[support] = 0
                x_new = np.linalg.lstsq(P[Ic, :], y[Ic], rcond=None)[0] if np.sum(Ic) > rank else x
                l_hat = P @ x_new
                s_hat = y - l_hat

            L[:, j] = l_hat
            S_out[:, j] = s_hat

            if (j + 1) % max(1, n // 5) == 0 and j > n_init:
                U, s_vals, Vt = svd(L[:, :j + 1], full_matrices=False)
                P = U[:, :rank]

        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
