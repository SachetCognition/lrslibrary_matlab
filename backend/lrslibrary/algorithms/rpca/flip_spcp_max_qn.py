"""Flip-Flop SPCP-max solved by Quasi-Newton.

Ports algorithms/rpca/flip-SPCP-max-QN/run_alg.m.
Reference: Aravkin et al. 2014.
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


@register("RPCA", "flip-SPCP-max-QN", "SPCP-max-QN (Aravkin et al. 2014)", speed_class=3)
class FlipSPCPMaxQN(Decomposer):
    """Flip-Flop version of SPCP-max solved by Quasi-Newton."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lambdaL = 0.25
        lambdaS = 0.01
        tol = 1e-3
        max_iter = 500

        L = np.tile(np.median(M, axis=1, keepdims=True), (1, n))
        S = M - L

        for _k in range(max_iter):
            L_prev = L.copy()
            S_prev = S.copy()

            U, s, Vt = svd(M - S, full_matrices=False)
            s_thresh = np.maximum(s - lambdaL, 0)
            L = U @ np.diag(s_thresh) @ Vt

            temp = M - L
            S = np.sign(temp) * np.maximum(np.abs(temp) - lambdaS, 0)

            change = np.linalg.norm(L - L_prev, "fro") + np.linalg.norm(S - S_prev, "fro")
            change /= np.linalg.norm(M, "fro") + 1e-16
            if change < tol:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
