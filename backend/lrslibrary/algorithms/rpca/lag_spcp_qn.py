"""Lagrangian SPCP solved by Quasi-Newton.

Ports algorithms/rpca/Lag-SPCP-QN/run_alg.m.
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


@register("RPCA", "Lag-SPCP-QN", "Lagrangian SPCP-QN (Aravkin et al. 2014)", speed_class=3)
class LagSPCPQN(Decomposer):
    """Lagrangian SPCP solved by Quasi-Newton method."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lambdaL = 0.25
        lambdaS = 0.01
        tol = 1e-3
        max_iter = 500

        # Initialize L0 with column median
        L = np.tile(np.median(M, axis=1, keepdims=True), (1, n))
        S = M - L

        for _k in range(max_iter):
            L_prev = L.copy()
            S_prev = S.copy()

            # L-step: SVT
            U, s, Vt = svd(M - S, full_matrices=False)
            s_thresh = np.maximum(s - lambdaL, 0)
            L = U @ np.diag(s_thresh) @ Vt

            # S-step: soft thresholding
            temp = M - L
            S = np.sign(temp) * np.maximum(np.abs(temp) - lambdaS, 0)

            # Convergence
            change = np.linalg.norm(L - L_prev, "fro") + np.linalg.norm(S - S_prev, "fro")
            change /= np.linalg.norm(M, "fro") + 1e-16
            if change < tol:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
