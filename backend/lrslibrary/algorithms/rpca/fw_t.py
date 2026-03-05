"""FW-T: Frank-Wolfe method for SPCP.

Ports algorithms/rpca/FW-T/run_alg.m.
Reference: Mu et al. 2014.
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


@register("RPCA", "FW-T", "FW-T (Mu et al. 2014)", speed_class=3)
class FWT(Decomposer):
    """SPCP solved by Frank-Wolfe method."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        D = M / (np.linalg.norm(M, "fro") + 1e-16)
        delta = 0.01
        rho = 1.0
        lambda_1 = delta * rho * np.linalg.norm(D, "fro")
        lambda_2 = delta * np.sqrt(rho) * np.linalg.norm(D, "fro") / np.sqrt(max(m, n))

        tol = 1e-3
        max_iter = 500

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            # Gradient
            grad = L + S - D

            # L-step: rank-1 update via leading singular vector
            U, s, Vt = svd(grad, full_matrices=False)
            u1 = U[:, 0:1]
            v1 = Vt[0:1, :]
            step_size = 2.0 / (_k + 2.0)
            L = (1 - step_size) * L - step_size * lambda_1 * (u1 @ v1)

            # S-step: soft thresholding
            grad = L + S - D
            S = S - step_size * grad
            S = np.sign(S) * np.maximum(np.abs(S) - step_size * lambda_2, 0)

            err = np.linalg.norm(D - L - S, "fro") / (np.linalg.norm(D, "fro") + 1e-16)
            if err < tol:
                break

        # Scale back
        L = L * np.linalg.norm(M, "fro")
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
