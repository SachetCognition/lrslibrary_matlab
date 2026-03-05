"""ScGrassMC algorithm for Matrix Completion.

Ports algorithms/mc/ScGrassMC/run_alg.m.
Reference: Ngo and Saad, 2012.
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


@register("MC", "ScGrassMC", "ScGrassMC (Ngo and Saad, 2012)", speed_class=2)
class ScGrassMC(Decomposer):
    """Scaled Grassmann Manifold for Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        tol = 1e-5
        max_iter = 200

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        U, s, Vt = svd(M * Omega, full_matrices=False)
        k = min(rank, len(s))
        X = U[:, :k]
        sigma = np.diag(s[:k])
        Y = Vt[:k, :].T

        for _it in range(max_iter):
            X_prev = X.copy()

            # Reconstruct
            L_est = X @ sigma @ Y.T
            residual = (L_est - M) * Omega

            # Gradient
            grad_X = residual @ Y @ sigma.T
            grad_Y = residual.T @ X @ sigma

            X = X - 0.01 * grad_X
            Y = Y - 0.01 * grad_Y

            # Re-orthogonalize
            X, Rx = np.linalg.qr(X)
            Y, Ry = np.linalg.qr(Y)
            sigma = Rx @ sigma @ Ry.T

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X @ sigma @ Y.T
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
