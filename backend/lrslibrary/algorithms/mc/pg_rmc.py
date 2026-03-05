"""PG-RMC algorithm for Matrix Completion.

Ports algorithms/mc/PG-RMC/run_alg.m.
Reference: Cambier and Absil, 2016.
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


@register("MC", "PG-RMC", "PG-RMC (Cambier and Absil, 2016)", speed_class=2)
class PGRMC(Decomposer):
    """Projected Gradient for Robust Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        tol = 1e-5
        max_iter = 300

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        U, s, Vt = svd(M * Omega, full_matrices=False)
        k = min(rank, len(s))
        X = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        step_size = 0.5

        for _it in range(max_iter):
            X_prev = X.copy()
            grad = (X - M) * Omega
            X = X - step_size * grad

            U, s, Vt = svd(X, full_matrices=False)
            X = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
