"""SVP algorithm for Matrix Completion.

Ports algorithms/mc/SVP/run_alg.m.
Reference: Jain et al. 2010.
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


@register("MC", "SVP", "SVP (Jain et al. 2010)", speed_class=1)
class SVP(Decomposer):
    """Singular Value Projection for Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        tol = 1e-5
        max_iter = 300
        step_size = 1.2

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        X = np.zeros((m, n), dtype=np.float64)

        for _it in range(max_iter):
            X_prev = X.copy()
            # Gradient step on observed entries
            X = X - step_size * (X - M) * Omega

            # Project to rank-k
            U, s, Vt = svd(X, full_matrices=False)
            k = min(rank, len(s))
            X = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
