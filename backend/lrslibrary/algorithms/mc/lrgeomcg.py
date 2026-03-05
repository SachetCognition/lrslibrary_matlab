"""LRGeomCG for Matrix Completion.

Ports algorithms/mc/LRGeomCG/run_alg.m.
Reference: Vandereycken, 2013.
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


@register("MC", "LRGeomCG", "LRGeomCG (Vandereycken, 2013)", speed_class=2)
class LRGeomCG(Decomposer):
    """Low-Rank Matrix Completion via Riemannian CG on fixed-rank manifold."""

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
        X = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        for _it in range(max_iter):
            X_prev = X.copy()
            grad = (X - M) * Omega

            U, s, Vt = svd(X - 0.5 * grad, full_matrices=False)
            X = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
