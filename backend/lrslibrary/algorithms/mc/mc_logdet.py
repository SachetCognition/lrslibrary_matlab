"""MC_logdet algorithm for Matrix Completion.

Ports algorithms/mc/MC_logdet/run_alg.m.
Reference: Kang and Bhatt, 2014.
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


@register("MC", "MC_logdet", "MC-logdet (Kang and Bhatt, 2014)", speed_class=3)
class MCLogdet(Decomposer):
    """Matrix Completion via log-det heuristic."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        tol = 1e-5
        max_iter = 200

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        X = M * Omega
        eps = 1e-3

        for _k in range(max_iter):
            X_prev = X.copy()
            U, s, Vt = svd(X, full_matrices=False)
            # Reweighted nuclear norm
            weights = 1.0 / (s + eps)
            s_new = np.maximum(s - weights, 0)
            svp = int(np.sum(s_new > 0))
            if svp > 0:
                X = U[:, :svp] @ np.diag(s_new[:svp]) @ Vt[:svp, :]
            else:
                X = np.zeros_like(M)

            # Project observed entries
            X = X * (~Omega) + M * Omega

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
