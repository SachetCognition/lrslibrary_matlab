"""Fixed Point Continuation (FPC) for Matrix Completion.

Ports algorithms/mc/FPC/run_alg.m.
Reference: Ma et al. 2011.
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


@register("MC", "FPC", "Fixed Point Continuation (Ma et al. 2011)", speed_class=2)
class FPC(Decomposer):
    """Fixed Point Continuation for Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        tau = 5.0 * np.sqrt(m * n)
        delta = 1.2
        tol = 1e-4
        max_iter = 500

        # Observation mask
        if params is not None and params.Omega is not None:
            Omega = params.Omega
        else:
            Omega = np.ones((m, n), dtype=bool)

        X = np.zeros((m, n), dtype=np.float64)

        for _k in range(max_iter):
            X_prev = X.copy()

            # Gradient step
            G = X - delta * (X - M) * Omega

            # SVD shrinkage
            U, s, Vt = svd(G, full_matrices=False)
            s_thresh = np.maximum(s - delta * tau, 0)
            svp = int(np.sum(s_thresh > 0))
            if svp > 0:
                X = U[:, :svp] @ np.diag(s_thresh[:svp]) @ Vt[:svp, :]
            else:
                X = np.zeros_like(M)

            change = np.linalg.norm(X - X_prev, "fro") / (np.linalg.norm(X_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = X
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
