"""OptSpace algorithm for Matrix Completion.

Ports algorithms/mc/OptSpace/run_alg.m.
Reference: Keshavan et al. 2010.
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


@register("MC", "OptSpace", "OptSpace (Keshavan et al. 2010)", speed_class=2)
class OptSpace(Decomposer):
    """OptSpace for Matrix Completion."""

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

        M_E = M * Omega
        U, s, Vt = svd(M_E, full_matrices=False)
        k = min(rank, len(s))
        X = U[:, :k] * np.sqrt(m)
        Y = Vt[:k, :].T * np.sqrt(n)
        S_diag = np.diag(s[:k]) / np.sqrt(m * n)

        for _it in range(max_iter):
            S_prev = S_diag.copy()
            # Gradient descent on Grassmann manifold
            XSY = X @ S_diag @ Y.T
            residual = (XSY - M) * Omega

            grad_X = residual @ Y @ S_diag.T
            grad_Y = residual.T @ X @ S_diag

            X = X - 0.01 * grad_X
            Y = Y - 0.01 * grad_Y

            # Orthogonalize
            X, _ = np.linalg.qr(X)
            Y, _ = np.linalg.qr(Y)

            # Update S
            S_diag = X.T @ M_E @ Y

            numer = np.linalg.norm(S_diag - S_prev, 'fro')
            change = numer / (np.linalg.norm(S_prev, 'fro') + 1e-16)
            if change < tol:
                break

        L = X @ S_diag @ Y.T
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
