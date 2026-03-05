"""PRMF algorithm.

Ports algorithms/rpca/PRMF/run_alg.m.
Reference: Wang et al. 2012.
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


@register("RPCA", "PRMF", "PRMF (Wang et al. 2012)", speed_class=3)
class PRMF(Decomposer):
    """Probabilistic Robust Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        col_norms = np.linalg.norm(M, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1.0
        X = M / col_norms

        rk = 2
        tol = 1e-2
        max_iter = 100

        U, s, Vt = svd(X, full_matrices=False)
        k = min(rk, len(s))
        P = U[:, :k] @ np.diag(np.sqrt(s[:k]))
        Q = np.diag(np.sqrt(s[:k])) @ Vt[:k, :]

        for _it in range(max_iter):
            L_norm = P @ Q
            residual = X - L_norm

            # Robust weighting
            sigma = np.std(residual, ddof=1)
            weights = 1.0 / (1.0 + (residual / (sigma + 1e-10)) ** 2)

            # Update P and Q
            # P_new = (W*X)*Q' * inv(Q*(W_col)*Q' + reg)
            W_col = np.mean(weights, axis=0, keepdims=True)  # (1, n)
            P_new = (weights * X) @ Q.T @ np.linalg.inv(
                Q @ np.diag(W_col.ravel()) @ Q.T + 1e-6 * np.eye(k)
            )
            W_row = np.mean(weights, axis=1, keepdims=True)  # (m, 1)
            PtWP = P_new.T @ (W_row * P_new) + 1e-6 * np.eye(k)
            Q_new = np.linalg.inv(PtWP) @ (P_new.T @ (weights * X))

            numer = np.linalg.norm(P_new @ Q_new - P @ Q, 'fro')
            change = numer / (np.linalg.norm(P @ Q, 'fro') + 1e-16)
            P, Q = P_new, Q_new
            if change < tol:
                break

        L = (P @ Q) * col_norms
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
