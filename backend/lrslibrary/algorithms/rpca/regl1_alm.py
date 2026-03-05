"""RegL1-ALM algorithm.

Ports algorithms/rpca/RegL1-ALM/run_alg.m.
Reference: Zheng et al. 2012.
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


@register("RPCA", "RegL1-ALM", "RegL1-ALM (Zheng et al. 2012)", speed_class=3)
class RegL1ALM(Decomposer):
    """Low-Rank Matrix Approximation under Robust L1-Norm."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        r = 1
        lam = 1e-3
        max_iter = 100
        tol = 1e-5

        # Initialize U, V via SVD
        U_mat, s, Vt = svd(M, full_matrices=False)
        U = U_mat[:, :r] @ np.diag(np.sqrt(s[:r]))
        V = np.diag(np.sqrt(s[:r])) @ Vt[:r, :]

        for _it in range(max_iter):
            L = U @ V
            residual = M - L
            sigma = np.std(residual, ddof=1) + 1e-10

            # L1 weights
            W = 1.0 / (np.abs(residual) + sigma)

            # Update U: each column weight summed across rows
            WM = W * M
            WV = WM @ V.T  # (m, r)
            # V @ diag(w_col) @ V' where w_col = mean of W per column
            w_col = np.mean(W, axis=0)  # (n,)
            VV = V @ np.diag(w_col) @ V.T + lam * np.eye(r)
            U_new = WV @ np.linalg.inv(VV)

            # Update V
            WU = U_new.T @ WM  # (r, n)
            w_row = np.mean(W, axis=1)  # (m,)
            UU = U_new.T @ np.diag(w_row) @ U_new + lam * np.eye(r)
            V_new = np.linalg.inv(UU) @ WU

            numer = np.linalg.norm(U_new @ V_new - U @ V, 'fro')
            change = numer / (np.linalg.norm(U @ V, 'fro') + 1e-16)
            U, V = U_new, V_new
            if change < tol:
                break

        L = U @ V
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
