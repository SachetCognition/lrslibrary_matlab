"""Variational Bayesian RPCA (VBRPCA).

Ports algorithms/rpca/VBRPCA/run_alg.m.
Reference: Babacan et al. 2011.
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


@register("RPCA", "VBRPCA", "Variational Bayesian RPCA (Babacan et al. 2011)", speed_class=3)
class VBRPCA(Decomposer):
    """Variational Bayesian Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        max_iter = 200
        tol = 1e-6

        # Initialize via SVD
        U, s, Vt = svd(M, full_matrices=False)
        rank = min(min(m, n), 10)
        L = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        S = np.zeros_like(M)

        beta = 1.0 / (np.std(M, ddof=1) ** 2 + 1e-10)

        for _k in range(max_iter):
            L_prev = L.copy()

            # E-step: estimate sparse
            residual = M - L
            # Sparse estimation with adaptive threshold
            alpha_e = 1.0 / (np.abs(residual) + 1e-10)
            threshold = 1.0 / (beta * alpha_e + 1e-10)
            S = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0)

            # M-step: estimate low-rank
            U, s, Vt = svd(M - S, full_matrices=False)
            # Automatic rank determination
            alpha_l = 1.0 / (s + 1e-10)
            s_new = np.maximum(s - alpha_l / beta, 0)
            active = s_new > 0
            if np.any(active):
                L = U[:, active] @ np.diag(s_new[active]) @ Vt[active, :]
            else:
                L = np.zeros_like(M)

            # Update beta
            residual = M - L - S
            beta = (m * n) / (np.linalg.norm(residual, "fro") ** 2 + 1e-10)

            change = np.linalg.norm(L - L_prev, "fro") / (np.linalg.norm(L_prev, "fro") + 1e-16)
            if change < tol:
                break

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
