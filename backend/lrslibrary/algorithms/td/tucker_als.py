"""Tucker-ALS: Tucker decomposition via ALS.

Ports algorithms/td/Tucker-ALS/run_alg.m.
Reference: Kolda and Bader, 2009.
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


@register("TD", "Tucker-ALS", "Tucker-ALS (Kolda and Bader, 2009)", speed_class=2, is_tensor=True)
class TuckerALS(Decomposer):
    """Tucker decomposition via Alternating Least Squares."""

    is_tensor = True

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        if M.ndim > 2:
            shape = M.shape
            M_2d = M.reshape(shape[0], -1)
        else:
            shape = M.shape
            M_2d = M

        m, n = M_2d.shape
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        U, s, Vt = svd(M_2d, full_matrices=False)
        k = min(rank, len(s))
        U_k = U[:, :k]
        V_k = Vt[:k, :].T

        for _it in range(max_iter):
            U_prev = U_k.copy()
            # Update core and factor matrices
            core = U_k.T @ M_2d @ V_k
            U_new, _, _ = svd(M_2d @ V_k @ core.T, full_matrices=False)
            U_k = U_new[:, :k]
            V_new, _, _ = svd(M_2d.T @ U_k @ core, full_matrices=False)
            V_k = V_new[:, :k]

            change = np.linalg.norm(U_k - U_prev, "fro") / (np.linalg.norm(U_prev, "fro") + 1e-16)
            if change < tol:
                break

        core = U_k.T @ M_2d @ V_k
        L_2d = U_k @ core @ V_k.T

        L = L_2d.reshape(shape) if data.ndim > 2 else L_2d
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
