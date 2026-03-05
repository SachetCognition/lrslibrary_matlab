"""PCP (Principal Component Pursuit) algorithm.

Ports algorithms/rpca/PCP/PCP.m.
Reference: Candes et al. 2009.
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


@register("RPCA", "PCP", "PCP (Candes et al. 2009)", speed_class=1)
class PCP(Decomposer):
    """Principal Component Pursuit via ADMM."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-5
        beta = 0.25 / np.mean(np.abs(M))
        max_iter = 1000

        S = np.zeros((m, n), dtype=np.float64)
        L = np.zeros((m, n), dtype=np.float64)
        Lambda = np.zeros((m, n), dtype=np.float64)

        for _iter in range(max_iter):
            nrm_LS = np.linalg.norm(np.hstack([S, L]), "fro")

            # S-subproblem (soft thresholding)
            X = Lambda / beta + M
            Y = X - L
            dS = S.copy()
            S = np.sign(Y) * np.maximum(np.abs(Y) - lam / beta, 0)
            dS = S - dS

            # L-subproblem (singular value thresholding)
            Y = X - S
            dL = L.copy()
            U, s, Vt = svd(Y, full_matrices=False)
            ind = s > 1.0 / beta
            if np.any(ind):
                L = U[:, ind] @ np.diag(s[ind] - 1.0 / beta) @ Vt[ind, :]
            else:
                L = np.zeros_like(M)
            dL = L - dL

            # Stopping criterion
            rel_chg = np.linalg.norm(np.hstack([dS, dL]), "fro") / (1.0 + nrm_LS)
            if rel_chg < tol:
                break

            # Update dual variable
            Lambda = Lambda - beta * (S + L - M)

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
