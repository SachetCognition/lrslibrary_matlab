"""ADM (Alternating Direction Method) algorithm for Robust PCA.

Ports algorithms/rpca/ADM/ADM.m.
Reference: Yuan and Yang, 2009.
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


@register("RPCA", "ADM", "ADM (Yuan and Yang, 2009)", speed_class=2)
class ADM(Decomposer):
    """Alternating Direction Method for low-rank + sparse decomposition."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        C = data.astype(np.float64)
        m, n = C.shape

        t = 0.01
        tau = t / (1.0 - t)
        beta = 0.25 / np.mean(np.abs(C))
        tol = 1e-6
        max_iter = 100

        A = np.zeros((m, n), dtype=np.float64)
        B = np.zeros((m, n), dtype=np.float64)
        Lambda = np.zeros((m, n), dtype=np.float64)

        for _iter in range(max_iter):
            nrm_AB = np.linalg.norm(np.hstack([A, B]), "fro")

            # A-subproblem (soft thresholding)
            X = Lambda / beta + C
            Y = X - B
            dA = A.copy()
            A = np.sign(Y) * np.maximum(np.abs(Y) - tau / beta, 0)
            dA = A - dA

            # B-subproblem (singular value thresholding)
            Y = X - A
            dB = B.copy()
            U, s, Vt = svd(Y, full_matrices=False)
            ind = s > 1.0 / beta
            if np.any(ind):
                B = U[:, ind] @ np.diag(s[ind] - 1.0 / beta) @ Vt[ind, :]
            else:
                B = np.zeros_like(C)
            dB = B - dB

            # Stopping criterion
            rel_chg = np.linalg.norm(np.hstack([dA, dB]), "fro") / (1.0 + nrm_AB)
            if rel_chg < tol:
                break

            # Update dual variable
            Lambda = Lambda - beta * (A + B - C)

        # ADM returns Sparse=A, LowRank=B
        L = B
        S = A
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
