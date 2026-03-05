"""LRR-ALM: Low-Rank Representation via ALM.

Ports algorithms/lrr/ALM/run_alg.m.
Reference: Lin et al. 2011.
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


@register("LRR", "LRR-ALM", "LRR-ALM (Lin et al. 2011)", speed_class=2)
class LRRALM(Decomposer):
    """Low-Rank Representation via Augmented Lagrange Multiplier."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-7
        max_iter = 500

        Z = np.zeros((n, n), dtype=np.float64)
        E = np.zeros((m, n), dtype=np.float64)
        Y1 = np.zeros((m, n), dtype=np.float64)
        Y2 = np.zeros((n, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M, 2)
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        MtM = M.T @ M

        for _k in range(max_iter):
            # Update Z
            temp = MtM + np.eye(n)
            rhs = M.T @ (M - E + Y1 / mu) + (Z - Y2 / mu)
            Z_raw = np.linalg.solve(temp, rhs)
            U, s, Vt = svd(Z_raw, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                Z = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                Z = np.zeros((n, n))

            # Update E
            temp_E = M - M @ Z + Y1 / mu
            E = np.sign(temp_E) * np.maximum(np.abs(temp_E) - lam / mu, 0)

            # Update multipliers
            res1 = M - M @ Z - E
            Y1 += mu * res1
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(res1, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        L = M @ Z
        S = E
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
