"""LRR-IALM: Low-Rank Representation via Inexact ALM.

Ports algorithms/lrr/IALM/run_alg.m.
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


@register("LRR", "LRR-IALM", "LRR-IALM (Lin et al. 2011)", speed_class=2)
class LRRIALM(Decomposer):
    """Low-Rank Representation via Inexact ALM."""

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
        Y = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M, 2)
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")
        MtM = M.T @ M

        for _k in range(max_iter):
            rhs = M.T @ (M - E + Y / mu)
            Z_raw = np.linalg.solve(MtM + np.eye(n) / mu, rhs)
            U, s, Vt = svd(Z_raw, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                Z = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                Z = np.zeros((n, n))

            temp_E = M - M @ Z + Y / mu
            E = np.sign(temp_E) * np.maximum(np.abs(temp_E) - lam / mu, 0)

            residual = M - M @ Z - E
            Y += mu * residual
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(residual, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        L = M @ Z
        S = E
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
