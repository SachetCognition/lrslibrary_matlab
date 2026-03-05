"""TFOCS-EC algorithm.

Ports algorithms/rpca/TFOCS-EC/run_alg.m.
Reference: Becker et al. 2011.
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


@register("RPCA", "TFOCS-EC", "TFOCS-EC (Becker et al. 2011)", speed_class=3)
class TFOCSEC(Decomposer):
    """TFOCS with equality constraints for RPCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-6
        max_iter = 500

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M, 2)
        rho = 1.5
        d_norm = np.linalg.norm(M, "fro")

        for _k in range(max_iter):
            U, s, Vt = svd(M - S + Y / mu, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                L = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                L = np.zeros_like(M)

            temp = M - L + Y / mu
            S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

            residual = M - L - S
            Y = Y + mu * residual
            mu = min(mu * rho, 1e10)

            err = np.linalg.norm(residual, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        S_out = M - L
        O = hard_threshold(S_out)
        return DecompositionResult(L=L, S=S_out, O=O)
