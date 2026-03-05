"""IALM + LMSVDS algorithm for Robust PCA.

Ports algorithms/rpca/IALM_LMSVDS/run_alg.m.
Reference: Liu et al. 2012.
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


@register("RPCA", "IALM_LMSVDS", "IALM + LMSVDS (Liu et al. 2012)", speed_class=2)
class IALMLMSVDS(Decomposer):
    """Inexact ALM with LMSVDS for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        D = data.astype(np.float64)
        m, n = D.shape

        lam = 1.0 / np.sqrt(m)
        tol = 1e-7
        max_iter = 1000

        Y = D.copy()
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A_hat = np.zeros((m, n), dtype=np.float64)
        E_hat = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(D, "fro")

        for iteration in range(1, max_iter + 1):
            temp_T = D - A_hat + (1.0 / mu) * Y
            E_hat = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)

            U, s, Vt = svd(D - E_hat + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                A_hat = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                A_hat = np.zeros_like(D)

            Z = D - A_hat - E_hat
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            stop_criterion = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if stop_criterion < tol:
                break

        O = hard_threshold(E_hat)
        return DecompositionResult(L=A_hat, S=E_hat, O=O,
                                   metadata={"iterations": iteration})
