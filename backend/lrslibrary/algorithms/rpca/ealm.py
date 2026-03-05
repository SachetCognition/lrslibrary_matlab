"""Exact ALM (EALM) algorithm for Robust PCA.

Ports algorithms/rpca/EALM/exact_alm_rpca.m.
Reference: Lin et al. 2009.
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


@register("RPCA", "EALM", "Exact ALM (Lin et al. 2009)", speed_class=2)
class EALM(Decomposer):
    """Exact Augmented Lagrange Multiplier method for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        D = data.astype(np.float64)
        m, n = D.shape

        lam = 1.0 / np.sqrt(m)
        tol = 1e-7
        max_iter = 1000

        # Initialize
        Y = np.sign(D)
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A_hat = np.zeros((m, n), dtype=np.float64)
        E_hat = np.zeros((m, n), dtype=np.float64)
        d_norm = np.linalg.norm(D, "fro")
        tol_proj = 1e-6 * d_norm

        mu = 0.5 / norm_two
        rho = 6.0

        converged = False
        iteration = 0

        while not converged:
            iteration += 1

            # Primal problem by alternating projection
            primal_converged = False
            primal_iter = 0
            while not primal_converged and primal_iter < 100:
                primal_iter += 1
                temp_T = D - A_hat + (1.0 / mu) * Y
                temp_E = np.maximum(temp_T - lam / mu, 0) + np.minimum(
                    temp_T + lam / mu, 0
                )

                U, s, Vt = svd(D - temp_E + (1.0 / mu) * Y, full_matrices=False)

                svp = int(np.sum(s > 1.0 / mu))
                if svp > 0:
                    temp_A = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
                else:
                    temp_A = np.zeros_like(D)

                if (
                    np.linalg.norm(A_hat - temp_A, "fro") < tol_proj
                    and np.linalg.norm(E_hat - temp_E, "fro") < tol_proj
                ):
                    primal_converged = True

                A_hat = temp_A
                E_hat = temp_E

            Z = D - A_hat - E_hat
            Y = Y + mu * Z
            mu = rho * mu

            stop_criterion = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if stop_criterion < tol:
                converged = True
            if not converged and iteration >= max_iter:
                converged = True

        O = hard_threshold(E_hat)
        return DecompositionResult(
            L=A_hat, S=E_hat, O=O,
            metadata={"iterations": iteration},
        )
