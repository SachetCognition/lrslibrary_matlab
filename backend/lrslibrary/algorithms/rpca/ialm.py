"""Inexact ALM (IALM) algorithm for Robust PCA.

Ports algorithms/rpca/IALM/inexact_alm_rpca.m (lines 1-99).
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


@register("RPCA", "IALM", "Inexact ALM (Lin et al. 2009)", speed_class=1)
class IALM(Decomposer):
    """Inexact Augmented Lagrange Multiplier method for Robust PCA.

    Minimizes |A|_* + lambda * |E|_1 subject to D = A + E
    using an inexact ALM approach.
    """

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        """Run IALM decomposition.

        Args:
            data: Input matrix D of shape (m, n).
            params: Optional decomposition parameters.

        Returns:
            DecompositionResult with L (A_hat, low-rank) and S (E_hat, sparse).
        """
        D = data.astype(np.float64)
        m, n = D.shape

        # Default parameters (matching inexact_alm_rpca.m lines 33-35)
        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-7
        max_iter = 1000

        # Initialize (lines 38-54)
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

        converged = False
        iteration = 0

        while not converged:
            iteration += 1

            # E_hat update (soft thresholding) - lines 59-61
            temp_T = D - A_hat + (1.0 / mu) * Y
            E_hat = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)

            # SVD and A_hat update - lines 63-74
            U, s, Vt = svd(D - E_hat + (1.0 / mu) * Y, full_matrices=False)

            # Singular value thresholding
            svp = np.sum(s > 1.0 / mu)
            if svp > 0:
                A_hat = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                A_hat = np.zeros_like(D)

            # Update Y and mu - lines 78-81
            Z = D - A_hat - E_hat
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            # Check convergence - lines 83-87
            stop_criterion = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if stop_criterion < tol:
                converged = True

            if not converged and iteration >= max_iter:
                converged = True

        O = hard_threshold(E_hat)

        return DecompositionResult(
            L=A_hat,
            S=E_hat,
            O=O,
            metadata={
                "iterations": iteration,
                "stop_criterion": stop_criterion,
                "svp": int(svp) if svp > 0 else 0,
            },
        )
