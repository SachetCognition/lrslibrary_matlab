"""IALM for Matrix Completion.

Ports algorithms/mc/IALM-MC/run_alg.m.
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


@register("MC", "IALM-MC", "IALM for MC (Lin et al. 2009)", speed_class=2)
class IALM_MC(Decomposer):
    """Inexact ALM for Matrix Completion."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-7
        max_iter = 500

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        P_Omega_M = M * Omega

        Y = P_Omega_M.copy()
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A = np.zeros((m, n), dtype=np.float64)
        E = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(P_Omega_M, "fro")

        for _k in range(max_iter):
            temp_T = M - A + (1.0 / mu) * Y
            E = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)
            E = E * Omega

            U, s, Vt = svd(M - E + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                A = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                A = np.zeros_like(M)

            Z = (M - A - E) * Omega
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        L = A
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
