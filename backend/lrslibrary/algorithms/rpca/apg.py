"""APG (Accelerated Proximal Gradient) algorithm for Robust PCA.

Ports algorithms/rpca/APG/run_alg.m.
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


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


@register("RPCA", "APG", "APG (Lin et al. 2009)", speed_class=1)
class APG(Decomposer):
    """Accelerated Proximal Gradient for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        lam = 1.0 / np.sqrt(max(m, n))

        tol = 1e-7
        max_iter = 1000
        mu = 1e-2
        mu_bar = 1e10
        rho = 1.5

        L = np.zeros((m, n), dtype=np.float64)
        S = np.zeros((m, n), dtype=np.float64)
        L_prev = L.copy()
        S_prev = S.copy()
        t = 1.0
        t_prev = 1.0

        for _k in range(max_iter):
            # Acceleration
            YL = L + ((t_prev - 1.0) / t) * (L - L_prev)
            YS = S + ((t_prev - 1.0) / t) * (S - S_prev)

            L_prev = L.copy()
            S_prev = S.copy()

            # Gradient step
            GL = YL - 0.5 * (YL + YS - M)
            GS = YS - 0.5 * (YL + YS - M)

            # Proximal step for L (singular value thresholding)
            U, s, Vt = svd(GL, full_matrices=False)
            s_thresh = np.maximum(s - mu / 2.0, 0)
            L = U @ np.diag(s_thresh) @ Vt

            # Proximal step for S (soft thresholding)
            S = _soft_threshold(GS, lam * mu / 2.0)

            # Update step size
            t_prev = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0

            # Check convergence
            err = np.linalg.norm(M - L - S, "fro") / (np.linalg.norm(M, "fro") + 1e-16)
            if err < tol:
                break

            mu = min(mu * rho, mu_bar)

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
