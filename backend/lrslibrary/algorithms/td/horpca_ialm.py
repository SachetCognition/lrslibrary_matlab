"""HoRPCA-IALM: Higher-order RPCA via IALM.

Ports algorithms/td/HoRPCA-IALM/run_alg.m.
Reference: Goldfarb and Qin, 2014.
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


@register(
    "TD", "HoRPCA-IALM", "HoRPCA-IALM (Goldfarb and Qin, 2014)",
    speed_class=3, is_tensor=True,
)
class HoRPCAIALM(Decomposer):
    """Higher-order RPCA via Inexact ALM."""

    is_tensor = True

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        if M.ndim > 2:
            shape = M.shape
            M_2d = M.reshape(shape[0], -1)
        else:
            shape = M.shape
            M_2d = M

        m, n = M_2d.shape
        lam = 1.0 / np.sqrt(max(m, n))
        tol = 1e-7
        max_iter = 500

        Y = M_2d.copy()
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lam
        dual_norm = max(norm_two, norm_inf)
        Y = Y / dual_norm

        A = np.zeros((m, n), dtype=np.float64)
        E = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(M_2d, "fro")

        for _k in range(max_iter):
            temp_T = M_2d - A + (1.0 / mu) * Y
            E = np.maximum(temp_T - lam / mu, 0) + np.minimum(temp_T + lam / mu, 0)

            U, s, Vt = svd(M_2d - E + (1.0 / mu) * Y, full_matrices=False)
            svp = int(np.sum(s > 1.0 / mu))
            if svp > 0:
                A = U[:, :svp] @ np.diag(s[:svp] - 1.0 / mu) @ Vt[:svp, :]
            else:
                A = np.zeros_like(M_2d)

            Z = M_2d - A - E
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(Z, "fro") / d_norm if d_norm > 0 else 0.0
            if err < tol:
                break

        L = A.reshape(shape) if data.ndim > 2 else A
        S_comp = E.reshape(shape) if data.ndim > 2 else E
        O = hard_threshold(S_comp)
        return DecompositionResult(L=L, S=S_comp, O=O)
