"""HoRPCA-S-NCX: Higher-order RPCA (Nonconvex).

Ports algorithms/td/HoRPCA-S-NCX/run_alg.m.
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
    "TD", "HoRPCA-S-NCX", "HoRPCA-S-NCX (Goldfarb and Qin, 2014)",
    speed_class=3, is_tensor=True,
)
class HoRPCASNCX(Decomposer):
    """Higher-order RPCA with Nonconvex penalty."""

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
        gamma = 1e-2
        tol = 1e-6
        max_iter = 500

        A = np.zeros((m, n), dtype=np.float64)
        E = np.zeros((m, n), dtype=np.float64)
        Y = np.zeros((m, n), dtype=np.float64)
        mu = 1.25 / np.linalg.norm(M_2d, 2)
        rho = 1.5
        d_norm = np.linalg.norm(M_2d, "fro")

        for _k in range(max_iter):
            temp = M_2d - A + Y / mu
            E = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

            D = M_2d - E + Y / mu
            U, s, Vt = svd(D, full_matrices=False)
            s_thresh = np.maximum(s - 1.0 / (mu * (s + gamma)), 0)
            svp = int(np.sum(s_thresh > 0))
            if svp > 0:
                A = U[:, :svp] @ np.diag(s_thresh[:svp]) @ Vt[:svp, :]
            else:
                A = np.zeros_like(M_2d)

            Z = M_2d - A - E
            Y += mu * Z
            mu = min(mu * rho, 1e10)

            if np.linalg.norm(Z, "fro") / (d_norm + 1e-16) < tol:
                break

        L = A.reshape(shape) if data.ndim > 2 else A
        S_comp = E.reshape(shape) if data.ndim > 2 else E
        O = hard_threshold(S_comp)
        return DecompositionResult(L=L, S=S_comp, O=O)
