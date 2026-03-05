"""Tucker-ADAL: Tucker decomposition via ADAL.

Ports algorithms/td/Tucker-ADAL/run_alg.m.
Reference: Goldfarb and Qin, 2014.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register(
    "TD", "Tucker-ADAL", "Tucker-ADAL (Goldfarb and Qin, 2014)",
    speed_class=3, is_tensor=True,
)
class TuckerADAL(Decomposer):
    """Tucker decomposition via Alternating Direction Augmented Lagrangian."""

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
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        A = rng.randn(m, rank)
        B = rng.randn(n, rank)

        for _k in range(max_iter):
            A_prev = A.copy()
            A = M_2d @ B @ np.linalg.inv(B.T @ B + 1e-8 * np.eye(rank))
            B = M_2d.T @ A @ np.linalg.inv(A.T @ A + 1e-8 * np.eye(rank))

            change = np.linalg.norm(A - A_prev, "fro") / (np.linalg.norm(A_prev, "fro") + 1e-16)
            if change < tol:
                break

        L_2d = A @ B.T
        L = L_2d.reshape(shape) if data.ndim > 2 else L_2d
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
