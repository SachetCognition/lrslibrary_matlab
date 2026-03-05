"""CP-APR: CP with Alternating Poisson Regression.

Ports algorithms/td/CP-APR/run_alg.m.
Reference: Chi and Kolda, 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("TD", "CP-APR", "CP-APR (Chi and Kolda, 2012)", speed_class=3, is_tensor=True)
class CPAPR(Decomposer):
    """CP decomposition via Alternating Poisson Regression."""

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
        M_pos = np.maximum(M_2d, 1e-10)
        rank = min(2, min(m, n))
        max_iter = 200

        rng = np.random.RandomState(42)
        A = np.abs(rng.randn(m, rank)) + 0.1
        B = np.abs(rng.randn(n, rank)) + 0.1

        for _k in range(max_iter):
            AB = A @ B.T + 1e-10
            # KL-divergence update for A
            A = A * ((M_pos / AB) @ B) / (np.sum(B, axis=0, keepdims=True) + 1e-10)
            AB = A @ B.T + 1e-10
            # KL-divergence update for B
            B = B * ((M_pos / AB).T @ A) / (np.sum(A, axis=0, keepdims=True) + 1e-10)

        L_2d = A @ B.T
        L = L_2d.reshape(shape) if data.ndim > 2 else L_2d
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
