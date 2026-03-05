"""PNMF: Projective NMF.

Ports algorithms/nmf/PNMF/run_alg.m.
Reference: Yuan and Oja, 2005.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "PNMF", "PNMF (Yuan and Oja, 2005)", speed_class=2)
class PNMF(Decomposer):
    """Projective Non-negative Matrix Factorization."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        M_pos = np.maximum(M, 0)
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1

        for _k in range(max_iter):
            W_prev = W.copy()
            WtW = W @ W.T
            # Projective update
            num = M_pos @ M_pos.T @ W
            den = WtW @ M_pos @ M_pos.T @ W + 1e-10
            W = W * np.sqrt(num / den)
            W = np.maximum(W, 1e-10)

            # Normalize columns
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            W = W / norms

            change = np.linalg.norm(W - W_prev, "fro") / (np.linalg.norm(W_prev, "fro") + 1e-16)
            if change < tol:
                break

        L = W @ W.T @ M
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
