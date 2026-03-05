"""MC-NMF algorithm for Matrix Completion.

Ports algorithms/mc/MC-NMF/run_alg.m.
Reference: Xu et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("MC", "MC-NMF", "MC-NMF (Xu et al. 2012)", speed_class=2)
class MCNMF(Decomposer):
    """Matrix Completion via NMF."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        M_pos = np.maximum(M, 0)

        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1
        H = np.abs(rng.randn(rank, n)) + 0.1

        for _k in range(max_iter):
            WH = W @ H
            # Update H
            num_H = W.T @ (M_pos * Omega)
            den_H = W.T @ (WH * Omega) + 1e-10
            H = H * (num_H / den_H)

            WH = W @ H
            # Update W
            num_W = (M_pos * Omega) @ H.T
            den_W = (WH * Omega) @ H.T + 1e-10
            W = W * (num_W / den_W)

            WH = W @ H
            numer = np.linalg.norm((M_pos - WH) * Omega, 'fro')
            denom = np.linalg.norm(M_pos * Omega, 'fro') + 1e-16
            res = numer / denom
            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
