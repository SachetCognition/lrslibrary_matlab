"""LMaFit algorithm for Matrix Completion.

Ports algorithms/mc/LMaFit/run_alg.m.
Reference: Wen et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("MC", "LMaFit", "LMaFit (Wen et al. 2012)", speed_class=1)
class LMaFit(Decomposer):
    """Low-rank Matrix Fitting."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        rank = min(2, min(m, n))
        tol = 1e-5
        max_iter = 200

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        rng = np.random.RandomState(42)
        X = rng.randn(m, rank)
        Y = rng.randn(rank, n)
        Z = M * Omega

        for _k in range(max_iter):
            # Fix Y, solve for X
            X = Z @ Y.T @ np.linalg.inv(Y @ Y.T + 1e-8 * np.eye(rank))
            # Fix X, solve for Y
            Y = np.linalg.inv(X.T @ X + 1e-8 * np.eye(rank)) @ X.T @ Z
            # Update Z on observed entries
            XY = X @ Y
            Z = M * Omega + XY * (~Omega)

            numer = np.linalg.norm((Z - XY) * Omega, 'fro')
            res = numer / (np.linalg.norm(M * Omega, 'fro') + 1e-16)
            if res < tol:
                break

        L = X @ Y
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
