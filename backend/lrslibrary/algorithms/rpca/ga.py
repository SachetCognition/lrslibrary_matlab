"""GA (Grassmann Average) algorithm.

Ports algorithms/rpca/GA/run_alg.m.
Reference: Hauberg et al. 2014.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("RPCA", "GA", "Grassmann Average (Hauberg et al. 2014)", speed_class=2)
class GA(Decomposer):
    """Grassmann Average for background estimation."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        # Grassmann average: iteratively compute weighted average direction
        # Initialize with mean
        X = M.T  # n x m
        mu = np.mean(X, axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-16)

        max_iter = 50
        for _k in range(max_iter):
            # Project data onto current mean
            projections = X @ mu
            signs = np.sign(projections)
            # Weighted average
            mu_new = (signs[:, None] * X).mean(axis=0)
            mu_new = mu_new / (np.linalg.norm(mu_new) + 1e-16)

            if np.linalg.norm(mu_new - mu) < 1e-6:
                break
            mu = mu_new

        # Rescale to data range
        L_col = mu * (np.max(M) - np.min(M)) + np.min(M)
        L = np.tile(L_col.reshape(-1, 1), (1, n))
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
