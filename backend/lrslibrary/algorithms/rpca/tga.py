"""TGA (Trimmed Grassmann Average) algorithm.

Ports algorithms/rpca/TGA/run_alg.m.
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


@register("RPCA", "TGA", "Trimmed Grassmann Average (Hauberg et al. 2014)", speed_class=2)
class TGA(Decomposer):
    """Trimmed Grassmann Average for robust background estimation."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape

        X = M.T  # n x m
        trim_pct = 50  # percentage to keep

        mu = np.mean(X, axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-16)

        max_iter = 50
        for _k in range(max_iter):
            projections = X @ mu
            signs = np.sign(projections)
            abs_proj = np.abs(projections)

            # Trim: keep only top trim_pct% closest to mean
            threshold = np.percentile(abs_proj, 100 - trim_pct)
            mask = abs_proj >= threshold

            if np.sum(mask) == 0:
                break

            mu_new = (signs[mask, None] * X[mask]).mean(axis=0)
            mu_new = mu_new / (np.linalg.norm(mu_new) + 1e-16)

            if np.linalg.norm(mu_new - mu) < 1e-6:
                break
            mu = mu_new

        L_col = mu * (np.max(M) - np.min(M)) + np.min(M)
        L = np.tile(L_col.reshape(-1, 1), (1, n))
        S = M - L

        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
