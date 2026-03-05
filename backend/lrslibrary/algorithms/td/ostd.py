"""OSTD: Online Stochastic Tensor Decomposition.

Ports algorithms/td/OSTD/run_alg.m.
Reference: Sobral et al. 2015.
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


@register("TD", "OSTD", "OSTD (Sobral et al. 2015)", speed_class=2, is_tensor=True)
class OSTD(Decomposer):
    """Online Stochastic Tensor Decomposition."""

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

        # Process columns online
        n_init = min(10, n)
        U, s, Vt = svd(M_2d[:, :n_init], full_matrices=False)
        k = min(rank, len(s))
        basis = U[:, :k]

        L_2d = np.zeros_like(M_2d)
        for j in range(n):
            col = M_2d[:, j]
            coeffs = basis.T @ col
            L_2d[:, j] = basis @ coeffs

            # Online update
            if (j + 1) % max(1, n // 10) == 0:
                U, s, Vt = svd(L_2d[:, :j + 1], full_matrices=False)
                basis = U[:, :k]

        L = L_2d.reshape(shape) if data.ndim > 2 else L_2d
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
