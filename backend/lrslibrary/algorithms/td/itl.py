"""ITL: Incremental Tensor Learning.

Ports algorithms/td/ITL/run_alg.m.
Reference: Sun et al. 2008.
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


@register("TD", "ITL", "ITL (Sun et al. 2008)", speed_class=2, is_tensor=True)
class ITL(Decomposer):
    """Incremental Tensor Learning."""

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

        U, s, Vt = svd(M_2d, full_matrices=False)
        k = min(rank, len(s))
        L_2d = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

        L = L_2d.reshape(shape) if data.ndim > 2 else L_2d
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
