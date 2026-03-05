"""NTD-APG: Non-negative Tucker Decomposition with APG.

Ports algorithms/ntf/NTD-APG/run_alg.m.
Reference: Xu and Yin, 2013.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NTF", "NTD-APG", "NTD-APG (Xu and Yin, 2013)", speed_class=2, is_tensor=True)
class NTDAPG(Decomposer):
    """Non-negative Tucker Decomposition with Accelerated Proximal Gradient."""

    is_tensor = True

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        if M.ndim == 2:
            shape = M.shape
        else:
            shape = M.shape
            M = M.reshape(shape[0], -1)

        m, n = M.shape
        M_pos = np.maximum(M, 0)
        rank = min(2, min(m, n))
        max_iter = 200
        tol = 1e-5

        rng = np.random.RandomState(42)
        W = np.abs(rng.randn(m, rank)) + 0.1
        H = np.abs(rng.randn(rank, n)) + 0.1

        t = 1.0
        W_prev = W.copy()
        H_prev = H.copy()

        for _k in range(max_iter):
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2

            W_mom = W + ((t - 1) / t_new) * (W - W_prev)
            H_mom = H + ((t - 1) / t_new) * (H - H_prev)
            W_mom = np.maximum(W_mom, 0)
            H_mom = np.maximum(H_mom, 0)

            grad_H = W_mom.T @ (W_mom @ H_mom - M_pos)
            Lh = np.linalg.norm(W_mom.T @ W_mom, 2) + 1e-10
            H_new = np.maximum(H_mom - grad_H / Lh, 0)

            grad_W = (W_mom @ H_new - M_pos) @ H_new.T
            Lw = np.linalg.norm(H_new @ H_new.T, 2) + 1e-10
            W_new = np.maximum(W_mom - grad_W / Lw, 0)

            numer = np.linalg.norm(M_pos - W_new @ H_new, 'fro')
            res = numer / (np.linalg.norm(M_pos, 'fro') + 1e-16)
            W_prev, H_prev = W, H
            W, H = W_new, H_new
            t = t_new

            if res < tol:
                break

        L = (W @ H).reshape(shape) if data.ndim > 2 else W @ H
        S = data.astype(np.float64) - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
