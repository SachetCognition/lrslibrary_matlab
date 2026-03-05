"""NeNMF: Nesterov's NMF.

Ports algorithms/nmf/NeNMF/run_alg.m.
Reference: Guan et al. 2012.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("NMF", "NeNMF", "NeNMF (Guan et al. 2012)", speed_class=1)
class NeNMF(Decomposer):
    """Nesterov's optimal gradient method for NMF."""

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
        H = np.abs(rng.randn(rank, n)) + 0.1

        alpha_W = 1.0
        alpha_H = 1.0
        W_prev = W.copy()
        H_prev = H.copy()

        for _k in range(max_iter):
            # Nesterov momentum
            alpha_W_new = (1 + np.sqrt(1 + 4 * alpha_W**2)) / 2
            alpha_H_new = (1 + np.sqrt(1 + 4 * alpha_H**2)) / 2

            W_tilde = W + ((alpha_W - 1) / alpha_W_new) * (W - W_prev)
            H_tilde = H + ((alpha_H - 1) / alpha_H_new) * (H - H_prev)
            W_tilde = np.maximum(W_tilde, 0)
            H_tilde = np.maximum(H_tilde, 0)

            # Gradient step
            grad_H = W_tilde.T @ (W_tilde @ H_tilde - M_pos)
            L_H = np.linalg.norm(W_tilde.T @ W_tilde, 2) + 1e-10
            H_new = np.maximum(H_tilde - grad_H / L_H, 0)

            grad_W = (W_tilde @ H_new - M_pos) @ H_new.T
            L_W = np.linalg.norm(H_new @ H_new.T, 2) + 1e-10
            W_new = np.maximum(W_tilde - grad_W / L_W, 0)

            numer = np.linalg.norm(M_pos - W_new @ H_new, 'fro')
            res = numer / (np.linalg.norm(M_pos, 'fro') + 1e-16)

            W_prev, H_prev = W, H
            W, H = W_new, H_new
            alpha_W, alpha_H = alpha_W_new, alpha_H_new

            if res < tol:
                break

        L = W @ H
        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
