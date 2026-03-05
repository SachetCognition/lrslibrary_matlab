"""OR1MP algorithm for Matrix Completion.

Ports algorithms/mc/OR1MP/run_alg.m.
Reference: Wang et al. 2015.
"""

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)
from lrslibrary.registry import register


@register("MC", "OR1MP", "OR1MP (Wang et al. 2015)", speed_class=2)
class OR1MP(Decomposer):
    """Orthogonal Rank-One Matrix Pursuit for MC."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        M = data.astype(np.float64)
        m, n = M.shape
        rank = min(2, min(m, n))
        max_iter = rank * 3

        if params is not None and params.Omega is not None:
            Omega = params.Omega.astype(bool)
        else:
            Omega = np.ones((m, n), dtype=bool)

        R = M * Omega
        U_list = []
        V_list = []
        sigma_list = []

        for _k in range(max_iter):
            # Power iteration to find leading singular vector of R
            rng = np.random.RandomState(42 + _k)
            v = rng.randn(n)
            for _pi in range(20):
                u = R @ v
                u_norm = np.linalg.norm(u)
                if u_norm < 1e-12:
                    break
                u = u / u_norm
                v = R.T @ u
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    break
                v = v / v_norm

            sigma = u @ R @ v
            if sigma < 1e-10:
                break

            U_list.append(u)
            V_list.append(v)
            sigma_list.append(sigma)

            R = R - sigma * np.outer(u, v)

            if len(sigma_list) >= rank:
                break

        if len(sigma_list) > 0:
            U_mat = np.column_stack(U_list)
            V_mat = np.column_stack(V_list)
            S_mat = np.diag(sigma_list)
            L = U_mat @ S_mat @ V_mat.T
        else:
            L = np.zeros_like(M)

        S = M - L
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
