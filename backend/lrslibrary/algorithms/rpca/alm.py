"""ALM algorithm for Robust PCA.

Ports algorithms/rpca/ALM/alm.m.
Reference: Tang and Nehorai, 2011.
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


@register("RPCA", "ALM", "ALM (Tang and Nehorai, 2011)", speed_class=3)
class ALM(Decomposer):
    """Augmented Lagrange Multiplier method for Robust PCA."""

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        D = data.astype(np.float64)
        n, p = D.shape

        kappa = 1.1
        tau = 0.61
        lam = tau * kappa
        eta = (1.0 - tau) * kappa
        mu = 30.0 / np.linalg.norm(np.sign(D), 2)
        rho = 1.1
        alpha = 1.0
        beta_param = 0.2

        Y = np.zeros_like(D)
        E = np.zeros_like(D)
        A = D.copy()

        tol_out = 1e-7
        max_iter_out = 500
        max_iter_inner2 = 20
        sv = min(10, min(n, p))

        for _iter_out in range(max_iter_out):
            Ek = E.copy()

            # Inner loop
            G = D - Ek + Y / mu
            Akk = G.copy()

            for _iter_inner2 in range(max_iter_inner2):
                U, s, Vt = svd(Akk, full_matrices=False)

                diagS = s[:sv]
                svn = int(np.sum(diagS > beta_param))
                svp = svn

                if len(diagS) > 1:
                    ratio = diagS[:-1] / (diagS[1:] + 1e-16)
                    max_idx = int(np.argmax(ratio))
                    if ratio[max_idx] > 2:
                        svp = min(svn, max_idx + 1)

                if svp < sv:
                    sv = min(svp + 1, min(n, p))
                else:
                    sv = min(svp + 10, min(n, p))

                if svp > 0:
                    Ahk = U[:, :svp] @ np.diag(diagS[:svp] - beta_param) @ Vt[:svp, :]
                else:
                    Ahk = np.zeros_like(D)

                B_mat = 2.0 * Ahk - Akk + mu * beta_param * G
                ns = np.linalg.norm(B_mat)
                if ns > 0:
                    B_mat = B_mat / (1.0 + mu * beta_param) * max(0, 1.0 - beta_param * eta / ns)
                Akk = Akk + alpha * (B_mat - Ahk)

            # E update - element-wise soft thresholding for L1 penalty
            G = D - Ahk + Y / mu
            Ep = np.sign(G) * np.maximum(np.abs(G) - lam / mu, 0)

            A = Ahk
            E = Ep

            err_out = np.linalg.norm(D - A - E, "fro") / (np.linalg.norm(D, "fro") + 1e-16)
            Y = Y + mu * (D - A - E)
            mu = rho * mu

            if err_out < tol_out:
                break

        L = A
        S = E
        O = hard_threshold(S)
        return DecompositionResult(L=L, S=S, O=O)
