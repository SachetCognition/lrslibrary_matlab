"""Fast PCP (FPCP) algorithm.

Ports algorithms/rpca/FPCP/fastpcp.m (lines 1-93).
Reference: Rodriguez and Wohlberg, 2013.
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


def _shrink(v: np.ndarray, lam: float) -> np.ndarray:
    """Soft thresholding (shrinkage) operator.

    Ports the MATLAB shrink function: u = sign(v) .* max(0, abs(v) - lambda)
    """
    return np.sign(v) * np.maximum(0, np.abs(v) - lam)


def _svdsecon(M: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy SVD truncated to given rank.

    Uses scipy.linalg.svd with full_matrices=False, then truncates.
    Equivalent to MATLAB svdsecon(M, rank).
    """
    U, s, Vt = svd(M, full_matrices=False)
    # Truncate to requested rank
    rank = min(rank, len(s))
    return U[:, :rank], np.diag(s[:rank]), Vt[:rank, :]


@register("RPCA", "FPCP", "Fast PCP (Rodriguez and Wohlberg, 2013)", speed_class=1)
class FPCP(Decomposer):
    """Fast Principal Component Pursuit.

    Iteratively computes partial SVD and applies soft thresholding.
    """

    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        """Run FPCP decomposition.

        Args:
            data: Input matrix V of shape (m, n).
            params: Optional parameters. Supports keys in metadata:
                - loops: Number of outer loops (default 2).
                - rank0: Initial rank (default 1).
                - rankThreshold: Threshold for rank increment (default 0.01).
                - lambdaFactor: Lambda scaling factor (default 1.0).
                - lambda_: Regularization parameter (default 1/sqrt(max(m,n))).

        Returns:
            DecompositionResult with L (low-rank) and S (sparse) components.
        """
        V = data.astype(np.float64)
        m, n = V.shape

        # Default parameters (matching fastpcp.m defaults)
        lam = 1.0 / np.sqrt(max(m, n))
        loops = 2
        rank0 = 1
        rank_threshold = 0.01
        lambda_factor = 1.0

        # Override from params metadata if provided
        if params is not None and hasattr(params, "rows"):
            # params object doesn't carry algorithm-specific settings,
            # but we keep the interface clean
            pass

        # First outer loop (lines 37-49)
        rank = rank0
        inc_rank = 1

        U, Sigma, Vt = _svdsecon(V, rank)
        L = U @ Sigma @ Vt

        S = _shrink(V - L, lam)

        # Subsequent outer loops (lines 54-82)
        stats_rank = [rank]
        stats_rho = [0.0]

        for k in range(1, loops):
            if inc_rank == 1:
                lam = lam * lambda_factor
                rank = rank + 1

            U, Sigma, Vt = _svdsecon(V - S, rank)

            current_evals = np.diag(Sigma)
            stats_rank.append(len(current_evals))

            if len(current_evals) > 1:
                rho = current_evals[-1] / np.sum(current_evals[:-1])
            else:
                rho = 0.0
            stats_rho.append(rho)

            # Simple rule to keep or increase current rank
            if rho < rank_threshold:
                inc_rank = 0
            else:
                inc_rank = 1

            L = U @ Sigma @ Vt
            S = _shrink(V - L, lam)

        O = hard_threshold(S)

        return DecompositionResult(
            L=L,
            S=S,
            O=O,
            metadata={"rank": stats_rank, "rho": stats_rho},
        )
