"""Base classes and utilities for decomposition algorithms.

Ports the contract from run_algorithm.m (lines 58-97) and hard_threshold.m.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecompositionParams:
    """Parameters for decomposition algorithms.

    Mirrors the MATLAB params struct from run_algorithm.m lines 63-82.
    """

    rows: int | None = None
    cols: int | None = None
    Idx: np.ndarray | None = None
    Omega: np.ndarray | None = None


@dataclass
class DecompositionResult:
    """Result of a decomposition algorithm.

    Mirrors the MATLAB results struct from run_algorithm.m lines 92-97.
    """

    L: np.ndarray  # low-rank component
    S: np.ndarray  # sparse component
    O: np.ndarray  # hard-thresholded outliers
    cputime: float = 0.0
    metadata: dict = field(default_factory=dict)


class Decomposer(ABC):
    """Abstract base class for all decomposition algorithms.

    Subclasses must implement the decompose() method and set class-level
    attributes for method_id, algorithm_id, name, and speed_class.
    """

    method_id: str = ""
    algorithm_id: str = ""
    name: str = ""
    speed_class: int = 1
    is_tensor: bool = False

    @abstractmethod
    def decompose(
        self, data: np.ndarray, params: DecompositionParams | None = None
    ) -> DecompositionResult:
        """Run decomposition on data matrix/tensor.

        Args:
            data: Input matrix (m x n) or tensor.
            params: Optional decomposition parameters.

        Returns:
            DecompositionResult with L, S, O components.
        """
        ...


def hard_threshold(S: np.ndarray) -> np.ndarray:
    """Apply hard thresholding to sparse component.

    Ports hard_threshold.m:
        beta = 0.5 * (std(S(:)))^2
        O = double(0.5 * S.^2 > beta)

    Args:
        S: Sparse component matrix.

    Returns:
        Binary outlier matrix O.
    """
    beta = 0.5 * (np.std(S)) ** 2
    O = (0.5 * S**2 > beta).astype(np.float64)
    return O


def subsampling(data: np.ndarray, obs: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Create subsampling masks for matrix completion.

    Ports subsampling.m (lines 10-23 of utils/subsampling.m).

    Args:
        data: Input matrix/tensor.
        obs: Fraction of observed entries (0 to 1). Default 0.5.

    Returns:
        Tuple of (Idx, Omega) where Idx is array of observed linear indices
        and Omega is binary mask of same shape as data.
    """
    nae = data.size
    rp = np.random.permutation(nae)
    k = int(np.floor(obs * nae))
    Idx = rp[:k]
    Omega = np.zeros(data.shape, dtype=np.float64)
    Omega.flat[Idx] = 1.0
    return Idx, Omega
