"""Algorithm registry and dispatch system.

Ports the dispatch pattern from run_algorithm.m (lines 34-53).
"""

import time
from typing import Any

import numpy as np

from lrslibrary.algorithms.base import (
    Decomposer,
    DecompositionParams,
    DecompositionResult,
    hard_threshold,
)

# Module-level registry: (method_id, algorithm_id) -> Decomposer class
_registry: dict[tuple[str, str], type[Decomposer]] = {}


def register(
    method_id: str,
    algorithm_id: str,
    name: str,
    speed_class: int = 1,
    is_tensor: bool = False,
):
    """Decorator to register a decomposition algorithm.

    Args:
        method_id: Category (e.g., "RPCA", "MC", "NMF").
        algorithm_id: Unique algorithm identifier (e.g., "FPCP", "IALM").
        name: Human-readable algorithm name.
        speed_class: Speed rating 1 (fast) to 5 (slow).
        is_tensor: Whether the algorithm operates on tensors.

    Returns:
        Decorator that registers the class.
    """

    def decorator(cls: type[Decomposer]) -> type[Decomposer]:
        cls.method_id = method_id
        cls.algorithm_id = algorithm_id
        cls.name = name
        cls.speed_class = speed_class
        cls.is_tensor = is_tensor
        _registry[(method_id, algorithm_id)] = cls
        return cls

    return decorator


def run_algorithm(
    method_id: str,
    algorithm_id: str,
    data: np.ndarray,
    params: DecompositionParams | None = None,
) -> DecompositionResult:
    """Look up and execute a registered algorithm.

    Mirrors run_algorithm.m behavior: dispatches by method_id/algorithm_id,
    sets up default params, runs algorithm, applies hard_threshold to S.

    Args:
        method_id: Category (e.g., "RPCA").
        algorithm_id: Algorithm identifier (e.g., "FPCP").
        data: Input matrix or tensor.
        params: Optional parameters. If None, defaults are created.

    Returns:
        DecompositionResult with L, S, O components.

    Raises:
        ValueError: If method_id/algorithm_id combination is not registered.
    """
    key = (method_id, algorithm_id)
    if key not in _registry:
        raise ValueError(
            f"Unknown algorithm: method_id={method_id!r}, algorithm_id={algorithm_id!r}"
        )

    # Set up default params (mirrors run_algorithm.m lines 63-66)
    if params is None:
        params = DecompositionParams()
    if params.rows is None:
        params.rows = data.shape[0]
    if params.cols is None:
        params.cols = data.shape[1] if data.ndim >= 2 else 1

    cls = _registry[key]
    decomposer = cls()

    start = time.perf_counter()
    result = decomposer.decompose(data, params)
    elapsed = time.perf_counter() - start

    # Apply hard threshold to S if O is all zeros (algorithm didn't set it)
    if np.all(result.O == 0) and not np.all(result.S == 0):
        result.O = hard_threshold(result.S)

    result.cputime = elapsed
    return result


def list_algorithms(method_id: str | None = None) -> list[dict[str, Any]]:
    """Return metadata for registered algorithms.

    Args:
        method_id: If provided, filter by this method category.

    Returns:
        List of dicts with algorithm metadata.
    """
    results = []
    for (mid, aid), cls in _registry.items():
        if method_id is not None and mid != method_id:
            continue
        results.append(
            {
                "method_id": mid,
                "algorithm_id": aid,
                "name": cls.name,
                "speed_class": cls.speed_class,
                "is_tensor": cls.is_tensor,
            }
        )
    return results


def list_methods() -> list[str]:
    """Return unique method IDs from registered algorithms."""
    return sorted({mid for mid, _ in _registry})
