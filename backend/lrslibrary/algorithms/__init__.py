"""Algorithm package - auto-discovers and imports all algorithm submodules."""


def discover_algorithms() -> None:
    """Import all algorithm submodules to trigger @register decorators.

    Called lazily to avoid circular imports (registry <-> algorithms).
    """
    from lrslibrary.algorithms.rpca import fpcp, ialm  # noqa: F401
