"""Algorithm package - auto-discovers and imports all algorithm submodules."""

# Import all algorithm submodules to trigger @register decorators
from lrslibrary.algorithms.rpca import fpcp, ialm  # noqa: F401
