"""Algorithm listing API routes."""

from fastapi import APIRouter

from lrslibrary.models import AlgorithmInfo
from lrslibrary.registry import list_algorithms, list_methods

router = APIRouter(tags=["algorithms"])


@router.get("/algorithms", response_model=list[AlgorithmInfo])
def get_algorithms():
    """Return list of all registered algorithms."""
    return list_algorithms()


@router.get("/algorithms/{method_id}", response_model=list[AlgorithmInfo])
def get_algorithms_by_method(method_id: str):
    """Return algorithms filtered by method_id."""
    return list_algorithms(method_id=method_id)


@router.get("/methods", response_model=list[str])
def get_methods():
    """Return list of unique method IDs."""
    return list_methods()
