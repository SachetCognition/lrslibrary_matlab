"""Pydantic models for API request/response validation."""

from typing import Literal

from pydantic import BaseModel


class AlgorithmInfo(BaseModel):
    """Algorithm metadata returned by list endpoints."""

    method_id: str
    algorithm_id: str
    name: str
    speed_class: int
    is_tensor: bool


class JobRequest(BaseModel):
    """Request body for submitting a decomposition job."""

    method_id: str
    algorithm_id: str
    video_id: str
    params: dict = {}


class JobStatus(BaseModel):
    """Status response for a decomposition job."""

    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    progress: float = 0.0
    cputime: float | None = None
    result_urls: dict | None = None
    error: str | None = None
    nframes: int | None = None


class VideoMetadata(BaseModel):
    """Metadata for an uploaded video."""

    video_id: str
    width: int
    height: int
    nframes: int
    fps: float
