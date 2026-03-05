"""Job submission and status API routes.

Phase 1: Synchronous execution (no Celery).
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from lrslibrary.api.routes.video import get_video_store
from lrslibrary.config import RESULTS_DIR
from lrslibrary.models import JobRequest, JobStatus
from lrslibrary.registry import run_algorithm
from lrslibrary.video.converter import video_to_2d
from lrslibrary.video.exporter import frames_to_png, matrix_results_to_frames
from lrslibrary.video.loader import load_video

router = APIRouter(tags=["jobs"])

# In-memory job store (Phase 1)
_job_store: dict[str, dict] = {}


@router.post("/jobs", response_model=JobStatus)
def create_job(request: JobRequest):
    """Submit a decomposition job.

    Phase 1: runs synchronously and returns results immediately.
    """
    video_store = get_video_store()
    if request.video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")

    video_meta = video_store[request.video_id]
    job_id = str(uuid.uuid4())

    # Update status to running
    _job_store[job_id] = {
        "status": "running",
        "progress": 0.0,
        "cputime": None,
        "result_urls": None,
        "error": None,
    }

    try:
        # Load video and convert to matrix
        video_data = load_video(video_meta["path"])
        matrix = video_to_2d(video_data)

        # Run algorithm
        result = run_algorithm(
            method_id=request.method_id,
            algorithm_id=request.algorithm_id,
            data=matrix,
        )

        # Convert results to frames
        frames_dict = matrix_results_to_frames(
            L=result.L,
            S=result.S,
            O=result.O,
            height=video_meta["height"],
            width=video_meta["width"],
        )

        # Save result frames as PNGs
        job_results_dir = RESULTS_DIR / job_id
        result_urls = {}
        for component in ("L", "S", "O"):
            component_dir = job_results_dir / component
            frames_to_png(frames_dict[component], component_dir, prefix=component)
            result_urls[component] = f"/api/jobs/{job_id}/results/{component}"

        _job_store[job_id] = {
            "status": "complete",
            "progress": 1.0,
            "cputime": result.cputime,
            "result_urls": result_urls,
            "error": None,
            "results_dir": str(job_results_dir),
            "nframes": video_meta["nframes"],
        }

    except Exception as e:
        _job_store[job_id] = {
            "status": "failed",
            "progress": 0.0,
            "cputime": None,
            "result_urls": None,
            "error": str(e),
        }

    store = _job_store[job_id]
    return JobStatus(
        job_id=job_id,
        status=store["status"],
        progress=store["progress"],
        cputime=store["cputime"],
        result_urls=store["result_urls"],
        error=store.get("error"),
        nframes=store.get("nframes"),
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Return job status."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    store = _job_store[job_id]
    return JobStatus(
        job_id=job_id,
        status=store["status"],
        progress=store["progress"],
        cputime=store["cputime"],
        result_urls=store["result_urls"],
        error=store.get("error"),
        nframes=store.get("nframes"),
    )


@router.get("/jobs/{job_id}/results/{component}/frame/{frame_num}")
def get_result_frame(job_id: str, component: str, frame_num: int):
    """Return a result frame PNG for a specific component."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    store = _job_store[job_id]
    if store["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")

    if component not in ("L", "S", "O"):
        raise HTTPException(status_code=400, detail="Invalid component. Use L, S, or O")

    results_dir = Path(store["results_dir"])
    frame_path = results_dir / component / f"{component}_{frame_num:04d}.png"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    from fastapi.responses import FileResponse

    return FileResponse(str(frame_path), media_type="image/png")
