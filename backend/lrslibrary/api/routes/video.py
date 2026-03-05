"""Video upload and frame serving API routes."""

import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from lrslibrary.config import UPLOAD_DIR
from lrslibrary.models import VideoMetadata
from lrslibrary.video.loader import load_video

router = APIRouter(tags=["video"])

# In-memory store for video metadata (Phase 1 - no database)
_video_store: dict[str, dict] = {}


@router.post("/video/upload", response_model=VideoMetadata)
async def upload_video(file: UploadFile):
    """Upload a video file and return metadata.

    Accepts .avi and .mp4 files.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".avi", ".mp4"):
        raise HTTPException(status_code=400, detail="Only .avi and .mp4 files are supported")

    video_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{video_id}{ext}"

    # Save uploaded file
    content = await file.read()
    save_path.write_bytes(content)

    # Load and get metadata
    try:
        video_data = load_video(save_path)
    except (FileNotFoundError, ValueError) as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    # Store metadata and path
    _video_store[video_id] = {
        "path": str(save_path),
        "width": video_data["width"],
        "height": video_data["height"],
        "nframes": video_data["nframes"],
        "fps": video_data["fps"],
    }

    return VideoMetadata(
        video_id=video_id,
        width=video_data["width"],
        height=video_data["height"],
        nframes=video_data["nframes"],
        fps=video_data["fps"],
    )


@router.get("/video/{video_id}/frame/{frame_num}")
def get_video_frame(video_id: str, frame_num: int):
    """Return a specific frame as PNG."""
    if video_id not in _video_store:
        raise HTTPException(status_code=404, detail="Video not found")

    meta = _video_store[video_id]
    if frame_num < 0 or frame_num >= meta["nframes"]:
        raise HTTPException(status_code=400, detail="Frame number out of range")

    # Read the specific frame
    cap = cv2.VideoCapture(meta["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame")

    # Convert to grayscale and encode as PNG in memory (no temp files)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    success, png_bytes = cv2.imencode(".png", frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return StreamingResponse(
        iter([png_bytes.tobytes()]),
        media_type="image/png",
    )


def get_video_store() -> dict[str, dict]:
    """Accessor for the video store (used by jobs route)."""
    return _video_store
