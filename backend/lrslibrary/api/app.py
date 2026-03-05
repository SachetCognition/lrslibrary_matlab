"""FastAPI application setup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lrslibrary.api.routes import algorithms, jobs, video
from lrslibrary.config import CORS_ORIGINS

# Trigger algorithm auto-discovery
import lrslibrary.algorithms  # noqa: F401

app = FastAPI(
    title="LRSLibrary",
    description="Low-Rank and Sparse Decomposition Library API",
    version="0.1.0",
)

# CORS middleware (allow all origins for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(algorithms.router, prefix="/api")
app.include_router(video.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
