"""
Main FastAPI application for NICU AI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


app = FastAPI(
    title="NICU AI",
    description=(
        "Upload one patient PDF and generate a NICU discharge summary."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    router,
    prefix="/api",
)


@app.get("/")
def root():
    return {
        "message": "NICU AI backend is running",
        "docs": "/docs",
    }