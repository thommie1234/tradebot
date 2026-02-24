"""Sovereign Bot Web Dashboard — FastAPI application."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import auth, dashboard, positions, history, charts, system, stream

app = FastAPI(title="Sovereign Bot Dashboard", version="1.0.0")

# CORS — allow local access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(positions.router)
app.include_router(history.router)
app.include_router(charts.router)
app.include_router(system.router)
app.include_router(stream.router)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=False)
