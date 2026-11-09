"""Serve the Electrify production bundle and the bounded v2 demo API."""
from collections import Counter
from pathlib import Path
import re

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api_v2 import router

root = Path(__file__).resolve().parent.parent
dist = root / "ui" / "dist"
app = FastAPI(title="Electrify", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router, prefix="/api")
limits = {"session": 160, "study": 48, "branch": 160, "step": 600, "edits": 320}
counts = Counter()
active_writes = 0
session_routes = re.compile(r"^/api/v2/sessions/[a-f0-9]{32}/(studies(?:/[a-f0-9]{32}/branch)?|step|edits)$")


@app.middleware("http")
async def bound_demo_writes(request: Request, call_next):
    global active_writes
    path = request.url.path
    if request.method != "POST" or not path.startswith("/api/"):
        return await call_next(request)
    match = session_routes.fullmatch(path)
    if path == "/api/v2/sessions":
        kind = "session"
    elif match:
        route = match[1]
        kind = "branch" if route.endswith("/branch") else "study" if route == "studies" else route
    else:
        return JSONResponse({"detail": "Unknown demo route"}, status_code=404)
    size = request.headers.get("content-length", "")
    if not size.isdecimal() or not 0 < int(size) <= 8192:
        return JSONResponse({"detail": "Invalid demo request size"}, status_code=413)
    if counts[kind] >= limits[kind] or active_writes >= 6:
        return JSONResponse({"detail": "Demo limit reached or busy. Try again later."}, status_code=429)
    counts[kind] += 1
    active_writes += 1
    try:
        return await call_next(request)
    finally:
        active_writes -= 1


@app.get("/evidence")
def evidence():
    return FileResponse(root / "artifacts/gridshift-v2-wam-001/RESULTS.md", media_type="text/plain")


@app.get("/")
def index():
    return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
