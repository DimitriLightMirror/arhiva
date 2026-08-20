"""FastAPI application: REST ingest, job status, results, downloads, web UI.

Endpoints:
- GET  /                              -> web test page
- POST /api/scan                      -> upload a scan, returns {job_id}
- GET  /api/jobs                      -> list jobs
- GET  /api/jobs/{id}                 -> job status/progress
- GET  /api/jobs/{id}/result          -> full JSON result
- GET  /api/jobs/{id}/pdf             -> searchable PDF download
- GET  /api/jobs/{id}/json            -> JSON sidecar download
- GET  /api/jobs/{id}/pages/{n}.jpg   -> preprocessed page preview image
- GET  /api/health                    -> health check incl. tesseract status
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .ocr_engine import SUPPORTED_EXTENSIONS, tesseract_available
from .pipeline import DocumentPipeline, FolderWatcher, JobRegistry

settings = get_settings()
settings.ensure_dirs()

app = FastAPI(title="arhivadoc.eu OCR backend", version="0.1.0")
registry = JobRegistry()
pipeline = DocumentPipeline(settings)


def submit_document(filename: str, file_bytes: bytes) -> str:
    """Create a job and run the pipeline in a background thread."""
    job = registry.create(filename)
    thread = threading.Thread(
        target=pipeline.run, args=(job, file_bytes, registry), daemon=True
    )
    thread.start()
    return job.id


@app.on_event("startup")
def start_watcher() -> None:
    watcher = FolderWatcher(registry, submit_document, settings)
    watcher.start()
    app.state.watcher = watcher


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    job_id = submit_document(file.filename or "scan", data)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs() -> dict:
    # Purge stale finished jobs and their preview directories before listing.
    registry.cleanup(settings.data_dir, settings.job_retention_hours)
    jobs = [
        {
            "id": j.id, "filename": j.filename, "status": j.status,
            "stage": j.stage, "created_at": j.created_at,
        }
        for j in registry.snapshot()
    ]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id, "filename": job.filename, "status": job.status,
        "stage": job.stage, "error": job.error,
    }


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> JSONResponse:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job not finished (status: {job.status})")
    return JSONResponse(job.result)


@app.get("/api/jobs/{job_id}/pdf")
def job_pdf(job_id: str) -> FileResponse:
    job = registry.get(job_id)
    if not job or not job.pdf_path or not Path(job.pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF not available")
    return FileResponse(
        job.pdf_path, media_type="application/pdf",
        filename=f"{Path(job.filename).stem}_searchable.pdf",
    )


@app.get("/api/jobs/{job_id}/json")
def job_json(job_id: str) -> FileResponse:
    job = registry.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="JSON not available")
    json_path = Path(job.pdf_path).with_name("result.json")
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON sidecar not found")
    return FileResponse(
        str(json_path), media_type="application/json",
        filename=f"{Path(job.filename).stem}_result.json",
    )


@app.get("/api/jobs/{job_id}/pages/{page}.jpg")
def job_page_image(job_id: str, page: int) -> FileResponse:
    job = registry.get(job_id)
    if not job or page < 1 or page > len(job.page_previews):
        raise HTTPException(status_code=404, detail="Page not available")
    return FileResponse(job.page_previews[page - 1], media_type="image/jpeg")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "tesseract": tesseract_available(),
        "llm_provider": settings.llm_provider,
    }


# Static web test page served at "/".
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
