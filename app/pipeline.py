"""End-to-end document pipeline and job management.

Stages: ingest -> preprocess -> OCR -> layout -> Agent 1 (correction) ->
Agent 2 (classification & tagging) -> storage & export.

Jobs run in background threads; progress is exposed through an in-memory
job registry so the REST API (and the test web page) can poll status.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .agents.classification import ClassificationAgent, ClassificationResult
from .agents.correction import CorrectionAgent
from .config import Settings, get_settings
from .layout import analyze_layout
from .llm.client import get_llm_client
from .ocr_engine import OcrPageResult, ocr_page, rasterize_document
from .preprocess import preprocess_page
from .storage import build_searchable_pdf, store_document

STAGES = [
    "ingest",
    "preprocess",
    "ocr",
    "layout",
    "correction",
    "classification",
    "export",
    "done",
]


@dataclass
class Job:
    """In-memory state for one document processing job."""

    id: str
    filename: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "ingest"
    error: str = ""
    created_at: float = field(default_factory=time.time)
    result: dict | None = None
    pdf_path: str = ""
    page_previews: list[str] = field(default_factory=list)  # paths to jpg previews


class JobRegistry:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)

    def snapshot(self) -> list[Job]:
        """Return a shallow copy of all jobs under the registry lock."""
        with self._lock:
            return list(self._jobs.values())

    def cleanup(self, data_dir: str, max_age_hours: float) -> int:
        """Remove finished jobs older than *max_age_hours* and delete their
        preview directories.  Returns the number of jobs purged."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        with self._lock:
            old_ids = [
                jid
                for jid, job in self._jobs.items()
                if job.created_at < cutoff and job.status in ("done", "error")
            ]
            for jid in old_ids:
                job = self._jobs.pop(jid)
                for preview in job.page_previews:
                    try:
                        Path(preview).unlink(missing_ok=True)
                    except Exception:
                        pass
                job_dir = Path(data_dir) / jid
                if job_dir.exists():
                    try:
                        for f in job_dir.iterdir():
                            f.unlink(missing_ok=True)
                        job_dir.rmdir()
                    except Exception:
                        pass
                removed += 1
        return removed


class DocumentPipeline:
    """Orchestrates all processing stages for one document."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        llm = get_llm_client(self.settings)
        self.correction = CorrectionAgent(llm, self.settings)
        self.classification = ClassificationAgent(llm)

    # -- individual stages (kept separate for testability) ----------------

    def rasterize(self, file_bytes: bytes, filename: str) -> list:
        return rasterize_document(
            file_bytes, filename, dpi=self.settings.ocr_dpi
        )

    def preprocess_pages(self, pages: list) -> list[dict]:
        return [preprocess_page(p) for p in pages]

    def ocr_pages(self, preprocessed: list[dict]) -> list[OcrPageResult]:
        return [
            ocr_page(pp["binary"], i + 1, self.settings)
            for i, pp in enumerate(preprocessed)
        ]

    def analyze(self, preprocessed: list[dict], ocr_pages: list[OcrPageResult]) -> list[list]:
        return [
            analyze_layout(pp["binary"], op)
            for pp, op in zip(preprocessed, ocr_pages)
        ]

    def correct(self, ocr_pages: list[OcrPageResult]) -> list[str]:
        return [self.correction.correct_page(op) for op in ocr_pages]

    def classify(
        self, corrected_texts: list[str], regions_per_page: list[list]
    ) -> ClassificationResult:
        full_text = "\n\n".join(corrected_texts)
        has_plan = any(
            r.type == "PLAN" for regions in regions_per_page for r in regions
        )
        return self.classification.classify(full_text, has_plan_region=has_plan)

    # -- full run ----------------------------------------------------------

    def run(self, job: Job, file_bytes: bytes, registry: JobRegistry) -> Job:
        """Execute all stages for a job, updating progress in the registry."""
        s = self.settings
        try:
            registry.update(job, status="running", stage="ingest")
            pages = self.rasterize(file_bytes, job.filename)
            if not pages:
                raise ValueError("document has no decodable pages")

            registry.update(job, stage="preprocess")
            preprocessed = self.preprocess_pages(pages)

            registry.update(job, stage="ocr")
            ocr_results = self.ocr_pages(preprocessed)

            registry.update(job, stage="layout")
            regions_per_page = self.analyze(preprocessed, ocr_results)

            registry.update(job, stage="correction")
            corrected_texts = self.correct(ocr_results)

            registry.update(job, stage="classification")
            classification = self.classify(corrected_texts, regions_per_page)

            registry.update(job, stage="export")
            job_dir = Path(s.data_dir) / job.id
            job_dir.mkdir(parents=True, exist_ok=True)

            preview_paths: list[str] = []
            for i, pp in enumerate(preprocessed):
                preview = job_dir / f"page_{i + 1}.jpg"
                cv2.imwrite(str(preview), pp["image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                preview_paths.append(str(preview))

            pdf_bytes, pdf_method = build_searchable_pdf(
                file_bytes, job.filename,
                [pp["image"] for pp in preprocessed], ocr_results, s,
            )

            payload = self._build_result_payload(
                job, ocr_results, regions_per_page, corrected_texts,
                classification, pdf_method,
            )
            paths = store_document(
                s, job.id, file_bytes, job.filename, pdf_bytes, payload, classification
            )
            registry.update(
                job, status="done", stage="done", result=payload,
                pdf_path=paths["pdf_path"], page_previews=preview_paths,
            )
        except Exception as exc:  # surface the failure to the API caller
            registry.update(job, status="error", stage="error", error=str(exc))
        return job

    def _build_result_payload(
        self, job, ocr_results, regions_per_page, corrected_texts,
        classification, pdf_method,
    ) -> dict:
        pages_payload = []
        for ocr, regions, corrected in zip(ocr_results, regions_per_page, corrected_texts):
            pages_payload.append(
                {
                    "page_number": ocr.page_number,
                    "width": ocr.width,
                    "height": ocr.height,
                    "raw_text": ocr.text,
                    "corrected_text": corrected,
                    "mean_confidence": round(ocr.mean_confidence, 2),
                    "ocr_language": ocr.lang_used,
                    "warnings": ocr.warnings,
                    "regions": [r.to_dict() for r in regions],
                    "words": [
                        {
                            "text": w.text, "x": w.x, "y": w.y,
                            "w": w.w, "h": w.h, "conf": round(w.conf, 1),
                        }
                        for w in ocr.words
                    ],
                }
            )
        all_confs = [o.mean_confidence for o in ocr_results] or [0.0]
        return {
            "job_id": job.id,
            "filename": job.filename,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_pages": len(ocr_results),
            "overall_mean_confidence": round(sum(all_confs) / len(all_confs), 2),
            "pdf_method": pdf_method,
            "pages": pages_payload,
            "classification": classification.to_dict(),
        }


class FolderWatcher(threading.Thread):
    """Watch the scanner output folder and enqueue new files automatically.

    Simple polling implementation (no external dependency). A file is picked
    up once its size is stable for two consecutive polls, which avoids
    reading half-written scans from the Avision AD345GN.
    """

    POLL_INTERVAL = 3.0

    def __init__(self, registry: JobRegistry, submit, settings: Settings):
        super().__init__(daemon=True)
        self.registry = registry
        self.submit = submit  # callable(filename, file_bytes)
        self.settings = settings
        self._seen: dict[str, int] = {}
        self._stop_event = threading.Event()

    def run(self) -> None:
        from .ocr_engine import SUPPORTED_EXTENSIONS

        folder = Path(self.settings.watch_folder)
        folder.mkdir(parents=True, exist_ok=True)
        while not self._stop_event.is_set():
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                size = path.stat().st_size
                key = str(path)
                if key in self._seen and self._seen[key] == size and size > 0:
                    try:
                        self.submit(path.name, path.read_bytes())
                        path.unlink()  # consume the file after ingest
                    except Exception:
                        pass
                    self._seen.pop(key, None)
                else:
                    self._seen[key] = size
            self._stop_event.wait(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
