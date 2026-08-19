"""Storage & export: searchable PDF generation, JSON sidecar, archive routing.

Searchable PDF strategy:
1. If the ``ocrmypdf`` binary is installed, we run it on the original file
   (best quality, proper PDF/A-2b conversion, reuses its own Tesseract layer).
2. Otherwise we fall back to a reportlab/pikepdf-style approach implemented
   here: each original page is embedded as a JPEG image into a new PDF, and an
   invisible text layer (render mode 3) is drawn at the OCR word positions.
   This produces a fully searchable PDF without external binaries. A strict
   PDF/A conversion on top of that requires pikepdf + veraPDF; install
   ocrmypdf on the server for full PDF/A compliance (see README).
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2

from .agents.classification import ClassificationResult
from .config import Settings
from .ocr_engine import OcrPageResult


def ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def _sanitize_component(name: str) -> str:
    """Make a string safe for use as a folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "doc"


def archive_destination(
    settings: Settings, document_class: str, doc_id: str, when: datetime | None = None
) -> Path:
    """Compute /archive/<document_class>/<year>/<doc_id>/ and create it."""
    when = when or datetime.now()
    dest = (
        Path(settings.archive_root)
        / _sanitize_component(document_class)
        / f"{when.year:04d}"
        / _sanitize_component(doc_id)
    )
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def build_searchable_pdf_ocrmypdf(
    original_bytes: bytes, filename: str, settings: Settings
) -> bytes | None:
    """Run ocrmypdf on the original file; return PDF bytes or None on failure."""
    import tempfile

    ext = ".pdf" if filename.lower().endswith(".pdf") else ".png"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext}"
        dst = Path(tmp) / "output.pdf"
        src.write_bytes(original_bytes)
        lang = settings.tesseract_lang
        try:
            subprocess.run(
                [
                    "ocrmypdf", "--skip-text", "--output-type", "pdfa",
                    "-l", lang, str(src), str(dst),
                ],
                check=True, capture_output=True, timeout=600,
            )
            return dst.read_bytes()
        except Exception:
            return None


def build_searchable_pdf_fallback(
    page_images: list, ocr_pages: list[OcrPageResult]
) -> bytes:
    """Searchable PDF without ocrmypdf: JPEG pages + invisible text layer.

    Word positions come from Tesseract TSV; text is drawn in render mode 3
    (invisible) so the visual output is exactly the original scan.
    """
    from reportlab.lib.colors import black
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)

    for image, ocr in zip(page_images, ocr_pages):
        h, w = image.shape[:2]
        c.setPageSize((w, h))

        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        c.drawImage(
            ImageReader(io.BytesIO(jpg.tobytes())), 0, 0, width=w, height=h,
            preserveAspectRatio=False, mask="auto",
        )

        # Invisible text layer (render mode 3 = neither fill nor stroke).
        for word in ocr.words:
            if not word.text:
                continue
            font_size = max(word.h * 0.85, 4)
            t = c.beginText(word.x, h - word.y - word.h * 0.85)
            t.setTextRenderMode(3)
            t.setFont("Helvetica", font_size)
            t.setFillColor(black)
            try:
                t.textOut(word.text)
            except Exception:
                continue  # skip glyphs not encodable in Helvetica
            c.drawText(t)
        c.showPage()

    c.save()
    return buf.getvalue()


def build_searchable_pdf(
    original_bytes: bytes,
    filename: str,
    page_images: list,
    ocr_pages: list[OcrPageResult],
    settings: Settings,
) -> tuple[bytes, str]:
    """Return (pdf_bytes, method) where method is 'ocrmypdf' or 'fallback'."""
    if ocrmypdf_available():
        pdf = build_searchable_pdf_ocrmypdf(original_bytes, filename, settings)
        if pdf:
            return pdf, "ocrmypdf"
    return build_searchable_pdf_fallback(page_images, ocr_pages), "fallback"


def write_json_sidecar(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def store_document(
    settings: Settings,
    doc_id: str,
    original_bytes: bytes,
    original_filename: str,
    pdf_bytes: bytes,
    result_payload: dict,
    classification: ClassificationResult,
) -> dict:
    """Route a finished document into the archive tree and write artifacts.

    Returns a dict with the created paths.
    """
    dest = archive_destination(settings, classification.document_class, doc_id)
    orig_name = _sanitize_component(original_filename) or "scan.bin"
    (dest / f"original_{orig_name}").write_bytes(original_bytes)
    pdf_path = dest / "searchable.pdf"
    pdf_path.write_bytes(pdf_bytes)
    json_path = dest / "result.json"
    write_json_sidecar(json_path, result_payload)
    return {
        "archive_dir": str(dest),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
    }
