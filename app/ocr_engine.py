"""Classical OCR layer: rasterization + Tesseract TSV extraction.

Tesseract is driven through pytesseract. The primary language is Romanian
(``ron``) with a ``ron+eng`` fallback when the Romanian traineddata is not
installed; if neither is available we degrade to whatever languages exist
and record a warning in the result.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .config import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class WordBox:
    """A single OCR word with its bounding box and confidence."""

    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    block: int
    par: int
    line: int
    word_num: int


@dataclass
class OcrPageResult:
    """OCR result for one page."""

    page_number: int
    width: int
    height: int
    text: str = ""
    words: list[WordBox] = field(default_factory=list)
    mean_confidence: float = 0.0
    lang_used: str = ""
    warnings: list[str] = field(default_factory=list)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def available_languages() -> list[str]:
    try:
        return pytesseract.get_languages()
    except Exception:
        return []


def resolve_language(settings: Settings) -> tuple[str, list[str]]:
    """Pick the best usable Tesseract language string and any warnings."""
    warnings: list[str] = []
    if not tesseract_available():
        return settings.tesseract_lang, ["tesseract binary not found on PATH"]

    langs = set(available_languages())

    def ok(lang_string: str) -> bool:
        return all(part in langs for part in lang_string.split("+"))

    if ok(settings.tesseract_lang):
        return settings.tesseract_lang, warnings
    warnings.append(
        f"primary language '{settings.tesseract_lang}' not installed; "
        f"trying fallback '{settings.tesseract_fallback_lang}'"
    )
    if ok(settings.tesseract_fallback_lang):
        return settings.tesseract_fallback_lang, warnings
    warnings.append("fallback language unavailable; using 'eng'")
    return "eng", warnings


def rasterize_document(file_bytes: bytes, filename: str, dpi: int = 300) -> list[np.ndarray]:
    """Convert an uploaded document into a list of BGR page images.

    PDFs are rasterized with PyMuPDF; TIFF/PNG/JPEG are decoded directly
    (multi-page TIFFs are expanded via PIL).
    """
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    pages: list[np.ndarray] = []

    if ext == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return pages

    # Image formats (PIL handles multi-page TIFF)
    pil = Image.open(io.BytesIO(file_bytes))
    for frame_idx in range(getattr(pil, "n_frames", 1)):
        pil.seek(frame_idx)
        rgb = pil.convert("RGB")
        arr = np.array(rgb)
        pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return pages


def ocr_page(binary: np.ndarray, page_number: int, settings: Settings) -> OcrPageResult:
    """Run Tesseract on one binarized page and return words + confidence."""
    lang, warnings = resolve_language(settings)
    height, width = binary.shape[:2]
    result = OcrPageResult(
        page_number=page_number, width=width, height=height,
        lang_used=lang, warnings=warnings,
    )
    if not tesseract_available():
        result.warnings.append("OCR skipped: tesseract not installed")
        return result

    tsv = pytesseract.image_to_data(
        binary, lang=lang, output_type=pytesseract.Output.DICT, config="--psm 3"
    )
    words: list[WordBox] = []
    confs: list[float] = []
    lines: dict[tuple[int, int, int], list[str]] = {}

    n = len(tsv.get("text", []))
    for i in range(n):
        text = (tsv["text"][i] or "").strip()
        try:
            conf = float(tsv["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        wb = WordBox(
            text=text,
            x=int(tsv["left"][i]), y=int(tsv["top"][i]),
            w=int(tsv["width"][i]), h=int(tsv["height"][i]),
            conf=conf,
            block=int(tsv["block_num"][i]), par=int(tsv["par_num"][i]),
            line=int(tsv["line_num"][i]), word_num=int(tsv["word_num"][i]),
        )
        words.append(wb)
        confs.append(conf)
        key = (wb.block, wb.par, wb.line)
        lines.setdefault(key, []).append(text)

    result.words = words
    result.text = "\n".join(" ".join(parts) for parts in lines.values())
    result.mean_confidence = float(np.mean(confs)) if confs else 0.0
    return result


def low_confidence_words(result: OcrPageResult, threshold: float) -> list[WordBox]:
    """Words whose confidence is below the flagging threshold."""
    return [w for w in result.words if w.conf < threshold]
