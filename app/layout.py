"""Layout analysis: segment a page into TEXT / TABLE / IMAGE / PLAN regions.

Heuristic, OpenCV-based approach:
- TABLE: morphological detection of long horizontal/vertical ruling lines.
- IMAGE: large dense non-text blobs (photos, stamps, scans within a scan).
- PLAN: large sparse line-art areas (cadastral drawings) — high edge content
  but very low ink coverage, typically with long connected strokes.
- TEXT: everything else that contains OCR words.

Regions are axis-aligned boxes in page pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ocr_engine import OcrPageResult


@dataclass
class Region:
    """A detected page region."""

    type: str  # TEXT | TABLE | IMAGE | PLAN
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "type": self.type, "x": self.x, "y": self.y,
            "w": self.w, "h": self.h, "score": round(self.score, 3),
        }


def _detect_tables(binary: np.ndarray, min_dim: int = 120) -> list[Region]:
    """Find ruled tables via morphological line extraction."""
    inv = 255 - binary
    h, w = binary.shape[:2]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 20), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 40, 20)))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    grid = cv2.add(horiz, vert)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)

    regions: list[Region] = []
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_dim or ch < min_dim:
            continue
        # A real table has both orientations of lines inside its box
        sub_h = np.count_nonzero(horiz[y : y + ch, x : x + cw])
        sub_v = np.count_nonzero(vert[y : y + ch, x : x + cw])
        if sub_h > 0 and sub_v > 0:
            score = min(1.0, (sub_h + sub_v) / float(cw * ch) * 20)
            regions.append(Region("TABLE", x, y, cw, ch, score))
    return regions


def _ink_density(binary: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    sub = binary[y : y + h, x : x + w]
    if sub.size == 0:
        return 0.0
    return float(np.count_nonzero(255 - sub)) / sub.size


def _detect_images_and_plans(
    binary: np.ndarray, ocr: OcrPageResult
) -> tuple[list[Region], list[Region]]:
    """Detect IMAGE (dense blobs) and PLAN (sparse line art) regions."""
    h, w = binary.shape[:2]
    page_area = float(h * w)
    inv = 255 - binary

    # Close small gaps so strokes of one drawing/photo merge into blobs.
    closed = cv2.morphologyEx(
        inv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images: list[Region] = []
    plans: list[Region] = []
    min_area = page_area * 0.03  # ignore blobs smaller than 3% of the page

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = float(cw * ch)
        if area < min_area:
            continue
        density = _ink_density(binary, x, y, cw, ch)
        words_inside = sum(
            1
            for wd in ocr.words
            if x <= wd.x + wd.w / 2 <= x + cw and y <= wd.y + wd.h / 2 <= y + ch
        )
        word_density = words_inside / (area / 10000.0)  # words per 100x100 px

        if density > 0.45 and word_density < 0.5:
            # Very dense, almost no recognized words -> photo / stamp / seal
            images.append(Region("IMAGE", x, y, cw, ch, min(1.0, density)))
        elif area > page_area * 0.15 and 0.01 < density < 0.25 and word_density < 0.8:
            # Large, sparse, stroke-dominated -> cadastral plan / drawing
            plans.append(Region("PLAN", x, y, cw, ch, min(1.0, area / page_area)))
    return images, plans


def _detect_text(binary: np.ndarray, ocr: OcrPageResult) -> list[Region]:
    """Group OCR words into text block regions, using Tesseract block ids."""
    blocks: dict[tuple[int, int], list] = {}
    for wd in ocr.words:
        blocks.setdefault((wd.block, wd.par), []).append(wd)

    regions: list[Region] = []
    for words in blocks.values():
        x0 = min(w.x for w in words)
        y0 = min(w.y for w in words)
        x1 = max(w.x + w.w for w in words)
        y1 = max(w.y + w.h for w in words)
        conf = float(np.mean([w.conf for w in words]))
        regions.append(Region("TEXT", x0, y0, x1 - x0, y1 - y0, conf / 100.0))
    return regions


def _boxes_overlap(a: Region, b: Region, ratio: float = 0.5) -> bool:
    """True if boxes overlap by at least ``ratio`` of the smaller box."""
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return False
    inter = (x1 - x0) * (y1 - y0)
    smaller = min(a.w * a.h, b.w * b.h)
    return smaller > 0 and inter / smaller >= ratio


def analyze_layout(binary: np.ndarray, ocr: OcrPageResult) -> list[Region]:
    """Full layout segmentation of one page.

    Precedence: TABLE > PLAN > IMAGE > TEXT. Text blocks that fall mostly
    inside a table are kept (cells contain text) but plans/images suppress
    overlapping text regions to avoid phantom labels on drawings.
    """
    tables = _detect_tables(binary)
    images, plans = _detect_images_and_plans(binary, ocr)
    texts = _detect_text(binary, ocr)

    regions: list[Region] = []
    regions.extend(tables)
    regions.extend(plans)
    regions.extend(images)

    for t in texts:
        if any(_boxes_overlap(t, p) for p in plans):
            continue
        if any(_boxes_overlap(t, im, ratio=0.8) for im in images):
            continue
        regions.append(t)
    return regions
