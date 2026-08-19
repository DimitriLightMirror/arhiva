"""Image preprocessing: deskew, denoise, adaptive binarization, orientation.

All functions operate on OpenCV BGR images (numpy arrays). The preprocessing
steps are deliberately conservative so that cadastral plans and stamps are
not destroyed before layout analysis and OCR.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_orientation(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect page orientation via Tesseract OSD and rotate upright.

    Returns (possibly rotated image, rotation degrees applied).
    Falls back to the original image when OSD fails (e.g. drawings only).
    """
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(bgr, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
    except Exception:
        return bgr, 0

    if angle == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 90
    if angle == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180), 180
    if angle == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 270
    return bgr, 0


def deskew(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate small skew angle from text-like edges and rotate to fix it.

    Uses a Hough-line based estimate on the binarized image. Only corrects
    angles within +/- 15 degrees; larger rotations are an orientation problem
    handled by detect_orientation().
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.shape[0] < 100:
        return bgr, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); normalize to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 15 or abs(angle) < 0.05:
        return bgr, 0.0

    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, float(angle)


def denoise(bgr: np.ndarray) -> np.ndarray:
    """Light non-local-means denoise that preserves thin strokes."""
    return cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)


def binarize(bgr: np.ndarray) -> np.ndarray:
    """Adaptive (Sauvola-like) binarization, robust to uneven illumination."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    return binary


def preprocess_page(bgr: np.ndarray, correct_orientation: bool = True) -> dict:
    """Full preprocessing chain for one page image.

    Returns a dict with the cleaned color image (for display/overlay), the
    binarized image (for OCR/layout) and diagnostic angles.
    """
    out = bgr
    rotation = 0
    if correct_orientation:
        out, rotation = detect_orientation(out)
    out, skew_angle = deskew(out)
    out = denoise(out)
    binary = binarize(out)
    return {
        "image": out,
        "binary": binary,
        "rotation": rotation,
        "skew_angle": skew_angle,
    }
