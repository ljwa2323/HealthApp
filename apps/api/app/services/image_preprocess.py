"""Image preprocessing pipeline for OCR quality improvement.

Steps (each with failure guards — if a step cannot be applied reliably it
leaves the image unchanged rather than risking damage):
  1. perspective correction  - detect the dominant document quad and
     warp it to a rectangle (fixes oblique photos)
  2. deskew                   - rotate small residual tilt (<15 deg)
  3. CLAHE contrast enhance   - equalize lighting/shadow on the L channel

The original file is never touched: this pipeline only produces the copy
that is sent to OCR.
"""

from __future__ import annotations

import math
import statistics

import cv2
import numpy as np


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(summed)]       # top-left
    ordered[2] = points[np.argmax(summed)]       # bottom-right
    ordered[1] = points[np.argmin(diff)]         # top-right (min x-y)
    ordered[3] = points[np.argmax(diff)]         # bottom-left (max x-y)
    return ordered


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """Warp the dominant quadrilateral (the paper) to a rectangle."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < h * w * 0.15:
        return img  # no confident document region
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return img
    pts = _order_corners(approx.reshape(4, 2).astype(np.float32))
    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[2] - pts[3])
    d3 = np.linalg.norm(pts[0] - pts[3])
    d4 = np.linalg.norm(pts[1] - pts[2])
    width = max(int(d1), int(d2))
    height = max(int(d3), int(d4))
    if width <= 0 or height <= 0 or width / height > 8 or height / width > 8:
        return img
    dst = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        img, matrix, (width, height), borderValue=(255, 255, 255)
    )


def deskew(img: np.ndarray) -> np.ndarray:
    """Rotate small text-line tilt back to horizontal (<15 deg).

    Trigger conditions (learned from the real-photo ablation set):
    - at least 5 near-horizontal lines detected,
    - near-horizontal lines must dominate all detected lines (>=50%),
      otherwise the 'tilt' is just table-line noise (e.g. straight photos
      whose grid lines yield a few 4-degree readings),
    - the horizontal angles must agree (std <= 2.5 deg; upscaling noise
      spreads readings by ~2 deg),
    - |median| in [1, 15] deg.
    If any condition fails the image is left untouched.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=max(80, int(w * 0.25)),
        maxLineGap=12,
    )
    if lines is None:
        return img
    horizontal: list[float] = []
    total = 0
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        total += 1
        if -10 <= angle <= 10:
            horizontal.append(angle)
    if len(horizontal) < 5:
        return img
    if len(horizontal) / total < 0.5:
        return img  # horizontal lines are not the dominant structure
    median = statistics.median(horizontal)
    if abs(median) < 1.0 or abs(median) > 15:
        return img
    if len(horizontal) >= 2 and statistics.stdev(horizontal) > 2.5:
        return img  # angles disagree too much; likely noise
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), median, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel of LAB to even out lighting/shadow."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(
    img: np.ndarray,
    *,
    perspective: bool = True,
    deskew_on: bool = True,
    clahe: bool = False,
) -> np.ndarray:
    """Apply the preprocessing chain to an OCR-bound copy.

    CLAHE is off by default: ablation on the real-photo test set showed it
    hurts 4 of 5 images (net +1.21pp CER) and only helped one.
    """
    result = img
    if perspective:
        result = correct_perspective(result)
    if deskew_on:
        result = deskew(result)
    if clahe:
        result = clahe_enhance(result)
    return result
