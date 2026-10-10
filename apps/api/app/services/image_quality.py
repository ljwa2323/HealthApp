"""Image quality gate for document uploads.

Computes cheap, deterministic quality metrics on the uploaded photo and
flags reasons why recognition is likely to fail, so the frontend can ask
the user to retake BEFORE bad OCR results are produced.

Deliberately NOT strict: the gate only reports issues; it never blocks
the upload (the caller decides).

Reasons (may be multiple):
  - "分辨率过低"      short side below RES_MIN_SIDE
  - "图片模糊"        Laplacian variance below BLUR_LAPLACIAN_MIN
  - "光线太暗"        mean gray below DARK_BRIGHTNESS_MIN

NOTE: "对比度过低" (gray std) was removed after calibration — white-background
lab reports all score low on raw gray std (even perfectly sharp ones), so it
mis-flags good photos. Resolution and blur are the trustworthy signals.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- thresholds (calibrated on the real-photo test set) ---
RES_MIN_SIDE = 400
BLUR_LAPLACIAN_MIN = 60.0
DARK_BRIGHTNESS_MIN = 45.0


@dataclass
class ImageQuality:
    ok: bool
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _load_gray(content: bytes) -> tuple[np.ndarray | None, int, int, str | None]:
    """Decode bytes into grayscale + dimensions, or (None, 0, 0, reason)."""
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.load()
    except (UnidentifiedImageError, OSError) as exc:
        return None, 0, 0, f"Unsupported or corrupted image: {exc}"
    if img.mode not in ("L", "RGB", "RGBA", "CMYK"):
        try:
            img = img.convert("RGB")
        except OSError as exc:
            return None, 0, 0, f"Unsupported image mode: {exc}"
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return gray, w, h, None


def analyze_image_quality(content: bytes) -> ImageQuality:
    gray, width, height, error = _load_gray(content)
    if error is not None or gray is None:
        return ImageQuality(ok=False, issues=[error or "无法读取图片"], metrics={})

    short_side = min(width, height)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    issues: list[str] = []
    if short_side < RES_MIN_SIDE:
        issues.append("分辨率过低")
    if laplacian_var < BLUR_LAPLACIAN_MIN:
        issues.append("图片模糊")
    if brightness < DARK_BRIGHTNESS_MIN:
        issues.append("光线太暗")

    return ImageQuality(
        ok=not issues,
        issues=issues,
        metrics={
            "width": width,
            "height": height,
            "short_side": short_side,
            "laplacian_var": round(laplacian_var, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
        },
    )
