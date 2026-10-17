"""Image quality gate for document uploads (text / imaging split).

Computes cheap, deterministic quality metrics on the uploaded photo and
flags reasons why recognition (text) or AI reading (imaging) is likely to
fail, so the frontend can ask the user to retake BEFORE bad OCR results
are produced.

Deliberately NOT strict: the gate only reports issues; it never blocks
the upload (the caller decides). The user may still choose to keep the
photo.

Two branches, selected by `doc_type`:

- "text"    : white-background reports (lab sheets, prescriptions…).
              Issues: 分辨率过低 / 图片模糊 / 光线太暗
- "imaging" : medical image photos (X-ray / CT / MRI film shots).
              Issues: 分辨率过低 / 画面模糊 / 画面过暗 / 画面过曝

`doc_type="auto"` picks the branch with a cheap heuristic (dark-background
film photos vs white-background documents). Explicit values always win.

The imaging thresholds are intentionally conservative starting points —
calibrate them on your own film-photo samples (the text thresholds were
calibrated on the real-photo lab-report test set; film photos are a
different image family, see NOTES below).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- text thresholds (calibrated on the real-photo lab-report test set) ---
RES_MIN_SIDE = 400
BLUR_LAPLACIAN_MIN = 60.0
DARK_BRIGHTNESS_MIN = 45.0

# --- imaging thresholds (starting points; recalibrate on film samples) ---
# Film photos need more pixels for detail than text documents. 512 is a
# pragmatic floor: a web-resolution film shot (e.g. 860x573) is still
# legible for AI reading; only genuinely tiny images get flagged.
RES_MIN_SIDE_IMAGING = 512
# Film images have naturally less high-frequency energy than dense text,
# so the blur bar must be much lower to avoid mis-flagging sharp films.
BLUR_LAPLACIAN_MIN_IMAGING = 25.0
# Whole-frame near-black (lens cap / unexposed) — not the same as the
# document "too dark" test, which assumes a white background.
DARK_BRIGHTNESS_MIN_IMAGING = 18.0
# Saturated-pixel ratio (values > 250). Overexposed / lightbox-glare
# washes out structure.
OVEREXPOSE_RATIO_MAX = 0.35
# NOTE: no tilt check on the imaging branch. Tilt detection is calibrated
# for text documents (OCR needs lines horizontal); anatomy has no fixed
# line direction — a skeleton illustration measured 33.7° "tilt" purely
# from rib/scapula contours, and patients are often scanned at odd angles
# anyway, so the check is both meaningless and a false-positive source.

# --- auto-split heuristic thresholds ---
# Share of pixels < 80 needed (together with dark corners) to call the
# shot an imaging film photo. Calibrated on real film/illustration
# samples (e.g. a blue skeleton render: dark_ratio 0.425, corner 74)
# vs the white-background lab-report test set (dark_ratio < 0.08).
DARK_RATIO_IMAGING = 0.35
CORNER_DARK_MAX = 110.0        # corner mean gray for film photos
BRIGHT_RATIO_TEXT = 0.30       # share of pixels > 200
CORNER_BRIGHT_MIN = 140.0      # corner mean gray for documents


@dataclass
class ImageQuality:
    ok: bool
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # Branch actually used: "auto" | "text" | "imaging"
    doc_type: str = "auto"


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


def _corner_mean(gray: np.ndarray) -> float:
    """Mean gray of the four corner patches (document vs film background)."""
    h, w = gray.shape
    patch_h = max(8, int(h * 0.12))
    patch_w = max(8, int(w * 0.12))
    corners = [
        gray[:patch_h, :patch_w],
        gray[:patch_h, -patch_w:],
        gray[-patch_h:, :patch_w],
        gray[-patch_h:, -patch_w:],
    ]
    return float(np.mean(np.concatenate([c.ravel() for c in corners])))


def _center_dark_ratio(gray: np.ndarray) -> float:
    """Share of dark pixels (< 80) in the central 60% region.

    Film photos / medical illustrations have a dark subject in the
    middle; white-background documents keep the middle bright no matter
    the lighting.
    """
    h, w = gray.shape
    ch0, ch1 = int(h * 0.2), int(h * 0.8)
    cw0, cw1 = int(w * 0.2), int(w * 0.8)
    return float((gray[ch0:ch1, cw0:cw1] < 80).mean())


def _guess_doc_type(
    gray: np.ndarray,
    *,
    bright_ratio: float,
    dark_ratio: float,
) -> str:
    """Cheap text-vs-imaging split; "text" is the safe default.

    Imaging: film photos are mostly dark (negative-style black background)
    with dark corners. Text: white-background documents are mostly bright.
    Ambiguous input falls back to "text" so existing document behaviour
    is unchanged; the caller may pass an explicit doc_type to override.

    Order matters: the bright-document test runs first, because dense
    black text on a white sheet can produce a high central dark share
    (e.g. 0.50) that must not be mistaken for an imaging subject.
    """
    corner_mean = _corner_mean(gray)
    if bright_ratio > BRIGHT_RATIO_TEXT and corner_mean > CORNER_BRIGHT_MIN:
        return "text"
    if dark_ratio > DARK_RATIO_IMAGING and corner_mean < CORNER_DARK_MAX:
        return "imaging"
    return "text"


def analyze_image_quality(
    content: bytes,
    doc_type: str = "auto",
) -> ImageQuality:
    """Run the quality gate. `doc_type` is "auto" | "text" | "imaging".

    Always returns an ImageQuality (never raises for bad images); the
    caller decides whether issues block anything.
    """
    if doc_type not in ("auto", "text", "imaging"):
        doc_type = "auto"

    gray, width, height, error = _load_gray(content)
    if error is not None or gray is None:
        return ImageQuality(
            ok=False,
            issues=[error or "无法读取图片"],
            metrics={},
            doc_type=doc_type,
        )

    short_side = min(width, height)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    bright_ratio = float((gray > 200).mean())
    dark_ratio = float((gray < 80).mean())
    saturated_ratio = float((gray > 250).mean())

    used_type = doc_type
    if used_type == "auto":
        used_type = _guess_doc_type(
            gray,
            bright_ratio=bright_ratio,
            dark_ratio=dark_ratio,
        )

    issues: list[str] = []
    metrics: dict = {
        "width": width,
        "height": height,
        "short_side": short_side,
        "laplacian_var": round(laplacian_var, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "bright_ratio": round(bright_ratio, 3),
        "dark_ratio": round(dark_ratio, 3),
        "center_dark_ratio": round(_center_dark_ratio(gray), 3),
        "saturated_ratio": round(saturated_ratio, 3),
    }

    if used_type == "imaging":
        if short_side < RES_MIN_SIDE_IMAGING:
            issues.append("分辨率过低")
        if laplacian_var < BLUR_LAPLACIAN_MIN_IMAGING:
            issues.append("画面模糊")
        if brightness < DARK_BRIGHTNESS_MIN_IMAGING:
            issues.append("画面过暗")
        if saturated_ratio > OVEREXPOSE_RATIO_MAX:
            issues.append("画面过曝")
    else:
        # Text branch: original white-background document checks.
        if short_side < RES_MIN_SIDE:
            issues.append("分辨率过低")
        if laplacian_var < BLUR_LAPLACIAN_MIN:
            issues.append("图片模糊")
        if brightness < DARK_BRIGHTNESS_MIN:
            issues.append("光线太暗")

    return ImageQuality(
        ok=not issues,
        issues=issues,
        metrics=metrics,
        doc_type=used_type,
    )
