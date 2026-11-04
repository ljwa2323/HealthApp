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

from .pdf_support import is_pdf_bytes

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

# --- C3: physical-photo issues on the text branch (conservative) ---
# A photo of a report sheet that is crumpled / tilted / creased / glare-lit
# hurts OCR. Each check reports only when it is fairly certain; the gate
# stays non-blocking either way.
SKEW_OFF_AXIS_DEG = 8.0        # line more than N degrees off 0/90 counts
SKEW_OFF_AXIS_RATIO_MAX = 0.30  # share of long lines off-axis to flag
SKEW_MIN_LINES = 4              # too few lines -> no verdict (handwriting)
FOLD_DIP = 18.0                 # column/row mean dip (dark crease shadow)
FOLD_DARK_SHARE_MAX = 0.015     # share of dark columns/rows to flag
FOLD_MAX_DIP = 25.0             # and the crease must be this much darker
GLARE_OVER_PAPER = 12.0         # glare must be this much brighter than paper
GLARE_MIN_RATIO = 0.005         # connected bright blob >= 0.5% of frame


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


def _skew_ratio(gray: np.ndarray) -> float:
    """Share of long straight lines that sit noticeably off 0/90 degrees.

    A photo of a flat report has horizontal/vertical table and text lines;
    a crumpled or perspective-tilted shot rotates those lines off-axis.
    Handwriting-heavy or line-free images yield few lines and return 0.0
    (no verdict) rather than a false positive.
    """
    h, w = gray.shape
    min_len = int(0.15 * min(h, w))
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(60, min_len),
        minLineLength=min_len,
        maxLineGap=12,
    )
    if lines is None or len(lines) < SKEW_MIN_LINES:
        return 0.0
    off = 0
    total = 0
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = ln
        dx, dy = x2 - x1, y2 - y1
        if (dx * dx + dy * dy) < min_len * min_len:
            continue
        total += 1
        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
        # Deviation from the nearest axis (horizontal 0/180 or vertical 90).
        dev = min(angle, 180.0 - angle, abs(90.0 - angle))
        if dev > SKEW_OFF_AXIS_DEG:
            off += 1
    return off / total if total else 0.0


def _fold_ratio(gray: np.ndarray) -> tuple[float, float]:
    """Share of dark columns/rows and the deepest dip.

    A crease casts a long thin shadow across the sheet; after heavy
    blurring the text disappears and the crease survives as a dip in the
    column (vertical crease) or row (horizontal crease) mean profile.
    Returns (dark_share, max_dip) — a crease is short (a few percent of
    the frame) but deep; the caller requires both.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), 11).astype(np.float64)
    col = blurred.mean(axis=0)
    row = blurred.mean(axis=1)

    def _stats(profile: np.ndarray) -> tuple[float, float]:
        base = float(np.median(profile))
        dip = base - profile
        dark_share = float((dip > FOLD_DIP).mean())
        return dark_share, float(dip.max())

    col_share, col_dip = _stats(col)
    row_share, row_dip = _stats(row)
    if col_share > row_share:
        return col_share, col_dip
    return row_share, row_dip


def _glare_ratio(gray: np.ndarray) -> float:
    """Largest connected bright blob as a share of the frame.

    Glare (a desk lamp or window reflection on the sheet) is a solid blob
    distinctly brighter than the paper itself. Small specular highlights
    on ink/photo prints are eroded away by the opening morphology.
    """
    paper = float(np.median(gray))
    threshold = max(250, min(254, paper + GLARE_OVER_PAPER))
    bright = (gray > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return 0.0
    total_px = gray.shape[0] * gray.shape[1]
    largest = max(int(st[4]) for st in stats[1:])
    return largest / total_px


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
    caller decides whether issues block anything. PDF uploads are
    rasterized (first page) before the checks so they go through the
    same gate as photos.
    """
    if doc_type not in ("auto", "text", "imaging"):
        doc_type = "auto"

    pdf_page_count = 0
    if is_pdf_bytes(content):
        from .pdf_support import pdf_to_first_page

        image, pdf_page_count = pdf_to_first_page(content)
        if image is None:
            return ImageQuality(
                ok=False,
                issues=["PDF 无法解析或已加密"],
                metrics={"pdf_page_count": 0},
                doc_type=doc_type,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        content = buffer.getvalue()

    gray, width, height, error = _load_gray(content)
    if error is not None or gray is None:
        return ImageQuality(
            ok=False,
            issues=[error or "无法读取图片"],
            metrics={"pdf_page_count": pdf_page_count},
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
    if pdf_page_count:
        metrics["pdf_page_count"] = pdf_page_count

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
        # C3: physical-photo issues (crumple / crease / glare).
        # Screen captures have no paper: their corners match the frame mean
        # (no lighting gradient) and large areas are render-pure white. In
        # that case the crease/glare/skew checks fire on table lines and
        # white cells — skip them, keep resolution/blur/brightness.
        corner_mean = _corner_mean(gray)
        is_capture = abs(corner_mean - brightness) < 10.0 and bright_ratio > 0.35
        metrics["screen_capture"] = bool(is_capture)
        if not is_capture:
            skew_ratio = _skew_ratio(gray)
            fold_share, fold_dip = _fold_ratio(gray)
            glare_ratio = _glare_ratio(gray)
            metrics["skew_ratio"] = round(skew_ratio, 3)
            metrics["fold_ratio"] = round(fold_share, 4)
            metrics["fold_dip"] = round(fold_dip, 1)
            metrics["glare_ratio"] = round(glare_ratio, 4)
            if skew_ratio > SKEW_OFF_AXIS_RATIO_MAX:
                issues.append("纸张不平整或倾斜")
            if fold_share > FOLD_DARK_SHARE_MAX and fold_dip > FOLD_MAX_DIP:
                issues.append("纸张有折痕")
            if glare_ratio > GLARE_MIN_RATIO:
                issues.append("局部反光")

    return ImageQuality(
        ok=not issues,
        issues=issues,
        metrics=metrics,
        doc_type=used_type,
    )
