"""RapidOCR provider for HealthApp.

Local, offline, zero-cost OCR using PP-OCRv4 models on the ONNX runtime.
Implements the same ``OCRProvider`` boundary as the NVIDIA provider, so the
canonical data model and ingestion pipeline stay vendor-independent.

Strategy for recognition quality without spending money:
1. Upscale small photos so the detector sees more detail.
2. Rely on the built-in orientation classifier (cls) to correct rotated
   photos before recognition.
3. Keep per-block confidence so low-confidence text can be routed to the
   human-review step instead of silently polluting facts.
"""

from __future__ import annotations

import asyncio
import io
import statistics
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..schemas import OCRBlock
from .ingestion import OCRProvider, OCRResult


def _sort_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder OCR blocks into natural reading order, with a self-check.

    Strategy: cluster blocks into rows by y-center proximity, sort inside a
    row by x, sort rows top to bottom. Then VERIFY the clustering: a row
    whose vertical span far exceeds the typical block height means several
    physical rows were merged (this happens on dense multi-column tables),
    and sorting by x would scramble columns. If too many rows look like
    that, the detection order was already better — keep it.
    """
    if len(blocks) <= 1:
        return blocks
    items: list[dict[str, Any]] = []
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        items.append(
            {
                "block": block,
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "height": y1 - y0,
                "x0": x0,
            }
        )
    items.sort(key=lambda item: item["cy"])
    heights = [item["height"] for item in items]
    median_h = statistics.median(heights)
    row_gap = max(median_h * 0.5, 8.0)
    rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_cy: float | None = None
    for item in items:
        if last_cy is None or abs(item["cy"] - last_cy) <= row_gap:
            current.append(item)
        else:
            rows.append(current)
            current = [item]
        last_cy = item["cy"]
    if current:
        rows.append(current)

    # Self-check: rows merged across physical lines have a huge vertical
    # span relative to the typical block height. Judge by the FRACTION OF
    # BLOCKS inside bad rows (a single merged mega-row can hold dozens of
    # blocks while being one of many rows).
    span_limit = 2.0 * median_h
    bad_blocks = sum(
        len(row)
        for row in rows
        if (max(i["block"]["bbox"][3] for i in row)
            - min(i["block"]["bbox"][1] for i in row)) > span_limit
    )
    if rows and bad_blocks / len(blocks) > 0.15:
        return blocks  # clustering failed; detection order was better

    ordered: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: item["x0"])
        ordered.extend(item["block"] for item in row)
    return ordered


class RapidOCRProvider(OCRProvider):
    """OCR via RapidOCR (PP-OCRv4 det + cls + rec, ONNX runtime)."""

    engine_name = "rapidocr-ppocrv4"

    def __init__(
        self,
        preprocess: bool = True,
        sort_reading_order: bool = True,
    ) -> None:
        # Deferred import: the heavy ORM/onnx modules load only when this
        # provider is actually selected.
        from rapidocr_onnxruntime import RapidOCR

        # Default config is CPU inference (use_cuda=False). Passing
        # det/cls/rec_use_cuda kwargs triggers a bug in rapidocr 1.2.3's
        # param updater (KeyError: 'model_path'), so keep the constructor bare.
        self._engine = RapidOCR()
        self.preprocess = preprocess
        self.sort_reading_order = sort_reading_order

    @staticmethod
    def _prepare_image(content: bytes, media_type: str | None) -> Image.Image:
        """Decode the upload into an RGB image, upscaling tiny photos."""
        del media_type  # only images are routed here; PDF handling comes later
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError(
                "Unsupported or corrupted image. Please upload PNG/JPEG."
            ) from exc

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Small photos lose fine print. Upscale when the short side is tiny.
        width, height = image.size
        min_side = 600
        scale = min_side / min(width, height)
        if scale > 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    async def recognize(
        self,
        content: bytes,
        media_type: str | None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> OCRResult:
        image = self._prepare_image(content, media_type)

        import cv2
        import numpy as np

        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if preprocess_fn is not None:
            array = preprocess_fn(array)
        elif self.preprocess:
            from .image_preprocess import preprocess_image

            array = preprocess_image(array)
        # RapidOCR is CPU-bound and synchronous; keep the event loop free.
        result, _ = await asyncio.to_thread(self._engine, array)

        blocks: list[dict[str, Any]] = []
        if result:
            for index, item in enumerate(result, start=1):
                box, text, score = item[0], item[1], float(item[2])
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                blocks.append(
                    {
                        "block_id": f"ocr_block_{index}",
                        "page": 1,
                        "text": str(text).strip(),
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "confidence": score,
                    }
                )
        if self.sort_reading_order:
            blocks = _sort_reading_order(blocks)
        text_lines = [block["text"] for block in blocks]

        return OCRResult(
            text="\n".join(text_lines).strip(),
            blocks=blocks,
            engine=self.engine_name,
        )
