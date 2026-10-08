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
from typing import Any

from PIL import Image, UnidentifiedImageError

from ..schemas import OCRBlock
from .ingestion import OCRProvider, OCRResult


class RapidOCRProvider(OCRProvider):
    """OCR via RapidOCR (PP-OCRv4 det + cls + rec, ONNX runtime)."""

    engine_name = "rapidocr-ppocrv4"

    def __init__(self) -> None:
        # Deferred import: the heavy ORM/onnx modules load only when this
        # provider is actually selected.
        from rapidocr_onnxruntime import RapidOCR

        # Default config is CPU inference (use_cuda=False). Passing
        # det/cls/rec_use_cuda kwargs triggers a bug in rapidocr 1.2.3's
        # param updater (KeyError: 'model_path'), so keep the constructor bare.
        self._engine = RapidOCR()

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
    ) -> OCRResult:
        image = self._prepare_image(content, media_type)

        import numpy as np

        array = np.array(image)
        # RapidOCR is CPU-bound and synchronous; keep the event loop free.
        result, _ = await asyncio.to_thread(self._engine, array)

        blocks: list[dict[str, Any]] = []
        text_lines: list[str] = []
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
                text_lines.append(str(text).strip())

        return OCRResult(
            text="\n".join(text_lines).strip(),
            blocks=blocks,
            engine=self.engine_name,
        )
