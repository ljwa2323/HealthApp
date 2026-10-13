"""Verify check-date extraction (keyword-first) on images that carry dates."""
import asyncio
from pathlib import Path

from app.services.field_extractor import extract_event
from app.services.nvidia_providers import build_default_providers
from app.schemas import SourceArtifact

D = Path(r"C:\Users\31683\Desktop\图片识别测试")


async def main() -> None:
    ocr, _ = build_default_providers()
    for name in ["test3.jpeg", "test4.jpeg", "web19.jpg", "web01.jpg", "test.jpg"]:
        path = D / name
        if not path.exists():
            continue
        result = await ocr.recognize(path.read_bytes(), "image/jpeg")
        artifact = SourceArtifact(
            artifact_id="a", patient_id="p", file_name=name,
            sha256="x", storage_uri="x",
            ocr_text=result.text, ocr_blocks=result.blocks,
        )
        event = extract_event("p", artifact, result)
        t = event.event_time if event else None
        print(f"{name:<12} start={t.start if t else None!s:<12} precision={t.precision if t else None}")


if __name__ == "__main__":
    asyncio.run(main())
