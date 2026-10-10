"""Smoke test: rule-based field extraction on the test set."""
import asyncio
import sys
from pathlib import Path

from app.services.field_extractor import extract_event
from app.services.nvidia_providers import build_default_providers
from app.schemas import SourceArtifact

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")


async def main() -> None:
    ocr, _normalizer = build_default_providers()
    for name in ["test3.jpeg", "test4.jpeg", "test5.jpg", "test.jpg"]:
        path = TEST_DIR / name
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        result = await ocr.recognize(path.read_bytes(), mime)
        artifact = SourceArtifact(
            artifact_id="art_smoke",
            patient_id="p_smoke",
            file_name=name,
            sha256="x",
            storage_uri="x",
            ocr_text=result.text,
            ocr_blocks=result.blocks,
        )
        event = extract_event("p_smoke", artifact, result)
        print("=" * 60)
        print(f"### {name}  (OCR blocks: {len(result.blocks)})")
        if event is None:
            print("  -> NO EVENT (no facts parsed)")
            continue
        print(f"  event: {event.event_type} | {event.title} | {event.event_time.precision}")
        print(f"  facts: {len(event.facts)}")
        for fact in event.facts:
            v = fact.value
            vstr = (
                f"{v.number} {v.unit or ''}".strip()
                if hasattr(v, "number")
                else f"[text] {v.text}"
            )
            ref = fact.reference_range
            refstr = (
                f"{ref.low}-{ref.high} {ref.unit or ''}".strip()
                if ref and (ref.low is not None or ref.high is not None)
                else (ref.text if ref else "-")
            )
            print(
                f"    {fact.concept.display:<12} | {fact.concept.local_display[:28]:<28} "
                f"| {vstr:<14} | ref={refstr:<12} | {fact.interpretation:<6} | conf={fact.confidence}"
            )


if __name__ == "__main__":
    asyncio.run(main())
