"""Debug aligned rows using the actual extractor logic."""
import asyncio
import sys
from pathlib import Path

from app.services.field_extractor import cluster_rows
from app.services.nvidia_providers import build_default_providers

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")


async def main(name: str) -> None:
    ocr, _ = build_default_providers()
    path = TEST_DIR / name
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    result = await ocr.recognize(path.read_bytes(), mime)
    rows = cluster_rows(result.blocks)
    print(f"### {name}: blocks={len(result.blocks)} aligned_rows={len(rows)}")
    for i, row in enumerate(rows):
        texts = " | ".join(b["block"]["text"][:16] for b in row)
        print(f"r{i:2d}: {texts[:90]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "test4.jpeg"))
