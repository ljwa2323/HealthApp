"""Debug: inspect test.jpg block y-distribution to see why rows merge."""
import asyncio
from pathlib import Path

from app.services.nvidia_providers import build_default_providers
from app.services.field_extractor import cluster_rows

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")


async def main() -> None:
    ocr, _ = build_default_providers()
    result = await ocr.recognize((TEST_DIR / "test.jpg").read_bytes(), "image/jpeg")
    blocks = result.blocks
    print(f"total blocks: {len(blocks)}")
    # Show first 24 blocks sorted by y with their spans
    items = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        items.append((y0, x0, y1 - y0, b["text"][:18], b["block_id"]))
    items.sort()
    for y0, x0, h, text, bid in items[:26]:
        print(f"y0={y0:7.1f} x0={x0:7.1f} h={h:5.1f} | {text}")
    rows = cluster_rows(blocks)
    print(f"\nclustered rows: {len(rows)}")
    for i, row in enumerate(rows):
        top = min(b["block"]["bbox"][1] for b in row)
        bottom = max(b["block"]["bbox"][3] for b in row)
        texts = " ".join(b["block"]["text"][:12] for b in row)
        print(f"row{i}: span={bottom - top:6.1f} n={len(row):2d} | {texts[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
