"""Debug column-row alignment for a given image."""
import asyncio
import statistics
import sys
from pathlib import Path

from app.services.nvidia_providers import build_default_providers

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")


def debug_columns(items, med_h):
    widths = [it["w"] for it in items]
    med_w = statistics.median(widths) if widths else 40.0
    col_gap = 0.6 * med_w
    by_x = sorted(items, key=lambda it: it["x0"])
    columns = []
    current = []
    prev_x1 = None
    for it in by_x:
        if prev_x1 is not None and it["x0"] - prev_x1 > col_gap:
            columns.append(current)
            current = []
        current.append(it)
        prev_x1 = it["x1"]
    if current:
        columns.append(current)
    print(f"med_w={med_w:.1f} col_gap={col_gap:.1f} n_cols={len(columns)}")
    row_gap_y = 0.7 * med_h
    for ci, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda it: it["y0"])
        rows_c = []
        cur = []
        prev_y0 = None
        for it in col_sorted:
            if prev_y0 is not None and it["y0"] - prev_y0 > row_gap_y:
                rows_c.append(cur)
                cur = []
            cur.append(it)
            prev_y0 = it["y0"]
        if cur:
            rows_c.append(cur)
        print(f"\n--- col{ci} (n={len(col)}, rows={len(rows_c)}) ---")
        for ri, r in enumerate(rows_c):
            t = " | ".join(f'{it["block"]["text"][:14]}' for it in r)
            y0s = "/".join(f'{it["y0"]:.0f}' for it in r)
            print(f"  r{ri}: y0=[{y0s}]  {t}")


async def main(name: str) -> None:
    ocr, _ = build_default_providers()
    path = TEST_DIR / name
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    result = await ocr.recognize(path.read_bytes(), mime)
    items = []
    for block in result.blocks:
        x0, y0, x1, y1 = block["bbox"]
        items.append(
            {
                "block": block,
                "x0": x0, "x1": x1, "y0": y0, "w": x1 - x0, "h": y1 - y0,
            }
        )
    heights = [it["h"] for it in items]
    med_h = statistics.median(heights)
    print(f"### {name}: blocks={len(items)} med_h={med_h:.1f}")
    debug_columns(items, med_h)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "test.jpg"))
