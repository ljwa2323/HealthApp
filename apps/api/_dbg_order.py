import asyncio
import statistics
from pathlib import Path

from app.services.rapid_ocr_provider import RapidOCRProvider, _sort_reading_order

D = Path(r"C:\Users\31683\Desktop\图片识别测试")


def cluster_debug(blocks):
    items = []
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        items.append(
            {
                "block": block,
                "cy": (y0 + y1) / 2.0,
                "height": y1 - y0,
            }
        )
    items.sort(key=lambda item: item["cy"])
    median_h = statistics.median(item["height"] for item in items)
    row_gap = max(median_h * 0.5, 8.0)
    rows = []
    current = []
    last_cy = None
    for item in items:
        if last_cy is None or abs(item["cy"] - last_cy) <= row_gap:
            current.append(item)
        else:
            rows.append(current)
            current = [item]
        last_cy = item["cy"]
    if current:
        rows.append(current)
    span_limit = 2.0 * median_h
    bad = 0
    for row in rows:
        span = max(i["block"]["bbox"][3] for i in row) - min(
            i["block"]["bbox"][1] for i in row
        )
        if span > span_limit:
            bad += 1
    print(
        f"median_h={median_h:.1f} rows={len(rows)} bad={bad} ratio={bad/len(rows):.2f} "
        f"span_limit={span_limit:.1f} -> "
        f"{'FALLBACK' if rows and bad/len(rows) > 0.3 else 'SORT'}"
    )
    for i, row in enumerate(rows[:6]):
        texts = [it["block"]["text"][:8] for it in row][:6]
        span = max(it["block"]["bbox"][3] for it in row) - min(
            it["block"]["bbox"][1] for it in row
        )
        print(f"  row{i}: n={len(row)} span={span:.0f} {'BAD' if span>span_limit else 'ok'} {texts}")


async def main() -> None:
    p = RapidOCRProvider(preprocess=True, sort_reading_order=False)
    for name in ["test.jpg", "test3.jpeg", "test2.webp"]:
        res = await p.recognize((D / name).read_bytes(), "image/jpeg")
        print(f"--- {name} (blocks={len(res.blocks)}) ---")
        cluster_debug(res.blocks)
        out = _sort_reading_order(res.blocks)
        print(f"    sort changed={len(out)!=len(res.blocks) or any(a is not b for a,b in zip(out,res.blocks))}")


if __name__ == "__main__":
    asyncio.run(main())
