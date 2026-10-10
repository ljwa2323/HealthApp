"""Test: does 2x upscaling rescue test1's Chinese text detection?"""
import asyncio
import re
from pathlib import Path

from PIL import Image

from app.services.rapid_ocr_provider import RapidOCRProvider

D = Path(r"C:\Users\31683\Desktop\图片识别测试")
GT = D / "test1.gt.txt"


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


async def main() -> None:
    raw = (D / "test1.webp").read_bytes()
    gt_norm = normalize(
        "\n".join(
            l.strip()
            for l in GT.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        )
    )
    img = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
    p = RapidOCRProvider(preprocess=False, sort_reading_order=False)
    for scale in (1, 2, 3):
        w, h = img.size
        big = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        import io

        buf = io.BytesIO()
        big.save(buf, format="PNG")
        res = await p.recognize(buf.getvalue(), "image/png")
        ocr_norm = normalize("".join(b["text"] for b in res.blocks))
        cer = levenshtein(gt_norm, ocr_norm) / len(gt_norm)
        chinese_blocks = sum(
            1 for b in res.blocks if re.search(r"[\u4e00-\u9fff]", b["text"])
        )
        print(
            f"scale={scale}x ({w*scale}x{h*scale}): blocks={len(res.blocks)} "
            f"含中文块={chinese_blocks} CER={cer*100:.2f}%"
        )


if __name__ == "__main__":
    asyncio.run(main())
