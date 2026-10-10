"""Ablation study: isolate the contribution of each preprocessing step.

Modes:
  baseline   - no preprocessing
  perspective only
  deskew only
  clahe only
  all        - perspective + deskew + clahe (current pipeline default)

Prints a per-image CER matrix and a per-mode overall CER.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.services.image_preprocess import clahe_enhance, correct_perspective, deskew
from app.services.rapid_ocr_provider import RapidOCRProvider

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODES = {
    "baseline": None,
    "perspective": lambda img: correct_perspective(img),
    "deskew": lambda img: deskew(img),
    "clahe": lambda img: clahe_enhance(img),
    "all": lambda img: (
        clahe_enhance(deskew(correct_perspective(img)))
    ),
}


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


def load_gt(gt_path: Path) -> str:
    lines = []
    for raw in gt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def run_ocr_blocking(provider: RapidOCRProvider, data: bytes, mime: str, fn):
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            provider.recognize(data, mime, preprocess_fn=fn)
        )
        return result.text
    finally:
        loop.close()


def main() -> None:
    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    provider = RapidOCRProvider(preprocess=False)
    # per image -> per mode CER
    matrix: dict[str, dict[str, float]] = {}
    totals: dict[str, list] = {m: [0, 0] for m in MODES}  # mode -> [dist, gt_len]

    for img in images:
        gt_path = TEST_DIR / (img.stem + ".gt.txt")
        if not gt_path.exists():
            print(f"{img.name}: 缺少 GT，跳过")
            continue
        gt_norm = normalize(load_gt(gt_path))
        mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
        raw = img.read_bytes()
        matrix[img.name] = {}
        for mode, fn in MODES.items():
            text = run_ocr_blocking(provider, raw, mime, fn)
            ocr_norm = normalize(text)
            dist = levenshtein(gt_norm, ocr_norm)
            cer = dist / len(gt_norm) if gt_norm else 0.0
            matrix[img.name][mode] = cer
            totals[mode][0] += dist
            totals[mode][1] += len(gt_norm)

    header = "图片".ljust(14) + "".join(m.ljust(14) for m in MODES)
    print(header)
    for name, row in matrix.items():
        cells = "".join(f"{row[m] * 100:6.2f}%".ljust(14) for m in MODES)
        print(name.ljust(14) + cells)

    print("\n总体 CER:")
    for mode in MODES:
        dist, length = totals[mode]
        print(f"  {mode.ljust(14)} {dist / length * 100:6.2f}%  ({dist}/{length})")

    # delta vs baseline
    base = totals["baseline"][0] / totals["baseline"][1]
    print("\n相对 baseline 变化:")
    for mode in MODES:
        dist, length = totals[mode]
        delta = dist / length - base
        print(f"  {mode.ljust(14)} {delta * 100:+.2f} 个百分点")


if __name__ == "__main__":
    main()
