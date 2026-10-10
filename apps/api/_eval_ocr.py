"""OCR evaluation: baseline vs deskew vs deskew+reading-order sort.

For each image in the test folder, runs RapidOCR in three modes and computes
character error rate (CER) against the matching .gt.txt, then writes a
comparison Markdown report.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import time
from pathlib import Path

from app.services.rapid_ocr_provider import RapidOCRProvider

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")
REPORT = TEST_DIR / "OCR预处理对比报告.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODES = [
    ("无预处理", False, False),
    ("+deskew", True, False),
    ("+deskew+排序", True, True),
]


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


def error_examples(gt: str, ocr: str, limit: int = 2) -> list[str]:
    matcher = difflib.SequenceMatcher(None, gt, ocr, autojunk=False)
    examples: list[str] = []
    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag in ("replace", "delete") and i2 - i1 > 0:
            g = gt[i1:i2][:40]
            o = ocr[j1:j2][:40]
            if g.strip() and o.strip() and g != o:
                examples.append(f"GT:「{g}」 → OCR:「{o}」")
                if len(examples) >= limit:
                    break
    return examples


def run_ocr_blocking(
    provider: RapidOCRProvider, data: bytes, mime: str
) -> tuple[str, list[dict]]:
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(provider.recognize(data, mime))
        return result.text, result.blocks
    finally:
        loop.close()


def main() -> None:
    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    providers = {
        label: RapidOCRProvider(preprocess=p, sort_reading_order=s)
        for label, p, s in MODES
    }
    labels = [label for label, _, _ in MODES]
    started = time.time()
    data: dict[str, dict] = {}

    for img in images:
        gt_path = TEST_DIR / (img.stem + ".gt.txt")
        if not gt_path.exists():
            data[img.name] = {"gt_chars": 0, "skip": "缺少 GT", "results": {}}
            continue
        gt_norm = normalize(load_gt(gt_path))
        entry: dict = {"gt_chars": len(gt_norm), "results": {}}
        mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
        raw = img.read_bytes()
        for label, provider in providers.items():
            text, blocks = run_ocr_blocking(provider, raw, mime)
            ocr_norm = normalize(text)
            dist = levenshtein(gt_norm, ocr_norm) if gt_norm else 0
            entry["results"][label] = {
                "blocks": len(blocks),
                "dist": dist,
                "cer": dist / len(gt_norm) if gt_norm else 0.0,
                "examples": error_examples(gt_norm, ocr_norm),
            }
        data[img.name] = entry

    elapsed = round(time.time() - started, 1)
    scored = [v for v in data.values() if "skip" not in v]

    lines: list[str] = []
    lines.append("# OCR 预处理与排序对比报告\n")
    lines.append(f"- 测试时间：2026-10-09")
    lines.append(f"- OCR 引擎：RapidOCR（PP-OCRv4，本地 ONNX）")
    lines.append(f"- 管线：deskew（条件触发）+ 文本行阅读顺序排序")
    lines.append(f"- 测试图片：{len(scored)} 张真实报告照片（{TEST_DIR}）")
    lines.append(f"- 总耗时：{elapsed}s\n")

    lines.append("## 总体对比\n")
    totals = {label: [0, 0] for label in labels}
    for v in scored:
        for label in labels:
            totals[label][0] += v["results"][label]["dist"]
            totals[label][1] += v["gt_chars"]
    lines.append("| 模式 | 总体 CER | 识别错误数 | 相对无预处理 |")
    lines.append("|---|---|---|---|")
    base_cer = totals[labels[0]][0] / totals[labels[0]][1]
    for label in labels:
        dist, length = totals[label]
        cer = dist / length
        delta = f"{cer * 100 - base_cer * 100:+.2f} 个百分点"
        lines.append(
            f"| {label} | **{cer * 100:.2f}%** | {dist} | {delta} |"
        )
    lines.append("")

    lines.append("## 单张对比\n")
    lines.append(
        "| 图片 | GT字符 | " + " | ".join(f"{l} CER" for l in labels)
        + " | " + f"{labels[-1]}相对{labels[0]}" + " |"
    )
    lines.append("|---|---|" + "---|" * len(labels) + "---|")
    rows_sorted = sorted(data.items(), key=lambda kv: kv[1]["gt_chars"], reverse=True)
    for name, entry in rows_sorted:
        if "skip" in entry:
            lines.append(f"| {name} | - | " + "|".join(["-"] * len(labels)) + f" | {entry['skip']} |")
            continue
        cells = [f"{entry['results'][l]['cer'] * 100:.2f}%" for l in labels]
        delta = (entry["results"][labels[-1]]["cer"] - entry["results"][labels[0]]["cer"]) * 100
        lines.append(
            f"| {name} | {entry['gt_chars']} | " + " | ".join(cells)
            + f" | {delta:+.2f} 个百分点 |"
        )
    lines.append("")

    lines.append("## 最终管线错误示例（每张最多 2 条）\n")
    final_label = labels[-1]
    for name, entry in rows_sorted:
        if "skip" in entry:
            continue
        ex = entry["results"][final_label]["examples"]
        if ex:
            lines.append(f"**{name}**")
            for e in ex:
                lines.append(f"- {e}")
            lines.append("")

    lines.append("## 小结\n")
    for name, entry in rows_sorted:
        if "skip" in entry:
            continue
        r0 = entry["results"][labels[0]]["cer"]
        rf = entry["results"][labels[-1]]["cer"]
        delta = (rf - r0) * 100
        trend = "改善" if delta < -0.5 else ("变差" if delta > 0.5 else "持平")
        lines.append(
            f"- {name}：{r0 * 100:.2f}% → {rf * 100:.2f}%（{trend}，{delta:+.2f} 个百分点）"
        )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{REPORT}")
    for label in labels:
        dist, length = totals[label]
        print(f"{label} 总体 CER: {dist / length * 100:.2f}% ({dist}/{length})")
    for name, entry in sorted(data.items()):
        if "skip" in entry:
            continue
        vals = [entry["results"][l]["cer"] * 100 for l in labels]
        print(f"  {name}: " + " -> ".join(f"{v:.2f}%" for v in vals))


if __name__ == "__main__":
    main()
