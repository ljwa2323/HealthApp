"""Compute combined CER for images that pass the quality gate."""
import re
from pathlib import Path

D = Path(r"C:\Users\31683\Desktop\图片识别测试")
# final pipeline CER per image (from the last evaluation run)
CER = {
    "test.jpg": 0.2860,
    "test3.jpeg": 0.0842,
    "test4.jpeg": 0.1192,
    "test5.jpg": 0.0419,
}


def gt_len(name: str) -> int:
    text = "\n".join(
        l.strip()
        for l in (D / name).read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    )
    return len(re.sub(r"\s+", "", text))


total_len = 0
total_dist = 0
print(f"{'图片':<12}{'GT字符':>6}{'CER':>8}{'错误数':>6}")
for name, cer in CER.items():
    gt_name = name.rsplit(".", 1)[0] + ".gt.txt"
    n = gt_len(gt_name)
    d = round(cer * n)
    total_len += n
    total_dist += d
    print(f"{name:<12}{n:>6}{cer * 100:>7.2f}%{d:>6}")

print(f"\n通过质量检测的图（test1/test2 被建议重拍，已排除）")
print(f"汇总: GT 字符 {total_len}，错误 {total_dist}")
print(f"总体 CER: {total_dist / total_len * 100:.2f}%")
print(f"总体准确率: {(1 - total_dist / total_len) * 100:.2f}%")
