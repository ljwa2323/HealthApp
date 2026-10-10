"""Generate 4 synthetic degraded lab-report test images + ground-truth files.

Each image renders the SAME clean report then applies a controlled
degradation that mimics real phone photos:
  1. perspective  - large-angle oblique shot of the paper
  2. dark_shadow  - dim lighting + a diagonal shadow gradient
  3. blur         - mild defocus/motion blur
  4. rotate_lowcontrast - small rotation + washed-out low contrast

Outputs go to the user's Desktop as JPEG + a matching .gt.txt per image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DESKTOP = Path(r"C:\Users\31683\Desktop")
FONT = "C:/Windows/Fonts/msyh.ttc"

REPORT_LINES = [
    "XX市中心医院检验报告单",
    "姓名：测试患者    性别：女    年龄：35",
    "序号    项目代码    项目名称        检验结果    参考范围      单位",
    "1       WBC        白细胞计数      6.28        3.50~9.50    10^9/L",
    "2       RBC        红细胞计数      4.52        4.30~5.80    10^12/L",
    "3       HGB        血红蛋白        135          130~175       g/l",
    "4       PLT        血小板计数      218          125~350       10^9/L",
    "5       NEUT%      中性粒细胞比值  58.4         40.00~75.00   %",
    "6       LYM%       淋巴细胞比值    30.2         20.00~50.00   %",
    "7       MONO%      单核细胞比值    5.8          3.00~10.00    %",
    "8       EO%        嗜酸性粒细胞比值 2.1         0.40~8.00     %",
    "9       BASO%      嗜碱性粒细胞比值 0.4         0~1.00        %",
    "10      HCT        红细胞压积      41.3         40.00~50.00   %",
    "采样时间：2026-10-08 09:30    检验者：李  审核者：王",
]

GT_ROWS = [
    "序号\t项目代码\t项目名称\t检验结果\t参考范围\t单位",
    "1\tWBC\t白细胞计数\t6.28\t3.50~9.50\t10^9/L",
    "2\tRBC\t红细胞计数\t4.52\t4.30~5.80\t10^12/L",
    "3\tHGB\t血红蛋白\t135\t130~175\tg/l",
    "4\tPLT\t血小板计数\t218\t125~350\t10^9/L",
    "5\tNEUT%\t中性粒细胞比值\t58.4\t40.00~75.00\t%",
    "6\tLYM%\t淋巴细胞比值\t30.2\t20.00~50.00\t%",
    "7\tMONO%\t单核细胞比值\t5.8\t3.00~10.00\t%",
    "8\tEO%\t嗜酸性粒细胞比值\t2.1\t0.40~8.00\t%",
    "9\tBASO%\t嗜碱性粒细胞比值\t0.4\t0~1.00\t%",
    "10\tHCT\t红细胞压积\t41.3\t40.00~50.00\t%",
]


def render_clean_report() -> np.ndarray:
    img = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(img)
    title = ImageFont.truetype(FONT, 34)
    body = ImageFont.truetype(FONT, 24)
    y = 40
    for line in REPORT_LINES:
        draw.text((60, y), line, font=title if y == 40 else body, fill="black")
        y += 48
    return np.array(img)


def save_and_gt(name: str, image_bgr: np.ndarray) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    jpg_path = DESKTOP / f"{name}.jpg"
    img.save(jpg_path, "JPEG", quality=82)
    gt_path = DESKTOP / f"{name}.gt.txt"
    gt_path.write_text(
        "# Ground Truth - " + name + ".jpg（合成退化测试图）\n"
        "# 合成报告内容，列用制表符分隔\n"
        "XX市中心医院检验报告单\n"
        "姓名：测试患者    性别：女    年龄：35\n"
        + "\n".join(GT_ROWS)
        + "\n采样时间：2026-10-08 09:30    检验者：李  审核者：王\n",
        encoding="utf-8",
    )
    print(f"saved: {jpg_path.name}  +  {gt_path.name}")


def degrade_perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32(
        [
            [w * 0.10, h * 0.18],
            [w * 0.92, h * 0.05],
            [w * 0.16, h * 0.92],
            [w * 0.95, h * 0.85],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, matrix, (w, h), borderValue=(235, 235, 230)
    )


def degrade_dark_shadow(img: np.ndarray) -> np.ndarray:
    dark = (img.astype(np.float32) * 0.42).clip(0, 255)
    h, w = img.shape[:2]
    gradient = np.linspace(0.55, 1.0, w, dtype=np.float32)
    mask = np.tile(gradient, (h, 1)).reshape(h, w, 1)
    out = dark * mask
    out[..., 0] *= 0.82  # slight warm/uneven tint on blue channel
    return out.clip(0, 255).astype(np.uint8)


def degrade_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=1.7, sigmaY=1.7)


def degrade_rotate_lowcontrast(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h), borderValue=(235, 235, 230)
    )
    low = rotated.astype(np.float32) * 0.62 + 72.0
    return low.clip(0, 255).astype(np.uint8)


def main() -> None:
    clean = render_clean_report()
    degrade_perspective(clean.copy())
    jobs = [
        ("synth_1_perspective", degrade_perspective),
        ("synth_2_dark_shadow", degrade_dark_shadow),
        ("synth_3_blur", degrade_blur),
        ("synth_4_rotate_lowcontrast", degrade_rotate_lowcontrast),
    ]
    for name, fn in jobs:
        save_and_gt(name, fn(clean.copy()))


if __name__ == "__main__":
    main()
