"""Calibrate image quality thresholds against the real-photo test set."""
from pathlib import Path

from app.services.image_quality import analyze_image_quality

D = Path(r"C:\Users\31683\Desktop\图片识别测试")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    for img in sorted(p for p in D.iterdir() if p.suffix.lower() in IMG_EXTS):
        q = analyze_image_quality(img.read_bytes())
        m = q.metrics
        flag = "OK " if q.ok else "!! "
        print(
            f"{flag}{img.name}: {m.get('width')}x{m.get('height')} "
            f"short={m.get('short_side')} lap={m.get('laplacian_var')} "
            f"bright={m.get('brightness')} contrast={m.get('contrast')} "
            f"issues={q.issues}"
        )


if __name__ == "__main__":
    main()
