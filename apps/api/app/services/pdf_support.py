"""PDF upload support: rasterize the first page to an RGB image.

Medical reports are frequently scanned to PDF while the rest of the
pipeline is image-based (RapidOCR + vision model). A PDF upload is
therefore rendered to a JPEG-like RGB image before recognition; the
original PDF is kept as the stored artifact. Multi-page PDFs use the
first page as the recognition target and report the page count so the
caller can surface "more pages exist" to the user.
"""

from __future__ import annotations

import io

from PIL import Image


def is_pdf_bytes(content: bytes) -> bool:
    """Cheap magic-byte check: PDFs start with %PDF."""
    return content.startswith(b"%PDF")


def pdf_to_first_page(
    content: bytes, dpi: int = 200
) -> tuple[Image.Image | None, int]:
    """Render the first page of a PDF.

    Returns (RGB image or None on failure, page_count). Never raises:
    a corrupted/encrypted PDF yields (None, 0) so the caller can fall
    back to a clean error message.
    """
    try:
        import pymupdf  # PyMuPDF >= 1.24 uses the non-deprecated name
    except ImportError:  # pragma: no cover - older 1.x wheels
        import fitz as pymupdf  # type: ignore[no-redef]
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        if page_count == 0:
            doc.close()
            return None, 0
        pix = doc[0].get_pixmap(dpi=dpi)
        doc.close()
    except Exception:  # noqa: BLE001 - corrupted / encrypted / unsupported
        return None, 0
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, page_count


# Hard cap so a 500-page slide deck can't trigger 500 model calls. Real
# medical PDFs are short; this only guards against accidental uploads.
PDF_MAX_PAGES = 30


def pdf_to_pages(
    content: bytes, dpi: int = 200
) -> tuple[list[Image.Image], int]:
    """Render every page of a PDF to RGB images.

    Returns (images, page_count). Images are truncated to PDF_MAX_PAGES.
    Never raises: a corrupted/encrypted PDF yields ([], 0).
    """
    try:
        import pymupdf  # PyMuPDF >= 1.24 uses the non-deprecated name
    except ImportError:  # pragma: no cover - older 1.x wheels
        import fitz as pymupdf  # type: ignore[no-redef]
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        n = min(page_count, PDF_MAX_PAGES)
        images: list[Image.Image] = []
        for i in range(n):
            pix = doc[i].get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        doc.close()
        if not images:
            return [], page_count
        return images, page_count
    except Exception:  # noqa: BLE001 - corrupted / encrypted / unsupported
        return [], 0
