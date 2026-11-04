from __future__ import annotations

import io
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from ..schemas import (
    ClinicalEvent,
    EventTime,
    IngestionResult,
    OCRBlock,
    SourceArtifact,
)


@dataclass
class OCRResult:
    text: str
    blocks: list[dict]
    engine: str


class OCRProvider:
    """OCR adapter boundary.

    Production implementations can wrap PaddleOCR, PP-Structure,
    a cloud OCR API, or another OCR engine. The canonical data model
    must not depend on a specific vendor.
    """

    async def recognize(self, content: bytes, media_type: str | None) -> OCRResult:
        raise NotImplementedError


class DemoOCRProvider(OCRProvider):
    async def recognize(self, content: bytes, media_type: str | None) -> OCRResult:
        return OCRResult(
            text=(
                "Demo OCR placeholder. Configure a production OCR provider "
                "before extracting real medical records."
            ),
            blocks=[],
            engine="demo",
        )


class ClinicalNormalizer:
    """LLM normalization boundary.

    The normalizer receives OCR text plus layout metadata and must return
    schema-valid ClinicalEvent objects. It should never discard OCR source
    data and every extracted fact should carry evidence pointers.
    """

    async def normalize(
        self,
        patient_id: str,
        artifact: SourceArtifact,
        ocr: OCRResult,
    ) -> ClinicalEvent | None:
        return None


async def ingest_document(
    *,
    patient_id: str,
    file_name: str,
    media_type: str | None,
    content: bytes,
    ocr_provider: OCRProvider | None = None,
    normalizer: ClinicalNormalizer | None = None,
    doc_type_hint: str | None = None,
) -> IngestionResult:
    if ocr_provider is None or normalizer is None:
        from .nvidia_providers import build_default_providers

        default_ocr, default_normalizer = build_default_providers()
        ocr_provider = ocr_provider or default_ocr
        normalizer = normalizer or default_normalizer

    from .auth_store import get_store

    # PDF uploads are rasterized page-by-page for recognition; the original
    # PDF is what gets stored, and the dedup hash is computed on it.
    from .pdf_support import is_pdf_bytes, pdf_to_pages

    raw_content = content
    pdf_page_count = 0
    # Per-page JPEG payloads for multi-page PDFs; empty for image uploads.
    page_jpegs: list[bytes] = []
    if is_pdf_bytes(content):
        images, pdf_page_count = pdf_to_pages(content)
        if not images:
            raise ValueError("PDF 无法解析或已加密，请换用图片格式上传")
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            page_jpegs.append(buf.getvalue())
        content = page_jpegs[0]  # single-page fallback uses the first page

    digest = sha256(raw_content).hexdigest()
    artifact_id = f"art_{uuid4().hex[:12]}"
    store = get_store()

    # C1 dedup: the same file (same content hash) uploaded before is
    # reported to the user; storage and recognition still run (the earlier
    # artifact keeps its own file so deleting one event can't orphan the
    # other's evidence).
    duplicate_of: dict | None = None
    existing = store.find_artifact_by_sha256(patient_id, digest)
    if existing is not None and existing.artifact_id != artifact_id:
        duplicate_of = {
            "artifact_id": existing.artifact_id,
            "file_name": existing.file_name,
            "created_at": existing.created_at,
        }

    saved = store.save_artifact_file(
        patient_id=patient_id,
        artifact_id=artifact_id,
        file_name=file_name,
        media_type=media_type,
        content=raw_content,
        sha256=digest,
    )

    artifact = SourceArtifact(
        artifact_id=artifact_id,
        patient_id=patient_id,
        file_name=saved.file_name,
        media_type=media_type,
        sha256=digest,
        storage_uri=saved.storage_path,
    )

    # Pure-image path: the quality gate classified this as a film/imaging
    # shot (dark background, no report text to read). Save the original as
    # evidence and emit an imaging event with no facts instead of running
    # OCR — OCR on a film photo risks mis-classified "laboratory" garbage
    # and buys nothing when there is no text.
    if doc_type_hint == "imaging":
        artifact.document_type = "imaging"
        event = ClinicalEvent(
            event_id=f"evt_{uuid4().hex[:10]}",
            patient_id=patient_id,
            event_type="imaging",
            title=Path(file_name).stem or "影像记录",
            event_time=EventTime(precision="unknown", source="document"),
            source_artifact_ids=[artifact.artifact_id],
            facts=[],
            summary="纯影像原图，未识别到文字",
        )
        return IngestionResult(
            artifact=artifact,
            status="needs_review",
            pipeline_steps=[
                "upload_complete",
                "sha256_dedup_key_created",
                "source_file_saved",
                "imaging_no_ocr",
                *(["pdf_rasterized"] if pdf_page_count else []),
            ],
            extracted_event=event,
            duplicate_of=duplicate_of,
            pdf_page_count=pdf_page_count,
        )

    # Multi-page PDFs: OCR every page, concatenate text and blocks. Block
    # coordinates are per-page; we only use blocks for text-based evidence
    # lookup, so cross-page coordinates don't matter.
    if page_jpegs:
        ocr_texts: list[str] = []
        ocr_blocks_raw: list = []
        ocr_engine = ""
        for jpg in page_jpegs:
            o = await ocr_provider.recognize(jpg, media_type)
            ocr_engine = o.engine
            ocr_texts.append(o.text)
            ocr_blocks_raw.extend(o.blocks)
        ocr = OCRResult(text="\n\n".join(ocr_texts), blocks=ocr_blocks_raw, engine=ocr_engine)
    else:
        ocr = await ocr_provider.recognize(content, media_type)
    artifact.ocr_engine = ocr.engine
    artifact.ocr_text = ocr.text
    artifact.ocr_blocks = [
        OCRBlock.model_validate(block) for block in ocr.blocks
    ]

    # Vision normalizers read the raw image directly (bypass OCR text);
    # text normalizers consume the OCR result. For vision models, also
    # hand over the reconstructed table grid (if the layout is tabular) so
    # multi-column tables are extracted per-cell instead of scrambled.
    table_text: str | None = None
    grid_matrix: list[list[str]] | None = None
    grid_low_conf: list[list[bool]] | None = None
    grid: object = None
    event: ClinicalEvent | None = None
    llm_error: str | None = None
    if hasattr(normalizer, "normalize_image"):
        # Grid reconstruction only for single-page uploads: multi-page PDFs
        # concatenate OCR blocks, which would make reconstruct_grid slice
        # across page boundaries.
        if not page_jpegs:
            try:
                from .table_grid import reconstruct_grid

                grid = reconstruct_grid(artifact.ocr_blocks)
                if grid is not None:
                    table_text = grid.to_text()
                    grid_matrix = grid.to_matrix()
                    grid_low_conf = grid.low_confidence_matrix()
            except Exception:  # noqa: BLE001 - grid is advisory
                table_text = None
        # GLM failures degrade to "OCR text only" instead of failing the
        # whole upload. Multi-page PDFs are recognized page by page: a page
        # that errors does not kill the other pages, facts/narrative are
        # merged into one event.
        page_errors: list[str] = []
        pages = page_jpegs or [content]
        for idx, jpg in enumerate(pages):
            try:
                ev = await normalizer.normalize_image(
                    patient_id,
                    artifact,
                    jpg,
                    table_text=table_text if idx == 0 else None,
                    require_correction=grid_matrix is not None and idx == 0,
                )
                if ev is None:
                    continue
                if event is None:
                    event = ev
                    # Tabular layouts ship the grid to the frontend as a third
                    # review mode. Prefer the GLM-corrected matrix; fall back
                    # to the raw OCR matrix.
                    if event.grid is not None and grid is not None:
                        event.grid = grid.apply_corrections(event.grid) or grid_matrix
                        event.grid_low_conf = grid_low_conf
                else:
                    event.facts.extend(ev.facts)
                    if ev.narrative:
                        event.narrative = (event.narrative or "") + "\n\n" + ev.narrative
            except Exception as exc:  # noqa: BLE001 - one bad page, keep going
                page_errors.append(f"第{idx + 1}页: {exc}")
        if event is None and page_errors:
            llm_error = "; ".join(page_errors)[:300]
    else:
        try:
            event = await normalizer.normalize(patient_id, artifact, ocr)
        except Exception as exc:  # noqa: BLE001 - degrade to OCR text
            llm_error = str(exc)[:300]

    # OCR x AI cross-check: compare the AI-extracted facts against the raw
    # OCR text so the user sees which items the two readings agree on and
    # which ones need manual verification.
    from .cross_check import (
        cross_check_facts,
        cross_check_grid,
        locate_evidence,
        medication_missing_fields,
    )

    cross_check: list[dict] = []
    if event is not None and event.facts:
        # Point each fact at the OCR block that carries its concept, so the
        # evidence field stops being empty and the source region is locatable.
        locate_evidence(event.facts, artifact.ocr_blocks, artifact.artifact_id)
        cross_check = cross_check_facts(ocr.text, event.facts)
        # Medication rows routinely lose dose/frequency/route while keeping
        # the drug name; surface that as a pending check instead of silence.
        cross_check.extend(medication_missing_fields(event.facts))

    grid_cross_check: list[dict] = []
    if event is not None and event.grid and grid_matrix:
        grid_cross_check = cross_check_grid(grid_matrix, event.grid)

    steps = [
        "upload_complete",
        "sha256_dedup_key_created",
        "source_file_saved",
        "ocr_complete",
        *(["pdf_rasterized"] if pdf_page_count else []),
    ]
    status: str = "ocr_complete"

    if event is not None:
        steps.extend(
            [
                "document_classification_complete",
                "llm_normalization_complete",
            ]
        )
        status = "needs_review"
        artifact.document_type = event.event_type
        artifact.document_time = event.event_time
        artifact.institution = event.institution
    elif llm_error:
        # Recognition failed but OCR text is intact: flag it so the user
        # can read the raw text and retry instead of seeing a bare error.
        steps.extend(
            [
                "document_classification_pending",
                "llm_normalization_failed",
            ]
        )
        status = "needs_review"
    else:
        steps.extend(
            [
                "document_classification_pending",
                "llm_normalization_pending",
            ]
        )
        status = "needs_review"

    return IngestionResult(
        artifact=artifact,
        status=status,  # type: ignore[arg-type]
        pipeline_steps=steps,
        extracted_event=event,
        cross_check=cross_check,
        grid_cross_check=grid_cross_check,
        duplicate_of=duplicate_of,
        llm_error=llm_error,
        pdf_page_count=pdf_page_count,
    )
