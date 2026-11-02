from __future__ import annotations

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

    digest = sha256(content).hexdigest()
    artifact_id = f"art_{uuid4().hex[:12]}"
    store = get_store()
    saved = store.save_artifact_file(
        patient_id=patient_id,
        artifact_id=artifact_id,
        file_name=file_name,
        media_type=media_type,
        content=content,
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
            ],
            extracted_event=event,
        )

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
    if hasattr(normalizer, "normalize_image"):
        try:
            from .table_grid import reconstruct_grid

            grid = reconstruct_grid(artifact.ocr_blocks)
            if grid is not None:
                table_text = grid.to_text()
                grid_matrix = grid.to_matrix()
        except Exception:  # noqa: BLE001 - grid is advisory
            table_text = None
        event = await normalizer.normalize_image(
            patient_id, artifact, content, table_text=table_text
        )
        # Tabular layouts ship the grid to the frontend as a third review
        # mode (分条 / 表格 / 全部) alongside facts and narrative.
        if event is not None and grid_matrix:
            event.grid = grid_matrix
    else:
        event = await normalizer.normalize(patient_id, artifact, ocr)

    # OCR x AI cross-check: compare the AI-extracted facts against the raw
    # OCR text so the user sees which items the two readings agree on and
    # which ones need manual verification.
    from .cross_check import cross_check_facts

    cross_check: list[dict] = []
    if event is not None and event.facts:
        cross_check = cross_check_facts(ocr.text, event.facts)

    steps = [
        "upload_complete",
        "sha256_dedup_key_created",
        "source_file_saved",
        "ocr_complete",
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
    )
