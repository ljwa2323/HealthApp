from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from ..schemas import ClinicalEvent, IngestionResult, OCRBlock, SourceArtifact


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
) -> IngestionResult:
    ocr_provider = ocr_provider or DemoOCRProvider()
    normalizer = normalizer or ClinicalNormalizer()

    digest = sha256(content).hexdigest()
    artifact_id = f"art_{uuid4().hex[:12]}"

    artifact = SourceArtifact(
        artifact_id=artifact_id,
        patient_id=patient_id,
        file_name=file_name,
        media_type=media_type,
        sha256=digest,
        storage_uri=f"local://uploads/{artifact_id}/{file_name}",
    )

    ocr = await ocr_provider.recognize(content, media_type)
    artifact.ocr_engine = ocr.engine
    artifact.ocr_text = ocr.text
    artifact.ocr_blocks = [
        OCRBlock.model_validate(block) for block in ocr.blocks
    ]

    event = await normalizer.normalize(patient_id, artifact, ocr)

    steps = [
        "upload_complete",
        "sha256_dedup_key_created",
        "ocr_complete",
        "document_classification_pending",
        "llm_normalization_pending",
    ]

    return IngestionResult(
        artifact=artifact,
        status="needs_review",
        pipeline_steps=steps,
        extracted_event=event,
    )
