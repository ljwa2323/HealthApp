from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TimePrecision(str, Enum):
    minute = "minute"
    hour = "hour"
    day = "day"
    month = "month"
    year = "year"
    interval = "interval"
    unknown = "unknown"


class EventTime(BaseModel):
    start: datetime | date | str | None = None
    end: datetime | date | str | None = None
    precision: TimePrecision = TimePrecision.unknown
    source: Literal["document", "patient", "system", "inferred"] = "document"


class Concept(BaseModel):
    display: str
    system: str | None = None
    code: str | None = None
    local_display: str | None = None
    local_code: str | None = None


class EvidencePointer(BaseModel):
    artifact_id: str
    page: int | None = None
    block_id: str | None = None
    quote: str | None = None


class QuantityValue(BaseModel):
    type: Literal["quantity"] = "quantity"
    number: float
    unit: str | None = None


class TextValue(BaseModel):
    type: Literal["text"] = "text"
    text: str


class CodeValue(BaseModel):
    type: Literal["code"] = "code"
    code: str
    display: str | None = None
    system: str | None = None


FactValue = QuantityValue | TextValue | CodeValue


class ReferenceRange(BaseModel):
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    text: str | None = None


class ClinicalFact(BaseModel):
    fact_id: str
    kind: Literal[
        "observation",
        "condition",
        "medication",
        "procedure",
        "allergy",
        "symptom",
        "note",
    ]
    concept: Concept
    value: FactValue | None = None
    reference_range: ReferenceRange | None = None
    interpretation: Literal["low", "normal", "high", "abnormal", "unknown"] | None = None
    time: EventTime
    evidence: list[EvidencePointer] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: Literal["extracted", "confirmed", "corrected", "rejected"] = "extracted"
    attributes: dict[str, Any] = Field(default_factory=dict)


class ClinicalEvent(BaseModel):
    event_id: str
    patient_id: str
    event_type: Literal[
        "laboratory",
        "imaging",
        "visit",
        "hospitalization",
        "procedure",
        "medication",
        "vital",
        "symptom",
        "other",
    ]
    title: str
    event_time: EventTime
    institution: str | None = None
    source_artifact_ids: list[str] = Field(default_factory=list)
    facts: list[ClinicalFact] = Field(default_factory=list)
    summary: str | None = None


class SourceArtifact(BaseModel):
    artifact_id: str
    patient_id: str
    file_name: str
    media_type: str | None = None
    document_type: str | None = None
    document_time: EventTime | None = None
    institution: str | None = None
    sha256: str | None = None
    storage_uri: str | None = None
    ocr_text: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Patient(BaseModel):
    patient_id: str
    display_name: str
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    birth_date: date | None = None


class DerivedInsight(BaseModel):
    insight_id: str
    patient_id: str
    kind: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model: str
    input_fact_ids: list[str]
    text: str
    confidence: float | None = None


class IngestionResult(BaseModel):
    artifact: SourceArtifact
    status: Literal["uploaded", "ocr_complete", "needs_review", "ready"]
    pipeline_steps: list[str]
    extracted_event: ClinicalEvent | None = None


class AgentContext(BaseModel):
    task: str
    patient_id: str
    time_range: dict[str, str | None]
    observations: list[dict[str, Any]]
    medications: list[dict[str, Any]]
    conditions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    data_gaps: list[str]
    evidence_refs: list[str]
