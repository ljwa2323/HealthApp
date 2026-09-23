from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ClinicalEvent, Patient
from .services.context_builder import build_agent_context
from .services.ingestion import ingest_document

app = FastAPI(
    title="HealthApp API",
    version="0.1.0",
    description="Longitudinal patient-owned health record API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_PATIENT = Patient(
    patient_id="p_demo_001",
    display_name="Demo Patient",
    sex="female",
)

EVENTS: list[ClinicalEvent] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: str) -> Patient:
    if patient_id != DEMO_PATIENT.patient_id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return DEMO_PATIENT


@app.get("/api/patients/{patient_id}/timeline", response_model=list[ClinicalEvent])
def get_timeline(patient_id: str) -> list[ClinicalEvent]:
    return sorted(
        [event for event in EVENTS if event.patient_id == patient_id],
        key=lambda event: str(event.event_time.start or ""),
        reverse=True,
    )


@app.post("/api/patients/{patient_id}/timeline", response_model=ClinicalEvent)
def add_event(patient_id: str, event: ClinicalEvent) -> ClinicalEvent:
    if event.patient_id != patient_id:
        raise HTTPException(status_code=400, detail="patient_id mismatch")
    EVENTS.append(event)
    return event


@app.post("/api/patients/{patient_id}/documents")
async def upload_document(
    patient_id: str,
    file: UploadFile = File(...),
):
    content = await file.read()
    return await ingest_document(
        patient_id=patient_id,
        file_name=file.filename or "upload",
        media_type=file.content_type,
        content=content,
    )


@app.get("/api/patients/{patient_id}/agent-context")
def agent_context(
    patient_id: str,
    task: str = Query(default="general_longitudinal_review"),
    start: str | None = None,
    end: str | None = None,
):
    return build_agent_context(
        patient_id=patient_id,
        task=task,
        events=EVENTS,
        start=start,
        end=end,
    )


@app.post("/api/dev/load-demo")
def load_demo() -> dict:
    """Load the repository example file when running from the repo root."""
    global EVENTS

    candidates = [
        Path("examples/patient_record.json"),
        Path("../../examples/patient_record.json"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Demo file not found")

    payload = json.loads(path.read_text(encoding="utf-8"))
    EVENTS = [ClinicalEvent.model_validate(item) for item in payload["events"]]
    return {"loaded": len(EVENTS)}
