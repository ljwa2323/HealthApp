from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import get_current_user, require_patient_access
from .config import get_nvidia_api_key
from .schemas import (
    AuthResponse,
    AuthUser,
    AssistantAnalyzeRequest,
    AssistantChatRequest,
    AssistantReply,
    ChangePasswordRequest,
    ClinicalEvent,
    EventCorrectionRequest,
    LoginRequest,
    Patient,
    RegisterRequest,
)
from .services.auth_store import UserRecord, get_store
from .services.context_builder import build_agent_context
from .services.health_agent import analyze_health, answer_question
from .services.ingestion import ingest_document

app = FastAPI(
    title="HealthApp API",
    version="0.2.0",
    description="Longitudinal patient-owned health record API with user registration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

optional_bearer = HTTPBearer(auto_error=False)


def _to_auth_user(user: UserRecord) -> AuthUser:
    return AuthUser(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        patient_id=user.patient_id,
        sex=user.sex,
    )


@app.on_event("startup")
def startup_banner() -> None:
    get_store()
    key = get_nvidia_api_key()
    if key:
        print("NVIDIA providers enabled (OCR + LLM).")
    else:
        print("NVIDIA_API_KEY missing; falling back to demo OCR/normalizer.")
    print("User registration auth enabled.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "nvidia_configured": bool(get_nvidia_api_key()),
        "auth": "registration",
    }


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    store = get_store()
    try:
        user = store.register(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
            sex=payload.sex,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = store.create_session(user.user_id)
    return AuthResponse(token=token, user=_to_auth_user(user))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    store = get_store()
    try:
        user = store.authenticate(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token = store.create_session(user.user_id)
    return AuthResponse(token=token, user=_to_auth_user(user))


@app.post("/api/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
) -> dict:
    if credentials and credentials.credentials:
        get_store().delete_session(credentials.credentials)
    return {"ok": True}


@app.post("/api/auth/change-password")
def change_password(
    payload: ChangePasswordRequest,
    user: UserRecord = Depends(get_current_user),
) -> dict:
    try:
        get_store().change_password(
            user_id=user.user_id,
            current_password=payload.current_password,
            new_password=payload.new_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/auth/me", response_model=AuthUser)
def me(user: UserRecord = Depends(get_current_user)) -> AuthUser:
    return _to_auth_user(user)


@app.get("/api/patients/{patient_id}", response_model=Patient)
def get_patient(
    patient_id: str,
    user: UserRecord = Depends(require_patient_access),
) -> Patient:
    return get_store().to_patient(user)


@app.get("/api/patients/{patient_id}/timeline", response_model=list[ClinicalEvent])
def get_timeline(
    patient_id: str,
    user: UserRecord = Depends(require_patient_access),
) -> list[ClinicalEvent]:
    return get_store().list_events(user.patient_id)


@app.post("/api/patients/{patient_id}/timeline", response_model=ClinicalEvent)
def add_event(
    patient_id: str,
    event: ClinicalEvent,
    user: UserRecord = Depends(require_patient_access),
) -> ClinicalEvent:
    if event.patient_id != user.patient_id:
        raise HTTPException(status_code=400, detail="patient_id mismatch")
    return get_store().add_event(event)


@app.put(
    "/api/patients/{patient_id}/timeline/{event_id}",
    response_model=ClinicalEvent,
)
def update_event(
    patient_id: str,
    event_id: str,
    payload: EventCorrectionRequest,
    user: UserRecord = Depends(require_patient_access),
) -> ClinicalEvent:
    store = get_store()
    existing = store.get_event(user.patient_id, event_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Event not found")

    data = existing.model_dump(mode="json")
    if payload.title is not None:
        data["title"] = payload.title.strip() or data["title"]
    if payload.summary is not None:
        data["summary"] = payload.summary
    if payload.institution is not None:
        data["institution"] = payload.institution
    if payload.event_time_start is not None:
        data["event_time"]["start"] = payload.event_time_start or None
        if payload.event_time_start:
            data["event_time"]["precision"] = "day"
            data["event_time"]["source"] = "patient"

    if payload.facts is not None:
        fact_map = {fact["fact_id"]: fact for fact in data.get("facts", [])}
        for correction in payload.facts:
            fact_id = correction.get("fact_id")
            if not fact_id or fact_id not in fact_map:
                continue
            target = fact_map[fact_id]
            if correction.get("label"):
                target["concept"]["display"] = correction["label"]
                target["concept"]["local_display"] = correction["label"]
            if "value_text" in correction and correction["value_text"] is not None:
                value_text = str(correction["value_text"]).strip()
                number = None
                unit = None
                # Prefer "88 umol/L" style values.
                parts = value_text.split(maxsplit=1)
                try:
                    number = float(parts[0])
                    unit = parts[1] if len(parts) > 1 else None
                    target["value"] = {
                        "type": "quantity",
                        "number": number,
                        "unit": unit,
                    }
                except ValueError:
                    target["value"] = {"type": "text", "text": value_text}
            if correction.get("interpretation"):
                target["interpretation"] = correction["interpretation"]
            target["status"] = correction.get("status") or "corrected"
            target["time"]["source"] = "patient"

    updated = ClinicalEvent.model_validate(data)
    return store.add_event(updated)


@app.delete("/api/patients/{patient_id}/timeline/{event_id}")
def delete_event(
    patient_id: str,
    event_id: str,
    user: UserRecord = Depends(require_patient_access),
) -> dict:
    deleted = get_store().delete_event(user.patient_id, event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True, "event_id": event_id}


@app.post("/api/patients/{patient_id}/documents")
async def upload_document(
    patient_id: str,
    file: UploadFile = File(...),
    user: UserRecord = Depends(require_patient_access),
):
    content = await file.read()
    try:
        result = await ingest_document(
            patient_id=user.patient_id,
            file_name=file.filename or "upload",
            media_type=file.content_type,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

    if result.extracted_event is not None:
        get_store().add_event(result.extracted_event)
    return result


@app.get("/api/patients/{patient_id}/artifacts/{artifact_id}")
def get_artifact_file(
    patient_id: str,
    artifact_id: str,
    user: UserRecord = Depends(require_patient_access),
):
    artifact = get_store().get_artifact(user.patient_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Original report not found")
    path = Path(artifact.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original report file missing on disk")
    return FileResponse(
        path=path,
        media_type=artifact.media_type or "application/octet-stream",
        filename=artifact.file_name,
    )


@app.get("/api/patients/{patient_id}/agent-context")
def agent_context(
    patient_id: str,
    task: str = Query(default="general_longitudinal_review"),
    start: str | None = None,
    end: str | None = None,
    user: UserRecord = Depends(require_patient_access),
):
    events = get_store().list_events(user.patient_id)
    return build_agent_context(
        patient_id=user.patient_id,
        task=task,
        events=events,
        start=start,
        end=end,
    )


@app.post(
    "/api/patients/{patient_id}/assistant/analyze",
    response_model=AssistantReply,
)
async def assistant_analyze(
    patient_id: str,
    payload: AssistantAnalyzeRequest,
    user: UserRecord = Depends(require_patient_access),
) -> AssistantReply:
    events = get_store().list_events(user.patient_id)
    if not events:
        raise HTTPException(
            status_code=400,
            detail="No timeline records available for analysis",
        )
    context = build_agent_context(
        patient_id=user.patient_id,
        task=payload.task,
        events=events,
    )
    try:
        result = await analyze_health(context)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AssistantReply(
        answer=result["answer"],
        evidence_refs=result.get("evidence_refs") or [],
        used_fact_ids=[item for item in (result.get("used_fact_ids") or []) if item],
        context=result.get("context"),
    )


@app.post(
    "/api/patients/{patient_id}/assistant/chat",
    response_model=AssistantReply,
)
async def assistant_chat(
    patient_id: str,
    payload: AssistantChatRequest,
    user: UserRecord = Depends(require_patient_access),
) -> AssistantReply:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    events = get_store().list_events(user.patient_id)
    if not events:
        raise HTTPException(
            status_code=400,
            detail="No timeline records available for chat",
        )
    context = build_agent_context(
        patient_id=user.patient_id,
        task=payload.task,
        events=events,
    )
    try:
        result = await answer_question(
            context=context,
            question=question,
            history=[item.model_dump() for item in payload.history],
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AssistantReply(
        answer=result["answer"],
        evidence_refs=result.get("evidence_refs") or [],
        used_fact_ids=[],
        context=result.get("context"),
    )


@app.post("/api/dev/load-demo")
def load_demo(user: UserRecord = Depends(get_current_user)) -> dict:
    """Load the repository example into the current user's timeline."""
    candidates = [
        Path("examples/patient_record.json"),
        Path("../../examples/patient_record.json"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Demo file not found")

    payload = json.loads(path.read_text(encoding="utf-8"))
    events: list[ClinicalEvent] = []
    for item in payload["events"]:
        item = dict(item)
        item["patient_id"] = user.patient_id
        events.append(ClinicalEvent.model_validate(item))
    loaded = get_store().replace_events(user.patient_id, events)
    return {"loaded": loaded, "patient_id": user.patient_id}
