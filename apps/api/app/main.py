from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import get_current_user, require_patient_access
from .config import get_deepseek_api_key, get_glm_api_key, get_nvidia_api_key
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
from .services.health_assistant import answer_question, answer_question_stream
from .services.ingestion import ingest_document
from .services.intent_router import route_question

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
    if get_nvidia_api_key():
        print("NVIDIA providers enabled (OCR + LLM).")
    elif get_glm_api_key():
        print("GLM vision normalizer enabled (RapidOCR + GLM vision).")
    elif get_deepseek_api_key():
        print("DeepSeek normalizer enabled (RapidOCR + DeepSeek LLM).")
    else:
        print("No cloud key: rule-based extraction enabled.")
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
    from .services.image_quality import analyze_image_quality

    # First pass: cheap heuristic split (text vs imaging) so bad photos
    # are flagged before OCR. After ingestion the classifier knows the
    # real document type; if it differs from the heuristic, re-run the
    # gate with the authoritative type so issues match the document.
    quality = analyze_image_quality(content)
    try:
        result = await ingest_document(
            patient_id=user.patient_id,
            file_name=file.filename or "upload",
            media_type=file.content_type,
            content=content,
            # Heuristic split from the quality gate: an "imaging" verdict
            # short-circuits OCR (pure film shot, nothing to read).
            doc_type_hint=quality.doc_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

    # Map the classified event type onto the quality-gate branches.
    # Only an explicit "imaging" classification switches the gate to the
    # imaging checks. Any other outcome (laboratory, unclassified, or a
    # mis-classified film photo) keeps the heuristic split — overwriting
    # an auto-"imaging" verdict with the text checks would re-introduce
    # false "too dark" warnings for dark film photos.
    classified_type = (result.artifact.document_type or "").lower()
    expected = "imaging" if classified_type == "imaging" else quality.doc_type
    if quality.doc_type != expected:
        quality = analyze_image_quality(content, doc_type=expected)

    result.quality = {
        "ok": quality.ok,
        "issues": quality.issues,
        "metrics": quality.metrics,
        "doc_type": quality.doc_type,
    }
    # The event is NOT written to the timeline here: the patient reviews the
    # extracted fields first and explicitly adds the event via
    # POST /api/patients/{patient_id}/timeline.
    return result


@app.post("/api/patients/{patient_id}/documents/{artifact_id}/split")
async def split_document(
    patient_id: str,
    artifact_id: str,
    user: UserRecord = Depends(require_patient_access),
):
    """On-demand splitting.

    At upload time narrative documents (discharge summaries, progress
    notes…) are kept as raw text without pre-splitting, so the initial
    extraction stays fast. When the user taps 分条展示, this endpoint
    re-reads the saved original and forces structured extraction.
    """
    artifact = get_store().get_artifact(user.patient_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(artifact.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original file missing on disk")
    content = path.read_bytes()

    from .schemas import SourceArtifact
    from .services.cross_check import cross_check_facts
    from .services.nvidia_providers import GLMVisionNormalizer
    from .services.rapid_ocr_provider import RapidOCRProvider
    from .services.table_grid import reconstruct_grid

    source = SourceArtifact(
        artifact_id=artifact.artifact_id,
        patient_id=user.patient_id,
        file_name=artifact.file_name,
        media_type=artifact.media_type,
        storage_uri=artifact.storage_path,
    )
    # Re-run the local OCR (fast; OCR text is not persisted on the
    # artifact row) to rebuild the table grid and cross-check.
    table_text: str | None = None
    grid_matrix: list[list[str]] | None = None
    grid: object = None
    cross_check: list[dict] = []
    grid_cross_check: list[dict] = []
    ocr = None
    try:
        ocr = await RapidOCRProvider().recognize(content, artifact.media_type)
        grid = reconstruct_grid(
            [
                {
                    "text": b["text"],
                    "bbox": b["bbox"],
                    "confidence": b.get("confidence"),
                }
                for b in ocr.blocks
            ]
        )
        if grid is not None:
            table_text = grid.to_text()
            grid_matrix = grid.to_matrix()
    except Exception:  # noqa: BLE001 - grid/OCR are advisory
        pass

    try:
        event = await GLMVisionNormalizer().normalize_image(
            user.patient_id,
            source,
            content,
            force_structured=True,
            table_text=table_text,
            require_correction=grid_matrix is not None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Split failed: {exc}") from exc
    if event is None:
        raise HTTPException(status_code=502, detail="Split failed: no result")
    if grid is not None:
        event.grid = grid.apply_corrections(event.grid) or grid_matrix

    if ocr is not None:
        try:
            from .services.cross_check import (
                cross_check_facts,
                cross_check_grid,
                locate_evidence,
                medication_missing_fields,
            )

            # Same evidence pointers as the first ingestion pass, so a
            # re-split document keeps its facts traceable to OCR blocks.
            if event.facts:
                locate_evidence(event.facts, ocr.blocks, artifact.artifact_id)
            cross_check = cross_check_facts(ocr.text, event.facts)
            cross_check.extend(medication_missing_fields(event.facts))
            if event.grid and grid_matrix:
                grid_cross_check = cross_check_grid(grid_matrix, event.grid)
        except Exception:  # noqa: BLE001 - cross-check is advisory only
            cross_check = []
            grid_cross_check = []

    return {
        "event": event,
        "facts": event.facts,
        "cross_check": cross_check,
        "grid_cross_check": grid_cross_check,
    }


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
    try:
        result = await answer_question(
            question="请基于我的时间轴做一次整体健康回顾。",
            events=events,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AssistantReply(
        answer=result["answer"],
        evidence_refs=[],
        used_fact_ids=result.get("used_fact_ids") or [],
        context=result.get("context"),
    )


@app.get("/api/patients/{patient_id}/conversations")
def list_conversations(
    patient_id: str,
    q: str = "",
    user: UserRecord = Depends(require_patient_access),
):
    return {"conversations": get_store().list_conversations(user.patient_id, search=q)}


@app.post("/api/patients/{patient_id}/conversations")
def create_conversation(
    patient_id: str,
    payload: dict,
    user: UserRecord = Depends(require_patient_access),
):
    title = (payload.get("title") or "新对话").strip()[:80]
    return get_store().create_conversation(user.patient_id, title)


@app.get("/api/patients/{patient_id}/conversations/{conversation_id}")
def get_conversation(
    patient_id: str,
    conversation_id: str,
    user: UserRecord = Depends(require_patient_access),
):
    conv = get_store().get_conversation(user.patient_id, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.patch("/api/patients/{patient_id}/conversations/{conversation_id}")
def rename_conversation(
    patient_id: str,
    conversation_id: str,
    payload: dict,
    user: UserRecord = Depends(require_patient_access),
):
    title = (payload.get("title") or "").strip()[:80]
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    ok = get_store().rename_conversation(user.patient_id, conversation_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@app.delete("/api/patients/{patient_id}/conversations/{conversation_id}")
def delete_conversation(
    patient_id: str,
    conversation_id: str,
    user: UserRecord = Depends(require_patient_access),
):
    ok = get_store().delete_conversation(user.patient_id, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


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

    # 意图路由先行：拿时间窗用于 SQL 下推，同时传给回答服务避免二次路由
    route = await route_question(question)
    effective_start = (
        route.time_start - timedelta(days=7) if route.time_start else None
    )
    events = get_store().list_events(
        user.patient_id,
        time_start=effective_start,
        time_end=route.time_end,
    )
    if not events:
        # 时间窗内无数据：回退全量并清空窗口，让 LLM 说明"该时间窗无记录"
        events = get_store().list_events(user.patient_id)
        effective_start = None
        route = replace(route, time_start=None)
    if not events:
        raise HTTPException(
            status_code=400,
            detail="No timeline records available for chat",
        )

    store = get_store()
    conv_id = payload.conversation_id
    if conv_id:
        conv = store.get_conversation(user.patient_id, conv_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # 自动建对话，标题取问题前 20 字
        conv = store.create_conversation(user.patient_id, question[:20])
        conv_id = conv["conversation_id"]

    store.append_message(user.patient_id, conv_id, "user", question)

    try:
        result = await answer_question(
            question=question,
            events=events,
            history=[item.model_dump() for item in payload.history],
            route=replace(route, time_start=effective_start),
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    store.append_message(user.patient_id, conv_id, "assistant", result["answer"])

    return AssistantReply(
        answer=result["answer"],
        evidence_refs=[],
        used_fact_ids=result.get("used_fact_ids") or [],
        context=result.get("context"),
        conversation_id=conv_id,
    )


@app.post("/api/patients/{patient_id}/assistant/chat/stream")
async def assistant_chat_stream(
    patient_id: str,
    payload: AssistantChatRequest,
    user: UserRecord = Depends(require_patient_access),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # 意图路由先行：拿时间窗用于 SQL 下推，同时传给回答服务避免二次路由
    route = await route_question(question)
    effective_start = (
        route.time_start - timedelta(days=7) if route.time_start else None
    )
    events = get_store().list_events(
        user.patient_id,
        time_start=effective_start,
        time_end=route.time_end,
    )
    if not events:
        # 时间窗内无数据：回退全量并清空窗口，让 LLM 说明"该时间窗无记录"
        events = get_store().list_events(user.patient_id)
        effective_start = None
        route = replace(route, time_start=None)
    if not events:
        raise HTTPException(
            status_code=400,
            detail="No timeline records available for chat",
        )

    store = get_store()
    conv_id = payload.conversation_id
    if conv_id:
        conv = store.get_conversation(user.patient_id, conv_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = store.create_conversation(user.patient_id, question[:20])
        conv_id = conv["conversation_id"]

    store.append_message(user.patient_id, conv_id, "user", question)
    collected: list[str] = []

    async def event_generator():
        try:
            async for piece in answer_question_stream(
                question=question,
                events=events,
                history=[item.model_dump() for item in payload.history],
                route=replace(route, time_start=effective_start),
            ):
                if piece.get("type") == "chunk":
                    collected.append(piece.get("text", ""))
                if piece.get("type") == "meta":
                    piece["conversation_id"] = conv_id
                yield f"data: {json.dumps(piece, ensure_ascii=False)}\n\n"
            full = "".join(collected)
            if full:
                store.append_message(user.patient_id, conv_id, "assistant", full)
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
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
