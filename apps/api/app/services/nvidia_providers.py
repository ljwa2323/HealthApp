from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any
from uuid import uuid4

import httpx
from openai import APIStatusError, AsyncOpenAI
from PIL import Image, UnidentifiedImageError

from ..config import (
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_INLINE_B64_CHARS,
    NVCF_ASSETS_URL,
    OCR_INVOKE_URL,
    get_nvidia_api_key,
)
from ..schemas import ClinicalEvent, SourceArtifact
from .ingestion import ClinicalNormalizer, OCRProvider, OCRResult

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _prepare_ocr_image(content: bytes, media_type: str | None) -> tuple[bytes, str]:
    """Convert uploads into a JPEG that fits NVIDIA inline OCR size limits."""
    del media_type
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Unsupported or corrupted image. Please upload PNG/JPEG."
        ) from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prefer a moderate resolution: faster OCR and still readable for reports.
    max_side = 1600
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )

    # Quick path: try a few qualities without expensive optimize passes.
    for quality in (72, 60, 48, 36):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False)
        candidate = buffer.getvalue()
        if len(base64.b64encode(candidate)) < MAX_INLINE_B64_CHARS:
            return candidate, "image/jpeg"
        # Shrink once and continue if still too large.
        width, height = image.size
        image = image.resize(
            (max(1, int(width * 0.75)), max(1, int(height * 0.75))),
            Image.Resampling.BILINEAR,
        )

    raise ValueError(
        "Image is still too large after compression for NVIDIA OCR. "
        "Please crop or upload a smaller photo."
    )


def _bbox_from_points(points: list[dict[str, float]] | None) -> list[float] | None:
    if not points:
        return None
    xs = [float(p["x"]) for p in points if "x" in p]
    ys = [float(p["y"]) for p in points if "y" in p]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _strip_model_wrappers(text: str) -> str:
    cleaned = text.strip()
    # Drop accidental reasoning / think wrappers if the model emits them.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _balanced_json_slice(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair_common_json_issues(text: str) -> str:
    repaired = text
    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Replace smart quotes
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return repaired


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_model_wrappers(text)
    candidates = [cleaned]
    sliced = _balanced_json_slice(cleaned)
    if sliced:
        candidates.append(sliced)
        candidates.append(_repair_common_json_issues(sliced))
    candidates.append(_repair_common_json_issues(cleaned))

    errors: list[str] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
            errors.append("JSON root is not an object")
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    preview = cleaned[:280].replace("\n", "\\n")
    raise ValueError(
        "LLM returned invalid JSON "
        f"({errors[-1] if errors else 'unknown'}). Preview: {preview}"
    )


def _http_error_detail(response: httpx.Response) -> str:
    body = (response.text or "").strip()
    if not body:
        return f"HTTP {response.status_code}"
    if len(body) > 800:
        body = body[:800] + "..."
    return f"HTTP {response.status_code}: {body}"


async def _sleep_backoff(attempt: int) -> None:
    await asyncio.sleep(min(2 ** attempt, 12))


class NvidiaOCRProvider(OCRProvider):
    engine_name = "nvidia/nemotron-ocr-v2"

    def __init__(self, api_key: str | None = None, timeout: float = 180.0) -> None:
        self.api_key = api_key or get_nvidia_api_key()
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not configured")
        self.timeout = timeout

    async def recognize(self, content: bytes, media_type: str | None) -> OCRResult:
        if media_type and media_type.startswith("application/pdf"):
            raise ValueError(
                "NVIDIA OCR currently supports images only. "
                "Convert PDF pages to PNG/JPEG before upload."
            )

        prepared_bytes, image_type = _prepare_ocr_image(content, media_type)
        image_b64 = base64.b64encode(prepared_bytes).decode("ascii")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if len(image_b64) < MAX_INLINE_B64_CHARS:
                image_url = f"data:{image_type};base64,{image_b64}"
            else:
                asset_id = await self._upload_asset(client, prepared_bytes, image_type)
                image_url = f"data:{image_type};asset_id,{asset_id}"
                headers["NVCF-INPUT-ASSET-REFERENCES"] = asset_id

            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": image_url,
                    }
                ]
            }
            last_error = "NVIDIA OCR failed"
            for attempt in range(4):
                response = await client.post(
                    OCR_INVOKE_URL,
                    headers=headers,
                    json=payload,
                )
                if response.status_code < 400:
                    return self._parse_ocr_response(response.json())
                last_error = f"NVIDIA OCR failed: {_http_error_detail(response)}"
                if response.status_code not in RETRYABLE_HTTP_STATUS or attempt == 3:
                    raise ValueError(last_error)
                await _sleep_backoff(attempt)

        raise ValueError(last_error)

    async def _upload_asset(
        self,
        client: httpx.AsyncClient,
        content: bytes,
        content_type: str,
    ) -> str:
        create_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        create_body = {
            "contentType": content_type,
            "description": "HealthApp OCR input image",
        }
        created = await client.post(
            NVCF_ASSETS_URL,
            headers=create_headers,
            json=create_body,
        )
        if created.status_code >= 400:
            raise ValueError(f"NVIDIA assets create failed: {_http_error_detail(created)}")
        created_json = created.json()
        asset_id = created_json.get("assetId") or created_json.get("asset_id")
        upload_url = created_json.get("uploadUrl") or created_json.get("upload_url")
        if not asset_id or not upload_url:
            raise RuntimeError("NVIDIA assets API did not return assetId/uploadUrl")

        upload_headers = {
            "Content-Type": content_type,
            "x-amz-meta-nvcf-asset-description": "HealthApp OCR input image",
        }
        uploaded = await client.put(upload_url, headers=upload_headers, content=content)
        if uploaded.status_code >= 400:
            raise ValueError(f"NVIDIA assets upload failed: {_http_error_detail(uploaded)}")
        return str(asset_id)

    def _parse_ocr_response(self, body: dict[str, Any]) -> OCRResult:
        detections: list[dict[str, Any]] = []
        data = body.get("data") or []
        if isinstance(data, list):
            for page in data:
                page_index = int(page.get("index", 0)) + 1
                for det in page.get("text_detections") or []:
                    prediction = det.get("text_prediction") or {}
                    text = (prediction.get("text") or "").strip()
                    if not text:
                        continue
                    confidence = prediction.get("confidence")
                    bbox = _bbox_from_points((det.get("bounding_box") or {}).get("points"))
                    detections.append(
                        {
                            "block_id": f"ocr_block_{len(detections) + 1}",
                            "page": page_index,
                            "text": text,
                            "bbox": bbox,
                            "confidence": float(confidence)
                            if confidence is not None
                            else None,
                        }
                    )

        if not detections and isinstance(body.get("text"), str):
            detections.append(
                {
                    "block_id": "ocr_block_1",
                    "page": 1,
                    "text": body["text"],
                    "bbox": None,
                    "confidence": None,
                }
            )

        text = "\n".join(block["text"] for block in detections).strip()
        return OCRResult(text=text, blocks=detections, engine=self.engine_name)


class NvidiaClinicalNormalizer(ClinicalNormalizer):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or get_nvidia_api_key()
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not configured")
        self.client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=self.api_key,
            timeout=180.0,
            max_retries=0,
        )
        self.model = LLM_MODEL

    async def normalize(
        self,
        patient_id: str,
        artifact: SourceArtifact,
        ocr: OCRResult,
    ) -> ClinicalEvent | None:
        if not ocr.text.strip():
            return None

        schema_hint = {
            "event_id": "evt_...",
            "patient_id": patient_id,
            "event_type": "laboratory|imaging|visit|hospitalization|procedure|medication|vital|symptom|other",
            "title": "short title",
            "event_time": {
                "start": "ISO-8601 datetime or date if known, else null",
                "end": None,
                "precision": "minute|hour|day|month|year|interval|unknown",
                "source": "document",
            },
            "institution": "string or null",
            "source_artifact_ids": [artifact.artifact_id],
            "facts": [
                {
                    "fact_id": "fact_...",
                    "kind": "observation|condition|medication|procedure|allergy|symptom|note",
                    "concept": {
                        "display": "name",
                        "system": None,
                        "code": None,
                        "local_display": "original name",
                        "local_code": None,
                    },
                    "value": {"type": "quantity", "number": 0, "unit": "unit"},
                    "reference_range": {
                        "low": None,
                        "high": None,
                        "unit": None,
                        "text": None,
                    },
                    "interpretation": "low|normal|high|abnormal|unknown",
                    "time": {
                        "start": None,
                        "end": None,
                        "precision": "unknown",
                        "source": "document",
                    },
                    "evidence": [
                        {
                            "artifact_id": artifact.artifact_id,
                            "page": 1,
                            "block_id": "ocr_block_1",
                            "quote": "short quote from OCR",
                        }
                    ],
                    "confidence": 0.0,
                    "status": "extracted",
                    "attributes": {},
                }
            ],
            "summary": "one sentence",
        }

        system_prompt = (
            "You are a medical document structuring assistant for HealthApp. "
            "Convert OCR text from clinical reports into one ClinicalEvent JSON object. "
            "Use only information present in the OCR text. "
            "Do not invent lab values. "
            "Every fact must include evidence with artifact_id and a short quote. "
            "Output a single valid JSON object only. "
            "Do not use markdown fences, comments, trailing commas, or thinking tags."
        )
        user_prompt = (
            f"patient_id: {patient_id}\n"
            f"artifact_id: {artifact.artifact_id}\n"
            f"file_name: {artifact.file_name}\n"
            f"ocr_engine: {ocr.engine}\n\n"
            f"Target JSON shape example:\n{json.dumps(schema_hint, ensure_ascii=True)}\n\n"
            f"OCR text:\n{ocr.text}\n\n"
            f"OCR blocks JSON:\n{json.dumps(ocr.blocks[:80], ensure_ascii=True)}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    top_p=1,
                    max_tokens=8192,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_effort": "none",
                    },
                )
                content = completion.choices[0].message.content or ""
                try:
                    payload = _extract_json_object(content)
                    break
                except ValueError as parse_exc:
                    last_error = parse_exc
                    # Ask the model to repair invalid JSON once or twice.
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Fix the following into one valid JSON object. "
                                "Return JSON only, no markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Parse error: {parse_exc}\n\n"
                                f"Broken text:\n{content}"
                            ),
                        },
                    ]
                    await _sleep_backoff(attempt)
                    continue
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code not in RETRYABLE_HTTP_STATUS or attempt == 4:
                    raise ValueError(
                        f"NVIDIA LLM failed: HTTP {exc.status_code}: {exc.message}"
                    ) from exc
                await _sleep_backoff(attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 4:
                    raise
                await _sleep_backoff(attempt)

        if payload is None:
            if last_error is not None:
                raise ValueError(str(last_error)) from last_error
            raise ValueError("NVIDIA LLM failed after retries")

        payload["patient_id"] = patient_id
        payload["source_artifact_ids"] = [artifact.artifact_id]
        if not payload.get("event_id"):
            payload["event_id"] = f"evt_{uuid4().hex[:12]}"

        for index, fact in enumerate(payload.get("facts") or []):
            if not isinstance(fact, dict):
                continue
            if not fact.get("fact_id"):
                fact["fact_id"] = f"fact_{uuid4().hex[:10]}"
            fact.setdefault("status", "extracted")
            evidence = fact.get("evidence") or []
            if not evidence:
                fact["evidence"] = [
                    {
                        "artifact_id": artifact.artifact_id,
                        "page": 1,
                        "block_id": None,
                        "quote": None,
                    }
                ]
            else:
                for item in evidence:
                    if isinstance(item, dict):
                        item["artifact_id"] = artifact.artifact_id

        try:
            return ClinicalEvent.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"LLM JSON failed schema validation: {exc}") from exc


def build_default_providers() -> tuple[OCRProvider, ClinicalNormalizer]:
    api_key = get_nvidia_api_key()
    if not api_key:
        from .ingestion import DemoOCRProvider

        return DemoOCRProvider(), ClinicalNormalizer()
    return NvidiaOCRProvider(api_key=api_key), NvidiaClinicalNormalizer(api_key=api_key)
