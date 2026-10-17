from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any
from uuid import uuid4

import httpx
from openai import APIStatusError, AsyncOpenAI, RateLimitError
from PIL import Image, UnidentifiedImageError

from ..config import (
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_INLINE_B64_CHARS,
    NVCF_ASSETS_URL,
    OCR_INVOKE_URL,
    get_deepseek_api_key,
    get_glm_api_key,
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
        self.extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }

    async def _chat_completion(self, messages: list[dict[str, str]]):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=1,
            max_tokens=16384,
            stream=False,
            extra_body=self.extra_body,
        )

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
                completion = await self._chat_completion(messages)
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
        # GLM/other models sometimes omit top-level fields; fill sensible defaults
        # so schema validation does not throw away otherwise-good facts.
        payload.setdefault("event_type", "laboratory")
        payload.setdefault("title", artifact.file_name or "Health record")
        if not payload.get("event_time"):
            payload["event_time"] = {"precision": "unknown", "source": "document"}

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


class DeepSeekClinicalNormalizer(NvidiaClinicalNormalizer):
    """OpenAI-compatible LLM normalizer backed by DeepSeek.

    OCR stays local (RapidOCR); this model only does the post-OCR
    structuring step: turn messy OCR text (including multi-column layout
    that the rule-based extractor cannot disambiguate) into a schema-valid
    ClinicalEvent. Runs in the cloud, so no local GPU/30B model needed.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or get_deepseek_api_key()
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.api_key,
            timeout=180.0,
            max_retries=0,
        )
        self.model = "deepseek-chat"
        # DeepSeek's OpenAI-compatible endpoint does not accept NVIDIA/NIM
        # specific extra_body fields; leave it empty.
        self.extra_body = {}


class GLMClinicalNormalizer(NvidiaClinicalNormalizer):
    """OpenAI-compatible LLM normalizer backed by Zhipu GLM (glm-4-flash).

    Runs on the free-tier Chinese model, directly reachable from mainland
    networks. Same prompt/schema/JSON-repair logic as the NVIDIA normalizer.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or get_glm_api_key()
        if not self.api_key:
            raise ValueError("GLM_API_KEY is not configured")
        self.client = AsyncOpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=self.api_key,
            timeout=180.0,
            max_retries=0,
        )
        self.model = "glm-5.3"
        # GLM-5.3 always thinks; use the lowest reasoning level to save tokens.
        self.extra_body = {"thinking": {"type": "enabled", "level": "low"}}


class GLMVisionNormalizer(ClinicalNormalizer):
    """Vision-model normalizer: image in, schema-valid ClinicalEvent out.

    Uses the free GLM-4.6V-Flash multimodal model, so the OCR step is bypassed
    entirely: the model sees the actual layout (multi-column tables included)
    and returns structured facts. 429 rate-limits are retried with backoff.
    """

    model = "glm-4.6v-flash"
    MAX_SIDE = 1400
    JPEG_QUALITY = 82

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "glm-4.6v-flash",
    ) -> None:
        self.api_key = api_key or get_glm_api_key()
        if not self.api_key:
            raise ValueError("GLM_API_KEY is not configured")
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=self.api_key,
            timeout=240.0,
            max_retries=0,
        )

    def _image_to_data_url(self, content: bytes) -> str:
        image = Image.open(io.BytesIO(content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        scale = min(1.0, self.MAX_SIDE / max(w, h))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.JPEG_QUALITY)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _build_prompt(self, artifact: SourceArtifact) -> str:
        return (
            "你是医疗文档结构化助手。请直接看图，先判断这是哪一种医疗文档，再按对应规则提取。\n"
            "document_type 只可能是以下 9 种之一：\n"
            "- laboratory: 检验/化验报告（血常规、生化、尿常规、甲功、乙肝等），逐项提取 项目名/数值/单位/参考范围/高低\n"
            "- medication: 处方/用药单，提取 药名/剂量/频次(如 qd、bid、每日3次)/用法(口服、静滴、外用)/数量\n"
            "- imaging: 影像检查文字报告（X光/CT/MRI/超声报告单），提取 检查部位/影像所见/诊断结论\n"
            "- visit: 门诊病历/就诊记录，提取 就诊日期/科室/主诉/诊断\n"
            "- hospitalization: 住院病历/出院小结，提取 入院日期/出院日期/科室/诊断\n"
            "- procedure: 手术/操作记录，提取 手术名称/麻醉方式/术者/术前诊断/术后诊断/日期\n"
            "- vital: 生命体征记录（血压/心率/体温等），提取 项目/数值/单位\n"
            "- symptom: 症状记录/症状量表，提取 症状名/描述/持续时间\n"
            "- other: 以上都不是的其他医疗文档，提取要点摘要。注意：诊断证明书、疾病证明、请假条、病情说明等单页证明文书属于 other（不是 visit），即使上面印有科室/姓名\n"
            "注意：纯影像原图（胶片照片，没有文字报告）不在这里，不要选 imaging。\n"
            "报告可能是多列布局，请按表格行列对应关系提取，不要漏项，不要把左右两列混在一起。\n"
            "先判断文档是结构化还是叙述型：\n"
            "- format=structured：数据型/分项型文档（化验单、表格、处方、逐项记录），提取 facts；\n"
            "- format=narrative：叙述型整段文字文档（出院记录、入院记录、病程记录、诊断意见、证明文书等一整段话），不要把一段话硬拆成多条 facts——此时 facts 留空，把全文按阅读顺序完整抄进 narrative_text（保持原有语句顺序，不改写、不省略、不翻译）。\n"
            "输出一个 JSON 对象（不要 markdown 代码块、不要注释），格式如下：\n"
            "{\n"
            '  "document_type": "9种之一",\n'
            '  "format": "structured 或 narrative",\n'
            '  "report_date": "YYYY-MM-DD 或 null（图上找不到日期就写 null）",\n'
            '  "narrative_text": "仅 format=narrative 时填写完整原文；structured 时写 null",\n'
            '  "facts": [\n'
            '    {"kind": "observation|medication|condition|symptom|note|procedure", "concept": "项目名或药名或诊断名", "value": 数值或字符串, "unit": "单位(检验/体征有则填，否则null)", "reference": "参考范围(检验有则填，否则null)", "interpretation": "low|normal|high|unknown(仅检验填，否则unknown)", "attrs": {"该类型专属字段": "值"}}\n'
            "  ]\n"
            "}\n"
            "规则：\n"
            "1. kind 与 document_type 对应（检验→observation，处方→medication，诊断类→condition，症状→symptom，影像所见/结论→note，手术→procedure）；\n"
            "2. attrs 只放该类型的专属字段，固定用以下键名：\n"
            "   medication: dose(剂量)、frequency(频次)、route(用法)、quantity(数量)\n"
            "   imaging: body_part(检查部位)、conclusion(诊断结论)\n"
            "   visit: department(科室)、chief_complaint(主诉)\n"
            "   hospitalization: department(科室)、admission_date(入院日期)、discharge_date(出院日期)\n"
            "   procedure: anesthesia(麻醉方式)、surgeon(术者)、pre_diagnosis(术前诊断)、post_diagnosis(术后诊断)\n"
            "   symptom: duration(持续时间)\n"
            "   laboratory、vital、other 的 attrs 留空 {}；\n"
            "3. value 若是文字结果（如 阴性/阳性/未见）则填字符串；\n"
            "4. 没有对应信息就写 null，不要编造。\n"
            f"文件名：{artifact.file_name}。只输出这个 JSON 对象。"
        )

    async def normalize_image(
        self,
        patient_id: str,
        artifact: SourceArtifact,
        content: bytes,
    ) -> ClinicalEvent | None:
        if not content:
            return None
        data_url = self._image_to_data_url(content)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt(artifact)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8192,
                    stream=False,
                )
                content_txt = completion.choices[0].message.content or ""
                if not content_txt.strip():
                    raise ValueError("empty vision model response")
                payload = _extract_json_object(content_txt)
                break
            except RateLimitError as exc:
                last_error = exc
                await asyncio.sleep(20 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 11:
                    raise ValueError(f"GLM vision failed: {exc}") from exc
                await asyncio.sleep(8 * (attempt + 1))

        if payload is None:
            raise ValueError(f"GLM vision failed after retries: {last_error}")

        return self._to_clinical_event(patient_id, artifact, payload)

    def _to_clinical_event(
        self,
        patient_id: str,
        artifact: SourceArtifact,
        payload: dict[str, Any],
    ) -> ClinicalEvent:
        VALID_TYPES = {
            "laboratory", "imaging", "visit", "hospitalization",
            "procedure", "medication", "vital", "symptom", "other",
        }
        VALID_KINDS = {
            "observation", "condition", "medication", "procedure",
            "allergy", "symptom", "note",
        }
        doc_type = str(payload.get("document_type") or "").strip().lower()
        event_type = doc_type if doc_type in VALID_TYPES else "other"

        event: dict[str, Any] = {
            "event_id": f"evt_{uuid4().hex[:12]}",
            "patient_id": patient_id,
            "event_type": event_type,
            "title": artifact.file_name or "Health record",
            "source_artifact_ids": [artifact.artifact_id],
            "event_time": {"precision": "unknown", "source": "document"},
        }
        if payload.get("title"):
            event["title"] = str(payload["title"])
        if payload.get("report_date"):
            event["event_time"] = {
                "start": payload["report_date"],
                "precision": "day",
                "source": "document",
            }

        # Narrative documents (discharge summaries, progress notes, etc.)
        # are kept as raw readable text instead of being split into facts.
        # The user reviews/edits the text on the right and decides whether
        # to add it to the timeline; the AI already judged the split.
        if str(payload.get("format") or "structured").strip().lower() == "narrative":
            narrative = str(
                payload.get("narrative_text") or payload.get("summary") or ""
            ).strip()
            if narrative:
                event["narrative"] = narrative
                event["summary"] = narrative
                event["facts"] = []
                return ClinicalEvent.model_validate(event)

        facts: list[dict[str, Any]] = []
        for raw in payload.get("facts") or []:
            if not isinstance(raw, dict):
                continue
            concept = raw.get("concept") or raw.get("name")
            if not concept:
                continue
            kind = str(raw.get("kind") or "observation").strip().lower()
            if kind not in VALID_KINDS:
                kind = "observation"
            concept_display = str(concept)
            fact: dict[str, Any] = {
                "fact_id": f"fact_{uuid4().hex[:10]}",
                "kind": kind,
                "concept": {"display": concept_display},
                "time": {"precision": "unknown", "source": "document"},
                "status": "extracted",
                "evidence": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "page": 1,
                        "block_id": None,
                        "quote": None,
                    }
                ],
            }
            if payload.get("report_date"):
                fact["time"] = {
                    "start": payload["report_date"],
                    "precision": "day",
                    "source": "document",
                }

            value = raw.get("value")
            unit = raw.get("unit")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fact["value"] = {
                    "type": "quantity",
                    "number": value,
                    "unit": str(unit) if unit else None,
                }
            elif value is not None and str(value) != "":
                fact["value"] = {"type": "text", "text": str(value)}
            ref = raw.get("reference") or raw.get("reference_range")
            if ref:
                fact["reference_range"] = {"text": str(ref), "unit": str(unit) if unit else None}
            interp = str(raw.get("interpretation") or "unknown").lower()
            if interp not in {"low", "normal", "high", "abnormal", "unknown"}:
                interp = "unknown"
            fact["interpretation"] = interp

            # Type-specific fields (dose/frequency/route/department/anesthesia…)
            # land in `attributes` so the frontend can render per-type templates.
            attrs: dict[str, Any] = {}
            raw_attrs = raw.get("attrs") or raw.get("attributes") or {}
            if isinstance(raw_attrs, dict):
                for key, val in raw_attrs.items():
                    if val is None or str(val).strip() == "":
                        continue
                    attrs[str(key)] = str(val)
            detail = raw.get("detail")
            if detail:
                attrs.setdefault("detail", str(detail))
            if attrs:
                fact["attributes"] = attrs
            facts.append(fact)

        event["facts"] = facts
        return ClinicalEvent.model_validate(event)


def build_default_providers() -> tuple[OCRProvider, ClinicalNormalizer]:
    nvidia_key = get_nvidia_api_key()
    if nvidia_key:
        return NvidiaOCRProvider(api_key=nvidia_key), NvidiaClinicalNormalizer(api_key=nvidia_key)

    glm_key = get_glm_api_key()
    if glm_key:
        try:
            from .rapid_ocr_provider import RapidOCRProvider

            # Vision model reads the image directly (bypasses OCR text),
            # solving multi-column layouts that text-only models cannot.
            # Paid glm-4.6v: 92.4% field accuracy on the 20-image web set
            # (vs 45.2% rule-based) and no rate limits; cost ~0.15 CNY/20 imgs.
            return RapidOCRProvider(), GLMVisionNormalizer(
                api_key=glm_key, model="glm-4.6v"
            )
        except Exception:  # noqa: BLE001 - fall through
            pass

    deepseek_key = get_deepseek_api_key()
    if deepseek_key:
        # Local offline OCR + cloud LLM normalization.
        try:
            from .rapid_ocr_provider import RapidOCRProvider

            return RapidOCRProvider(), DeepSeekClinicalNormalizer(api_key=deepseek_key)
        except Exception:  # noqa: BLE001 - fall through to local rule-based
            pass

    # No cloud key: prefer local, offline OCR (zero cost, no rate limits).
    # Field extraction is rule-based (coordinates + dictionaries + regex);
    # an LLM normalizer can wrap or replace it later without model changes.
    try:
        from .field_extractor import RuleBasedNormalizer
        from .rapid_ocr_provider import RapidOCRProvider

        return RapidOCRProvider(), RuleBasedNormalizer()
    except Exception:  # noqa: BLE001 - RapidOCR unavailable, keep the demo fallback.
        from .ingestion import DemoOCRProvider

        return DemoOCRProvider(), ClinicalNormalizer()
