from __future__ import annotations

import json
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from ..config import LLM_BASE_URL, LLM_MODEL, get_nvidia_api_key
from ..schemas import AgentContext, ClinicalEvent
from .context_builder import build_agent_context
from .nvidia_providers import RETRYABLE_HTTP_STATUS, _sleep_backoff


SYSTEM_PROMPT = (
    "You are HealthApp's longitudinal health analysis assistant. "
    "You only use the provided structured JSON context (events, observations, "
    "medications, conditions, data_gaps, evidence_refs). "
    "Do not invent lab values, diagnoses, or medications that are not in the context. "
    "If evidence is insufficient, say so clearly and list data gaps. "
    "Write in clear Simplified Chinese. "
    "This is decision support for the patient, not a medical diagnosis. "
    "Mention that clinical decisions should be made with a licensed clinician."
)


def _client() -> AsyncOpenAI:
    api_key = get_nvidia_api_key()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured")
    return AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=api_key,
        timeout=180.0,
        max_retries=0,
    )


def build_patient_context(
    *,
    patient_id: str,
    events: list[ClinicalEvent],
    task: str = "general_longitudinal_review",
) -> AgentContext:
    return build_agent_context(
        patient_id=patient_id,
        task=task,
        events=events,
    )


async def _chat_once(messages: list[dict[str, str]]) -> str:
    client = _client()
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            completion = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.2,
                top_p=1,
                max_tokens=4096,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_effort": "none",
                },
            )
            return (completion.choices[0].message.content or "").strip()
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code not in RETRYABLE_HTTP_STATUS or attempt == 3:
                raise ValueError(
                    f"NVIDIA LLM failed: HTTP {exc.status_code}: {exc.message}"
                ) from exc
            await _sleep_backoff(attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 3:
                raise
            await _sleep_backoff(attempt)
    if last_error is not None:
        raise last_error
    raise ValueError("NVIDIA LLM failed after retries")


async def analyze_health(context: AgentContext) -> dict[str, Any]:
    context_json = context.model_dump(mode="json")
    user_prompt = (
        "Please run a longitudinal health analysis using ONLY this context JSON.\n"
        "Return markdown with these sections:\n"
        "1. 总体概况\n"
        "2. 关键关注的变化\n"
        "3. 数据缺口\n"
        "4. 可与医生讨论的问题\n\n"
        f"CONTEXT_JSON:\n{json.dumps(context_json, ensure_ascii=False)}"
    )
    answer = await _chat_once(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return {
        "answer": answer,
        "context": context_json,
        "used_fact_ids": [item.get("fact_id") for item in context.observations][:20],
        "evidence_refs": context.evidence_refs,
    }


async def answer_question(
    *,
    context: AgentContext,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    context_json = context.model_dump(mode="json")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Structured patient context JSON follows. "
                "Answer later questions using only this context.\n\n"
                f"CONTEXT_JSON:\n{json.dumps(context_json, ensure_ascii=False)}"
            ),
        },
        {
            "role": "assistant",
            "content": "已加载结构化健康上下文。我会仅基于这些事实回答。",
        },
    ]
    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})

    answer = await _chat_once(messages)
    return {
        "answer": answer,
        "context": context_json,
        "evidence_refs": context.evidence_refs,
    }
