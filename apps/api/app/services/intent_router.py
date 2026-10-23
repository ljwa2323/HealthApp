from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

from openai import AsyncOpenAI

from ..config import get_glm_api_key


VALID_TOPICS = {
    "renal_function",
    "glucose_metabolism",
    "liver_function",
    "lipid",
    "blood_pressure",
    "blood_routine",
    "general",
}

VALID_QUESTION_TYPES = {
    "trend_query",
    "latest_value",
    "medication_review",
    "summary",
    "specific_event",
    "unknown",
}


@dataclass
class Intent:
    topic: str = "general"
    question_type: str = "summary"
    time_start: date | None = None
    time_end: date | None = None
    matched_concepts: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


ROUTER_SYSTEM = (
    "你是 HealthApp 的问题路由器。把用户的健康问题解析成结构化 JSON。\n"
    "只输出 JSON，不要解释。\n\n"
    "字段说明：\n"
    "- topic: 临床主题，只能是 renal_function / glucose_metabolism / liver_function / "
    "lipid / blood_pressure / blood_routine / general 之一。判断不出就用 general。\n"
    "- question_type: 问题类型，只能是 trend_query（趋势变化）/ latest_value（最新值）/ "
    "medication_review（用药情况）/ summary（整体总结）/ specific_event（某次具体事件）/ "
    "unknown 之一。\n"
    "- time_start / time_end: ISO 日期 YYYY-MM-DD。用户说'近半年'就从今天往前推 6 个月；"
    "说'最近三个月'就推 3 个月；说'今年'就从 1 月 1 日开始；没提时间就都给 null。\n"
    "- matched_concepts: 用户问题里提到的具体指标名（中文），比如 ['肌酐', 'eGFR']。"
    "没提到具体指标就给空数组。\n\n"
    "今天是 {today}。"
)


def _client() -> AsyncOpenAI:
    api_key = get_glm_api_key()
    if not api_key:
        raise ValueError("GLM_API_KEY is not configured")
    return AsyncOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=api_key,
        timeout=15.0,
        max_retries=1,
    )


def _fallback(question: str) -> Intent:
    # 路由失败时的兜底：全时间窗口、general 主题
    return Intent(topic="general", question_type="summary")


async def route_question(question: str, *, today: date | None = None) -> Intent:
    today = today or date.today()
    prompt = ROUTER_SYSTEM.format(today=today.isoformat())

    try:
        client = _client()
        completion = await client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        data = json.loads(content)
    except Exception:
        return _fallback(question)

    topic = data.get("topic", "general")
    if topic not in VALID_TOPICS:
        topic = "general"

    qtype = data.get("question_type", "summary")
    if qtype not in VALID_QUESTION_TYPES:
        qtype = "summary"

    def _parse_date(v):
        if not v or not isinstance(v, str):
            return None
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None

    return Intent(
        topic=topic,
        question_type=qtype,
        time_start=_parse_date(data.get("time_start")),
        time_end=_parse_date(data.get("time_end")),
        matched_concepts=[c for c in data.get("matched_concepts", []) if isinstance(c, str)][:10],
        raw=data,
    )
