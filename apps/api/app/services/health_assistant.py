from __future__ import annotations

import json
import re
from typing import AsyncIterator, Any

from openai import AsyncOpenAI

from ..config import get_deepseek_api_key
from .gap_extractor import extract_data_gaps
from .intent_router import route_question
from .series_aggregator import aggregate_context


SYSTEM_PROMPT = (
    "你是 HealthApp 的纵向健康分析助手。"
    "你只能使用提供给你的结构化 JSON 上下文回答问题。"
    "禁止编造上下文中没有的检验数值、诊断或用药。"
    "每个具体数值后面必须标注对应的 [fact_id]，例如'肌酐 96 μmol/L [f_xxx]'。"
    "如果上下文里没有相关数据，直接说明，不要猜测。"
    "回答用简体中文，分小节写，简洁清楚。"
    "\n"
    "当数据不足以给出明确结论时，不要只说'材料不足'。你要像一个医生看报告一样："
    "(1)先说你看到了什么异常或可疑之处，并基于现有数据给出你能给出的判断；"
    "(2)再说要确认判断还缺什么关键检查或指标（参考 data_gaps 字段）；"
    "(3)指出已有数据本身有什么问题（参考 data_quality_issues 字段，比如缺日期、缺单位、OCR识别错误），"
    "建议用户在时间轴上补充或修正这些信息；"
    "(4)最后具体建议用户补充上传什么检查报告或去医院做什么检查。"
    "如果数据已经足够，就直接给结论，不要额外要求用户做检查。"
    "即使数据不全，也要先给出基于现有数据的初步判断，不要陷入只索要材料而不给结论。"
    "\n"
    "这是患者自助参考，不是医疗诊断，最后提醒一句'具体诊疗请咨询执业医生'。"
)

_FACT_ID_RE = re.compile(r"\[([a-zA-Z0-9_]+)\]")


def _client() -> AsyncOpenAI:
    api_key = get_deepseek_api_key()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured")
    return AsyncOpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key,
        timeout=120.0,
        max_retries=1,
    )


def _build_context_payload(events, route_result, agg, gap_report) -> dict[str, Any]:
    series_json = []
    for s in agg.series:
        series_json.append({
            "concept": s.concept,
            "unit": s.unit,
            "ref_range": s.ref_range,
            "trend": s.trend,
            "delta_pct": s.delta_pct,
            "points": [
                {
                    "t": p.t, "v": p.v, "unit": p.unit,
                    "flag": p.flag, "fact_id": p.fact_id, "quote": p.quote,
                }
                for p in s.points
            ],
        })

    return {
        "topic": route_result.topic,
        "question_type": route_result.question_type,
        "time_window": {
            "start": route_result.time_start.isoformat() if route_result.time_start else None,
            "end": route_result.time_end.isoformat() if route_result.time_end else None,
        },
        "series": series_json,
        "active_medications": [
            {"drug": m.drug, "dose": m.dose, "since": m.since, "stopped": m.stopped}
            for m in agg.active_medications
        ],
        "recent_events": agg.recent_events,
        "data_gaps": gap_report.gaps,
        "data_quality_issues": detect_data_quality_issues(events) if events else [],
        "panel_known": gap_report.panel_known,
        "observed_concepts": gap_report.observed_concepts,
    }


def _collect_used_fact_ids(agg) -> list[str]:
    ids: list[str] = []
    for s in agg.series:
        for p in s.points:
            ids.append(p.fact_id)
    return ids


def detect_data_quality_issues(events: list) -> list[str]:
    """
    扫描 events，发现已有数据的字段缺失问题，
    用于建议用户补充/修正具体信息。
    """
    issues: list[str] = []
    no_date_events = 0
    no_unit_facts = 0
    no_ref_range = 0
    no_date_facts = 0
    noisy_concepts = 0

    for event in events:
        event_date = event.event_time.start if event.event_time else None
        if not event_date:
            no_date_events += 1
        for fact in event.facts:
            if fact.kind != "observation":
                continue
            # 有数值但没日期
            fact_date = fact.time.start if fact.time else None
            if not fact_date and not event_date:
                no_date_facts += 1
            # 有数值但没单位
            if fact.value and fact.value.type == "quantity":
                if not fact.value.unit:
                    no_unit_facts += 1
                if not fact.reference_range or (
                    fact.reference_range.low is None and fact.reference_range.high is None
                ):
                    no_ref_range += 1
            # 概念名里混着数字/OCR 噪声
            display = fact.concept.display or ""
            if any(ch.isdigit() for ch in display) and len(display) > 8:
                noisy_concepts += 1

    if no_date_events:
        issues.append(f"有 {no_date_events} 条记录没有标注日期，建议在时间轴上补充事件日期")
    if no_date_facts:
        issues.append(f"有 {no_date_facts} 个检验数值没有检测日期，无法判断时间先后")
    if no_unit_facts:
        issues.append(f"有 {no_unit_facts} 个数值缺少单位，建议核对原始报告补充")
    if no_ref_range:
        issues.append(f"有 {no_ref_range} 个数值缺少参考范围，无法判断是否异常")
    if noisy_concepts:
        issues.append(f"有 {noisy_concepts} 个指标名称疑似 OCR 识别错误，建议核对修正")

    return issues


def validate_citations(answer: str, allowed_fact_ids: list[str]) -> dict[str, Any]:
    """检查答案里 [xxx] 引用的 fact_id 是否都在本次 context 里。"""
    allowed = set(allowed_fact_ids)
    cited = set(_FACT_ID_RE.findall(answer))
    hallucinated = sorted(cited - allowed)
    return {
        "cited_fact_ids": sorted(cited),
        "allowed_fact_ids": sorted(allowed),
        "hallucinated_citations": hallucinated,
        "hallucination_count": len(hallucinated),
    }


def compute_confidence(
    context: dict[str, Any],
    citation_check: dict[str, Any] | None = None,
) -> float:
    """
    根据本次上下文的数据丰富度估算置信度（0.1 ~ 0.95）。
    - 基础分 0.3
    - 命中的 series 数量加分
    - 有日期的数据点占比加分
    - 有幻觉引用扣分
    - data_gaps 越多扣分
    """
    score = 0.4

    series = context.get("series") or []
    score += min(len(series) * 0.02, 0.2)

    dated = 0
    total_points = 0
    for s in series:
        for p in s.get("points", []):
            total_points += 1
            if p.get("t"):
                dated += 1
    if total_points:
        score += (dated / total_points) * 0.15

    gaps = context.get("data_gaps") or []
    score -= len(gaps) * 0.02

    if citation_check and citation_check.get("hallucination_count"):
        score -= 0.1 * citation_check["hallucination_count"]

    return round(max(0.1, min(0.95, score)), 2)


def confidence_footer(confidence: float) -> str:
    """根据置信度生成结尾提示语。"""
    if confidence >= 0.8:
        return f"\n\n---\n置信度 {confidence:.2f}：本次答案基于较完整的已有记录，仅供参考，具体诊疗请咨询执业医生。"
    elif confidence >= 0.5:
        return f"\n\n---\n置信度 {confidence:.2f}：有部分数据支持上述判断，但仍有缺口。建议补充缺失的检查报告后再做更完整评估，必要时咨询医生。"
    elif confidence >= 0.3:
        return f"\n\n---\n置信度 {confidence:.2f}：现有数据不足以给出可靠判断。上文已列出你看到的异常和还缺的检查，建议补充上传相关报告或去医院做对应检查后再来问。"
    else:
        return f"\n\n---\n置信度 {confidence:.2f}：数据严重不足，无法做出判断。建议携带已有资料到医院面诊。"


async def _prepare(question: str, events: list) -> tuple[dict, list[dict[str, str]], list[str]]:
    """路由 + 聚合 + 缺口 + 拼 messages。返回 (context_payload, messages, used_fact_ids)。"""
    route = await route_question(question)
    agg = aggregate_context(
        events=events,
        topic=route.topic,
        matched_concepts=route.matched_concepts,
        time_start=route.time_start,
        time_end=route.time_end,
    )
    gap_report = extract_data_gaps(events=events, topic=route.topic)
    context_payload = _build_context_payload(events, route, agg, gap_report)
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
    used = _collect_used_fact_ids(agg)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "结构化患者上下文如下，请只基于它回答后面的问题。\n\n"
                f"CONTEXT_JSON:\n{context_json}\n\n"
                f"患者问题：{question}"
            ),
        },
    ]
    return context_payload, messages, used


async def answer_question(
    *,
    question: str,
    events: list,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    context_payload, messages, used = await _prepare(question, events)

    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    client = _client()
    completion = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        stream=False,
    )
    answer = (completion.choices[0].message.content or "").strip()
    citations = validate_citations(answer, used)
    confidence = compute_confidence(context_payload, citations)
    answer = answer + confidence_footer(confidence)

    return {
        "answer": answer,
        "context": context_payload,
        "used_fact_ids": used,
        "citation_check": citations,
        "confidence": confidence,
    }


async def answer_question_stream(
    *,
    question: str,
    events: list,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式版本：先 yield 路由/上下文元数据，再逐块 yield 答案文本，最后 yield 校验结果。"""
    context_payload, messages, used = await _prepare(question, events)

    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    # 先推元数据，让前端知道路由结果和有哪些 fact_id
    yield {
        "type": "meta",
        "context": context_payload,
        "used_fact_ids": used,
    }

    client = _client()
    stream = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        stream=True,
    )

    answer_parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            answer_parts.append(delta)
            yield {"type": "chunk", "text": delta}

    full_answer = "".join(answer_parts).strip()
    citations = validate_citations(full_answer, used)
    confidence = compute_confidence(context_payload, citations)
    footer = confidence_footer(confidence)
    full_answer = full_answer + footer
    yield {"type": "chunk", "text": footer}
    yield {
        "type": "done",
        "answer": full_answer,
        "citation_check": citations,
        "confidence": confidence,
    }
