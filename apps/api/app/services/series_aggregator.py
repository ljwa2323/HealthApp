from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

from ..schemas import ClinicalEvent, ClinicalFact, EventTime


@dataclass
class SeriesPoint:
    t: str
    v: float | None
    unit: str | None
    flag: str | None
    fact_id: str
    quote: str | None


@dataclass
class Series:
    concept: str
    loinc: str | None
    unit: str | None
    ref_range: str | None
    points: list[SeriesPoint] = field(default_factory=list)
    trend: str = "insufficient_data"
    delta_pct: float | None = None


@dataclass
class MedicationState:
    drug: str
    dose: str | None
    since: str | None
    stopped: str | None


@dataclass
class AggregatedContext:
    series: list[Series] = field(default_factory=list)
    active_medications: list[MedicationState] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)


def _parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _time_in_window(t: date | None, start: date | None, end: date | None) -> bool:
    # 没指定时间窗时，不按日期过滤（无日期的事件也保留）
    if not start and not end:
        return True
    if t is None:
        return False
    if start and t < start:
        return False
    if end and t > end:
        return False
    return True


def _compute_trend(values: list[float]) -> tuple[str, float | None]:
    """根据点列判断趋势。至少 3 个点才判断方向。"""
    if len(values) < 2:
        return "insufficient_data", None
    first, last = values[0], values[-1]
    if first == 0:
        return "stable", 0.0
    delta = (last - first) / first * 100
    if len(values) >= 3:
        # 简单斜率判断：首尾 + 中间点是否单调
        rising = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        falling = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        if rising and delta > 5:
            return "rising_slowly" if delta < 20 else "rising", round(delta, 1)
        if falling and delta < -5:
            return "falling_slowly" if delta > -20 else "falling", round(delta, 1)
    if delta > 10:
        return "up_jump", round(delta, 1)
    if delta < -10:
        return "down_drop", round(delta, 1)
    return "stable", round(delta, 1)


def aggregate_context(
    *,
    events: Iterable[ClinicalEvent],
    topic: str | None = None,
    matched_concepts: list[str] | None = None,
    time_start: date | None = None,
    time_end: date | None = None,
) -> AggregatedContext:
    """
    把 events 聚合成给 LLM 看的聚焦上下文：
    - series: 按 concept 分组的时间序列（只保留时间窗内的 observation）
    - active_medications: 折叠后的当前用药
    - recent_events: 事件级骨架
    """
    events = list(events)

    # 按 concept 分组 observation facts
    buckets: dict[str, list[tuple[date, SeriesPoint, ClinicalFact]]] = {}
    medications: list[ClinicalFact] = []

    for event in events:
        event_date = _parse_date(event.event_time.start)

        for fact in event.facts:
            fact_date = _parse_date(fact.time.start) or event_date
            if not _time_in_window(fact_date, time_start, time_end):
                continue

            if fact.kind == "observation":
                display = fact.concept.display or "unknown"
                value = fact.value.number if fact.value and fact.value.type == "quantity" else None
                quote = fact.evidence[0].quote if fact.evidence else None
                point = SeriesPoint(
                    t=fact_date.isoformat() if fact_date else "",
                    v=value,
                    unit=fact.value.unit if fact.value and fact.value.type == "quantity" else None,
                    flag=fact.interpretation,
                    fact_id=fact.fact_id,
                    quote=quote,
                )
                buckets.setdefault(display, []).append((fact_date, point, fact))

            elif fact.kind == "medication":
                medications.append(fact)

    # 组装 series
    series_list: list[Series] = []
    for concept, items in buckets.items():
        items.sort(key=lambda x: x[0] or date.min)
        facts = [item[2] for item in items]
        ref = facts[0].reference_range
        ref_str = None
        if ref and ref.low is not None and ref.high is not None:
            ref_str = f"{ref.low}-{ref.high}"
        unit = facts[0].value.unit if facts[0].value and facts[0].value.type == "quantity" else None
        values = [p.v for _, p, _ in items if p.v is not None]
        trend, delta = _compute_trend(values)
        series_list.append(Series(
            concept=concept,
            loinc=facts[0].concept.code,
            unit=unit,
            ref_range=ref_str,
            points=[p for _, p, _ in items],
            trend=trend,
            delta_pct=delta,
        ))

    # 折叠用药：按 drug 分组，取最新的一条状态
    med_by_drug: dict[str, list[tuple[date, ClinicalFact]]] = {}
    for fact in medications:
        d = _parse_date(fact.time.start)
        drug = fact.concept.display or "unknown"
        med_by_drug.setdefault(drug, []).append((d or date.min, fact))

    active_meds: list[MedicationState] = []
    for drug, entries in med_by_drug.items():
        entries.sort(key=lambda x: x[0])
        latest = entries[-1][1]
        status = latest.attributes.get("status", "active")
        dose = latest.attributes.get("dose") or latest.attributes.get("dosage")
        stopped = None
        if status == "stopped" or status == "discontinued":
            stopped = entries[-1][0].isoformat()
        active_meds.append(MedicationState(
            drug=drug,
            dose=dose,
            since=entries[0][0].isoformat() if entries[0][0] != date.min else None,
            stopped=stopped,
        ))

    # recent_events 简单去重
    seen_events: set[tuple] = set()
    recent_events = []
    for event in events:
        event_date = _parse_date(event.event_time.start)
        if not _time_in_window(event_date, time_start, time_end):
            continue
        key = (event.event_id,)
        if key in seen_events:
            continue
        seen_events.add(key)
        recent_events.append({
            "date": event_date.isoformat() if event_date else None,
            "type": event.event_type,
            "title": event.title,
            "institution": event.institution,
        })

    return AggregatedContext(
        series=series_list,
        active_medications=active_meds,
        recent_events=recent_events,
    )
