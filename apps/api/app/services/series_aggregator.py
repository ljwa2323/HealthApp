from __future__ import annotations

import re
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
class MedicationCourse:
    """一段连续用药：同一味药被拆成几段，是因为中间停过。"""
    since: str | None
    until: str | None
    records: int
    status: str | None = None
    last: str | None = None


@dataclass
class MedicationState:
    drug: str
    dose: str | None
    since: str | None
    stopped: str | None
    fact_id: str | None = None
    frequency: str | None = None
    route: str | None = None
    quantity: str | None = None
    courses: list[MedicationCourse] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)


@dataclass
class ProcedureRecord:
    name: str
    date: str | None
    institution: str | None
    source: str  # "fact" 来自 procedure 类事实，"narrative" 来自事件正文
    fact_id: str | None = None
    event_id: str | None = None
    quote: str | None = None


@dataclass
class AggregatedContext:
    series: list[Series] = field(default_factory=list)
    # 档案里记过的所有药，含已停用的；判断"还在吃"用 ongoing_medications
    medications: list[MedicationState] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)
    clinical_notes: list[dict] = field(default_factory=list)
    procedures: list[ProcedureRecord] = field(default_factory=list)

    @property
    def ongoing_medications(self) -> list[MedicationState]:
        """还在吃的药。没有停药日期的算在用。"""
        return [m for m in self.medications if not m.stopped]


CLINICAL_NOTE_KINDS = {"condition", "procedure", "symptom", "note"}

# 用药属性在不同来源里的写法不统一，第一条命中的就是取值
_MED_ATTR_KEYS = {
    "dose": ("dose", "dosage", "剂量"),
    "frequency": ("frequency", "freq", "频次", "次数"),
    "route": ("route", "用法", "给药途径"),
    "quantity": ("quantity", "qty", "数量", "总量"),
    "status": ("status", "状态", "是否停药"),
}

# 两条用药记录间隔超过这么久，就当作另一段疗程（常见供药周期都在 3 个月内）
COURSE_GAP_DAYS = 90

# 出院小结里描述术式的句子，用于补齐没有 procedure 类事实的病例
_PROCEDURE_RE = re.compile(
    "|".join(
        (
            "手术", "切除", "置换", "置入", "吻合", "清创", "引流", "造瘘",
            "腔镜", "开胸", "开腹", "搭桥", "起搏器", "支架植入",
        )
    )
)
_SENTENCE_SPLIT = re.compile(r"[。；;\n]")
_CLAUSE_SPLIT = re.compile(r"[，,、]")


def _procedure_clause(sentence: str) -> str:
    """从"行腹腔镜阑尾切除术，术后恢复良好"里取术式那一小段。"""
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip()
        if clause and _PROCEDURE_RE.search(clause):
            return (clause[1:] if clause.startswith("行") else clause).strip() or clause
    return sentence.strip()

# 手术记录被 OCR 拆成字段级事实时，只有名称类字段是术式，
# 麻醉方式、术前术后诊断这些取出来会变成"做过全身麻醉"这种假记录
_PROCEDURE_NAME_LABELS = ("手术名称", "术式", "手术", "操作")
_NON_PROCEDURE_LABELS = (
    "麻醉", "诊断", "日期", "时间", "术者", "助手", "切口", "体位",
    "出血", "输液", "备注", "科室",
)


def _procedure_name(fact: ClinicalFact) -> str | None:
    label = (fact.concept.display or "").strip()
    text = fact.value.text if fact.value and fact.value.type == "text" else None

    if label and any(bad in label for bad in _NON_PROCEDURE_LABELS):
        return None
    if text and (not label or any(good in label for good in _PROCEDURE_NAME_LABELS)):
        return text.strip()
    return (text or label).strip() or None


def _med_attr(fact: ClinicalFact, field_name: str) -> str | None:
    for key in _MED_ATTR_KEYS[field_name]:
        value = fact.attributes.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _latest_attr(entries: list[tuple[date, ClinicalFact]], field_name: str) -> str | None:
    """从后往前找第一条带这个属性的记录。

    停药那条往往只有 status，没有剂量频次，只取最新一条会把字段全丢。
    """
    for _, fact in reversed(entries):
        value = _med_attr(fact, field_name)
        if value:
            return value
    return None


def _split_courses(entries: list[tuple[date, ClinicalFact]]) -> list[dict]:
    """把同一味药的记录切成若干段用药期。

    数据库里没有"这次疗程结束"的字段，只能看记录之间的间隔：
    同一段连续开药的记录间隔通常在一个供药周期内，隔了三个月再开是另一段疗程。
    """
    courses: list[dict] = []
    current: list[tuple[date, ClinicalFact]] = []
    for entry in entries:
        d, fact = entry
        # 写了停药的记录是这一段的结尾，不管隔多久都不算新开一段
        closes = (_med_attr(fact, "status") or "").lower() in ("stopped", "discontinued")
        if (
            current
            and not closes
            and d != date.min
            and current[-1][0] != date.min
            and (d - current[-1][0]).days > COURSE_GAP_DAYS
        ):
            courses.append(_finish_course(current))
            current = []
        current.append(entry)
        if closes:
            courses.append(_finish_course(current))
            current = []
    if current:
        courses.append(_finish_course(current))
    return courses


def _finish_course(entries: list[tuple[date, ClinicalFact]]) -> dict:
    dated = [(d, fact) for d, fact in entries if d != date.min]
    status = None
    for _, fact in reversed(entries):
        value = _med_attr(fact, "status")
        if value:
            status = value.lower()
            break
    if not dated:
        return {"since": None, "until": None, "last": None, "records": len(entries), "status": status}
    since = dated[0][0].isoformat()
    last = dated[-1][0].isoformat()
    # until 只表示明确的结束日期；没有停药记录时留空，单次记录也一样
    until = last if status in ("stopped", "discontinued") else None
    return {"since": since, "until": until, "last": last, "records": len(dated), "status": status}


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
    - medications: 折叠后的用药，含已停用的；在用视图是 ongoing_medications
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

    # 折叠用药：同一味药按记录日期切疗程，中间停超过 COURSE_GAP_DAYS 就算另一段
    med_by_drug: dict[str, list[tuple[date, ClinicalFact]]] = {}
    for fact in medications:
        d = _parse_date(fact.time.start)
        drug = fact.concept.display or "unknown"
        med_by_drug.setdefault(drug, []).append((d or date.min, fact))

    med_states: list[MedicationState] = []
    for drug, entries in med_by_drug.items():
        entries.sort(key=lambda x: x[0])
        courses = _split_courses(entries)
        latest = entries[-1][1]
        latest_course = courses[-1]
        # 只有最后一段明确写了停药，才算这味药已停
        stopped = latest_course["until"] if latest_course["status"] in ("stopped", "discontinued") else None
        med_states.append(MedicationState(
            drug=drug,
            dose=_latest_attr(entries, "dose"),
            since=courses[0]["since"],
            stopped=stopped,
            fact_id=latest.fact_id,
            frequency=_latest_attr(entries, "frequency"),
            route=_latest_attr(entries, "route"),
            quantity=_latest_attr(entries, "quantity"),
            courses=[MedicationCourse(**course) for course in courses],
            fact_ids=list(dict.fromkeys(fact.fact_id for _, fact in entries if fact.fact_id)),
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
            "event_id": event.event_id,
            "date": event_date.isoformat() if event_date else None,
            "type": event.event_type,
            "title": event.title,
            "institution": event.institution,
        })

    # 非检验类事实（诊断/手术/症状/叙述）结构化进 context，
    # 让 LLM 回答手术史、诊断等问题时有数据可引用
    clinical_notes: list[dict] = []
    for event in events:
        event_date = _parse_date(event.event_time.start)
        for fact in event.facts:
            if fact.kind not in CLINICAL_NOTE_KINDS:
                continue
            fact_date = _parse_date(fact.time.start) or event_date
            if not _time_in_window(fact_date, time_start, time_end):
                continue
            text = None
            if fact.value:
                if fact.value.type == "text":
                    text = fact.value.text
                elif fact.value.number is not None:
                    text = str(fact.value.number)
            clinical_notes.append({
                "kind": fact.kind,
                "concept": fact.concept.display or "unknown",
                "text": text,
                "date": fact_date.isoformat() if fact_date else None,
                "fact_id": fact.fact_id,
            })

    # 手术记录：procedure 类事实，加上出院小结里描述术式的句子。
    # 后者没有 fact_id，只能引用到事件，但比整台手术漏掉好。
    procedures: list[ProcedureRecord] = []
    seen_procedures: set[tuple[str, str]] = set()
    for event in events:
        event_date = _parse_date(event.event_time.start)

        for fact in event.facts:
            if fact.kind != "procedure":
                continue
            fact_date = _parse_date(fact.time.start) or event_date
            if not _time_in_window(fact_date, time_start, time_end):
                continue
            name = _procedure_name(fact)
            if not name:
                continue
            key = (name[:20], fact_date.isoformat() if fact_date else "")
            if key in seen_procedures:
                continue
            seen_procedures.add(key)
            procedures.append(ProcedureRecord(
                name=name,
                date=fact_date.isoformat() if fact_date else None,
                institution=event.institution,
                source="fact",
                fact_id=fact.fact_id,
                event_id=event.event_id,
            ))

        text = (event.narrative or "").strip()
        if not text or not _time_in_window(event_date, time_start, time_end):
            continue
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if len(sentence) < 4 or not _PROCEDURE_RE.search(sentence):
                continue
            key = (sentence[:20], event_date.isoformat() if event_date else "")
            if key in seen_procedures:
                continue
            seen_procedures.add(key)
            clauses = _procedure_clause(sentence)
            procedures.append(ProcedureRecord(
                name=clauses[:120],
                date=event_date.isoformat() if event_date else None,
                institution=event.institution,
                source="narrative",
                event_id=event.event_id,
                quote=sentence[:120],
            ))

    return AggregatedContext(
        series=series_list,
        medications=med_states,
        recent_events=recent_events,
        clinical_notes=clinical_notes,
        procedures=procedures,
    )
