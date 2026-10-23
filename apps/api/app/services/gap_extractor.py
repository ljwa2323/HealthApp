from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

from ..schemas import ClinicalEvent, ClinicalFact


# 临床主题 → 评估该主题应查的指标。
# required 写常用中文名，匹配 fact.concept.display（中英文都做子串包含）。
# max_age_days: 超过这个天数没复查也算缺口。
# 字典不全时不会崩：panel_known=False 时只报 observed_concepts，由 LLM 自行判断。
TOPIC_PANELS: dict[str, dict] = {
    "renal_function": {
        "required": [
            ["肌酐", "Creatinine"],
            ["eGFR"],
            ["尿素", "尿素氮", "BUN"],
            ["尿微量白蛋白"],
            ["胱抑素", "Cystatin"],
            ["尿酸", "UA"],
        ],
        "max_age_days": 365,
    },
    "glucose_metabolism": {
        "required": [
            ["空腹血糖", "FPG", "GLU"],
            ["糖化血红蛋白", "HbA1c"],
            ["餐后", "OGTT"],
            ["胰岛素", "INS"],
        ],
        "max_age_days": 180,
    },
    "liver_function": {
        "required": [
            ["ALT", "谷丙转氨酶", "丙氨酸氨基转移酶"],
            ["AST", "谷草转氨酶", "门冬氨酸氨基转移酶"],
            ["TBIL", "总胆红素"],
            ["DBIL", "直接胆红素"],
            ["IBIL", "间接胆红素"],
            ["ALB", "白蛋白"],
            ["GLB", "球蛋白"],
            ["TP", "总蛋白"],
            ["ALP", "碱性磷酸酶"],
            ["GGT", "谷氨酰转肽酶", "r-谷氨酰转肽酶"],
        ],
        "max_age_days": 365,
    },
    "lipid": {
        "required": [
            ["总胆固醇", "TC"],
            ["甘油三酯", "TG"],
            ["LDL"],
            ["HDL"],
        ],
        "max_age_days": 365,
    },
    "blood_pressure": {
        "required": [["收缩压"], ["舒张压"]],
        "max_age_days": 90,
    },
    "blood_routine": {
        "required": [
            ["白细胞", "WBC"],
            ["红细胞", "RBC"],
            ["血红蛋白", "HGB"],
            ["血小板", "PLT"],
        ],
        "max_age_days": 365,
    },
}


@dataclass
class GapReport:
    gaps: list[str] = field(default_factory=list)
    panel_known: bool = True
    observed_concepts: list[str] = field(default_factory=list)


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().lower().replace(" ", "")


def _matches_concept(fact: ClinicalFact, concept: str) -> bool:
    needle = _normalize(concept)
    if not needle:
        return False
    haystacks = [
        _normalize(fact.concept.display),
        _normalize(fact.concept.local_display),
    ]
    return any(needle in h or h in needle for h in haystacks if h)


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


def extract_data_gaps(
    *,
    events: Iterable[ClinicalEvent],
    topic: str | None,
    today: date | None = None,
) -> GapReport:
    """
    返回三层信息：
    - gaps: 字典命中时，确定知道该查但没查的指标
    - panel_known: 这个主题我们有没有标准清单。False 时 gaps 为空，
      不要让 LLM 假装知道还缺什么。
    - observed_concepts: 病人这次实际有的指标，让 LLM 知道它手头有什么数据。
    """
    all_facts: list[ClinicalFact] = []
    for event in events:
        all_facts.extend(event.facts)

    observed = sorted({fact.concept.display for fact in all_facts if fact.concept.display})

    if not topic or topic not in TOPIC_PANELS:
        return GapReport(panel_known=False, observed_concepts=observed)

    panel = TOPIC_PANELS[topic]
    required = panel["required"]
    max_age = panel["max_age_days"]
    today = today or date.today()

    gaps: list[str] = []
    for alias_group in required:
        # alias_group 是同一指标的不同写法，如 ["ALT", "谷丙转氨酶"]
        # 只要 facts 里匹配到任意一个别名，就算"已检测过"
        display_name = alias_group[0]
        latest: date | None = None
        any_match = False
        for fact in all_facts:
            matched = any(_matches_concept(fact, alias) for alias in alias_group)
            if not matched:
                continue
            any_match = True
            fact_date = _parse_date(fact.time.start)
            if fact_date is None:
                continue
            if latest is None or fact_date > latest:
                latest = fact_date

        if not any_match:
            gaps.append(f"从未检测过{display_name}")
        elif latest is not None:
            age_days = (today - latest).days
            if age_days > max_age:
                months = age_days // 30
                gaps.append(f"{display_name}最近一次是{latest.isoformat()}，已超过{months}个月未复查")

    return GapReport(gaps=gaps, panel_known=True, observed_concepts=observed)
