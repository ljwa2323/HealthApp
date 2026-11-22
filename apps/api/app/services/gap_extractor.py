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
    "thyroid_function": {
        "required": [
            ["促甲状腺激素", "TSH"],
            ["游离三碘甲状腺原氨酸", "FT3"],
            ["游离甲状腺素", "FT4"],
            ["抗甲状腺过氧化物酶抗体", "TPOAb", "甲状腺过氧化物酶抗体"],
            ["抗甲状腺球蛋白抗体", "TGAb", "甲状腺球蛋白抗体"],
        ],
        "max_age_days": 365,
    },
    "coagulation": {
        "required": [
            ["凝血酶原时间", "PT"],
            ["活化部分凝血活酶时间", "APTT"],
            ["INR", "国际标准化比值"],
            ["纤维蛋白原", "FIB"],
        ],
        "max_age_days": 365,
    },
    "tumor_markers": {
        "required": [
            ["甲胎蛋白", "AFP"],
            ["癌胚抗原", "CEA"],
            ["糖类抗原19-9", "CA19-9", "CA199"],
            ["糖类抗原125", "CA125"],
            ["糖类抗原15-3", "CA15-3", "CA153"],
            ["前列腺特异性抗原", "PSA"],
        ],
        "max_age_days": 365,
    },
    "cardiac_enzymes": {
        "required": [
            ["肌酸激酶", "CK"],
            ["肌酸激酶同工酶", "CK-MB"],
            ["乳酸脱氢酶", "LDH"],
            ["肌钙蛋白", "cTnI", "高敏肌钙蛋白", "hs-cTnI"],
            ["脑钠肽", "BNP", "NT-proBNP", "N端前脑钠肽"],
        ],
        "max_age_days": 180,
    },
    "electrolytes": {
        "required": [
            ["钾", "K+", "钾离子"],
            ["钠", "Na+", "钠离子"],
            ["氯", "Cl-", "氯离子"],
            ["钙", "Ca", "血钙"],
            ["镁", "Mg"],
            ["磷", "无机磷", "Phosphorus"],
        ],
        "max_age_days": 90,
    },
    "uric_acid": {
        "required": [
            ["尿酸", "UA", "血尿酸"],
        ],
        "max_age_days": 365,
    },
    "inflammation": {
        "required": [
            ["C反应蛋白", "CRP", "超敏C反应蛋白"],
            ["血沉", "ESR"],
            ["降钙素原", "PCT"],
            ["白细胞介素-6", "IL-6"],
        ],
        "max_age_days": 90,
    },
    "urine_routine": {
        "required": [
            ["尿蛋白", "PRO"],
            ["尿糖", "尿葡萄糖"],
            ["尿酮体", "KET"],
            ["尿白细胞"],
            ["尿潜血", "尿红细胞", "BLD"],
        ],
        "max_age_days": 365,
    },
    "iron_metabolism": {
        "required": [
            ["血清铁", "Fe", "血铁"],
            ["铁蛋白", "Ferritin"],
            ["总铁结合力", "TIBC"],
            ["转铁蛋白", "Transferrin"],
        ],
        "max_age_days": 365,
    },
    "bone_metabolism": {
        "required": [
            ["25-羟基维生素D", "25-OH-VD", "维生素D"],
            ["甲状旁腺激素", "PTH"],
        ],
        "max_age_days": 365,
    },
    "hepatitis_b": {
        "required": [
            ["乙肝表面抗原", "HBsAg"],
            ["乙肝表面抗体", "HBsAb"],
            ["乙肝e抗原", "HBeAg"],
            ["乙肝e抗体", "HBeAb"],
            ["乙肝核心抗体", "HBcAb"],
        ],
        "max_age_days": 365,
    },
    "sex_hormones": {
        "required": [
            ["促卵泡激素", "FSH"],
            ["促黄体生成素", "LH"],
            ["雌二醇", "E2"],
            ["孕酮", "Progesterone"],
            ["睾酮", "Testosterone"],
            ["催乳素", "PRL"],
        ],
        "max_age_days": 180,
    },
    "autoimmune": {
        "required": [
            ["类风湿因子", "RF"],
            ["抗环瓜氨酸肽抗体", "抗CCP"],
            ["抗核抗体", "ANA"],
        ],
        "max_age_days": 365,
    },
    "blood_gas": {
        "required": [
            ["pH"],
            ["二氧化碳分压", "PaCO2"],
            ["氧分压", "PaO2"],
            ["碳酸氢根", "HCO3-"],
            ["乳酸", "Lac"],
        ],
        "max_age_days": 90,
    },
    "trace_elements": {
        "required": [
            ["锌", "Zn"],
            ["铜", "Cu"],
            ["硒", "Se"],
        ],
        "max_age_days": 365,
    },
    # 非检验面板：按事件类型 / fact.kind 判断档案里有没有这类记录。
    # 没有 required 指标清单，只有"有没有"的缺口检查。
    "imaging_review": {
        "match_event_types": ["imaging"],
        "gap_text": "档案中没有影像报告（CT / MRI / X光等）",
    },
    "surgery_history": {
        "match_kinds": ["procedure"],
        "gap_text": "档案中没有手术或操作记录",
    },
    "diagnosis": {
        "match_kinds": ["condition"],
        "gap_text": "档案中没有诊断记录（如慢病、出院诊断）",
    },
    "symptom_review": {
        "match_kinds": ["symptom"],
        "gap_text": "档案中没有症状自述记录",
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
    # Forward containment only: the fact's name must contain the panel alias.
    # The reverse direction (alias contains fact name) let a one-character OCR
    # fragment such as "白" satisfy 白蛋白, which then reported a made-up
    # "already tested 20 months ago" entry for ALB/GLB/TP.
    return any(needle in h for h in haystacks if h)


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
    max_age = panel.get("max_age_days", 365)
    today = today or date.today()

    # 非检验面板：按事件类型 / fact.kind 判断档案里有没有这类记录
    match_event_types = panel.get("match_event_types") or []
    match_kinds = panel.get("match_kinds") or []
    if match_event_types or match_kinds:
        present_types = {event.event_type for event in events}
        present_kinds = {fact.kind for event in events for fact in event.facts}
        has = True
        if match_event_types:
            has = has and all(t in present_types for t in match_event_types)
        if match_kinds:
            has = has and all(k in present_kinds for k in match_kinds)
        gap_text = panel.get("gap_text") or "档案中缺少相关记录"
        if not has:
            return GapReport(gaps=[gap_text], panel_known=True, observed_concepts=observed)
        return GapReport(gaps=[], panel_known=True, observed_concepts=observed)

    required = panel["required"]

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
