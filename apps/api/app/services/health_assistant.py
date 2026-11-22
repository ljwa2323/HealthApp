from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import AsyncIterator, Any

from openai import AsyncOpenAI

from ..config import get_deepseek_api_key
from .gap_extractor import extract_data_gaps
from .intent_router import route_question
from .record_lookup import (
    lookup_by_plan,
    lookup_drug_names,
    lookup_fact_ids,
    lookup_records,
    normalize_drug,
)
from .series_aggregator import aggregate_context


SYSTEM_PROMPT = (
    "你是 HealthApp 的健康档案助手。回答要短：默认 200 字以内，问用药清单最多 300 字。\n"
    "第一句直接给结论，后面只留支撑结论的记录。不用 Markdown 标题，不分小节，不复述问题，"
    "不解释自己怎么查的，不提字段名（lookup_result、data_gaps 之类）。"
    "列表只在真有多条要列时用，不超过 5 条；一句话能说完就一句话。"
    "只回答问到的事：问用药就只讲用药，不要顺带评价其他指标、不要补体检建议。\n"
    "\n"
    "内容分两层：\n"
    "档案层：检验数值、诊断、用药、手术、症状只能来自上下文，数值后标 [fact_id]，如'肌酐 96 μmol/L [f_xxx]'，"
    "上下文里没有的不许补。\n"
    "常识层：解释指标、说明一般处理原则可以用通用医学常识，但要用'一般来说'这类说法点明，"
    "不写具体指南名称或版本，不给精确百分比（如'五年生存率 70%'），只给定性判断，"
    "并说清与本人情况的差异和不确定的地方。两层不许混写。\n"
    "\n"
    "问'有没有做过某台手术''有没有吃过某种药'时以 lookup_result 为准："
    "有 matched 就只引用 matched 里的条目并标 id，叙述性记录标事件 id（如 [evt_xxx]）；"
    "matched 为空直说档案里没查到；只有 candidates 时把候选列出来让用户确认，不要替用户认定。"
    "问'我的用药情况怎么样'这类清单时把 matched 里的药全部列出（药名、剂量频次、起止、是否还在用），"
    "答案里的药名只能来自 lookup_result 或 active_medications。"
    "讲时长按 duration_text：先说服药期多久，再说按频次折算的服药天数；"
    "多段就分段说，中间空档不算在吃药。\n"
    "\n"
    "问'要不要做某手术''该不该用某药''怎么办'这类问题时，按三步写，每步一到两句："
    "档案里支持什么、缺什么；通用层面的考量（用通用医学常识，标'一般来说'）；需要面诊才能定。"
    "不要只回'资料不足'，也不要把常识写成对这个人下的结论。\n"
    "看指标时先说看到什么异常，再说还缺什么检查，最后说建议补什么；数据够了就直接给判断。\n"
    "引用标签只能是记录 id。回答用简体中文。"
)

_FACT_ID_RE = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# A number written right before a [fact_id] tag, e.g. "ALT 55 U/L [f_alt]".
# The gap may hold a unit, so it allows up to 12 chars but no newline, sentence
# punctuation or brackets: that keeps "ALT 55 U/L，AST 30 U/L [f_alt]" from
# pairing 55 with the other analyte's tag, and keeps a reference range written
# in brackets from being read as the measured value.
_NUM_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)([^\n\[\]，,。；;、（）()]{0,12})\[([a-zA-Z0-9_]+)\]")

# Words that mark the neighbouring number as a threshold rather than a result.
_RANGE_HINTS = ("参考", "正常", "范围", "上限", "下限", "高于", "低于", "超过")

# 模型偶尔会把 payload 的字段名当成引用标签写出来（"[lookup_result]"），
# 这类不是记录 id，计入幻觉会让前端误报，直接忽略。
_NON_CITABLE_TOKENS = {
    "lookup_result", "data_gaps", "data_quality_issues", "clinical_notes",
    "recent_events", "procedures", "active_medications", "series",
    "observed_concepts", "panel_known", "fact_id", "event_id", "context",
}

# 带剂型后缀的词当药名看，用来发现"档案里没有却写进答案"的药
_DRUG_FORM_RE = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9]{2,12}(?:缓释片|控释片|肠溶片|分散片|咀嚼片|胶囊|颗粒|"
    r"口服液|注射液|滴眼液|乳膏|软膏|糖浆|片|丸))"
)
_GENERIC_FORM_WORDS = {"药片", "片剂", "胶囊剂", "颗粒剂", "药物", "药品", "中成药", "西药", "口服药"}
# 以"片"结尾但不是药的常用词，避免把"药盒照片"当成药名
_NON_DRUG_FORM_WORDS = {
    "照片", "图片", "卡片", "芯片", "名片", "胶片", "底片", "相片", "镜片",
    "刀片", "纸片", "铁片", "大片", "贴片", "切片", "药盒照片",
}
_ADVICE_WORDS = ("建议", "可以", "考虑", "推荐", "如需", "必要时", "可尝试", "应服用", "也可")
# 判定"这是在说某种药"的上下文，避免把照片、卡片这类词算成药名
_DRUG_CONTEXT_WORDS = ("服用", "口服", "吃过", "用药", "剂量", "每日", "每次", "一片", "半片", "停药", "开药")
# 剂量写法常以"片"结尾：每次1片、每日2片、一次半片
_DOSAGE_PREFIXES = ("每次", "每日", "每天", "一天", "一次", "一日", "每回")
_DOSAGE_ONLY_RE = re.compile(r"^(?:每次|每日|每天|一天|一次|一日|每回)?[0-9一二两三四半.]{1,4}$")


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


def _build_context_payload(events, route_result, agg, gap_report, lookup=None) -> dict[str, Any]:
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
            {
                "drug": m.drug,
                "dose": m.dose,
                "frequency": m.frequency,
                "route": m.route,
                "quantity": m.quantity,
                "since": m.since,
                "stopped": m.stopped,
                "courses": [
                    {"since": c.since, "until": c.until, "records": c.records}
                    for c in m.courses
                ],
                "fact_id": m.fact_id,
            }
            for m in agg.active_medications
        ],
        "procedures": [
            {
                "name": p.name,
                "date": p.date,
                "institution": p.institution,
                "source": p.source,
                "fact_id": p.fact_id,
                "event_id": p.event_id,
            }
            for p in agg.procedures
        ],
        "recent_events": agg.recent_events,
        "clinical_notes": agg.clinical_notes,
        "lookup_result": lookup,
        "data_gaps": gap_report.gaps,
        "data_quality_issues": detect_data_quality_issues(events) if events else [],
        "panel_known": gap_report.panel_known,
        "observed_concepts": gap_report.observed_concepts,
    }


def _collect_fact_values(agg) -> dict[str, float | None]:
    """fact_id -> recorded numeric value, for every fact the LLM may cite."""
    values: dict[str, float | None] = {}
    for s in agg.series:
        for p in s.points:
            values[p.fact_id] = p.v
    for m in agg.active_medications:
        # 同一味药有开头、停药多条记录，模型引哪一条都该算合法
        for fact_id in (m.fact_ids or [m.fact_id]):
            if fact_id:
                values.setdefault(fact_id, None)
    for n in agg.clinical_notes:
        if n.get("fact_id"):
            values.setdefault(n["fact_id"], None)
    # 叙述性记录没有 fact_id，只能按事件引用；事件 id 一律可引用
    for p in agg.procedures:
        if p.event_id:
            values.setdefault(p.event_id, None)
    for e in agg.recent_events:
        if e.get("event_id"):
            values.setdefault(e["event_id"], None)
    return values


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


def _values_close(answer_value: float, record_value: float) -> bool:
    """Allow rounding noise, reject a rewritten number."""
    return abs(answer_value - record_value) <= max(0.05, abs(record_value) * 0.02)


def validate_citations(
    answer: str,
    allowed_fact_ids: list[str],
    fact_values: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """检查答案里的 [fact_id] 引用是否有效，并核对引用旁边的数值。

    - 引用了不存在的 fact_id -> hallucinated_citations
    - 引用了存在的 fact_id 但把数值写错 -> mismatched_values
      只靠 id 校验发现不了后者：模型挂一个真实 id、配一个错数字，仍然"合法"。
    """
    allowed = set(allowed_fact_ids)
    cited = set(_FACT_ID_RE.findall(answer))
    hallucinated = sorted(cited - allowed - _NON_CITABLE_TOKENS)

    mismatched: list[dict[str, Any]] = []
    if fact_values:
        seen: set[tuple[str, str]] = set()
        for match in _NUM_PAIR_RE.finditer(answer):
            num_text, gap, fact_id = match.group(1), match.group(2), match.group(3)
            if fact_id not in allowed:
                continue
            # "0-40 [f_alt]" or "比上限 40 高 [f_alt]" is a reference bound,
            # not the value the model claims to have read.
            if any(hint in gap for hint in _RANGE_HINTS):
                continue
            prefix = answer[max(0, match.start() - 12):match.start()]
            if any(hint in prefix for hint in _RANGE_HINTS):
                continue
            record = fact_values.get(fact_id)
            if record is None:
                continue
            try:
                answer_value = float(num_text)
            except ValueError:
                continue
            if (fact_id, num_text) in seen:
                continue
            seen.add((fact_id, num_text))
            if not _values_close(answer_value, record):
                mismatched.append({
                    "fact_id": fact_id,
                    "answer_value": answer_value,
                    "record_value": record,
                })

    return {
        "cited_fact_ids": sorted(cited),
        "allowed_fact_ids": sorted(allowed),
        "hallucinated_citations": hallucinated,
        "hallucination_count": len(hallucinated),
        "mismatched_values": mismatched,
        "mismatch_count": len(mismatched),
    }


def _matches_allowed_drug(mentioned: str, allowed: set[str]) -> bool:
    """抽取出来的词可能带着前面的动词（"你在用肝泰乐片"），按包含关系比。"""
    norm = normalize_drug(mentioned)
    if not norm:
        return True
    if norm in allowed:
        return True
    return any(name and (name in norm or norm in name) for name in allowed)


def _drugs_in_answer(answer: str) -> list[str]:
    """从答案里挑出像药名的词（带剂型后缀），用来核对有没有提到档案外的药。"""
    found = []
    for match in _DRUG_FORM_RE.finditer(answer):
        name = match.group(1)
        if name in _GENERIC_FORM_WORDS or name in _NON_DRUG_FORM_WORDS:
            continue
        # "每次1片""每次半片"是剂量写法，不是药名；含字母数字的药名（维生素B12）仍要查
        if _DOSAGE_ONLY_RE.match(name) or name.startswith(_DOSAGE_PREFIXES):
            continue
        near = answer[max(0, match.start() - 15): match.end() + 15]
        if any(word in near for word in _ADVICE_WORDS):
            continue  # 建议/推荐里提到的药不算"记录里没有"
        if not any(word in near for word in _DRUG_CONTEXT_WORDS):
            continue  # 没有用药语境，多半是照片、卡片一类的词
        found.append(name)
    return found


def _mentions(name: str, answer: str) -> bool:
    """答案里提没提到这条记录。名字常被写短（"肝泰乐片"写成"肝泰乐"、
    "腹腔镜胆囊切除术"写成"胆囊切除"），按最长公共子串判断，不用全名匹配。"""
    if not name:
        return False
    if name in answer:
        return True
    longest = SequenceMatcher(None, name, answer).find_longest_match(
        0, len(name), 0, len(answer)
    ).size
    return longest >= max(3, len(name) // 2)


_AFFIRM_WORDS = {
    "procedure": ("做过", "接受过", "实施过", "有过手术"),
    "medication": ("吃过", "服用过", "吃过药", "一直在吃", "还在吃", "在服用"),
}
_DENY_WORDS = ("没有找到", "未找到", "没有相关记录", "没有这类记录", "未见", "查无")
# 答案已经说明"只能当候选、需要确认"时，附带的列举不算肯定说法
_HEDGE_RE = re.compile(r"没有完全对上|没有完全匹配|候选|需要.{0,4}确认|不确定|无法确认|待确认")
# 肯定词前面有否定词，是"不能确认你做过"这种写法，不算下结论
_NEGATION_RE = re.compile(r"不能|无法|没有|未见|并非|不确定|难以|存疑|是否|如果|假如|若|要是")


def _sentence_around(text: str, index: int) -> str:
    start = max(text.rfind(mark, 0, index) for mark in ("。", "；", "\n")) + 1
    ends = [pos for pos in (text.find(mark, index) for mark in ("。", "；", "\n")) if pos != -1]
    end = min(ends) if ends else len(text)
    return text[start:end], start


def check_lookup_consistency(answer: str, lookup: dict | None) -> dict | None:
    """检索已经把答案查出来了，看答案有没有跟着走。

    三种情况都算没跟：检索有命中、答案却没引用也没提到；检索没命中、
    答案却给了肯定说法（"你确实做过""你一直在吃"）；答案提到了档案里
    根本没有的药名。
    """
    if not lookup:
        return None

    matched = lookup.get("matched") or []
    matched_ids = set(lookup_fact_ids(lookup))
    cited = set(_FACT_ID_RE.findall(answer))
    names = [item.get("name") or item.get("drug") or "" for item in matched]
    named = any(_mentions(name, answer) for name in names)

    if lookup.get("kind") == "medication":
        allowed = {normalize_drug(n) for n in lookup_drug_names(lookup)}
        for mentioned in _drugs_in_answer(answer):
            if not _matches_allowed_drug(mentioned, allowed):
                return {"kind": "medication", "reason": f"答案提到的药不在档案记录里：{mentioned}"}

    if lookup.get("mode") == "list" and matched:
        if not (cited & matched_ids) and not named:
            return {"kind": lookup["kind"], "reason": f"检索到 {len(matched)} 条记录，答案一条都没有提到"}

    denies = re.search("|".join(_DENY_WORDS), answer)
    affirm = None
    for phrase in _AFFIRM_WORDS.get(lookup.get("kind"), ()):
        for match in re.finditer(phrase, answer):
            sentence, sentence_start = _sentence_around(answer, match.start())
            # 同一句里写着"只能算候选、需要确认"的，不算下了结论
            if _HEDGE_RE.search(sentence):
                continue
            # 肯定词前面有否定词，比如"不能确认你做过""是否做过还没核实"
            if _NEGATION_RE.search(sentence[: match.start() - sentence_start]):
                continue
            affirm = phrase
            break
        if affirm:
            break

    if matched:
        if not (cited & matched_ids) and not named:
            return {"kind": lookup["kind"], "reason": "检索到记录，但答案没有引用或提到它们"}
        if denies and not named:
            return {"kind": lookup["kind"], "reason": "检索到记录，答案却说没有"}
        return None

    if affirm and not denies:
        return {"kind": lookup["kind"], "reason": f"检索没有命中，答案却给出了肯定说法：{affirm}"}
    return None


_GUIDELINE_RE = re.compile(r"《[^》]{0,20}指南》|临床指南|诊治指南|专家共识|guideline", re.I)
_RATE_CLAIM_RE = re.compile(
    r"(生存率|存活率|成功率|有效率|复发率|发生率|风险|概率)[^。；\n]{0,10}?(\d+(?:\.\d+)?%|\d+(?:\.\d+)?成)"
)
_UNCERTAINTY_WORDS = (
    "不确定", "因人而异", "个体差异", "取决于", "不能确定", "仅供参考",
    "具体情况", "面诊", "需要医生", "视情况",
)


def check_advice_answer(answer: str, question_type: str | None) -> list[str]:
    """建议类回答的粗检：允许写通用常识，但不许写成有出处的硬结论。

    能查的只有三件事：有没有点名具体指南、有没有给没有记录支撑的比例、
    有没有交代不确定性。查不出"常识本身对不对"，那要靠人看。
    """
    if question_type != "advice":
        return []

    flags: list[str] = []
    if _GUIDELINE_RE.search(answer):
        flags.append("回答点名了具体指南或共识，档案里没有这类依据可核对")

    unsupported: list[str] = []
    for match in _RATE_CLAIM_RE.finditer(answer):
        # 挨着 [fact_id] 的数字属于患者档案，不算编造
        if _FACT_ID_RE.search(answer[max(0, match.start() - 10): match.end() + 8]):
            continue
        unsupported.append(match.group(0).strip())
    if unsupported:
        flags.append("回答给了没有记录支撑的比例或概率：" + "、".join(dict.fromkeys(unsupported)))

    if not any(word in answer for word in _UNCERTAINTY_WORDS):
        flags.append("建议类回答没有说明不确定性和适用条件")
    return flags


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
    """一句话收尾，不再复述上文已经写过的内容。"""
    if confidence >= 0.8:
        return f"\n\n置信度 {confidence:.2f} · 仅供参考，具体请咨询医生"
    if confidence >= 0.5:
        return f"\n\n置信度 {confidence:.2f} · 资料不全，仅供参考"
    return f"\n\n置信度 {confidence:.2f} · 资料不足，仅供参考"


async def _prepare(
    question: str,
    events: list,
    route=None,
    history: list[dict[str, str]] | None = None,
) -> tuple[dict, list[dict[str, str]], list[str], dict[str, float | None]]:
    """路由 + 聚合 + 缺口 + 拼 messages。

    返回 (context_payload, messages, used_fact_ids, fact_values)。
    """
    if route is None:
        route = await route_question(question)
    agg = aggregate_context(
        events=events,
        topic=route.topic,
        matched_concepts=route.matched_concepts,
        time_start=route.time_start,
        time_end=route.time_end,
    )
    gap_report = extract_data_gaps(events=events, topic=route.topic)
    lookup = lookup_by_plan(getattr(route, "retrieval", None), agg)
    if lookup is None:
        # 路由没给计划（或字段不合法）时，用问句正则兜底
        lookup = lookup_records(question, agg)
    context_payload = _build_context_payload(events, route, agg, gap_report, lookup)
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
    fact_values = _collect_fact_values(agg)
    used = list(fact_values)

    # 对话历史要排在当前问题之前，否则模型看到的顺序是"先问后答"
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": (
            "结构化患者上下文如下。档案事实以它为准；通用医学常识按系统说明单独成段写。\n\n"
            f"CONTEXT_JSON:\n{context_json}\n\n"
            f"患者问题：{question}"
        ),
    })
    return context_payload, messages, used, fact_values


async def answer_question(
    *,
    question: str,
    events: list,
    history: list[dict[str, str]] | None = None,
    route=None,
) -> dict[str, Any]:
    context_payload, messages, used, fact_values = await _prepare(question, events, route, history)

    client = _client()
    completion = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        stream=False,
    )
    answer = (completion.choices[0].message.content or "").strip()
    citations = validate_citations(answer, used, fact_values)
    lookup_check = check_lookup_consistency(answer, context_payload.get("lookup_result"))
    if lookup_check:
        citations["lookup_conflict"] = lookup_check
    advice_flags = check_advice_answer(answer, context_payload.get("question_type"))
    if advice_flags:
        citations["advice_flags"] = advice_flags
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
    route=None,
) -> AsyncIterator[dict[str, Any]]:
    """流式版本：先 yield 路由/上下文元数据，再逐块 yield 答案文本，最后 yield 校验结果。"""
    context_payload, messages, used, fact_values = await _prepare(question, events, route, history)

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
        temperature=0.0,
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
    citations = validate_citations(full_answer, used, fact_values)
    lookup_check = check_lookup_consistency(full_answer, context_payload.get("lookup_result"))
    if lookup_check:
        citations["lookup_conflict"] = lookup_check
    advice_flags = check_advice_answer(full_answer, context_payload.get("question_type"))
    if advice_flags:
        citations["advice_flags"] = advice_flags
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
