"""按问题做确定性检索：手术史、用药史。

存在性类问题（"有没有做过手术""有没有吃过某个药"）交给模型去翻上下文，
模型倾向于补一个"通常会有"的答案。这里先在事件里把答案查出来，
再作为事实交给模型，让它只负责表述。
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

from .series_aggregator import AggregatedContext


# 剂型与包装后缀，比较药名时剥掉，避免"肝泰乐片"和"肝泰乐"判成两个药
_DOSAGE_FORM_SUFFIXES = (
    "注射液", "注射剂", "滴眼液", "滴鼻液", "口服液", "混悬液", "乳膏", "软膏",
    "缓释片", "控释片", "分散片", "肠溶片", "薄膜衣片", "胶囊", "颗粒",
    "片剂", "喷雾剂", "气雾剂", "洗剂", "酊剂", "栓剂", "片", "丸", "散", "膏",
)

# 用户记不清药名时夹带的词，先剥掉再当检索前缀
# 用户记不清药名时夹带的词，先剥掉再当检索前缀
_HINT_NOISE = (
    "我有没有吃过", "我有没有", "有没有", "服用过", "吃过", "用过",
    "吃了多久", "吃了多长时间", "多长时间", "多久", "的药",
    "一种", "一个", "名字", "药名", "类似", "好像", "记得", "那个", "那种",
    "什么", "啥", "叫", "药", "吗", "呢",
)

_QUESTION_WORDS = (
    "我", "有没有", "是否", "做过", "做过吗", "之前", "以前", "曾经",
    "手术", "的吗", "了吗", "没有", "有没有做过",
)

_PUNCT_RE = re.compile(r"[，。、？！?,;；:：\s()（）]")

SIMILAR_FLOOR = 0.5


def _norm_drug(name: str) -> str:
    text = _PUNCT_RE.sub("", name or "")
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", text)
    for suffix in _DOSAGE_FORM_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _hint_variants(hint: str) -> list[str]:
    variants = [hint]
    for prefix in ("于", "的", "是", "有", "我", "就", "还", "那个"):
        if hint.startswith(prefix) and len(hint) > len(prefix):
            variants.append(hint[len(prefix):])
    return [v for v in variants if v]


def extract_drug_hint(question: str) -> str:
    """从问句里剥出药名线索，"有没有吃过叫肝什么的药" -> "肝"。"""
    text = question or ""
    for word in sorted(_HINT_NOISE, key=len, reverse=True):
        text = text.replace(word, "")
    return _PUNCT_RE.sub("", text)


def detect_lookup(question: str) -> dict | None:
    """判断这句话是不是在问"有没有 / 吃过多久"，与路由模型无关。"""
    text = question or ""
    if not text:
        return None

    asks_presence = any(w in text for w in ("有没有", "是否", "做过", "吃过", "服用过", "用过", "吃了多久", "多久"))
    if not asks_presence:
        return None

    if "手术" in text:
        return {"kind": "procedure"}
    if any(w in text for w in ("药", "吃过", "服用", "用药")):
        return {"kind": "medication", "hint": extract_drug_hint(text)}
    return None


def _drug_score(hint: str, drug: str) -> tuple[float, str]:
    q = _norm_drug(hint)
    d = _norm_drug(drug)
    if not q or not d:
        return 0.0, "none"
    if q == d:
        return 1.0, "exact"
    if q in d:
        return 0.85, "prefix" if d.startswith(q) else "contains"
    if d in q:
        return 0.7, "contained"
    ratio = SequenceMatcher(None, q, d).ratio()
    return (ratio, "similar") if ratio >= SIMILAR_FLOOR else (ratio, "none")


def _span_text(days: int) -> str:
    months = days / 30.44
    return f"约 {months:.1f} 个月" if months >= 1 else f"{days} 天"


_FREQ_PATTERNS = (
    (re.compile(r"隔日|隔天|每两日|每2日|qod|every other", re.I), 0.5),
    (re.compile(r"每\s*(?:周|星期)"), 1 / 7),
    (re.compile(r"每\s*月"), 1 / 30),
    (re.compile(r"每\s*[日天]"), 1.0),
)
_FREQ_RATIO_RE = re.compile(r"(\d+)\s*[日天]\s*(\d+)\s*次")
_FREQ_PER_WEEK_RE = re.compile(r"每\s*(?:周|星期)\s*(\d+)\s*次")


def _dose_multiplier(frequency: str | None) -> float | None:
    """把频次换算成"每过 1 天，实际吃到的天数占比"。

    "两天吃一次"得来 0.5，"每周一次"得来 1/7，换算不出来的返回 None。
    """
    text = (frequency or "").strip()
    if not text:
        return None
    ratio = _FREQ_RATIO_RE.search(text)
    if ratio:
        every, times = int(ratio.group(1)), int(ratio.group(2))
        if every > 0 and times > 0:
            return min(1.0, times / every)
    per_week = _FREQ_PER_WEEK_RE.search(text)
    if per_week:
        return min(1.0, int(per_week.group(1)) / 7)
    for pattern, multiplier in _FREQ_PATTERNS:
        if pattern.search(text):
            return multiplier
    return None


def _course_info(course, frequency: str | None, is_last: bool) -> dict:
    """一段用药期的时间跨度。只用记录本身，不往"今天"外推。"""
    since, until, last = course.since, course.until, getattr(course, "last", None)
    days = None
    if since:
        start = date.fromisoformat(since[:10])
        end_text = until or (last if course.records > 1 else None)
        if end_text:
            days = (date.fromisoformat(end_text[:10]) - start).days
    # 没有停药记录又在最后一段：不能断定已经停了，也不能断定还在吃
    ongoing = is_last and not until
    dose_days = None
    multiplier = _dose_multiplier(frequency)
    if days is not None and multiplier is not None:
        dose_days = round(days * multiplier)
    return {
        "since": since,
        "until": until,
        "last": last,
        "days": days,
        "dose_days": dose_days,
        "records": course.records,
        "ongoing": ongoing,
    }


def _med_duration(med, today: date) -> dict:
    """按疗程算时长：用药期有多长、实际吃了多少天，分开说。"""
    courses = getattr(med, "courses", None) or []
    if not courses:
        return {
            "days": None, "span_days": None, "dose_days": None,
            "text": None, "ongoing": False, "courses": [],
        }

    infos = [
        _course_info(course, med.frequency, is_last=(index == len(courses) - 1))
        for index, course in enumerate(courses)
    ]
    dated = [info for info in infos if info["since"]]

    if not dated:
        return {
            "days": None, "span_days": None, "dose_days": None,
            "text": "记录里没有写用药日期，算不出时长",
            "ongoing": False, "courses": infos,
        }

    total_days = sum(info["days"] for info in infos if info["days"]) if any(info["days"] for info in infos) else None
    dose_days = sum(info["dose_days"] for info in infos if info["dose_days"]) if any(info["dose_days"] for info in infos) else None
    first = date.fromisoformat(dated[0]["since"][:10])
    last_text = dated[-1]["until"] or dated[-1]["last"]
    last_day = date.fromisoformat(last_text[:10]) if last_text else first
    span_days = (last_day - first).days if last_day > first else None
    ongoing = any(info["ongoing"] for info in infos)

    pieces: list[str] = []
    for index, info in enumerate(dated, start=1):
        if len(dated) > 1:
            if info["days"] is None:
                piece = f"第{index}段 {info['since']} 只有一次记录"
            else:
                end = info["until"] or info["last"]
                piece = f"第{index}段 {info['since']}～{end}（{info['days']} 天）"
        elif info["days"] is None:
            piece = f"{info['since']} 只有一次记录，没有起止"
        else:
            end = info["until"] or info["last"]
            piece = f"{info['since']}～{end}，用药期{_span_text(info['days'])}"
            if info["dose_days"] is not None:
                piece += f"，按{med.frequency}算约 {info['dose_days']} 天在服药"
        pieces.append(piece)

    if len(dated) > 1:
        text = f"共 {len(dated)} 段：" + "；".join(pieces) + "。"
        if dose_days is not None:
            text += f"估计服药 {dose_days} 天（{med.frequency}）"
        elif total_days is not None:
            text += f"累计用药期{_span_text(total_days)}"
    else:
        text = pieces[0]
    if ongoing:
        text += "；最后一段没有停药记录"

    return {
        "days": total_days,
        "span_days": span_days,
        "dose_days": dose_days,
        "text": text,
        "ongoing": ongoing,
        "courses": infos,
    }


def _procedure_keyword(question: str) -> str | None:
    text = question or ""
    for word in _QUESTION_WORDS:
        text = text.replace(word, "")
    text = _PUNCT_RE.sub("", text)
    return text if len(text) >= 2 else None


def lookup_procedures(agg: AggregatedContext, question: str, keyword: str | None = None) -> dict:
    keyword = keyword if keyword is not None else _procedure_keyword(question)
    matched = []
    for record in agg.procedures:
        if keyword and keyword not in record.name:
            continue
        matched.append({
            "name": record.name,
            "date": record.date,
            "institution": record.institution,
            "source": record.source,
            "fact_id": record.fact_id,
            "event_id": record.event_id,
        })

    result = {
        "kind": "procedure",
        "keyword": keyword,
        "matched": matched,
        "matched_count": len(matched),
        "total_in_file": len(agg.procedures),
    }
    if not matched:
        if agg.procedures:
            result["note"] = (
                f"档案里有 {len(agg.procedures)} 条手术相关记录，但没有一条包含'{keyword}'"
                if keyword
                else "档案里没有手术相关记录"
            )
        else:
            result["note"] = "档案里没有手术相关记录"
    return result


def _med_item(med, today: date) -> dict:
    duration = _med_duration(med, today)
    return {
        "drug": med.drug,
        "dose": med.dose,
        "frequency": med.frequency,
        "route": med.route,
        "quantity": med.quantity,
        "since": med.since,
        "stopped": med.stopped,
        "duration_days": duration["days"],
        "span_days": duration["span_days"],
        "dose_days": duration["dose_days"],
        "duration_text": duration["text"],
        "courses": duration["courses"],
        "ongoing": duration["ongoing"],
        "fact_id": med.fact_id,
    }


def lookup_medications(agg: AggregatedContext, question: str, today: date, hint: str | None = None) -> dict:
    hint = extract_drug_hint(question) if hint is None else hint
    variants = _hint_variants(hint)

    scored = []
    for med in agg.medications:
        best = (0.0, "none")
        for variant in variants:
            score, kind = _drug_score(variant, med.drug)
            if score > best[0]:
                best = (score, kind)
        if best[0] <= 0:
            continue
        item = _med_item(med, today)
        item["score"] = round(best[0], 2)
        item["match"] = best[1]
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)

    # 单字线索只当候选：库里以"肝"开头的药可能不止一个，不能替用户认定
    single_char_hint = len(_norm_drug(hint)) <= 1
    matched = [item for item in scored if not single_char_hint and item["match"] in ("exact", "prefix", "contains")]
    candidates = [item for item in scored if item not in matched]

    result = {
        "kind": "medication",
        "hint": hint,
        "matched": matched,
        "candidates": candidates[:5],
        "matched_count": len(matched),
        "total_in_file": len(agg.medications),
    }
    if not matched and not candidates:
        result["note"] = (
            f"档案里的 {len(agg.medications)} 种药都不含'{hint}'"
            if agg.medications and hint
            else "档案里没有用药记录"
        )
    elif not matched and candidates:
        result["note"] = "没有完全对上的药名，下面是相近的候选，需要用户确认是哪一种"
    return result


def lookup_records(question: str, agg: AggregatedContext, *, today: date | None = None) -> dict | None:
    """正则兜底：路由没有给检索计划时用，判断依据是问句本身。"""
    today = today or date.today()
    request = detect_lookup(question)
    if not request:
        return None
    if request["kind"] == "procedure":
        return lookup_procedures(agg, question)
    return lookup_medications(agg, question, today)


_MED_FIELDS = ("dose", "frequency", "route", "quantity", "since")


def lookup_medication_list(agg: AggregatedContext, *, scope: str = "all", today: date) -> dict:
    """用药清单：不问某一种药，而是把档案里记过的药全部列出来。"""
    items = [_med_item(med, today) for med in agg.medications]
    if scope == "active":
        items = [item for item in items if item["ongoing"]]
    items.sort(key=lambda item: item["since"] or "", reverse=True)

    missing = {f: sum(1 for item in items if not item.get(f)) for f in _MED_FIELDS}
    result = {
        "kind": "medication",
        "mode": "list",
        "scope": scope,
        "matched": items,
        "matched_count": len(items),
        "total_in_file": len(agg.medications),
        "missing_fields": {k: v for k, v in missing.items() if v},
    }
    if not items:
        result["note"] = "档案里的用药都已停用" if agg.medications else "档案里没有用药记录"
    return result


def lookup_procedure_list(agg: AggregatedContext) -> dict:
    matched = [
        {
            "name": record.name,
            "date": record.date,
            "institution": record.institution,
            "source": record.source,
            "fact_id": record.fact_id,
            "event_id": record.event_id,
        }
        for record in agg.procedures
    ]
    result = {
        "kind": "procedure",
        "mode": "list",
        "matched": matched,
        "matched_count": len(matched),
        "total_in_file": len(agg.procedures),
    }
    if not matched:
        result["note"] = "档案里没有手术相关记录"
    return result


LOOKUP_KINDS = ("medication", "procedure")
LOOKUP_MODES = ("presence", "list", "duration")


def lookup_by_plan(plan: dict | None, agg: AggregatedContext, *, today: date | None = None) -> dict | None:
    """按路由给出的检索计划执行，不再从问句里猜要不要查。"""
    if not plan or not plan.get("need_lookup"):
        return None
    kind = plan.get("kind")
    mode = plan.get("mode") or "presence"
    if kind not in LOOKUP_KINDS or mode not in LOOKUP_MODES:
        return None

    today = today or date.today()
    hint = (plan.get("hint") or "").strip()
    scope = plan.get("scope") or "all"

    if kind == "procedure":
        # 手术没有"在用/已停"这个概念，scope 一律按全部记录查，但要写回结果，
        # 免得传了 active 却什么都不发生、调用方还以为过滤生效了
        result = (
            lookup_procedure_list(agg)
            if mode == "list" and not hint
            else lookup_procedures(agg, "", keyword=hint or None)
        )
        result["scope"] = "all"
        if scope == "active":
            note = "手术记录没有在用/停用的区分，按全部记录查"
            result["note"] = f"{result['note']}；{note}" if result.get("note") else note
        return result
    if not hint:
        # 问"用过哪些药"却没给药名，只能列清单；按空线索匹配会把每种药都判成 0 分
        return lookup_medication_list(agg, scope=scope, today=today)
    return lookup_medications(agg, "", today, hint=hint)


def lookup_drug_names(result: dict | None) -> list[str]:
    """检索结果里出现过的药名（含候选），用于核对答案有没有提到档案外的药。"""
    if not result or result.get("kind") != "medication":
        return []
    names = []
    for item in (result.get("matched") or []) + (result.get("candidates") or []):
        if item.get("drug"):
            names.append(_norm_drug(item["drug"]))
    return [name for name in dict.fromkeys(names) if name]


def normalize_drug(name: str) -> str:
    return _norm_drug(name)


def lookup_fact_ids(result: dict | None) -> list[str]:
    """检索结果里带 fact_id 的条目，用于校验答案有没有真的引用它们。"""
    if not result:
        return []
    ids = [item["fact_id"] for item in result.get("matched", []) if item.get("fact_id")]
    return list(dict.fromkeys(ids))
