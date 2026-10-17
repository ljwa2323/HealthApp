# -*- coding: utf-8 -*-
"""Cross-check AI-extracted facts against the raw OCR text.

The vision model (GLM) reads the image directly and extracts structured
facts; RapidOCR gives an independent second reading of the printed text.
Comparing the two surfaces items the user should double-check (OCR failed
to read something, or the two readings disagree) without blocking the
flow — the user reviews and edits the facts anyway.

Per-fact outcome:
- verified:  the concept and its value were both found in the OCR text.
- not_found: the concept was not found in the OCR text (handwriting or
             tiny grid cells OCR could not read; the AI read it from
             the pixels directly).
- conflict:  the concept was found, but the numbers/text seen near it in
             the OCR context differ from the AI value.
"""
from __future__ import annotations

import re
from typing import Any

from ..schemas import ClinicalFact

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Full-width -> half-width for digits/letters and common punctuation so
# OCR output and AI values can be compared in a normalized space.
_FULLWIDTH = {
    ord(c): ord(half)
    for c, half in zip(
        "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ：；，。／．－（）（）％＋－",
        "0123456789abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ:;,./.-()()%+-",
    )
}
_FULLWIDTH.setdefault(ord("　"), ord(" "))
_FULLWIDTH.setdefault(ord("："), ord(":"))
_FULLWIDTH.setdefault(ord("；"), ord(";"))
_FULLWIDTH.setdefault(ord("，"), ord(","))
_FULLWIDTH.setdefault(ord("。"), ord("."))
_FULLWIDTH.setdefault(ord("／"), ord("/"))
_FULLWIDTH.setdefault(ord("．"), ord("."))
_FULLWIDTH.setdefault(ord("－"), ord("-"))
_FULLWIDTH.setdefault(ord("（"), ord("("))
_FULLWIDTH.setdefault(ord("）"), ord(")"))
_FULLWIDTH.setdefault(ord("％"), ord("%"))
_FULLWIDTH.setdefault(ord("＋"), ord("+"))


def _norm(text: str) -> str:
    text = text.translate(_FULLWIDTH).lower()
    # Keep digits, letters and . - / : + % ; spaces; drop other punctuation.
    return re.sub(r"[^\w.\-/:%+;\s]", "", text, flags=re.UNICODE)


def _fact_value_text(fact: ClinicalFact) -> str:
    value = fact.value
    if value is None:
        return ""
    if value.type == "quantity":
        return str(value.number)
    if value.type == "code":
        return str(value.code)
    return str(getattr(value, "text", "") or "")


def _context_window(norm_text: str, pos: int, size: int = 300) -> str:
    """Slice around a concept occurrence; the value usually sits on the
    same line or within the next few rows of a report table."""
    start = max(0, pos - 60)
    return norm_text[start : pos + size]


def cross_check_facts(
    ocr_text: str | None,
    facts: list[ClinicalFact],
) -> list[dict[str, Any]]:
    if not ocr_text or not facts:
        return []
    norm_text = _norm(ocr_text)
    results: list[dict[str, Any]] = []
    for fact in facts:
        concept = (fact.concept.display or "").strip()
        if not concept:
            continue
        c_norm = _norm(concept)
        if not c_norm:
            continue
        pos = norm_text.find(c_norm)
        if pos < 0:
            results.append(
                {
                    "fact_id": fact.fact_id,
                    "concept": concept,
                    "status": "not_found",
                    "detail": f"OCR 未读到「{concept}」，可能为手写或 OCR 漏读，"
                    "AI 基于图像直接判读，请人工核对",
                }
            )
            continue

        window = _context_window(norm_text, pos)
        value_text = _fact_value_text(fact)
        value_norm = _norm(value_text)
        val_tokens = _NUMBER_RE.findall(value_norm)

        if val_tokens:
            ctx_tokens = _NUMBER_RE.findall(window)

            def _num(t: str) -> float | None:
                try:
                    return float(t)
                except ValueError:
                    return None

            # Compare numerically: AI "135.0" must match OCR "135".
            ctx_values = {v for v in (_num(t) for t in ctx_tokens) if v is not None}
            matched = [t for t in val_tokens if _num(t) in ctx_values]
            if len(matched) == len(val_tokens):
                results.append(
                    {
                        "fact_id": fact.fact_id,
                        "concept": concept,
                        "status": "verified",
                        "detail": f"「{concept}」数值与 OCR 文本一致",
                    }
                )
            else:
                seen = sorted(ctx_tokens, key=len, reverse=True)[:6]
                results.append(
                    {
                        "fact_id": fact.fact_id,
                        "concept": concept,
                        "status": "conflict",
                        "detail": f"「{concept}」附近 OCR 读到 "
                        f"{'/'.join(seen) if seen else '其他内容'}，与 AI 提取值"
                        f"（{value_text}）不一致，请人工核对",
                    }
                )
        else:
            # Text result (e.g. 阴性/阳性/未见): substring match in context.
            if value_norm and (
                value_norm in window or value_norm in norm_text
            ):
                results.append(
                    {
                        "fact_id": fact.fact_id,
                        "concept": concept,
                        "status": "verified",
                        "detail": f"「{concept}」内容与 OCR 文本一致",
                    }
                )
            else:
                results.append(
                    {
                        "fact_id": fact.fact_id,
                        "concept": concept,
                        "status": "not_found",
                        "detail": f"「{concept}」的值未在 OCR 文本中找到，"
                        "AI 基于图像直接判读，请人工核对",
                    }
                )
    return results
