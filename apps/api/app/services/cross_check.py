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

from ..schemas import ClinicalFact, EvidencePointer, OCRBlock

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Reference ranges as printed on reports: "3.5-9.5", "3.5~9.5", "3.5至9.5".
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~—至]\s*(\d+(?:\.\d+)?)")

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
        # Keep the unit in the comparison string: "135 g/L" must not pass
        # against an OCR context that only carries the garbled "135 7/6".
        unit = getattr(value, "unit", None) or ""
        return f"{value.number} {unit}".strip()
    if value.type == "code":
        return str(value.code)
    return str(getattr(value, "text", "") or "")


def _unit_ok(value_norm: str, window: str) -> bool:
    """The unit part of a quantity (letters, /, %, ^) must appear in the
    context window, or the cross-check cannot confirm the unit."""
    unit_part = re.sub(r"[\d\s.\-:]+", "", value_norm)
    if not unit_part or len(unit_part) < 2:
        return True  # no meaningful unit to verify (e.g. pure count)
    return unit_part in window


def _reference_mismatch(
    fact: ClinicalFact,
    window: str,
    value_floats: list[float],
    value_norm: str,
) -> tuple[bool, str]:
    """Compare the fact's reference range with a range printed in the window.

    Only the range printed NEXT TO the fact's own value is compared: on a
    multi-column report the 300-char window also covers neighbouring rows,
    and their ranges must not be mistaken for this fact's. When OCR never
    read a range beside the value, nothing is reported — reports often crop
    or garble the reference column, and a missing range is not evidence.
    """
    ref = fact.reference_range
    if ref is None:
        return False, ""
    if ref.low is not None and ref.high is not None:
        lo, hi = float(ref.low), float(ref.high)
    else:
        parsed = _RANGE_RE.search(_norm(str(ref.text or "")))
        if not parsed:
            return False, ""
        lo, hi = float(parsed.group(1)), float(parsed.group(2))

    # Anchor on the fact value inside the window, then read the range that
    # follows it (printed values sit left of their reference range).
    anchor = -1
    if value_floats:
        for match in _NUMBER_RE.finditer(window):
            if any(_near(v, float(match.group())) for v in value_floats):
                anchor = match.start()
                break
    elif value_norm:
        anchor = window.find(value_norm)
    if anchor < 0:
        return False, ""
    tail = window[anchor : anchor + 80]
    win_ranges = _RANGE_RE.findall(tail)
    if not win_ranges:
        return False, ""
    for low_txt, high_txt in win_ranges:
        if _near(lo, float(low_txt)) and _near(hi, float(high_txt)):
            return False, ""
    return True, f"{win_ranges[0][0]}-{win_ranges[0][1]}"


def _context_window(norm_text: str, pos: int, size: int = 300) -> str:
    """Slice around a concept occurrence; the value usually sits on the
    same line or within the next few rows of a report table."""
    start = max(0, pos - 60)
    return norm_text[start : pos + size]


def _near(a: float, b: float) -> bool:
    """Tolerance-aware comparison (same as the field-evaluation matcher)."""
    return abs(a - b) <= 0.5 or (a != 0 and abs(a - b) / abs(a) <= 0.05)


def _concept_tokens(c_norm: str) -> list[str]:
    """Split a concept like "ast谷草转氨酶" into matchable tokens.

    The vision model often emits combined names (EN abbrev + CN full name);
    OCR text usually contains only one form. Any token hit counts as found.
    """
    tokens = [t for t in re.split(r"[\s/]+", c_norm) if len(t) >= 2]
    return tokens


def locate_evidence(
    facts: list[ClinicalFact],
    blocks: list[OCRBlock] | list[dict[str, Any]],
    artifact_id: str,
) -> None:
    """Attach the OCR block that carries each fact's concept as evidence.

    The vision model returns no block ids. When the OCR text contains the
    concept, that block is a verifiable pointer to the source region; facts
    OCR never read (handwriting) keep the empty pointer and stay marked by
    the cross-check instead.

    Accepts OCRBlock models or the raw provider dicts, since the upload
    pipeline persists models while the split endpoint re-runs OCR in memory.
    """
    if not facts or not blocks:
        return
    prepared: list[tuple[str, int, str, str]] = []
    for block in blocks:
        if isinstance(block, dict):
            text = str(block.get("text") or "")
            block_id = str(block.get("block_id") or "")
            page = int(block.get("page") or 1)
        else:
            text = block.text or ""
            block_id = block.block_id or ""
            page = block.page or 1
        if text and block_id:
            prepared.append((block_id, page, text, _norm(text)))
    if not prepared:
        return
    for fact in facts:
        concept = (fact.concept.display or "").strip()
        if not concept:
            continue
        tokens = _concept_tokens(_norm(concept))
        if not tokens:
            continue
        hit: tuple[str, int, str, str] | None = None
        for token in tokens:
            for entry in prepared:
                if token in entry[3]:
                    hit = entry
                    break
            if hit is not None:
                break
        if hit is None:
            continue
        fact.evidence = [
            EvidencePointer(
                artifact_id=artifact_id,
                page=hit[1],
                block_id=hit[0],
                quote=hit[2][:160],
            )
        ]


# Medication facts must carry dose/frequency/route; the vision model tends
# to drop the "2 支 / 外用 / 日 3 次" part while keeping the drug name.
# Prompt rules already ask for them, so the remaining gap must not stay
# silent: flag missing fields for the user to complete against the image.
_MED_REQUIRED_ATTRS: dict[str, tuple[str, ...]] = {
    "dose": ("dose", "dosage", "剂量"),
    "frequency": ("frequency", "freq", "频次", "次数"),
    "route": ("route", "用法", "给药途径"),
}


def medication_missing_fields(facts: list[ClinicalFact]) -> list[dict[str, Any]]:
    """Return cross-check warnings for medication facts missing key fields."""
    warnings: list[dict[str, Any]] = []
    for fact in facts:
        if fact.kind != "medication":
            continue
        attrs = fact.attributes or {}
        missing: list[str] = []
        for label, keys in _MED_REQUIRED_ATTRS.items():
            if not any(str(attrs.get(k, "")).strip() for k in keys):
                missing.append(label)
        if missing:
            warnings.append(
                {
                    "fact_id": fact.fact_id,
                    "concept": fact.concept.display,
                    "status": "missing_field",
                    "detail": f"用药「{fact.concept.display}」缺少{', '.join(missing)}，"
                    "请对照原图补全（剂量/频次/用法）",
                }
            )
    return warnings


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
        tokens = _concept_tokens(c_norm)
        if not tokens:
            continue

        pos = -1
        for token in tokens:
            hit = norm_text.find(token)
            if hit >= 0:
                pos = hit
                break
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

            val_floats = [v for v in (_num(t) for t in val_tokens) if v is not None]
            ctx_floats = [v for v in (_num(t) for t in ctx_tokens) if v is not None]

            # Tolerance-aware: AI "135.0" matches OCR "135"; a nearby row's
            # number must NOT be required to match exactly (multi-column
            # OCR windows mix adjacent cells). Unit and reference range
            # must also agree.
            ref_bad, ocr_ref = _reference_mismatch(
                fact, window, val_floats, value_norm
            )
            if (
                val_floats
                and any(
                    _near(v, c) for v in val_floats for c in ctx_floats
                )
                and _unit_ok(value_norm, window)
                and not ref_bad
            ):
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
                reason = (
                    f"参考范围与 OCR 读到的「{ocr_ref}」不一致"
                    if ref_bad
                    else (
                        "单位与 OCR 附近内容不一致"
                        if val_floats and not _unit_ok(value_norm, window)
                        else "数值不一致"
                    )
                )
                results.append(
                    {
                        "fact_id": fact.fact_id,
                        "concept": concept,
                        "status": "conflict",
                        "detail": f"「{concept}」{reason}：OCR 附近读到 "
                        f"{'/'.join(seen) if seen else '其他内容'}，与 AI 提取值"
                        f"（{value_text}）不一致，请人工核对",
                    }
                )
        else:
            # Text result (e.g. 阴性/阳性/未见): must appear inside the
            # concept's own context window; a global substring hit elsewhere
            # in the document is not evidence for this fact.
            if value_norm and value_norm in window:
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
                        "detail": f"「{concept}」的值未在该概念附近的 OCR 文本中找到，"
                        "AI 基于图像直接判读，请人工核对",
                    }
                )
    return results


def cross_check_grid(
    ocr_matrix: list[list[str]],
    corrected_matrix: list[list[str]],
) -> list[dict[str, Any]]:
    """Per-cell OCR x AI cross-check for the table view.

    The vision model re-reads the image and corrects garbled cells; every
    cell that differs from the raw OCR reading is a place the two readings
    disagree — flag it so the user can verify the change before adding the
    record to the timeline. Cells the two readings agree on stay quiet.
    """
    results: list[dict[str, Any]] = []
    for r, row in enumerate(corrected_matrix):
        for c, corrected in enumerate(row):
            ocr_val = (
                ocr_matrix[r][c]
                if r < len(ocr_matrix) and c < len(ocr_matrix[r])
                else ""
            )
            c_norm = _norm(corrected)
            o_norm = _norm(ocr_val)
            if c_norm == o_norm or (o_norm and o_norm in c_norm):
                continue  # readings agree (or correction only embellished)
            if not c_norm and not o_norm:
                continue  # both empty
            if not c_norm:
                continue  # AI cleared a cell: keep it silent, user sees it
            if not o_norm and c_norm in {"-", "null", "none", "n/a", "无", "/"}:
                continue  # AI placeholder for an empty cell, not a reading
            results.append(
                {
                    "row": r,
                    "col": c,
                    "status": "conflict",
                    "ocr": ocr_val,
                    "corrected": corrected,
                    "detail": (
                        f"AI 将 OCR 值「{ocr_val or '-'}」纠正为「{corrected}」"
                        if ocr_val
                        else f"AI 补充了该格内容「{corrected}」，OCR 未读到"
                    ),
                }
            )
    return results
