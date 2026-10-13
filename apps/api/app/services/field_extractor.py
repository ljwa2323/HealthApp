"""Rule-based field extraction for HealthApp.

Turns OCR blocks (text + bbox + confidence) into schema-valid ClinicalEvent
objects WITHOUT any LLM: row clustering from block coordinates -> per-row
text -> regex + dictionary parsing (lab item, value, unit, reference range)
-> ClinicalFact assembly.

This is the zero-cost first stage of the "field extraction" line. A future
LLM normalizer can replace or wrap this adapter; the canonical data model
does not change.
"""

from __future__ import annotations

import re
import statistics
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..schemas import (
    ClinicalEvent,
    ClinicalFact,
    Concept,
    EventTime,
    EvidencePointer,
    QuantityValue,
    ReferenceRange,
    TextValue,
)
from .ingestion import ClinicalNormalizer, OCRResult, SourceArtifact

# ---------------------------------------------------------------------------
# Dictionaries (canonical display, local aliases)
# ---------------------------------------------------------------------------

# (canonical display, [aliases]) — aliases are matched case-insensitively
# against the name part of a row after whitespace/punctuation removal.
LAB_ITEMS: list[tuple[str, list[str]]] = [
    # CBC
    ("白细胞计数", ["白细胞计数", "白细胞", "WBC", "白血球"]),
    ("红细胞计数", ["红细胞计数", "红细胞", "RBC", "红血球"]),
    ("血红蛋白", ["血红蛋白", "HGB", "HB", "血色素"]),
    ("红细胞压积", ["红细胞压积", "红细胞比容", "HCT", "PCV"]),
    ("平均红细胞体积", ["平均红细胞体积", "MCV"]),
    ("平均血红蛋白量", ["平均血红蛋白量", "平均血红蛋白含量", "MCH"]),
    ("平均血红蛋白浓度", ["平均血红蛋白浓度", "MCHC"]),
    ("血小板计数", ["血小板计数", "血小板", "PLT"]),
    ("中性粒细胞百分比", ["中性粒细胞百分比", "中性粒细胞比率", "NEUT%", "NEUT"]),
    ("淋巴细胞百分比", ["淋巴细胞百分比", "淋巴细胞比率", "LYMPH%", "LYM%", "LYMPH"]),
    ("单核细胞百分比", ["单核细胞百分比", "单核细胞比率", "MONO%", "MONO"]),
    ("嗜酸性粒细胞百分比", ["嗜酸性粒细胞百分比", "嗜酸性粒细胞比率", "EO%", "EO"]),
    ("嗜碱性粒细胞百分比", ["嗜碱性粒细胞百分比", "嗜碱性粒细胞比率", "BASO%", "BASO"]),
    ("中性粒细胞绝对值", ["中性粒细胞绝对值", "NEUT#"]),
    ("淋巴细胞绝对值", ["淋巴细胞绝对值", "LYMPH#", "LYM#"]),
    # Liver
    ("谷丙转氨酶", ["谷丙转氨酶", "丙氨酸氨基转移酶", "ALT", "GPT"]),
    ("谷草转氨酶", ["谷草转氨酶", "天门冬氨酸氨基转移酶", "AST", "GOT"]),
    ("总胆红素", ["总胆红素", "TBIL", "TB"]),
    ("直接胆红素", ["直接胆红素", "DBIL", "DB"]),
    ("间接胆红素", ["间接胆红素", "IBIL", "IB"]),
    ("碱性磷酸酶", ["碱性磷酸酶", "ALP", "AKP"]),
    ("γ-谷氨酰转肽酶", ["Γ-谷氨酰转肽酶", "谷氨酰转肽酶", "γ谷氨酰转移酶", "GGT", "γ-GT"]),
    ("总蛋白", ["总蛋白", "TP"]),
    ("白蛋白", ["白蛋白", "ALB"]),
    ("球蛋白", ["球蛋白", "GLOB"]),
    ("白球比", ["白球比", "白蛋白/球蛋白", "A/G"]),
    ("胆碱酯酶", ["胆碱酯酶", "CHE"]),
    ("乙肝前S1抗原", ["乙肝前S1抗原", "乙型肝炎前S1抗原", "前S1抗原", "PRES1"]),
    # Kidney
    ("肌酐", ["肌酐", "血肌酐", "CREA", "CR", "CRE"]),
    ("尿素", ["尿素", "尿素氮", "血尿素", "BUN", "UREA"]),
    ("尿酸", ["尿酸", "UA", "URIC"]),
    ("胱抑素C", ["胱抑素C", "胱抑素c", "CYS-C", "CYSC"]),
    ("β2-微球蛋白", ["Β2-微球蛋白", "B2-微球蛋白", "微球蛋白β2", "Β2-MG"]),
    # Lipids
    ("总胆固醇", ["总胆固醇", "胆固醇", "TC", "CHOL"]),
    ("甘油三酯", ["甘油三酯", "TG", "TRIG"]),
    ("高密度脂蛋白胆固醇", ["高密度脂蛋白胆固醇", "高密度脂蛋白", "HDL-C", "HDL"]),
    ("低密度脂蛋白胆固醇", ["低密度脂蛋白胆固醇", "低密度脂蛋白", "LDL-C", "LDL"]),
    ("载脂蛋白A1", ["载脂蛋白A1", "载脂蛋白AI", "APOA1", "APO-A1"]),
    ("载脂蛋白B", ["载脂蛋白B", "APOB", "APO-B"]),
    ("脂蛋白a", ["脂蛋白A", "脂蛋白a", "LP(A)", "LPA"]),
    # Hepatitis / tumor markers
    ("乙肝表面抗原", ["乙肝表面抗原", "乙型肝炎表面抗原", "HBSAG", "乙肝表面抗原定量"]),
    ("乙肝表面抗体", ["乙肝表面抗体", "乙型肝炎表面抗体", "HBSAB", "抗-HBS"]),
    ("乙肝e抗原", ["乙肝E抗原", "乙型肝炎E抗原", "HBEAG"]),
    ("乙肝e抗体", ["乙肝E抗体", "乙型肝炎E抗体", "HBEAB", "抗-HBE"]),
    ("乙肝核心抗体", ["乙肝核心抗体", "乙型肝炎核心抗体", "HBCAB", "抗-HBC"]),
    ("乙肝核心IgM抗体", ["乙肝核心IGM抗体", "核心抗体IGM", "HBCAB-IGM", "HBCABIGM"]),
    ("甲肝抗体", ["甲肝抗体", "甲型肝炎抗体", "HAV", "HAV-IgM", "HAVIGM"]),
    ("丙肝抗体", ["丙肝抗体", "丙型肝炎抗体", "HCV", "抗-HCV", "HCVAB"]),
    ("甲胎蛋白", ["甲胎蛋白", "AFP"]),
    ("癌胚抗原", ["癌胚抗原", "CEA"]),
    ("糖类抗原125", ["糖类抗原125", "CA125"]),
    ("糖类抗原199", ["糖类抗原199", "CA199", "CA19-9"]),
    # Others
    ("血糖", ["血糖", "空腹血糖", "GLU", "GLUCOSE"]),
    ("糖化血红蛋白", ["糖化血红蛋白", "HBA1C", "GHB"]),
    ("C反应蛋白", ["C反应蛋白", "超敏C反应蛋白", "CRP", "HS-CRP"]),
    ("淀粉酶", ["淀粉酶", "AMY", "AMS"]),
    ("肌酸激酶", ["肌酸激酶", "CK"]),
    ("乳酸脱氢酶", ["乳酸脱氢酶", "LDH", "LD"]),
    ("钾", ["钾", "血钾", "血清钾", "K"]),
    ("钠", ["钠", "血钠", "血清钠", "NA"]),
    ("氯", ["氯", "血氯", "血清氯", "CL"]),
    ("钙", ["钙", "血钙", "血清钙", "CA"]),
    ("镁", ["镁", "血镁", "血清镁", "MG"]),
    ("磷", ["磷", "血磷", "血清磷", "P", "IP"]),
    ("铁", ["铁", "血清铁", "FE"]),
    ("总铁结合力", ["总铁结合力", "TIBC"]),
    ("铁蛋白", ["铁蛋白", "FERRITIN", "SF"]),
    ("叶酸", ["叶酸", "FA", "FOLATE"]),
    ("维生素B12", ["维生素B12", "维生素B12", "VITB12", "VB12"]),
    ("促甲状腺激素", ["促甲状腺激素", "TSH"]),
    ("游离三碘甲状腺原氨酸", ["游离三碘甲状腺原氨酸", "游离T3", "FT3"]),
    ("游离甲状腺素", ["游离甲状腺素", "游离T4", "FT4"]),
    ("三碘甲状腺原氨酸", ["三碘甲状腺原氨酸", "总T3", "TT3", "T3"]),
    ("甲状腺素", ["甲状腺素", "总T4", "TT4", "T4"]),
    ("尿蛋白", ["尿蛋白", "PRO"]),
    ("尿潜血", ["尿潜血", "潜血", "BLD", "OB"]),
    ("尿糖", ["尿糖", "GLU-U", "UGLU"]),
    ("尿酮体", ["尿酮体", "酮体", "KET"]),
]

# Aliases are matched against the normalized name part.
ALIAS_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in LAB_ITEMS:
    for _alias in _aliases:
        ALIAS_LOOKUP[_alias.upper()] = _canonical

# Units — longest match wins, so µ/mol variants are tried before short ones.
UNITS: list[str] = [
    "µmol/L", "μmol/L", "umol/L",
    "mmol/L", "mEq/L", "mg/dL", "mg/L", "ng/mL", "ng/ml",
    "pg/mL", "pg/ml", "µg/L", "μg/L", "ug/L", "µg/mL", "μg/mL",
    "×10⁹/L", "×10^9/L", "x10^9/L", "10^9/L", "10⁹/L",
    "×10¹²/L", "×10^12/L", "x10^12/L", "10^12/L", "10¹²/L",
    "U/L", "IU/L", "mIU/L", "mIU/mL", "IU/mL", "u/L", "ku/L", "kU/L", "KU/L",
    "g/L", "g/dL", "g%", "mL/min", "fL", "fL/红细胞", "pg", "%", "ng/dl",
]
UNITS_SORTED = sorted(UNITS, key=len, reverse=True)
# Normalized (whitespace/`-` removed, lowercased) unit -> canonical unit
UNIT_NORM: dict[str, str] = {}
for _u in UNITS:
    UNIT_NORM[re.sub(r"[\s\-/]", "", _u).lower()] = _u

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

REF_RANGE_RE = re.compile(
    r"(?:\(|\（)?\s*(?P<low>-?\d+\.?\d*)\s*[~～至\-]\s*"
    r"(?P<high>-?\d+\.?\d*)\s*(?:\)|\）)?"
)
REF_LT_RE = re.compile(r"(?:<|≤)\s*(?P<max>\d+\.?\d*)")
REF_GT_RE = re.compile(r"(?:>|≥)\s*(?P<min>\d+\.?\d*)")
NUMBER_RE = re.compile(r"(?P<prefix>>|<|≥|≤|-)?\s*(?P<num>\d+\.?\d*)")
DATE_RE = re.compile(
    r"(?:20\d{2}\s*[年./-]\s*\d{1,2}\s*[月./-]\s*\d{1,2}\s*日?)"
    r"|(?:\d{1,2}\s*月\s*\d{1,2}\s*日)"
)
DATE_KEYWORDS = (
    "检验日期", "送检时间", "采样时间", "标本采集时间", "采集时间",
    "检验时间", "检查日期", "检查时间", "报告日期",
)


def _find_document_date(
    blocks: list[dict[str, Any]],
    text: str,
) -> re.Match[str] | None:
    """Locate the report (examination) date.

    Date lines carrying a date keyword (检验日期/送检时间/采样时间/…) are
    authoritative; only when none is found fall back to the first date-like
    token anywhere in the OCR text.
    """
    for block in blocks:
        t = block.get("text") or ""
        if not t:
            continue
        for kw in DATE_KEYWORDS:
            if kw in t:
                m = DATE_RE.search(t)
                if m:
                    return m
    return DATE_RE.search(text or "")
HDR_WORDS = {
    "项目", "名称", "结果", "参考", "单位", "范围", "检测方法", "方法", "测定结果",
}


def _is_header_row(row_blocks: list[dict[str, Any]]) -> bool:
    text = row_text(row_blocks)
    if not text or len(text) > 40:
        return False
    return any(word in text for word in HDR_WORDS)
SKIP_WORDS = {
    "姓名", "性别", "年龄", "科室", "床号", "门诊号", "住院号", "病历号",
    "送检", "标本", "临床诊断", "诊断", "备注", "采样号", "检测时间",
    "报告日期", "报告时间", "免费", "免", "编号", "号码", "检验编号",
}


def _norm(s: str) -> str:
    return re.sub(r"[\s\u3000]", "", s).upper()


def _norm_unit(u: str) -> str:
    return re.sub(r"[\s\-/]", "", u).lower()


def cluster_rows(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Reconstruct the table grid: column clustering by x, then row
    clustering within each column, then row-number alignment.

    Clustering by block y-center alone chain-merges table rows because cell
    heights vary and adjacent rows interleave (a row's last cell can sit at
    nearly the same y as the next row's first cell). Splitting the problem
    per-column keeps row boundaries clean; the row index inside each column
    is the physical row number.

    Falls back to y-center row clustering when no multi-column structure
    (e.g. free text) is detected.
    """
    if not blocks:
        return []
    items: list[dict[str, Any]] = []
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        items.append(
            {
                "block": block,
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "w": x1 - x0,
                "h": y1 - y0,
            }
        )
    heights = [it["h"] for it in items]
    med_h = statistics.median(heights) if heights else 12.0
    widths = [it["w"] for it in items]
    med_w = statistics.median(widths) if widths else 40.0

    # 1) Column clustering by x0 gap.
    col_gap = 0.6 * med_w
    by_x = sorted(items, key=lambda it: it["x0"])
    columns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_x1: float | None = None
    for it in by_x:
        if prev_x1 is not None and it["x0"] - prev_x1 > col_gap:
            columns.append(current)
            current = []
        current.append(it)
        prev_x1 = it["x1"]
    if current:
        columns.append(current)

    if len(columns) <= 1:
        # Not a table layout: fall back to plain y-center row clustering.
        return _cluster_rows_by_cy(items, med_h)

    # 2) Row clustering inside each column (by y0 gap). Header rows are
    #    separated out: they appear in only some columns and would offset
    #    the row-number alignment.
    row_gap_y = 0.7 * med_h
    col_rows: list[list[list[dict[str, Any]]]] = []
    for col in columns:
        col_sorted = sorted(col, key=lambda it: it["y0"])
        col_blocks_rows: list[list[dict[str, Any]]] = []
        cur_row: list[dict[str, Any]] = []
        prev_y0: float | None = None
        for it in col_sorted:
            if prev_y0 is not None and it["y0"] - prev_y0 > row_gap_y:
                col_blocks_rows.append(cur_row)
                cur_row = []
            cur_row.append(it)
            prev_y0 = it["y0"]
        if cur_row:
            col_blocks_rows.append(cur_row)
        col_rows.append([r for r in col_blocks_rows if not _is_header_row(r)])

    # 3) Align other columns to the anchor column (the one whose rows look
    #    most like lab item names) by nearest y. Header rows that do not
    #    appear in every column are absorbed or dropped; the table rows
    #    themselves pair correctly. (A perfect match for dense 6-column
    #    tables needs header-driven column semantics — out of scope here.)
    anchor_idx = max(
        range(len(col_rows)),
        key=lambda ci: (_dictionary_score(col_rows[ci]), len(col_rows[ci])),
    )

    def _row_y(row: list[dict[str, Any]]) -> float:
        return statistics.median(it["y0"] for it in row)

    anchor_rows = col_rows[anchor_idx]
    anchor_y = [_row_y(row) for row in anchor_rows]
    aligned_rows: list[list[dict[str, Any]]] = [list(row) for row in anchor_rows]
    aligned_y: list[float] = list(anchor_y)

    for ci, col_blocks_rows in enumerate(col_rows):
        if ci == anchor_idx:
            continue
        for row in col_blocks_rows:
            y_ref = _row_y(row)
            best_k = min(range(len(aligned_y)), key=lambda k: abs(y_ref - aligned_y[k]))
            aligned_rows[best_k].extend(row)

    for row in aligned_rows:
        row.sort(key=lambda it: it["x0"])
    return aligned_rows


def _dictionary_score(col_rows: list[list[dict[str, Any]]]) -> int:
    """Prefer the column whose rows look like lab item names."""
    score = 0
    for row in col_rows:
        text = _norm(re.sub(r"[\s,，。;；:：()（）<>＜＞]", "", row_text(row)))
        if any(
            len(alias) > 1 and (alias in text or text in alias)
            for alias in ALIAS_LOOKUP
        ):
            score += 1
    return score


def _cluster_rows_by_cy(
    items: list[dict[str, Any]], med_h: float
) -> list[list[dict[str, Any]]]:
    by_cy = sorted(items, key=lambda it: it["cy"])
    row_gap = max(med_h * 0.55, 6.0)
    rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_cy: float | None = None
    for it in by_cy:
        if last_cy is None or abs(it["cy"] - last_cy) <= row_gap:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
        last_cy = it["cy"]
    if current:
        rows.append(current)
    for row in rows:
        row.sort(key=lambda it: it["x0"])
    return rows


def row_text(row: list[dict[str, Any]]) -> str:
    return " ".join(item["block"]["text"].strip() for item in row if item["block"]["text"].strip())


def _find_unit(text: str) -> tuple[str | None, str]:
    """Return (canonical_unit, remainder) — longest unit match removed."""
    best: tuple[int, str | None] = (-1, None)
    for unit in UNITS_SORTED:
        idx = text.find(unit)
        if idx >= 0 and (idx > best[0]):
            best = (idx, unit)
    if best[1] is None:
        return None, text
    unit = best[1]
    remainder = text[: best[0]] + " " + text[best[0] + len(unit):]
    return UNIT_NORM.get(_norm_unit(unit), unit), remainder


def parse_row(row: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse one table row into a fact candidate (dict) or None."""
    text = row_text(row)
    if not text:
        return None

    remainder = text
    reference: dict[str, Any] | None = None

    # 1. Reference range (range a-b, <x, >x)
    m = REF_RANGE_RE.search(remainder)
    if m:
        low = float(m.group("low"))
        high = float(m.group("high"))
        if low > high:  # OCR sometimes reverses "0~10" into "10~0"
            low, high = high, low
        reference = {"low": low, "high": high, "text": m.group(0).strip()}
        remainder = remainder[: m.start()] + " " + remainder[m.end():]
    else:
        m2 = REF_LT_RE.search(remainder)
        if m2:
            reference = {"low": None, "high": float(m2.group("max")), "text": m2.group(0).strip()}
            remainder = remainder[: m2.start()] + " " + remainder[m2.end():]
        else:
            m3 = REF_GT_RE.search(remainder)
            if m3:
                reference = {"low": float(m3.group("min")), "high": None, "text": m3.group(0).strip()}
                remainder = remainder[: m3.start()] + " " + remainder[m3.end():]

    # 2. Unit
    unit, remainder = _find_unit(remainder)

    # 3. Value number(s) — after reference/unit removal, take the LAST
    #    number token (result values sit at the row's right end; row
    #    sequence numbers at the left must not win). A 阳性/阴性 marker
    #    shifts the value to whatever follows it.
    flag: str | None = None
    value: float | None = None
    numbers = list(NUMBER_RE.finditer(remainder))
    if numbers:
        candidates = numbers
        marker = re.search(r"阳性|阴性", remainder)
        if marker:
            after = [m for m in numbers if m.start() >= marker.end()]
            if after:
                candidates = after
        # A leading 1..50 integer followed by a letter/char is a row
        # sequence number, not a lab value (e.g. "10 AFP 甲胎蛋白").
        first = candidates[0]
        tail_after_first = remainder[first.end():].lstrip()
        if (
            len(candidates) == 1
            and first.start() <= 2
            and 1 <= float(first.group("num")) <= 50
            and tail_after_first
            and not tail_after_first[0].isdigit()
            and "." not in first.group("num")
        ):
            candidates = []
        # Merge a split decimal: "0." followed by " 23" -> 0.23.
        merged: list[tuple[int, int, str]] = []
        idx = 0
        while idx < len(candidates):
            m = candidates[idx]
            tok = m.group("num")
            if tok.endswith(".") and idx + 1 < len(candidates):
                nxt = candidates[idx + 1]
                nxt_tok = nxt.group("num")
                if "." not in nxt_tok and nxt.start() <= m.end() + 2:
                    merged.append((m.start(), nxt.end(), tok + nxt_tok))
                    idx += 2
                    continue
            merged.append((m.start(), m.end(), tok))
            idx += 1
        if merged:
            start, end, tok = merged[-1]
            value = float(tok)
            prefix_match = re.search(r"[><≥≤-]\s*$", remainder[:start])
            prefix = prefix_match.group(0).strip() if prefix_match else None
            flag = prefix if prefix in (">", "<", "≥", "≤") else None
            if prefix == "-":
                value = -value
            remainder = remainder[:start] + " " + remainder[end:]

    # 4. Name part: strip punctuation/whitespace, match the lab dictionary.
    name_part = _norm(re.sub(r"[\s,，。;；:：()（）<>＜＞]", "", remainder))
    if not name_part:
        return None
    # Header rows carry no values — skip them.
    if value is None and unit is None and reference is None:
        return None
    # Personal-info / header rows (not lab items): skip.
    if any(word in name_part for word in SKIP_WORDS):
        return None

    canonical: str | None = None
    matched_alias: str | None = None
    for alias in sorted(ALIAS_LOOKUP, key=len, reverse=True):
        if len(alias) <= 2:
            # Single/double-char aliases (K, P, CA, NA...) need word
            # boundaries: "KU/L" must not match K, "Pres1" must not match P.
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", name_part):
                canonical = ALIAS_LOOKUP[alias]
                matched_alias = alias
                break
        elif alias in name_part or name_part in alias:
            canonical = ALIAS_LOOKUP[alias]
            matched_alias = alias
            break

    concept = Concept(
        display=canonical or name_part,
        local_display=text,
        local_code=matched_alias or None,
    )

    confidence = 0.92 if canonical else 0.55
    if unit:
        confidence += 0.03
    if value is not None:
        confidence += 0.03
    if reference:
        confidence -= 0.01

    blocks_conf = [b.get("confidence") for b in (item["block"] for item in row)]
    confs = [c for c in blocks_conf if isinstance(c, (int, float))]
    avg_conf = sum(confs) / len(confs) if confs else 1.0
    confidence = min(0.99, confidence * avg_conf)

    evidence = [
        EvidencePointer(
            artifact_id="__artifact__",
            block_id=item["block"]["block_id"],
            quote=item["block"]["text"][:80],
        )
        for item in row
    ]
    fact = {
        "concept": concept,
        "local_name": text,
        "value_number": value,
        "value_flag": flag,
        "unit": unit,
        "reference": reference,
        "confidence": round(confidence, 3),
        "evidence": evidence,
    }
    return fact


def _interpretation(value: float | None, reference: dict[str, Any] | None) -> str:
    if value is None or reference is None:
        return "unknown"
    low = reference.get("low")
    high = reference.get("high")
    if low is not None and high is not None:
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "normal"
    if low is None and high is not None:
        return "normal" if value <= high else "high"
    if high is None and low is not None:
        return "normal" if value >= low else "low"
    return "unknown"


def extract_event(
    patient_id: str,
    artifact: SourceArtifact,
    ocr: OCRResult,
) -> ClinicalEvent | None:
    """Rule-based extraction: blocks -> ClinicalEvent (or None)."""
    blocks: list[dict[str, Any]] = list(ocr.blocks)
    if not blocks:
        return None

    rows = cluster_rows(blocks)
    facts: list[ClinicalFact] = []
    for row in rows:
        candidate = parse_row(row)
        if candidate is None:
            continue
        value = candidate["value_number"]
        if value is not None:
            attrs: dict[str, Any] = {}
            if candidate["value_flag"]:
                attrs["flag"] = candidate["value_flag"]
            fvalue: QuantityValue | TextValue = QuantityValue(
                number=value,
                unit=candidate["unit"],
            )
        else:
            fvalue = TextValue(text=candidate["local_name"])
            attrs = {}

        ref = candidate["reference"]
        reference_range = None
        if ref:
            reference_range = ReferenceRange(
                low=ref.get("low"),
                high=ref.get("high"),
                unit=candidate["unit"],
                text=ref.get("text"),
            )

        evidence = [
            EvidencePointer(
                artifact_id=artifact.artifact_id,
                block_id=item.block_id,
                quote=item.quote,
            )
            for item in candidate["evidence"]
        ]
        facts.append(
            ClinicalFact(
                fact_id=f"fact_{uuid4().hex[:10]}",
                kind="observation",
                concept=candidate["concept"],
                value=fvalue,
                reference_range=reference_range,
                interpretation=_interpretation(value, ref),
                time=EventTime(
                    start=None,
                    precision="unknown",
                    source="document",
                ),
                evidence=evidence,
                confidence=candidate["confidence"],
                status="extracted",
                attributes=attrs,
            )
        )

    if not facts:
        return None

    # Document date: prefer a line carrying a date keyword (检验日期/送检
    # 时间/采样时间…), else the first date-like token in the raw OCR text.
    m = _find_document_date(blocks, ocr.text or "")
    event_time: EventTime
    if m:
        raw = m.group(0).replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
        parts = [p for p in re.split(r"[-.]", raw) if p]
        if len(parts) >= 3:
            try:
                event_time = EventTime(
                    start=datetime(int(parts[0]), int(parts[1]), int(parts[2])),
                    precision="day",
                    source="document",
                )
            except ValueError:
                event_time = EventTime(precision="unknown", source="document")
        else:
            event_time = EventTime(precision="unknown", source="document")
    else:
        event_time = EventTime(precision="unknown", source="document")

    return ClinicalEvent(
        event_id=f"evt_{uuid4().hex[:10]}",
        patient_id=patient_id,
        event_type="laboratory",
        title="检验报告",
        event_time=event_time,
        institution=None,
        source_artifact_ids=[artifact.artifact_id],
        facts=facts,
        summary=f"共提取 {len(facts)} 项检验结果",
    )


class RuleBasedNormalizer(ClinicalNormalizer):
    """Zero-cost field extraction: coordinates + dictionaries + regex."""

    async def normalize(
        self,
        patient_id: str,
        artifact: SourceArtifact,
        ocr: OCRResult,
    ) -> ClinicalEvent | None:
        if not ocr.text.strip():
            return None
        return extract_event(patient_id, artifact, ocr)
