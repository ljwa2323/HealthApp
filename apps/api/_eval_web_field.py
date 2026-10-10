"""Field-level extraction evaluation on the 20 web lab-report images vs hand-written GT.

Metrics per image:
  gt_items      - number of lab items in the GT (excluding header/info lines)
  extracted     - number of facts extracted by the rule-based extractor
  matched       - item fully correct (concept name matched AND numeric/text value correct)
  concept_only  - concept matched but value wrong/missing
  missed        - GT items not recovered
Overall: field accuracy = matched / extracted ; field recall = matched / gt_items
"""
import asyncio
import difflib
import re
from pathlib import Path

from app.services.field_extractor import extract_event
from app.services.nvidia_providers import build_default_providers
from app.schemas import SourceArtifact

TEST_DIR = Path(r"C:\Users\31683\Desktop\图片识别测试")

# canonical display names that may differ from report wording
NAME_HINTS = {
    "红细胞计数": ["红细胞", "红细胞数", "红细胞数目", "红细胞计数", "红细胞总数"],
    "血红蛋白": ["血红蛋白", "血红蛋白浓度", "血红蛋白量", "血红蛋白浓度(hgb)", "hgb"],
    "血小板计数": ["血小板", "血小板数", "血小板数目", "血小板总数", "血小板计数"],
    "白细胞计数": ["白细胞", "白细胞数", "白细胞数目", "白细胞总数", "白细胞计数"],
    "红细胞压积": ["红细胞压积", "红细胞比容", "红细胞比积", "血细胞比容", "hct"],
    "平均红细胞体积": ["平均红细胞体积", "平均红细胞容积", "mcv"],
    "平均红细胞血红蛋白含量": ["平均红细胞血红蛋白含量", "平均血红蛋白含量", "平均血红蛋白量", "平均rbc血红蛋白含量", "mch"],
    "平均红细胞血红蛋白浓度": ["平均红细胞血红蛋白浓度", "平均血红蛋白浓度", "平均rbc血红蛋白浓度", "mchc"],
    "白球比": ["白球比", "白蛋白/球蛋白", "白蛋白/球蛋白(a/g)", "a/g"],
    "谷草转氨酶": ["谷草转氨酶", "天门冬氨酸氨基转移酶", "天冬氨酸氨基转移酶", "ast"],
    "谷丙转氨酶": ["谷丙转氨酶", "丙氨酸氨基转移酶", "alt"],
    "碱性磷酸酶": ["碱性磷酸酶", "alp", "akp"],
    "γ-谷氨酰转肽酶": ["γ-谷氨酰转肽酶", "谷氨酰转肽酶", "谷氨酰氨转移酶", "γ-谷氨酰基转移酶", "l-γ-谷氨酰转肽酶", "ggt"],
    "总胆红素": ["总胆红素", "tbil"],
    "直接胆红素": ["直接胆红素", "dbil"],
    "间接胆红素": ["间接胆红素", "间接胆红素(ibil)", "ibil"],
    "总蛋白": ["总蛋白", "tp"],
    "白蛋白": ["白蛋白", "alb"],
    "球蛋白": ["球蛋白", "glob", "glb"],
    "肌酐": ["肌酐", "crea", "cr"],
    "尿素": ["尿素", "尿素氮", "bun", "urea"],
    "尿酸": ["尿酸", "ua"],
    "葡萄糖": ["葡萄糖", "glu"],
    "钙": ["钙", "总钙", "ca"],
    "磷": ["磷", "p"],
    "镁": ["镁", "mg"],
    "钾": ["钾", "k"],
    "钠": ["钠", "na"],
    "氯": ["氯", "cl"],
    "乳酸脱氢酶": ["乳酸脱氢酶", "ldh"],
    "肌酸激酶": ["肌酸激酶", "磷酸肌酸激酶", "ck"],
    "羟丁酸脱氢酶": ["羟丁酸脱氢酶", "α-羟丁酸脱氢酶", "hbdh"],
    "总胆固醇": ["总胆固醇", "cholesterol", "chol"],
    "甘油三酯": ["甘油三酯", "甘油三脂", "tg"],
    "高密度脂蛋白胆固醇": ["高密度脂蛋白胆固醇", "hdl-c"],
    "低密度脂蛋白胆固醇": ["低密度脂蛋白胆固醇", "ldl-c"],
    "促甲状腺激素": ["促甲状腺激素", "促甲状腺激素测定", "血清促甲状腺激素", "tsh"],
    "游离三碘甲状腺原氨酸": ["游离三碘甲状腺原氨酸", "游离三碘甲状原氨酸", "游离三碘甲状腺原氨", "ft3"],
    "游离甲状腺素": ["游离甲状腺素", "游离甲状腺素测定", "ft4"],
    "甲状腺素": ["甲状腺素", "t4"],
    "三碘甲状腺原氨酸": ["三碘甲状腺原氨酸", "t3"],
    "抗甲状腺球蛋白抗体": ["抗甲状腺球蛋白抗体", "tga", "tg-ab"],
    "抗甲状腺过氧化物酶抗体": ["抗甲状腺过氧化物酶抗体", "抗甲状腺过氧化物酶抗体测定", "anti-tpo", "tpo-ab", "甲状腺过氧化物酶抗体"],
    "乙肝表面抗原": ["乙肝表面抗原", "hbsag"],
    "乙肝表面抗体": ["乙肝表面抗体", "hbsab"],
    "乙肝e抗原": ["乙肝e抗原", "hbeag"],
    "乙肝e抗体": ["乙肝e抗体", "hbeab"],
    "乙肝核心抗体": ["乙肝核心抗体", "hbcab"],
}


def norm_text(s: str) -> str:
    return re.sub(r"[\s()（）\[\]【】\-_~×*#]", "", str(s)).lower()


def name_match(ext_display: str, gt_name: str) -> bool:
    """Return True when the extracted concept matches the GT item name."""
    a = norm_text(ext_display)
    b = norm_text(gt_name)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    hints = NAME_HINTS.get(ext_display) or NAME_HINTS.get(gt_name) or []
    for h in hints:
        hn = norm_text(h)
        if hn and (hn in a or a in hn or hn in b or b in hn):
            return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.55


def extract_number(s: str):
    """First number in a value string (handles '<2.00', '>1300.0', '1+ 0.3')."""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def value_match(gt_val: str, ext_val) -> bool:
    """Compare GT value text with an extracted fact value."""
    gt_num = extract_number(gt_val)
    if isinstance(ext_val, (int, float)):
        if gt_num is None:
            return False
        diff = abs(ext_val - gt_num)
        return diff <= 0.5 or (gt_num != 0 and diff / abs(gt_num) <= 0.05)
    # text value
    g = norm_text(gt_val)
    e = norm_text(ext_val)
    if not g or not e:
        return False
    return g == e or g in e or e in g


RANGE_RE = re.compile(
    r"^[-+]?\d+(?:\.\d+)?\s*[-~–]\s*[-+]?\d+(?:\.\d+)?"
)
CODE_RE = re.compile(r"^[A-Za-z0-9./#*^+<>~_-]{1,12}$")
UNIT_RE = re.compile(
    r"^(U/L|g/L|g/l|mmol/L|mmol/l|umol/L|μmol/L|mol/L|10\^9/L|10\^12/L|"
    r"×10\^9/L|×10\^12/L|x10\^9/L|x10\^12/L|%|pg|fl|fL|ng/ml|uIU/ml|IU/ml|u/ml|"
    r"U/ml|ug/ml|mg/L|/μL|/ul|/uL|/HPF|/HP|/LPF|/LP|/CH|cell/μ|ku/L|KU/L|EU/dl|ms/cm|MS/CM|"
    r"/ul|/μl|IU/mL|mIU/ml|s/co|S/CO)$",
    re.IGNORECASE,
)
TEXT_VALUE_RE = re.compile(
    r"^[+-]$|^[+-]?$|^\d?\+$|阴性|阳性|弱阳性|浅黄色|黄色|澄清|未见|已复核|"
    r"未分类|无结果|非均一|O型|阴性\(-\)|阳性\(\+\)|>=\d|>\d|<=\d|<\d"
)


def parse_gt(path: Path):
    """Parse a GT file into (info_lines, items) where items = [(name, value)]."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = [c.strip() for c in line.split("\t")]
        if not cells or not cells[0]:
            continue
        if cells[0] in ("No", "序号", "编号", "项目", "项目名称", "检验项目", "名称", "代号", "英文缩写", "英文"):
            continue
        if any(h in cells[0] for h in ("年龄", "性别", "临床诊断", "标本", "检验项目", "送检",
                                        "申请", "检验时间", "报告时间", "采样", "接收", "备注", "报告单")):
            continue
        # locate the item name: first cell that contains Chinese (or is PH值 style)
        name_idx = None
        for i, c in enumerate(cells):
            if not c:
                continue
            if i == 0 and (CODE_RE.fullmatch(c) or c in ("项目", "名称")):
                continue
            if any("\u4e00" <= ch <= "\u9fff" for ch in c) or c.startswith("PH") or c.startswith("pH"):
                name_idx = i
                break
        if name_idx is None:
            continue
        name = cells[name_idx]
        # value = first candidate after name that is numeric-but-not-range or text value
        value = None
        for c in cells[name_idx + 1:]:
            if not c:
                continue
            if UNIT_RE.fullmatch(c):
                continue
            if RANGE_RE.fullmatch(c):
                continue
            # pure-letter abbreviations (WBC, ALT...) are codes, not values
            if CODE_RE.fullmatch(c) and not re.search(r"\d", c) and not TEXT_VALUE_RE.search(c):
                continue
            if re.search(r"\d", c) or TEXT_VALUE_RE.search(c):
                value = c
                break
        if value is None:
            continue
        items.append((name, value))
    return items


async def main() -> None:
    ocr, _ = build_default_providers()
    results = []
    total_gt = total_ext = total_match = 0

    for gt_path in sorted(TEST_DIR.glob("web*.gt.txt")):
        img = TEST_DIR / (gt_path.name[:-len(".gt.txt")] + ".jpg")
        if not img.exists():
            continue
        gt_items = parse_gt(gt_path)
        result = await ocr.recognize(img.read_bytes(), "image/jpeg")
        artifact = SourceArtifact(
            artifact_id="art_web", patient_id="p_web", file_name=img.name,
            sha256="x", storage_uri="x",
            ocr_text=result.text, ocr_blocks=result.blocks,
        )
        event = extract_event("p_web", artifact, result)
        facts = event.facts if event else []

        matched = 0
        concept_only = 0
        used_gt = [False] * len(gt_items)
        # greedy: for each extracted fact, find the best unmatched GT item
        for fact in facts:
            best = None
            best_score = -1.0
            for gi, (gn, gv) in enumerate(gt_items):
                if used_gt[gi]:
                    continue
                if not name_match(fact.concept.display, gn):
                    continue
                score = difflib.SequenceMatcher(
                    None, norm_text(fact.concept.display), norm_text(gn)
                ).ratio()
                if score > best_score:
                    best_score, best = score, gi
            if best is None:
                continue
            gn, gv = gt_items[best]
            v = fact.value
            ext_num = v.number if hasattr(v, "number") else None
            ext_text = v.text if hasattr(v, "text") else None
            if value_match(gv, ext_num if ext_num is not None else ext_text):
                matched += 1
                used_gt[best] = True
            else:
                concept_only += 1

        missed = sum(1 for i, u in enumerate(used_gt) if not u)
        extracted = len(facts)
        total_gt += len(gt_items)
        total_ext += extracted
        total_match += matched
        results.append((gt_path.stem, len(gt_items), extracted, matched, concept_only, missed))

    print(f"{'image':<8} {'GT':>3} {'ext':>4} {'match':>5} {'cOnly':>5} {'miss':>5}  acc  rec")
    for name, gt, ext, match, conly, miss in results:
        acc = match / ext if ext else 0
        rec = match / gt if gt else 0
        print(f"{name:<8} {gt:>3} {ext:>4} {match:>5} {conly:>5} {miss:>5}  {acc:4.0%} {rec:4.0%}")
    print("-" * 64)
    print(f"TOTAL  GT={total_gt} extracted={total_ext} matched={total_match}")
    print(f"FIELD ACCURACY = {total_match/total_ext:4.1%}  (matched / extracted)")
    print(f"FIELD RECALL   = {total_match/total_gt:4.1%}  (matched / GT items)")


if __name__ == "__main__":
    asyncio.run(main())
