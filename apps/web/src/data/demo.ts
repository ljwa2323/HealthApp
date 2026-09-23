import type { TimelineEvent, TrendPoint } from "../types";

export const events: TimelineEvent[] = [
  {
    id: "evt-20260902",
    date: "2026-09-02",
    displayDate: "9月2日",
    type: "imaging",
    title: "胸部 CT",
    institution: "示例医院",
    summary: "报告记录右肺上叶约 5 mm 实性结节。",
    source: "胸部CT报告.pdf · OCR 99%",
    facts: [
      {
        id: "fact-nodule",
        label: "右肺上叶结节",
        value: "约 5 mm",
        interpretation: "abnormal",
        evidence: "第 1 页：右肺上叶见实性结节影，直径约5 mm",
      },
    ],
  },
  {
    id: "evt-20260318",
    date: "2026-03-18",
    displayDate: "3月18日",
    type: "laboratory",
    title: "年度体检",
    institution: "示例体检中心",
    summary: "肌酐 88 μmol/L，轻度高于该报告参考范围。",
    source: "生化检验报告.jpg · OCR 99%",
    facts: [
      {
        id: "fact-creatinine-2026",
        label: "肌酐",
        value: "88 μmol/L",
        interpretation: "high",
        evidence: "第 2 页：肌酐 88 ↑ μmol/L 41-81",
      },
      {
        id: "fact-egfr-2026",
        label: "eGFR",
        value: "82 mL/min/1.73m²",
        interpretation: "normal",
        evidence: "第 2 页：eGFR 82 mL/min/1.73m²",
      },
    ],
  },
  {
    id: "evt-20250108",
    date: "2025-01-08",
    displayDate: "2025年1月8日",
    type: "laboratory",
    title: "肾功能检查",
    institution: "示例医院",
    summary: "肌酐 76 μmol/L，在报告参考范围内。",
    source: "门诊检验单.jpg · OCR 99%",
    facts: [
      {
        id: "fact-creatinine-2025",
        label: "肌酐",
        value: "76 μmol/L",
        interpretation: "normal",
        evidence: "第 1 页：肌酐 76 μmol/L 41-81",
      },
    ],
  },
  {
    id: "evt-20241120",
    date: "2024-11-20",
    displayDate: "2024年11月20日",
    type: "medication",
    title: "门诊处方",
    institution: "示例医院",
    summary: "开始使用阿托伐他汀 20 mg，每晚一次。",
    source: "门诊处方.png · OCR 98%",
    facts: [
      {
        id: "fact-med",
        label: "阿托伐他汀",
        value: "20 mg qn",
        evidence: "阿托伐他汀钙片 20mg，每晚1次",
      },
    ],
  },
];

export const creatinineTrend: TrendPoint[] = [
  { date: "2024-06", value: 74 },
  { date: "2025-01", value: 76 },
  { date: "2025-08", value: 81 },
  { date: "2026-03", value: 88 },
];

export const kidneyContext = {
  task: "回顾近两年肾功能变化",
  timeRange: "2024-09 至 2026-09",
  facts: [
    ["2025-01-08", "肌酐", "76 μmol/L"],
    ["2026-03-18", "肌酐", "88 μmol/L"],
    ["2026-03-18", "eGFR", "82 mL/min/1.73m²"],
  ],
  gaps: ["当前记录中没有尿白蛋白/肌酐比结果", "2026年3月后尚无复查肾功能"],
  evidence: ["fact-creatinine-2025", "fact-creatinine-2026", "fact-egfr-2026"],
};
