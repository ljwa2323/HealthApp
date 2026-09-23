# 纵向健康数据模型

## 1. 设计目标

数据结构主要服务两个对象：

- 人类用户，需要快速浏览、纠错、追溯
- LLM/Agent，需要低冗余、时间明确、结构稳定、可按任务裁剪

目标不是把所有文档压成一段摘要，而是实现接近无损的语义压缩。

## 2. 核心实体

### Patient

只保存稳定身份和少量长期属性，避免在每次事件中重复。

```json
{
  "patient_id": "p_demo_001",
  "sex": "female",
  "birth_date": "1989-05-12"
}
```

### SourceArtifact

表示原始证据。

关键字段：

- `artifact_id`
- `type`: image / pdf / text / manual
- `sha256`
- `storage_uri`
- `ocr_text`
- `ocr_blocks`
- `document_time`
- `institution`
- `document_type`

同一文件通过哈希去重。

### ClinicalEvent

时间轴上的一个临床事件容器。

例如：

- 一次门诊
- 一次住院
- 一次检验采样
- 一次 CT 检查
- 一次处方
- 一段服药过程
- 一次患者自测血压

核心字段：

```json
{
  "event_id": "evt_20260923_lab_001",
  "event_type": "laboratory",
  "event_time": {
    "start": "2026-09-23T08:20:00+08:00",
    "end": null,
    "precision": "minute"
  },
  "source_artifact_ids": ["art_001"],
  "fact_ids": ["fact_cr_001", "fact_egfr_001"]
}
```

### ClinicalFact

最重要的原子单位。

一个 fact 只表达一个可验证陈述，例如：

- 肌酐 = 96 μmol/L
- eGFR = 82 mL/min/1.73m²
- CT 报告提示右肺上叶结节 5 mm
- 开始服用阿托伐他汀 20 mg qn

推荐结构：

```json
{
  "fact_id": "fact_cr_001",
  "kind": "observation",
  "concept": {
    "display": "Creatinine",
    "system": "LOINC",
    "code": "2160-0"
  },
  "value": {
    "type": "quantity",
    "number": 96,
    "unit": "umol/L"
  },
  "reference_range": {
    "low": 41,
    "high": 81,
    "unit": "umol/L"
  },
  "interpretation": "high",
  "time": {
    "start": "2026-09-23T08:20:00+08:00",
    "precision": "minute"
  },
  "evidence": [
    {
      "artifact_id": "art_001",
      "page": 1,
      "block_id": "ocr_block_18",
      "quote": "肌酐 96 ↑ μmol/L 41-81"
    }
  ],
  "confidence": 0.99,
  "status": "confirmed"
}
```

## 3. 为什么采用 Event + Fact

如果把整份报告直接存成一个 JSON，大量字段会相互重复，而且很难跨文档比较。

Event 负责表达临床上下文和时间。

Fact 负责表达可比较的原子医学事实。

例如一张肝肾功能报告只需要一个 laboratory event，其中挂载几十个 observation facts。

## 4. 时间建模

时间必须允许不完整。

```json
{
  "start": "2026-09-01",
  "end": null,
  "precision": "day",
  "source": "document"
}
```

`precision` 支持：

- minute
- hour
- day
- month
- year
- interval
- unknown

禁止为了统一格式而虚构不存在的小时或分钟。

## 5. 纵向状态不是重复存储

慢病状态、长期药物列表、最近一次关键指标等，原则上都应从事实层计算得到。

例如：

```text
current_medications = fold(medication facts ordered by time)
active_conditions = resolve(condition facts + status changes)
latest_labs = latest observation by concept
```

可以缓存这些视图，但事实主表保持唯一。

## 6. 近无损压缩

HealthApp 不追求删除原文，而是通过双轨保存实现：

```text
original artifact
      +
OCR text / evidence blocks
      +
canonical facts
```

LLM 日常只读取 canonical facts。

当需要核验、解释歧义或恢复完整上下文时，再按 evidence pointer 回读原文。

因此高频计算使用紧凑结构，同时保留完整证据。

## 7. 术语标准化

优先支持：

- LOINC：检验与观察
- SNOMED CT：症状、疾病、临床概念
- ICD-10：诊断编码兼容
- RxNorm / ATC：药物
- UCUM：单位
- HPO：表型，可选

中国医院本地名称可以保留 `local_code` 和 `local_display`。

标准化失败时绝不能丢弃原始名称。

## 8. AI 衍生对象

所有 LLM 输出进入独立的 `DerivedInsight`。

```json
{
  "insight_id": "ins_001",
  "kind": "trend_summary",
  "created_at": "2026-09-23T10:00:00Z",
  "model": "model-name",
  "input_fact_ids": ["fact_cr_001", "fact_cr_002"],
  "text": "近一年肌酐总体稳定。",
  "confidence": 0.86
}
```

这种设计避免 AI 输出污染医学事实层。

## 9. 面向 Agent 的紧凑上下文

Agent 默认不读取 OCR 全文，而读取任务相关 JSON：

```json
{
  "task": "renal_function_review",
  "patient": {"age": 37, "sex": "female"},
  "observations": [
    ["2025-03-04", "creatinine", 82, "umol/L"],
    ["2026-02-16", "creatinine", 89, "umol/L"],
    ["2026-09-23", "creatinine", 96, "umol/L"]
  ],
  "medications": [],
  "conditions": [],
  "data_gaps": ["No urine albumin results in record"],
  "evidence_refs": ["fact_cr_001", "fact_cr_002", "fact_cr_003"]
}
```

这是给 LLM 的压缩视图，不是数据库的唯一真实来源。
