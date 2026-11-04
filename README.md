# HealthApp

A patient-owned longitudinal health record application.

HealthApp turns scattered laboratory reports, imaging reports, prescriptions, discharge summaries and patient-entered observations into a provenance-preserving clinical timeline that can be used by both people and AI agents.

## Core principles

1. **Source first**: original files, OCR text and layout blocks remain the evidence layer.
2. **Time first**: clinical facts carry the time the event happened, not only the upload time.
3. **Fact first**: normalized atomic facts are the primary machine-readable layer.
4. **Inference is separate**: AI summaries and suggestions never overwrite source-derived facts.
5. **Traceability by default**: every important fact can point back to the original report location.

## Repository structure

- `apps/web`: responsive React MVP
- `apps/api`: FastAPI API and canonical schemas
- `docs/architecture.md`: system architecture
- `docs/data-model.md`: longitudinal health data model
- `docs/product-spec.md`: product and UX specification
- `docs/progress/`: 阶段性中文进度说明（识别、时间轴、AI 咨询）
- `examples/patient_record.json`: example canonical patient record

## MVP workflow

```text
Photo / PDF
    ↓
OCR + layout
    ↓
Document classification
    ↓
LLM normalization
    ↓
Confidence check / user confirmation
    ↓
ClinicalEvent + ClinicalFact
    ↓
Timeline / trends / Agent context
```

## Run the web app

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`.

## Run the API

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The API is available at `http://localhost:8000` and OpenAPI documentation at `/docs`.

To load the demo timeline:

```bash
curl -X POST http://localhost:8000/api/dev/load-demo
```

## What is implemented now

- responsive overview, timeline, upload and AI context interfaces
- registration auth and NVIDIA OCR/LLM ingestion adapters
- laboratory reference-range extraction and timeline display
- canonical longitudinal schemas with provenance
- document ingestion boundary and Agent context builder
- interactive health assistant (analyze / chat)
- demo patient timeline

阶段性详细进展见 [`docs/progress/`](./docs/progress/README.md)。

## Next implementation priorities

1. 合入 `dev/yexiaohan` 识别链路（RapidOCR、质量门、多页 PDF、交叉核对）
2. AI 助手置信度校准、SSE 对话持久化压测、用户反馈
3. 手写体改为人工/语音录入窗口（不做自动 OCR）
4. PostgreSQL 与对象存储
5. 术语归一化与单位换算
6. 重复/冲突消解与人工审核落库
7. 认证加密与审计日志
8. FHIR 导入导出
9. 移动端 PWA
