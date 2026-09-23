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
- canonical longitudinal schemas
- source artifact and OCR provenance model
- document ingestion boundary
- task-specific Agent context builder
- demo patient timeline
- frontend upload call to the API

The OCR provider and LLM normalizer are intentionally adapters in the first commit. They can be connected to PaddleOCR, another OCR engine, and the chosen LLM without changing the canonical data model.

## Next implementation priorities

1. PostgreSQL and object storage
2. production OCR provider
3. LLM extraction with JSON schema validation
4. terminology normalization and unit conversion
5. duplicate and conflict resolution
6. human review screen
7. longitudinal trend query APIs
8. authentication, encryption and audit logging
9. FHIR import/export adapters
10. mobile PWA packaging
