# HealthApp

A longitudinal personal health record application designed for patient-owned health information management.

## Core idea

HealthApp converts scattered medical records, laboratory reports, imaging reports, prescriptions, discharge summaries, and patient-entered observations into a provenance-preserving longitudinal timeline that both humans and AI agents can use.

The project follows three principles:

1. **Source first**: original files and OCR text are immutable evidence.
2. **Time first**: every clinical fact carries the time it happened, not only the time it was uploaded.
3. **Inference separated from fact**: AI summaries and suggestions never overwrite source-derived clinical facts.

## Repository structure

- `apps/web`: React web MVP
- `apps/api`: FastAPI backend MVP
- `docs/architecture.md`: system architecture
- `docs/data-model.md`: longitudinal health data model
- `docs/product-spec.md`: product and UX specification
- `examples/patient_record.json`: example canonical patient record

## MVP workflow

Upload photo/PDF → OCR → document classification → LLM normalization → human confirmation when needed → canonical timeline → longitudinal views → AI context builder.

## Status

Initial architecture and runnable MVP scaffold.
