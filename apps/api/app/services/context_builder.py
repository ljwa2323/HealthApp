from __future__ import annotations

from datetime import datetime
from typing import Iterable

from ..schemas import AgentContext, ClinicalEvent


def _in_range(value: str | None, start: str | None, end: str | None) -> bool:
    if not value:
        return True
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if start and parsed < datetime.fromisoformat(start.replace("Z", "+00:00")):
            return False
        if end and parsed > datetime.fromisoformat(end.replace("Z", "+00:00")):
            return False
    except (TypeError, ValueError):
        return True
    return True


def build_agent_context(
    *,
    patient_id: str,
    task: str,
    events: Iterable[ClinicalEvent],
    start: str | None = None,
    end: str | None = None,
) -> AgentContext:
    observations: list[dict] = []
    medications: list[dict] = []
    conditions: list[dict] = []
    compact_events: list[dict] = []
    evidence_refs: list[str] = []

    for event in events:
        if event.patient_id != patient_id:
            continue

        event_start = str(event.event_time.start) if event.event_time.start else None
        if not _in_range(event_start, start, end):
            continue

        compact_events.append(
            {
                "event_id": event.event_id,
                "time": event_start,
                "type": event.event_type,
                "title": event.title,
                "institution": event.institution,
            }
        )

        for fact in event.facts:
            item = {
                "fact_id": fact.fact_id,
                "time": str(fact.time.start) if fact.time.start else event_start,
                "concept": fact.concept.display,
                "code": fact.concept.code,
                "system": fact.concept.system,
                "value": fact.value.model_dump() if fact.value else None,
                "interpretation": fact.interpretation,
                "confidence": fact.confidence,
            }

            if fact.kind == "observation":
                observations.append(item)
            elif fact.kind == "medication":
                medications.append(item)
            elif fact.kind == "condition":
                conditions.append(item)

            evidence_refs.extend(
                pointer.artifact_id for pointer in fact.evidence
            )

    gaps: list[str] = []
    if not observations:
        gaps.append("No observations available in the selected time range.")
    if not medications:
        gaps.append("No medication history available in the selected time range.")
    if not conditions:
        gaps.append("No condition history available in the selected time range.")

    return AgentContext(
        task=task,
        patient_id=patient_id,
        time_range={"start": start, "end": end},
        observations=observations,
        medications=medications,
        conditions=conditions,
        events=compact_events,
        data_gaps=gaps,
        evidence_refs=sorted(set(evidence_refs)),
    )
