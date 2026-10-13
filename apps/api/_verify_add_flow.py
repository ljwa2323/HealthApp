"""Verify add-to-timeline flow: upload does NOT auto-write; explicit POST does."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

D = Path(r"C:\Users\31683\Desktop\图片识别测试")

with TestClient(app) as client:
    r = client.post("/api/auth/login", json={"email": "ocr_test2@example.com", "password": "test123456"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    pid = r.json()["user"]["patient_id"]
    headers = {"Authorization": f"Bearer {token}"}

    def timeline_count() -> int:
        return len(client.get(f"/api/patients/{pid}/timeline", headers=headers).json())

    before = timeline_count()

    with (D / "test3.jpeg").open("rb") as fh:
        up = client.post(
            f"/api/patients/{pid}/documents",
            headers=headers,
            files={"file": ("test3.jpeg", fh, "image/jpeg")},
        )
    assert up.status_code == 200, up.text
    body = up.json()
    ev = body.get("extracted_event")
    assert ev is not None and len(ev["facts"]) == 10
    after_upload = timeline_count()
    print(f"before={before} after_upload={after_upload} -> auto-write removed: {after_upload == before}")

    # Build payload the way the frontend does (facts get time; manual facts appended).
    facts = [{"fact_id": f["fact_id"], "kind": "observation", "concept": f["concept"],
              "value": f["value"], "reference_range": f.get("reference_range"),
              "interpretation": f.get("interpretation"), "status": "confirmed",
              "time": {"precision": "unknown"}} for f in ev["facts"]]
    manual = {"fact_id": "manual_123", "kind": "observation",
              "concept": {"display": "血尿酸", "local_display": "血尿酸"},
              "value": {"type": "quantity", "number": 360, "unit": "umol/L"},
              "reference_range": {"low": 155, "high": 422, "text": "155-422", "unit": "umol/L"},
              "interpretation": "normal", "status": "extracted",
              "time": {"precision": "unknown"}}
    payload = {
        "event_id": ev["event_id"],
        "patient_id": pid,
        "event_type": ev["event_type"],
        "title": ev["title"],
        "event_time": ev.get("event_time") or {"precision": "unknown"},
        "institution": ev.get("institution"),
        "summary": ev.get("summary"),
        "source_artifact_ids": [body["artifact"]["artifact_id"]],
        "facts": facts + [manual],
    }
    r = client.post(f"/api/patients/{pid}/timeline", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    saved = r.json()
    after_add = timeline_count()
    print(f"after_add={after_add} added: {after_add == before + 1}")
    print(f"saved facts={len(saved['facts'])} (10 extracted + 1 manual)")

    # cleanup: delete the event to keep the test account timeline tidy
    d = client.delete(f"/api/patients/{pid}/timeline/{saved['event_id']}", headers=headers)
    assert d.status_code == 200, d.text
    print(f"final count={timeline_count()} (cleaned up)")
