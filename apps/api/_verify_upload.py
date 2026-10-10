"""End-to-end: upload test3 via the API and inspect extracted_event."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

D = Path(r"C:\Users\31683\Desktop\图片识别测试")
client = TestClient(app)

login = client.post(
    "/api/auth/login",
    json={"email": "ocr_test2@example.com", "password": "test123456"},
)
assert login.status_code == 200, login.text
token = login.json()["token"]
patient_id = login.json()["user"]["patient_id"]
headers = {"Authorization": f"Bearer {token}"}

with (D / "test3.jpeg").open("rb") as f:
    resp = client.post(
        f"/api/patients/{patient_id}/documents",
        files={"file": ("test3.jpeg", f, "image/jpeg")},
        headers=headers,
    )
print(f"status={resp.status_code}")
body = resp.json()
event = body.get("extracted_event")
print(f"pipeline_steps: {body.get('pipeline_steps')}")
print(f"status: {body.get('status')}")
if event:
    print(f"event_type={event['event_type']} title={event['title']} facts={len(event['facts'])}")
    for fact in event["facts"][:3]:
        v = fact["value"]
        vs = f"{v['number']} {v.get('unit') or ''}".strip() if v["type"] == "quantity" else f"[text] {v['text']}"
        print(
            f"  {fact['concept']['display']} | {vs} | "
            f"ref={fact['reference_range']} | {fact['interpretation']} | conf={fact['confidence']}"
        )
else:
    print("NO extracted_event")
    print(body.get("artifact", {}).get("ocr_text", "")[:200])
