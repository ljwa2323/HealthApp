"""End-to-end check: upload endpoint returns quality field correctly."""
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

for name in ["test1.webp", "test3.jpeg", "test5.jpg"]:
    with (D / name).open("rb") as f:
        resp = client.post(
            f"/api/patients/{patient_id}/documents",
            files={"file": (name, f, "image/jpeg")},
            headers=headers,
        )
    body = resp.json()
    q = body.get("quality")
    print(
        f"{name}: status={resp.status_code} quality.ok={q.get('ok') if q else None} "
        f"issues={q.get('issues') if q else None} "
        f"short_side={q.get('metrics', {}).get('short_side') if q else None}"
    )
