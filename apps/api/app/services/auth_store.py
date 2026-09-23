from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..schemas import ClinicalEvent, Patient


Sex = Literal["female", "male", "other", "unknown"]


@dataclass
class UserRecord:
    user_id: str
    email: str
    display_name: str
    patient_id: str
    sex: Sex
    password_hash: str
    password_salt: str
    created_at: str


@dataclass
class ArtifactRecord:
    artifact_id: str
    patient_id: str
    file_name: str
    media_type: str | None
    storage_path: str
    created_at: str


class AuthStore:
    def __init__(self, db_path: Path | None = None) -> None:
        # auth_store.py lives at apps/api/app/services/
        # parents[4] = repo root, parents[3] = apps/ (legacy path used earlier)
        repo_root = Path(__file__).resolve().parents[4]
        apps_root = Path(__file__).resolve().parents[3]
        canonical = repo_root / "data" / "healthapp.sqlite3"
        legacy = apps_root / "data" / "healthapp.sqlite3"
        if db_path is not None:
            self.db_path = db_path
        elif legacy.exists() and not canonical.exists():
            self.db_path = legacy
        elif legacy.exists() and canonical.exists():
            # Prefer the database that already has users.
            self.db_path = legacy if legacy.stat().st_size >= canonical.stat().st_size else canonical
        else:
            self.db_path = canonical
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    patient_id TEXT NOT NULL UNIQUE,
                    sex TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    event_time TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    media_type TEXT,
                    storage_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_patient
                ON events(patient_id);

                CREATE INDEX IF NOT EXISTS idx_artifacts_patient
                ON artifacts(patient_id);
                """
            )

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
        salt_value = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_value.encode("utf-8"),
            120_000,
        ).hex()
        return digest, salt_value

    def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str,
        sex: Sex = "unknown",
    ) -> UserRecord:
        normalized = email.strip().lower()
        if "@" not in normalized or "." not in normalized.split("@")[-1]:
            raise ValueError("Invalid email address")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        if not display_name.strip():
            raise ValueError("Display name is required")

        password_hash, password_salt = self._hash_password(password)
        user = UserRecord(
            user_id=f"user_{uuid4().hex[:12]}",
            email=normalized,
            display_name=display_name.strip(),
            patient_id=f"p_{uuid4().hex[:12]}",
            sex=sex,
            password_hash=password_hash,
            password_salt=password_salt,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, email, display_name, patient_id, sex,
                        password_hash, password_salt, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.user_id,
                        user.email,
                        user.display_name,
                        user.patient_id,
                        user.sex,
                        user.password_hash,
                        user.password_salt,
                        user.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Email is already registered") from exc
        return user

    def authenticate(self, email: str, password: str) -> UserRecord:
        user = self.get_user_by_email(email)
        if user is None:
            raise ValueError("Invalid email or password")
        digest, _ = self._hash_password(password, user.password_salt)
        if not secrets.compare_digest(digest, user.password_hash):
            raise ValueError("Invalid email or password")
        return user

    def change_password(
        self,
        *,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> None:
        if len(new_password) < 6:
            raise ValueError("Password must be at least 6 characters")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                raise ValueError("User not found")
            user = self._row_to_user(row)
            digest, _ = self._hash_password(current_password, user.password_salt)
            if not secrets.compare_digest(digest, user.password_hash):
                raise ValueError("Current password is incorrect")
            password_hash, password_salt = self._hash_password(new_password)
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?
                WHERE user_id = ?
                """,
                (password_hash, password_salt, user_id),
            )

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (token, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (token, user_id, datetime.now(timezone.utc).isoformat()),
            )
        return token

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def get_user_by_token(self, token: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.*
                FROM sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (email.strip().lower(),),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_patient_id(self, patient_id: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def to_patient(self, user: UserRecord) -> Patient:
        return Patient(
            patient_id=user.patient_id,
            display_name=user.display_name,
            sex=user.sex,
        )

    def list_events(self, patient_id: str) -> list[ClinicalEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM events
                WHERE patient_id = ?
                ORDER BY COALESCE(event_time, '') DESC, created_at DESC
                """,
                (patient_id,),
            ).fetchall()
        return [ClinicalEvent.model_validate(json.loads(row["payload_json"])) for row in rows]

    def add_event(self, event: ClinicalEvent) -> ClinicalEvent:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (event_id, patient_id, payload_json, event_time, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    event_time = excluded.event_time
                """,
                (
                    event.event_id,
                    event.patient_id,
                    event.model_dump_json(),
                    str(event.event_time.start) if event.event_time.start else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return event

    def replace_events(self, patient_id: str, events: list[ClinicalEvent]) -> int:
        with self._connect() as conn:
            conn.execute("DELETE FROM events WHERE patient_id = ?", (patient_id,))
            for event in events:
                if event.patient_id != patient_id:
                    raise ValueError("patient_id mismatch in event payload")
                conn.execute(
                    """
                    INSERT INTO events (event_id, patient_id, payload_json, event_time, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.patient_id,
                        event.model_dump_json(),
                        str(event.event_time.start) if event.event_time.start else None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        return len(events)

    def uploads_root(self) -> Path:
        return self.db_path.parent / "uploads"

    def save_artifact_file(
        self,
        *,
        patient_id: str,
        artifact_id: str,
        file_name: str,
        media_type: str | None,
        content: bytes,
    ) -> ArtifactRecord:
        safe_name = Path(file_name).name or "upload.bin"
        target_dir = self.uploads_root() / patient_id / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        target_path.write_bytes(content)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            patient_id=patient_id,
            file_name=safe_name,
            media_type=media_type,
            storage_path=str(target_path),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, patient_id, file_name, media_type, storage_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    media_type = excluded.media_type,
                    storage_path = excluded.storage_path
                """,
                (
                    record.artifact_id,
                    record.patient_id,
                    record.file_name,
                    record.media_type,
                    record.storage_path,
                    record.created_at,
                ),
            )
        return record

    def get_event(self, patient_id: str, event_id: str) -> ClinicalEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM events
                WHERE patient_id = ? AND event_id = ?
                """,
                (patient_id, event_id),
            ).fetchone()
        if row is None:
            return None
        return ClinicalEvent.model_validate(json.loads(row["payload_json"]))

    def delete_event(self, patient_id: str, event_id: str) -> bool:
        event = self.get_event(patient_id, event_id)
        if event is None:
            return False
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM events WHERE patient_id = ? AND event_id = ?",
                (patient_id, event_id),
            )
        for artifact_id in event.source_artifact_ids:
            self.delete_artifact(patient_id, artifact_id)
        return True

    def delete_artifact(self, patient_id: str, artifact_id: str) -> bool:
        artifact = self.get_artifact(patient_id, artifact_id)
        if artifact is None:
            return False
        path = Path(artifact.storage_path)
        if path.is_file():
            path.unlink()
        # Clean empty parent dirs when possible.
        try:
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except OSError:
            pass
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM artifacts WHERE patient_id = ? AND artifact_id = ?",
                (patient_id, artifact_id),
            )
        return True

    def get_artifact(self, patient_id: str, artifact_id: str) -> ArtifactRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT artifact_id, patient_id, file_name, media_type, storage_path, created_at
                FROM artifacts
                WHERE patient_id = ? AND artifact_id = ?
                """,
                (patient_id, artifact_id),
            ).fetchone()
        if row is None:
            return None
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            patient_id=row["patient_id"],
            file_name=row["file_name"],
            media_type=row["media_type"],
            storage_path=row["storage_path"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            email=row["email"],
            display_name=row["display_name"],
            patient_id=row["patient_id"],
            sex=row["sex"],
            password_hash=row["password_hash"],
            password_salt=row["password_salt"],
            created_at=row["created_at"],
        )


_STORE: AuthStore | None = None


def get_store() -> AuthStore:
    global _STORE
    if _STORE is None:
        _STORE = AuthStore()
    return _STORE
