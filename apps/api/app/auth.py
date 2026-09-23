from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .services.auth_store import UserRecord, get_store

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserRecord:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = get_store().get_user_by_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_patient_access(
    patient_id: str,
    user: UserRecord = Depends(get_current_user),
) -> UserRecord:
    if user.patient_id != patient_id:
        raise HTTPException(status_code=403, detail="Access denied for this patient")
    return user
