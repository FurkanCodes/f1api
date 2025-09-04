from __future__ import annotations
from typing import Iterable, Set
from datetime import datetime
from fastapi import HTTPException


def to_iso(dt) -> str | None:
    try:
        if dt is None:
            return None
        if isinstance(dt, datetime):
            return dt.isoformat()
        # Fallback to str if already string-like
        return str(dt)
    except Exception:
        return None


def validate_fields(requested: Iterable[str] | None, allowed: Set[str]) -> Set[str]:
    if requested is None:
        return allowed
    req = {f.lower() for f in requested}
    invalid = req - allowed
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported fields: {sorted(invalid)}. Allowed: {sorted(allowed)}")
    return req

