from __future__ import annotations
from typing import Iterable, Set, Any, List, Tuple, Dict
from datetime import datetime, date
import math
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
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


def json_safe(value: Any) -> Any:
    """Recursively convert common Pandas/NumPy/datetime values to JSON-safe types.

    - NaN/NaT -> None
    - numpy scalars -> Python scalars
    - pandas Timestamp / datetime / date -> ISO string
    - dict/list/tuple/set -> recurse
    - fallback -> str(value)
    """
    try:
        # None and basic primitives
        if value is None or isinstance(value, (str, int, float, bool)):
            # normalize NaN floats
            if isinstance(value, float) and math.isnan(value):
                return None
            return value

        # NumPy scalars -> native Python
        if np is not None and isinstance(value, np.generic):  # type: ignore[attr-defined]
            v = value.item()
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        # Pandas missing
        if pd is not None and (pd.isna(value) if hasattr(pd, "isna") else False):  # type: ignore[call-arg]
            return None

        # Datetime-like
        if isinstance(value, (datetime, date)):
            return to_iso(value)
        # Many objects (e.g., pandas.Timestamp) expose isoformat
        if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
            try:
                return value.isoformat()  # type: ignore[no-any-return]
            except Exception:
                pass

        # Containers
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(v) for v in value]

        return str(value)
    except Exception:
        return None


def paginate_list(items: List[Any], page: int = 1, per_page: int = 50, *, max_per_page: int = 500) -> Tuple[List[Any], Dict[str, int | bool]]:
    """Slice a list and return items along with pagination metadata.

    Ensures page>=1 and per_page within 1..max_per_page. Returns:
    - sliced items
    - metadata: {page, per_page, total, total_pages, has_next, has_prev}
    """
    try:
        if page is None or page < 1:
            page = 1
        if per_page is None or per_page < 1:
            per_page = 1
        if per_page > max_per_page:
            per_page = max_per_page
        total = len(items)
        total_pages = max(1, (total + per_page - 1) // per_page)
        if page > total_pages:
            page = total_pages
        start = (page - 1) * per_page
        end = start + per_page
        sliced = items[start:end]
        meta: Dict[str, int | bool] = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }
        return sliced, meta
    except Exception:
        # Fallback: return original with minimal metadata
        return items, {
            "page": 1,
            "per_page": len(items),
            "total": len(items),
            "total_pages": 1,
            "has_next": False,
            "has_prev": False,
        }
