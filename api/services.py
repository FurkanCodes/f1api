import threading
from collections import OrderedDict
from typing import Tuple
from urllib.parse import unquote

import fastf1
import fastf1.plotting

from .config import get_settings


# Initialize FastF1 plotting and disk cache once
_settings = get_settings()
fastf1.plotting.setup_mpl()
fastf1.Cache.enable_cache(_settings.fastf1_cache_dir)


# Simple in-memory LRU cache for loaded sessions
_session_cache: "OrderedDict[Tuple[int, str, str], fastf1.core.Session]" = OrderedDict()
_cache_lock = threading.Lock()


def get_session_cached(year: int, event: str, session_type: str):
    """Return a fully-loaded FastF1 session with small LRU caching.

    Keyed by (year, event, session_type). Thread-safe.
    """
    key = (year, event, session_type)
    with _cache_lock:
        if key in _session_cache:
            sess = _session_cache.pop(key)
            _session_cache[key] = sess
            return sess

    sess = fastf1.get_session(year, event, session_type)
    sess.load()

    with _cache_lock:
        _session_cache[key] = sess
        # Trim LRU if beyond capacity
        while len(_session_cache) > _settings.session_cache_max:
            _session_cache.popitem(last=False)
    return sess


def _normalize_event_name(name: str) -> str:
    s = unquote(str(name)).strip()
    s = " ".join(s.split())
    return s.lower()


def resolve_event_name(year: int, user_event: str) -> str:
    """Resolve a user-provided event string to the exact EventName from schedule.

    Performs case-insensitive and whitespace-normalized matching.
    Raises ValueError if no match is found.
    """
    schedule = fastf1.get_event_schedule(year)
    target = _normalize_event_name(user_event)
    for evt in schedule.to_dict("records"):
        name = evt.get("EventName")
        if name and _normalize_event_name(name) == target:
            return str(name)
    # Try loose contains match as a second chance
    for evt in schedule.to_dict("records"):
        name = evt.get("EventName")
        if name and target in _normalize_event_name(name):
            return str(name)
    raise ValueError(f"Event '{user_event}' not found in {year} schedule. Check /sessions/{year} for exact names.")


def clear_session_cache() -> int:
    """Clear the in-memory session LRU cache. Returns number of entries cleared."""
    with _cache_lock:
        n = len(_session_cache)
        _session_cache.clear()
        return n
