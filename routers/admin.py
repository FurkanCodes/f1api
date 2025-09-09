from fastapi import APIRouter
from api.services import clear_session_cache


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/cache/clear")
async def cache_clear():
    cleared = clear_session_cache()
    return {"cleared": cleared, "remaining": 0}


@router.post("/drivers-cache/clear")
async def drivers_cache_clear():
    try:
        from routers.seasons import clear_drivers_cache, _standings_cache, _ergast_stand_cache, _ergast_latest_cache  # type: ignore
        n = clear_drivers_cache()
        # Also clear standings cache to force recompute
        try:
            _standings_cache.clear()
        except Exception:
            pass
        try:
            _ergast_stand_cache.clear()
            _ergast_latest_cache.clear()
        except Exception:
            pass
    except Exception:
        n = 0
    return {"cleared": n}


@router.post("/prewarm/standings")
async def prewarm_standings(year: int):
    """Precompute and cache standings for a year (Ergast and fallback)."""
    try:
        from routers.seasons import _fetch_ergast_standings, _fetch_ergast_latest_points, _compute_championship_standings, _standings_cache  # type: ignore
        # Preload Ergast caches
        _fetch_ergast_standings(year)
        _fetch_ergast_latest_points(year)
        # Preload fallback computation as well
        standings = await _compute_championship_standings(year)
        import time as _t
        _standings_cache[year] = (_t.time(), standings)
        return {"status": "ok", "year": year}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
