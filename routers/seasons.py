from datetime import datetime
from typing import List, Optional, Dict, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import Request, Response
import hashlib
import fastf1
import time
try:
    import fastf1.plotting as f1plot
except Exception:  # pragma: no cover
    f1plot = None  # type: ignore

from api.models import DriversResponse, SeasonScheduleResponse
from api.concurrency import run_in_thread
from api.utils import json_safe, paginate_list


router = APIRouter(tags=["Seasons"])

# Simple in-memory cache for drivers per year with TTL
_drivers_cache: Dict[int, Tuple[float, List[dict]]] = {}
from api.config import get_settings
_settings = get_settings()
_DRIVERS_CACHE_TTL_SECONDS = int(getattr(_settings, "drivers_cache_ttl_seconds", 6 * 3600))

# Championship standings cache per year (abbr -> {points, position})
_standings_cache: Dict[int, Tuple[float, Dict[str, Dict[str, int | float]]]] = {}
# Ergast caches
_ergast_stand_cache: Dict[int, Tuple[float, List[dict], List[dict]]] = {}
_ergast_latest_cache: Dict[int, Tuple[float, Dict[str, float]]] = {}
_ERGAST_CACHE_TTL_SECONDS = int(getattr(_settings, "ergast_cache_ttl_seconds", 1800))


def clear_drivers_cache() -> int:
    n = len(_drivers_cache)
    _drivers_cache.clear()
    return n


async def _load_drivers_for_year(year: int) -> List[dict]:
    schedule = await run_in_thread(fastf1.get_event_schedule, year, include_testing=False)
    if schedule.empty:
        raise ValueError(f"No events found for the year {year}")

    now_utc = datetime.utcnow()
    completed_events = schedule[schedule["EventDate"] < now_utc]
    event_to_load = completed_events.iloc[-1] if not completed_events.empty else schedule.iloc[0]

    session = fastf1.get_session(year, event_to_load["EventName"], "R")
    await run_in_thread(
        session.load,
        laps=False,
        telemetry=False,
        weather=False,
        messages=False,
    )

    drivers_info: List[dict] = []
    for drv_number in session.drivers:
        di = session.get_driver(drv_number)
        # Convert driver info (pandas Series-like) to a plain dict with all fields
        try:
            entry = di.to_dict()  # type: ignore[attr-defined]
        except Exception:
            entry = dict(di)

        # Normalize values to JSON‑safe types
        entry = {str(k): json_safe(v) for k, v in entry.items()}

        # Ensure some common aliases exist for convenience if available
        entry["abbreviation"] = entry.get("Abbreviation") or entry.get("Driver")
        entry["full_name"] = entry.get("FullName")
        entry["team"] = entry.get("TeamName")
        entry["country"] = entry.get("CountryCode")

        drivers_info.append(entry)

    return drivers_info


async def _compute_championship_standings(year: int) -> Dict[str, Dict[str, int | float]]:
    """Aggregate championship points up to now by summing race + sprint results.

    Returns mapping per driver abbreviation:
    {
      'VER': { 'total': 255.0, 'latest': 25.0, 'position': 1 }, ...
    }
    where 'latest' is points from the most recent completed event (race + sprint).
    """
    schedule = await run_in_thread(fastf1.get_event_schedule, year, include_testing=False)
    if schedule.empty:
        return {}

    totals: Dict[str, float] = {}
    latest_evt_pts: Dict[str, float] = {}
    last_completed_name: Optional[str] = None

    # Iterate all scheduled events in order; treat events with available race results as completed.
    for _, evt in schedule.iterrows():
        evt_name = evt.get("EventName")
        if not evt_name:
            continue
        # Event accumulator for "latest" tracking
        evt_points: Dict[str, float] = {}

        # Race points
        try:
            sess = fastf1.get_session(year, evt_name, "R")
            await run_in_thread(
                sess.load,
                laps=False,
                telemetry=False,
                weather=False,
                messages=False,
            )
            res = getattr(sess, "results", None)
            if res is not None and not res.empty:
                for r in res.to_dict("records"):
                    abbr = r.get("Abbreviation") or r.get("Driver")
                    pts = r.get("Points")
                    try:
                        abbr_s = str(abbr) if abbr else None
                        # Fallback if Points missing: compute from finishing position
                        val = None
                        if pts is not None:
                            try:
                                val = float(pts)
                            except Exception:
                                val = None
                        if val is None:
                            pos = r.get("Position") or r.get("PositionOrder") or r.get("PositionText")
                            try:
                                posi = int(str(pos).strip().replace("DNF", "").replace("DQ", ""))
                            except Exception:
                                posi = None
                            if posi is not None and 1 <= posi <= 10:
                                race_points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                                val = float(race_points[posi - 1])
                            else:
                                val = 0.0
                        if abbr_s is not None:
                            totals[abbr_s] = totals.get(abbr_s, 0.0) + val
                            evt_points[abbr_s] = evt_points.get(abbr_s, 0.0) + val
                    except Exception:
                        pass
        except Exception:
            pass

        # Sprint points when applicable
        try:
            # Try to load a sprint; if not available, this will raise
            s_sess = fastf1.get_session(year, evt_name, "S")
            await run_in_thread(
                s_sess.load,
                laps=False,
                telemetry=False,
                weather=False,
                messages=False,
            )
            sres = getattr(s_sess, "results", None)
            if sres is not None and not sres.empty:
                for r in sres.to_dict("records"):
                    abbr = r.get("Abbreviation") or r.get("Driver")
                    pts = r.get("Points")
                    try:
                        abbr_s = str(abbr) if abbr else None
                        val = None
                        if pts is not None:
                            try:
                                val = float(pts)
                            except Exception:
                                val = None
                        if val is None:
                            pos = r.get("Position") or r.get("PositionOrder") or r.get("PositionText")
                            try:
                                posi = int(str(pos).strip().replace("DNF", "").replace("DQ", ""))
                            except Exception:
                                posi = None
                            if posi is not None and 1 <= posi <= 8:
                                sprint_points = [8, 7, 6, 5, 4, 3, 2, 1]
                                val = float(sprint_points[posi - 1])
                            else:
                                val = 0.0
                        if abbr_s is not None:
                            totals[abbr_s] = totals.get(abbr_s, 0.0) + val
                            evt_points[abbr_s] = evt_points.get(abbr_s, 0.0) + val
                    except Exception:
                        pass
        except Exception:
            pass

        # If we collected any points for this event, declare it the latest completed
        if evt_points:
            latest_evt_pts = evt_points
            last_completed_name = str(evt_name)

    # Rank by points (descending); break ties by abbreviation
    sorted_items = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    standings: Dict[str, Dict[str, int | float]] = {}
    pos = 1
    for abbr, pts in sorted_items:
        standings[abbr] = {"total": float(pts), "latest": float(latest_evt_pts.get(abbr, 0.0)), "position": int(pos)}
        pos += 1
    return standings


def _ergast_get(url: str) -> dict | None:
    try:
        import json, urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _fetch_ergast_standings(year: int) -> tuple[list[dict], list[dict]] | None:
    """Fetch season standings from Ergast (drivers and constructors).

    Returns (drivers, constructors) lists or None on failure.
    """
    now = time.time()
    cached = _ergast_stand_cache.get(year)
    if cached and (now - cached[0]) <= _ERGAST_CACHE_TTL_SECONDS:
        return cached[1], cached[2]
    base = f"https://ergast.com/api/f1/{year}"
    dj = _ergast_get(base + "/driverStandings.json")
    cj = _ergast_get(base + "/constructorStandings.json")
    if not dj or not cj:
        return None

    try:
        d_lists = dj.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        c_lists = cj.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        d_items = (d_lists[0].get("DriverStandings", []) if d_lists else [])
        c_items = (c_lists[0].get("ConstructorStandings", []) if c_lists else [])

        drivers: list[dict] = []
        for it in d_items:
            drv = it.get("Driver", {})
            code = drv.get("code") or (drv.get("familyName", "")[:3].upper() if drv.get("familyName") else None)
            name = f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip()
            team = None
            cons = it.get("Constructors", [])
            if cons:
                team = cons[0].get("name")
            try:
                points = float(it.get("points", 0))
            except Exception:
                points = 0.0
            try:
                pos = int(it.get("position", 0)) or None
            except Exception:
                pos = None
            drivers.append({
                "abbreviation": code,
                "full_name": name,
                "team": team,
                "points": points,
                "position": pos,
            })

        constructors: list[dict] = []
        for it in c_items:
            cons = it.get("Constructor", {})
            name = cons.get("name")
            try:
                points = float(it.get("points", 0))
            except Exception:
                points = 0.0
            try:
                pos = int(it.get("position", 0)) or None
            except Exception:
                pos = None
            if name:
                constructors.append({"team": name, "points": points, "position": pos})

        _ergast_stand_cache[year] = (now, drivers, constructors)
        return drivers, constructors
    except Exception:
        return None


def _ergast_points_map(year: int) -> Dict[str, Dict[str, int | float]] | None:
    """Return mapping from driver code to totals/position from Ergast."""
    data = _fetch_ergast_standings(year)
    if not data:
        return None
    drivers, _ = data
    # Fetch latest event points from Ergast if available
    latest = _fetch_ergast_latest_points(year)
    mp: Dict[str, Dict[str, int | float]] = {}
    for d in drivers:
        abbr = str(d.get("abbreviation") or "").upper()
        if not abbr:
            continue
        mp[abbr] = {
            "total": float(d.get("points") or 0.0),
            "position": int(d.get("position") or 0) or None,  # type: ignore
            "latest": float(latest.get(abbr, 0.0)) if latest else 0.0,
        }
    return mp


def _fetch_ergast_latest_points(year: int) -> Dict[str, float] | None:
    """Fetch latest completed round (race + sprint) and return points per driver code."""
    now = time.time()
    cached = _ergast_latest_cache.get(year)
    if cached and (now - cached[0]) <= _ERGAST_CACHE_TTL_SECONDS:
        return cached[1]
    base = f"https://ergast.com/api/f1/{year}"
    rj = _ergast_get(base + "/last/results.json")
    sj = _ergast_get(base + "/last/sprint.json")
    if not rj and not sj:
        return None

    pts: Dict[str, float] = {}
    try:
        if rj:
            races = rj.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races:
                for res in races[0].get("Results", []):
                    drv = res.get("Driver", {})
                    code = drv.get("code") or (drv.get("familyName", "")[:3].upper() if drv.get("familyName") else None)
                    try:
                        p = float(res.get("points", 0))
                    except Exception:
                        p = 0.0
                    if code:
                        pts[code] = pts.get(code, 0.0) + p
        if sj:
            races = sj.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races:
                for res in races[0].get("SprintResults", []):
                    drv = res.get("Driver", {})
                    code = drv.get("code") or (drv.get("familyName", "")[:3].upper() if drv.get("familyName") else None)
                    try:
                        p = float(res.get("points", 0))
                    except Exception:
                        p = 0.0
                    if code:
                        pts[code] = pts.get(code, 0.0) + p
        _ergast_latest_cache[year] = (now, pts)
        return pts
    except Exception:
        return None


@router.get("/sessions/{year}", response_model=SeasonScheduleResponse)
async def get_season_schedule(year: int, request: Request):
    try:
        schedule = await run_in_thread(fastf1.get_event_schedule, year)
        payload = {"year": year, "events": schedule.to_dict("records")}
        content = jsonable_encoder(payload, exclude_none=True)
        raw = JSONResponse(content=content)
        # ETag support
        body_bytes = raw.body if raw.body is not None else jsonable_encoder(content)
        etag = hashlib.md5(str(content).encode("utf-8")).hexdigest()
        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})
        raw.headers["ETag"] = etag
        return raw
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drivers/{year}", response_model=DriversResponse)
async def get_drivers(
    year: int,
    abbreviation: Optional[List[str]] = Query(None, description="Filter by driver abbreviation; repeatable e.g. ?abbreviation=VER&abbreviation=HAM"),
    team: Optional[List[str]] = Query(None, description="Filter by team name (case-insensitive contains); repeatable"),
    country: Optional[List[str]] = Query(None, description="Filter by country code (case-insensitive); repeatable"),
    sort_by: Optional[str] = Query(None, description="Sort key: abbreviation|full_name|team|country or original keys like Abbreviation, FullName"),
    order: str = Query("asc", description="Sort order: asc or desc"),
    fields: Optional[List[str]] = Query(None, description="Limit fields; repeatable. Supports aliases and original FastF1 keys."),
    request: Request = None,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(50, ge=1, le=500, description="Items per page (max 500)"),
):
    try:
        # Load or reuse cache
        cached = _drivers_cache.get(year)
        now = time.time()
        if cached is None or (now - cached[0]) > _DRIVERS_CACHE_TTL_SECONDS:
            drivers_info = await _load_drivers_for_year(year)
            _drivers_cache[year] = (now, drivers_info)
        else:
            drivers_info = cached[1]

        # Merge championship standings (cached per year)
        # Prefer Ergast totals for reliability; fall back to local computation
        standings = _ergast_points_map(year) or {}
        if not standings:
            s_cached = _standings_cache.get(year)
            if s_cached is None or (now - s_cached[0]) > _DRIVERS_CACHE_TTL_SECONDS:
                standings = await _compute_championship_standings(year)
                _standings_cache[year] = (now, standings)
            else:
                standings = s_cached[1]

        items = drivers_info

        # Filtering
        if abbreviation:
            abbrs = {a.upper() for a in abbreviation}
            items = [d for d in items if (str(d.get("abbreviation") or "").upper() in abbrs)]

        if team:
            team_set = {t.strip().lower() for t in team}
            def team_match(n: Optional[str]) -> bool:
                if not n:
                    return False
                nl = n.lower()
                return any(t in nl for t in team_set)
            items = [d for d in items if team_match(str(d.get("team") or d.get("TeamName") or ""))]

        if country:
            countries = {c.strip().lower() for c in country}
            items = [d for d in items if str(d.get("country") or d.get("CountryCode") or "").lower() in countries]

        # Sorting
        key_map = {
            None: "abbreviation",
            "abbreviation": "abbreviation",
            "full_name": "full_name",
            "team": "team",
            "country": "country",
            "Abbreviation": "abbreviation",
            "FullName": "full_name",
            "TeamName": "team",
            "CountryCode": "country",
        }
        k = key_map.get(sort_by, sort_by) if sort_by is not None else key_map[None]

        def sort_key(d: dict):
            v = d.get(k)
            if v is None:
                return ""
            if isinstance(v, str):
                return v.lower()
            return v

        reverse = str(order).lower() == "desc"
        try:
            items = sorted(items, key=sort_key, reverse=reverse)
        except Exception:
            items = sorted(items, key=lambda d: str(d.get(k) or "").lower(), reverse=reverse)

        # Enrich with championship points and positions
        def enrich(d: dict) -> dict:
            abbr = str(d.get("abbreviation") or d.get("Abbreviation") or "").upper()
            st = standings.get(abbr)
            if st:
                total = st.get("total")
                latest = st.get("latest")
                posn = st.get("position")
                # Explicit cumulative vs latest
                d["season_total_points"] = total
                d["season_position"] = posn
                d["latest_event_points"] = latest
                # Back-compat aliases
                d["championship_points"] = total
                d["championship_position"] = posn
            return d

        items = [enrich(dict(x)) for x in items]

        # Field selection
        if fields:
            requested = set(fields)
            def project(d: dict) -> dict:
                return {f: d[f] for f in requested if f in d}
            items = [project(d) for d in items]

        # Pagination
        items_paged, meta = paginate_list(items, page=page, per_page=per_page)

        payload = {"year": year, "drivers": items_paged, "count": len(items_paged), **meta}
        content = jsonable_encoder(payload, exclude_none=True)
        etag = hashlib.md5(str(content).encode("utf-8")).hexdigest()
        if request is not None:
            inm = request.headers.get("if-none-match")
            if inm and inm == etag:
                return Response(status_code=304, headers={"ETag": etag})
        resp = JSONResponse(content=content)
        resp.headers["ETag"] = etag
        return resp
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve driver data for {year}. Reason: {str(e)}",
        )


@router.get("/standings/{year}")
async def get_standings(year: int, request: Request):
    try:
        # Prefer official Ergast standings when available
        erg = _fetch_ergast_standings(year)
        if erg:
            drivers_list, constructors_list = erg
            # Augment drivers with latest_event_points if available
            latest = _fetch_ergast_latest_points(year)
            if latest:
                for d in drivers_list:
                    code = str(d.get("abbreviation") or "").upper()
                    d["latest_event_points"] = float(latest.get(code, 0.0))
            payload = {"year": year, "drivers": drivers_list, "constructors": constructors_list}
            content = jsonable_encoder(payload, exclude_none=True)
            etag = hashlib.md5(str(content).encode("utf-8")).hexdigest()
            inm = request.headers.get("if-none-match")
            if inm and inm == etag:
                return Response(status_code=304, headers={"ETag": etag})
            resp = JSONResponse(content=content)
            resp.headers["ETag"] = etag
            return resp

        # Load drivers (for mapping names/teams) from cache, then compute totals
        now = time.time()
        cached = _drivers_cache.get(year)
        if cached is None or (now - cached[0]) > _DRIVERS_CACHE_TTL_SECONDS:
            drivers_info = await _load_drivers_for_year(year)
            _drivers_cache[year] = (now, drivers_info)
        else:
            drivers_info = cached[1]

        # Load standings (totals + latest + pos)
        s_cached = _standings_cache.get(year)
        if s_cached is None or (now - s_cached[0]) > _DRIVERS_CACHE_TTL_SECONDS:
            standings = await _compute_championship_standings(year)
            _standings_cache[year] = (now, standings)
        else:
            standings = s_cached[1]

        # Build driver standings list with metadata
        by_abbr = {str(d.get("abbreviation") or d.get("Abbreviation") or "").upper(): d for d in drivers_info}
        drivers_list: list[dict] = []
        for abbr, st in sorted(standings.items(), key=lambda kv: (int(kv[1].get("position", 9999)))):
            meta = by_abbr.get(abbr, {})
            drivers_list.append({
                "abbreviation": abbr,
                "full_name": meta.get("full_name") or meta.get("FullName"),
                "team": meta.get("team") or meta.get("TeamName"),
                "points": st.get("total"),
                "latest_event_points": st.get("latest"),
                "position": st.get("position"),
            })

        # Fallback: if no points yet (e.g., pre-season), list roster with zeroes
        if not drivers_list and drivers_info:
            for d in drivers_info:
                abbr = str(d.get("abbreviation") or d.get("Abbreviation") or "").upper()
                drivers_list.append({
                    "abbreviation": abbr,
                    "full_name": d.get("full_name") or d.get("FullName"),
                    "team": d.get("team") or d.get("TeamName"),
                    "points": 0.0,
                    "latest_event_points": 0.0,
                    "position": None,
                })
            # Sort alphabetically for stable order
            drivers_list = sorted(drivers_list, key=lambda x: (str(x.get("full_name") or x.get("abbreviation") or "").lower()))

        # Constructors by summing driver totals grouped by team
        team_points: dict[str, float] = {}
        for item in drivers_list:
            team = item.get("team")
            pts = item.get("points") or 0.0
            if team:
                team_points[str(team)] = team_points.get(str(team), 0.0) + float(pts)
        constructors_list = [
            {"team": t, "points": p}
            for t, p in sorted(team_points.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        # Add positions
        for idx, it in enumerate(constructors_list, start=1):
            it["position"] = idx if any(tp > 0 for _, tp in team_points.items()) else None

        payload = {"year": year, "drivers": drivers_list, "constructors": constructors_list}
        content = jsonable_encoder(payload, exclude_none=True)
        etag = hashlib.md5(str(content).encode("utf-8")).hexdigest()
        if request is not None:
            inm = request.headers.get("if-none-match")
            if inm and inm == etag:
                return Response(status_code=304, headers={"ETag": etag})
        resp = JSONResponse(content=content)
        resp.headers["ETag"] = etag
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/teams/{year}")
async def get_teams(
    year: int,
    name: Optional[List[str]] = Query(None, description="Filter by team name (contains, case-insensitive); repeatable"),
    fields: Optional[List[str]] = Query(None, description="Limit fields; repeatable"),
    sort_by: Optional[str] = Query("name", description="Sort key: name or TeamName"),
    order: str = Query("asc", description="Sort order: asc|desc"),
    include_drivers: bool = Query(False, description="Include team driver list (abbreviation, full_name, country)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
):
    try:
        # Reuse drivers to enumerate teams
        cached = _drivers_cache.get(year)
        if cached is None:
            drivers = await _load_drivers_for_year(year)
            _drivers_cache[year] = (time.time(), drivers)
        else:
            drivers = cached[1]

        team_map: Dict[str, dict] = {}
        for d in drivers:
            tname = d.get("team") or d.get("TeamName")
            if not tname:
                continue
            key = str(tname)
            if key not in team_map:
                color = None
                try:
                    if f1plot is not None and hasattr(f1plot, "get_team_color"):
                        color = f1plot.get_team_color(key)  # type: ignore[attr-defined]
                except Exception:
                    color = None
                team_map[key] = {
                    "name": key,
                    "TeamName": key,
                    "color": color,
                }
            if include_drivers:
                lst = team_map[key].setdefault("drivers", [])
                lst.append({
                    "abbreviation": d.get("abbreviation") or d.get("Abbreviation"),
                    "full_name": d.get("full_name") or d.get("FullName"),
                    "country": d.get("country") or d.get("CountryCode"),
                })

        items = list(team_map.values())

        # Filter by name contains
        if name:
            needles = {n.strip().lower() for n in name}
            items = [it for it in items if any(nn in it["name"].lower() for nn in needles)]

        # Sort
        k = "name" if sort_by in (None, "name", "TeamName") else "name"
        reverse = str(order).lower() == "desc"
        items = sorted(items, key=lambda x: str(x.get(k) or "").lower(), reverse=reverse)

        # Fields
        if fields:
            req = set(fields)
            items = [{f: it[f] for f in req if f in it} for it in items]

        items_paged, meta = paginate_list(items, page=page, per_page=per_page)
        return {"year": year, "teams": items_paged, "count": len(items_paged), **meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/circuits/{year}")
async def get_circuits(
    year: int,
    fields: Optional[List[str]] = Query(None, description="Limit fields; repeatable"),
    sort_by: Optional[str] = Query("round", description="Sort key: round|date|name|EventName"),
    order: str = Query("asc", description="Sort order: asc|desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
):
    try:
        schedule = await run_in_thread(fastf1.get_event_schedule, year)
        records = schedule.to_dict("records")
        items: List[dict] = []
        for r in records:
            name = r.get("EventName")
            off = r.get("OfficialEventName") or r.get("EventOfficialName")
            loc = r.get("Location") or r.get("EventLocation")
            country = r.get("Country")
            date = r.get("EventDate") or r.get("Session1Date")
            rnd = r.get("RoundNumber") or r.get("Round")
            item = dict(r)
            item.update({
                "name": name,
                "official_name": off,
                "location": loc,
                "country": country,
                "date": str(date) if date is not None else None,
                "round": int(rnd) if rnd is not None else None,
            })
            items.append(item)

        key_map = {
            None: "round",
            "round": "round",
            "date": "date",
            "name": "name",
            "EventName": "name",
        }
        k = key_map.get(sort_by, sort_by) if sort_by is not None else key_map[None]
        reverse = str(order).lower() == "desc"
        items = sorted(items, key=lambda x: str(x.get(k) or "").lower(), reverse=reverse)

        if fields:
            req = set(fields)
            items = [{f: it[f] for f in req if f in it} for it in items]

        items_paged, meta = paginate_list(items, page=page, per_page=per_page)
        return {"year": year, "circuits": items_paged, "count": len(items_paged), **meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
