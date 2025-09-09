from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import hashlib
import pandas as pd
import numpy as np

from api.models import (
    SessionBasicResponse,
    SessionResultsResponse,
    LapTimesResponse,
    LapTimeItem,
    PitStopsResponse,
    PitStopItem,
    FlagsResponse,
    TrackStatusSegment,
)
from api.services import get_session_cached, resolve_event_name
from api.concurrency import run_in_thread
from api.utils import to_iso, paginate_list


router = APIRouter(tags=["Sessions"])


@router.get("/session/{year}/{event}/{session_type}", response_model=SessionBasicResponse)
async def get_session_info(year: int, event: str, session_type: str):
    try:
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)

        track_status = None
        if hasattr(session, "track_status") and session.track_status is not None:
            try:
                track_status = session.track_status.to_dict()
            except Exception:
                track_status = None

        weather = None
        if hasattr(session, "weather_data") and session.weather_data is not None:
            try:
                weather = session.weather_data.reset_index(drop=False).to_dict("records")
            except Exception:
                weather = None

        return {
            "year": year,
            "event": event,
            "session_type": session_type,
            "session_name": getattr(session, "name", None),
            "date": to_iso(getattr(session, "date", None)),
            "track_status": track_status,
            "weather": weather,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/flags/{year}/{event}/{session_type}", response_model=FlagsResponse)
async def get_flags(year: int, event: str, session_type: str):
    try:
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)

        segments: List[TrackStatusSegment] = []
        ts = getattr(session, "track_status", None)
        if ts is None:
            return {"year": year, "event": event, "session_type": session_type, "segments": []}

        try:
            df = ts.reset_index(drop=False)
        except Exception:
            # Try to coerce to DataFrame
            try:
                df = ts.to_frame(name="Status").reset_index(drop=False)
            except Exception:
                df = None  # type: ignore

        if df is None or len(df) == 0:
            return {"year": year, "event": event, "session_type": session_type, "segments": []}

        # Identify status column
        status_col = None
        for col in ("Status", "TrackStatus", "StatusCode"):
            if col in df.columns:
                status_col = col
                break
        if status_col is None:
            # if single unnamed column exists after index, pick last column
            status_col = df.columns[-1]

        # Label mapping (best-effort; codes vary by feed)
        label_map = {
            1: "GREEN",
            2: "YELLOW",
            3: "RED",
            4: "SC",
            5: "VSC",
            6: "VSC_END",
        }

        # Walk rows and cut segments on changes
        prev_code = None
        seg_start_time = None
        prev_time = None

        for _, row in df.iterrows():
            t = row.get("Time") or row.get("Date") or row.get("SessionTime")
            code = row.get(status_col)
            # Convert pandas Timestamp to python datetime
            if hasattr(t, "to_pydatetime"):
                t = t.to_pydatetime()

            if prev_code is None:
                prev_code = int(code) if pd.notna(code) else None
                seg_start_time = t
                prev_time = t
                continue

            code_int = int(code) if pd.notna(code) else None
            if code_int != prev_code:
                # close previous
                duration = None
                try:
                    duration = (prev_time - seg_start_time).total_seconds() if seg_start_time and prev_time else None
                except Exception:
                    duration = None
                segments.append(
                    TrackStatusSegment(
                        start_time=to_iso(seg_start_time),
                        end_time=to_iso(prev_time),
                        code=prev_code,
                        label=label_map.get(prev_code, str(prev_code) if prev_code is not None else None),
                        duration_seconds=duration,
                    )
                )
                # start new
                prev_code = code_int
                seg_start_time = t
            prev_time = t

        # close last
        if prev_code is not None:
            duration = None
            try:
                duration = (prev_time - seg_start_time).total_seconds() if seg_start_time and prev_time else None
            except Exception:
                duration = None
            segments.append(
                TrackStatusSegment(
                    start_time=to_iso(seg_start_time),
                    end_time=to_iso(prev_time),
                    code=prev_code,
                    label=label_map.get(prev_code, str(prev_code)),
                    duration_seconds=duration,
                )
            )

        return {"year": year, "event": event, "session_type": session_type, "segments": segments}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/pits/{year}/{event}/{session_type}", response_model=PitStopsResponse)
async def get_pit_stops(
    year: int,
    event: str,
    session_type: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(100, ge=1, le=2000, description="Items per page (max 2000)"),
):
    try:
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
        laps = session.laps

        items: List[PitStopItem] = []
        # Prefer PitInTime/PitOutTime if present
        has_in = "PitInTime" in laps.columns
        has_out = "PitOutTime" in laps.columns

        if has_in or has_out:
            for _, lap in laps.iterrows():
                pit_in = lap.get("PitInTime") if has_in else None
                pit_out = lap.get("PitOutTime") if has_out else None
                if pd.isna(pit_in) and pd.isna(pit_out):
                    continue
                # Estimate in/out lap numbers
                in_lap = int(lap.get("LapNumber")) if pd.notna(lap.get("LapNumber")) else None
                out_lap = in_lap + 1 if in_lap is not None else None

                # Compute duration when both are available
                dur = None
                try:
                    if pd.notna(pit_in) and pd.notna(pit_out):
                        dur = float((pit_out - pit_in).total_seconds())
                except Exception:
                    dur = None

                items.append(
                    PitStopItem(
                        driver=str(lap.get("Driver")),
                        in_lap=in_lap,
                        out_lap=out_lap,
                        pit_in_time=str(pit_in) if pd.notna(pit_in) else None,
                        pit_out_time=str(pit_out) if pd.notna(pit_out) else None,
                        duration_seconds=dur,
                    )
                )
        else:
            # Fallback: derive from 'PitOutLap' / 'IsPitOut' flags if present
            in_flags = set()
            out_flags = set(laps.index[laps.get("PitOutLap", False)].tolist()) if "PitOutLap" in laps.columns else set()
            if "IsPitOut" in laps.columns:
                out_flags |= set(laps.index[laps["IsPitOut"].fillna(False)].tolist())

            for idx, lap in laps.iterrows():
                in_lap = int(lap.get("LapNumber")) if pd.notna(lap.get("LapNumber")) else None
                out_lap = in_lap + 1 if in_lap is not None else None
                if idx in in_flags or idx in out_flags:
                    items.append(
                        PitStopItem(
                            driver=str(lap.get("Driver")),
                            in_lap=in_lap,
                            out_lap=out_lap,
                        )
                    )

        items_paged, meta = paginate_list(items, page=page, per_page=per_page, max_per_page=2000)
        return {"year": year, "event": event, "session_type": session_type, "items": items_paged, **meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results/{year}/{event}/{session_type}", response_model=SessionResultsResponse)
async def get_session_results(
    year: int,
    event: str,
    session_type: str,
    request: Request,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(50, ge=1, le=500, description="Items per page (max 500)"),
):
    try:
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
        results = session.results
        if results is not None and not results.empty:
            results_dict = results.to_dict("records")
            for result in results_dict:
                for key, value in list(result.items()):
                    if pd.isna(value):
                        result[key] = None
                    elif isinstance(value, pd.Timedelta):
                        result[key] = str(value)
                    elif isinstance(value, np.integer):
                        result[key] = int(value)
                    elif isinstance(value, np.floating):
                        result[key] = float(value)
        else:
            results_dict = []

        # Pagination for results
        results_paged, meta = paginate_list(results_dict, page=page, per_page=per_page)

        payload = {
            "session_info": {
                "year": year,
                "event": event,
                "session_type": session_type,
                "session_name": getattr(session, "name", None),
            },
            "results": results_paged,
            **meta,
        }
        content = jsonable_encoder(payload, exclude_none=True)
        etag = hashlib.md5(str(content).encode("utf-8")).hexdigest()
        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})
        resp = JSONResponse(content=content)
        resp.headers["ETag"] = etag
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/laptimes/{year}/{event}/{session_type}", response_model=LapTimesResponse)
async def get_lap_times(
    year: int,
    event: str,
    session_type: str,
    drivers: Optional[List[str]] = Query(None, description="Driver abbreviations, e.g. drivers=VER&drivers=HAM"),
    exclude_pit: bool = Query(False, description="Exclude in/out laps"),
    exclude_invalid: bool = Query(False, description="Exclude invalid/accurate=False laps"),
    request: Request = None,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(100, ge=1, le=2000, description="Items per page (max 2000)"),
):
    try:
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
        laps = session.laps
        if drivers:
            drivers = [d.upper() for d in drivers]
            laps = laps[laps["Driver"].isin(drivers)]
        try:
            if exclude_pit:
                if "PitInTime" in laps.columns:
                    laps = laps[laps["PitInTime"].isna()]
                if "PitOutTime" in laps.columns:
                    laps = laps[laps["PitOutTime"].isna()]
            if exclude_invalid:
                if "Deleted" in laps.columns:
                    laps = laps[~laps["Deleted"].fillna(False)]
                if "IsAccurate" in laps.columns:
                    laps = laps[laps["IsAccurate"].fillna(True)]
        except Exception:
            pass

        lap_data: List[LapTimeItem] = []
        for _, lap in laps.iterrows():
            def td_to_seconds(val):
                try:
                    return float(val.total_seconds())
                except Exception:
                    return None

            lap_data.append(
                LapTimeItem(
                    driver=str(lap.get("Driver")),
                    lap_number=int(lap.get("LapNumber")),
                    lap_time=str(lap.get("LapTime")) if pd.notna(lap.get("LapTime")) else None,
                    lap_time_seconds=td_to_seconds(lap.get("LapTime")) if pd.notna(lap.get("LapTime")) else None,
                    sector1_time=str(lap.get("Sector1Time")) if pd.notna(lap.get("Sector1Time")) else None,
                    sector2_time=str(lap.get("Sector2Time")) if pd.notna(lap.get("Sector2Time")) else None,
                    sector3_time=str(lap.get("Sector3Time")) if pd.notna(lap.get("Sector3Time")) else None,
                    compound=lap.get("Compound") if pd.notna(lap.get("Compound")) else None,
                    tyre_life=int(lap.get("TyreLife")) if pd.notna(lap.get("TyreLife")) else None,
                    fresh_tyre=bool(lap.get("FreshTyre")) if pd.notna(lap.get("FreshTyre")) else None,
                    team=lap.get("Team") if pd.notna(lap.get("Team")) else None,
                    is_personal_best=bool(lap.get("IsPersonalBest")) if pd.notna(lap.get("IsPersonalBest")) else None,
                )
            )

        # Pagination
        lap_paged, meta = paginate_list(lap_data, page=page, per_page=per_page, max_per_page=2000)
        payload = {"session_info": {"year": year, "event": event, "session_type": session_type}, "lap_times": lap_paged, **meta}
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
