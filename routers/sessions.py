from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from api.models import SessionBasicResponse, SessionResultsResponse, LapTimesResponse, LapTimeItem
from api.services import get_session_cached, resolve_event_name
from api.concurrency import run_in_thread
from api.utils import to_iso


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


@router.get("/results/{year}/{event}/{session_type}", response_model=SessionResultsResponse)
async def get_session_results(year: int, event: str, session_type: str):
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

        return {
            "session_info": {
                "year": year,
                "event": event,
                "session_type": session_type,
                "session_name": getattr(session, "name", None),
            },
            "results": results_dict,
        }
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

        return {"session_info": {"year": year, "event": event, "session_type": session_type}, "lap_times": lap_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
