import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
from datetime import datetime

import fastf1
import fastf1.plotting

import pandas as pd
import numpy as np
from collections import OrderedDict
import threading

# Enable FastF1 plotting and cache
fastf1.plotting.setup_mpl()
# Allow overriding cache dir via env var
_cache_dir = os.getenv("FASTF1_CACHE_DIR", "./cache")
fastf1.Cache.enable_cache(_cache_dir)

# Simple in-memory LRU cache for loaded sessions
_SESSION_CACHE_MAX = int(os.getenv("SESSION_CACHE_MAX", "3"))
_session_cache: "OrderedDict[tuple[int, str, str], fastf1.core.Session]" = OrderedDict()
_cache_lock = threading.Lock()

def get_session_cached(year: int, event: str, session_type: str):
    key = (year, event, session_type)
    with _cache_lock:
        if key in _session_cache:
            sess = _session_cache.pop(key)
            _session_cache[key] = sess  # mark as most recently used
            return sess

    # Not cached -> create and fully load
    sess = fastf1.get_session(year, event, session_type)
    sess.load()

    with _cache_lock:
        _session_cache[key] = sess
        if len(_session_cache) > _SESSION_CACHE_MAX:
            _session_cache.popitem(last=False)
    return sess

# ---------- Pydantic models for nicer docs ----------

class SessionInfo(BaseModel):
    year: int
    event: str
    session_type: str
    session_name: Optional[str] = None

class SeasonScheduleResponse(BaseModel):
    year: int
    events: List[Dict[str, Any]]

class SessionBasicResponse(BaseModel):
    year: int
    event: str
    session_type: str
    session_name: Optional[str]
    date: Optional[str]
    track_status: Optional[Dict[str, Any]] = None
    weather: Optional[List[Dict[str, Any]]] = None

class SessionResultsResponse(BaseModel):
    session_info: SessionInfo
    results: List[Dict[str, Any]]

class TelemetryPayload(BaseModel):
    distance: Optional[List[float]] = None
    speed: Optional[List[float]] = None
    throttle: Optional[List[float]] = None
    brake: Optional[List[float]] = None
    rpm: Optional[List[float]] = None
    gear: Optional[List[int]] = None
    drs: Optional[List[int]] = None

class TelemetrySummary(BaseModel):
    max_speed: Optional[float] = None
    avg_speed: Optional[float] = None
    min_speed: Optional[float] = None
    avg_throttle: Optional[float] = None
    brake_usage_pct: Optional[float] = None
    max_rpm: Optional[float] = None
    avg_gear: Optional[float] = None
    drs_open_pct: Optional[float] = None
    total_distance: Optional[float] = None

class TelemetryResponse(BaseModel):
    driver: str
    lap_number: int
    lap_time: Optional[str]
    compound: Optional[str]
    telemetry: Optional[TelemetryPayload] = None
    summary: Optional[TelemetrySummary] = None

class BrakingZone(BaseModel):
    start_distance: float
    end_distance: float
    length_m: float
    duration_ms: float
    entry_speed: Optional[float] = None
    min_speed: Optional[float] = None
    exit_speed: Optional[float] = None
    avg_decel_mps2: Optional[float] = None
    avg_brake: Optional[float] = None

class BrakingResponse(BaseModel):
    driver: str
    lap_number: int
    zones: List[BrakingZone]

class CornerItem(BaseModel):
    index: int
    distance: float
    min_speed: float

class CornersResponse(BaseModel):
    driver: str
    lap_number: int
    corners: List[CornerItem]

class SectorDeltaResponse(BaseModel):
    driver1: str
    driver2: str
    s1_times: Dict[str, Optional[str]]
    s2_times: Dict[str, Optional[str]]
    s3_times: Dict[str, Optional[str]]
    total_times: Dict[str, Optional[str]]
    gaps: Dict[str, Optional[str]]

class LapTimeItem(BaseModel):
    driver: str
    lap_number: int
    lap_time: Optional[str]
    lap_time_seconds: Optional[float]
    sector1_time: Optional[str]
    sector2_time: Optional[str]
    sector3_time: Optional[str]
    compound: Optional[str] = None
    tyre_life: Optional[int] = None
    fresh_tyre: Optional[bool] = None
    team: Optional[str] = None
    is_personal_best: Optional[bool] = None

class LapTimesResponse(BaseModel):
    session_info: SessionInfo
    lap_times: List[LapTimeItem]

class CompareFastest(BaseModel):
    driver: str
    lap_time: Optional[str]
    compound: Optional[str]
    max_speed: Optional[float]
    avg_speed: Optional[float]

class CompareAverage(BaseModel):
    driver: str
    average_lap_time: Optional[str]
    total_laps: int

class CompareResponse(BaseModel):
    driver1: Dict[str, Any]
    driver2: Dict[str, Any]
    gap: Optional[str] = None
    average_gap: Optional[str] = None

class DriverInfo(BaseModel):
    abbreviation: str
    full_name: Optional[str]
    team: Optional[str]
    country: Optional[str]

class DriversResponse(BaseModel):
    year: int
    drivers: List[DriverInfo]
    count: int

# ---------- App with Swagger UI settings ----------

app = FastAPI(
    title="F1 Data Analysis API",
    description="Comprehensive F1 data analysis tool ",
    version="1.0.0",
    # Keep default /docs and /redoc, just tune Swagger behavior
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # hide giant models sidebar by default
        "docExpansion": "list",          # collapse operations, show as list
        "displayRequestDuration": True
    },
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally compress large responses (telemetry/laps)
if os.getenv("ENABLE_GZIP", "0").lower() in {"1", "true", "yes"}:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---------- OpenAPI customization (logo/contact/license/tags) ----------

TAGS_METADATA = [
    {"name": "Root", "description": "Service status & metadata."},
    {"name": "Seasons", "description": "Season schedules and driver rosters."},
    {"name": "Sessions", "description": "Session info, results, and lap data."},
    {"name": "Analysis", "description": "Comparisons and telemetry endpoints."},
]

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["tags"] = TAGS_METADATA
    openapi_schema["info"]["contact"] = {
        "name": "F1 Data Analysis",
    
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore

# ---------- Routes ----------

@app.get("/", tags=["Root"])
async def root():
    """Service status.

    Use the documented endpoints for seasons, sessions, telemetry, comparisons, and analysis.
    See README for analysis workflows; Swagger includes per-endpoint tips below.
    """
    return {"message": "F1 Data Analysis API", "version": "1.0.0"}

@app.get("/sessions/{year}", response_model=SeasonScheduleResponse, tags=["Seasons"])
async def get_season_schedule(year: int):
    """Season schedule for a year.

    Use this to discover exact `EventName` strings and available session types.

    Examples:
    - List events: `/sessions/2024`
    - Use returned `EventName` with other routes (e.g., `Monaco Grand Prix`).
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        return {
            "year": year,
            "events": schedule.to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/session/{year}/{event}/{session_type}",
    response_model=SessionBasicResponse,
    tags=["Sessions"]
)
async def get_session_info(year: int, event: str, session_type: str):
    """Basic session information and context.

    Returns session name, date, and—if available—track status and weather snapshots.

    Tips:
    - Weather/track_status may be missing for some sessions.
    - Use to log covariates for pace/strategy models.
    """
    try:
        session = get_session_cached(year, event, session_type)

        # Weather/track data may be missing depending on session
        track_status = None
        if hasattr(session, "track_status") and session.track_status is not None:
            # track_status is a pandas series indexed by time -> convert safely
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
            "date": str(getattr(session, "date", None)) if getattr(session, "date", None) else None,
            "track_status": track_status,
            "weather": weather,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/results/{year}/{event}/{session_type}",
    response_model=SessionResultsResponse,
    tags=["Sessions"]
)
async def get_session_results(year: int, event: str, session_type: str):
    """Session results/standings, normalized for JSON.

    Timedeltas are returned as strings; NaN→null. Use for tables and finishing orders.
    """
    try:
        session = get_session_cached(year, event, session_type)

        results = session.results
        if results is not None and not results.empty:
            results_dict = results.to_dict("records")
            # Clean up for JSON serialization
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

@app.get(
    "/telemetry/{year}/{event}/{session_type}",
    response_model=TelemetryResponse,
    tags=["Analysis"]
)
async def get_driver_telemetry(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(..., description="Driver abbreviation (e.g., VER, HAM)", examples={"example": {"value": "VER"}}),
    lap: Optional[int] = Query(None, description="Specific lap number; fastest lap if not provided", examples={"example": {"value": 44}}),
    sample_every: Optional[int] = Query(None, ge=1, description="Keep every Nth telemetry row to reduce payload size"),
    max_points: Optional[int] = Query(None, ge=10, le=20000, description="Downsample to approximately this many points"),
    distance_step: Optional[float] = Query(None, gt=0, description="Resample to fixed distance step (meters); takes precedence over time_step_ms"),
    time_step_ms: Optional[int] = Query(None, gt=0, description="Resample to fixed time step (milliseconds)"),
    fields: Optional[List[str]] = Query(None, description="Which series to return: distance,speed,throttle,brake,rpm,gear,drs"),
    summary_only: bool = Query(False, description="Return only summary metrics; omit arrays")
):
    """Lap telemetry for a driver with analysis-friendly controls.

    Usage:
    - Fastest lap arrays: set `driver=VER`.
    - Align for overlays: add `distance_step=5` (meters). Optionally add `max_points`.
    - Smaller payload: restrict `fields` (e.g., `fields=speed&fields=throttle`).
    - Summaries only: `summary_only=true` to get metrics without arrays.
    - Specific lap: set `lap` to a lap number.
    """
    try:
        session = get_session_cached(year, event, session_type)

        driver_laps = session.laps.pick_driver(driver)
        if driver_laps is None or driver_laps.empty:
            raise HTTPException(status_code=404, detail=f"No laps found for driver {driver}")

        if lap is None:
            selected_lap = driver_laps.pick_fastest()
        else:
            sel = driver_laps[driver_laps["LapNumber"] == lap]
            if sel.empty:
                raise HTTPException(status_code=404, detail=f"Lap {lap} not found for driver {driver}")
            selected_lap = sel.iloc[0]

        telemetry = selected_lap.get_telemetry()
        # Optional resampling to a fixed grid (distance or time)
        try:
            if distance_step is not None:
                dist = telemetry["Distance"].astype(float).to_numpy()
                if len(dist) > 1:
                    new_dist = np.arange(float(dist[0]), float(dist[-1]), float(distance_step))
                    speed = np.interp(new_dist, dist, telemetry["Speed"].astype(float).to_numpy())
                    throttle = np.interp(new_dist, dist, telemetry["Throttle"].astype(float).to_numpy())
                    brake = np.interp(new_dist, dist, telemetry["Brake"].astype(float).to_numpy())
                    rpm = np.interp(new_dist, dist, telemetry["RPM"].astype(float).to_numpy())
                    idx_float = np.interp(new_dist, dist, np.arange(len(dist)))
                    nn_idx = np.round(idx_float).astype(int).clip(0, len(dist) - 1)
                    gear = telemetry["nGear"].to_numpy()[nn_idx].astype(int)
                    drs = telemetry["DRS"].to_numpy()[nn_idx].astype(int)
                    telemetry = pd.DataFrame({
                        "Distance": new_dist,
                        "Speed": speed,
                        "Throttle": throttle,
                        "Brake": brake,
                        "RPM": rpm,
                        "nGear": gear,
                        "DRS": drs,
                    })
            elif time_step_ms is not None and "Time" in telemetry.columns:
                t = telemetry["Time"].dt.total_seconds().astype(float).to_numpy()
                if len(t) > 1:
                    step = float(time_step_ms) / 1000.0
                    new_t = np.arange(float(t[0]), float(t[-1]), step)
                    speed = np.interp(new_t, t, telemetry["Speed"].astype(float).to_numpy())
                    throttle = np.interp(new_t, t, telemetry["Throttle"].astype(float).to_numpy())
                    brake = np.interp(new_t, t, telemetry["Brake"].astype(float).to_numpy())
                    rpm = np.interp(new_t, t, telemetry["RPM"].astype(float).to_numpy())
                    idx_float = np.interp(new_t, t, np.arange(len(t)))
                    nn_idx = np.round(idx_float).astype(int).clip(0, len(t) - 1)
                    gear = telemetry["nGear"].to_numpy()[nn_idx].astype(int)
                    drs = telemetry["DRS"].to_numpy()[nn_idx].astype(int)
                    distance = np.interp(new_t, t, telemetry["Distance"].astype(float).to_numpy())
                    telemetry = pd.DataFrame({
                        "Distance": distance,
                        "Speed": speed,
                        "Throttle": throttle,
                        "Brake": brake,
                        "RPM": rpm,
                        "nGear": gear,
                        "DRS": drs,
                    })
        except Exception:
            # If resampling fails for any reason, fall back to raw data
            pass
        if sample_every is not None and sample_every > 1:
            telemetry = telemetry.iloc[::sample_every]
        if max_points is not None and max_points > 0 and len(telemetry) > max_points:
            idx = np.linspace(0, len(telemetry) - 1, num=max_points).round().astype(int)
            telemetry = telemetry.iloc[idx]

        # Build dynamic telemetry payload and summary
        selected_fields = set([f.lower() for f in fields]) if fields else {"distance","speed","throttle","brake","rpm","gear","drs"}
        payload: Dict[str, Any] = {
            "driver": driver,
            "lap_number": int(selected_lap["LapNumber"]),
            "lap_time": str(selected_lap.get("LapTime")) if pd.notna(selected_lap.get("LapTime")) else None,
            "compound": selected_lap.get("Compound"),
        }

        def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
            try:
                max_speed = float(df["Speed"].max()) if "speed" in selected_fields or summary_only else float(df["Speed"].max())
                avg_speed = float(df["Speed"].mean())
                min_speed = float(df["Speed"].min())
                avg_throttle = float(df["Throttle"].mean()) if "Throttle" in df.columns else None
                brake_usage_pct = float((df["Brake"] > 0.01).mean() * 100.0) if "Brake" in df.columns else None
                max_rpm = float(df["RPM"].max()) if "RPM" in df.columns else None
                avg_gear = float(df["nGear"].mean()) if "nGear" in df.columns else None
                drs_open_pct = float((df["DRS"] > 0).mean() * 100.0) if "DRS" in df.columns else None
                total_distance = float(df["Distance"].iloc[-1] - df["Distance"].iloc[0]) if "Distance" in df.columns and len(df) > 1 else None
                return {
                    "max_speed": max_speed,
                    "avg_speed": avg_speed,
                    "min_speed": min_speed,
                    "avg_throttle": avg_throttle,
                    "brake_usage_pct": brake_usage_pct,
                    "max_rpm": max_rpm,
                    "avg_gear": avg_gear,
                    "drs_open_pct": drs_open_pct,
                    "total_distance": total_distance,
                }
            except Exception:
                return {}

        if summary_only:
            payload["summary"] = build_summary(telemetry)
            payload["telemetry"] = None
            return payload

        tele_out: Dict[str, Any] = {}
        if "distance" in selected_fields:
            tele_out["distance"] = telemetry["Distance"].astype(float).tolist()
        if "speed" in selected_fields:
            tele_out["speed"] = telemetry["Speed"].astype(float).tolist()
        if "throttle" in selected_fields and "Throttle" in telemetry.columns:
            tele_out["throttle"] = telemetry["Throttle"].astype(float).tolist()
        if "brake" in selected_fields and "Brake" in telemetry.columns:
            tele_out["brake"] = telemetry["Brake"].astype(float).tolist()
        if "rpm" in selected_fields and "RPM" in telemetry.columns:
            tele_out["rpm"] = telemetry["RPM"].astype(float).tolist()
        if "gear" in selected_fields and "nGear" in telemetry.columns:
            tele_out["gear"] = telemetry["nGear"].astype(int).tolist()
        if "drs" in selected_fields and "DRS" in telemetry.columns:
            tele_out["drs"] = telemetry["DRS"].astype(int).tolist()

        payload["telemetry"] = tele_out
        payload["summary"] = build_summary(telemetry)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/laptimes/{year}/{event}/{session_type}",
    response_model=LapTimesResponse,
    tags=["Sessions"]
)
async def get_lap_times(
    year: int,
    event: str,
    session_type: str,
    drivers: Optional[List[str]] = Query(None, description="Driver abbreviations, e.g. drivers=VER&drivers=HAM"),
    exclude_pit: bool = Query(False, description="Exclude in/out laps"),
    exclude_invalid: bool = Query(False, description="Exclude invalid/accurate=False laps")
):
    """Per-lap dataset for pace analysis.

    Recommendations:
    - Race pace: add `exclude_pit=true&exclude_invalid=true`.
    - Filter drivers with repeated `drivers=ABR` params.
    - Engineer features client-side (stints, degradation) from returned fields.
    """
    try:
        session = get_session_cached(year, event, session_type)

        laps = session.laps
        if drivers:
            laps = laps[laps["Driver"].isin(drivers)]
        # Optional filters
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

        return {
            "session_info": {
                "year": year,
                "event": event,
                "session_type": session_type,
            },
            "lap_times": lap_data,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/compare/{year}/{event}/{session_type}",
    response_model=CompareResponse,
    tags=["Analysis"]
)
async def compare_drivers(
    year: int,
    event: str,
    session_type: str,
    driver1: str = Query(..., description="First driver abbreviation", examples={"example": {"value": "VER"}}),
    driver2: str = Query(..., description="Second driver abbreviation", examples={"example": {"value": "HAM"}}),
    lap_type: str = Query("fastest", description="'fastest' or 'average'", pattern="^(fastest|average)$"),
    include_telemetry: bool = Query(False, description="Include telemetry traces for overlays"),
    fields: Optional[List[str]] = Query(None, description="Telemetry series to include if included: distance,speed,throttle,brake,rpm,gear,drs"),
    distance_step: Optional[float] = Query(None, gt=0, description="Resample telemetry to fixed distance step (meters) for overlay"),
    sample_every: Optional[int] = Query(None, ge=1, description="Downsample telemetry by stride if included"),
    max_points: Optional[int] = Query(None, ge=10, le=20000, description="Target number of telemetry points if included"),
    exclude_pit: bool = Query(False, description="Exclude pit in/out laps from analysis"),
    exclude_invalid: bool = Query(False, description="Exclude invalid/accurate=False laps from analysis")
):
    """Compare two drivers on fastest or average laps; optionally include overlay telemetry.

    Tips:
    - Quali overlay: `lap_type=fastest&include_telemetry=true&distance_step=5&fields=speed`.
    - Race pace: `lap_type=average&exclude_pit=true&exclude_invalid=true`.
    - Telemetry arrays are added only when `include_telemetry=true`.
    """
    try:
        session = get_session_cached(year, event, session_type)

        driver1_laps = session.laps.pick_driver(driver1)
        driver2_laps = session.laps.pick_driver(driver2)
        # Optional lap filters
        def apply_filters(df):
            try:
                if exclude_pit:
                    if "PitInTime" in df.columns:
                        df = df[df["PitInTime"].isna()]
                    if "PitOutTime" in df.columns:
                        df = df[df["PitOutTime"].isna()]
                if exclude_invalid:
                    if "Deleted" in df.columns:
                        df = df[~df["Deleted"].fillna(False)]
                    if "IsAccurate" in df.columns:
                        df = df[df["IsAccurate"].fillna(True)]
            except Exception:
                pass
            return df

        driver1_laps = apply_filters(driver1_laps)
        driver2_laps = apply_filters(driver2_laps)
        if (driver1_laps is None or driver1_laps.empty) or (driver2_laps is None or driver2_laps.empty):
            raise HTTPException(status_code=404, detail="One or both drivers have no laps")

        if lap_type == "fastest":
            lap1 = driver1_laps.pick_fastest()
            lap2 = driver2_laps.pick_fastest()

            tel1 = lap1.get_telemetry()
            tel2 = lap2.get_telemetry()

            comparison = {
                "driver1": CompareFastest(
                    driver=driver1,
                    lap_time=str(lap1.get("LapTime")),
                    compound=lap1.get("Compound"),
                    max_speed=float(tel1["Speed"].max()) if not tel1.empty else None,
                    avg_speed=float(tel1["Speed"].mean()) if not tel1.empty else None,
                ).model_dump(),
                "driver2": CompareFastest(
                    driver=driver2,
                    lap_time=str(lap2.get("LapTime")),
                    compound=lap2.get("Compound"),
                    max_speed=float(tel2["Speed"].max()) if not tel2.empty else None,
                    avg_speed=float(tel2["Speed"].mean()) if not tel2.empty else None,
                ).model_dump(),
                "gap": str(abs(lap1.get("LapTime") - lap2.get("LapTime"))) if pd.notna(lap1.get("LapTime")) and pd.notna(lap2.get("LapTime")) else None,
            }
            if include_telemetry:
                sel_fields = set([f.lower() for f in fields]) if fields else {"distance","speed","throttle","brake","rpm","gear","drs"}
                # Optional resample by distance for better overlays
                if distance_step is not None and len(tel1) > 1 and len(tel2) > 1:
                    d1 = tel1["Distance"].astype(float).to_numpy()
                    d2 = tel2["Distance"].astype(float).to_numpy()
                    end = float(min(d1[-1], d2[-1]))
                    new_dist = np.arange(0.0, end, float(distance_step))
                    def resample(df, new_d):
                        base = {"Distance": new_d}
                        base["Speed"] = np.interp(new_d, df["Distance"].astype(float).to_numpy(), df["Speed"].astype(float).to_numpy())
                        if "Throttle" in df.columns:
                            base["Throttle"] = np.interp(new_d, df["Distance"].astype(float).to_numpy(), df["Throttle"].astype(float).to_numpy())
                        if "Brake" in df.columns:
                            base["Brake"] = np.interp(new_d, df["Distance"].astype(float).to_numpy(), df["Brake"].astype(float).to_numpy())
                        if "RPM" in df.columns:
                            base["RPM"] = np.interp(new_d, df["Distance"].astype(float).to_numpy(), df["RPM"].astype(float).to_numpy())
                        # nearest-neighbor for gear/drs
                        idx_float = np.interp(new_d, df["Distance"].astype(float).to_numpy(), np.arange(len(df)))
                        nn_idx = np.round(idx_float).astype(int).clip(0, len(df)-1)
                        if "nGear" in df.columns:
                            base["nGear"] = df["nGear"].to_numpy()[nn_idx].astype(int)
                        if "DRS" in df.columns:
                            base["DRS"] = df["DRS"].to_numpy()[nn_idx].astype(int)
                        return pd.DataFrame(base)
                    tel1 = resample(tel1, new_dist)
                    tel2 = resample(tel2, new_dist)
                # Optional downsampling
                if sample_every is not None and sample_every > 1:
                    tel1 = tel1.iloc[::sample_every]
                    tel2 = tel2.iloc[::sample_every]
                if max_points is not None and max_points > 0:
                    def uniform_down(df):
                        if len(df) <= max_points:
                            return df
                        idx = np.linspace(0, len(df)-1, num=max_points).round().astype(int)
                        return df.iloc[idx]
                    tel1 = uniform_down(tel1)
                    tel2 = uniform_down(tel2)

                def build_tel(df):
                    out = {}
                    if "distance" in sel_fields:
                        out["distance"] = df["Distance"].astype(float).tolist()
                    if "speed" in sel_fields:
                        out["speed"] = df["Speed"].astype(float).tolist()
                    if "throttle" in sel_fields and "Throttle" in df.columns:
                        out["throttle"] = df["Throttle"].astype(float).tolist()
                    if "brake" in sel_fields and "Brake" in df.columns:
                        out["brake"] = df["Brake"].astype(float).tolist()
                    if "rpm" in sel_fields and "RPM" in df.columns:
                        out["rpm"] = df["RPM"].astype(float).tolist()
                    if "gear" in sel_fields and "nGear" in df.columns:
                        out["gear"] = df["nGear"].astype(int).tolist()
                    if "drs" in sel_fields and "DRS" in df.columns:
                        out["drs"] = df["DRS"].astype(int).tolist()
                    return out

                comparison["driver1"].update({"telemetry": build_tel(tel1)})
                comparison["driver2"].update({"telemetry": build_tel(tel2)})
        else:
            avg1 = driver1_laps["LapTime"].mean() if not driver1_laps.empty else None
            avg2 = driver2_laps["LapTime"].mean() if not driver2_laps.empty else None

            comparison = {
                "driver1": CompareAverage(
                    driver=driver1,
                    average_lap_time=str(avg1) if avg1 is not None else None,
                    total_laps=len(driver1_laps),
                ).model_dump(),
                "driver2": CompareAverage(
                    driver=driver2,
                    average_lap_time=str(avg2) if avg2 is not None else None,
                    total_laps=len(driver2_laps),
                ).model_dump(),
                "average_gap": str(abs(avg1 - avg2)) if (avg1 is not None and avg2 is not None) else None,
            }

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/analysis/braking/{year}/{event}/{session_type}",
    response_model=BrakingResponse,
    tags=["Analysis"]
)
async def braking_zones(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(..., description="Driver abbreviation"),
    lap: Optional[int] = Query(None, description="Specific lap; fastest if omitted"),
    brake_threshold: float = Query(0.1, ge=0, description="Brake threshold (0-1 or percent if raw telemetry is 0-100)"),
    min_duration_ms: int = Query(200, ge=0, description="Minimum zone duration in ms"),
    min_length_m: float = Query(5.0, ge=0, description="Minimum zone length in meters")
):
    """Detect braking zones and compute zone-level metrics.

    Parameters:
    - brake_threshold: normalized threshold (0..1) after auto-scaling if input is 0..100.
    - min_duration_ms / min_length_m: suppress micro spikes and noise.

    Output includes: start/end distance, length, duration, entry/min/exit speed,
    average decel (m/s²), and average brake value per zone.
    """
    try:
        session = get_session_cached(year, event, session_type)
        laps = session.laps.pick_driver(driver)
        if laps is None or laps.empty:
            raise HTTPException(status_code=404, detail=f"No laps for {driver}")
        if lap is None:
            sel = laps.pick_fastest()
        else:
            tmp = laps[laps["LapNumber"] == lap]
            if tmp.empty:
                raise HTTPException(status_code=404, detail=f"Lap {lap} not found for {driver}")
            sel = tmp.iloc[0]

        tel = sel.get_telemetry()
        # Normalize brake to 0..1 if necessary
        if "Brake" not in tel.columns:
            raise HTTPException(status_code=400, detail="Brake channel not available")
        brake = tel["Brake"].astype(float).to_numpy()
        scale = 100.0 if np.nanmax(brake) > 1.5 else 1.0
        br_norm = brake / scale
        mask = br_norm > brake_threshold

        # Find contiguous segments of True in mask
        idx = np.arange(len(mask))
        change = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0] - 1

        zones: List[BrakingZone] = []
        dist = tel["Distance"].astype(float).to_numpy() if "Distance" in tel.columns else None
        speed_kmh = tel["Speed"].astype(float).to_numpy() if "Speed" in tel.columns else None
        time_s = tel["Time"].dt.total_seconds().astype(float).to_numpy() if "Time" in tel.columns else None

        for s, e in zip(starts, ends):
            if s >= len(mask) or e >= len(mask) or e <= s:
                continue
            # Duration and length
            dur_ms = None
            if time_s is not None:
                dur_ms = max(0.0, (time_s[e] - time_s[s]) * 1000.0)
            length_m = None
            if dist is not None:
                length_m = max(0.0, float(dist[e] - dist[s]))

            if dur_ms is not None and dur_ms < min_duration_ms:
                continue
            if length_m is not None and length_m < min_length_m:
                continue

            entry_sp = float(speed_kmh[s]) if speed_kmh is not None else None
            min_sp = float(np.nanmin(speed_kmh[s:e+1])) if speed_kmh is not None else None
            exit_sp = float(speed_kmh[e]) if speed_kmh is not None else None
            avg_br = float(np.nanmean(br_norm[s:e+1])) if (e > s) else float(br_norm[s])
            avg_decel = None
            if entry_sp is not None and min_sp is not None and dur_ms and dur_ms > 0:
                # Convert km/h to m/s for decel
                entry_mps = entry_sp / 3.6
                min_mps = min_sp / 3.6
                avg_decel = (entry_mps - min_mps) / (dur_ms / 1000.0)

            zones.append(BrakingZone(
                start_distance=float(dist[s]) if dist is not None else 0.0,
                end_distance=float(dist[e]) if dist is not None else 0.0,
                length_m=length_m or 0.0,
                duration_ms=dur_ms or 0.0,
                entry_speed=entry_sp,
                min_speed=min_sp,
                exit_speed=exit_sp,
                avg_decel_mps2=avg_decel,
                avg_brake=avg_br,
            ))

        return {"driver": driver, "lap_number": int(sel["LapNumber"]), "zones": [z.model_dump() for z in zones]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/analysis/corners/{year}/{event}/{session_type}",
    response_model=CornersResponse,
    tags=["Analysis"]
)
async def corner_minimums(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(...),
    lap: Optional[int] = Query(None),
    min_gap_m: float = Query(30.0, ge=0, description="Minimum distance between detected corners"),
    window_points: int = Query(21, ge=3, description="Local minima detection window (points)"),
    top_n: Optional[int] = Query(None, ge=1, description="Return only the N slowest corners")
):
    """Detect corner apex proxies via local speed minima along the lap.

    Parameters:
    - min_gap_m: enforce a minimum distance between picked minima to avoid duplicates.
    - window_points: local window size for minima detection (odd integer recommended).
    - top_n: optionally keep only the slowest N corners.

    Output: list of corners with distance and min_speed for quick comparison.
    """
    try:
        session = get_session_cached(year, event, session_type)
        laps = session.laps.pick_driver(driver)
        if laps is None or laps.empty:
            raise HTTPException(status_code=404, detail=f"No laps for {driver}")
        if lap is None:
            sel = laps.pick_fastest()
        else:
            tmp = laps[laps["LapNumber"] == lap]
            if tmp.empty:
                raise HTTPException(status_code=404, detail=f"Lap {lap} not found for {driver}")
            sel = tmp.iloc[0]

        tel = sel.get_telemetry()
        if "Speed" not in tel.columns or "Distance" not in tel.columns:
            raise HTTPException(status_code=400, detail="Required telemetry channels missing")

        # Simple local minima detection by comparing neighbors
        sp = tel["Speed"].astype(float).to_numpy()
        dist = tel["Distance"].astype(float).to_numpy()
        n = len(sp)
        corners: List[CornerItem] = []
        for i in range(1, n - 1):
            # Check window bounds
            lo = max(0, i - window_points // 2)
            hi = min(n, i + window_points // 2 + 1)
            if sp[i] == np.nanmin(sp[lo:hi]) and sp[i] < sp[i - 1] and sp[i] < sp[i + 1]:
                # Enforce minimum gap between picks
                if corners and (dist[i] - corners[-1].distance) < min_gap_m:
                    # Keep the lower speed one
                    if sp[i] < corners[-1].min_speed:
                        corners[-1] = CornerItem(index=i, distance=float(dist[i]), min_speed=float(sp[i]))
                    continue
                corners.append(CornerItem(index=i, distance=float(dist[i]), min_speed=float(sp[i])))

        # Optionally keep only N slowest corners
        if top_n is not None and top_n > 0:
            corners = sorted(corners, key=lambda c: c.min_speed)[:top_n]
            # Re-sort by distance for plotting order
            corners = sorted(corners, key=lambda c: c.distance)

        return {"driver": driver, "lap_number": int(sel["LapNumber"]), "corners": [c.model_dump() for c in corners]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get(
    "/analysis/sector-deltas/{year}/{event}/{session_type}",
    response_model=SectorDeltaResponse,
    tags=["Analysis"]
)
async def sector_deltas(
    year: int,
    event: str,
    session_type: str,
    driver1: str = Query(...),
    driver2: str = Query(...)
):
    """Sector and total time gaps between two drivers' fastest laps.

    Returns each driver's sector times and total lap time as strings plus gaps
    (s1_gap, s2_gap, s3_gap, total_gap). Use to prioritize deeper telemetry analysis.
    """
    try:
        session = get_session_cached(year, event, session_type)
        d1 = session.laps.pick_driver(driver1)
        d2 = session.laps.pick_driver(driver2)
        if d1 is None or d1.empty or d2 is None or d2.empty:
            raise HTTPException(status_code=404, detail="One or both drivers have no laps")
        l1 = d1.pick_fastest()
        l2 = d2.pick_fastest()

        def td_str(val):
            try:
                return str(val) if pd.notna(val) else None
            except Exception:
                return None

        def td_gap(a, b):
            try:
                if pd.notna(a) and pd.notna(b):
                    return str(a - b)
            except Exception:
                pass
            return None

        s1_times = {driver1: td_str(l1.get("Sector1Time")), driver2: td_str(l2.get("Sector1Time"))}
        s2_times = {driver1: td_str(l1.get("Sector2Time")), driver2: td_str(l2.get("Sector2Time"))}
        s3_times = {driver1: td_str(l1.get("Sector3Time")), driver2: td_str(l2.get("Sector3Time"))}
        total_times = {driver1: td_str(l1.get("LapTime")), driver2: td_str(l2.get("LapTime"))}

        gaps = {
            "s1_gap": td_gap(l1.get("Sector1Time"), l2.get("Sector1Time")),
            "s2_gap": td_gap(l1.get("Sector2Time"), l2.get("Sector2Time")),
            "s3_gap": td_gap(l1.get("Sector3Time"), l2.get("Sector3Time")),
            "total_gap": td_gap(l1.get("LapTime"), l2.get("LapTime"))
        }

        return {
            "driver1": driver1,
            "driver2": driver2,
            "s1_times": s1_times,
            "s2_times": s2_times,
            "s3_times": s3_times,
            "total_times": total_times,
            "gaps": gaps,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drivers/{year}", response_model=DriversResponse, tags=["Seasons"])
async def get_drivers(year: int):
    """Driver roster for a season.

    Returns each driver's abbreviation, full name, team, and country. Use the
    abbreviations (e.g., VER, HAM) as stable identifiers across endpoints.
    """
    try:
        # Get the event schedule for the year
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            raise ValueError(f"No events found for the year {year}")
        # print("schedule", schedule)
        # Get current date to compare it to the last race in the schedule
        now_utc = datetime.utcnow()
  
        # Find completed events
        completed_events = schedule[schedule['EventDate'] < now_utc]

        event_to_load = None
        if not completed_events.empty:
            event_to_load = completed_events.iloc[-1]
        else:
            event_to_load = schedule.iloc[0]
            # print(f"Season not yet started. Loading drivers from the first scheduled event {event_to_load['EventName']}")
        # Get the session object
        session = fastf1.get_session(year, event_to_load["EventName"], "R")

        # OPTIMIZED: Load only essential data. This makes the API call much faster.
        session.load(laps=False, telemetry=False, weather=False, messages=False)

        drivers_info: List[DriverInfo] = []

        for drv_number in session.drivers:
            # 'driver_data' is a Pandas Series with info for that driver
            driver_data = session.get_driver(drv_number)
           
            drivers_info.append(
                DriverInfo(
                    abbreviation=driver_data['Abbreviation'],
                    full_name=driver_data['FullName'],
                    team=driver_data['TeamName'],
                    country=driver_data['CountryCode'],
                )
            )

        return {"year": year, "drivers": [di.model_dump() for di in drivers_info], "count": len(drivers_info)}
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in get_drivers for year {year}: {e}")
        # Return a more appropriate 404 if data isn't found
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve driver data for {year}. Reason: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
