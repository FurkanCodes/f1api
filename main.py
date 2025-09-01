from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

import fastf1
import fastf1.plotting

import pandas as pd
import numpy as np

# Enable FastF1 plotting and cache
fastf1.plotting.setup_mpl()
fastf1.Cache.enable_cache("./cache")

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
    distance: List[float]
    speed: List[float]
    throttle: List[float]
    brake: List[float]
    rpm: List[float]
    gear: List[int]
    drs: List[int]

class TelemetryResponse(BaseModel):
    driver: str
    lap_number: int
    lap_time: Optional[str]
    compound: Optional[str]
    telemetry: TelemetryPayload

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
    return {"message": "F1 Data Analysis API", "version": "1.0.0"}

@app.get("/sessions/{year}", response_model=SeasonScheduleResponse, tags=["Seasons"])
async def get_season_schedule(year: int):
    """Get all sessions for a given year"""
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
    """Get basic F1 session information"""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

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
    """Get session results and standings"""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

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
    lap: Optional[int] = Query(None, description="Specific lap number; fastest lap if not provided", examples={"example": {"value": 44}})
):
    """Get telemetry data for a specific driver and lap"""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

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

        return {
            "driver": driver,
            "lap_number": int(selected_lap["LapNumber"]),
            "lap_time": str(selected_lap.get("LapTime")) if pd.notna(selected_lap.get("LapTime")) else None,
            "compound": selected_lap.get("Compound"),
            "telemetry": {
                "distance": telemetry["Distance"].astype(float).tolist(),
                "speed": telemetry["Speed"].astype(float).tolist(),
                "throttle": telemetry["Throttle"].astype(float).tolist(),
                "brake": telemetry["Brake"].astype(float).tolist(),
                "rpm": telemetry["RPM"].astype(float).tolist(),
                "gear": telemetry["nGear"].astype(int).tolist(),
                "drs": telemetry["DRS"].astype(int).tolist(),
            },
        }
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
    drivers: Optional[List[str]] = Query(None, description="Driver abbreviations, e.g. drivers=VER&drivers=HAM")
):
    """Get lap times analysis for drivers"""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

        laps = session.laps
        if drivers:
            laps = laps[laps["Driver"].isin(drivers)]

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
    lap_type: str = Query("fastest", description="'fastest' or 'average'", pattern="^(fastest|average)$")
):
    """Compare two drivers' performance"""
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

        driver1_laps = session.laps.pick_driver(driver1)
        driver2_laps = session.laps.pick_driver(driver2)
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
                ).dict(),
                "driver2": CompareFastest(
                    driver=driver2,
                    lap_time=str(lap2.get("LapTime")),
                    compound=lap2.get("Compound"),
                    max_speed=float(tel2["Speed"].max()) if not tel2.empty else None,
                    avg_speed=float(tel2["Speed"].mean()) if not tel2.empty else None,
                ).dict(),
                "gap": str(abs(lap1.get("LapTime") - lap2.get("LapTime"))) if pd.notna(lap1.get("LapTime")) and pd.notna(lap2.get("LapTime")) else None,
            }
        else:
            avg1 = driver1_laps["LapTime"].mean() if not driver1_laps.empty else None
            avg2 = driver2_laps["LapTime"].mean() if not driver2_laps.empty else None

            comparison = {
                "driver1": CompareAverage(
                    driver=driver1,
                    average_lap_time=str(avg1) if avg1 is not None else None,
                    total_laps=len(driver1_laps),
                ).dict(),
                "driver2": CompareAverage(
                    driver=driver2,
                    average_lap_time=str(avg2) if avg2 is not None else None,
                    total_laps=len(driver2_laps),
                ).dict(),
                "average_gap": str(abs(avg1 - avg2)) if (avg1 is not None and avg2 is not None) else None,
            }

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drivers/{year}", response_model=DriversResponse, tags=["Seasons"])
async def get_drivers(year: int):
    """Get all drivers for a given yearr"""
    try:
        # Get a race session to extract driver info
        schedule = fastf1.get_event_schedule(year)
        # Prefer a race event; fall back to first event if formats differ
        race_rows = schedule[schedule["EventFormat"].isin(["conventional", "sprint", "sprint_shootout"])]
        first_event = (race_rows.iloc[0] if not race_rows.empty else schedule.iloc[0])

        session = fastf1.get_session(year, first_event["EventName"], "R")
        session.load()

        drivers_info: List[DriverInfo] = []
        for drv in session.drivers:
            d = session.get_driver(drv)
            drivers_info.append(
                DriverInfo(
                    abbreviation=str(drv),
                    full_name=d.get("FullName"),
                    team=d.get("TeamName"),
                    country=d.get("CountryCode"),
                )
            )

        return {"year": year, "drivers": [di.dict() for di in drivers_info]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
