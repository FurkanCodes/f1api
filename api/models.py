from typing import Optional, List, Dict, Any
from pydantic import BaseModel


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

