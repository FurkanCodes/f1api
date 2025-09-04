from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException
import numpy as np
import pandas as pd

from api.models import (
    TelemetryResponse,
    CompareResponse,
    BrakingResponse,
    CornersResponse,
    SectorDeltaResponse,
    CompareFastest,
    CompareAverage,
)
from api.services import get_session_cached, resolve_event_name
from api.concurrency import run_in_thread
from api.utils import validate_fields


router = APIRouter(tags=["Analysis"])


@router.get("/telemetry/{year}/{event}/{session_type}", response_model=TelemetryResponse)
async def get_driver_telemetry(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(..., description="Driver abbreviation (e.g., VER, HAM)"),
    lap: Optional[int] = Query(None, description="Specific lap number; fastest lap if not provided"),
    sample_every: Optional[int] = Query(None, ge=1, description="Keep every Nth telemetry row to reduce payload size"),
    max_points: Optional[int] = Query(None, ge=10, le=20000, description="Downsample to approximately this many points"),
    distance_step: Optional[float] = Query(None, gt=0, description="Resample to fixed distance step (meters); takes precedence over time_step_ms"),
    time_step_ms: Optional[int] = Query(None, gt=0, description="Resample to fixed time step (milliseconds)"),
    fields: Optional[List[str]] = Query(None, description="Which series to return: distance,speed,throttle,brake,rpm,gear,drs"),
    summary_only: bool = Query(False, description="Return only summary metrics; omit arrays"),
):
    try:
        driver = driver.upper()
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)

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
                    telemetry = pd.DataFrame(
                        {
                            "Distance": new_dist,
                            "Speed": speed,
                            "Throttle": throttle,
                            "Brake": brake,
                            "RPM": rpm,
                            "nGear": gear,
                            "DRS": drs,
                        }
                    )
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
                    telemetry = pd.DataFrame(
                        {
                            "Distance": distance,
                            "Speed": speed,
                            "Throttle": throttle,
                            "Brake": brake,
                            "RPM": rpm,
                            "nGear": gear,
                            "DRS": drs,
                        }
                    )
        except Exception:
            pass
        if sample_every is not None and sample_every > 1:
            telemetry = telemetry.iloc[::sample_every]
        if max_points is not None and max_points > 0 and len(telemetry) > max_points:
            idx = np.linspace(0, len(telemetry) - 1, num=max_points).round().astype(int)
            telemetry = telemetry.iloc[idx]

        allowed = {"distance", "speed", "throttle", "brake", "rpm", "gear", "drs"}
        selected_fields = validate_fields(fields, allowed)
        payload: Dict[str, Any] = {
            "driver": driver,
            "lap_number": int(selected_lap["LapNumber"]),
            "lap_time": str(selected_lap.get("LapTime")) if pd.notna(selected_lap.get("LapTime")) else None,
            "compound": selected_lap.get("Compound"),
        }

        def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
            try:
                max_speed = float(df["Speed"].max())
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


@router.get("/compare/{year}/{event}/{session_type}", response_model=CompareResponse)
async def compare_drivers(
    year: int,
    event: str,
    session_type: str,
    driver1: str = Query(..., description="First driver abbreviation"),
    driver2: str = Query(..., description="Second driver abbreviation"),
    lap_type: str = Query("fastest", description="'fastest' or 'average'", pattern="^(fastest|average)$"),
    include_telemetry: bool = Query(False, description="Include telemetry traces for overlays"),
    fields: Optional[List[str]] = Query(None, description="Telemetry fields to include"),
    distance_step: Optional[float] = Query(None, gt=0, description="Resample to fixed distance step (meters)"),
    sample_every: Optional[int] = Query(None, ge=1),
    max_points: Optional[int] = Query(None, ge=10, le=20000),
):
    try:
        driver1 = driver1.upper()
        driver2 = driver2.upper()
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
        driver1_laps = session.laps.pick_driver(driver1)
        driver2_laps = session.laps.pick_driver(driver2)

        if driver1_laps is None or driver1_laps.empty or driver2_laps is None or driver2_laps.empty:
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
                sel_fields = validate_fields(fields, {"distance", "speed", "throttle", "brake", "rpm", "gear", "drs"})
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
                        idx_float = np.interp(new_d, df["Distance"].astype(float).to_numpy(), np.arange(len(df)))
                        nn_idx = np.round(idx_float).astype(int).clip(0, len(df) - 1)
                        if "nGear" in df.columns:
                            base["nGear"] = df["nGear"].to_numpy()[nn_idx].astype(int)
                        if "DRS" in df.columns:
                            base["DRS"] = df["DRS"].to_numpy()[nn_idx].astype(int)
                        return pd.DataFrame(base)

                    tel1 = resample(tel1, new_dist)
                    tel2 = resample(tel2, new_dist)
                if sample_every is not None and sample_every > 1:
                    tel1 = tel1.iloc[::sample_every]
                    tel2 = tel2.iloc[::sample_every]
                if max_points is not None and max_points > 0:
                    def uniform_down(df):
                        if len(df) <= max_points:
                            return df
                        idx = np.linspace(0, len(df) - 1, num=max_points).round().astype(int)
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
                "driver1": CompareAverage(driver=driver1, average_lap_time=str(avg1) if avg1 is not None else None, total_laps=len(driver1_laps)).model_dump(),
                "driver2": CompareAverage(driver=driver2, average_lap_time=str(avg2) if avg2 is not None else None, total_laps=len(driver2_laps)).model_dump(),
                "average_gap": str(abs(avg1 - avg2)) if (avg1 is not None and avg2 is not None) else None,
            }

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/braking/{year}/{event}/{session_type}", response_model=BrakingResponse)
async def braking_zones(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(..., description="Driver abbreviation"),
    lap: Optional[int] = Query(None, description="Specific lap; fastest if omitted"),
    brake_threshold: float = Query(0.1, ge=0, description="Brake threshold (0-1 or percent if raw telemetry is 0-100)"),
    min_duration_ms: int = Query(200, ge=0, description="Minimum zone duration in ms"),
    min_length_m: float = Query(5.0, ge=0, description="Minimum zone length in meters"),
):
    try:
        driver = driver.upper()
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
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
        if "Brake" not in tel.columns:
            raise HTTPException(status_code=400, detail="Brake channel not available")
        brake = tel["Brake"].astype(float).to_numpy()
        scale = 100.0 if np.nanmax(brake) > 1.5 else 1.0
        br_norm = brake / scale
        mask = br_norm > brake_threshold

        idx = np.arange(len(mask))
        change = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0] - 1

        zones = []
        dist = tel["Distance"].astype(float).to_numpy() if "Distance" in tel.columns else None
        speed_kmh = tel["Speed"].astype(float).to_numpy() if "Speed" in tel.columns else None
        time_s = tel["Time"].dt.total_seconds().astype(float).to_numpy() if "Time" in tel.columns else None

        for s, e in zip(starts, ends):
            if s >= len(mask) or e >= len(mask) or e <= s:
                continue
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
            min_sp = float(np.nanmin(speed_kmh[s:e + 1])) if speed_kmh is not None else None
            exit_sp = float(speed_kmh[e]) if speed_kmh is not None else None
            avg_br = float(np.nanmean(br_norm[s:e + 1])) if (e > s) else float(br_norm[s])
            avg_decel = None
            if entry_sp is not None and min_sp is not None and dur_ms and dur_ms > 0:
                entry_mps = entry_sp / 3.6
                min_mps = min_sp / 3.6
                avg_decel = (entry_mps - min_mps) / (dur_ms / 1000.0)

            zones.append(
                {
                    "start_distance": float(dist[s]) if dist is not None else 0.0,
                    "end_distance": float(dist[e]) if dist is not None else 0.0,
                    "length_m": length_m or 0.0,
                    "duration_ms": dur_ms or 0.0,
                    "entry_speed": entry_sp,
                    "min_speed": min_sp,
                    "exit_speed": exit_sp,
                    "avg_decel_mps2": avg_decel,
                    "avg_brake": avg_br,
                }
            )

        return {"driver": driver, "lap_number": int(sel["LapNumber"]), "zones": zones}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/corners/{year}/{event}/{session_type}", response_model=CornersResponse)
async def corner_minimums(
    year: int,
    event: str,
    session_type: str,
    driver: str = Query(...),
    lap: Optional[int] = Query(None),
    min_gap_m: float = Query(30.0, ge=0),
    window_points: int = Query(9, ge=3),
    top_n: Optional[int] = Query(None, ge=1),
):
    try:
        driver = driver.upper()
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
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

        sp = tel["Speed"].astype(float).to_numpy()
        dist = tel["Distance"].astype(float).to_numpy()
        n = len(sp)
        corners = []
        for i in range(1, n - 1):
            lo = max(0, i - window_points // 2)
            hi = min(n, i + window_points // 2 + 1)
            if sp[i] == np.nanmin(sp[lo:hi]) and sp[i] < sp[i - 1] and sp[i] < sp[i + 1]:
                if corners and (dist[i] - corners[-1]["distance"]) < min_gap_m:
                    if sp[i] < corners[-1]["min_speed"]:
                        corners[-1] = {"index": i, "distance": float(dist[i]), "min_speed": float(sp[i])}
                    continue
                corners.append({"index": i, "distance": float(dist[i]), "min_speed": float(sp[i])})

        if top_n is not None and top_n > 0:
            corners = sorted(corners, key=lambda c: c["min_speed"])[:top_n]
            corners = sorted(corners, key=lambda c: c["distance"])

        return {"driver": driver, "lap_number": int(sel["LapNumber"]), "corners": corners}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/sector-deltas/{year}/{event}/{session_type}", response_model=SectorDeltaResponse)
async def sector_deltas(
    year: int,
    event: str,
    session_type: str,
    driver1: str = Query(...),
    driver2: str = Query(...),
):
    try:
        driver1 = driver1.upper()
        driver2 = driver2.upper()
        canonical_event = await run_in_thread(resolve_event_name, year, event)
        session = await run_in_thread(get_session_cached, year, canonical_event, session_type)
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
            "total_gap": td_gap(l1.get("LapTime"), l2.get("LapTime")),
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
