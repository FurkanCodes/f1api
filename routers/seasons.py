from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
import fastf1

from api.models import DriversResponse, DriverInfo, SeasonScheduleResponse
from api.concurrency import run_in_thread


router = APIRouter(tags=["Seasons"])


@router.get("/sessions/{year}", response_model=SeasonScheduleResponse)
async def get_season_schedule(year: int):
    try:
        schedule = await run_in_thread(fastf1.get_event_schedule, year)
        return {"year": year, "events": schedule.to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drivers/{year}", response_model=DriversResponse)
async def get_drivers(year: int):
    try:
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

        drivers_info: List[DriverInfo] = []
        for drv_number in session.drivers:
            di = session.get_driver(drv_number)
            drivers_info.append(
                DriverInfo(
                    abbreviation=di["Abbreviation"],
                    full_name=di["FullName"],
                    team=di["TeamName"],
                    country=di["CountryCode"],
                )
            )

        return {"year": year, "drivers": [d.model_dump() for d in drivers_info], "count": len(drivers_info)}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve driver data for {year}. Reason: {str(e)}",
        )
