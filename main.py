import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

import fastf1

import pandas as pd
import numpy as np

from api.config import get_settings
from api.logging_config import configure_logging
from api.services import get_session_cached
from api.openapi import customize_openapi
from api.models import (
    SeasonScheduleResponse,
    SessionBasicResponse,
    SessionResultsResponse,
    TelemetryResponse,
    LapTimesResponse,
    CompareResponse,
    BrakingResponse,
    CornersResponse,
    SectorDeltaResponse,
    DriversResponse,
    LapTimeItem,
    CompareAverage,
    CompareFastest,
    DriverInfo,
    BrakingZone,
    CornerItem,
)
from routers.root import router as root_router
from routers.seasons import router as seasons_router
from routers.sessions import router as sessions_router
from routers.analysis import router as analysis_router
from routers.admin import router as admin_router

settings = get_settings()
logger = configure_logging()

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
    allow_origins=list(settings.allowed_origins) if isinstance(settings.allowed_origins, (list, tuple)) else [str(settings.allowed_origins)],
    allow_credentials=settings.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally compress large responses (telemetry/laps)
if settings.enable_gzip:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

app.openapi = lambda: customize_openapi(app)  # type: ignore

# Gradually include modular routers (Root, Seasons)
app.include_router(root_router)
app.include_router(seasons_router)
app.include_router(sessions_router)
app.include_router(analysis_router)
app.include_router(admin_router)

# ---------- Routes ----------

# Root endpoint now served by routers/root.py

# Seasons schedule endpoint now served by routers/seasons.py

# Sessions endpoints are now served by routers/sessions.py

# Sessions endpoints are now served by routers/sessions.py

# Analysis endpoints are now served by routers/analysis.py

# Sessions endpoints are now served by routers/sessions.py

# Analysis endpoints are now served by routers/analysis.py

# Analysis endpoints are now served by routers/analysis.py

# Analysis endpoints are now served by routers/analysis.py

# Analysis endpoints are now served by routers/analysis.py

# Drivers endpoint now served by routers/seasons.py
# Health endpoint now served by routers/root.py
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
