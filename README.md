# f1api

API for F1 season, session, lap, telemetry, and analysis data built on FastAPI + FastF1. It’s designed for charts, dashboards, notebooks, or any client that consumes clean JSON.

## Overview

- Purpose: Serve F1 season schedules, drivers, session metadata, results, lap times, telemetry, and analysis in a consistent HTTP API.
- Data source: Uses FastF1 to load timing data. First request for a session downloads and caches on disk; subsequent requests are much faster.
- Consumers: Frontend apps, scripts, notebooks—anything that calls HTTP/JSON.

## Quick Start

- Install: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload`
- Explore: Open `/docs` (Swagger UI) or `/redoc` in a browser.



## Configuration

- `FASTF1_CACHE_DIR`: Directory for FastF1 disk cache (default `./cache`).
- `SESSION_CACHE_MAX`: In‑memory session cache capacity (default `3`).
- `ENABLE_GZIP`: Set `1` to enable response compression (disabled by default).
- `ALLOWED_ORIGINS`: Comma-separated CORS origins (default `*`).
- `ALLOW_CREDENTIALS`: `1`/`0` to allow credentials with CORS (default `0`).

## Core Concepts

- Year: Four‑digit season (e.g., `2024`).
- Event: The exact `EventName` from the year’s schedule (see `/sessions/{year}`).
- Session Type: Common values include `R` (Race), `Q` (Qualifying), `FP1`, `FP2`, `FP3`, `S` (Sprint), `SS` (Sprint Shootout). Use what appears for the event.
- Driver Abbreviation: Three‑letter code (e.g., `VER`, `HAM`). See `/drivers/{year}`.
- Data availability: Some channels/fields (weather/messages) may be missing depending on session and data source.

## Response Conventions

- JSON safe: Pandas/NumPy types are normalized (e.g., numbers to float/int, timedeltas to strings, NaN → `null`).
- Optional fields: Some response fields are optional and may be `null` or omitted if not available.
- Pydantic v2: Models serialize with `model_dump()` internally.

## Endpoints Overview

- Seasons
  - `GET /sessions/{year}` — Season schedule.
  - `GET /drivers/{year}` — Drivers and teams for the season.
- Sessions
  - `GET /session/{year}/{event}/{session_type}` — Session metadata (name, date, weather, track status when available).
  - `GET /results/{year}/{event}/{session_type}` — Results/standings for the session.
  - `GET /laptimes/{year}/{event}/{session_type}` — Rich per‑lap data; optional filters.
- Telemetry
  - `GET /telemetry/{year}/{event}/{session_type}` — Per‑lap telemetry with resampling/downsampling, field selection, and summaries.
- Comparisons
  - `GET /compare/{year}/{event}/{session_type}` — Fastest/average comparison; optional overlay telemetry.
- Analysis
  - `GET /analysis/braking/{year}/{event}/{session_type}` — Braking zones per lap.
  - `GET /analysis/corners/{year}/{event}/{session_type}` — Corner minimum speeds via local minima.
  - `GET /analysis/sector-deltas/{year}/{event}/{session_type}` — Sector and total time gaps between two drivers.
  - Admin
    - `POST /admin/cache/clear` — Clear in-memory session cache.

## Endpoint Guide (What and Why)

- Root
  - `/`: Service banner with version. Why: quick sanity check and discoverability.
  - `/health`: Health probe. Why: readiness/liveness checks for ops/monitoring.

- Seasons
  - `GET /sessions/{year}`: Season schedule (rounds, `EventName`, dates, format). Why: discovery of canonical `EventName` for other routes.
  - `GET /drivers/{year}`: Driver roster (abbreviation, full name, team, country). Why: source of stable driver codes like `VER`, `HAM`.

- Sessions
  - `GET /session/{year}/{event}/{session_type}`: Session context (name, date, optional weather, track status). Why: adds conditions/timing context for analysis and UI.
  - `GET /results/{year}/{event}/{session_type}`: Results/standings, JSON‑normalized. Why: canonical outcome for tables and leaderboards.
  - `GET /laptimes/{year}/{event}/{session_type}`: Per‑lap dataset with rich fields (lap time, sectors, compound/life, flags). Filters: `drivers`, `exclude_pit`, `exclude_invalid`. Why: core feed for pace, stint, degradation analysis.

- Telemetry
  - `GET /telemetry/{year}/{event}/{session_type}`: Driver lap telemetry arrays with resampling (`distance_step` or `time_step_ms`), downsampling (`sample_every`, `max_points`), `fields` selection, and `summary_only`. Why: powers charts and overlays with compact payloads and quick summaries.

- Comparisons
  - `GET /compare/{year}/{event}/{session_type}`: Compare `driver1` vs `driver2` on `fastest` or `average` laps; optional overlay telemetry (`include_telemetry`, `distance_step`, `fields`, `sample_every`, `max_points`). Filters (`exclude_pit`, `exclude_invalid`). Why: one‑call comparison for dashboards with headline gaps and traces.

- Analysis
  - `GET /analysis/braking/{year}/{event}/{session_type}`: Braking zones with length, duration, entry/min/exit speed, avg brake, avg decel. Params: `driver`, optional `lap`, `brake_threshold`, `min_duration_ms`, `min_length_m`. Why: turns raw telemetry into actionable braking features.
  - `GET /analysis/corners/{year}/{event}/{session_type}`: Corner apex proxies via local speed minima (distance, min_speed). Params: `driver`, optional `lap`, `min_gap_m`, `window_points`, `top_n`. Why: quick corner benchmarking without complex track segmentation.
  - `GET /analysis/sector-deltas/{year}/{event}/{session_type}`: Sector times and total lap times + per‑sector/total gaps for two drivers. Params: `driver1`, `driver2`. Why: fast insight into where time is gained/lost.

- Admin
  - `POST /admin/cache/clear`: Clears in‑memory session cache; returns number cleared. Why: operational lever to reclaim memory/reset state in dev or constrained hosting. Protect behind trusted network in production.

### Design Notes
- Discovery first: Seasons + Drivers provide canonical inputs for all other routes.
- Analysis‑ready: JSON normalization, filtering, resampling/downsampling, and field selection for practical dashboards and plots.
- Case‑insensitive inputs: `event` and driver codes are normalized to reduce friction.
- Operational control: Simple admin cache clear for quick resets.

## Seasons

### GET `/sessions/{year}`
- Returns the schedule; use `EventName` for other endpoints.
- Useful fields: round, `EventName`, dates, `EventFormat`.

### GET `/drivers/{year}`
- Returns `{ year, drivers, count }` where each driver has `abbreviation`, `full_name`, `team`, `country`.

## Sessions

### GET `/session/{year}/{event}/{session_type}`
- Returns: `session_name`, `date`, `track_status` (if available), `weather` (if available).

### GET `/results/{year}/{event}/{session_type}`
- Returns: `session_info` and `results` with JSON‑normalized values.

### GET `/laptimes/{year}/{event}/{session_type}`
- Query params:
  - `drivers`: Repeat to filter (e.g., `drivers=VER&drivers=HAM`).
  - `exclude_pit`: Exclude pit in/out laps when columns exist.
  - `exclude_invalid`: Exclude deleted or inaccurate laps when columns exist.
- Returns: Array with driver, lap number, lap time, lap time seconds, sector times, compound, tyre life, team, personal best flags, etc.

## Telemetry

### GET `/telemetry/{year}/{event}/{session_type}`
- Required:
  - `driver`: Three‑letter code (e.g., `VER`).
- Optional:
  - Lap choice: `lap` for a specific lap; otherwise uses fastest lap.
  - Resampling:
    - `distance_step`: Resample to a fixed distance grid (meters). Best for overlaying traces across drivers.
    - `time_step_ms`: Resample to a fixed time grid (milliseconds).
  - Downsampling:
    - `sample_every`: Keep every Nth row.
    - `max_points`: Uniformly reduce to about this many points.
  - Field selection:
    - `fields`: Repeatable; any of `distance,speed,throttle,brake,rpm,gear,drs`.
  - Summary only:
    - `summary_only=true`: Return only summary metrics (no arrays).
- Response:
  - Always includes: `driver`, `lap_number`, `lap_time`, `compound`.
  - `telemetry` (optional): Only requested series.
  - `summary` (optional): `max/avg/min speed`, `avg throttle`, `brake usage %`, `max rpm`, `avg gear`, `DRS open %`, `total distance`.

Telemetry tips
- Downsample by stride: `sample_every=10`
- Target point count: `max_points=2000`
- Resample by distance: `distance_step=5.0` (meters)
- Resample by time: `time_step_ms=50`
- Select fields: `fields=speed&fields=throttle` (omit others)
- Summary only: `summary_only=true` (metrics without arrays)

## Comparisons

### GET `/compare/{year}/{event}/{session_type}`
- Required:
  - `driver1`, `driver2`.
- Optional:
  - `lap_type`: `fastest` or `average`.
  - `include_telemetry`: Adds telemetry arrays under each driver.
  - `distance_step`: Resample to same distance grid for overlays.
  - `sample_every`, `max_points`, `fields`: Same semantics as telemetry endpoint.
  - `exclude_pit`, `exclude_invalid`: Filter laps before computing fastest/average.
- Response:
  - `fastest`: Each driver’s lap time, compound, max/avg speed; overall `gap`.
  - `average`: Each driver’s average lap time, total laps; `average_gap`.
  - `include_telemetry=true`: Adds `telemetry` per driver (field‑selectable).

Compare endpoint tips
- Include overlay telemetry: `include_telemetry=true`
- Use a common distance grid: add `distance_step=5.0`
- Also supports `fields`, `sample_every`, `max_points`
- Lap filters: `exclude_pit=true&exclude_invalid=true`

## Analysis

### GET `/analysis/braking/{year}/{event}/{session_type}`
- Params: `driver`, optional `lap` (else fastest), `brake_threshold` (0–1 normalized), `min_duration_ms`, `min_length_m`.
- Output: List of zones with `start_distance`, `end_distance`, `length_m`, `duration_ms`, `entry/min/exit speed`, `avg_brake`, `avg_decel_mps2`.
- Note: Brake input is normalized if source is 0–100; thresholds apply after normalization.

### GET `/analysis/corners/{year}/{event}/{session_type}`
- Params: `driver`, optional `lap` (else fastest), `min_gap_m` (spacing), `window_points` (local minima window), `top_n`.
- Output: Detected corners (local speed minima) with `distance` and `min_speed`.
- Note: Heuristic detection; tune `min_gap_m` and `window_points` per track.

### GET `/analysis/sector-deltas/{year}/{event}/{session_type}`
- Params: `driver1`, `driver2`.
- Output: S1/S2/S3 times and total lap times for both drivers, plus `s1_gap`, `s2_gap`, `s3_gap`, `total_gap` as strings.

Analysis endpoint tips
- Braking zones:
  - `GET /analysis/braking/{year}/{event}/{session_type}?driver=VER&brake_threshold=0.1&min_duration_ms=200&min_length_m=5`
- Corners:
  - `GET /analysis/corners/{year}/{event}/{session_type}?driver=HAM&top_n=10&min_gap_m=40`
- Sector deltas:
  - `GET /analysis/sector-deltas/{year}/{event}/{session_type}?driver1=VER&driver2=HAM`

## Input Strategy

- Find events: `GET /sessions/{year}`. Event matching is case-insensitive and whitespace-normalized, but using the exact `EventName` is still recommended.
- Find drivers: `GET /drivers/{year}`. Use the `abbreviation` (e.g., `VER`). Driver inputs are case-insensitive.
- Session types: Use the types shown for the event (e.g., Race = `R`).

## Performance & Caching

- Disk cache (FastF1): Controlled by `FASTF1_CACHE_DIR` (default `./cache`). First load is slower; later loads are fast.
- In‑memory cache: Recently loaded sessions are retained up to `SESSION_CACHE_MAX`.
- Shrink payloads: Use `fields`, `summary_only`, `distance_step`, `sample_every`, `max_points`.
- Compression: Set `ENABLE_GZIP=1` to compress large responses if your environment supports it.

## Error Handling & Troubleshooting

- 400 Bad Request:
  - Malformed inputs, unsupported event/session type, or telemetry channel not available.
- 404 Not Found:
  - No laps for a driver, lap number not found, or driver not present in that session.
- Event name mismatch:
  - Use `GET /sessions/{year}` and copy `EventName`. Naming varies across seasons.
- Large responses:
  - Use field selection and downsampling/resampling. Prefer `summary_only=true` when exploring.
- GZip errors:
  - If `ENABLE_GZIP=1` causes issues, disable it and rely on field selection/downsampling.

## Practical Examples

- Season schedule (2024):
  - `/sessions/2024`
- Drivers list (2025):
  - `/drivers/2025`
- Session info (Race, Monaco 2024):
  - `/session/2024/Monaco%20Grand%20Prix/R`
- Results (same):
  - `/results/2024/Monaco%20Grand%20Prix/R`
- Lap times for VER and HAM excluding pit/invalid:
  - `/laptimes/2024/Monaco%20Grand%20Prix/R?drivers=VER&drivers=HAM&exclude_pit=true&exclude_invalid=true`
- Telemetry (VER fastest lap, distance grid, selected fields):
  - `/telemetry/2024/Monaco%20Grand%20Prix/R?driver=VER&distance_step=5&fields=speed&fields=throttle&max_points=1500`
- Summary‑only telemetry:
  - `/telemetry/2024/Monaco%20Grand%20Prix/R?driver=VER&summary_only=true`
- Compare VER vs HAM with overlay telemetry:
  - `/compare/2024/Monaco%20Grand%20Prix/R?driver1=VER&driver2=HAM&include_telemetry=true&distance_step=5&fields=speed&max_points=1000`
- Braking zones (threshold/duration/length tuned):
  - `/analysis/braking/2024/Monaco%20Grand%20Prix/R?driver=VER&brake_threshold=0.1&min_duration_ms=200&min_length_m=5`
- Corner minimums (top 10 slowest corners, spaced ≥40 m):
  - `/analysis/corners/2024/Monaco%20Grand%20Prix/R?driver=HAM&top_n=10&min_gap_m=40`
- Sector deltas (VER vs HAM):
  - `/analysis/sector-deltas/2024/Monaco%20Grand%20Prix/R?driver1=VER&driver2=HAM`

