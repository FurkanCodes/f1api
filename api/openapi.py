from fastapi.openapi.utils import get_openapi

TAGS_METADATA = [
    {"name": "Root", "description": "Service status & metadata."},
    {"name": "Seasons", "description": "Season schedules and driver rosters."},
    {"name": "Sessions", "description": "Session info, results, and lap data."},
    {"name": "Analysis", "description": "Comparisons and telemetry endpoints."},
]


def customize_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["tags"] = TAGS_METADATA
    schema["info"]["contact"] = {
        "name": "F1 Data Analysis",
    }
    schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
    app.openapi_schema = schema
    return app.openapi_schema

