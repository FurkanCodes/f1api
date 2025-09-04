from fastapi import APIRouter
from api.services import clear_session_cache


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/cache/clear")
async def cache_clear():
    cleared = clear_session_cache()
    return {"cleared": cleared, "remaining": 0}

