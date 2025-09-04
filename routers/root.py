from fastapi import APIRouter

router = APIRouter(tags=["Root"])


@router.get("/")
async def root():
    return {"message": "F1 Data Analysis API", "version": "1.0.0"}


@router.get("/health")
async def health():
    return {"status": "ok"}

