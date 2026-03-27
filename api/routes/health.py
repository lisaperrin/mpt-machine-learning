from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "portfolio-optimization-api", "version": "2.1.0"}
