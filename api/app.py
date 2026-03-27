import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.cache import get_master_data
from api.routes import all_routers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload data on startup for faster first request."""
    logger.info("Preloading market data on startup...")
    try:
        get_master_data()
        logger.info("Data preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload data: {e}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Portfolio Optimization API",
        description="Portfolio optimization with MPT, factor-based, and ML strategies",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in all_routers:
        app.include_router(router)

    return app
