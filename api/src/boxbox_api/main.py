"""FastAPI application factory (``uvicorn boxbox_api.main:create_app --factory``)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from boxbox_api.routers import evaluation, meta, state, strategy
from boxbox_api.services import OutOfRangeError
from boxbox_api.settings import Settings
from boxbox_api.store import NotFoundError, Store, load_store


def create_app(settings: Settings | None = None, store: Store | None = None) -> FastAPI:
    """Build the app; artifacts are loaded once (or injected, for tests)."""
    resolved = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store or load_store(resolved.artifacts_dir, resolved.default_profile)
        yield

    app = FastAPI(title="BoxBox API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, error: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(OutOfRangeError)
    async def out_of_range(_: Request, error: OutOfRangeError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    for router in (meta.router, state.router, strategy.router, evaluation.router):
        app.include_router(router, prefix="/api")
    return app
