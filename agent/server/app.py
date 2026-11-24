"""FastAPI application factory for the Financial Reasoning Agent production gateway."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.documents.service import DocumentService
from agent.graph import create_financial_agent
from agent.memory.working import get_checkpointer
from agent.nodes import hitl_nodes
from agent.server.auth import require_principal
from agent.server.limits import RequestSizeLimitMiddleware
from agent.server.routes import router as api_router
from agent.server.store import RecordStore
from agent.telemetry import init_telemetry
from agent.worker import Worker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes telemetry, durable checkpointer, and compiled LangGraph agent."""
    init_telemetry(is_eval_mode=False)
    if not hasattr(app.state, "agent"):
        app.state.agent = create_financial_agent(checkpointer=get_checkpointer())
    worker = Worker(app.state.store, app.state.agent, app.state.documents)
    app.state.worker = worker
    ledger = hitl_nodes.AUDIT_LEDGER
    app.state.audit_ready = (
        ledger.init_schema() if hasattr(ledger, "init_schema") else True
    )
    if os.getenv("FINAGENT_EMBEDDED_WORKER", "1") == "1":
        worker.start(int(os.getenv("WORKER_CONCURRENCY", "2")))
    try:
        yield
    finally:
        worker.stop()


def create_app() -> FastAPI:
    """Builds and configures the production ASGI FastAPI gateway."""
    app = FastAPI(
        title="Autonomous Financial Reasoning Gateway",
        description="Production ASGI server for multi-modal SEC financial analysis with LangGraph, SSE streaming, and HITL governance.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.store = RecordStore()
    app.state.documents = DocumentService(app.state.store)

    allowed_origins = os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Trace-Id"],
    )
    app.add_middleware(RequestSizeLimitMiddleware)

    @app.middleware("http")
    async def request_tracing_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = request.headers.get("X-Trace-Id", f"trace_{uuid.uuid4().hex[:12]}")
        start_time = time.perf_counter()

        request.state.trace_id = trace_id
        public_paths = {
            "/api/v1/config",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/api/v1/health",
        }
        if (
            request.method != "OPTIONS"
            and request.url.path.startswith("/api/v1/")
            and request.url.path not in public_paths
        ):
            import asyncio

            try:
                await asyncio.to_thread(require_principal, request)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=exc.headers,
                )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                limit = (
                    int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
                    + 512 * 1024
                    if request.url.path == "/api/v1/documents"
                    else 1024 * 1024
                )
                if int(content_length) > limit:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "The request exceeds the size limit."},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "Invalid content length."}
                )
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "unknown")
        logger.exception("Unhandled gateway error (trace_id=%s)", trace_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "The request could not be completed.",
                "trace_id": trace_id,
            },
        )

    app.include_router(api_router)

    @app.get("/ready")
    async def readiness() -> JSONResponse:
        import asyncio

        store_ready = await asyncio.to_thread(app.state.store.ping)
        auth_ready = (
            bool(os.getenv("FINAGENT_API_TOKEN"))
            if os.getenv(
                "FINAGENT_AUTH_MODE",
                "supabase" if os.getenv("SUPABASE_URL") else "token",
            )
            == "token"
            else bool(
                os.getenv("SUPABASE_URL")
                and (
                    os.getenv("SUPABASE_PUBLISHABLE_KEY")
                    or os.getenv("SUPABASE_ANON_KEY")
                )
            )
        )
        audit_ready = bool(getattr(app.state, "audit_ready", False))
        ready = (
            store_ready and auth_ready and audit_ready and hasattr(app.state, "agent")
        )
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "checks": {
                    "storage": store_ready,
                    "authentication": auth_ready,
                    "audit": audit_ready,
                    "agent": hasattr(app.state, "agent"),
                },
            },
        )

    @app.get("/health", include_in_schema=False)
    @app.get("/healthz", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    async def root_health() -> dict[str, str]:
        return {"status": "healthy", "service": "finagent-production-gateway"}

    return app


app = create_app()
