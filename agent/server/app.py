"""FastAPI application factory for the Financial Reasoning Agent production gateway."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.graph import create_financial_agent
from agent.memory.working import get_checkpointer
from agent.server.routes import router as api_router
from agent.telemetry import init_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes telemetry, durable checkpointer, and compiled LangGraph agent."""
    init_telemetry(is_eval_mode=False)
    checkpointer = get_checkpointer()
    app.state.agent = create_financial_agent(checkpointer=checkpointer)
    yield
    # Clean shutdown
    if hasattr(app.state, "agent"):
        del app.state.agent


def create_app() -> FastAPI:
    """Builds and configures the production ASGI FastAPI gateway."""
    app = FastAPI(
        title="Autonomous Financial Reasoning Gateway",
        description="Production ASGI server for multi-modal SEC financial analysis with LangGraph, SSE streaming, and HITL governance.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 1. CORS Configuration
    # Explicit origin allowlist (Streamlit UI). allow_credentials=True with
    # wildcard origins is an invalid combination browsers reject, and wildcard
    # CORS is unsafe for a gateway that will carry auth tokens (Phase 13).
    allowed_origins = os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Trace-Id"],
    )

    # 2. Distributed Tracing & Request ID Middleware
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
        trace_id = request.headers.get("X-Trace-Id", f"trace_{uuid.uuid4().hex[:12]}")
        start_time = time.perf_counter()

        request.state.trace_id = trace_id
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    # 3. Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": str(exc),
                "trace_id": getattr(request.state, "trace_id", "unknown"),
            },
        )

    # 4. Attach API Routers
    app.include_router(api_router)

    return app


app = create_app()