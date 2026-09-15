"""FastAPI application entry point.

Re-exports the production gateway (agent/server/app.py) so both
`uvicorn agent.server.app:app` and `uvicorn api.main:app` serve the same
application. This file used to be a bare /health stub, which made it easy to
accidentally serve an API without the agent, routes, or HITL governance.
"""

from agent.server.app import app

__all__ = ["app"]
