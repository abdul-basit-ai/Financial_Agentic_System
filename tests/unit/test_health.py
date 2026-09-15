from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """api.main re-exports the production gateway; /health (and /healthz)
    serve the gateway liveness payload that K8s probes depend on."""
    for path in ("/health", "/healthz"):
        response = client.get(path)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["service"] == "finagent-production-gateway"


def test_api_main_is_gateway():
    """api.main:app must be the real gateway, not a bare stub — the stub made
    it easy to serve an API without agent routes or HITL governance."""
    from agent.server.app import app as gateway_app

    assert app is gateway_app
