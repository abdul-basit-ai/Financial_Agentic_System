from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agent.documents.storage import FileStorage
from agent.server import auth

LOCAL = {"Authorization": "Bearer test-worker-token"}
OWNER_A = "1da6eb67-c690-44b8-ab26-2da5faccd9f0"
OWNER_B = "2f99110b-6601-4382-9703-7fc814813d01"


def identity_service(monkeypatch):
    monkeypatch.setenv("FINAGENT_AUTH_MODE", "supabase")

    def verify(method, path, headers):
        assert (method, path) == ("GET", "/auth/v1/user")
        owner = {"Bearer token-a": OWNER_A, "Bearer token-b": OWNER_B}.get(
            headers["Authorization"]
        )
        return SimpleNamespace(
            status_code=200 if owner else 401, json=lambda: {"id": owner}
        )

    monkeypatch.setattr(auth, "supabase_request", verify)


@pytest.mark.parametrize(
    "header",
    [
        {},
        {"Authorization": "Bearer invalid-fixture-token"},
        {"Authorization": "Basic unrelated"},
    ],
)
def test_private_resources_require_verified_bearer_credentials(api_app, header):
    with TestClient(api_app) as client:
        for path in (
            "documents",
            "approvals",
            "threads",
            "threads/sample/state",
            "jobs/sample",
        ):
            assert client.get("/api/v1/" + path, headers=header).status_code == 401
        assert (
            client.post(
                "/api/v1/query/async", headers=header, json={"query": "Revenue?"}
            ).status_code
            == 401
        )


def test_owner_is_derived_from_remote_identity_across_resources(api_app, monkeypatch):
    identity_service(monkeypatch)
    monkeypatch.setenv("FINAGENT_EMBEDDED_WORKER", "0")
    a, b = {"Authorization": "Bearer token-a"}, {"Authorization": "Bearer token-b"}
    store = api_app.state.store
    with TestClient(api_app) as client:
        result = client.post(
            "/api/v1/query/async",
            headers=a,
            json={"query": "Revenue?", "thread_id": "shared-id", "tenant_id": OWNER_B},
        )
        assert result.status_code == 202
        job = result.json()["job_id"]
        assert store.get("threads", "shared-id", OWNER_A)
        assert store.get("threads", "shared-id", OWNER_B) is None
        assert client.get("/api/v1/threads", headers=b).json()["items"] == []
        saved = client.get("/api/v1/threads", headers=a).json()["items"]
        assert [item["thread_id"] for item in saved] == ["shared-id"]
        assert "state_values" not in saved[0]
        assert client.get(f"/api/v1/jobs/{job}", headers=b).status_code == 404
        assert (
            client.get("/api/v1/threads/shared-id/state", headers=b).status_code == 404
        )
        assert (
            client.post(
                "/api/v1/threads/shared-id/resume",
                headers=b,
                json={"action": "APPROVE", "analyst_id": OWNER_A},
            ).status_code
            == 404
        )
        upload = client.post(
            "/api/v1/documents",
            headers=a,
            files={"file": ("report.pdf", b"%PDF-1.7 fixture", "application/pdf")},
        )
        assert upload.status_code == 202
        doc = upload.json()["document_id"]
        assert "storage_key" not in upload.json()
        assert client.get("/api/v1/documents", headers=b).json()["items"] == []
        for suffix in ("", "/evidence", "/download"):
            assert (
                client.get(f"/api/v1/documents/{doc}" + suffix, headers=b).status_code
                == 404
            )
        assert client.delete(f"/api/v1/documents/{doc}", headers=b).status_code == 404
        assert (
            client.post(
                "/api/v1/query/async",
                headers=b,
                json={"query": "Revenue?", "document_ids": [doc]},
            ).status_code
            == 404
        )
        assert (
            client.get(f"/api/v1/documents/{doc}/download", headers=a).headers[
                "cache-control"
            ]
            == "private, no-store"
        )
        assert (
            client.post(
                "/api/v1/query/async",
                headers=b,
                json={"query": "Independent?", "thread_id": "shared-id"},
            ).status_code
            == 202
        )


@pytest.mark.parametrize("user", [None, [], {"id": ""}, {"id": "not-a-user-uuid"}])
def test_malformed_identity_provider_response_fails_closed(api_app, monkeypatch, user):
    monkeypatch.setenv("FINAGENT_AUTH_MODE", "supabase")
    monkeypatch.setattr(
        auth,
        "supabase_request",
        lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: user),
    )
    with TestClient(api_app) as client:
        assert client.get("/api/v1/documents", headers=LOCAL).status_code == 503


def test_real_request_sizes_and_admission_limits(api_app, monkeypatch):
    monkeypatch.setenv("FINAGENT_EMBEDDED_WORKER", "0")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "64")
    monkeypatch.setenv("MAX_PENDING_JOBS", "1")
    with TestClient(api_app, headers=LOCAL) as client:
        too_big = client.post(
            "/api/v1/query/async",
            json={
                "query": "Revenue?",
                "metadata": {"description": "x" * (1024 * 1024)},
            },
        )
        assert too_big.status_code == 413
        assert (
            client.post(
                "/api/v1/documents",
                files={"file": ("a.pdf", b"%PDF-" + b"x" * 64, "application/pdf")},
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/v1/query/async", json={"query": "Revenue?", "thread_id": "first"}
            ).status_code
            == 202
        )
        assert (
            client.post(
                "/api/v1/query/async", json={"query": "Profit?", "thread_id": "second"}
            ).status_code
            == 429
        )
        assert client.get("/api/v1/threads/second/state").status_code == 404


def test_duplicate_submissions_and_durable_rate_limit(api_app, monkeypatch):
    monkeypatch.setenv("MAX_QUERIES_PER_MINUTE", "1")
    monkeypatch.setenv("FINAGENT_EMBEDDED_WORKER", "0")
    with TestClient(api_app, headers=LOCAL) as client:

        def submit(_):
            return client.post(
                "/api/v1/query/async", json={"query": "Revenue?", "thread_id": "shared"}
            ).status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(submit, range(2))) == [202, 409]
    from agent.server.app import create_app

    restarted = create_app()
    restarted.state.agent = api_app.state.agent
    with TestClient(restarted, headers=LOCAL) as client:
        resp = client.post(
            "/api/v1/query/async", json={"query": "Profit?", "thread_id": "another"}
        )
        assert resp.status_code == 429
        assert "retry-after" in resp.headers


def test_authenticated_reviewer_and_duplicate_resume(api_app, monkeypatch):
    identity_service(monkeypatch)
    monkeypatch.setenv("FINAGENT_EMBEDDED_WORKER", "0")
    from fastapi import HTTPException

    from agent.server.routes import _queue_resume
    from agent.server.schemas import HITLApprovalRequest

    thread = "paused"
    store = api_app.state.store
    store.put(
        "threads",
        thread,
        OWNER_A,
        {
            "thread_id": thread,
            "is_paused": True,
            "review_id": "review-1",
            "input_query": "Revenue?",
            "state_values": {},
        },
    )
    decision = HITLApprovalRequest(action="APPROVE", analyst_id="supplied-in-body")

    def submit(_):
        try:
            return _queue_resume(store, thread, decision, auth.Principal(OWNER_A))
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(submit, range(2)))
    assert outcomes.count(409) == 1
    jobs = store.list("jobs", OWNER_A)
    assert len(jobs) == 1 and jobs[0]["decision"]["analyst_id"] == OWNER_A


def test_file_storage_private_keys_and_secure_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("FINAGENT_FILES_PATH", str(tmp_path))
    storage = FileStorage()
    doc = str(uuid4())
    key = storage.put(OWNER_A, doc, b"fixture content")
    assert key == f"{OWNER_A}/{doc}.pdf"
    assert storage.read(key) == b"fixture content"
    with pytest.raises(ValueError):
        storage.put("two/components", doc, b"invalid")
    assert list((tmp_path / OWNER_A).iterdir()) == [tmp_path / key]
    monkeypatch.setenv("SUPABASE_URL", "http://storage.example.test")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "fixture")
    with pytest.raises(RuntimeError):
        FileStorage()


def test_duplicate_upload_at_capacity_reuses_document_and_ingest_job(
    api_app, monkeypatch
):
    monkeypatch.setenv("MAX_DOCUMENTS", "1")
    service = api_app.state.documents

    def create(_):
        return service.create(OWNER_A, "report.pdf", b"%PDF-same report")["document_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, range(2)))
    assert len(set(results)) == 1
    assert len(service.store.list("jobs", OWNER_A)) == 1
    assert len(service.store.list("documents", OWNER_A)) == 1


def test_document_evidence_is_filtered_before_any_result_limit(api_app):
    from agent.documents.models import Evidence

    service = api_app.state.documents
    owner = "owner"
    doc = "requested"
    service.store.put("documents", doc, owner, {"document_id": doc, "status": "ready"})
    row = Evidence(
        evidence_id="selected-cell",
        document_id=doc,
        filename="a.pdf",
        document_hash="fixture",
        page=1,
        text="Revenue grew.",
    )
    service.store.put("evidence", row.evidence_id, owner, row.model_dump())
    with service.store.connection() as conn:
        conn.execute(
            "WITH RECURSIVE rows(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM rows WHERE n<50001) INSERT INTO workspace_records(kind,id,owner_id,payload,updated_at) SELECT 'evidence','extra-' || n,?,json_object('document_id','other'),'2999-01-01' FROM rows",
            (owner,),
        )
        conn.commit()
    assert [e.evidence_id for e in service.evidence(owner, [doc])] == [row.evidence_id]
    with service.store.transaction(owner) as tx:
        tx.clear_evidence(doc)
        assert tx.count("evidence") == 50001


def test_upload_registration_failure_cleans_up_unreferenced_file(api_app):
    service = api_app.state.documents
    with service.store.connection() as conn:
        conn.execute(
            "CREATE TRIGGER fail_ingest BEFORE INSERT ON workspace_records WHEN NEW.kind='jobs' BEGIN SELECT RAISE(ABORT, 'temporary fixture failure'); END"
        )
        conn.commit()
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        service.create("owner", "source.pdf", b"%PDF-fixture")
    assert service.store.list("documents", "owner") == []
    assert not list(service.storage.root.rglob("*.pdf"))
