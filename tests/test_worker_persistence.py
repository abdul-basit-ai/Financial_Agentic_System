import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from agent.server.store import RecordStore
from agent.worker import Worker


def queued(job_id="job-1"):
    return {
        "job_id": job_id,
        "thread_id": "analysis",
        "kind": "query",
        "status": "QUEUED",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "query_payload": {"query": "Revenue?"},
    }


class Agent:
    def __init__(self, answer):
        self.answer = answer
        self.called = 0

    def invoke(self, state, config):
        self.called += 1
        assert config["configurable"]["owner_id"] == "owner"
        assert config["configurable"]["thread_id"] == "owner:analysis"
        return {"final_answer": self.answer, "is_terminal": True}

    def get_state(self, config):
        return SimpleNamespace(next=(), values={})


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    return RecordStore(path=str(tmp_path / "jobs.sqlite"))


def test_store_reopens_and_independent_workers_claim_once(store):
    store.put("jobs", "job-1", "owner", queued())
    second = RecordStore(path=store.path)
    assert second.get("jobs", "job-1", "owner") == queued()
    gate = threading.Barrier(2)

    def claim(s):
        gate.wait(timeout=3)
        return s.claim_job("worker")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, [store, second]))
    assert sum(r is not None for r in results) == 1
    assert second.get("jobs", "job-1", "owner")["attempts"] == 1
    assert second.get("jobs", "job-1", "another-owner") is None


def test_expired_lease_can_be_recovered_and_completed_once(store):
    store.put("jobs", "job-1", "owner", queued())
    abandoned = store.claim_job("terminated", lease_seconds=-1)
    agent = Agent("new result")
    worker = Worker(RecordStore(path=store.path), agent)
    assert worker.run_once()
    job = store.get("jobs", "job-1", "owner")
    assert (
        job["status"] == "COMPLETED" and job["result"]["final_answer"] == "new result"
    )
    assert job["attempts"] == 2 and job["lease_token"] != abandoned["lease_token"]
    assert not worker.run_once()
    assert agent.called == 1


def test_renewal_preserves_payload_and_refuses_obsolete_lease(store):
    store.put("jobs", "job-1", "owner", queued())
    first = store.claim_job("abandoned", lease_seconds=-1)
    second = store.claim_job("replacement", lease_seconds=15)
    second["status"] = "COMPLETED"
    assert store.renew("owner", second, 60)
    persisted = store.get("jobs", "job-1", "owner")
    assert persisted["status"] == "RUNNING"
    assert persisted["lease_until"] > second["lease_until"]
    assert not store.renew("owner", first, 60)


def test_stale_worker_cannot_replace_new_result_or_thread_state(store):
    entered, release = threading.Event(), threading.Event()

    class Slow(Agent):
        def invoke(self, state, config):
            entered.set()
            assert release.wait(timeout=10)
            return super().invoke(state, config)

    store.put("jobs", "job-1", "owner", queued())
    old = Worker(store, Slow("stale"))
    new = Worker(RecordStore(path=store.path), Agent("valid"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(old.run_once)
        try:
            assert entered.wait(timeout=5)
            job = store.get("jobs", "job-1", "owner")
            job["lease_until"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
            store.put("jobs", "job-1", "owner", job)
            assert new.run_once()
        finally:
            release.set()
        assert pending.result(timeout=5)
    assert store.get("jobs", "job-1", "owner")["result"]["final_answer"] == "valid"
    assert (
        store.get("threads", "analysis", "owner")["state_values"]["final_answer"]
        == "valid"
    )


def test_worker_failure_is_terminal_and_sanitized(store):
    class Fails(Agent):
        def invoke(self, state, config):
            raise RuntimeError("internal test diagnostic")

    store.put("jobs", "job-1", "owner", queued())
    worker = Worker(store, Fails("unused"))
    assert worker.run_once()
    job = store.get("jobs", "job-1", "owner")
    assert job["status"] == "FAILED" and not job["result"]
    assert job["error"] and "internal test diagnostic" not in job["error"]
    thread = store.get("threads", "analysis", "owner")
    assert (
        thread["state_values"]["is_terminal"]
        and thread["state_values"]["error"] == job["error"]
    )
    assert not worker.run_once()


def test_restart_reuses_an_already_checkpointed_terminal_result(store):
    class Checkpointed(Agent):
        def get_state(self, config):
            return SimpleNamespace(
                next=(),
                values={"final_answer": "already computed", "is_terminal": True},
            )

        def invoke(self, state, config):
            raise AssertionError("A completed checkpoint must not rerun.")

    store.put("jobs", "job-1", "owner", queued())
    store.claim_job("dead-process", lease_seconds=-1)
    assert Worker(store, Checkpointed("unused")).run_once()
    assert (
        store.get("jobs", "job-1", "owner")["result"]["final_answer"]
        == "already computed"
    )


def test_thread_registration_rolls_back_when_job_persistence_fails(
    api_app, monkeypatch
):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("FINAGENT_EMBEDDED_WORKER", "0")
    store = api_app.state.store
    with store.connection() as conn:
        conn.execute(
            "CREATE TRIGGER fail_job BEFORE INSERT ON workspace_records WHEN NEW.kind='jobs' BEGIN SELECT RAISE(ABORT, 'fixture outage'); END"
        )
        conn.commit()
    with TestClient(
        api_app,
        raise_server_exceptions=False,
        headers={"Authorization": "Bearer test-worker-token"},
    ) as client:
        body = {"query": "Revenue?", "thread_id": "retry-this-id"}
        assert client.post("/api/v1/query/async", json=body).status_code == 500
        assert client.get("/api/v1/threads/retry-this-id/state").status_code == 404
        with store.connection() as conn:
            conn.execute("DROP TRIGGER fail_job")
            conn.commit()
        assert client.post("/api/v1/query/async", json=body).status_code == 202


def test_active_worker_renews_lease_during_work(store, monkeypatch):
    monkeypatch.setenv("WORKER_LEASE_SECONDS", "3")
    renewed = threading.Event()
    original = store.renew

    def renew(owner, job, seconds):
        ok = original(owner, job, seconds)
        if ok:
            renewed.set()
        return ok

    monkeypatch.setattr(store, "renew", renew)

    class LongRunning(Agent):
        def invoke(self, state, config):
            assert renewed.wait(timeout=6)
            return super().invoke(state, config)

    store.put("jobs", "job-1", "owner", queued())
    assert Worker(store, LongRunning("finished")).run_once()
    assert (
        renewed.is_set()
        and store.get("jobs", "job-1", "owner")["status"] == "COMPLETED"
    )


@pytest.mark.parametrize("audit_available", [False, True])
def test_report_answers_are_only_released_after_audit(
    store, monkeypatch, audit_available
):
    import agent.worker as worker_module
    from agent.nodes import hitl_nodes

    sources = SimpleNamespace(evidence=lambda owner, ids: [])
    answer = {
        "final_answer": "approved figure",
        "answer_status": "grounded",
        "citations": [{"evidence_id": "cell-1"}],
        "calculation": {"result": 8},
        "is_terminal": True,
    }
    monkeypatch.setattr(
        worker_module, "answer_question", lambda question, corpus: answer.copy()
    )
    events = []

    def audit(**kwargs):
        events.append(kwargs)
        if not audit_available:
            raise RuntimeError("audit write failure")

    monkeypatch.setattr(hitl_nodes.AUDIT_LEDGER, "record_transition", audit)
    job = queued()
    job["query_payload"]["document_ids"] = ["document"]
    store.put("jobs", job["job_id"], "owner", job)
    assert Worker(store, Agent("unused"), sources).run_once()
    final = store.get("jobs", job["job_id"], "owner")
    thread = store.get("threads", "analysis", "owner")["state_values"]
    assert final["status"] == ("COMPLETED" if audit_available else "FAILED")
    assert events[0]["actor_id"] == "owner"
    assert events[0]["state_payload"]["citations"] == ["cell-1"]
    assert events[0]["metadata"]["event_id"]
    if audit_available:
        assert (
            final["result"]["final_answer"]
            == thread["final_answer"]
            == "approved figure"
        )
    else:
        assert not final["result"] and thread.get("final_answer") is None
        assert thread["is_terminal"]
