from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from langgraph.types import Command

from agent.documents.models import Evidence
from agent.documents.reasoning import answer_question
from agent.documents.service import DocumentService
from agent.server.store import RecordStore, utc_now
from agent.state.schema import AgentStateV1

logger = logging.getLogger(__name__)


def thread_config(owner_id: str, thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": f"{owner_id}:{thread_id}",
            "owner_id": owner_id,
            "tenant_id": owner_id,
        }
    }


def _checkpoint_id(snapshot: Any) -> str | None:
    config = getattr(snapshot, "config", None) or {}
    value = config.get("configurable", {}).get("checkpoint_id")
    return str(value) if value else None


def _interrupted(snapshot: Any) -> bool:
    if not snapshot:
        return False
    return bool(
        "hitl_gate" in (snapshot.next or ())
        or any(
            getattr(task, "interrupts", ()) for task in getattr(snapshot, "tasks", ())
        )
    )


def _audit_document_job(owner: str, job: dict[str, Any]) -> None:
    from agent.nodes import hitl_nodes

    hitl_nodes._ensure_audit_schema()
    result = job.get("result") or {}
    ids = (
        [job["document_id"]]
        if job.get("kind") == "ingest"
        else job.get("query_payload", {}).get("document_ids", [])
    )
    stage = "ingest" if job.get("kind") == "ingest" else "document_answer"
    content = {
        "answer": result.get("final_answer"),
        "calculation": result.get("calculation"),
        "references": [
            (c.get("evidence_id"), c.get("quote")) for c in result.get("citations", [])
        ],
        "document_hash": result.get("sha256"),
        "evidence_count": result.get("evidence_count"),
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    hitl_nodes.AUDIT_LEDGER.record_transition(
        trace_id=job["job_id"],
        node_name=stage,
        actor_id=owner,
        action_type="DOCUMENT_JOB_COMPLETED",
        state_payload={
            "documents": ids,
            "citations": [c.get("evidence_id") for c in result.get("citations", [])],
            "calculation": result.get("calculation"),
            "answer_status": result.get("answer_status") or result.get("status"),
            "document_hash": result.get("sha256"),
            "evidence_count": result.get("evidence_count"),
            "result_digest": digest,
        },
        metadata={
            "event_id": f"{owner}:{job['job_id']}:{job.get('created_at','')}:{stage}:{digest}"
        },
    )


class Worker:
    def __init__(
        self, store: RecordStore, agent: Any, documents: DocumentService | None = None
    ):
        self.store = store
        self.agent = agent
        self.documents = documents or DocumentService(store)
        self.worker_id = str(uuid.uuid4())
        self.lease_seconds = max(3, int(os.getenv("WORKER_LEASE_SECONDS", "600")))
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def _heartbeat(
        self, stop: threading.Event, owner: str, job: dict[str, Any]
    ) -> None:
        while not stop.wait(self.lease_seconds / 3):
            try:
                if not self.store.renew(owner, job, self.lease_seconds):
                    return
            except Exception:
                logger.exception("Could not renew job lease for %s", job["job_id"])

    def _graph_result(
        self, owner: str, job: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        payload = job["query_payload"]
        config = thread_config(owner, job["thread_id"])
        is_resume = job.get("kind") == "resume"
        entry: Any = (
            Command(resume=job["decision"])
            if is_resume
            else AgentStateV1(
                input=payload["query"],
                company_identifier=payload.get("company_identifier"),
                owner_id=owner,
            ).model_dump()
        )
        if job["attempts"] > 1 or is_resume:
            prior = self.agent.get_state(config)
            moved = (
                is_resume
                and job.get("checkpoint_id")
                and _checkpoint_id(prior) != job["checkpoint_id"]
            )
            if prior and prior.values and (job["attempts"] > 1 or moved):
                if not prior.next or (_interrupted(prior) and (not is_resume or moved)):
                    return dict(prior.values), prior
                if not _interrupted(prior):
                    entry = None
        result = self.agent.invoke(entry, config=config)
        return result, self.agent.get_state(config)

    def run_once(self) -> bool:
        job = self.store.claim_job(self.worker_id, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        owner = job.pop("owner_id")
        done = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat, args=(done, owner, job), daemon=True
        )
        heartbeat.start()
        started = time.monotonic()
        document: dict[str, Any] | None = None
        evidence: list[Evidence] = []
        thread_changes: dict[str, Any] = {}
        try:
            if job.get("attempts", 0) > 3:
                raise RuntimeError("Retry limit reached.")
            if job.get("kind") == "ingest":
                with self.store.transaction(owner) as tx:
                    if not tx.lease_current(job):
                        return True
                    pending = tx.get("documents", job["document_id"])
                    if pending and pending.get("status") in {"queued", "processing"}:
                        tx.put(
                            "documents",
                            job["document_id"],
                            {
                                **pending,
                                "status": "processing",
                                "updated_at": utc_now(),
                            },
                        )
                document, evidence = self.documents.prepare_ingest(
                    owner, job["document_id"]
                )
                job.update(
                    status="COMPLETED" if document["status"] == "ready" else "FAILED",
                    result={k: v for k, v in document.items() if k != "storage_key"},
                    error=document.get("error"),
                )
            else:
                payload = job["query_payload"]
                if payload.get("document_ids"):
                    sources = self.documents.evidence(owner, payload["document_ids"])
                    result = answer_question(payload["query"], sources)
                    result.update(
                        input=payload["query"],
                        document_ids=payload["document_ids"],
                        owner_id=owner,
                    )
                    snapshot = None
                else:
                    result, snapshot = self._graph_result(owner, job)
                paused = _interrupted(snapshot)
                checkpoint = _checkpoint_id(snapshot)
                thread_changes = {
                    "state_values": result,
                    "is_paused": paused,
                    "next_nodes": (
                        list(snapshot.next) if snapshot is not None and paused else []
                    ),
                    "hitl_status": result.get("hitl_status", "NONE"),
                    "checkpoint_id": checkpoint,
                    "review_id": (checkpoint or uuid.uuid4().hex) if paused else None,
                }
                job.update(
                    status="SUSPENDED_HITL" if paused else "COMPLETED",
                    result=result,
                    error=None,
                )
        except Exception:
            logger.exception("Job failed: %s", job["job_id"])
            message = "The analysis could not complete. Check the worker logs using the job ID."
            job.update(status="FAILED", error=message, result=None)
            if job.get("kind") == "query":
                thread_changes = {
                    "is_paused": False,
                    "next_nodes": [],
                    "state_values": {"error": message, "is_terminal": True},
                }
        finally:
            try:
                with self.store.transaction(owner) as tx:
                    if tx.lease_current(job):
                        if job["status"] == "COMPLETED" and (
                            document is not None
                            or job.get("query_payload", {}).get("document_ids")
                        ):
                            try:
                                _audit_document_job(owner, job)
                            except Exception:
                                logger.exception(
                                    "Could not record document audit for %s",
                                    job["job_id"],
                                )
                                message = (
                                    "The required audit record could not be saved."
                                )
                                job.update(status="FAILED", error=message, result=None)
                                if document is not None:
                                    document.update(
                                        status="failed", error=message, evidence_count=0
                                    )
                                    evidence = []
                                else:
                                    thread_changes = {
                                        "state_values": {
                                            "error": message,
                                            "is_terminal": True,
                                        },
                                        "is_paused": False,
                                        "next_nodes": [],
                                    }
                        if document is not None:
                            self.documents.save_ingest(tx, document, evidence)
                        elif job.get("kind") == "ingest" and job["status"] == "FAILED":
                            record = tx.get("documents", job["document_id"])
                            if record:
                                tx.put(
                                    "documents",
                                    job["document_id"],
                                    {
                                        **record,
                                        "status": "failed",
                                        "error": "The report could not be processed.",
                                        "updated_at": utc_now(),
                                    },
                                )
                        if thread_changes:
                            thread = tx.get("threads", job["thread_id"]) or {}
                            tx.put(
                                "threads",
                                job["thread_id"],
                                {**thread, **thread_changes, "updated_at": utc_now()},
                            )
                        job.update(
                            updated_at=utc_now(),
                            execution_time_ms=round(
                                (time.monotonic() - started) * 1000, 3
                            ),
                        )
                        tx.put("jobs", job["job_id"], job)
            finally:
                done.set()
                heartbeat.join(timeout=1)
        return True

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.run_once():
                    self.stop_event.wait(0.25)
            except Exception:
                logger.exception("Worker could not process a job")
                self.stop_event.wait(2)

    def start(self, concurrency: int = 2) -> None:
        for index in range(max(1, min(concurrency, 8))):
            thread = threading.Thread(
                target=self._loop, name=f"financial-worker-{index}", daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)


def main() -> None:
    from agent.graph import create_financial_agent

    worker = Worker(RecordStore(), create_financial_agent())
    worker.start(int(os.getenv("WORKER_CONCURRENCY", "2")))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
