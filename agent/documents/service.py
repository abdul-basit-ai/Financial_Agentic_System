from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import PurePath
from typing import Any

from agent.documents.models import Evidence
from agent.documents.parser import DocumentParseError, parse_pdf
from agent.documents.storage import FileStorage
from agent.server.store import AdmissionLimitError, RecordStore, Transaction, utc_now

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, store: RecordStore, storage: FileStorage | None = None):
        self.store = store
        self.storage = storage or FileStorage()

    def create(
        self,
        owner_id: str,
        filename: str,
        content: bytes,
        company_identifier: str | None = None,
    ) -> dict[str, Any]:
        if not content.startswith(b"%PDF-"):
            raise ValueError("Upload a valid PDF report.")
        if len(content) > int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))):
            raise ValueError("The report exceeds the upload size limit.")
        digest = hashlib.sha256(content).hexdigest()
        key = None
        try:
            with self.store.transaction(owner_id) as tx:
                existing = tx.find_document(digest)
                if existing and existing.get("status") == "deleting":
                    raise ValueError("Report deletion is in progress.")
                if existing and existing.get("status") != "failed":
                    return existing
                if not existing and tx.count("documents") >= int(
                    os.getenv("MAX_DOCUMENTS", "100")
                ):
                    raise AdmissionLimitError(
                        "The report storage limit has been reached."
                    )
                tx.admit()
                document_id = (
                    existing["document_id"]
                    if existing
                    else str(uuid.uuid5(uuid.NAMESPACE_URL, owner_id + ":" + digest))
                )
                key = self.storage.put(owner_id, document_id, content)
                now = utc_now()
                record = {
                    "document_id": document_id,
                    "filename": PurePath(filename.replace("\\", "/")).name[:200],
                    "company_identifier": company_identifier,
                    "status": "queued",
                    "page_count": 0,
                    "created_at": (existing or {}).get("created_at", now),
                    "updated_at": now,
                    "sha256": digest,
                    "storage_key": key,
                    "error": None,
                }
                tx.put("documents", document_id, record)
                job_id = "ingest_" + document_id
                tx.put(
                    "jobs",
                    job_id,
                    {
                        "job_id": job_id,
                        "kind": "ingest",
                        "document_id": document_id,
                        "thread_id": "",
                        "status": "QUEUED",
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                return record
        except Exception:
            if key:
                try:
                    with self.store.transaction(owner_id) as cleanup:
                        if cleanup.find_document(digest) is None:
                            self.storage.delete(key)
                except Exception:
                    logger.exception("Could not clean up a failed report registration")
            raise

    def prepare_ingest(
        self, owner_id: str, document_id: str
    ) -> tuple[dict[str, Any], list[Evidence]]:
        record = self.store.get("documents", document_id, owner_id)
        if not record or record.get("status") == "deleting":
            raise ValueError("Report not found.")
        try:
            content = self.storage.read(record["storage_key"])
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("Source checksum mismatch")
            evidence, page_count = parse_pdf(content, document_id, record["filename"])
            record.update(
                status="ready",
                page_count=page_count,
                evidence_count=len(evidence),
                error=None,
            )
            return record, evidence
        except Exception as exc:
            logger.exception("Report ingestion failed for %s", document_id)
            message = (
                str(exc)
                if isinstance(exc, DocumentParseError)
                else "The report could not be processed."
            )
            record.update(
                status="failed", page_count=0, evidence_count=0, error=message
            )
            return record, []

    def save_ingest(
        self, tx: Transaction, record: dict[str, Any], evidence: list[Evidence]
    ) -> None:
        doc_id = record["document_id"]
        current = tx.get("documents", doc_id)
        if not current or current.get("status") == "deleting":
            raise ValueError("Report no longer available.")
        tx.replace_evidence(
            doc_id, [item.model_dump(exclude_none=True) for item in evidence]
        )
        record["updated_at"] = utc_now()
        tx.put("documents", doc_id, record)

    def ingest(self, owner_id: str, document_id: str) -> dict[str, Any]:
        record, evidence = self.prepare_ingest(owner_id, document_id)
        with self.store.transaction(owner_id) as tx:
            self.save_ingest(tx, record, evidence)
        return record

    def evidence(self, owner_id: str, document_ids: list[str]) -> list[Evidence]:
        selected = list(dict.fromkeys(document_ids))
        with self.store.transaction(owner_id, write=False) as tx:
            for doc_id in selected:
                doc = tx.get("documents", doc_id)
                if not doc or doc.get("status") != "ready":
                    raise ValueError(
                        "Every selected report must be available and ready."
                    )
            return [Evidence.model_validate(row) for row in tx.evidence(selected)]

    def delete(self, owner_id: str, document_id: str) -> None:
        with self.store.transaction(owner_id) as tx:
            record = tx.get("documents", document_id)
            if not record:
                raise ValueError("Report not found.")
            if record.get("status") in {"queued", "processing"}:
                raise ValueError(
                    "Wait for report processing to finish before deleting it."
                )
            token = record.get("deletion_token") or uuid.uuid4().hex
            tx.put(
                "documents",
                document_id,
                {**record, "status": "deleting", "deletion_token": token},
            )
        with self.store.transaction(owner_id) as tx:
            current = tx.get("documents", document_id)
            if not current or current.get("deletion_token") != token:
                return
            self.storage.delete(record["storage_key"])
            tx.clear_evidence(document_id)
            tx.delete("documents", document_id)
            tx.delete("jobs", "ingest_" + document_id)
