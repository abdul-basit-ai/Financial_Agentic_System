"""Cryptographic Merkle-chain audit ledger.

Provides tamper-evident SHA-256 state tracking for application events.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    entry_index: int
    timestamp_utc: str
    trace_id: str
    node_name: str
    actor_id: str
    action_type: str
    state_hash: str
    prev_hash: str
    entry_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MerkleAuditLedger:
    """Manages an append-only, tamper-evident cryptographic hash chain of state updates."""

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def _canonical_hash(self, data: Any) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def record_transition(
        self,
        trace_id: str,
        node_name: str,
        actor_id: str,
        action_type: str,
        state_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        event_id = (metadata or {}).get("event_id")
        if event_id:
            for previous in self.entries:
                if previous.metadata.get("event_id") == event_id:
                    return previous
        entry_idx = self.entries[-1].entry_index + 1 if self.entries else 0
        prev_hash = self.entries[-1].entry_hash if self.entries else self.GENESIS_HASH
        timestamp = datetime.now(UTC).isoformat()

        state_hash = self._canonical_hash(state_payload)

        compound_raw = f"{entry_idx}:{prev_hash}:{trace_id}:{node_name}:{actor_id}:{action_type}:{state_hash}:{timestamp}:{self._canonical_hash(metadata or {})}"
        entry_hash = hashlib.sha256(compound_raw.encode("utf-8")).hexdigest()

        entry = AuditEntry(
            entry_index=entry_idx,
            timestamp_utc=timestamp,
            trace_id=trace_id,
            node_name=node_name,
            actor_id=actor_id,
            action_type=action_type,
            state_hash=state_hash,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def verify_integrity(self) -> tuple[bool, str]:
        """Validates that the Merkle hash chain has not been tampered with or altered."""
        for i, entry in enumerate(self.entries):
            if entry.entry_index != i:
                return False, f"Broken index sequence at index {i}."
            expected_prev = (
                self.entries[i - 1].entry_hash if i > 0 else self.GENESIS_HASH
            )
            if entry.prev_hash != expected_prev:
                return False, f"Broken chain link at index {i}: prev_hash mismatch."

            compound_raw = (
                f"{entry.entry_index}:{entry.prev_hash}:{entry.trace_id}:{entry.node_name}:{entry.actor_id}:"
                f"{entry.action_type}:{entry.state_hash}:{entry.timestamp_utc}:{self._canonical_hash(entry.metadata)}"
            )
            recomputed = hashlib.sha256(compound_raw.encode("utf-8")).hexdigest()
            if recomputed != entry.entry_hash:
                return (
                    False,
                    f"Tampered record at index {i}: entry_hash recomputation failed.",
                )

        return True, "Audit chain fully verified. Zero discrepancies found."
