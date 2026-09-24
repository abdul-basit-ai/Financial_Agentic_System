"""Shared pytest configuration.

Forces deterministic (no-LLM) agent behavior during tests: .env may carry a
real OpenRouter/OpenAI key on dev machines, and without this pin the e2e graph
tests would issue real paid API calls and slow down/flake. Set
FINAGENT_TESTS_LLM=1 to opt into live-LLM test runs deliberately.

Also patches all external I/O tools (Neo4j, Postgres, Redis, SentenceTransformer)
so the full test suite runs offline without hanging on unreachable databases
or multi-GB model downloads.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest


def pytest_configure(config) -> None:
    if os.getenv("FINAGENT_TESTS_LLM", "").strip() != "1":
        os.environ.setdefault("OPENAI_API_KEY", "")
        os.environ.setdefault("OPENROUTER_API_KEY", "")

    # Install a lightweight SentenceTransformer stub BEFORE any project import
    # can pull torch. This prevents multi-GB model downloads / OOM on 8GB hosts.
    if "sentence_transformers" not in sys.modules:

        class _FakeST:
            def __init__(self, *a, **k):
                pass

            def encode(self, texts, normalize_embeddings=True, **k):
                single = isinstance(texts, str)
                if single:
                    texts = [texts]
                out = []
                for t in texts:
                    rng = np.random.default_rng(abs(hash(str(t))) % (2**32))
                    v = rng.standard_normal(384).astype(np.float32)
                    if normalize_embeddings:
                        v /= np.linalg.norm(v)
                    out.append(v)
                return out[0] if single else np.stack(out)

        mod = types.ModuleType("sentence_transformers")
        mod.SentenceTransformer = _FakeST
        sys.modules["sentence_transformers"] = mod


# ─── Shared offline tool stubs ──────────────────────────────────────────────


def _make_offline_tools():
    """Build deterministic stubs for external tool dependencies."""
    from agent.tools.base import ToolMetrics, ToolResult
    from agent.tools.graph_tool import GraphQueryOutput
    from agent.tools.memory_tool import MemoryQueryOutput
    from agent.tools.table_tool import TableExtractOutput
    from agent.tools.vector_tool import VectorSearchOutput

    metrics = ToolMetrics(latency_ms=0.0)

    def _memory_stub(payload, client=None):
        return ToolResult(
            tool_name="memory_retrieval",
            success=True,
            data=MemoryQueryOutput(
                episodic_memories=[], matched_archetype=None, total_found=0
            ),
            metrics=metrics,
        )

    def _graph_stub(payload, client=None):
        return ToolResult(
            tool_name="graph_retrieval",
            success=True,
            data=GraphQueryOutput(
                company_identifier=payload.company_identifier, records=[], total_found=0
            ),
            metrics=metrics,
        )

    def _vector_stub(payload, client=None):
        return ToolResult(
            tool_name="vector_retrieval",
            success=True,
            data=VectorSearchOutput(query=payload.query_text, chunks=[], total_found=0),
            metrics=metrics,
        )

    def _table_stub(payload, client=None):
        return ToolResult(
            tool_name="table_extract",
            success=True,
            data=TableExtractOutput(
                record_id=payload.record_id,
                headers=[],
                rows=[],
                total_rows=0,
                total_values=0,
            ),
            metrics=metrics,
        )

    return {
        "agent.nodes.memory_node.memory_retrieval_tool": _memory_stub,
        "agent.nodes.parallel_nodes.graph_retrieval_tool": _graph_stub,
        "agent.nodes.parallel_nodes.vector_retrieval_tool": _vector_stub,
        "agent.nodes.parallel_nodes.table_extract_tool": _table_stub,
        "agent.tools.graph_tool.list_company_record_ids": lambda c, client=None: [],
        "agent.nodes.retrieval_node.vector_retrieval_tool": _vector_stub,
        "agent.nodes.retrieval_node.graph_retrieval_tool": _graph_stub,
    }


@pytest.fixture(autouse=True)
def offline_tools(monkeypatch):
    """Auto-applied: replace all external I/O tools with deterministic stubs."""
    if os.getenv("FINAGENT_INTEGRATION_TESTS", "").strip() == "1":
        return

    for target, fn in _make_offline_tools().items():
        monkeypatch.setattr(target, fn)

    import agent.nodes.hitl_nodes as hitl_mod
    from agent.guardrails.audit import MerkleAuditLedger

    monkeypatch.setattr(hitl_mod, "AUDIT_LEDGER", MerkleAuditLedger())
    monkeypatch.setattr(hitl_mod, "_schema_initialized", True)

    monkeypatch.setattr(
        "agent.nodes.memory_node.MemoryPromotionEngine.promote_session",
        lambda self, session_id, state: None,
    )

    from langgraph.checkpoint.memory import MemorySaver

    monkeypatch.setattr(
        "agent.memory.working.get_checkpointer", lambda redis_url=None: MemorySaver()
    )

    # Skip persistent audit DB tests (Postgres may hang on connect forever)
    monkeypatch.setattr(
        "agent.guardrails.persistent_audit.PersistentMerkleAuditLedger.init_schema",
        lambda self: False,
    )
