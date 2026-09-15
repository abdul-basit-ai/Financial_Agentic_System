"""Process-wide cached clients for heavy tool dependencies.

Every retrieval/memory tool call used to construct a fresh client object,
reloading the SentenceTransformer encoder from disk each time (~1-2s plus
hundreds of MB of churn, multiplied across parallel fan-out branches). The
lru_cache singletons here keep one encoder and one client per store for the
process lifetime.

Imports are deliberately function-local: the tool modules import this module,
so importing them here at module level would be circular. Tests that need
isolation should keep using the `client=` injection parameter on the tool
entrypoints rather than clearing these caches.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agent.tools.vector_tool import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_sentence_encoder(model_name: str = EMBEDDING_MODEL_NAME) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_vector_tool_client() -> Any:
    from agent.tools.vector_tool import VectorRetrievalTool

    return VectorRetrievalTool()


@lru_cache(maxsize=1)
def get_episodic_store() -> Any:
    from agent.memory.episodic import EpisodicMemoryStore

    return EpisodicMemoryStore()


@lru_cache(maxsize=1)
def get_procedural_bank() -> Any:
    from agent.memory.procedural import ProceduralMemoryBank

    return ProceduralMemoryBank()


@lru_cache(maxsize=1)
def get_memory_tool_client() -> Any:
    from agent.tools.memory_tool import MemoryRetrievalTool

    # Share the cached episodic/procedural instances so the whole memory
    # tier loads the encoder exactly once.
    return MemoryRetrievalTool(
        episodic_store=get_episodic_store(),
        procedural_bank=get_procedural_bank(),
    )
