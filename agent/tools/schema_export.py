"""Export JSON Schemas for every Phase 4 tool (MCP Phase 10 readiness).

Run directly to (re)generate agent/tools/schemas/<tool>_schema.json files.
Schemas are derived from the Pydantic input/output models, so they can never
drift from the code — regeneration is idempotent.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from agent.tools.decomposer import DecompositionInput, DecompositionOutput
from agent.tools.fusion_tool import ContextFusionInput, ContextFusionOutput
from agent.tools.graph_tool import GraphQueryInput, GraphQueryOutput
from agent.tools.safe_math import SafeMathInput, SafeMathOutput
from agent.tools.vector_tool import VectorSearchInput, VectorSearchOutput

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "safe_math": {
        "input": SafeMathInput.model_json_schema(),
        "output": SafeMathOutput.model_json_schema(),
    },
    "graph_retrieval": {
        "input": GraphQueryInput.model_json_schema(),
        "output": GraphQueryOutput.model_json_schema(),
    },
    "vector_retrieval": {
        "input": VectorSearchInput.model_json_schema(),
        "output": VectorSearchOutput.model_json_schema(),
    },
    "context_fusion": {
        "input": ContextFusionInput.model_json_schema(),
        "output": ContextFusionOutput.model_json_schema(),
    },
    "query_decomposition": {
        "input": DecompositionInput.model_json_schema(),
        "output": DecompositionOutput.model_json_schema(),
    },
}


def export_schemas(out_dir: str = SCHEMA_DIR) -> dict[str, str]:
    """Writes one JSON file per tool. Returns mapping of tool -> file path."""
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    for tool_name, schemas in TOOL_SCHEMAS.items():
        payload = {
            "tool": tool_name,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            **schemas,
        }
        path = os.path.join(out_dir, f"{tool_name}_schema.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        written[tool_name] = path
    return written


def load_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Loads a previously exported schema (used by Phase 10 MCP server)."""
    path = os.path.join(SCHEMA_DIR, f"{tool_name}_schema.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    for tool, path in export_schemas().items():
        print(f"exported {tool} -> {path}")
