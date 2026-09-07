"""Model Context Protocol (MCP) Resource providers for financial reasoning."""

from __future__ import annotations

import json
from typing import Any

from agent.prompts import load_prompt

NEO4J_SCHEMA_DEFINITION = {
    "nodes": {
        "Company": {"properties": ["id", "name", "ticker", "sector"]},
        "Report": {"properties": ["id", "year", "quarter", "source_file", "record_id", "question", "split"]},
        "Table": {"properties": ["id", "title", "page_number"]},
        "Row": {"properties": ["id", "label", "unit", "scale", "category"]},
        "Value": {"properties": ["id", "amount", "normalized_amount", "year"]},
        "TextChunk": {"properties": ["id", "content", "position"]},
        "Metric": {"properties": ["id", "name", "category", "subcategory"]},
    },
    "relationships": [
        "(Company)-[:FILED]->(Report)",
        "(Report)-[:CONTAINS_TABLE]->(Table)",
        "(Table)-[:HAS_ROW]->(Row)",
        "(Row)-[:HAS_VALUE]->(Value)",
        "(Value)-[:IN_REPORT]->(Report)",
        "(Value)-[:NEXT_YEAR]->(Value)",
        "(Metric)-[:MEASURED_BY]->(Row)",
        "(Report)-[:HAS_CONTEXT]->(TextChunk)",
    ],
}


def get_graph_schema_resource() -> str:
    """Returns the structural Neo4j knowledge graph schema."""
    return json.dumps(NEO4J_SCHEMA_DEFINITION, indent=2)


def get_prompt_template_resource(prompt_name: str) -> str:
    """Returns the versioned system prompt template and hash metadata."""
    try:
        content, version, sha_hash = load_prompt(prompt_name)
        payload = {
            "name": version,
            "hash": sha_hash,
            "template": content,
        }
        return json.dumps(payload, indent=2)
    except FileNotFoundError:
        return json.dumps({"error": f"Prompt template '{prompt_name}' not found."})


def get_filing_reference_resource(record_id: str) -> str:
    """Returns static filing metadata format."""
    mock_meta = {
        "record_id": record_id,
        "uri": f"financial://filings/{record_id}",
        "access": "read-only",
        "description": "Normalized FinQA filing context container",
    }
    return json.dumps(mock_meta, indent=2)