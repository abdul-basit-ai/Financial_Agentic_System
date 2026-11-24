"""Record-scoped evidence comparison for FinQA."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from evaluation.metrics import evidence_hit, ranked_metrics


@dataclass
class _Evidence:
    record_id: str | None
    gold_ids: frozenset[str]
    text: str
    kind: str


def _value_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("column_header") or record.get("year"),
        record.get("raw") or record.get("amount"),
    ]
    return " ".join(str(part) for part in parts if part is not None)


def _collect_evidence(
    envelopes: list[dict[str, Any]],
) -> tuple[list[_Evidence], list[_Evidence], list[_Evidence]]:
    graph_parts: dict[tuple[Any, ...], dict[Any, dict[str, Any]]] = defaultdict(dict)
    vector: dict[tuple[Any, ...], _Evidence] = {}
    table: dict[tuple[Any, ...], _Evidence] = {}

    for envelope in envelopes:
        data = envelope.get("data")
        if not envelope.get("success", True) or not isinstance(data, dict):
            continue
        for item in data.get("records", []):
            group_key = (
                item.get("report_id"),
                item.get("row_evidence_id"),
                item.get("row_index"),
                item.get("row_label"),
            )
            cell_key = item.get("evidence_id") or (
                item.get("column_index"),
                item.get("year"),
            )
            graph_parts[group_key][cell_key] = item

        for item in data.get("chunks", []):
            record_id = item.get("record_id")
            key = (
                record_id,
                item.get("section"),
                item.get("evidence_id") or item.get("chunk_index"),
            )
            vector[key] = _Evidence(
                record_id,
                frozenset(item.get("gold_ids") or []),
                str(item.get("text_content") or ""),
                "text",
            )

        for row in data.get("rows", []):
            record_id = data.get("record_id")
            table_key = (record_id, row.get("evidence_id") or row.get("row_index"))
            gold_id = row.get("gold_id")
            table[table_key] = _Evidence(
                record_id,
                frozenset([gold_id] if gold_id else []),
                " ".join(
                    [
                        str(row.get("row_label") or ""),
                        *map(_value_text, row.get("cells", [])),
                    ]
                ),
                "table",
            )

    graph = []
    for (record_id, _, _, label), cells in graph_parts.items():
        ordered = sorted(cells.values(), key=lambda cell: cell.get("column_index") or 0)
        gold_id = ordered[0].get("gold_id")
        graph.append(
            _Evidence(
                record_id,
                frozenset([gold_id] if gold_id else []),
                " ".join([str(label or ""), *map(_value_text, ordered)]),
                "table",
            )
        )
    return graph, list(vector.values()), list(table.values())


def score_evidence(
    record_id: str,
    gold_ids: list[str],
    gold_texts: list[str],
    envelopes: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, float]:
    graph, vector, table = _collect_evidence(envelopes)
    gold = [
        (
            gold_ids[index] if index < len(gold_ids) else None,
            gold_texts[index] if index < len(gold_texts) else "",
        )
        for index in range(max(len(gold_ids), len(gold_texts)))
    ]

    def matches(candidate: _Evidence, gold_id: str | None, gold_text: str) -> bool:
        if candidate.record_id and candidate.record_id != record_id:
            return False
        if gold_id and candidate.gold_ids:
            return candidate.record_id == record_id and gold_id in candidate.gold_ids
        if gold_id and gold_id.startswith(("text_", "table_")):
            if not gold_id.startswith(f"{candidate.kind}_"):
                return False
        return evidence_hit(gold_text, candidate.text)

    def ranked(items: list[_Evidence], limit: int) -> dict[str, float]:
        hits = [
            {
                idx
                for idx, (gid, content) in enumerate(gold)
                if matches(item, gid, content)
            }
            for item in items[:limit]
        ]
        return ranked_metrics(hits, len(gold), limit)

    fused = []
    for rank in range(max(len(graph), len(vector))):
        if rank < len(graph):
            fused.append(graph[rank])
        if rank < len(vector):
            fused.append(vector[rank])
    fused.extend(table)
    return {
        **ranked(fused, k),
        "recall_graph": ranked(graph, len(graph))["recall_at_k"] if graph else 0.0,
        "recall_vector": ranked(vector, len(vector))["recall_at_k"] if vector else 0.0,
        "recall_table": ranked(table, len(table))["recall_at_k"] if table else 0.0,
    }
