"""Query decomposition tool for multi-hop financial reasoning."""

from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
COMMON_METRIC_KEYWORDS = [
    "operating income",
    "net income",
    "operating revenue",
    "revenue",
    "sales",
    "gross profit",
    "gross margin",
    "operating margin",
    "total assets",
    "total liabilities",
    "stockholders' equity",
    "cash flow",
    "diluted eps",
    "eps",
]


class SubTask(BaseModel):
    task_id: str = Field(..., description="Unique subtask identifier, e.g. 'task_1'")
    target_tool: Literal["graph_retrieval", "vector_retrieval", "safe_math"] = Field(...)
    query_payload: dict = Field(..., description="Parameters to pass directly to target tool")
    dependencies: list[str] = Field(default_factory=list, description="IDs of tasks that must resolve first")


class DecompositionOutput(BaseModel):
    original_query: str
    is_multi_hop: bool
    sub_tasks: list[SubTask]
    reasoning_plan: str


def _extract_metric_hint(text: str) -> str | None:
    low = text.lower()
    for kw in COMMON_METRIC_KEYWORDS:
        if kw in low:
            return kw
    return None


def _needs_narrative_evidence(query: str, metric_hint: str | None) -> bool:
    """Text-vs-graph routing heuristic.

    FinQA values live in tables AND in pre/post narrative text (gold indices
    reference both). The graph tool only covers tabular row values, so queries
    with these signals must ALSO schedule vector retrieval or they will
    silently miss text-only evidence.
    """
    q = query.lower()
    text_signal_words = [
        "why", "reason", "drive", "driven", "cause", "explain", "explanation",
        "discuss", "describe", "according to", "context", "background",
    ]
    numeric_text_words = [
        "stated", "reported", "mentioned", "disclosed", "noted in", "per the filing",
        "as disclosed", "estimate", "expect", "anticipate", "guidance",
    ]
    if any(w in q for w in text_signal_words):
        return True
    if any(w in q for w in numeric_text_words):
        return True
    # Metric without a strong tabular anchor (e.g. "percent of", "amount of X
    # used for") — likely computed from text-extracted values.
    if metric_hint is None:
        return True
    return False


def decompose_query(query: str, company: str = "UNKNOWN") -> DecompositionOutput:
    clean_q = query.strip()
    years = [int(y) for y in YEAR_RE.findall(clean_q)]
    years_sorted = sorted(set(years))
    metric_hint = _extract_metric_hint(clean_q)

    sub_tasks: list[SubTask] = []
    is_multi_hop = False

    # Pattern A: Difference / YoY change between two fiscal years
    if len(years_sorted) >= 2 and any(w in clean_q.lower() for w in ["change", "difference", "increase", "decrease", "growth"]):
        is_multi_hop = True
        y1, y2 = years_sorted[0], years_sorted[1]
        needs_narrative = _needs_narrative_evidence(clean_q, metric_hint)
        plan = f"Retrieve {metric_hint or 'metric'} for {y1} and {y2} in parallel, then calculate difference/growth."
        if needs_narrative:
            plan += " Narrative evidence scheduled as backup for text-only values."

        t1 = SubTask(
            task_id="task_1",
            target_tool="graph_retrieval",
            query_payload={"company_identifier": company, "year": y1, "metric_name": metric_hint},
            dependencies=[],
        )
        t2 = SubTask(
            task_id="task_2",
            target_tool="graph_retrieval",
            query_payload={"company_identifier": company, "year": y2, "metric_name": metric_hint},
            dependencies=[],
        )
        # Concrete values are injected into expression prior to calling safe_math
        t3 = SubTask(
            task_id="task_3",
            target_tool="safe_math",
            query_payload={"expression": "subtract(task_2.amount, task_1.amount)"},
            dependencies=["task_1", "task_2"],
        )
        sub_tasks.extend([t1, t2, t3])
        if needs_narrative:
            t4 = SubTask(
                task_id="task_4",
                target_tool="vector_retrieval",
                query_payload={"query_text": clean_q, "top_k": 5},
                dependencies=[],
            )
            sub_tasks.append(t4)

    # Pattern B: Explanation query requiring narrative and tabular evidence
    elif any(w in clean_q.lower() for w in ["why", "reason", "drive", "driven", "cause"]):
        is_multi_hop = True
        plan = "Retrieve structured numbers from graph and explanatory commentary from vector chunks in parallel."

        t1 = SubTask(
            task_id="task_1",
            target_tool="graph_retrieval",
            query_payload={"company_identifier": company, "year": years_sorted[0] if years_sorted else None, "metric_name": metric_hint},
            dependencies=[],
        )
        t2 = SubTask(
            task_id="task_2",
            target_tool="vector_retrieval",
            query_payload={"query_text": clean_q, "top_k": 5},
            dependencies=[],
        )
        sub_tasks.extend([t1, t2])

    # Default: Single-hop graph lookup
    else:
        needs_narrative = _needs_narrative_evidence(clean_q, metric_hint)
        plan = (
            "Direct single-hop graph query plus narrative context lookup."
            if needs_narrative
            else "Direct single-hop graph query."
        )
        t1 = SubTask(
            task_id="task_1",
            target_tool="graph_retrieval",
            query_payload={"company_identifier": company, "year": years_sorted[0] if years_sorted else None, "metric_name": metric_hint},
            dependencies=[],
        )
        sub_tasks.append(t1)
        if needs_narrative:
            # FinQA gold evidence frequently lives in pre/post text; a pure
            # graph lookup would silently miss it. Fan out to the vector store.
            t2 = SubTask(
                task_id="task_2",
                target_tool="vector_retrieval",
                query_payload={"query_text": clean_q, "top_k": 5},
                dependencies=[],
            )
            sub_tasks.append(t2)

    return DecompositionOutput(
        original_query=clean_q,
        is_multi_hop=is_multi_hop,
        sub_tasks=sub_tasks,
        reasoning_plan=plan,
    )


class DecompositionInput(BaseModel):
    query: str = Field(..., description="User financial question")
    company: str = Field(default="UNKNOWN", description="Target company ticker")


def query_decomposition_tool(payload: DecompositionInput) -> ToolResult[DecompositionOutput]:
    """Instrumented tool entrypoint for query decomposition."""
    return ToolResult.execute_instrumented(
        tool_name="query_decomposition",
        fn=decompose_query,
        query=payload.query,
        company=payload.company,
    )