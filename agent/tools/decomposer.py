"""Query decomposition tool for multi-hop financial reasoning.

Plan contract (record anchoring): every plan schedules a vector_retrieval
task FIRST, scoped to the company's filings by the worker. Graph tasks
declare a dependency on that vector task so the fan-out can inject the
anchored record_id (the filing the question is actually about) into their
payloads before dispatch. safe_math depends on the graph tasks and
references their values as task_N.amount.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import ToolResult

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# "average payment volume per transaction" / "revenue per employee" /
# "operating margin as a percentage of sales" — a ratio between two column
# or line-item phrases named IN the question itself.
PER_RATIO_RE = re.compile(
    r"\b(?:average\s+|avg\s+)?([a-z][a-z \-]{2,40}?)\s+(?:per|/\s*|as a percentage of|as a % of)\s+([a-z][a-z \-]{2,40})",
    re.IGNORECASE,
)
GROWTH_WORDS = ("growth", "rate", "percent", "%", "percentage", "ratio")
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
    target_tool: Literal["graph_retrieval", "vector_retrieval", "safe_math"] = Field(
        ...
    )
    query_payload: dict = Field(
        ..., description="Parameters to pass directly to target tool"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of tasks that must resolve first"
    )


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


def _vector_anchor_task(company: str, query: str) -> SubTask:
    """The first task of every plan: scoped semantic search that (a) anchors
    graph retrieval to the right filing via its top chunk's record_id and
    (b) provides narrative evidence for driver-style questions."""
    return SubTask(
        task_id="task_1",
        target_tool="vector_retrieval",
        query_payload={"query_text": query, "top_k": 5, "company_identifier": company},
        dependencies=[],
    )


_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:what|how|which|where)\s+(?:is|was|are|were|did|does|do|has|have)?\s*",
    re.IGNORECASE,
)
_LEADING_NOISE_RE = re.compile(r"^(?:the|a|an|average|avg)\s+", re.IGNORECASE)
# Connectors end a metric phrase: "transaction FOR american express" is not
# the denominator the question names.
_PHRASE_CUT_RE = re.compile(
    r"\s+\b(?:for|of|in|at|during|by|from|to|and)\b\s+.*$", re.IGNORECASE
)


def _per_ratio_parts(query: str) -> tuple[str, str] | None:
    """Extracts cleaned (numerator, denominator) phrases from a per-ratio
    question, or None when the pattern doesn't apply."""
    m = PER_RATIO_RE.search(_QUESTION_PREFIX_RE.sub("", query))
    if not m:
        return None

    def _clean(phrase: str) -> str:
        text = _PHRASE_CUT_RE.sub("", phrase)
        while True:
            stripped = _LEADING_NOISE_RE.sub("", text).strip(" -")
            if stripped == text:
                break
            text = stripped
        return re.sub(r"\s+", " ", text)

    numerator, denominator = _clean(m.group(1)), _clean(m.group(2))
    if len(numerator) < 3 or len(denominator) < 3:
        return None
    return numerator, denominator


def decompose_query(query: str, company: str = "UNKNOWN") -> DecompositionOutput:
    clean_q = query.strip()
    years = [int(y) for y in YEAR_RE.findall(clean_q)]
    years_sorted = sorted(set(years))
    metric_hint = _extract_metric_hint(clean_q)
    low_q = clean_q.lower()

    sub_tasks: list[SubTask] = [_vector_anchor_task(company, clean_q)]
    is_multi_hop = False

    # Pattern P: per-ratio ("average payment volume per transaction") — two
    # operands named in the question, quotient is the answer. This ratio
    # family is unreachable by the keyword list; the operands come straight
    # from the question text and the graph tool's word-level matching binds
    # them to row labels / column headers.
    per_parts = _per_ratio_parts(clean_q)
    if per_parts:
        numerator, denominator = per_parts
        is_multi_hop = True
        plan = (
            f"Anchor to the right {company} filing, retrieve '{numerator}' and "
            f"'{denominator}', then divide."
        )
        year = years_sorted[0] if years_sorted else None
        t2 = SubTask(
            task_id="task_2",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": year,
                "metric_name": numerator,
            },
            dependencies=["task_1"],
        )
        t3 = SubTask(
            task_id="task_3",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": year,
                "metric_name": denominator,
            },
            dependencies=["task_1"],
        )
        t4 = SubTask(
            task_id="task_4",
            target_tool="safe_math",
            query_payload={"expression": "divide(task_2.amount, task_3.amount)"},
            dependencies=["task_2", "task_3"],
        )
        sub_tasks.extend([t2, t3, t4])
        return DecompositionOutput(
            original_query=clean_q,
            is_multi_hop=is_multi_hop,
            sub_tasks=sub_tasks,
            reasoning_plan=plan,
        )

    # Pattern A: Difference / YoY change between two fiscal years.
    # Growth-rate phrasing requires a RELATIVE change: divide the difference
    # by the base year. Plain subtract systematically mis-answered every
    # growth/ratio question even with perfect retrieval.
    if len(years_sorted) >= 2 and any(
        w in low_q for w in ["change", "difference", "increase", "decrease", "growth"]
    ):
        is_multi_hop = True
        y1, y2 = years_sorted[0], years_sorted[1]
        relative = any(w in low_q for w in GROWTH_WORDS)
        plan = (
            f"Anchor to the right {company} filing, retrieve {metric_hint or 'metric'} for "
            f"{y1} and {y2}, then calculate the "
            f"{'percentage change' if relative else 'change'}."
        )

        t2 = SubTask(
            task_id="task_2",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": y1,
                "metric_name": metric_hint,
            },
            dependencies=["task_1"],
        )
        t3 = SubTask(
            task_id="task_3",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": y2,
                "metric_name": metric_hint,
            },
            dependencies=["task_1"],
        )
        # Concrete values are injected into the expression prior to execution
        expression = (
            "divide(subtract(task_3.amount, task_2.amount), task_2.amount)"
            if relative
            else "subtract(task_3.amount, task_2.amount)"
        )
        t4 = SubTask(
            task_id="task_4",
            target_tool="safe_math",
            query_payload={"expression": expression},
            dependencies=["task_2", "task_3"],
        )
        sub_tasks.extend([t2, t3, t4])

    # Pattern B: Explanation query requiring narrative and tabular evidence
    elif any(w in low_q for w in ["why", "reason", "drive", "driven", "cause"]):
        is_multi_hop = True
        plan = "Anchor to the right filing, then retrieve its numbers and explanatory commentary together."

        t2 = SubTask(
            task_id="task_2",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": years_sorted[0] if years_sorted else None,
                "metric_name": metric_hint,
            },
            dependencies=["task_1"],
        )
        sub_tasks.append(t2)

    # Default: Single-hop graph lookup, still anchored
    else:
        plan = f"Anchor to the right {company} filing, then look up the {metric_hint or 'requested'} line items."

        t2 = SubTask(
            task_id="task_2",
            target_tool="graph_retrieval",
            query_payload={
                "company_identifier": company,
                "year": years_sorted[0] if years_sorted else None,
                "metric_name": metric_hint,
            },
            dependencies=["task_1"],
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


def query_decomposition_tool(
    payload: DecompositionInput,
) -> ToolResult[DecompositionOutput]:
    """Instrumented tool entrypoint for query decomposition."""
    return ToolResult.execute_instrumented(
        tool_name="query_decomposition",
        fn=decompose_query,
        query=payload.query,
        company=payload.company,
    )
