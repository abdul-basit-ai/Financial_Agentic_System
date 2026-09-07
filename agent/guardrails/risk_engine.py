"""Deterministic financial risk assessment engine.

Evaluates continuous metrics, Benford's law conformity, statistical MAD Z-scores,
materiality thresholds, and retrieval confidence without external LLM dependencies.
"""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

# Benford's Law theoretical digit frequencies for d in 1..9
BENFORD_EXPECTED: dict[int, float] = {
    d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)
}
# Chi-Square critical threshold for df = 8 at alpha = 0.01
CHI_SQUARE_CRITICAL_99 = 20.09
MATERIAL_DOLLAR_THRESHOLD = 100_000_000.0  # $100M in raw table units
MIN_VECTOR_SIMILARITY_THRESHOLD = 0.65
MAD_Z_SCORE_THRESHOLD = 10.0  # Tight enough for true outliers (z>20 for a 3.5x
# magnitude deviation in a same-magnitude cluster), loose enough that normal
# financial variance within one order of magnitude (z typically < 5) passes.


class RiskTriggerReason(BaseModel):
    category: str = Field(..., description="Category: ANOMALY, MATERIALITY, INTEGRITY, CONFIDENCE")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")
    message: str = Field(..., description="Detailed explanation of the risk trigger")
    metric_value: float | None = Field(default=None, description="Observed numeric value")


class RiskAssessmentResult(BaseModel):
    requires_hitl: bool
    composite_risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[RiskTriggerReason] = Field(default_factory=list)
    inspected_metrics_count: int


def compute_modified_z_scores(values: list[float]) -> list[float]:
    """Computes Modified Z-score using Median Absolute Deviation (MAD)."""
    if len(values) < 3:
        return [0.0] * len(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = (
        sorted_vals[n // 2]
        if n % 2 != 0
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    )

    abs_deviations = sorted(abs(x - median) for x in values)
    mad = (
        abs_deviations[n // 2]
        if n % 2 != 0
        else (abs_deviations[n // 2 - 1] + abs_deviations[n // 2]) / 2.0
    )

    if mad == 0.0:
        return [0.0] * len(values)

    return [round(0.6745 * (x - median) / mad, 4) for x in values]


def verify_benford_law_conformity(numbers: list[float]) -> tuple[bool, float]:
    """Tests first-digit distribution against Benford's Law via Chi-Square goodness-of-fit."""
    first_digits: list[int] = []
    for num in numbers:
        abs_val = abs(num)
        if abs_val > 0.0:
            first_char = str(f"{abs_val:.10e}").lstrip("0.")[0]
            if first_char.isdigit() and int(first_char) != 0:
                first_digits.append(int(first_char))

    # Requires at least 20 numbers for valid chi-square sample size
    if len(first_digits) < 20:
        return True, 0.0

    total_count = len(first_digits)
    observed_counts = {d: first_digits.count(d) for d in range(1, 10)}

    chi_square = 0.0
    for d in range(1, 10):
        expected_count = BENFORD_EXPECTED[d] * total_count
        observed = observed_counts[d]
        chi_square += ((observed - expected_count) ** 2) / expected_count

    is_conformant = chi_square <= CHI_SQUARE_CRITICAL_99
    return is_conformant, round(chi_square, 4)


class FinancialRiskEngine:
    """Evaluates state metrics against financial and governance risk policies."""

    @classmethod
    def evaluate_state(cls, state_dict: dict[str, Any]) -> RiskAssessmentResult:
        reasons: list[RiskTriggerReason] = []
        extracted_numbers: list[float] = []

        # 1. Inspect Completed Tool Results
        tool_results = state_dict.get("tool_results", [])
        sub_task_results = state_dict.get("sub_task_results", {})

        # Collect amounts from graph records
        for res in tool_results:
            data = res.get("data")
            if not isinstance(data, dict):
                continue

            # Graph metric records
            records = data.get("records", [])
            for r in records:
                amt = r.get("amount")
                norm_amt = r.get("normalized_amount")
                row_label = str(r.get("row_label", "")).lower()

                target_val = norm_amt if norm_amt is not None else amt
                if target_val is not None:
                    extracted_numbers.append(float(target_val))

                # Check Extreme Margins (> 100% or < -100%)
                if any(m in row_label for m in ["margin", "rate", "percentage"]) and amt is not None:
                    if abs(float(amt)) > 100.0:
                        reasons.append(
                            RiskTriggerReason(
                                category="ANOMALY",
                                severity="CRITICAL",
                                message=f"Impossible margin detected on '{r.get('row_label')}': {amt}%",
                                metric_value=float(amt),
                            )
                        )

                # Check Material Transactions on RAW amounts (table-native units).
                # normalized_amount multiplies by 1e6/1e9 for 'in millions'
                # filings — using it would flag every large filing as material
                # (false-positive alarm fatigue). Raw table values are the
                # auditor-comparable figure; threshold $100M (raw).
                if amt is not None and abs(float(amt)) >= MATERIAL_DOLLAR_THRESHOLD:
                    reasons.append(
                        RiskTriggerReason(
                            category="MATERIALITY",
                            severity="HIGH",
                            message=f"Material figure exceeding threshold on '{r.get('row_label')}': {float(amt):,.2f} (table units)",
                            metric_value=float(amt),
                        )
                    )

            # Vector chunk confidence checks — one signal per retrieval call,
            # not one per chunk: a single retrieval returning 5 mediocre
            # chunks is one confidence observation, not five escalations.
            chunks = data.get("chunks", [])
            chunk_sims = [
                float(c["similarity_score"])
                for c in chunks
                if c.get("similarity_score") is not None
            ]
            if chunk_sims:
                worst_sim = min(chunk_sims)
                if worst_sim < MIN_VECTOR_SIMILARITY_THRESHOLD:
                    reasons.append(
                        RiskTriggerReason(
                            category="CONFIDENCE",
                            severity="MEDIUM",
                            message=f"Low semantic confidence (worst {worst_sim:.4f} of {len(chunk_sims)} chunks) on retrieved narrative section.",
                            metric_value=worst_sim,
                        )
                    )

            # Math results check (Extreme percentages / Division anomalies)
            math_result = data.get("result")
            expr = str(data.get("expression", ""))
            if math_result is not None:
                extracted_numbers.append(float(math_result))
                if any(w in expr.lower() for w in ["growth", "change", "margin"]) and abs(float(math_result)) > 500.0:
                    reasons.append(
                        RiskTriggerReason(
                            category="ANOMALY",
                            severity="CRITICAL",
                            message=f"Extreme calculated variance/growth exceeding 500%: {math_result:.2f}%",
                            metric_value=float(math_result),
                        )
                    )

        # 2. Statistical Outlier Detection (MAD) — within same-magnitude clusters
        # Financial filings legitimately mix scales (thousands/millions/billions
        # in one table). Raw MAD across mixed magnitudes flags every small line
        # next to a large one (z > 100 observed), which is noise, not anomaly.
        # Cluster values by order of magnitude and detect outliers WITHIN a
        # cluster (needs >= 4 comparable values to be meaningful).
        magnitude_clusters: dict[int, list[float]] = {}
        for num in extracted_numbers:
            if num == 0.0:
                continue
            cluster_key = int(math.floor(math.log10(abs(num))))
            magnitude_clusters.setdefault(cluster_key, []).append(num)

        for cluster_values in magnitude_clusters.values():
            if len(cluster_values) < 4:
                continue
            z_scores = compute_modified_z_scores(cluster_values)
            for val, z in zip(cluster_values, z_scores):
                if abs(z) >= MAD_Z_SCORE_THRESHOLD:
                    reasons.append(
                        RiskTriggerReason(
                            category="ANOMALY",
                            severity="HIGH",
                            message=f"Statistical outlier within same-magnitude cluster (Modified Z-Score: {z:.2f}) on value: {val:,.2f}",
                            metric_value=val,
                        )
                    )

        # 3. Benford's Law Data Integrity Check
        benford_ok, chi_sq = verify_benford_law_conformity(extracted_numbers)
        if not benford_ok:
            reasons.append(
                RiskTriggerReason(
                    category="INTEGRITY",
                    severity="CRITICAL",
                    message=f"Data failed Benford's law conformity (Chi-Square: {chi_sq} > {CHI_SQUARE_CRITICAL_99}). Potential OCR/Table parsing corruption.",
                    metric_value=chi_sq,
                )
            )

        # 4. Composite Risk Score Formulation
        # HIGH/CRITICAL triggers escalate immediately; MEDIUM/LOW reasons are
        # informational UNLESS they accumulate heavily. A couple of mediocre
        # vector matches (2x MEDIUM = 0.7) hitting the composite ceiling would
        # escalate nearly every generic query -> HITL alarm fatigue.
        severity_weights = {"LOW": 0.1, "MEDIUM": 0.35, "HIGH": 0.7, "CRITICAL": 1.0}
        total_risk = 0.0
        for r in reasons:
            total_risk += severity_weights.get(r.severity, 0.2)

        composite_score = min(1.0, total_risk)
        has_escalation_trigger = any(
            r.severity in {"HIGH", "CRITICAL"} for r in reasons
        )
        # MEDIUM-only noise requires 3+ simultaneous reasons to cross 0.7 via
        # the composite; direct composite escalation needs >= 1.0 (saturation).
        requires_hitl = has_escalation_trigger or composite_score >= 1.0

        return RiskAssessmentResult(
            requires_hitl=requires_hitl,
            composite_risk_score=round(composite_score, 4),
            reasons=reasons,
            inspected_metrics_count=len(extracted_numbers),
        )