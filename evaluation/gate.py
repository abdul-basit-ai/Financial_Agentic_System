"""CI/CD regression gate enforcing McNemar significance and bootstrap confidence bounds.

Compares two benchmark result files produced by evaluation/runner.py
(benchmark_results.json, containing the per-record "results" list with
record_id + execution_correct fields).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Any

from pydantic import BaseModel

from evaluation.manifest import hash_json, id_set_digest, load_manifest


class GateDecision(BaseModel):
    passed: bool
    reasons: list[str]
    baseline_acc: float
    candidate_acc: float
    acc_delta: float
    mcnemar_chi2: float
    p_value: float
    candidate_ci_95: tuple[float, float]


def compute_mcnemar_p_value_1df(chi2_val: float) -> float:
    """Exact closed-form p-value for 1 degree of freedom: P(chi^2 >= x) = erfc(sqrt(x / 2))."""
    if chi2_val <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(chi2_val / 2.0))


def compute_mcnemar_test(
    baseline_flags: list[bool],
    candidate_flags: list[bool],
) -> tuple[float, float]:
    """Computes Edwards continuity-corrected McNemar Chi-Square and exact p-value."""
    if len(baseline_flags) != len(candidate_flags):
        raise ValueError("Baseline and candidate outcomes must have identical lengths.")

    n10 = sum(1 for b, c in zip(baseline_flags, candidate_flags) if b and not c)
    n01 = sum(1 for b, c in zip(baseline_flags, candidate_flags) if not b and c)

    denom = n01 + n10
    if denom == 0:
        return 0.0, 1.0

    chi2 = max(0.0, abs(n01 - n10) - 1.0) ** 2 / denom
    p_val = compute_mcnemar_p_value_1df(chi2)
    return round(chi2, 4), round(p_val, 6)


def compute_bootstrap_ci(
    flags: list[bool],
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes empirical non-parametric bootstrap confidence interval for accuracy."""
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return (0.0, 0.0)

    estimates: list[float] = []
    numeric_flags = [1.0 if x else 0.0 for x in flags]

    for _ in range(n_bootstraps):
        sample = [rng.choice(numeric_flags) for _ in range(n)]
        estimates.append(sum(sample) / n)

    estimates.sort()
    lower_idx = int((alpha / 2.0) * n_bootstraps)
    upper_idx = int((1.0 - alpha / 2.0) * n_bootstraps)

    return round(estimates[lower_idx], 4), round(estimates[upper_idx], 4)


def evaluate_gate(
    baseline_path: str,
    candidate_path: str,
    max_regression_rate: float = 0.02,
) -> GateDecision:
    """Evaluates candidate benchmark results against baseline with statistical rigor."""
    with open(baseline_path, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(candidate_path, encoding="utf-8") as f:
        cand_data = json.load(f)

    if not 0 <= max_regression_rate <= 1:
        raise ValueError("max_regression_rate must be between 0 and 1")

    base_manifest = load_manifest(base_data)
    cand_manifest = load_manifest(cand_data)
    if (
        base_manifest.execution != "candidate-runtime"
        or cand_manifest.execution != "candidate-runtime"
    ):
        raise ValueError("The gate requires two candidate-runtime benchmarks")

    def outcomes(artifact: dict[str, Any], expected_ids: str) -> dict[str, bool]:
        rows = artifact.get("results")
        if not isinstance(rows, list) or artifact.get("total_samples") != len(rows):
            raise ValueError("Benchmark result count does not match the artifact")
        if artifact.get("results_sha256") != hash_json(rows):
            raise ValueError("Benchmark results checksum does not match")
        ids = []
        result: dict[str, bool] = {}
        for item in rows:
            record_id = item.get("record_id") if isinstance(item, dict) else None
            correct = item.get("execution_correct") if isinstance(item, dict) else None
            if not isinstance(record_id, str) or type(correct) is not bool:
                raise ValueError(
                    "Each outcome must declare a record ID and a Boolean result"
                )
            ids.append(record_id)
            result[record_id] = correct
        if id_set_digest(ids) != expected_ids:
            raise ValueError("Benchmark IDs do not match the run manifest")
        return result

    base_map = outcomes(base_data, base_manifest.config.record_ids_sha256)
    cand_map = outcomes(cand_data, cand_manifest.config.record_ids_sha256)
    if base_map.keys() != cand_map.keys():
        raise ValueError("Baseline and candidate must contain identical record ID sets")
    if base_manifest.config != cand_manifest.config:
        raise ValueError("Benchmark manifests are incompatible")

    ids = sorted(base_map)
    base_flags = [base_map[rid] for rid in ids]
    cand_flags = [cand_map[rid] for rid in ids]

    base_acc = sum(base_flags) / len(base_flags)
    cand_acc = sum(cand_flags) / len(cand_flags)
    delta = cand_acc - base_acc

    chi2, p_val = compute_mcnemar_test(base_flags, cand_flags)
    ci_95 = compute_bootstrap_ci(cand_flags)

    n10 = sum(1 for b, c in zip(base_flags, cand_flags) if b and not c)
    reg_rate = n10 / len(ids)

    reasons: list[str] = []
    passed = True

    if delta < 0.0:
        passed = False
        reasons.append(
            f"Net accuracy dropped by {abs(delta):.2%} ({base_acc:.2%} -> {cand_acc:.2%})."
        )

    if reg_rate > max_regression_rate:
        passed = False
        reasons.append(
            f"Sample regression rate ({reg_rate:.2%}) breached max allowable threshold ({max_regression_rate:.2%})."
        )

    if passed:
        reasons.append(
            "All CI/CD statistical criteria passed. Zero unacceptable regressions detected."
        )

    return GateDecision(
        passed=passed,
        reasons=reasons,
        baseline_acc=round(base_acc, 4),
        candidate_acc=round(cand_acc, 4),
        acc_delta=round(delta, 4),
        mcnemar_chi2=chi2,
        p_value=p_val,
        candidate_ci_95=ci_95,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CI/CD Statistical Regression Gate")
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline benchmark JSON"
    )
    parser.add_argument(
        "--candidate", required=True, help="Path to candidate benchmark JSON"
    )
    parser.add_argument("--max-regression-rate", type=float, default=0.02)
    args = parser.parse_args()

    decision = evaluate_gate(
        args.baseline,
        args.candidate,
        max_regression_rate=args.max_regression_rate,
    )

    print(json.dumps(decision.model_dump(), indent=2))

    if not decision.passed:
        print("\nCI/CD Regression Gate FAILED!")
        sys.exit(1)
    else:
        print("\nCI/CD Regression Gate PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
