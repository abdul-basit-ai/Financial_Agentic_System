"""Validation suite for Phase 9 Code Interpreter Sandbox.

Verifies:
1. Deterministic mathematical execution with input variables and JSON output parsing.
2. AST pre-filter rejection of OS, networking, subprocess, and reflection primitives.
3. Defense against Python sandbox escape chains (__subclasses__, __globals__, etc.).
4. Clean pre-emptive termination of non-terminating loops (Turing timeout bound).
5. Environment scrubbing ensuring host database/API credentials are never leaked.
"""

from __future__ import annotations

import os
import pytest

from agent.sandbox.prefilter import SandboxSecurityError, audit_code_safety
from agent.sandbox.protocol import SandboxExecutionInput
from agent.sandbox.runner import SubprocessSandboxRunner
from agent.tools.sandbox_tool import code_sandbox_tool


# =====================================================================
# AST Pre-Filter Security Tests
# =====================================================================


def test_prefilter_allows_safe_scientific_code() -> None:
    code = """
import math
import statistics

values = [10.5, 20.0, 30.5, 40.0]
mean_val = statistics.mean(values)
sqrt_val = math.sqrt(mean_val)
print(f"Result: {sqrt_val}")
"""
    # Must parse without exception
    audit_code_safety(code)


def test_prefilter_blocks_forbidden_imports() -> None:
    forbidden_snippets = [
        "import os\nos.listdir('.')",
        "import sys\nsys.exit(0)",
        "import subprocess\nsubprocess.run(['ls'])",
        "import socket\nsocket.socket()",
        "import urllib.request\nurllib.request.urlopen('http://evil.com')",
        "from pathlib import Path\nPath('/etc').exists()",
        "from shutil import rmtree\nrmtree('/')",
    ]

    for snippet in forbidden_snippets:
        with pytest.raises(SandboxSecurityError) as exc_info:
            audit_code_safety(snippet)
        assert "unauthorized module" in str(exc_info.value).lower()


def test_prefilter_blocks_builtin_eval_and_exec() -> None:
    forbidden_calls = [
        "eval('2 + 2')",
        "exec('import os')",
        "open('/etc/passwd', 'r')",
        "__import__('os').system('whoami')",
        "getattr(math, 'sqrt')(16)",
    ]

    for snippet in forbidden_calls:
        with pytest.raises(SandboxSecurityError) as exc_info:
            audit_code_safety(snippet)
        assert "prohibited" in str(exc_info.value).lower()


def test_prefilter_blocks_dunder_escape_vectors() -> None:
    escape_snippets = [
        "cls = ().__class__.__bases__[0].__subclasses__()",
        "func = lambda: None\nglob = func.__globals__",
        "c = (1).__class__.__mro__",
    ]

    for snippet in escape_snippets:
        with pytest.raises(SandboxSecurityError) as exc_info:
            audit_code_safety(snippet)
        assert "sandbox escape defense" in str(exc_info.value).lower()


def test_prefilter_blocks_forbidden_builtin_aliasing() -> None:
    """Assigning a forbidden builtin to a variable is an escape primer
    (e.g. `e = eval` then `e('__import__("os")')`) and must be rejected
    outright, even before any call site appears."""
    alias_snippets = [
        "e = eval",
        "a = eval\nb = a",
        "x = exec",
        "opener = open",
        "imp = __import__",
        "e = eval\nr = e('1+1')",
    ]
    for snippet in alias_snippets:
        with pytest.raises(SandboxSecurityError) as exc_info:
            audit_code_safety(snippet)
        assert "alias escape defense" in str(exc_info.value).lower()


def test_prefilter_allows_safe_code_after_alias_rules() -> None:
    """Legitimate scientific code must still pass with alias defenses active."""
    safe_codes = [
        "import math\nx = math.sqrt(16)\nprint(x)",
        "import statistics\nm = statistics.mean([1, 2, 3])",
        "import json\nd = json.loads('{\"a\": 1}')",
        "val = 42\nprint(val * 2)",
    ]
    for code in safe_codes:
        audit_code_safety(code)  # must not raise


# =====================================================================
# Subprocess Sandbox Execution Tests
# =====================================================================


def test_sandbox_computes_financial_metrics_with_input_data() -> None:
    runner = SubprocessSandboxRunner()
    code = """
import json

revenues = input_data.get("revenues", [])
base_rev = revenues[0]
latest_rev = revenues[-1]

cagr = ((latest_rev / base_rev) ** (1.0 / (len(revenues) - 1)) - 1.0) * 100.0

output = {
    "cagr_percent": round(cagr, 2),
    "periods": len(revenues)
}
print(json.dumps(output))
"""
    payload = SandboxExecutionInput(
        code=code,
        input_variables={"revenues": [100.0, 115.0, 130.0, 148.0]},
        timeout_seconds=5.0,
    )

    output = runner.execute(payload)
    assert output.exit_code == 0
    assert output.timed_out is False
    assert output.parsed_result is not None
    assert output.parsed_result["cagr_percent"] == 13.96
    assert output.parsed_result["periods"] == 4


def test_sandbox_enforces_strict_timeout() -> None:
    runner = SubprocessSandboxRunner()
    infinite_loop_code = """
import time
while True:
    time.sleep(0.1)
"""
    payload = SandboxExecutionInput(
        code=infinite_loop_code,
        timeout_seconds=1.5,
    )

    output = runner.execute(payload)
    assert output.timed_out is True
    assert output.exit_code == -9
    assert output.execution_time_ms >= 1400.0


def test_sandbox_scrubs_environment_secrets() -> None:
    """Verifies that parent environment credentials are not leaked into the
    guest process environment. The guest cannot import os (prefilter blocks
    it), so we validate the sanitized env dict the runner constructs directly."""
    runner = SubprocessSandboxRunner()
    # Inject fake sensitive secrets into current host process
    os.environ["NEO4J_PASSWORD"] = "ultra_secret_neo4j_pwd"
    os.environ["POSTGRES_PASSWORD"] = "ultra_secret_pg_pwd"
    os.environ["OPENAI_API_KEY"] = "sk-test-redacted"

    sanitized = runner._build_sanitized_environment()
    passed_keys = set(sanitized.keys())

    assert "NEO4J_PASSWORD" not in passed_keys
    assert "POSTGRES_PASSWORD" not in passed_keys
    assert "OPENAI_API_KEY" not in passed_keys
    # Belt-and-suspenders: no secret-looking key survives at all
    for k in passed_keys:
        assert not any(
            s in k.upper()
            for s in ["KEY", "TOKEN", "SECRET", "PASSWORD", "NEO4J", "POSTGRES", "REDIS"]
        ), f"Sensitive env var '{k}' leaked into sandbox environment"


# =====================================================================
# Tool Envelope & MCP Schema Export Tests
# =====================================================================


def test_code_sandbox_tool_instrumentation() -> None:
    code = "import json\nprint(json.dumps({'status': 'ok'}))"
    payload = SandboxExecutionInput(code=code, timeout_seconds=5.0)

    result = code_sandbox_tool(payload)
    assert result.success is True
    assert result.data is not None
    assert result.data.parsed_result == {"status": "ok"}
    assert result.metrics.latency_ms > 0.0

    # Schema Exportability for MCP (Phase 10)
    schema = SandboxExecutionInput.model_json_schema()
    assert "properties" in schema
    assert "code" in schema["properties"]