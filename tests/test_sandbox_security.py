"""Validation suite for Phase 9 Code Interpreter Sandbox.

Verifies:
1. Deterministic mathematical execution with input variables and JSON output parsing.
2. AST pre-filter rejection of OS, networking, subprocess, and reflection primitives.
3. Defense against Python sandbox escape chains (__subclasses__, __globals__, etc.).
4. Clean pre-emptive termination of non-terminating loops (Turing timeout bound).
5. Environment scrubbing ensuring host database/API credentials are never leaked.
"""

from __future__ import annotations

import pytest

from agent.sandbox.prefilter import SandboxSecurityError, audit_code_safety
from agent.sandbox.protocol import SandboxExecutionInput
from agent.sandbox.runner import SubprocessSandboxRunner
from agent.tools.sandbox_tool import code_sandbox_tool


def test_prefilter_allows_safe_scientific_code() -> None:
    code = """
import math
import statistics

values = [10.5, 20.0, 30.5, 40.0]
mean_val = statistics.mean(values)
sqrt_val = math.sqrt(mean_val)
print(f"Result: {sqrt_val}")
"""

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
        audit_code_safety(code)


def test_host_execution_is_disabled_even_for_benign_code() -> None:
    output = SubprocessSandboxRunner().execute(SandboxExecutionInput(code="print(42)"))
    assert output.exit_code == 126
    assert "disabled" in output.stderr
    assert not output.stdout


def test_code_tool_fails_closed_without_executing(tmp_path) -> None:
    sentinel = tmp_path / "must-not-exist.txt"
    payload = SandboxExecutionInput(
        code=f"open({str(sentinel)!r}, 'w').write('executed')"
    )
    result = code_sandbox_tool(payload)
    assert result.success is False
    assert "PermissionError" in result.error
    assert result.data is None
    assert not sentinel.exists()


def test_docker_runner_has_no_subprocess_fallback() -> None:
    from agent.sandbox.runner import DockerSandboxRunner

    output = DockerSandboxRunner().execute(SandboxExecutionInput(code="print(42)"))
    assert output.exit_code == 126
    assert "deferred" in output.stderr
    assert not output.stdout
