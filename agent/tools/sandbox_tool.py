"""Phase 4-compliant tool wrapper for dynamic sandbox execution.

Execution preference: Docker container (air-gapped network, hard memory/CPU/
pids limits) with automatic fallback to the hardened subprocess runner when
the Docker daemon is unavailable — containment is never weaker than the
subprocess baseline, and the mode actually used is recorded on the output.
"""

from __future__ import annotations

from typing import Any

from agent.sandbox.protocol import SandboxExecutionInput, SandboxOutput
from agent.sandbox.runner import DockerSandboxRunner, SubprocessSandboxRunner
from agent.tools.base import ToolResult

_DOCKER_RUNNER = DockerSandboxRunner()
_SUBPROCESS_RUNNER = SubprocessSandboxRunner()

ACTIVE_SANDBOX_MODE: str = "unknown"


def execute_sandboxed_code(payload_dict: dict[str, Any]) -> SandboxOutput:
    """Direct execution entrypoint: Docker-isolated when available, hardened
    subprocess otherwise. The mode used is stamped into stderr for tracing."""
    global ACTIVE_SANDBOX_MODE
    input_model = SandboxExecutionInput(**payload_dict)

    if _DOCKER_RUNNER._is_docker_available():
        ACTIVE_SANDBOX_MODE = "docker"
        output = _DOCKER_RUNNER.execute(input_model)
        output.stderr = f"[sandbox] mode=docker\n{output.stderr}"
    else:
        ACTIVE_SANDBOX_MODE = "subprocess_fallback"
        output = _SUBPROCESS_RUNNER.execute(input_model)
        output.stderr = (
            f"[sandbox] mode=subprocess_fallback (docker unavailable)\n{output.stderr}"
        )
    return output


def code_sandbox_tool(payload: SandboxExecutionInput) -> ToolResult[SandboxOutput]:
    """Instrumented tool entrypoint executing arbitrary Python code safely."""
    return ToolResult.execute_instrumented(
        tool_name="code_interpreter",
        fn=execute_sandboxed_code,
        payload_dict=payload.model_dump(),
    )