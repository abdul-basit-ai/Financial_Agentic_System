"""Phase 4-compliant tool wrapper for dynamic sandbox execution."""

from __future__ import annotations

from typing import Any

from agent.sandbox.protocol import SandboxExecutionInput, SandboxOutput
from agent.sandbox.runner import SubprocessSandboxRunner
from agent.tools.base import ToolResult


def execute_sandboxed_code(payload_dict: dict[str, Any]) -> SandboxOutput:
    """Direct execution entrypoint wrapping the sandbox runner."""
    input_model = SandboxExecutionInput(**payload_dict)
    runner = SubprocessSandboxRunner()
    return runner.execute(input_model)


def code_sandbox_tool(payload: SandboxExecutionInput) -> ToolResult[SandboxOutput]:
    """Instrumented tool entrypoint executing arbitrary Python code safely."""
    return ToolResult.execute_instrumented(
        tool_name="code_interpreter",
        fn=execute_sandboxed_code,
        payload_dict=payload.model_dump(),
    )