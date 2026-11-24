from __future__ import annotations

from typing import Any

from agent.sandbox.protocol import SandboxExecutionInput, SandboxOutput
from agent.tools.base import ToolResult

ACTIVE_SANDBOX_MODE = "disabled"


def execute_sandboxed_code(payload_dict: dict[str, Any]) -> SandboxOutput:
    raise PermissionError(
        "Arbitrary Python execution is deferred. Use evidence-linked arithmetic."
    )


def code_sandbox_tool(payload: SandboxExecutionInput) -> ToolResult[SandboxOutput]:
    return ToolResult.execute_instrumented(
        tool_name="code_interpreter",
        fn=execute_sandboxed_code,
        payload_dict=payload.model_dump(),
    )
