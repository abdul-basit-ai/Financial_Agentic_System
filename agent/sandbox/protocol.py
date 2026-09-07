"""Data contracts and schemas for the code execution sandbox."""

from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class SandboxExecutionInput(BaseModel):
    """Input payload for sandbox script execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str = Field(..., description="Python source code to execute in the sandbox")
    input_variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured variables passed into the script via input_data.json",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum execution wall-clock time before pre-emptive SIGKILL",
    )
    memory_limit_mb: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Memory ceiling in megabytes",
    )


class ArtifactManifest(BaseModel):
    """Metadata for generated file artifacts (e.g. CSVs, plots, matrices)."""

    filename: str
    content_type: str
    size_bytes: int
    data_base64: str | None = None


class SandboxOutput(BaseModel):
    """Execution telemetry and output payload from the sandbox."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error")
    exit_code: int = Field(default=0, description="Process termination return code")
    execution_time_ms: float = Field(default=0.0, description="Measured wall-clock latency")
    parsed_result: Any | None = Field(
        default=None,
        description="Parsed JSON payload if stdout ends with a valid JSON document",
    )
    artifacts: list[ArtifactManifest] = Field(
        default_factory=list, description="Output files generated in the scratch space"
    )
    timed_out: bool = Field(default=False, description="Whether execution was aborted by timeout")
    oom_killed: bool = Field(
        default=False, description="Whether process was terminated due to memory exhaustion"
    )

    def extract_json_result(self) -> Any | None:
        """Attempts to parse the last non-empty line of stdout as a structured JSON object."""
        lines = [line.strip() for line in self.stdout.strip().splitlines() if line.strip()]
        if not lines:
            return None
        for line in reversed(lines):
            try:
                return json.loads(line)
            except Exception:
                continue
        return None