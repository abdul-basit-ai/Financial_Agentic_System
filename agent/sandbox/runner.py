from __future__ import annotations

from agent.sandbox.protocol import SandboxExecutionInput, SandboxOutput


class SubprocessSandboxRunner:
    def execute(self, payload: SandboxExecutionInput) -> SandboxOutput:
        return SandboxOutput(
            exit_code=126,
            stderr="Host Python execution is disabled. Use evidence-linked financial arithmetic.",
        )


class DockerSandboxRunner:
    def __init__(self, image: str = "finagent-sandbox:disabled") -> None:
        self.image = image

    def _is_docker_available(self) -> bool:
        return False

    def execute(self, payload: SandboxExecutionInput) -> SandboxOutput:
        return SandboxOutput(
            exit_code=126,
            stderr="Arbitrary Python execution is deferred. No subprocess or Docker fallback is permitted.",
        )
