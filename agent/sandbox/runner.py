"""Hardened sandbox runner.

Executes untrusted Python scripts within scrubbed environments, strictly bounded
by timeouts, memory ceilings, and isolated directory scopes.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agent.sandbox.prefilter import audit_code_safety
from agent.sandbox.protocol import ArtifactManifest, SandboxExecutionInput, SandboxOutput

# Optional POSIX resource limits
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]


# Minimal environment variables permitted inside the guest runner
SAFE_ENV_KEYS: set[str] = {
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
    "PYTHONPATH",
    "PYTHONHOME",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
}

RUNNER_WRAPPER_TEMPLATE = """\
import json
import sys
from pathlib import Path

# Load input variables
input_path = Path("input_data.json")
if input_path.exists():
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
else:
    input_data = {}

# User Script Execution Scope
globals_dict = {"input_data": input_data, "__name__": "__main__"}
with open("user_script.py", "r", encoding="utf-8") as f:
    code_content = f.read()

exec(code_content, globals_dict)
"""


class SubprocessSandboxRunner:
    """Executes sandboxed Python scripts in scrubbed, temporary subprocesses."""

    @staticmethod
    def _build_sanitized_environment() -> dict[str, str]:
        """Constructs a scrubbed environment stripped of all host database/API credentials."""
        scrubbed = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            if (
                k_upper in SAFE_ENV_KEYS
                and not any(
                    s in k_upper
                    for s in ["KEY", "TOKEN", "SECRET", "PASSWORD", "NEO4J", "POSTGRES", "REDIS"]
                )
            ):
                scrubbed[k] = v

        scrubbed["PYTHONDONTWRITEBYTECODE"] = "1"
        scrubbed["PYTHONUNBUFFERED"] = "1"
        return scrubbed

    @staticmethod
    def _set_posix_limits(memory_limit_mb: int) -> None:
        """Applies kernel cgroup/rlimit memory bounds on POSIX platforms."""
        if resource is None:
            return
        bytes_limit = memory_limit_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except (ValueError, OSError):
            pass

    def execute(self, payload: SandboxExecutionInput) -> SandboxOutput:
        # 1. Static AST Security Pre-Filter
        audit_code_safety(payload.code)

        # 2. Ephemeral Scratchpad Directory
        scratch_dir = Path(tempfile.mkdtemp(prefix="finagent_sandbox_"))
        script_file = scratch_dir / "user_script.py"
        input_file = scratch_dir / "input_data.json"
        wrapper_file = scratch_dir / "runner_wrapper.py"
        artifacts_dir = scratch_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            script_file.write_text(payload.code, encoding="utf-8")
            input_file.write_text(json.dumps(payload.input_variables), encoding="utf-8")
            wrapper_file.write_text(RUNNER_WRAPPER_TEMPLATE, encoding="utf-8")

            sanitized_env = self._build_sanitized_environment()
            preexec = (
                (lambda: self._set_posix_limits(payload.memory_limit_mb))
                if os.name != "nt" and resource is not None
                else None
            )

            start_time = time.perf_counter()
            timed_out = False
            oom_killed = False

            try:
                proc = subprocess.Popen(
                    [sys.executable, str(wrapper_file.name)],
                    cwd=str(scratch_dir),
                    env=sanitized_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=preexec,
                )
                stdout, stderr = proc.communicate(timeout=payload.timeout_seconds)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                exit_code = -9
                timed_out = True

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Detect Memory Limit Kill on POSIX
            if exit_code in (-9, 137) and not timed_out:
                oom_killed = True
                stderr += "\n[Sandbox Error] Process killed: memory limit exceeded (OOM)."

            # Harvest Generated Artifacts
            artifacts: list[ArtifactManifest] = []
            for file_path in artifacts_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_size <= 5 * 1024 * 1024:  # 5MB cap
                    data_bytes = file_path.read_bytes()
                    artifacts.append(
                        ArtifactManifest(
                            filename=file_path.name,
                            content_type="application/octet-stream",
                            size_bytes=len(data_bytes),
                            data_base64=base64.b64encode(data_bytes).decode("ascii"),
                        )
                    )

            output = SandboxOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_ms=round(duration_ms, 3),
                artifacts=artifacts,
                timed_out=timed_out,
                oom_killed=oom_killed,
            )
            output.parsed_result = output.extract_json_result()
            return output

        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)


class DockerSandboxRunner:
    """Containerized sandbox executing code inside an air-gapped Docker container."""

    def __init__(self, image: str = "python:3.12-slim") -> None:
        self.image = image
        self.fallback = SubprocessSandboxRunner()

    def _is_docker_available(self) -> bool:
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(self, payload: SandboxExecutionInput) -> SandboxOutput:
        # Fall back to hardened subprocess if Docker daemon is not running
        if not self._is_docker_available():
            return self.fallback.execute(payload)

        # 1. Static AST Security Pre-Filter
        audit_code_safety(payload.code)

        scratch_dir = Path(tempfile.mkdtemp(prefix="finagent_docker_sandbox_"))
        script_file = scratch_dir / "user_script.py"
        input_file = scratch_dir / "input_data.json"
        wrapper_file = scratch_dir / "runner_wrapper.py"

        try:
            script_file.write_text(payload.code, encoding="utf-8")
            input_file.write_text(json.dumps(payload.input_variables), encoding="utf-8")
            wrapper_file.write_text(RUNNER_WRAPPER_TEMPLATE, encoding="utf-8")

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",  # Air-gapped network namespace
                "--memory",
                f"{payload.memory_limit_mb}m",
                "--cpus",
                "1.0",
                "--pids-limit",
                "64",
                "-v",
                f"{scratch_dir}:/sandbox:ro",
                "-w",
                "/sandbox",
                self.image,
                "python",
                "runner_wrapper.py",
            ]

            start_time = time.perf_counter()
            timed_out = False
            try:
                proc = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=payload.timeout_seconds,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "[Sandbox Error] Docker container execution timed out."
                exit_code = -9
                timed_out = True

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            output = SandboxOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_ms=round(duration_ms, 3),
                timed_out=timed_out,
                oom_killed=(exit_code == 137 and not timed_out),
            )
            output.parsed_result = output.extract_json_result()
            return output

        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)