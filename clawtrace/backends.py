"""Shared backend detection and resolution for coding-agent CLIs.

Used by both the scoring pipeline and PII review to auto-detect whether
clawtrace is running under Claude Code, Codex, or OpenClaw and dispatch
to the corresponding automation CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SUPPORTED_BACKENDS = ("claude", "codex", "openclaw")
BACKEND_CHOICES = ("auto", *SUPPORTED_BACKENDS)
BACKEND_COMMANDS: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "openclaw": "openclaw",
}
BACKEND_ENV_MARKERS: dict[str, tuple[str, ...]] = {
    "claude": ("CLAUDECODE", "CLAUDE_CODE", "CLAUDECODE_SESSION_ID", "CLAUDE_PROJECT_DIR"),
    "codex": ("CODEX_THREAD_ID", "CODEX_SANDBOX", "CODEX_CI"),
    "openclaw": ("OPENCLAW_HOME", "OPENCLAW_STATE_DIR", "OPENCLAW_CONFIG_PATH"),
}
BACKEND_COMMAND_ALIASES: dict[str, tuple[str, ...]] = {
    "claude": ("claude",),
    "codex": ("codex",),
    "openclaw": ("openclaw",),
}


def _detect_current_agent_from_env(env: dict[str, str] | None = None) -> str | None:
    """Infer the current agent from the process environment."""
    env = os.environ if env is None else env
    for backend, keys in BACKEND_ENV_MARKERS.items():
        for key in keys:
            if env.get(key):
                return backend
    return None


def _get_process_field(pid: int, field: str) -> str:
    """Read a single process field from ps, returning an empty string on failure."""
    try:
        proc = subprocess.run(
            ["ps", f"-o{field}=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _classify_process_command(comm: str, command: str) -> str | None:
    """Map a process command to a supported backend."""
    fields = " ".join(part for part in (comm, command) if part).lower()
    if not fields:
        return None
    base = Path(comm).name.lower() if comm else ""
    for backend, aliases in BACKEND_COMMAND_ALIASES.items():
        for alias in aliases:
            if base == alias or f" {alias}" in f" {fields}" or f"/{alias}" in fields:
                return backend
    return None


def _detect_current_agent_from_process_tree(pid: int | None = None, *, max_depth: int = 6) -> str | None:
    """Walk parent processes to find a known coding-agent CLI."""
    current_pid = pid if pid is not None else os.getppid()
    depth = 0
    seen: set[int] = set()

    while current_pid > 1 and depth < max_depth and current_pid not in seen:
        seen.add(current_pid)
        comm = _get_process_field(current_pid, "comm")
        command = _get_process_field(current_pid, "command")
        detected = _classify_process_command(comm, command)
        if detected:
            return detected
        parent_text = _get_process_field(current_pid, "ppid")
        try:
            current_pid = int(parent_text)
        except ValueError:
            break
        depth += 1
    return None


def detect_current_agent(env: dict[str, str] | None = None) -> str | None:
    """Detect the current coding agent from env vars or process tree."""
    return _detect_current_agent_from_env(env) or _detect_current_agent_from_process_tree()


def resolve_backend(backend: str = "auto", env: dict[str, str] | None = None) -> str:
    """Resolve 'auto' backend selection to a concrete backend name.

    Priority: explicit value > CLAWTRACE_SCORER_BACKEND env > auto-detect.
    """
    env = os.environ if env is None else env
    requested = (backend or "auto").strip().lower()
    if requested != "auto":
        if requested not in SUPPORTED_BACKENDS:
            raise RuntimeError(f"Unsupported backend: {backend}")
        return requested

    override = (env.get("CLAWTRACE_SCORER_BACKEND") or "").strip().lower()
    if override:
        if override not in SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"Unsupported CLAWTRACE_SCORER_BACKEND value: {override}. "
                f"Use one of: {', '.join(SUPPORTED_BACKENDS)}."
            )
        return override

    detected = detect_current_agent(env)
    if detected:
        return detected

    raise RuntimeError(
        "Could not detect the current agent. "
        "Run clawtrace from a supported agent CLI, set CLAWTRACE_SCORER_BACKEND, "
        "or pass --backend explicitly."
    )


def require_backend_command(backend: str) -> str:
    """Return the CLI command for a backend, ensuring it is installed."""
    command = BACKEND_COMMANDS[backend]
    if shutil.which(command) is None:
        raise RuntimeError(f"{backend} CLI not found. Install it or choose a different --backend.")
    return command


def check_backend_runtime(backend: str, env: dict[str, str] | None = None) -> None:
    """Backend-specific runtime preflight hook (extensible, currently a no-op)."""
    _ = backend, env


def summarize_process_error(stderr: str, stdout: str = "") -> str:
    """Return the most actionable error line from subprocess output."""
    lines: list[str] = []
    for raw in f"{stderr}\n{stdout}".splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("WARNING: proceeding, even though we could not update PATH"):
            continue
        if line.startswith("note: run with `RUST_BACKTRACE=1`"):
            continue
        if line.startswith("thread '"):
            continue
        lines.append(line)

    if not lines:
        return ""

    for line in reversed(lines):
        lower = line.lower()
        if (
            lower.startswith("error:")
            or " error " in lower
            or "failed" in lower
            or "unauthorized" in lower
        ):
            return line

    return lines[-1]


def format_codex_runtime_error(returncode: int, stderr: str, stdout: str = "") -> str:
    """Normalize common Codex exec failures into actionable guidance."""
    combined = "\n".join(part.strip() for part in (stderr, stdout) if part and part.strip())
    lower = combined.lower()

    if (
        "failed to lookup address information" in lower
        or "temporary failure in name resolution" in lower
        or "name or service not known" in lower
        or "network is unreachable" in lower
        or "could not resolve host" in lower
    ):
        return (
            "Codex runs through `codex exec` in non-interactive mode. "
            "This process could not reach the Codex backend from the current environment. "
            "If you launched clawtrace inside a network-disabled Codex sandbox, "
            "rerun it from your host shell or with network access."
        )

    if (
        "401" in lower
        or "unauthorized" in lower
        or "not signed in" in lower
        or "authentication required" in lower
    ):
        return (
            "Codex runs through `codex exec` in non-interactive mode. "
            "`codex exec` reuses saved CLI authentication by default; for automation, "
            "run `codex login` or set `CODEX_API_KEY` before running clawtrace."
        )

    summary = summarize_process_error(stderr, stdout)
    if summary:
        return f"codex exited {returncode}: {summary}"
    return f"codex exited {returncode}"
