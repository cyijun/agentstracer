"""Tests for agentstracer.backends — shared backend detection and resolution."""

import pytest

from agentstracer.backends import (
    BACKEND_CHOICES,
    SUPPORTED_BACKENDS,
    _classify_process_command,
    _detect_current_agent_from_env,
    _detect_current_agent_from_process_tree,
    _get_process_field,
    check_backend_runtime,
    format_codex_runtime_error,
    require_backend_command,
    resolve_backend,
    summarize_process_error,
)


class TestConstants:
    def test_backend_choices_include_auto(self):
        assert BACKEND_CHOICES == ("auto", "claude", "codex", "openclaw")

    def test_supported_backends(self):
        assert set(SUPPORTED_BACKENDS) == {"claude", "codex", "openclaw"}


class TestDetection:
    def test_detect_from_env_claude(self):
        assert _detect_current_agent_from_env({"CLAUDECODE": "1"}) == "claude"

    def test_detect_from_env_codex(self):
        assert _detect_current_agent_from_env({"CODEX_THREAD_ID": "t-1"}) == "codex"

    def test_detect_from_env_openclaw(self):
        assert _detect_current_agent_from_env({"OPENCLAW_STATE_DIR": "/tmp"}) == "openclaw"

    def test_detect_from_env_empty(self):
        assert _detect_current_agent_from_env({}) is None

    def test_classify_process_command_claude(self):
        assert _classify_process_command("claude", "") == "claude"
        assert _classify_process_command("", "/usr/local/bin/claude -p") == "claude"

    def test_classify_process_command_codex(self):
        assert _classify_process_command("codex", "") == "codex"
        assert _classify_process_command("", "/opt/homebrew/bin/codex exec") == "codex"

    def test_classify_process_command_openclaw(self):
        assert _classify_process_command("openclaw", "") == "openclaw"

    def test_classify_process_command_unknown(self):
        assert _classify_process_command("bash", "/bin/bash") is None


class TestResolveBackend:
    def test_explicit_backend(self):
        assert resolve_backend("codex", {}) == "codex"
        assert resolve_backend("claude", {}) == "claude"
        assert resolve_backend("openclaw", {}) == "openclaw"

    def test_explicit_unsupported_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported backend"):
            resolve_backend("gemini", {})

    def test_env_override(self):
        env = {"AGENTSTRACE_SCORER_BACKEND": "openclaw"}
        assert resolve_backend("auto", env) == "openclaw"

    def test_env_override_invalid_raises(self):
        env = {"AGENTSTRACE_SCORER_BACKEND": "invalid"}
        with pytest.raises(RuntimeError, match="Unsupported AGENTSTRACE_SCORER_BACKEND"):
            resolve_backend("auto", env)

    def test_auto_detects_from_env(self):
        env = {"CODEX_THREAD_ID": "thread-123"}
        assert resolve_backend("auto", env) == "codex"


class TestErrorFormatting:
    def test_codex_network_error(self):
        msg = format_codex_runtime_error(1, "failed to lookup address information")
        assert "codex exec" in msg
        assert "host shell" in msg

    def test_codex_auth_error(self):
        msg = format_codex_runtime_error(1, "401 Unauthorized")
        assert "CODEX_API_KEY" in msg
        assert "codex login" in msg

    def test_codex_generic_error(self):
        msg = format_codex_runtime_error(1, "error: something broke")
        assert "codex exited 1" in msg
        assert "something broke" in msg

    def test_summarize_process_error_finds_error_line(self):
        stderr = "info: starting\nerror: connection refused\n"
        assert "connection refused" in summarize_process_error(stderr)

    def test_summarize_process_error_empty(self):
        assert summarize_process_error("") == ""


class TestCheckBackendRuntime:
    def test_is_noop(self):
        assert check_backend_runtime("codex") is None
        assert check_backend_runtime("claude") is None


class TestRequireBackendCommand:
    def test_found(self, monkeypatch):
        monkeypatch.setattr("agentstracer.backends.shutil.which", lambda cmd: "/usr/bin/" + cmd)
        assert require_backend_command("claude") == "claude"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.setattr("agentstracer.backends.shutil.which", lambda cmd: None)
        with pytest.raises(RuntimeError, match="CLI not found"):
            require_backend_command("codex")


class TestResolveBackendAutoNoAgent:
    def test_raises_when_no_agent(self, monkeypatch):
        monkeypatch.setattr("agentstracer.backends._detect_current_agent_from_process_tree", lambda **kw: None)
        with pytest.raises(RuntimeError, match="Could not detect the current agent"):
            resolve_backend("auto", {})


class TestProcessTreeDetection:
    def test_finds_claude_in_parent(self, monkeypatch):
        def fake_get_field(pid, field):
            if pid == 100 and field == "comm":
                return "claude"
            if pid == 100 and field == "command":
                return "/usr/local/bin/claude -p"
            if pid == 200 and field == "ppid":
                return "100"
            return ""

        monkeypatch.setattr("agentstracer.backends._get_process_field", fake_get_field)
        assert _detect_current_agent_from_process_tree(pid=200, max_depth=6) == "claude"

    def test_returns_none_at_max_depth(self, monkeypatch):
        def fake_get_field(pid, field):
            if field == "ppid":
                return str(pid + 1)
            return "bash"

        monkeypatch.setattr("agentstracer.backends._get_process_field", fake_get_field)
        assert _detect_current_agent_from_process_tree(pid=10, max_depth=2) is None

    def test_handles_cycle(self, monkeypatch):
        def fake_get_field(pid, field):
            if field == "ppid":
                return "10"
            return "bash"

        monkeypatch.setattr("agentstracer.backends._get_process_field", fake_get_field)
        assert _detect_current_agent_from_process_tree(pid=10, max_depth=10) is None


class TestGetProcessField:
    def test_timeout_returns_empty(self, monkeypatch):
        import subprocess
        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="ps", timeout=2)
        monkeypatch.setattr("agentstracer.backends.subprocess.run", fake_run)
        assert _get_process_field(1234, "comm") == ""

    def test_nonzero_returncode_returns_empty(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            "agentstracer.backends.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
        )
        assert _get_process_field(99999, "comm") == ""
