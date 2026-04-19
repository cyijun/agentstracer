"""Tests for agentstrace.scoring — pure functions (no LLM calls)."""

import json

import pytest

from agentstrace.backends import (
    _classify_process_command,
    _detect_current_agent_from_env,
    check_backend_runtime as _check_backend_runtime,
    format_codex_runtime_error as _format_codex_runtime_error,
)
from agentstrace.scoring import (
    SCORING_BACKEND_CHOICES,
    SCORING_BACKEND_RUNNERS,
    Segment,
    ScoringResult,
    Step,
    _extract_judge_result_from_value,
    _resolve_scoring_backend,
    call_judge,
    compute_basic_metrics,
    extract_tool_uses,
    format_session_for_judge,
    get_message_text,
    load_scoring_rubric,
    segment_session,
)


# ---------------------------------------------------------------------------
# Helpers to build test messages
# ---------------------------------------------------------------------------


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _asst_msg(text: str, tool_uses: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool_uses:
        msg["tool_uses"] = tool_uses
    return msg


def _tool_use(tool: str, inp: str = "", output: str = "", status: str = "success") -> dict:
    return {
        "tool": tool,
        "input": {"path": inp} if inp else {},
        "output": output,
        "status": status,
    }


# ---------------------------------------------------------------------------
# get_message_text
# ---------------------------------------------------------------------------


class TestGetMessageText:
    def test_string_content(self):
        assert get_message_text({"content": "hello"}) == "hello"

    def test_list_with_text_block(self):
        msg = {"content": [{"text": "hello", "type": "text"}]}
        assert get_message_text(msg) == "hello"

    def test_list_with_string(self):
        msg = {"content": ["hello"]}
        assert get_message_text(msg) == "hello"

    def test_empty(self):
        assert get_message_text({}) == ""
        assert get_message_text({"content": []}) == ""


# ---------------------------------------------------------------------------
# extract_tool_uses
# ---------------------------------------------------------------------------


class TestExtractToolUses:
    def test_from_tool_uses_field(self):
        msg = {"tool_uses": [{"tool": "Read", "input": {}, "output": "ok", "status": "success"}]}
        uses = extract_tool_uses(msg)
        assert len(uses) == 1
        assert uses[0]["tool"] == "Read"

    def test_from_content_blocks(self):
        msg = {
            "content": [
                {"tool": "Bash", "input": {"command": "ls"}, "output": "file.py", "status": "success"},
            ]
        }
        uses = extract_tool_uses(msg)
        assert len(uses) == 1
        assert uses[0]["tool"] == "Bash"
        assert uses[0]["first_arg"] == "ls"

    def test_no_tool_uses(self):
        assert extract_tool_uses({"content": "just text"}) == []
        assert extract_tool_uses({}) == []


# ---------------------------------------------------------------------------
# segment_session
# ---------------------------------------------------------------------------


class TestSegmentSession:
    def test_single_segment(self):
        messages = [
            _user_msg("Fix the bug"),
            _asst_msg("I'll look at it", [_tool_use("Read", "auth.py", "contents")]),
            _asst_msg("Found it", [_tool_use("Edit", "auth.py", "fixed")]),
        ]
        segments = segment_session(messages)
        assert len(segments) == 1
        assert segments[0].user_message == "Fix the bug"
        assert len(segments[0].steps) == 2
        assert segments[0].steps[0].action_tool == "Read"
        assert segments[0].steps[1].action_tool == "Edit"
        assert segments[0].user_response is None

    def test_multi_segment(self):
        messages = [
            _user_msg("Fix the bug"),
            _asst_msg("Done", [_tool_use("Edit", "auth.py", "fixed")]),
            _user_msg("Now add tests"),
            _asst_msg("Writing tests", [_tool_use("Write", "test.py", "ok")]),
        ]
        segments = segment_session(messages)
        assert len(segments) == 2
        assert segments[0].user_message == "Fix the bug"
        assert segments[0].user_response == "Now add tests"
        assert segments[1].user_message == "Now add tests"
        assert segments[1].user_response is None

    def test_no_tool_uses(self):
        messages = [
            _user_msg("Hello"),
            _asst_msg("Hi there"),
        ]
        segments = segment_session(messages)
        assert len(segments) == 1
        assert len(segments[0].steps) == 0

    def test_empty_messages(self):
        assert segment_session([]) == []

    def test_multiple_tool_uses_in_one_message(self):
        messages = [
            _user_msg("Check everything"),
            _asst_msg("Checking", [
                _tool_use("Read", "a.py", "ok"),
                _tool_use("Read", "b.py", "ok"),
                _tool_use("Read", "c.py", "ok"),
            ]),
        ]
        segments = segment_session(messages)
        assert len(segments) == 1
        assert len(segments[0].steps) == 3

    def test_reflect_on_previous_step(self):
        messages = [
            _user_msg("Do it"),
            _asst_msg("Starting", [_tool_use("Read", "f.py", "contents")]),
            _asst_msg("I see the issue"),
        ]
        segments = segment_session(messages)
        assert len(segments) == 1
        assert len(segments[0].steps) == 1
        assert segments[0].steps[0].reflect == "I see the issue"

    def test_assistant_only_session(self):
        messages = [
            _asst_msg("Auto-running", [_tool_use("Bash", "ls", "file.py")]),
        ]
        segments = segment_session(messages)
        assert len(segments) == 1
        assert segments[0].user_message == ""
        assert len(segments[0].steps) == 1


# ---------------------------------------------------------------------------
# compute_basic_metrics
# ---------------------------------------------------------------------------


class TestComputeBasicMetrics:
    def test_basic(self):
        seg = Segment(user_message="test", steps=[
            Step("", "Read", "a", "ok", "success", ""),
            Step("", "Bash", "b", "err", "failure", ""),
            Step("", "Edit", "c", "ok", "success", ""),
        ])
        detail = {
            "user_messages": 2,
            "input_tokens": 5000,
            "output_tokens": 3000,
            "duration_seconds": 120,
            "files_touched": '["a.py", "b.py"]',
            "outcome_badge": "tests_passed",
        }
        m = compute_basic_metrics([seg], detail)
        assert m["total_steps"] == 3
        assert m["tool_failures"] == 1
        assert m["segments"] == 1
        assert m["outcome_badge"] == "tests_passed"

    def test_empty(self):
        m = compute_basic_metrics([], {})
        assert m["total_steps"] == 0
        assert m["tool_failures"] == 0
        assert m["outcome_badge"] is None

    def test_multiple_segments(self):
        seg1 = Segment(user_message="a", steps=[
            Step("", "Read", "f", "ok", "success", ""),
        ])
        seg2 = Segment(user_message="b", steps=[
            Step("", "Bash", "c", "err", "error", ""),
            Step("", "Edit", "d", "ok", "success", ""),
        ])
        m = compute_basic_metrics([seg1, seg2], {})
        assert m["total_steps"] == 3
        assert m["segments"] == 2
        assert m["tool_failures"] == 1


# ---------------------------------------------------------------------------
# load_scoring_rubric
# ---------------------------------------------------------------------------


class TestLoadScoringRubric:
    def test_rubric_loads(self):
        rubric = load_scoring_rubric()
        assert "quality" in rubric.lower() or "score" in rubric.lower()
        assert len(rubric) > 100


# ---------------------------------------------------------------------------
# format_session_for_judge
# ---------------------------------------------------------------------------


class TestFormatSessionForJudge:
    def test_single_segment(self):
        seg = Segment(
            user_message="Fix bug",
            steps=[
                Step("Looking at it", "Read", "auth.py", "file contents", "success", ""),
                Step("Fixing", "Edit", "auth.py", "applied fix", "success", ""),
            ],
            user_response="thanks!",
        )
        metrics = {"total_steps": 2, "tool_failures": 0, "input_tokens": 5000,
                   "output_tokens": 3000, "outcome_badge": "tests_passed"}
        text = format_session_for_judge([seg], "Fix bug", metrics)
        assert "## User's Task" in text
        assert "Fix bug" in text
        assert "Step 1:" in text
        assert "Step 2:" in text
        assert "Read(auth.py)" in text
        assert "## Session Metrics" in text
        assert "Outcome: tests_passed" in text
        assert '"thanks!"' in text
        assert "Respond with JSON" in text

    def test_multi_segment_shows_turns(self):
        seg1 = Segment(user_message="Fix it", steps=[
            Step("", "Read", "f.py", "ok", "success", ""),
        ], user_response="Now test")
        seg2 = Segment(user_message="Now test", steps=[
            Step("", "Bash", "pytest", "pass", "success", ""),
        ])
        text = format_session_for_judge([seg1, seg2], "Fix it\nNow test")
        assert "Turn 1" in text
        assert "Turn 2" in text

    def test_no_user_response(self):
        seg = Segment(user_message="Do it", steps=[
            Step("", "Read", "f.py", "ok", "success", ""),
        ])
        text = format_session_for_judge([seg], "Do it")
        assert "No response — session ended" in text


# ---------------------------------------------------------------------------
# ScoringResult
# ---------------------------------------------------------------------------


class TestScoringResult:
    def test_basic(self):
        r = ScoringResult(segments=[], quality=4, reason="Good session")
        assert r.quality == 4
        assert r.reason == "Good session"
        assert r.taste_signals == []
        assert r.detail_json == "{}"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_backend_choices_include_auto(self):
        assert SCORING_BACKEND_CHOICES == ("auto", "claude", "codex", "openclaw")

    def test_detect_current_agent_from_env_codex(self):
        env = {"CODEX_THREAD_ID": "thread-123"}
        assert _detect_current_agent_from_env(env) == "codex"

    def test_detect_current_agent_from_env_openclaw(self):
        env = {"OPENCLAW_STATE_DIR": "/tmp/openclaw"}
        assert _detect_current_agent_from_env(env) == "openclaw"

    def test_detect_current_agent_from_env_unknown(self):
        assert _detect_current_agent_from_env({}) is None

    def test_classify_process_command_codex(self):
        assert _classify_process_command("codex", "") == "codex"
        assert _classify_process_command("", "/opt/homebrew/bin/codex exec") == "codex"

    def test_classify_process_command_claude(self):
        assert _classify_process_command("claude", "") == "claude"
        assert _classify_process_command("", "/usr/local/bin/claude -p") == "claude"

    def test_classify_process_command_openclaw(self):
        assert _classify_process_command("openclaw", "") == "openclaw"
        assert _classify_process_command("", "/usr/local/bin/openclaw agent --json") == "openclaw"

    def test_resolve_backend_explicit(self):
        assert _resolve_scoring_backend("codex", {}) == "codex"

    def test_resolve_backend_env_override(self):
        env = {"AGENTSTRACE_SCORER_BACKEND": "claude"}
        assert _resolve_scoring_backend("auto", env) == "claude"

    def test_resolve_backend_raises_without_current_agent(self):
        with pytest.raises(RuntimeError, match="Could not detect the current agent"):
            _resolve_scoring_backend("auto", {})

    def test_call_judge_dispatches_to_codex(self, monkeypatch):
        monkeypatch.setenv("CODEX_THREAD_ID", "thread-123")
        monkeypatch.setattr("agentstrace.scoring.load_scoring_rubric", lambda: "rubric")
        monkeypatch.setitem(
            SCORING_BACKEND_RUNNERS,
            "codex",
            lambda prompt_text, session_data, metadata, rubric, model: {
                "quality": 4,
                "reasoning": "Good session",
                "display_title": "Fix auth tests",
                "outcome": 4,
                "intent": 4,
                "taste": {"detected": False},
                "task_type": "debugging",
                "outcome_label": "tests_passed",
                "value_labels": [],
                "risk_level": [],
            },
        )
        result = call_judge(
            "prompt",
            session_data={"messages": []},
            metadata={"total_steps": 1},
        )
        assert result["quality"] == 4
        assert result["task_type"] == "debugging"

    def test_check_backend_runtime_codex_is_non_blocking(self):
        env = {"CODEX_SANDBOX_NETWORK_DISABLED": "1"}
        assert _check_backend_runtime("codex", env) is None

    def test_format_codex_runtime_error_for_network_failure(self):
        message = _format_codex_runtime_error(
            1,
            "ERROR failed to connect: failed to lookup address information: nodename nor servname provided, or not known",
        )
        assert "codex exec" in message
        assert "host shell" in message

    def test_format_codex_runtime_error_for_auth_failure(self):
        message = _format_codex_runtime_error(
            1,
            "401 Unauthorized",
        )
        assert "CODEX_API_KEY" in message
        assert "codex login" in message

    def test_call_judge_dispatches_to_openclaw(self, monkeypatch):
        monkeypatch.setattr("agentstrace.scoring.load_scoring_rubric", lambda: "rubric")
        monkeypatch.setitem(
            SCORING_BACKEND_RUNNERS,
            "openclaw",
            lambda prompt_text, session_data, metadata, rubric, model: {
                "quality": 5,
                "reasoning": "Excellent session",
                "display_title": "Add retry logic",
                "outcome": 5,
                "intent": 5,
                "taste": {"detected": False},
                "task_type": "feature",
                "outcome_label": "completed",
                "value_labels": ["tool_rich"],
                "risk_level": [],
            },
        )
        result = call_judge(
            "prompt",
            session_data={"messages": []},
            metadata={"total_steps": 1},
            backend="openclaw",
        )
        assert result["quality"] == 5
        assert result["task_type"] == "feature"

    def test_extract_judge_result_from_nested_openclaw_json(self):
        payload = {
            "reply": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "quality": 4,
                            "reasoning": "Solid session",
                            "display_title": "Refactor auth flow",
                            "outcome": 4,
                            "intent": 4,
                            "taste": {"detected": False},
                            "task_type": "refactor",
                            "outcome_label": "completed",
                            "value_labels": [],
                            "risk_level": [],
                        }),
                    }
                ]
            }
        }
        result = _extract_judge_result_from_value(payload)
        assert result["quality"] == 4
        assert result["task_type"] == "refactor"
