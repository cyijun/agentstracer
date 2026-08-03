"""Regression tests for provider-neutral training-data conversion."""

import pytest

from agentstracer.convert_to_training_data import (
    convert_session,
    convert_sessions_to_training,
)


def test_plain_assistant_message_is_a_final_reply():
    turns = convert_session({
        "session_id": "claude-session",
        "source": "claude",
        "messages": [
            {"role": "user", "content": "Fix it"},
            {"role": "assistant", "content": "Done."},
        ],
    })
    assert turns[0]["output"] == [{"type": "message", "content": "Done."}]


def test_async_process_polls_wait_for_terminal_result():
    session = {
        "session_id": "async-session",
        "source": "openclaw",
        "messages": [
            {"role": "user", "content": "Run it"},
            {"role": "assistant", "tool_uses": [{
                "tool": "exec",
                "input": {"cmd": "long-task", "timeout": 10},
                "output": {"text": "Command still running (session proc-1, pid 42)"},
                "status": "unknown",
            }]},
            {"role": "assistant", "tool_uses": [{
                "tool": "process",
                "input": {"sessionId": "proc-1", "action": "poll"},
                "output": {"text": "Command still running (session proc-1, pid 42)"},
                "status": "unknown",
            }]},
            {"role": "assistant", "tool_uses": [{
                "tool": "process",
                "input": {"sessionId": "proc-1", "action": "poll"},
                "output": {"text": "FINAL OUTPUT", "status": "completed"},
                "status": "unknown",
            }]},
            {"role": "assistant", "content": "Finished."},
        ],
    }
    output = convert_session(session)[0]["output"]
    calls = [item for item in output if item["type"] == "tool_call"]
    results = [item for item in output if item["type"] == "tool_result"]
    assert len(calls) == 1
    assert results == [{
        "type": "tool_result",
        "id": calls[0]["id"],
        "output": "FINAL OUTPUT",
        "status": "success",
    }]
    assert output[-1] == {"type": "message", "content": "Finished."}


def test_async_terminal_error_preserves_failure_status():
    session = {
        "session_id": "failed-session",
        "messages": [
            {"role": "user", "content": "Run"},
            {"role": "assistant", "tool_uses": [{
                "tool": "exec", "input": {"cmd": "false"},
                "output": {"text": "Command still running (session p2, pid 43)"},
            }]},
            {"role": "assistant", "tool_uses": [{
                "tool": "process", "input": {"sessionId": "p2"},
                "output": {"text": "exit 1", "status": "failed"},
            }]},
        ],
    }
    result = next(
        item for item in convert_session(session)[0]["output"]
        if item["type"] == "tool_result"
    )
    assert result["output"] == "exit 1"
    assert result["status"] == "error"


def test_nested_stats_are_exported_and_tool_ids_are_stable():
    session = {
        "session_id": "stable-session",
        "stats": {
            "user_messages": 7,
            "assistant_messages": 3,
            "tool_uses": 1,
            "input_tokens": 101,
            "output_tokens": 22,
            "duration_seconds": 9,
        },
        "messages": [
            {"role": "user", "content": "Inspect"},
            {"role": "assistant", "tool_uses": [{
                "tool": "read", "input": {"path": "README.md"},
                "output": {"text": "contents"}, "status": "success",
            }]},
            {"role": "assistant", "content": "Looks good."},
        ],
    }
    first = convert_session(session)
    second = convert_session(session)
    assert first == second
    assert first[0]["session_stats"] == session["stats"]
    tool_call = next(
        item for item in first[0]["output"] if item["type"] == "tool_call"
    )
    assert tool_call["id"].startswith("tc_")


def test_training_file_write_is_private_and_atomic(tmp_path):
    output = tmp_path / "training.jsonl"
    sessions = [{
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    }]
    convert_sessions_to_training(sessions, output)
    assert "world" in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600

    def broken_sessions():
        yield sessions[0]
        raise RuntimeError("conversion failed")

    previous = output.read_text()
    with pytest.raises(RuntimeError, match="conversion failed"):
        convert_sessions_to_training(broken_sessions(), output)
    assert output.read_text() == previous
    assert not list(tmp_path.glob(".training.jsonl.*.tmp"))
