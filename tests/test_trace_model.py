"""Tests for the canonical observability trace model."""

from datetime import datetime, timezone

import pytest

from agentstracer.anonymizer import Anonymizer
from agentstracer.trace_model import (
    AgentTrace,
    TraceObservation,
    sanitize_trace,
    timestamp_ns,
)


def _trace(observations=None):
    return AgentTrace(
        logical_id="session:turn:0",
        name="agent turn",
        source="test",
        source_session_id="session",
        observations=observations or [
            TraceObservation(
                logical_id="root", name="run", as_type="agent",
                start_ns=1_700_000_000_000_000_000,
                end_ns=1_700_000_001_000_000_000,
            ),
        ],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_700_000_000, 1_700_000_000_000_000_000),
        (1_700_000_000_000, 1_700_000_000_000_000_000),
        (1_700_000_000_000_000, 1_700_000_000_000_000_000),
        (1_700_000_000_000_000_000, 1_700_000_000_000_000_000),
        ("2023-11-14T22:13:20Z", 1_700_000_000_000_000_000),
        (datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc), 1_700_000_000_000_000_000),
    ],
)
def test_timestamp_ns_supports_agent_encodings(value, expected):
    assert timestamp_ns(value) == expected


def test_finalize_repairs_missing_parent_and_is_deterministic():
    trace = _trace([
        TraceObservation("root", "run", "agent", 10, 20),
        TraceObservation("child", "tool", "tool", 12, 11, "missing"),
    ]).finalize()
    child = trace.observations[1]
    assert child.parent_logical_id == "root"
    assert child.end_ns == child.start_ns
    assert child.metadata == {"time_repaired": True, "missing_parent": "missing"}

    same = _trace([
        TraceObservation("root", "run", "agent", 10, 20),
        TraceObservation("child", "tool", "tool", 12, 12, "root", metadata={
            "time_repaired": True, "missing_parent": "missing",
        }),
    ]).finalize()
    assert trace.trace_id == same.trace_id
    assert len(trace.trace_id) == 32
    assert len(trace.span_id("root")) == 16


def test_finalize_rejects_duplicate_ids_and_parent_cycles():
    with pytest.raises(ValueError, match="duplicate"):
        _trace([
            TraceObservation("root", "run", "agent", 1, 2),
            TraceObservation("root", "again", "span", 1, 2, "root"),
        ]).finalize()

    with pytest.raises(ValueError, match="cycle"):
        _trace([
            TraceObservation("root", "run", "agent", 1, 2),
            TraceObservation("a", "a", "span", 1, 2, "b"),
            TraceObservation("b", "b", "span", 1, 2, "a"),
        ]).finalize()


def test_sanitize_trace_redacts_nested_secrets_and_paths():
    trace = _trace()
    trace.project = "/Users/private-name/work/repo"
    trace.root.input = {
        "api_key": "sk-" + "a" * 48,
        "cwd": "/Users/private-name/work/repo",
        "nested": ["contact sensitive.person@corp.com"],
    }
    sanitize_trace(
        trace,
        anonymizer=Anonymizer(extra_usernames=["private-name"]),
        custom_strings=("work",),
    )
    encoded = str(trace.root.input)
    assert "sk-aaaa" not in encoded
    assert "private-name" not in encoded
    assert "sensitive.person@corp.com" not in encoded
    assert "work" not in encoded
    assert trace.trace_id is not None
