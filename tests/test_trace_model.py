"""Tests for the canonical observability trace model."""

from datetime import datetime, timezone

import pytest

from agentstracer.anonymizer import Anonymizer
from agentstracer.trace_model import (
    AgentTrace,
    TraceObservation,
    ns_to_iso,
    sanitize_trace,
    sanitize_value,
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


def test_timestamp_ns_preserves_nanosecond_precision():
    exact = 1_700_000_000_123_456_789
    assert timestamp_ns(exact) == exact
    assert timestamp_ns(str(exact)) == exact
    assert timestamp_ns("2023-11-14T22:13:20.123456789Z") == exact
    assert ns_to_iso(exact) == "2023-11-14T22:13:20.123456789+00:00"


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


def test_sanitize_value_redacts_dict_keys_without_losing_collisions():
    value = {
        "alice": "first",
        "bob": "second",
    }
    sanitized = sanitize_value(
        value,
        anonymizer=Anonymizer(),
        custom_strings=("alice", "bob"),
    )
    assert len(sanitized) == 2
    assert set(sanitized.values()) == {"first", "second"}
    encoded = str(sanitized)
    assert "alice" not in encoded
    assert "bob" not in encoded
    assert any("__redacted_" in key for key in sanitized)


def test_sanitize_trace_covers_identifiers_tags_and_relationships():
    trace = AgentTrace(
        logical_id="trace:/Users/alice/private",
        name="run alice",
        source="custom-alice",
        source_session_id="alice@example.com",
        session_id="session-alice",
        environment="alice-laptop",
        tags=["owner:alice"],
        metadata={"alice@example.com": "alice"},
        observations=[
            TraceObservation("root:alice", "root", "agent", 1, 3),
            TraceObservation(
                "child:alice", "child", "tool", 2, 3, "root:alice",
            ),
        ],
    )
    sanitize_trace(trace, anonymizer=Anonymizer(extra_usernames=["alice"]))
    encoded = str(trace)
    assert "alice" not in encoded.lower()
    assert trace.observations[1].parent_logical_id == trace.observations[0].logical_id


@pytest.mark.parametrize("field", ["name", "project", "metadata", "tags", "environment", "session_id"])
def test_finalize_fingerprint_covers_uploaded_trace_fields(field):
    trace = _trace().finalize()
    original_trace_id = trace.trace_id
    values = {
        "name": "changed name",
        "project": "changed project",
        "metadata": {"changed": True},
        "tags": ["changed"],
        "environment": "changed environment",
        "session_id": "changed session",
    }
    setattr(trace, field, values[field])
    trace.finalize()
    assert trace.trace_id != original_trace_id
