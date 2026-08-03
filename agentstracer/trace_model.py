"""Canonical trace model shared by local agent adapters and Langfuse export.

The existing conversation parser intentionally flattens provider logs into a
training-friendly ``messages`` list.  Observability needs a different shape:
stable identities, parent/child relationships, exact operation bounds, and a
way to state when a timestamp or relationship was inferred.  This module is
kept dependency-free so trace extraction and dry-runs remain fully local.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .anonymizer import Anonymizer
from .secrets import redact_text


OBSERVATION_TYPES = frozenset({
    "span", "generation", "event", "agent", "tool", "chain",
    "retriever", "evaluator", "embedding", "guardrail",
})
LEVELS = frozenset({"DEBUG", "DEFAULT", "WARNING", "ERROR"})
PATH_KEYS = frozenset({
    "cwd", "directory", "file", "file_path", "filepath", "filename",
    "path", "project_path", "session_file", "workdir", "workspace_dir",
})


def stable_hex(*parts: Any, nbytes: int) -> str:
    """Create a stable lowercase hex identifier of exactly ``nbytes`` bytes."""
    h = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="replace")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    value = h.digest()[:nbytes]
    # OTel forbids all-zero trace/span identifiers.
    if not any(value):
        value = b"\x01" + value[1:]
    return value.hex()


def _numeric_timestamp_to_ns(value: float) -> int | None:
    if not math.isfinite(value) or value <= 0:
        return None
    # Current agent logs use a mix of seconds, milliseconds, microseconds and
    # nanoseconds.  Magnitude is unambiguous for contemporary timestamps.
    if value >= 1e17:
        return int(value)
    if value >= 1e14:
        return int(value * 1_000)
    if value >= 1e11:
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def timestamp_ns(value: Any) -> int | None:
    """Normalize common agent timestamp encodings to Unix nanoseconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    if isinstance(value, (int, float)):
        return _numeric_timestamp_to_ns(float(value))
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return _numeric_timestamp_to_ns(float(text))
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


@dataclasses.dataclass(slots=True)
class TraceLink:
    """A non-parent causal relationship represented as an OTel span link."""

    target_logical_id: str
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class TraceObservation:
    logical_id: str
    name: str
    as_type: str
    start_ns: int
    end_ns: int
    parent_logical_id: str | None = None
    input: Any = None
    output: Any = None
    model: str | None = None
    usage: dict[str, int | float] = dataclasses.field(default_factory=dict)
    cost: dict[str, int | float] = dataclasses.field(default_factory=dict)
    model_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    level: str = "DEFAULT"
    status_message: str | None = None
    completion_start_ns: int | None = None
    links: list[TraceLink] = dataclasses.field(default_factory=list)

    def normalize(self) -> None:
        if self.as_type not in OBSERVATION_TYPES:
            self.as_type = "span"
        if self.level not in LEVELS:
            self.level = "DEFAULT"
        if self.end_ns < self.start_ns:
            self.end_ns = self.start_ns
            self.metadata.setdefault("time_repaired", True)
        if self.completion_start_ns is not None:
            self.completion_start_ns = max(
                self.start_ns, min(self.completion_start_ns, self.end_ns)
            )


@dataclasses.dataclass(slots=True)
class AgentTrace:
    """One Langfuse trace snapshot with a single explicit root observation."""

    logical_id: str
    name: str
    source: str
    source_session_id: str
    observations: list[TraceObservation]
    session_id: str | None = None
    project: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    tags: list[str] = dataclasses.field(default_factory=list)
    environment: str = "local-import"
    trace_id: str | None = None
    content_hash: str | None = None

    def finalize(self) -> "AgentTrace":
        if not self.observations:
            raise ValueError(f"trace {self.logical_id!r} has no observations")

        seen: set[str] = set()
        for observation in self.observations:
            observation.normalize()
            if observation.logical_id in seen:
                raise ValueError(
                    f"duplicate observation logical id {observation.logical_id!r}"
                )
            seen.add(observation.logical_id)

        roots = [o for o in self.observations if o.parent_logical_id is None]
        if len(roots) != 1:
            raise ValueError(
                f"trace {self.logical_id!r} must have exactly one root; got {len(roots)}"
            )
        for observation in self.observations:
            parent = observation.parent_logical_id
            if parent is not None and parent not in seen:
                observation.metadata.setdefault("missing_parent", parent)
                observation.parent_logical_id = roots[0].logical_id
            valid_links: list[TraceLink] = []
            for link in observation.links:
                if link.target_logical_id in seen:
                    valid_links.append(link)
                else:
                    observation.metadata.setdefault("missing_link_targets", []).append(
                        link.target_logical_id
                    )
            observation.links = valid_links

        self._check_parent_cycles()
        fingerprint_payload = {
            "logical_id": self.logical_id,
            "source": self.source,
            "source_session_id": self.source_session_id,
            "observations": [dataclasses.asdict(o) for o in self.observations],
        }
        self.content_hash = hashlib.sha256(
            canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        # Including the snapshot hash keeps immutable Langfuse v4 observations
        # safe when an active local trace later grows.
        self.trace_id = stable_hex(
            "agentstracer-trace", self.source, self.logical_id, self.content_hash,
            nbytes=16,
        )
        if self.session_id is None:
            self.session_id = stable_hex(
                "agentstracer-session", self.source, self.source_session_id,
                nbytes=16,
            )
        self.tags = list(dict.fromkeys([self.source, "agentstracer", *self.tags]))
        return self

    def _check_parent_cycles(self) -> None:
        parents = {o.logical_id: o.parent_logical_id for o in self.observations}
        for start in parents:
            cursor: str | None = start
            visited: set[str] = set()
            while cursor is not None:
                if cursor in visited:
                    raise ValueError(f"observation parent cycle at {cursor!r}")
                visited.add(cursor)
                cursor = parents.get(cursor)

    @property
    def root(self) -> TraceObservation:
        return next(o for o in self.observations if o.parent_logical_id is None)

    def span_id(self, logical_id: str) -> str:
        if not self.trace_id:
            raise ValueError("trace must be finalized before span ids are requested")
        return stable_hex(
            "agentstracer-span", self.trace_id, logical_id, nbytes=8,
        )

    def summary(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for observation in self.observations:
            types[observation.as_type] = types.get(observation.as_type, 0) + 1
        return {
            "trace_id": self.trace_id,
            "logical_id": self.logical_id,
            "source": self.source,
            "session_id": self.session_id,
            "start_time": ns_to_iso(self.root.start_ns),
            "end_time": ns_to_iso(self.root.end_ns),
            "observation_count": len(self.observations),
            "observation_types": types,
            "content_hash": self.content_hash,
        }


def _sanitize_string(
    value: str,
    *,
    key: str | None,
    anonymizer: Anonymizer,
    custom_strings: Iterable[str],
    redact_secrets: bool,
) -> str:
    if key and key.lower() in PATH_KEYS:
        result = anonymizer.path(value)
    else:
        result = anonymizer.text(value)
    for custom in custom_strings:
        if custom:
            result = result.replace(custom, "[REDACTED]")
    if redact_secrets:
        result, _, _ = redact_text(result)
    return result


def sanitize_value(
    value: Any,
    *,
    anonymizer: Anonymizer,
    custom_strings: Iterable[str] = (),
    redact_secrets: bool = True,
    key: str | None = None,
) -> Any:
    """Recursively sanitize every string before it can leave the machine."""
    if isinstance(value, str):
        return _sanitize_string(
            value,
            key=key,
            anonymizer=anonymizer,
            custom_strings=custom_strings,
            redact_secrets=redact_secrets,
        )
    if isinstance(value, dict):
        return {
            str(k): sanitize_value(
                v,
                anonymizer=anonymizer,
                custom_strings=custom_strings,
                redact_secrets=redact_secrets,
                key=str(k),
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_value(
                item,
                anonymizer=anonymizer,
                custom_strings=custom_strings,
                redact_secrets=redact_secrets,
                key=key,
            )
            for item in value
        ]
    if isinstance(value, Path):
        return _sanitize_string(
            str(value),
            key=key or "path",
            anonymizer=anonymizer,
            custom_strings=custom_strings,
            redact_secrets=redact_secrets,
        )
    return value


def sanitize_trace(
    trace: AgentTrace,
    *,
    anonymizer: Anonymizer,
    custom_strings: Iterable[str] = (),
    redact_secrets: bool = True,
) -> AgentTrace:
    """Sanitize a trace in place and finalize its immutable identifiers."""
    strings = tuple(custom_strings)
    trace.name = _sanitize_string(
        trace.name,
        key=None,
        anonymizer=anonymizer,
        custom_strings=strings,
        redact_secrets=redact_secrets,
    )
    trace.project = sanitize_value(
        trace.project,
        anonymizer=anonymizer,
        custom_strings=strings,
        redact_secrets=redact_secrets,
        key="project_path",
    )
    trace.metadata = sanitize_value(
        trace.metadata,
        anonymizer=anonymizer,
        custom_strings=strings,
        redact_secrets=redact_secrets,
    )
    for observation in trace.observations:
        observation.name = _sanitize_string(
            observation.name,
            key=None,
            anonymizer=anonymizer,
            custom_strings=strings,
            redact_secrets=redact_secrets,
        )
        for field_name in (
            "input", "output", "metadata", "model_parameters", "status_message",
        ):
            setattr(
                observation,
                field_name,
                sanitize_value(
                    getattr(observation, field_name),
                    anonymizer=anonymizer,
                    custom_strings=strings,
                    redact_secrets=redact_secrets,
                    key=field_name,
                ),
            )
        if observation.model:
            observation.model = _sanitize_string(
                observation.model,
                key=None,
                anonymizer=anonymizer,
                custom_strings=strings,
                redact_secrets=redact_secrets,
            )
        for link in observation.links:
            link.attributes = sanitize_value(
                link.attributes,
                anonymizer=anonymizer,
                custom_strings=strings,
                redact_secrets=redact_secrets,
            )
    return trace.finalize()


def bounds(values: Iterable[int | None], fallback: int | None = None) -> tuple[int, int]:
    present = [value for value in values if value is not None]
    if present:
        return min(present), max(present)
    now = fallback or int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)
    return now, now
