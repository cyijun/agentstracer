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
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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


def _numeric_timestamp_to_ns(value: int | float | Decimal) -> int | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        exact = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        return None
    if not exact.is_finite() or exact <= 0:
        return None
    # Current agent logs use a mix of seconds, milliseconds, microseconds and
    # nanoseconds.  Magnitude is unambiguous for contemporary timestamps.
    if exact >= Decimal("1e17"):
        multiplier = 1
    elif exact >= Decimal("1e14"):
        multiplier = 1_000
    elif exact >= Decimal("1e11"):
        multiplier = 1_000_000
    else:
        multiplier = 1_000_000_000
    return int(exact * multiplier)


def _datetime_to_ns(value: datetime) -> int:
    """Convert a datetime without going through a precision-losing float."""
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    utc = aware.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


_ISO_TIMESTAMP_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)


def timestamp_ns(value: Any) -> int | None:
    """Normalize common agent timestamp encodings to Unix nanoseconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return _datetime_to_ns(value)
    if isinstance(value, int):
        return _numeric_timestamp_to_ns(value)
    if isinstance(value, float):
        return _numeric_timestamp_to_ns(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return _numeric_timestamp_to_ns(Decimal(text))
    except InvalidOperation:
        pass
    match = _ISO_TIMESTAMP_RE.fullmatch(text)
    if match:
        timezone_text = match.group("tz") or "+00:00"
        if timezone_text == "Z":
            timezone_text = "+00:00"
        elif len(timezone_text) == 5:
            timezone_text = timezone_text[:3] + ":" + timezone_text[3:]
        try:
            seconds = datetime.fromisoformat(match.group("base") + timezone_text)
        except ValueError:
            return None
        fraction = (match.group("fraction") or "")[:9].ljust(9, "0")
        return _datetime_to_ns(seconds) + int(fraction or "0")
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _datetime_to_ns(dt)


def ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    if not nanoseconds:
        return base
    return base.replace("+00:00", f".{nanoseconds:09d}+00:00")


def file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)  # type: ignore[arg-type]
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
        if self.session_id is None:
            self.session_id = stable_hex(
                "agentstracer-session", self.source, self.source_session_id,
                nbytes=16,
            )
        self.tags = list(dict.fromkeys([self.source, "agentstracer", *self.tags]))
        fingerprint_payload = {
            "logical_id": self.logical_id,
            "name": self.name,
            "source": self.source,
            "source_session_id": self.source_session_id,
            "session_id": self.session_id,
            "project": self.project,
            "metadata": self.metadata,
            "tags": self.tags,
            "environment": self.environment,
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
        sanitized: dict[str, Any] = {}
        collision_counts: dict[str, int] = {}
        for raw_key, item_value in value.items():
            key_text = str(raw_key)
            sanitized_key = _sanitize_string(
                key_text,
                key=None,
                anonymizer=anonymizer,
                custom_strings=custom_strings,
                redact_secrets=redact_secrets,
            )
            candidate = sanitized_key
            if candidate in sanitized:
                suffix = collision_counts.get(sanitized_key, 1) + 1
                candidate = f"{sanitized_key}__redacted_{suffix}"
                while candidate in sanitized:
                    suffix += 1
                    candidate = f"{sanitized_key}__redacted_{suffix}"
                collision_counts[sanitized_key] = suffix
            else:
                collision_counts[sanitized_key] = 1
            sanitized[candidate] = sanitize_value(
                item_value,
                anonymizer=anonymizer,
                custom_strings=custom_strings,
                redact_secrets=redact_secrets,
                key=key_text,
            )
        return sanitized
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
    # Normalize relationships first so any repair metadata introduced by
    # finalize() is included in the sanitization pass below.
    trace.finalize()
    trace.logical_id = _sanitize_string(
        trace.logical_id, key=None, anonymizer=anonymizer,
        custom_strings=strings, redact_secrets=redact_secrets,
    )
    trace.source = _sanitize_string(
        trace.source, key=None, anonymizer=anonymizer,
        custom_strings=strings, redact_secrets=redact_secrets,
    )
    trace.source_session_id = _sanitize_string(
        trace.source_session_id, key=None, anonymizer=anonymizer,
        custom_strings=strings, redact_secrets=redact_secrets,
    )
    if trace.session_id is not None:
        trace.session_id = _sanitize_string(
            trace.session_id, key=None, anonymizer=anonymizer,
            custom_strings=strings, redact_secrets=redact_secrets,
        )
    trace.environment = _sanitize_string(
        trace.environment, key=None, anonymizer=anonymizer,
        custom_strings=strings, redact_secrets=redact_secrets,
    )
    trace.tags = [
        _sanitize_string(
            tag, key=None, anonymizer=anonymizer,
            custom_strings=strings, redact_secrets=redact_secrets,
        )
        for tag in trace.tags
    ]
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
    identifier_map: dict[str, str] = {}
    used_identifiers: set[str] = set()
    for observation in trace.observations:
        raw_identifier = observation.logical_id
        base_identifier = _sanitize_string(
            raw_identifier, key=None, anonymizer=anonymizer,
            custom_strings=strings, redact_secrets=redact_secrets,
        )
        sanitized_identifier = base_identifier
        suffix = 2
        while sanitized_identifier in used_identifiers:
            sanitized_identifier = f"{base_identifier}__redacted_{suffix}"
            suffix += 1
        identifier_map[raw_identifier] = sanitized_identifier
        used_identifiers.add(sanitized_identifier)

    for observation in trace.observations:
        raw_parent = observation.parent_logical_id
        observation.logical_id = identifier_map[observation.logical_id]
        observation.parent_logical_id = (
            identifier_map.get(raw_parent, raw_parent) if raw_parent is not None else None
        )
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
            link.target_logical_id = identifier_map.get(
                link.target_logical_id, link.target_logical_id
            )
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
    now = fallback or _datetime_to_ns(datetime.now(tz=timezone.utc))
    return now, now
