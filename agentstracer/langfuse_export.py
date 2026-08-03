"""Dependency-free Langfuse v4 exporter using OTLP/HTTP JSON.

Langfuse v4 observations are immutable OpenTelemetry spans.  Historical agent
logs therefore need complete spans, deterministic identifiers, exact source
timestamps, and a local export ledger that prevents accidental re-ingestion.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import random
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .config import CONFIG_DIR
from .trace_model import AgentTrace, TraceObservation, canonical_json, ns_to_iso

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = CONFIG_DIR / "langfuse_sync.db"
DEFAULT_BATCH_SPANS = 200
DEFAULT_MAX_IO_CHARS = 200_000


class LangfuseExportError(RuntimeError):
    pass


class LangfuseHTTPError(LangfuseExportError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Langfuse HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


@dataclass(frozen=True, slots=True)
class LangfuseConfig:
    base_url: str
    public_key: str
    secret_key: str = field(repr=False)
    timeout_seconds: float = 30.0
    max_retries: int = 3
    bypass_proxy: bool | None = None
    ingestion_version: str = "auto"

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        missing = [
            name for name, value in (
                ("LANGFUSE_BASE_URL", base_url),
                ("LANGFUSE_PUBLIC_KEY", public_key),
                ("LANGFUSE_SECRET_KEY", secret_key),
            )
            if not value
        ]
        if missing:
            raise LangfuseExportError(
                "missing Langfuse environment variables: " + ", ".join(missing)
            )
        ingestion_version = os.environ.get("LANGFUSE_INGESTION_VERSION", "auto").strip()
        if ingestion_version not in ("auto", "3", "4"):
            raise LangfuseExportError(
                "LANGFUSE_INGESTION_VERSION must be auto, 3, or 4"
            )
        return cls(
            base_url=str(base_url).rstrip("/"),
            public_key=str(public_key),
            secret_key=str(secret_key),
            ingestion_version=ingestion_version,
        )

    @property
    def traces_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/api/public/otel/v1/traces"):
            return base
        if base.endswith("/api/public/otel"):
            return base + "/v1/traces"
        return base + "/api/public/otel/v1/traces"

    @property
    def observations_url(self) -> str:
        base = self.base_url.rstrip("/")
        for suffix in ("/api/public/otel/v1/traces", "/api/public/otel"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base + "/api/public/v2/observations"

    @property
    def observations_v1_url(self) -> str:
        base = self.base_url.rstrip("/")
        for suffix in ("/api/public/otel/v1/traces", "/api/public/otel"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return base + "/api/public/observations"

    @property
    def auth_header(self) -> str:
        raw = f"{self.public_key}:{self.secret_key}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    @property
    def should_bypass_proxy(self) -> bool:
        """Direct-connect local/private Langfuse hosts unless overridden.

        macOS agent environments commonly set a global HTTP proxy whose
        ``no_proxy`` list does not include internal single-label hostnames.
        Sending those credentials to that proxy both fails and is avoidable.
        """
        if self.bypass_proxy is not None:
            return self.bypass_proxy
        override = os.environ.get("LANGFUSE_BYPASS_PROXY")
        if override is not None:
            return override.strip().lower() not in ("0", "false", "no", "off")
        hostname = urllib.parse.urlparse(self.base_url).hostname or ""
        if hostname == "localhost" or hostname.endswith(".local") or "." not in hostname:
            return True
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False
        return address.is_private or address.is_loopback or address.is_link_local


def _any_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"stringValue": "null"}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        # Protobuf JSON encodes int64 values as decimal strings.
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, bool, int, float)) for item in value
    ):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": canonical_json(value)}


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": _any_value(value)}


def _json_io(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    encoded = canonical_json(value)
    if len(encoded) <= max_chars:
        return encoded
    preview = encoded[: max(0, max_chars - 120)]
    return canonical_json({
        "_agentstracer_truncated": True,
        "original_chars": len(encoded),
        "preview": preview,
    })


def _metadata_attributes(prefix: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    attributes: list[dict[str, Any]] = []
    for raw_key, value in metadata.items():
        if value is None:
            continue
        # Langfuse reserves dot-separated metadata paths.  Keep source keys
        # flat and safe so they remain filterable in the UI.
        key = "".join(char if char.isalnum() or char == "_" else "_" for char in str(raw_key))
        if not key or any(segment in ("__proto__", "constructor", "prototype") for segment in key.split(".")):
            continue
        if isinstance(value, (dict, list, tuple)):
            value = canonical_json(value)
        attributes.append(_attribute(f"{prefix}.{key}", value))
    return attributes


def _trace_attributes(trace: AgentTrace, max_io_chars: int) -> list[dict[str, Any]]:
    metadata = {
        "source": trace.source,
        "logical_trace_id": trace.logical_id,
        "source_session_id": trace.source_session_id,
        "content_hash": trace.content_hash,
        "project": trace.project,
        **trace.metadata,
    }
    attributes = [
        _attribute("langfuse.trace.name", trace.name),
        _attribute("langfuse.session.id", trace.session_id),
        _attribute("session.id", trace.session_id),
        _attribute("langfuse.trace.tags", trace.tags),
        _attribute("langfuse.environment", trace.environment),
        _attribute("langfuse.version", __version__),
        *_metadata_attributes("langfuse.trace.metadata", metadata),
    ]
    trace_input = _json_io(trace.root.input, max_io_chars)
    trace_output = _json_io(trace.root.output, max_io_chars)
    if trace_input is not None:
        attributes.append(_attribute("langfuse.trace.input", trace_input))
    if trace_output is not None:
        attributes.append(_attribute("langfuse.trace.output", trace_output))
    return attributes


def observation_to_otlp(
    trace: AgentTrace,
    observation: TraceObservation,
    *,
    max_io_chars: int = DEFAULT_MAX_IO_CHARS,
) -> dict[str, Any]:
    if trace.trace_id is None:
        raise ValueError("trace must be finalized before OTLP serialization")
    attributes = [
        *_trace_attributes(trace, max_io_chars),
        _attribute("langfuse.observation.type", observation.as_type),
        _attribute("langfuse.observation.level", observation.level),
    ]
    input_value = _json_io(observation.input, max_io_chars)
    output_value = _json_io(observation.output, max_io_chars)
    if input_value is not None:
        attributes.append(_attribute("langfuse.observation.input", input_value))
    if output_value is not None:
        attributes.append(_attribute("langfuse.observation.output", output_value))
    if observation.status_message:
        attributes.append(_attribute(
            "langfuse.observation.status_message", observation.status_message
        ))
    if observation.model:
        attributes.append(_attribute("langfuse.observation.model.name", observation.model))
    if observation.model_parameters:
        attributes.append(_attribute(
            "langfuse.observation.model.parameters",
            canonical_json(observation.model_parameters),
        ))
    if observation.usage:
        attributes.append(_attribute(
            "langfuse.observation.usage_details", canonical_json(observation.usage)
        ))
    if observation.cost:
        attributes.append(_attribute(
            "langfuse.observation.cost_details", canonical_json(observation.cost)
        ))
    if observation.completion_start_ns is not None:
        attributes.append(_attribute(
            "langfuse.observation.completion_start_time",
            ns_to_iso(observation.completion_start_ns),
        ))
    attributes.extend(_metadata_attributes(
        "langfuse.observation.metadata", observation.metadata
    ))

    span: dict[str, Any] = {
        "traceId": trace.trace_id,
        "spanId": trace.span_id(observation.logical_id),
        "name": observation.name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(observation.start_ns),
        "endTimeUnixNano": str(observation.end_ns),
        "attributes": attributes,
        "status": {
            "code": 2 if observation.level == "ERROR" else 1,
            **({"message": observation.status_message} if observation.status_message else {}),
        },
        "flags": 1,
    }
    if observation.parent_logical_id:
        span["parentSpanId"] = trace.span_id(observation.parent_logical_id)
    if observation.links:
        span["links"] = [
            {
                "traceId": trace.trace_id,
                "spanId": trace.span_id(link.target_logical_id),
                "attributes": [
                    _attribute(str(key), value) for key, value in link.attributes.items()
                ],
            }
            for link in observation.links
        ]
    return span


def otlp_request(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                _attribute("service.name", "agentstracer"),
                _attribute("service.version", __version__),
                _attribute("telemetry.sdk.language", "python"),
            ]},
            "scopeSpans": [{
                "scope": {"name": "agentstracer.langfuse", "version": __version__},
                "spans": spans,
            }],
        }],
    }


def _topological_observations(trace: AgentTrace) -> list[TraceObservation]:
    pending = {observation.logical_id: observation for observation in trace.observations}
    emitted: set[str] = set()
    ordered: list[TraceObservation] = []
    while pending:
        ready = [
            observation for observation in pending.values()
            if observation.parent_logical_id is None or observation.parent_logical_id in emitted
        ]
        if not ready:
            raise ValueError(f"cannot topologically order trace {trace.logical_id}")
        ready.sort(key=lambda item: (item.start_ns, item.end_ns, item.logical_id))
        for observation in ready:
            ordered.append(observation)
            emitted.add(observation.logical_id)
            pending.pop(observation.logical_id)
    return ordered


class LangfuseOTLPClient:
    def __init__(self, config: LangfuseConfig):
        self.config = config
        self._urlopen = (
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open
            if config.should_bypass_proxy
            else urllib.request.urlopen
        )
        self._resolved_ingestion_version: str | None = (
            config.ingestion_version if config.ingestion_version in ("3", "4") else None
        )

    def _request(
        self,
        request: urllib.request.Request,
        *,
        retry: bool,
    ) -> tuple[int, bytes]:
        attempts = self.config.max_retries if retry else 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self._urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as exc:
                body = exc.read(4096).decode("utf-8", errors="replace")
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 >= attempts:
                    raise LangfuseHTTPError(exc.code, body) from exc
                last_error = exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    delay = 2**attempt
                time.sleep(delay + random.random() * 0.25)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(2**attempt + random.random() * 0.25)
        raise LangfuseExportError(f"could not reach Langfuse: {last_error}") from last_error

    def _observation_request(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": self.config.auth_header,
                "Accept": "application/json",
                "User-Agent": f"agentstracer/{__version__}",
            },
        )
        _, body = self._request(request, retry=True)
        try:
            value = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise LangfuseExportError("Langfuse returned non-JSON observation data") from exc
        return value if isinstance(value, dict) else {"data": value}

    def _resolve_ingestion_version(self) -> str:
        if self._resolved_ingestion_version is not None:
            return self._resolved_ingestion_version
        now = datetime.now(tz=timezone.utc)
        query = urllib.parse.urlencode({
            "fromStartTime": (now - timedelta(days=3650)).isoformat(),
            "toStartTime": (now + timedelta(days=1)).isoformat(),
            "limit": 1,
        })
        try:
            self._observation_request(self.config.observations_url + "?" + query)
            self._resolved_ingestion_version = "4"
        except LangfuseHTTPError as exc:
            if exc.status != 404:
                raise
            # Langfuse v3 exposes the v2 route but returns a specific 404 when
            # the deployment is not in v4 write mode.
            self._resolved_ingestion_version = "3"
        return self._resolved_ingestion_version

    def export_trace(
        self,
        trace: AgentTrace,
        *,
        batch_spans: int = DEFAULT_BATCH_SPANS,
        max_io_chars: int = DEFAULT_MAX_IO_CHARS,
    ) -> int:
        ingestion_version = self._resolve_ingestion_version()
        ordered = _topological_observations(trace)
        exported = 0
        for offset in range(0, len(ordered), batch_spans):
            chunk = ordered[offset: offset + batch_spans]
            spans = [
                observation_to_otlp(trace, observation, max_io_chars=max_io_chars)
                for observation in chunk
            ]
            payload = json.dumps(otlp_request(spans), separators=(",", ":")).encode("utf-8")
            headers = {
                "Authorization": self.config.auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"agentstracer/{__version__}",
            }
            if ingestion_version == "4":
                headers["x-langfuse-ingestion-version"] = "4"
            request = urllib.request.Request(
                self.config.traces_url,
                method="POST",
                data=payload,
                headers=headers,
            )
            _, body = self._request(request, retry=True)
            if body:
                try:
                    response = json.loads(body)
                except json.JSONDecodeError:
                    response = {}
                partial = response.get("partialSuccess") or response.get("partial_success")
                if isinstance(partial, dict) and int(partial.get("rejectedSpans") or partial.get("rejected_spans") or 0):
                    raise LangfuseExportError(
                        f"Langfuse rejected spans: {partial}"
                    )
            exported += len(chunk)
        return exported

    def get_observations(self, trace_id: str, limit: int = 1000) -> dict[str, Any]:
        version = self._resolve_ingestion_version()
        # Langfuse v3's public Observations endpoint rejects values above 100;
        # v4 raises the cap to 1000.  Verification only needs a non-empty page.
        maximum_limit = 100 if version == "3" else 1000
        params = {
            "traceId": trace_id,
            "limit": min(maximum_limit, max(1, limit)),
        }
        if version == "4":
            params["fields"] = "core,basic,time,usage,metadata,trace_context"
        url = self.config.observations_url if version == "4" else self.config.observations_v1_url
        return self._observation_request(url + "?" + urllib.parse.urlencode(params))

    def doctor(self) -> dict[str, Any]:
        version = self._resolve_ingestion_version()
        url = self.config.observations_url if version == "4" else self.config.observations_v1_url
        value = self._observation_request(url + "?limit=1")
        return {
            "ok": True,
            "status": 200,
            "base_url": self.config.base_url,
            "ingestion_version": version,
            "api": f"observations-v{2 if version == '4' else 1}",
            "response_has_data": isinstance(value, dict) and "data" in value,
        }


class SyncLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS langfuse_exports (
                trace_id TEXT PRIMARY KEY,
                logical_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                observation_count INTEGER NOT NULL,
                exported_at TEXT,
                error TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_langfuse_exports_logical ON langfuse_exports(logical_id, content_hash)"
        )
        self.conn.commit()


    def close(self) -> None:
        self.conn.close()

    def exported(self, trace: AgentTrace) -> bool:
        row = self.conn.execute(
            "SELECT status FROM langfuse_exports WHERE trace_id=?", (trace.trace_id,)
        ).fetchone()
        return bool(row and row[0] == "exported")

    def record(self, trace: AgentTrace, status: str, error: str | None = None) -> None:
        self.conn.execute("""
            INSERT INTO langfuse_exports (
                trace_id, logical_id, content_hash, source, status,
                observation_count, exported_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                status=excluded.status,
                observation_count=excluded.observation_count,
                exported_at=excluded.exported_at,
                error=excluded.error
        """, (
            trace.trace_id,
            trace.logical_id,
            trace.content_hash,
            trace.source,
            status,
            len(trace.observations),
            datetime.now(tz=timezone.utc).isoformat() if status == "exported" else None,
            error,
        ))
        self.conn.commit()


def smoke_test(
    client: LangfuseOTLPClient,
    *,
    verify_attempts: int = 20,
    verify_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Write and read back a content-safe synthetic agent trace."""
    now = time.time_ns()
    trace = AgentTrace(
        logical_id=f"integration-smoke:{now}",
        name="agentstracer-integration-smoke",
        source="synthetic",
        source_session_id="integration-smoke",
        environment="integration-test",
        project="agentstracer",
        observations=[
            TraceObservation(
                "root", "run-synthetic-agent", "agent", now, now + 4_000_000,
                input={"case": "synthetic"}, output={"ok": True},
            ),
            TraceObservation(
                "generation", "generate-synthetic", "generation",
                now + 1_000_000, now + 2_000_000, "root",
                input="ping", output="pong", model="synthetic-model",
                usage={"input": 1, "output": 1},
            ),
            TraceObservation(
                "tool", "synthetic-tool", "tool",
                now + 2_000_000, now + 3_000_000, "root",
                input={"value": 1}, output={"value": 2},
            ),
            TraceObservation(
                "event", "synthetic-event", "event",
                now + 3_500_000, now + 3_500_000, "root",
                input={"phase": "complete"},
            ),
        ],
    ).finalize()
    sent = client.export_trace(trace)
    data: list[dict[str, Any]] = []
    last_error: str | None = None
    reads = 0
    for attempt in range(max(1, verify_attempts)):
        reads += 1
        try:
            response = client.get_observations(trace.trace_id, limit=20)
            candidate = response.get("data") if isinstance(response, dict) else None
            data = candidate if isinstance(candidate, list) else []
            last_error = None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if len(data) >= len(trace.observations):
            break
        if attempt + 1 < max(1, verify_attempts):
            time.sleep(max(0.0, verify_delay_seconds))

    types: dict[str, int] = {}
    names: list[str] = []
    parented = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or item.get("observationType") or "unknown")
        types[item_type] = types.get(item_type, 0) + 1
        if item.get("parentObservationId") or item.get("parent_observation_id"):
            parented += 1
        if item.get("name"):
            names.append(str(item["name"]))
    expected_names = {observation.name for observation in trace.observations}
    name_set = set(names)
    ok = sent == len(trace.observations) and expected_names.issubset(name_set)
    return {
        "ok": ok,
        "trace_id": trace.trace_id,
        "ingestion_version": client._resolve_ingestion_version(),
        "spans_sent": sent,
        "observations_read": len(data),
        "types": types,
        "parented": parented,
        "names": sorted(names),
        "reads": reads,
        "last_error": last_error,
    }


def sync_traces(
    traces: Iterable[AgentTrace],
    *,
    client: LangfuseOTLPClient | None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    dry_run: bool = False,
    force: bool = False,
    verify: bool = False,
    verify_attempts: int = 6,
    verify_delay_seconds: float = 0.5,
    limit: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "traces_seen": 0,
        "traces_exported": 0,
        "traces_skipped": 0,
        "traces_failed": 0,
        "observations_seen": 0,
        "observations_exported": 0,
        "verified": 0,
        "verification_failed": 0,
        "verified_tool_names": [],
        "by_source": {},
        "errors": [],
        "trace_summaries": [],
    }
    verified_tool_names: set[str] = set()

    def verify_trace(trace: AgentTrace) -> None:
        assert client is not None
        verification_error: BaseException | None = None
        for attempt in range(max(1, verify_attempts)):
            try:
                response = client.get_observations(
                    trace.trace_id, limit=len(trace.observations)
                )
                data = response.get("data") if isinstance(response, dict) else None
                if isinstance(data, list) and data:
                    remote_names = {
                        str(item["name"])
                        for item in data
                        if isinstance(item, dict) and item.get("name")
                    }
                    expected_names = {item.name for item in trace.observations}
                    expected_tools = {
                        item.name for item in trace.observations if item.as_type == "tool"
                    }
                    verified_tool_names.update(remote_names & expected_tools)
                    # When the API returned the whole trace, verify node names
                    # as well as existence.  On capped pages, a subset is valid.
                    if (
                        len(data) >= len(trace.observations)
                        and not expected_names.issubset(remote_names)
                    ):
                        verification_error = LangfuseExportError(
                            "trace found but observation names do not match"
                        )
                        continue
                    summary["verified"] += 1
                    return
                verification_error = LangfuseExportError(
                    "trace accepted but not found during verification"
                )
            except Exception as exc:
                verification_error = exc
            if attempt + 1 < max(1, verify_attempts):
                time.sleep(max(0.0, verify_delay_seconds))
        summary["verification_failed"] += 1
        if len(summary["errors"]) < 20:
            summary["errors"].append({
                "trace_id": trace.trace_id,
                "logical_id": trace.logical_id,
                "error": "verification: " + str(verification_error),
            })

    ledger = None if dry_run else SyncLedger(ledger_path)
    try:
        for trace in traces:
            if limit is not None and summary["traces_seen"] >= limit:
                break
            if trace.trace_id is None:
                trace.finalize()
            summary["traces_seen"] += 1
            summary["observations_seen"] += len(trace.observations)
            source_stats = summary["by_source"].setdefault(trace.source, {
                "traces": 0, "observations": 0, "exported": 0, "failed": 0,
            })
            source_stats["traces"] += 1
            source_stats["observations"] += len(trace.observations)
            if len(summary["trace_summaries"]) < 20:
                summary["trace_summaries"].append(trace.summary())
            if dry_run:
                continue
            assert ledger is not None and client is not None
            if ledger.exported(trace) and not force:
                summary["traces_skipped"] += 1
                if verify:
                    verify_trace(trace)
                continue
            try:
                count = client.export_trace(trace)
                ledger.record(trace, "exported")
                summary["traces_exported"] += 1
                summary["observations_exported"] += count
                source_stats["exported"] += 1
                if verify:
                    verify_trace(trace)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                ledger.record(trace, "failed", message)
                summary["traces_failed"] += 1
                source_stats["failed"] += 1
                if len(summary["errors"]) < 20:
                    summary["errors"].append({
                        "trace_id": trace.trace_id,
                        "logical_id": trace.logical_id,
                        "error": message,
                    })
    finally:
        if ledger is not None:
            ledger.close()
    summary["verified_tool_names"] = sorted(verified_tool_names)
    return summary
