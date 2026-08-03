"""OTLP serialization, transport, verification, and idempotency tests."""

import json
import stat
import urllib.error
from io import BytesIO
from unittest.mock import patch

import pytest

from agentstracer.langfuse_export import (
    LangfuseConfig,
    LangfuseExportError,
    LangfuseOTLPClient,
    observation_to_otlp,
    smoke_test,
    sync_traces,
)
from agentstracer.trace_model import AgentTrace, TraceLink, TraceObservation


def _trace():
    return AgentTrace(
        logical_id="synthetic:turn:0",
        name="synthetic-agent-trace",
        source="synthetic",
        source_session_id="synthetic-session",
        project="test-project",
        observations=[
            TraceObservation(
                "root", "run-agent", "agent",
                1_700_000_000_000_000_000,
                1_700_000_002_000_000_000,
                input={"task": "test"}, output={"ok": True},
            ),
            TraceObservation(
                "generation", "generate", "generation",
                1_700_000_000_100_000_000,
                1_700_000_001_000_000_000,
                "root", input="hello", output="world", model="test-model",
                usage={"input": 3, "output": 2},
            ),
            TraceObservation(
                "tool", "search", "tool",
                1_700_000_001_100_000_000,
                1_700_000_001_900_000_000,
                "root", input={"q": "x"}, output={"hits": 1},
            ),
        ],
    ).finalize()


def _attributes(span):
    result = {}
    for attribute in span["attributes"]:
        value = attribute["value"]
        result[attribute["key"]] = next(iter(value.values()))
    return result


def test_config_normalizes_supported_endpoint_forms(monkeypatch):
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.local:3000/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    config = LangfuseConfig.from_env()
    assert config.traces_url == "http://langfuse.local:3000/api/public/otel/v1/traces"
    assert config.observations_url == "http://langfuse.local:3000/api/public/v2/observations"
    assert config.should_bypass_proxy is True
    assert "public" not in repr(config.auth_header)
    assert "secret" not in repr(config)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    with pytest.raises(LangfuseExportError, match="LANGFUSE_SECRET_KEY"):
        LangfuseConfig.from_env()


@pytest.mark.parametrize("base_url", [
    "ftp://langfuse.example.com",
    "https://user:password@langfuse.example.com",
    "https://langfuse.example.com?token=secret",
    "//langfuse.example.com",
    "https://langfuse.example.com\\@evil.example",
])
def test_config_rejects_unsafe_base_urls(base_url):
    with pytest.raises(LangfuseExportError):
        LangfuseConfig(base_url, "public", "secret")


def test_otlp_span_contains_langfuse_v4_attributes_and_parent_ids():
    trace = _trace()
    span = observation_to_otlp(trace, trace.observations[1])
    attrs = _attributes(span)
    assert span["traceId"] == trace.trace_id
    assert span["parentSpanId"] == trace.span_id("root")
    assert attrs["langfuse.observation.type"] == "generation"
    assert attrs["langfuse.observation.model.name"] == "test-model"
    assert json.loads(attrs["langfuse.observation.usage_details"]) == {
        "input": 3, "output": 2,
    }
    assert "langfuse.trace.input" not in attrs
    assert "langfuse.trace.output" not in attrs
    assert attrs["session.id"] == trace.session_id
    assert attrs["langfuse.session.id"] == trace.session_id

    root_attrs = _attributes(observation_to_otlp(trace, trace.root))
    assert json.loads(root_attrs["langfuse.observation.input"]) == {"task": "test"}
    assert json.loads(root_attrs["langfuse.observation.output"]) == {"ok": True}


def test_otlp_serializes_causal_span_links():
    trace = _trace()
    trace.observations[2].links.append(TraceLink(
        "generation", {"agentstracer.relationship": "requested_by"},
    ))
    span = observation_to_otlp(trace, trace.observations[2])
    assert span["links"][0]["traceId"] == trace.trace_id
    assert span["links"][0]["spanId"] == trace.span_id("generation")


def test_otlp_http_json_transport_uses_v4_header():
    received = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    def urlopen(request, timeout):
        received.append({
            "url": request.full_url,
            "auth": request.headers.get("Authorization"),
            "version": request.headers.get("X-langfuse-ingestion-version"),
            "body": json.loads(request.data),
        })
        return Response()

    config = LangfuseConfig(
        base_url="http://langfuse.example.com:3000",
        public_key="public", secret_key="secret",
        ingestion_version="4",
    )
    with patch("agentstracer.langfuse_export.urllib.request.urlopen", side_effect=urlopen):
        assert LangfuseOTLPClient(config).export_trace(_trace(), batch_spans=2) == 3

    assert len(received) == 2
    assert all(item["url"].endswith("/api/public/otel/v1/traces") for item in received)
    assert all(item["version"] == "4" for item in received)
    assert all(item["auth"] == config.auth_header for item in received)
    span_count = sum(
        len(item["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"])
        for item in received
    )
    assert span_count == 3


def test_export_batches_are_bounded_by_serialized_bytes():
    requests = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self, _size=None):
            return b"{}"

    def urlopen(request, timeout):
        requests.append(request)
        return Response()

    config = LangfuseConfig(
        "http://langfuse.example.com:3000", "public", "secret",
        ingestion_version="4",
    )
    with patch("agentstracer.langfuse_export.urllib.request.urlopen", side_effect=urlopen):
        LangfuseOTLPClient(config).export_trace(
            _trace(), batch_spans=3, max_batch_bytes=1,
        )
    assert len(requests) == 3


def test_transport_rejects_oversized_response():
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self, _size=None):
            return b"too-large"

    config = LangfuseConfig(
        "http://langfuse.example.com:3000", "public", "secret",
        ingestion_version="4",
    )
    with patch("agentstracer.langfuse_export.MAX_RESPONSE_BYTES", 2), \
         patch("agentstracer.langfuse_export.urllib.request.urlopen", return_value=Response()):
        with pytest.raises(LangfuseExportError, match="safety limit"):
            LangfuseOTLPClient(config).export_trace(_trace())


def test_v4_observation_read_follows_cursor_pages():
    client = LangfuseOTLPClient(LangfuseConfig(
        "http://langfuse.example.com:3000", "public", "secret",
        ingestion_version="4",
    ))
    with patch.object(client, "_observation_request", side_effect=[
        {"data": [{"id": "one"}], "meta": {"cursor": "next-page"}},
        {"data": [{"id": "two"}], "meta": {"cursor": None}},
    ]) as request:
        result = client.get_observations("trace", limit=2)
    assert [row["id"] for row in result["data"]] == ["one", "two"]
    assert "cursor=next-page" in request.call_args_list[1].args[0]


def test_v3_observation_read_follows_numbered_pages():
    client = LangfuseOTLPClient(LangfuseConfig(
        "http://langfuse.example.com:3000", "public", "secret",
        ingestion_version="3",
    ))
    with patch.object(client, "_observation_request", side_effect=[
        {"data": [{"id": "one"}], "meta": {"totalPages": 2}},
        {"data": [{"id": "two"}], "meta": {"totalPages": 2}},
    ]) as request:
        result = client.get_observations("trace", limit=2)
    assert [row["id"] for row in result["data"]] == ["one", "two"]
    assert "page=2" in request.call_args_list[1].args[0]


def test_doctor_falls_back_to_langfuse_v3_observations_api():
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":[],"meta":{}}'

    urls = []

    def urlopen(request, timeout):
        urls.append(request.full_url)
        if "/api/public/v2/observations" in request.full_url:
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {},
                BytesIO(b'{"message":"only available in v4 write mode"}'),
            )
        return Response()

    config = LangfuseConfig(
        base_url="http://langfuse.example.com:3000",
        public_key="public", secret_key="secret",
    )
    with patch("agentstracer.langfuse_export.urllib.request.urlopen", side_effect=urlopen):
        result = LangfuseOTLPClient(config).doctor()
    assert result["ok"] is True
    assert result["ingestion_version"] == "3"
    assert result["api"] == "observations-v1"
    assert any("/api/public/observations?limit=1" in url for url in urls)


def test_get_observations_caps_limit_for_each_langfuse_version():
    class Client(LangfuseOTLPClient):
        last_url = ""
        def _observation_request(self, url):
            self.last_url = url
            return {"data": []}

    v3 = Client(LangfuseConfig(
        base_url="http://langfuse.example.com:3000",
        public_key="public", secret_key="secret", ingestion_version="3",
    ))
    v4 = Client(LangfuseConfig(
        base_url="http://langfuse.example.com:3000",
        public_key="public", secret_key="secret", ingestion_version="4",
    ))
    v3.get_observations("trace", limit=900)
    v4.get_observations("trace", limit=900)
    assert "limit=100" in v3.last_url
    assert "limit=900" in v4.last_url


def test_langfuse_v3_export_omits_v4_ingestion_header():
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    def urlopen(request, timeout):
        requests.append(request)
        return Response()

    config = LangfuseConfig(
        base_url="http://langfuse.example.com:3000",
        public_key="public", secret_key="secret", ingestion_version="3",
    )
    with patch("agentstracer.langfuse_export.urllib.request.urlopen", side_effect=urlopen):
        LangfuseOTLPClient(config).export_trace(_trace())
    assert requests[0].headers.get("X-langfuse-ingestion-version") is None


def test_smoke_test_writes_and_reads_safe_trace():
    class Client:
        def export_trace(self, trace):
            assert trace.source == "synthetic"
            return len(trace.observations)

        def get_observations(self, trace_id, limit=20):
            return {"data": [
                {"name": "run-synthetic-agent", "type": "SPAN"},
                {"name": "generate-synthetic", "type": "GENERATION", "parentObservationId": "root"},
                {"name": "synthetic-tool", "type": "SPAN", "parentObservationId": "root"},
                {"name": "synthetic-event", "type": "EVENT", "parentObservationId": "root"},
            ]}

        def _resolve_ingestion_version(self):
            return "3"

    result = smoke_test(Client(), verify_attempts=1, verify_delay_seconds=0)
    assert result["ok"] is True
    assert result["observations_read"] == 4
    assert result["parented"] == 3


def test_sync_ledger_skips_successful_snapshot_and_polls_verification(tmp_path):
    class Client:
        def __init__(self):
            self.export_calls = 0
            self.read_calls = 0

        def export_trace(self, trace):
            self.export_calls += 1
            return len(trace.observations)

        def get_observations(self, trace_id, limit=1000):
            self.read_calls += 1
            return {
                "data": [] if self.read_calls == 1 else [
                    {"traceId": trace_id, "name": "run-agent", "type": "SPAN"},
                    {"traceId": trace_id, "name": "generate", "type": "GENERATION",
                     "parentObservationId": "root"},
                    {"traceId": trace_id, "name": "search", "type": "SPAN",
                     "parentObservationId": "root"},
                ],
            }

    client = Client()
    ledger = tmp_path / "ledger.db"
    first = sync_traces(
        [_trace()], client=client, ledger_path=ledger, verify=True,
        verify_attempts=3, verify_delay_seconds=0,
    )
    second = sync_traces(
        [_trace()], client=client, ledger_path=ledger, verify=True,
        verify_attempts=1, verify_delay_seconds=0,
    )
    assert first["traces_exported"] == 1
    assert first["verified"] == 1
    assert first["verification_failed"] == 0
    assert second["traces_skipped"] == 1
    assert second["verified"] == 1
    assert second["verified_tool_names"] == ["search"]
    assert client.export_calls == 1
    assert client.read_calls == 3
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600


def test_verification_failure_does_not_make_accepted_trace_retryable(tmp_path):
    class Client:
        def __init__(self):
            self.export_calls = 0

        def export_trace(self, trace):
            self.export_calls += 1
            return len(trace.observations)

        def get_observations(self, trace_id, limit=1000):
            return {"data": []}

    client = Client()
    ledger = tmp_path / "ledger.db"
    result = sync_traces(
        [_trace()], client=client, ledger_path=ledger, verify=True,
        verify_attempts=2, verify_delay_seconds=0,
    )
    retry = sync_traces([_trace()], client=client, ledger_path=ledger)
    assert result["traces_failed"] == 0
    assert result["verification_failed"] == 1
    assert retry["traces_skipped"] == 1
    assert client.export_calls == 1


def test_partial_batch_progress_resumes_without_resending(tmp_path):
    class Client(LangfuseOTLPClient):
        def __init__(self):
            super().__init__(LangfuseConfig(
                "https://langfuse.example.com", "public", "secret",
                ingestion_version="4",
            ))
            self.requests = []
            self.fail_second_request = True

        def export_trace(self, trace, *, start_offset=0, on_progress=None, **kwargs):
            return super().export_trace(
                trace,
                batch_spans=2,
                start_offset=start_offset,
                on_progress=on_progress,
                **kwargs,
            )

        def _request(self, request, *, retry):
            body = json.loads(request.data)
            spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
            self.requests.append([span["name"] for span in spans])
            if self.fail_second_request and len(self.requests) == 2:
                self.fail_second_request = False
                raise LangfuseExportError("second batch failed")
            return 200, b"{}"

    client = Client()
    ledger = tmp_path / "resume.db"
    first = sync_traces([_trace()], client=client, ledger_path=ledger)
    second = sync_traces([_trace()], client=client, ledger_path=ledger)
    assert first["traces_failed"] == 1
    assert second["traces_exported"] == 1
    assert client.requests == [
        ["run-agent", "generate"],
        ["search"],
        ["search"],
    ]
