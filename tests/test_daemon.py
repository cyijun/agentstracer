"""Tests for the workbench daemon HTTP API."""

import json
import io
import zipfile
from http.client import HTTPConnection
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest

from agentstracer.daemon import Scanner, WorkbenchHandler, run_server
from agentstracer.index import open_index, upsert_sessions


@pytest.fixture
def index_setup(tmp_path, monkeypatch):
    """Set up an index DB in a temp directory and seed it."""
    monkeypatch.setattr("agentstracer.index.INDEX_DB", tmp_path / "index.db")
    monkeypatch.setattr("agentstracer.index.BLOBS_DIR", tmp_path / "blobs")
    monkeypatch.setattr("agentstracer.index.CONFIG_DIR", tmp_path / "agentstracer_config")
    monkeypatch.setattr("agentstracer.daemon.CONFIG_DIR", tmp_path / "agentstracer_config")
    monkeypatch.setattr("agentstracer.daemon.FRONTEND_DIST", tmp_path / "nonexistent_dist")

    conn = open_index()
    sessions = [
        {
            "session_id": f"sess-{i}",
            "project": "test-project",
            "source": "claude",
            "model": "claude-sonnet-4",
            "start_time": f"2025-01-0{i+1}T00:00:00+00:00",
            "end_time": f"2025-01-0{i+1}T00:10:00+00:00",
            "messages": [
                {"role": "user", "content": f"Task {i}: fix the bug", "tool_uses": []},
                {"role": "assistant", "content": "Done.", "tool_uses": []},
            ],
            "stats": {
                "user_messages": 1, "assistant_messages": 1,
                "tool_uses": 0, "input_tokens": 100, "output_tokens": 50,
            },
        }
        for i in range(3)
    ]
    upsert_sessions(conn, sessions)
    conn.close()
    return tmp_path


@pytest.fixture
def server(index_setup):
    """Start a test HTTP server."""
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
    port = srv.server_address[1]
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield port
    srv.shutdown()


def _get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    return resp.status, json.loads(body) if resp.getheader("Content-Type", "").startswith("application/json") else body


def _request_raw(port, method, path, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    payload = resp.read()
    return resp.status, dict(resp.getheaders()), payload


def _post(port, path, data=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(data or {}).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp_body = resp.read().decode()
    return resp.status, json.loads(resp_body) if resp.getheader("Content-Type", "").startswith("application/json") else resp_body


class TestSessionsAPI:
    def test_list_sessions(self, server):
        status, data = _get(server, "/api/sessions")
        assert status == 200
        assert len(data) == 3

    def test_list_sessions_with_limit(self, server):
        status, data = _get(server, "/api/sessions?limit=2")
        assert status == 200
        assert len(data) == 2

    def test_get_session_detail(self, server):
        status, data = _get(server, "/api/sessions/sess-0")
        assert status == 200
        assert data["session_id"] == "sess-0"
        assert "messages" in data

    def test_get_session_not_found(self, server):
        status, data = _get(server, "/api/sessions/nonexistent")
        assert status == 404

    def test_update_session_status(self, server):
        status, data = _post(server, "/api/sessions/sess-0", {"status": "approved"})
        assert status == 200
        assert data["ok"] is True

        # Verify it persisted
        status, detail = _get(server, "/api/sessions/sess-0")
        assert detail["review_status"] == "approved"

    def test_rejects_cross_origin_browser_request(self, server):
        status, _, payload = _request_raw(
            server, "GET", "/api/sessions", headers={"Origin": "https://evil.example"},
        )
        assert status == 403
        assert b"Cross-origin" in payload

    def test_rejects_dns_rebinding_host(self, server):
        status, _, payload = _request_raw(
            server, "GET", "/api/sessions", headers={"Host": "evil.example"},
        )
        assert status == 403
        assert b"Loopback" in payload

    def test_accepts_exact_loopback_origin_without_wildcard_cors(self, server):
        status, headers, _ = _request_raw(
            server, "GET", "/api/sessions",
            headers={"Origin": f"http://127.0.0.1:{server}"},
        )
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers

    @pytest.mark.parametrize("path", [
        "/api/sessions?limit=-1", "/api/sessions?limit=nope", "/api/search?q=x&offset=-1",
    ])
    def test_invalid_pagination_returns_json_400(self, server, path):
        status, data = _get(server, path)
        assert status == 400
        assert "error" in data

    def test_rejects_non_object_json(self, server):
        status, _, payload = _request_raw(
            server, "POST", "/api/scan", body=b"[]",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert b"must be an object" in payload

    def test_rejects_non_json_content_type(self, server):
        status, _, _ = _request_raw(
            server, "POST", "/api/scan", body=b"{}",
            headers={"Content-Type": "text/plain"},
        )
        assert status == 415

    def test_invalid_score_returns_json_400(self, server):
        status, data = _post(server, "/api/sessions/sess-0", {"ai_quality_score": "bad"})
        assert status == 400
        assert "error" in data


class TestStatsAPI:
    def test_stats(self, server):
        status, data = _get(server, "/api/stats")
        assert status == 200
        assert data["total"] == 3
        assert "by_status" in data
        assert "by_source" in data


class TestProjectsAPI:
    def test_projects(self, server):
        status, data = _get(server, "/api/projects")
        assert status == 200
        assert len(data) >= 1
        assert data[0]["project"] == "test-project"


class TestBundlesAPI:
    def test_create_and_list(self, server):
        status, data = _post(server, "/api/bundles", {
            "session_ids": ["sess-0", "sess-1"],
            "note": "Test bundle",
        })
        assert status == 201
        assert "bundle_id" in data

        status, bundles = _get(server, "/api/bundles")
        assert status == 200
        assert len(bundles) == 1

    def test_create_empty_fails(self, server):
        status, data = _post(server, "/api/bundles", {"session_ids": []})
        assert status == 400

    def test_create_rejects_string_instead_of_id_list(self, server):
        status, _ = _post(server, "/api/bundles", {"session_ids": "sess-0"})
        assert status == 400

    def test_download_uses_redacted_export_pipeline(self, server):
        from agentstracer.index import create_bundle, open_index, upsert_sessions

        token = "ghp_AbCdEf0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        custom = "InternalProjectZephyr"
        conn = open_index()
        upsert_sessions(conn, [{
            "session_id": "secret-session", "project": "project", "source": "claude",
            "messages": [{
                "role": "user", "content": f"use {token}",
                "tool_uses": [{
                    "tool": "bash", "input": {"command": f"deploy {custom} --token {token}"},
                    "output": "ok", "status": "success",
                }],
            }],
            "stats": {"user_messages": 1, "assistant_messages": 0, "tool_uses": 1},
        }])
        bundle_id = create_bundle(
            conn, ["secret-session"], attestation=f"approved with {token} for {custom}",
        )
        conn.close()

        with patch("agentstracer.daemon.load_config", return_value={
            "redact_strings": [custom], "allowlist_entries": [], "redact_usernames": [],
        }):
            status, headers, payload = _request_raw(
                server, "GET", f"/api/bundles/{bundle_id}/download",
            )
        assert status == 200
        assert "hostname" not in headers.get("Content-Disposition", "")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            exported = archive.read("sessions.jsonl").decode()
            manifest = archive.read("manifest.json").decode()
        assert token not in exported
        assert custom not in exported
        assert token not in manifest
        assert custom not in manifest


class TestPoliciesAPI:
    def test_add_and_list(self, server):
        status, data = _post(server, "/api/policies", {
            "policy_type": "redact_string",
            "value": "my-secret",
            "reason": "API key",
        })
        assert status == 201

        status, policies = _get(server, "/api/policies")
        assert status == 200
        assert len(policies) == 1

    def test_add_missing_fields(self, server):
        status, data = _post(server, "/api/policies", {"policy_type": "redact_string"})
        assert status == 400


class TestStaticServing:
    def test_placeholder_when_no_frontend(self, server):
        conn = HTTPConnection("127.0.0.1", server, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read().decode()
        assert resp.status == 200
        assert "AgentsTrace Workbench" in body


class TestRunServerPortFallback:
    def test_fallback_to_free_port_on_oserror(self, index_setup):
        """If the default port is busy, run_server falls back without opening a browser."""
        real_server = MagicMock()
        real_server.server_address = ("127.0.0.1", 9999)
        real_server.serve_forever.side_effect = KeyboardInterrupt

        call_count = 0

        def fake_init(addr, handler):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Address already in use")
            return real_server

        with patch("agentstracer.daemon.ThreadingHTTPServer", side_effect=fake_init), \
             patch("agentstracer.daemon.Scanner"), \
             patch("webbrowser.open") as mock_open:
            run_server(port=8384, open_browser=True)

        mock_open.assert_not_called()


class TestScanner:
    def test_unchanged_project_is_not_reparsed(self):
        scanner = Scanner()
        project = {
            "source": "claude", "dir_name": "project", "session_count": 1,
            "total_size_bytes": 100, "latest_mtime_ns": 123,
        }
        connection = MagicMock()
        with patch("agentstracer.daemon.open_index", return_value=connection), \
             patch("agentstracer.daemon.discover_projects", return_value=[project]), \
             patch("agentstracer.daemon.parse_project_sessions", return_value=[{"session_id": "x"}]) as parse, \
             patch("agentstracer.daemon.upsert_sessions", return_value=1):
            assert scanner.scan_once() == {"claude": 1}
            assert scanner.scan_once() == {}
        assert parse.call_count == 1


class TestShareAPI:
    """The workbench remains local-only; Langfuse has its own explicit CLI."""

    def _create_and_export_bundle(self, port):
        """Helper: create a bundle and export it, return bundle_id."""
        status, data = _post(port, "/api/bundles", {
            "session_ids": ["sess-0", "sess-1"],
            "note": "Share test bundle",
        })
        assert status == 201
        bundle_id = data["bundle_id"]

        status, data = _post(port, f"/api/bundles/{bundle_id}/export")
        assert status == 200
        assert data["ok"] is True
        return bundle_id

    def test_share_is_explicitly_disabled(self, server):
        bundle_id = self._create_and_export_bundle(server)
        status, data = _post(server, f"/api/bundles/{bundle_id}/share")
        assert status == 503
        assert "local-only" in data["error"]
