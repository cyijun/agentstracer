"""Tests for the workbench daemon HTTP API."""

import json
from http.client import HTTPConnection
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest

from agentstracer.daemon import WorkbenchHandler, run_server
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
        from http.server import ThreadingHTTPServer

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
