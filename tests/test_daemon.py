"""Tests for the workbench daemon HTTP API."""

import json
import time
import urllib.error
from http.client import HTTPConnection
from io import BytesIO
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest

from clawtrace.daemon import WorkbenchHandler, run_server, _SHARE_COOLDOWN_SECONDS
from clawtrace.index import open_index, upsert_sessions


@pytest.fixture
def index_setup(tmp_path, monkeypatch):
    """Set up an index DB in a temp directory and seed it."""
    monkeypatch.setattr("clawtrace.index.INDEX_DB", tmp_path / "index.db")
    monkeypatch.setattr("clawtrace.index.BLOBS_DIR", tmp_path / "blobs")
    monkeypatch.setattr("clawtrace.index.CONFIG_DIR", tmp_path / "clawtrace_config")
    monkeypatch.setattr("clawtrace.daemon.CONFIG_DIR", tmp_path / "clawtrace_config")
    monkeypatch.setattr("clawtrace.daemon.FRONTEND_DIST", tmp_path / "nonexistent_dist")

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
        assert "ClawTrace Workbench" in body


class TestRunServerPortFallback:
    def test_fallback_to_free_port_on_oserror(self, index_setup):
        """If the default port is busy, run_server falls back to port 0 and opens the browser."""
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

        with patch("clawtrace.daemon.ThreadingHTTPServer", side_effect=fake_init), \
             patch("clawtrace.daemon.Scanner"), \
             patch("webbrowser.open") as mock_open:
            run_server(port=8384, open_browser=True)

        mock_open.assert_called_once_with("http://localhost:9999/traces")


def _mock_urlopen_factory(register_response=None, upload_response=None, upload_error=None):
    """Create a mock urlopen that handles /register and /upload calls."""
    register_resp = register_response or {
        "device_id": "test-device-id-0000-0000-000000000000",
        "device_token": "test-device-token-abc123",
    }
    upload_resp = upload_response or {
        "ok": True,
    }

    def mock_urlopen(req, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/register" in url:
            resp = MagicMock()
            resp.read.return_value = json.dumps(register_resp).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp
        elif "/upload" in url:
            if upload_error:
                raise upload_error
            resp = MagicMock()
            resp.read.return_value = json.dumps(upload_resp).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    return mock_urlopen


class TestShareAPI:
    """Tests for the share-to-GCS HTTP upload flow."""

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

    def test_share_success(self, server, monkeypatch):
        """Full success path: create, export, share via HTTP."""
        # Reset rate limit timer
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        mock_urlopen = _mock_urlopen_factory()
        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 200
        assert data["ok"] is True
        assert "gcs_uri" not in data
        assert "shared_at" in data
        assert data["bundle_hash"]
        assert "redaction_summary" in data
        assert isinstance(data["redaction_summary"]["total_redactions"], int)
        assert isinstance(data["redaction_summary"]["by_type"], dict)

    def test_share_rate_limiting(self, server, monkeypatch):
        """Two shares within cooldown → second gets 429."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        mock_urlopen = _mock_urlopen_factory()
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")
        assert status == 200

        # Immediately try again — should be rate limited
        status, data = _post(server, f"/api/bundles/{bundle_id}/share")
        assert status == 429
        assert "Rate limited" in data["error"]

    def test_share_duplicate_prevention(self, server, monkeypatch):
        """Already-shared bundle → 409 (unless force=true)."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        mock_urlopen = _mock_urlopen_factory()
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, _ = _post(server, f"/api/bundles/{bundle_id}/share")
        assert status == 200

        # Second share without force → 409
        WorkbenchHandler._last_share_time = 0.0
        status, data = _post(server, f"/api/bundles/{bundle_id}/share")
        assert status == 409
        assert "already shared" in data["error"]

        # With force=true → should re-share
        WorkbenchHandler._last_share_time = 0.0
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share", {"force": True})
        assert status == 200
        assert data["ok"] is True

    def test_share_http_error(self, server, monkeypatch):
        """HTTP error from ingest → daemon returns 502."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        error_resp = BytesIO(json.dumps({"error": "Internal server error"}).encode())
        http_error = urllib.error.HTTPError(
            url="http://test/upload",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=error_resp,
        )

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        mock_urlopen = _mock_urlopen_factory(upload_error=http_error)
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 502
        assert "error" in data

    def test_share_cf_409_treated_as_success(self, server, monkeypatch):
        """Cloud Function 409 (already in GCS) → daemon treats as success."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        error_resp = BytesIO(json.dumps({"error": "Bundle already uploaded"}).encode())
        http_error = urllib.error.HTTPError(
            url="http://test/upload",
            code=409,
            msg="Conflict",
            hdrs={},  # type: ignore[arg-type]
            fp=error_resp,
        )

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        mock_urlopen = _mock_urlopen_factory(upload_error=http_error)
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 200
        assert data["ok"] is True
        assert "gcs_uri" not in data
        assert data["shared_at"]

    def test_share_network_failure(self, server, monkeypatch):
        """Network failure → daemon returns 502 with friendly message."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        network_error = urllib.error.URLError("Connection refused")

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "test-dev-id",
            "device_token": "test-dev-token",
        })
        mock_urlopen = _mock_urlopen_factory(upload_error=network_error)
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 502
        assert "Could not reach upload service" in data["error"]

    def test_device_token_auto_registered(self, server, monkeypatch, tmp_path):
        """Device token is auto-registered on first share."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        # Start with no device credentials
        saved_configs = []
        original_load = lambda: {"repo": None, "source": None, "excluded_projects": [], "redact_strings": []}

        def tracking_save(config):
            saved_configs.append(dict(config))

        monkeypatch.setattr("clawtrace.daemon.load_config", original_load)
        monkeypatch.setattr("clawtrace.daemon.save_config", tracking_save)

        mock_urlopen = _mock_urlopen_factory()
        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 200
        # Verify config was saved with device credentials
        assert len(saved_configs) == 1
        assert saved_configs[0]["device_id"] == "test-device-id-0000-0000-000000000000"
        assert saved_configs[0]["device_token"] == "test-device-token-abc123"

    def test_device_token_reused(self, server, monkeypatch):
        """Existing device token is reused without extra /register call."""
        WorkbenchHandler._last_share_time = 0.0

        bundle_id = self._create_and_export_bundle(server)

        monkeypatch.setattr("clawtrace.daemon.load_config", lambda: {
            "device_id": "existing-device-id",
            "device_token": "existing-device-token",
        })

        register_called = []

        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/register" in url:
                register_called.append(True)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "ok": True,
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("clawtrace.daemon.urllib.request.urlopen", side_effect=mock_urlopen):
            status, data = _post(server, f"/api/bundles/{bundle_id}/share")

        assert status == 200
        assert "gcs_uri" not in data
        assert len(register_called) == 0  # No /register call made
