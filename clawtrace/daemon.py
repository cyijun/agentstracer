"""Local daemon for the scientist workbench — scanner + HTTP API."""

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .anonymizer import Anonymizer
from .badges import compute_all_badges
from .config import CONFIG_DIR, load_config, save_config
from .index import (
    EXPORT_FIELDS,
    add_policy,
    create_bundle,
    export_bundle_to_disk,
    get_bundle,
    get_bundles,
    get_dashboard_analytics,
    get_policies,
    get_session_detail,
    get_share_ready_stats,
    get_stats,
    open_index,
    query_sessions,
    remove_policy,
    search_fts,
    update_session,
    upsert_sessions,
)
from .parser import (
    CLAUDE_SOURCE,
    CODEX_SOURCE,
    OPENCLAW_SOURCE,
    discover_projects,
    parse_project_sessions,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8384
SCAN_INTERVAL = 60  # seconds

# NOTE: Network features removed - local-only mode
_share_rate_lock = threading.Lock()

# Sources supported in the workbench (scientist-facing subset)
WORKBENCH_SOURCES = {CLAUDE_SOURCE, CODEX_SOURCE, OPENCLAW_SOURCE}

# Path to the built frontend dist directory
FRONTEND_DIST = Path(__file__).parent / "web" / "frontend" / "dist"


class Scanner:
    """Periodically scans source directories and indexes new sessions."""

    def __init__(self, source_filter: str | None = None):
        self.source_filter = source_filter
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_scan_mtimes: dict[str, float] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def scan_once(self) -> dict[str, int]:
        """Run a single scan pass. Returns {source: new_session_count}."""
        conn = open_index()
        try:
            config = load_config()
            extra_usernames = config.get("redact_usernames", [])
            anonymizer = Anonymizer(extra_usernames=extra_usernames)

            results: dict[str, int] = {}
            projects = discover_projects()

            for project in projects:
                source = project.get("source", "")
                if source not in WORKBENCH_SOURCES:
                    continue
                if self.source_filter and source != self.source_filter:
                    continue

                try:
                    sessions = parse_project_sessions(
                        project["dir_name"],
                        anonymizer=anonymizer,
                        include_thinking=True,
                        source=source,
                    )
                    if sessions:
                        new_count = upsert_sessions(conn, sessions)
                        results[source] = results.get(source, 0) + new_count
                except Exception:
                    logger.exception("Error parsing project %s", project["dir_name"])

            return results
        finally:
            conn.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                results = self.scan_once()
                total_new = sum(results.values())
                if total_new > 0:
                    logger.info("Indexed %d new sessions: %s", total_new, results)
            except Exception:
                logger.exception("Scanner error")
            self._stop_event.wait(SCAN_INTERVAL)


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    """Send a JSON response."""
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    """Read and parse JSON body from request."""
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw)


def _parse_json_fields(rows: list[dict]) -> None:
    """Parse JSON string fields in session rows into Python objects.

    Also resolves LLM-classified badges: prefers ai_* values when present,
    falls back to heuristic values, then removes the ai_* keys from the dict.
    """
    for row in rows:
        for field in ("value_badges", "risk_badges", "files_touched", "commands_run",
                       "ai_value_badges", "ai_risk_badges"):
            if isinstance(row.get(field), str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, ValueError):
                    pass

        # Resolve: prefer LLM classification over heuristic
        if row.get("ai_task_type"):
            row["task_type"] = row["ai_task_type"]
        if row.get("ai_outcome_badge"):
            row["outcome_badge"] = row["ai_outcome_badge"]
        if row.get("ai_value_badges"):
            row["value_badges"] = row["ai_value_badges"]
        if row.get("ai_risk_badges"):
            row["risk_badges"] = row["ai_risk_badges"]

        # Remove ai_* fields from API response (frontend doesn't need them)
        for k in ("ai_task_type", "ai_outcome_badge", "ai_value_badges", "ai_risk_badges"):
            row.pop(k, None)

        # Rename DB column names → user-facing API names
        if "outcome_badge" in row:
            row["outcome_label"] = row.pop("outcome_badge")
        if "value_badges" in row:
            row["value_labels"] = row.pop("value_badges")
        if "risk_badges" in row:
            row["risk_level"] = row.pop("risk_badges")



def share_bundle(
    conn: sqlite3.Connection,
    bundle_id: str,
    *,
    force: bool = False,
    custom_strings: list[str] | None = None,
) -> dict[str, Any]:
    """Upload a bundle to the GCS ingest service.
    
    NOTE: Network features disabled - local-only mode.
    """
    return {
        "error": "Network features are disabled. This is a local-only build of clawtrace.",
        "status": 503,
    }


class WorkbenchHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the workbench API + static files."""

    _last_share_time: float = 0.0

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        # API routes
        if path == "/api/sessions":
            self._handle_list_sessions(params)
        elif path.startswith("/api/sessions/") and path.endswith("/redaction-report"):
            session_id = path[len("/api/sessions/"):-len("/redaction-report")]
            self._handle_redaction_report(session_id)
        elif path.startswith("/api/sessions/") and path.endswith("/redacted"):
            session_id = path[len("/api/sessions/"):-len("/redacted")]
            self._handle_session_redacted(session_id)
        elif path.startswith("/api/sessions/"):
            session_id = path[len("/api/sessions/"):]
            self._handle_get_session(session_id)
        elif path == "/api/search":
            self._handle_search(params)
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/dashboard":
            self._handle_dashboard()
        elif path == "/api/projects":
            self._handle_projects()
        elif path == "/api/share-ready":
            self._handle_share_ready()
        elif path == "/api/bundles":
            self._handle_list_bundles()
        elif path.startswith("/api/bundles/") and path.endswith("/preview"):
            bundle_id = path[len("/api/bundles/"):-len("/preview")]
            self._handle_preview_bundle(bundle_id)
        elif path.startswith("/api/bundles/") and path.endswith("/download"):
            bundle_id = path[len("/api/bundles/"):-len("/download")]
            self._handle_download_bundle(bundle_id)
        elif path.startswith("/api/bundles/"):
            bundle_id = path[len("/api/bundles/"):]
            self._handle_get_bundle(bundle_id)
        elif path == "/api/policies":
            self._handle_list_policies()
        elif path == "/api/allowlist":
            self._handle_list_allowlist()
        else:
            self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/sessions/"):
            session_id = path[len("/api/sessions/"):]
            self._handle_update_session(session_id)
        elif path == "/api/quick-share":
            self._handle_quick_share()
        elif path == "/api/bundles":
            self._handle_create_bundle()
        elif path.startswith("/api/bundles/") and path.endswith("/export"):
            bundle_id = path[len("/api/bundles/"):-len("/export")]
            self._handle_export_bundle(bundle_id)
        elif path.startswith("/api/bundles/") and path.endswith("/share"):
            bundle_id = path[len("/api/bundles/"):-len("/share")]
            self._handle_share(bundle_id)
        elif path == "/api/policies":
            self._handle_add_policy()
        elif path == "/api/allowlist":
            self._handle_add_allowlist()
        elif path == "/api/scan":
            self._handle_trigger_scan()
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/policies/"):
            policy_id = path[len("/api/policies/"):]
            self._handle_remove_policy(policy_id)
        elif path.startswith("/api/allowlist/"):
            entry_id = path[len("/api/allowlist/"):]
            self._handle_remove_allowlist(entry_id)
        else:
            _json_response(self, {"error": "Not found"}, 404)

    # --- API handlers ---

    def _handle_list_sessions(self, params: dict[str, list[str]]) -> None:
        conn = open_index()
        try:
            result = query_sessions(
                conn,
                status=params.get("status", [None])[0],
                source=params.get("source", [None])[0],
                project=params.get("project", [None])[0],
                task_type=params.get("task_type", [None])[0],
                search_text=params.get("q", [None])[0],
                sort=params.get("sort", ["start_time"])[0],
                order=params.get("order", ["desc"])[0],
                limit=int(params.get("limit", ["50"])[0]),
                offset=int(params.get("offset", ["0"])[0]),
            )
            _parse_json_fields(result)
            _json_response(self, result)
        finally:
            conn.close()

    def _handle_get_session(self, session_id: str) -> None:
        conn = open_index()
        try:
            detail = get_session_detail(conn, session_id)
            if detail is None:
                _json_response(self, {"error": "Session not found"}, 404)
                return
            _parse_json_fields([detail])
            _json_response(self, detail)
        finally:
            conn.close()

    def _handle_update_session(self, session_id: str) -> None:
        body = _read_body(self)
        conn = open_index()
        try:
            ok = update_session(
                conn, session_id,
                status=body.get("status"),
                notes=body.get("notes"),
                reason=body.get("reason"),
                ai_quality_score=body.get("ai_quality_score"),
                ai_score_reason=body.get("ai_score_reason"),
                ai_episode_quality=body.get("ai_episode_quality"),
                ai_quality_tier=body.get("ai_quality_tier"),
                ai_scoring_detail=body.get("ai_scoring_detail"),
                ai_task_type=body.get("ai_task_type"),
                ai_outcome_badge=body.get("ai_outcome_badge"),
                ai_value_badges=json.dumps(body["ai_value_badges"]) if isinstance(body.get("ai_value_badges"), list) else body.get("ai_value_badges"),
                ai_risk_badges=json.dumps(body["ai_risk_badges"]) if isinstance(body.get("ai_risk_badges"), list) else body.get("ai_risk_badges"),
            )
            if ok:
                _json_response(self, {"ok": True})
            else:
                _json_response(self, {"error": "Session not found"}, 404)
        finally:
            conn.close()

    def _handle_search(self, params: dict[str, list[str]]) -> None:
        q = params.get("q", [""])[0]
        if not q:
            _json_response(self, [])
            return
        conn = open_index()
        try:
            results = search_fts(
                conn, q,
                limit=int(params.get("limit", ["50"])[0]),
                offset=int(params.get("offset", ["0"])[0]),
            )
            _parse_json_fields(results)
            _json_response(self, results)
        finally:
            conn.close()

    def _handle_stats(self) -> None:
        conn = open_index()
        try:
            stats = get_stats(conn)
            _json_response(self, stats)
        finally:
            conn.close()

    def _handle_dashboard(self) -> None:
        conn = open_index()
        try:
            data = get_dashboard_analytics(conn)
            _json_response(self, data)
        finally:
            conn.close()

    def _handle_projects(self) -> None:
        conn = open_index()
        try:
            rows = conn.execute(
                "SELECT project, source, COUNT(*) as session_count, "
                "SUM(input_tokens + output_tokens) as total_tokens "
                "FROM sessions GROUP BY project, source ORDER BY project"
            ).fetchall()
            _json_response(self, [dict(r) for r in rows])
        finally:
            conn.close()

    def _handle_session_redacted(self, session_id: str) -> None:
        """Return session with secrets redacted — for pre-share review."""
        from .secrets import redact_session
        from .anonymizer import Anonymizer
        from .config import load_config
        conn = open_index()
        try:
            detail = get_session_detail(conn, session_id)
            if detail is None:
                _json_response(self, {"error": "Session not found"}, 404)
                return
            config = load_config()
            custom_strings = config.get("redact_strings", [])
            allowlist = config.get("allowlist_entries", [])
            detail, _, _ = redact_session(detail, custom_strings=custom_strings, user_allowlist=allowlist)
            # Anonymize paths and usernames
            extra = config.get("redact_usernames", [])
            anon = Anonymizer(extra_usernames=extra)
            for field in ("display_title", "project", "git_branch"):
                if detail.get(field) and isinstance(detail[field], str):
                    detail[field] = anon.text(detail[field])
            for msg in detail.get("messages", []):
                for field in ("content", "thinking"):
                    if msg.get(field) and isinstance(msg[field], str):
                        msg[field] = anon.text(msg[field])
                for tool_use in msg.get("tool_uses", []):
                    for field in ("input", "output"):
                        val = tool_use.get(field)
                        if val and isinstance(val, str):
                            tool_use[field] = anon.text(val)
            _json_response(self, detail)
        finally:
            conn.close()

    def _handle_redaction_report(self, session_id: str) -> None:
        """Return redacted session WITH the full redaction log for review."""
        from .secrets import redact_session
        from .anonymizer import Anonymizer
        from .config import load_config
        conn = open_index()
        try:
            detail = get_session_detail(conn, session_id)
            if detail is None:
                _json_response(self, {"error": "Session not found"}, 404)
                return
            config = load_config()
            custom_strings = config.get("redact_strings", [])
            allowlist = config.get("allowlist_entries", [])
            detail, redaction_count, redaction_log = redact_session(
                detail, custom_strings=custom_strings, user_allowlist=allowlist,
            )
            # Anonymize paths and usernames
            extra = config.get("redact_usernames", [])
            anon = Anonymizer(extra_usernames=extra)
            for field in ("display_title", "project", "git_branch"):
                if detail.get(field) and isinstance(detail[field], str):
                    detail[field] = anon.text(detail[field])
            for msg in detail.get("messages", []):
                for field in ("content", "thinking"):
                    if msg.get(field) and isinstance(msg[field], str):
                        msg[field] = anon.text(msg[field])
                for tool_use in msg.get("tool_uses", []):
                    for field in ("input", "output"):
                        val = tool_use.get(field)
                        if val and isinstance(val, str):
                            tool_use[field] = anon.text(val)
            _json_response(self, {
                "session_id": session_id,
                "redaction_count": redaction_count,
                "redaction_log": redaction_log,
                "redacted_session": detail,
            })
        finally:
            conn.close()

    def _handle_list_allowlist(self) -> None:
        """Return current allowlist entries from config."""
        from .config import load_config
        config = load_config()
        entries = config.get("allowlist_entries", [])
        _json_response(self, entries)

    def _handle_add_allowlist(self) -> None:
        """Add a new allowlist entry to config."""
        import uuid
        from .config import load_config, save_config
        body = _read_body(self)

        entry_type = body.get("type")
        if entry_type not in ("exact", "pattern", "category"):
            _json_response(self, {"error": "type must be exact, pattern, or category"}, 400)
            return

        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "type": entry_type,
            "added": datetime.now(timezone.utc).isoformat(),
        }
        if entry_type == "exact":
            if not body.get("text"):
                _json_response(self, {"error": "text required for exact type"}, 400)
                return
            entry["text"] = body["text"]
        elif entry_type == "pattern":
            if not body.get("regex"):
                _json_response(self, {"error": "regex required for pattern type"}, 400)
                return
            entry["regex"] = body["regex"]
        elif entry_type == "category":
            if not body.get("match_type"):
                _json_response(self, {"error": "match_type required for category type"}, 400)
                return
            entry["match_type"] = body["match_type"]

        if body.get("reason"):
            entry["reason"] = body["reason"]

        config = load_config()
        entries = config.get("allowlist_entries", [])
        entries.append(entry)
        config["allowlist_entries"] = entries
        save_config(config)
        _json_response(self, {"ok": True, "entry": entry})

    def _handle_remove_allowlist(self, entry_id: str) -> None:
        """Remove an allowlist entry by ID."""
        from .config import load_config, save_config
        config = load_config()
        entries = config.get("allowlist_entries", [])
        new_entries = [e for e in entries if e.get("id") != entry_id]
        if len(new_entries) == len(entries):
            _json_response(self, {"error": "Entry not found"}, 404)
            return
        config["allowlist_entries"] = new_entries
        save_config(config)
        _json_response(self, {"ok": True})

    def _handle_share_ready(self) -> None:
        """Return stats for approved sessions ready to share."""
        conn = open_index()
        try:
            stats = get_share_ready_stats(conn)
            _json_response(self, stats)
        finally:
            conn.close()

    def _handle_quick_share(self) -> None:
        """Combined create + share in one call."""
        _json_response(self, {
            "error": "Network features are disabled. This is a local-only build of clawtrace.",
        }, 503)

    def _handle_list_bundles(self) -> None:
        conn = open_index()
        try:
            bundles = get_bundles(conn)
            for b in bundles:
                b.pop("gcs_uri", None)
            _json_response(self, bundles)
        finally:
            conn.close()

    def _handle_get_bundle(self, bundle_id: str) -> None:
        conn = open_index()
        try:
            bundle = get_bundle(conn, bundle_id)
            if bundle is None:
                _json_response(self, {"error": "Bundle not found"}, 404)
                return
            bundle.pop("gcs_uri", None)
            _json_response(self, bundle)
        finally:
            conn.close()

    def _handle_create_bundle(self) -> None:
        body = _read_body(self)
        session_ids = body.get("session_ids", [])
        if not session_ids:
            _json_response(self, {"error": "session_ids required"}, 400)
            return
        conn = open_index()
        try:
            bundle_id = create_bundle(
                conn, session_ids,
                attestation=body.get("attestation"),
                note=body.get("note"),
            )
            _json_response(self, {"bundle_id": bundle_id}, 201)
        finally:
            conn.close()

    def _handle_preview_bundle(self, bundle_id: str) -> None:
        """Return a readable summary of an exported bundle."""
        conn = open_index()
        try:
            bundle = get_bundle(conn, bundle_id)
            if bundle is None:
                _json_response(self, {"error": "Bundle not found"}, 404)
                return

            # Check both default and custom export paths
            export_dir = CONFIG_DIR / "bundles" / bundle_id

            # If manifest stored an export_path, try that first
            manifest_data = bundle.get("manifest")
            if isinstance(manifest_data, dict) and manifest_data.get("export_path"):
                custom_dir = Path(manifest_data["export_path"])
                if (custom_dir / "sessions.jsonl").exists():
                    export_dir = custom_dir

            sessions_file = export_dir / "sessions.jsonl"
            manifest_file = export_dir / "manifest.json"

            # Check if exported
            if not sessions_file.exists():
                _json_response(self, {"error": "Bundle not exported yet. Export first."}, 400)
                return

            # Read manifest
            manifest = {}
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)

            # Build session previews from the JSONL
            previews = []
            total_tokens = 0
            total_messages = 0
            with open(sessions_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    session = json.loads(line)
                    msgs = session.get("messages", [])
                    input_tok = session.get("input_tokens", 0) or 0
                    output_tok = session.get("output_tokens", 0) or 0
                    total_tokens += input_tok + output_tok
                    total_messages += len(msgs)

                    # First user message as preview
                    first_user_msg = ""
                    for m in msgs:
                        if m.get("role") == "user":
                            content = m.get("content", "")
                            if isinstance(content, str):
                                first_user_msg = content[:200]
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, str):
                                        first_user_msg = block[:200]
                                        break
                                    if isinstance(block, dict) and block.get("text"):
                                        first_user_msg = block["text"][:200]
                                        break
                            break

                    previews.append({
                        "session_id": session.get("session_id"),
                        "project": session.get("project"),
                        "source": session.get("source"),
                        "model": session.get("model"),
                        "display_title": session.get("display_title", ""),
                        "message_count": len(msgs),
                        "input_tokens": input_tok,
                        "output_tokens": output_tok,
                        "first_user_message": first_user_msg,
                        "ai_quality_score": session.get("ai_quality_score"),
                    })

            file_size = sessions_file.stat().st_size

            _json_response(self, {
                "bundle_id": bundle_id,
                "status": bundle.get("status"),
                "session_count": len(previews),
                "total_tokens": total_tokens,
                "total_messages": total_messages,
                "file_size_bytes": file_size,
                "export_path": str(export_dir),
                "manifest": manifest,
                "sessions": previews,
            })
        finally:
            conn.close()

    def _handle_export_bundle(self, bundle_id: str) -> None:
        body = _read_body(self)
        output_path = body.get("output_path")

        conn = open_index()
        try:
            bundle = get_bundle(conn, bundle_id)
            if bundle is None:
                _json_response(self, {"error": "Bundle not found"}, 404)
                return

            export_dir, manifest = export_bundle_to_disk(conn, bundle_id, bundle, output_path=output_path)
            if export_dir is None:
                _json_response(self, {"error": "output_path must be under home directory or /tmp"}, 400)
                return

            _json_response(self, {
                "ok": True,
                "export_path": str(export_dir),
                "session_count": len(manifest["sessions"]),
            })
        finally:
            conn.close()

    def _handle_download_bundle(self, bundle_id: str) -> None:
        """Generate a zip of the bundle and serve it as a browser download."""
        conn = open_index()
        try:
            bundle = get_bundle(conn, bundle_id)
            if bundle is None:
                _json_response(self, {"error": "Bundle not found"}, 404)
                return

            # Build sessions JSONL content
            lines = []
            manifest_sessions = []
            for s in bundle.get("sessions", []):
                detail = get_session_detail(conn, s["session_id"])
                if detail:
                    clean = {k: v for k, v in detail.items() if k in EXPORT_FIELDS}
                    lines.append(json.dumps(clean, default=str))
                    manifest_sessions.append({
                        "session_id": s["session_id"],
                        "project": s.get("project"),
                        "source": s.get("source"),
                        "model": s.get("model"),
                    })

            sessions_content = "\n".join(lines) + ("\n" if lines else "")

            manifest = {
                "bundle_id": bundle_id,
                "session_count": len(manifest_sessions),
                "attestation": bundle.get("attestation"),
                "submission_note": bundle.get("submission_note"),
                "sessions": manifest_sessions,
            }
            manifest_content = json.dumps(manifest, indent=2, default=str)

            # Create in-memory zip
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("sessions.jsonl", sessions_content)
                zf.writestr("manifest.json", manifest_content)
            zip_bytes = buf.getvalue()

            # Mark as exported
            conn.execute(
                "UPDATE bundles SET status = 'exported', manifest = ? WHERE bundle_id = ?",
                (json.dumps(manifest, default=str), bundle_id),
            )
            conn.commit()

            # Serve the zip
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            filename = f"clawtrace-bundle-{bundle_id[:8]}-{date_str}.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(zip_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(zip_bytes)
        finally:
            conn.close()

    def _handle_share(self, bundle_id: str) -> None:
        """Share a bundle via the ingest service."""
        _json_response(self, {
            "error": "Network features are disabled. This is a local-only build of clawtrace.",
        }, 503)

    def _handle_list_policies(self) -> None:
        conn = open_index()
        try:
            policies = get_policies(conn)
            _json_response(self, policies)
        finally:
            conn.close()

    def _handle_add_policy(self) -> None:
        body = _read_body(self)
        policy_type = body.get("policy_type")
        value = body.get("value")
        if not policy_type or not value:
            _json_response(self, {"error": "policy_type and value required"}, 400)
            return
        conn = open_index()
        try:
            policy_id = add_policy(conn, policy_type, value, reason=body.get("reason"))
            _json_response(self, {"policy_id": policy_id}, 201)
        finally:
            conn.close()

    def _handle_remove_policy(self, policy_id: str) -> None:
        conn = open_index()
        try:
            ok = remove_policy(conn, policy_id)
            if ok:
                _json_response(self, {"ok": True})
            else:
                _json_response(self, {"error": "Policy not found"}, 404)
        finally:
            conn.close()

    def _handle_trigger_scan(self) -> None:
        """Trigger an immediate scan (used by the UI refresh button)."""
        scanner = getattr(self.server, "_scanner", None)
        if scanner:
            results = scanner.scan_once()
            _json_response(self, {"ok": True, "new_sessions": results})
        else:
            _json_response(self, {"error": "Scanner not available"}, 503)

    # --- Static file serving ---

    def _serve_static(self, path: str) -> None:
        """Serve frontend static files, falling back to index.html for SPA routing."""
        if path == "/" or path == "":
            path = "/index.html"

        file_path = (FRONTEND_DIST / path.lstrip("/")).resolve()
        if not file_path.is_relative_to(FRONTEND_DIST.resolve()):
            self.send_error(403)
            return

        # SPA fallback: if file doesn't exist, serve index.html
        if not file_path.exists() or not file_path.is_file():
            file_path = FRONTEND_DIST / "index.html"

        if not file_path.exists():
            # No frontend built yet — serve a placeholder
            self._serve_placeholder()
            return

        content_types = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".map": "application/json",
        }
        ext = file_path.suffix.lower()
        content_type = content_types.get(ext, "application/octet-stream")

        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self.send_error(404)

    def _serve_placeholder(self) -> None:
        """Serve a minimal HTML page when the frontend isn't built yet."""
        html = """<!DOCTYPE html>
<html>
<head><title>ClawTrace Workbench</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px; color: #333; }
h1 { font-size: 1.4em; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
pre { background: #f0f0f0; padding: 12px; border-radius: 6px; overflow-x: auto; }
.api-link { color: #0066cc; }
</style>
</head>
<body>
<h1>ClawTrace Workbench</h1>
<p>The API is running. The frontend hasn't been built yet.</p>
<p>To build the frontend:</p>
<pre>cd clawtrace/web/frontend
npm install
npm run build</pre>
<p>API endpoints available:</p>
<ul>
<li><a class="api-link" href="/api/stats">/api/stats</a> — Index statistics</li>
<li><a class="api-link" href="/api/sessions">/api/sessions</a> — Session list</li>
<li><a class="api-link" href="/api/projects">/api/projects</a> — Projects</li>
<li><a class="api-link" href="/api/bundles">/api/bundles</a> — Bundles</li>
<li><a class="api-link" href="/api/policies">/api/policies</a> — Policies</li>
</ul>
</body>
</html>"""
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    source_filter: str | None = None,
    remote: bool = False,
) -> None:
    """Start the workbench daemon — scanner + HTTP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    scanner = Scanner(source_filter=source_filter)

    # Start HTTP server first so it's responsive immediately
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), WorkbenchHandler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
        port = server.server_address[1]
    server._scanner = scanner  # type: ignore[attr-defined]

    url = f"http://localhost:{port}/traces"
    logger.info("Workbench running at %s", url)

    if remote:
        import socket
        hostname = socket.gethostname()
        print(f"\nRemote access — run this on your local machine:")
        print(f"  ssh -L {port}:localhost:{port} <user>@{hostname}")
        print(f"Then open {url}\n")

    # NOTE: webbrowser opening disabled - local-only mode
    # if open_browser and not remote:
    #     webbrowser.open(url)

    # Run initial scan in background, then start periodic scanner
    def _initial_scan() -> None:
        logger.info("Running initial scan...")
        results = scanner.scan_once()
        total = sum(results.values())
        logger.info("Initial scan complete: %d sessions indexed", total)
        scanner.start()
        logger.info("Background scanner started (interval: %ds)", SCAN_INTERVAL)

    threading.Thread(target=_initial_scan, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        scanner.stop()
        server.shutdown()
