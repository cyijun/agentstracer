"""Local SQLite + FTS5 index for the scientist workbench."""

import json
import hashlib
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .badges import compute_all_badges
from .config import CONFIG_DIR

INDEX_DB = CONFIG_DIR / "index.db"
BLOBS_DIR = CONFIG_DIR / "blobs"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id         TEXT PRIMARY KEY,
    project            TEXT NOT NULL,
    source             TEXT NOT NULL,
    model              TEXT,
    start_time         TEXT,
    end_time           TEXT,
    duration_seconds   INTEGER,
    git_branch         TEXT,
    user_messages      INTEGER DEFAULT 0,
    assistant_messages INTEGER DEFAULT 0,
    tool_uses          INTEGER DEFAULT 0,
    input_tokens       INTEGER DEFAULT 0,
    output_tokens      INTEGER DEFAULT 0,
    display_title      TEXT,
    outcome_badge      TEXT,
    value_badges       TEXT,
    risk_badges        TEXT,
    sensitivity_score  REAL DEFAULT 0.0,
    task_type          TEXT,
    files_touched      TEXT,
    commands_run       TEXT,
    review_status      TEXT DEFAULT 'new',
    selection_reason   TEXT,
    reviewer_notes     TEXT,
    reviewed_at        TEXT,
    blob_path          TEXT,
    content_hash       TEXT,
    raw_source_path    TEXT,
    indexed_at         TEXT NOT NULL,
    updated_at         TEXT,
    bundle_id          TEXT REFERENCES bundles(bundle_id),
    ai_quality_score   INTEGER,
    ai_score_reason    TEXT,
    ai_display_title   TEXT
);

CREATE TABLE IF NOT EXISTS bundles (
    bundle_id       TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    session_count   INTEGER,
    status          TEXT DEFAULT 'draft',
    attestation     TEXT,
    submission_note TEXT,
    bundle_hash     TEXT,
    manifest        TEXT,
    shared_at       TEXT,
    gcs_uri         TEXT
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id    TEXT PRIMARY KEY,
    policy_type  TEXT NOT NULL,
    value        TEXT NOT NULL,
    reason       TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(review_status);
CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    session_id,
    display_title,
    transcript_text,
    files_touched,
    commands_run
);
"""

# We use a regular FTS5 table (not contentless) so it stores its own content.
# This avoids rowid synchronization issues with INSERT OR REPLACE on the
# sessions table.  We join on session_id instead of rowid.
# The transcript_text column holds flattened message content for search.


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def open_index() -> sqlite3.Connection:
    """Open (and initialize if needed) the index database.

    Creates the database file, tables, indices, and FTS virtual table
    if they do not already exist. Returns a connection with
    row_factory set to sqlite3.Row for dict-like access.
    """
    _ensure_private_dir(CONFIG_DIR)
    _ensure_private_dir(BLOBS_DIR)

    conn = sqlite3.connect(str(INDEX_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(SCHEMA_SQL)

    # FTS5 creation must be separate -- executescript resets transactions
    # and CREATE VIRTUAL TABLE cannot be inside a multi-statement script
    # on some SQLite builds. We handle the case where FTS5 is unavailable.
    try:
        conn.execute(FTS_SCHEMA_SQL.strip())
        conn.commit()
    except sqlite3.OperationalError:
        # FTS5 extension not available -- full-text search will be disabled
        pass

    # Migrations: add columns that may be missing in older databases.
    for col, col_type in [
        ("ai_quality_score", "INTEGER"),
        ("ai_score_reason", "TEXT"),
        ("ai_episode_quality", "REAL"),
        ("ai_quality_tier", "TEXT"),
        ("ai_scoring_detail", "TEXT"),
        ("ai_task_type", "TEXT"),
        ("ai_outcome_badge", "TEXT"),
        ("ai_value_badges", "TEXT"),
        ("ai_risk_badges", "TEXT"),
        ("ai_display_title", "TEXT"),
        ("parent_session_id", "TEXT"),
        ("segment_index", "INTEGER"),
        ("segment_start_message", "INTEGER"),
        ("segment_end_message", "INTEGER"),
        ("segment_reason", "TEXT"),
        ("content_hash", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):
                raise
            # Column already exists — ignore.

    for col, col_type in [
        ("shared_at", "TEXT"),
        ("gcs_uri", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE bundles ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):
                raise

    _ensure_private_file(INDEX_DB)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{INDEX_DB}{suffix}")
        if sidecar.exists():
            _ensure_private_file(sidecar)
    return conn


def _ensure_private_dir(path: Path) -> None:
    """Create a sensitive-data directory and repair permissive modes."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _ensure_private_file(path: Path) -> None:
    """Best-effort repair for files containing raw local conversations."""
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _flatten_transcript(session: dict[str, Any]) -> str:
    """Extract all message content and tool I/O as plain text for FTS indexing."""
    parts: list[str] = []
    def append_value(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                append_value(nested)
        elif isinstance(value, list):
            for nested in value:
                append_value(nested)

    for msg in session.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    # Text blocks
                    text = block.get("text")
                    if text:
                        parts.append(text)
                    # Tool use input
                    tool_input = block.get("input")
                    append_value(tool_input)
                    # Tool result output
                    output = block.get("output")
                    append_value(output)
        append_value(msg.get("tool_uses", []))
        # Handle agentstracer's parsed format: tool uses stored as dicts with "tool" key
        tool = msg.get("tool")
        if tool:
            inp = msg.get("input")
            if isinstance(inp, dict):
                for v in inp.values():
                    if isinstance(v, str):
                        parts.append(v)
            out = msg.get("output")
            if isinstance(out, str):
                parts.append(out)
    return "\n".join(parts)


def _extract_files_touched(session: dict[str, Any]) -> list[str]:
    """Extract file paths from tool use inputs across all messages."""
    files: set[str] = set()
    for msg in session.get("messages", []):
        content = msg.get("content")
        blocks = []
        if isinstance(content, list):
            blocks = content
        # Also handle agentstracer parsed format
        if msg.get("tool"):
            blocks = [msg]

        for block in blocks:
            if not isinstance(block, dict):
                continue
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                continue
            for key in ("file_path", "path", "file", "filename"):
                val = inp.get(key)
                if isinstance(val, str) and val.strip():
                    files.add(val.strip())
    return sorted(files)


def _extract_commands_run(session: dict[str, Any]) -> list[str]:
    """Extract shell commands from bash/shell tool uses."""
    commands: list[str] = []
    for msg in session.get("messages", []):
        content = msg.get("content")
        blocks = []
        if isinstance(content, list):
            blocks = content
        if msg.get("tool"):
            blocks = [msg]

        for block in blocks:
            if not isinstance(block, dict):
                continue
            tool_name = block.get("tool") or block.get("name", "")
            if tool_name not in ("bash", "shell", "terminal", "execute_command"):
                continue
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                continue
            cmd = inp.get("command") or inp.get("cmd", "")
            if isinstance(cmd, str) and cmd.strip():
                commands.append(cmd.strip())
    return commands


def _compute_duration(session: dict[str, Any]) -> int | None:
    """Compute duration in seconds from start_time and end_time."""
    start = session.get("start_time")
    end = session.get("end_time")
    if not start or not end:
        return None
    try:
        start_dt = datetime.fromisoformat(str(start))
        end_dt = datetime.fromisoformat(str(end))
        delta = (end_dt - start_dt).total_seconds()
        if delta < 0:
            return None
        return int(delta)
    except (ValueError, TypeError):
        return None


def _generate_display_title(session: dict[str, Any]) -> str:
    """Generate a display title from the first user message, truncated."""
    # Prefer segment_title for child traces (already stripped of metadata)
    seg_title = session.get("segment_title")
    if seg_title:
        if len(seg_title) > 120:
            return seg_title[:117] + "..."
        return seg_title
    for msg in session.get("messages", []):
        role = msg.get("role", "")
        if role != "user":
            continue
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    text = block
                    break
                if isinstance(block, dict) and block.get("text"):
                    text = block["text"]
                    break
        text = text.strip()
        if text:
            # Truncate to first line, max 120 chars
            first_line = text.split("\n", 1)[0].strip()
            if len(first_line) > 120:
                return first_line[:117] + "..."
            return first_line
    return session.get("session_id", "untitled")


def _session_content_hash(session: dict[str, Any]) -> str:
    payload = json.dumps(
        session, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_blob_path(session_id: str, content_hash: str) -> Path:
    safe_id = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:32]
    return BLOBS_DIR / f"{safe_id}-{content_hash[:16]}.json"


def _write_blob(
    session_id: str,
    session: dict[str, Any],
    content_hash: str | None = None,
) -> Path:
    """Write full session JSON to blob storage. Returns the blob file path."""
    _ensure_private_dir(BLOBS_DIR)
    digest = content_hash or _session_content_hash(session)
    blob_path = _canonical_blob_path(session_id, digest)
    fd, tmp_name = tempfile.mkstemp(prefix=".blob-", suffix=".tmp", dir=BLOBS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, blob_path)
        _ensure_private_file(blob_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return blob_path


def upsert_sessions(conn: sqlite3.Connection, sessions: list[dict[str, Any]]) -> int:
    """Index parsed sessions into the database.

    Takes parsed session dicts (output of parser.parse_project_sessions).
    Stores metadata in sessions table, writes full session JSON to
    BLOBS_DIR/{session_id}.json, and updates FTS index.

    Returns the count of new sessions inserted (sessions that did not
    previously exist in the index).
    """
    if not sessions:
        return 0

    now = _now_iso()
    new_count = 0

    # Check FTS availability
    has_fts = _has_fts(conn)

    affected_bundles: set[str] = set()
    try:
        for session in sessions:
            session_id = session.get("session_id")
            if not session_id:
                continue

            project = session.get("project", "")
            source = session.get("source", "")
            if not project or not source:
                continue

            stats = session.get("stats", {})
            duration = _compute_duration(session)
            badges = compute_all_badges(session)
            display_title = badges["display_title"]
            files = badges["files_touched"]
            commands = badges["commands_run"]
            content_hash = _session_content_hash(session)

            existing = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
            is_new = existing is None
            old_hash = existing["content_hash"] if existing is not None else None
            if existing is not None and not old_hash and existing["blob_path"]:
                try:
                    old_path = Path(existing["blob_path"]).resolve()
                    if old_path.is_relative_to(BLOBS_DIR.resolve()):
                        with old_path.open(encoding="utf-8") as old_file:
                            old_hash = _session_content_hash(json.load(old_file))
                except (OSError, ValueError, json.JSONDecodeError):
                    old_hash = None
            content_changed = existing is not None and old_hash != content_hash
            if content_changed and existing["bundle_id"]:
                affected_bundles.add(existing["bundle_id"])

            blob_path = _write_blob(session_id, session, content_hash)

            if has_fts and not is_new:
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session_id,))

            conn.execute(
                """INSERT INTO sessions (
                session_id, project, source, model,
                start_time, end_time, duration_seconds,
                git_branch,
                user_messages, assistant_messages, tool_uses,
                input_tokens, output_tokens,
                display_title,
                outcome_badge, value_badges, risk_badges,
                sensitivity_score, task_type,
                files_touched, commands_run,
                blob_path, raw_source_path, content_hash,
                indexed_at, updated_at,
                parent_session_id, segment_index,
                segment_start_message, segment_end_message,
                segment_reason
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            ) ON CONFLICT(session_id) DO UPDATE SET
                project=excluded.project, source=excluded.source, model=excluded.model,
                start_time=excluded.start_time, end_time=excluded.end_time,
                duration_seconds=excluded.duration_seconds, git_branch=excluded.git_branch,
                user_messages=excluded.user_messages,
                assistant_messages=excluded.assistant_messages,
                tool_uses=excluded.tool_uses, input_tokens=excluded.input_tokens,
                output_tokens=excluded.output_tokens, display_title=excluded.display_title,
                outcome_badge=excluded.outcome_badge, value_badges=excluded.value_badges,
                risk_badges=excluded.risk_badges,
                sensitivity_score=excluded.sensitivity_score, task_type=excluded.task_type,
                files_touched=excluded.files_touched, commands_run=excluded.commands_run,
                blob_path=excluded.blob_path, raw_source_path=excluded.raw_source_path,
                content_hash=excluded.content_hash, updated_at=excluded.updated_at,
                parent_session_id=excluded.parent_session_id,
                segment_index=excluded.segment_index,
                segment_start_message=excluded.segment_start_message,
                segment_end_message=excluded.segment_end_message,
                segment_reason=excluded.segment_reason""",
                (
                    session_id, project, source, session.get("model"),
                    session.get("start_time"), session.get("end_time"), duration,
                    session.get("git_branch"),
                    stats.get("user_messages", 0), stats.get("assistant_messages", 0),
                    stats.get("tool_uses", 0), stats.get("input_tokens", 0),
                    stats.get("output_tokens", 0), display_title,
                    badges["outcome_badge"], json.dumps(badges["value_badges"]),
                    json.dumps(badges["risk_badges"]), badges["sensitivity_score"],
                    badges["task_type"], json.dumps(files), json.dumps(commands),
                    str(blob_path), session.get("raw_source_path"), content_hash,
                    now, now, session.get("parent_session_id"), session.get("segment_index"),
                    session.get("segment_message_range", [None, None])[0] if session.get("segment_message_range") else None,
                    session.get("segment_message_range", [None, None])[1] if session.get("segment_message_range") else None,
                    session.get("segment_reason"),
                ),
            )

            if content_changed:
                conn.execute(
                    """UPDATE sessions SET review_status='new', selection_reason=NULL,
                    reviewer_notes=NULL, reviewed_at=NULL, bundle_id=NULL,
                    ai_quality_score=NULL, ai_score_reason=NULL,
                    ai_episode_quality=NULL, ai_quality_tier=NULL,
                    ai_scoring_detail=NULL, ai_task_type=NULL,
                    ai_outcome_badge=NULL, ai_value_badges=NULL,
                    ai_risk_badges=NULL, ai_display_title=NULL
                    WHERE session_id=?""",
                    (session_id,),
                )

            if has_fts:
                conn.execute(
                    "INSERT INTO sessions_fts(session_id, display_title, transcript_text, files_touched, commands_run) VALUES(?, ?, ?, ?, ?)",
                    (session_id, display_title, _flatten_transcript(session), " ".join(files), " ".join(commands)),
                )

            if is_new:
                new_count += 1

        for bundle_id in affected_bundles:
            conn.execute(
                "UPDATE bundles SET session_count=(SELECT COUNT(*) FROM sessions WHERE bundle_id=?) WHERE bundle_id=?",
                (bundle_id, bundle_id),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return new_count


def _has_fts(conn: sqlite3.Connection) -> bool:
    """Check if the FTS virtual table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions_fts'"
    ).fetchone()
    return row is not None


def query_sessions(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    source: str | None = None,
    project: str | None = None,
    task_type: str | None = None,
    search_text: str | None = None,
    sort: str = "start_time",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    exclude_segmented_parents: bool = False,
) -> list[dict[str, Any]]:
    """Query sessions with optional filters.

    If search_text is provided and FTS is available, joins with the FTS
    index. Returns a list of dicts containing metadata (no messages).
    """
    # Validate sort column to prevent SQL injection
    allowed_sort_columns = {
        "start_time", "end_time", "indexed_at", "updated_at",
        "project", "source", "model", "review_status", "task_type",
        "user_messages", "assistant_messages", "tool_uses",
        "input_tokens", "output_tokens", "duration_seconds",
        "sensitivity_score", "ai_quality_score",
    }
    if sort not in allowed_sort_columns:
        sort = "start_time"
    if order.lower() not in ("asc", "desc"):
        order = "desc"

    limit = min(max(int(limit), 1), 1000)
    offset = max(int(offset), 0)
    params: list[Any] = []
    where_clauses: list[str] = []

    if search_text and _has_fts(conn):
        # FTS join query
        base = (
            "SELECT s.* FROM sessions s "
            "JOIN sessions_fts f ON s.session_id = f.session_id "
            "WHERE sessions_fts MATCH ?"
        )
        normalized_query = _normalize_fts_query(search_text)
        if not normalized_query:
            return []
        params.append(normalized_query)
    else:
        base = "SELECT * FROM sessions s WHERE 1=1"

    if status is not None:
        where_clauses.append("s.review_status = ?")
        params.append(status)
    if source is not None:
        where_clauses.append("s.source = ?")
        params.append(source)
    if project is not None:
        where_clauses.append("s.project = ?")
        params.append(project)
    if task_type is not None:
        where_clauses.append("COALESCE(s.ai_task_type, s.task_type) = ?")
        params.append(task_type)
    if exclude_segmented_parents:
        where_clauses.append("s.review_status != 'segmented'")

    sql = base
    for clause in where_clauses:
        sql += f" AND {clause}"
    sql += f" ORDER BY s.{sort} {order.upper()} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if search_text and "fts5" in str(exc).lower():
            return []
        raise
    return [dict(row) for row in rows]


def get_session_detail(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """Return full session detail including messages loaded from blob.

    Returns None if the session is not found.
    """
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None

    result = dict(row)

    # Load messages from blob
    blob_path_str = result.get("blob_path")
    blob_path = Path(blob_path_str).resolve() if blob_path_str else None
    blobs_root = BLOBS_DIR.resolve()
    if blob_path and not blob_path.is_relative_to(blobs_root):
        blob_path = None
    # Fallback: if stored path is stale, try the canonical location
    if (blob_path is None or not blob_path.exists()) and result.get("content_hash"):
        fallback = _canonical_blob_path(session_id, result["content_hash"])
        if fallback.exists():
            blob_path = fallback
    if blob_path and blob_path.exists():
        try:
            with open(blob_path) as f:
                blob_data = json.load(f)
            result["messages"] = blob_data.get("messages", [])
        except (json.JSONDecodeError, OSError):
            result["messages"] = []
    else:
        result["messages"] = []

    # Parse JSON fields
    for field in ("value_badges", "risk_badges", "files_touched", "commands_run"):
        val = result.get(field)
        if isinstance(val, str):
            try:
                result[field] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass

    return result


def update_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    status: str | None = None,
    notes: str | None = None,
    reason: str | None = None,
    ai_quality_score: int | None = None,
    ai_score_reason: str | None = None,
    ai_episode_quality: float | None = None,
    ai_quality_tier: str | None = None,
    ai_scoring_detail: str | None = None,
    ai_task_type: str | None = None,
    ai_outcome_badge: str | None = None,
    ai_value_badges: str | None = None,
    ai_risk_badges: str | None = None,
    ai_display_title: str | None = None,
) -> bool:
    """Update review fields on a session.

    Sets reviewed_at when status changes. Returns True if the session was
    found and updated, False otherwise.
    """
    if ai_quality_score is not None:
        ai_quality_score = int(ai_quality_score)
        if not (1 <= ai_quality_score <= 5):
            return False

    row = conn.execute(
        "SELECT session_id, review_status FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return False

    updates: list[str] = []
    params: list[Any] = []
    now = _now_iso()

    if status is not None:
        updates.append("review_status = ?")
        params.append(status)
        if status != row["review_status"]:
            updates.append("reviewed_at = ?")
            params.append(now)

    if notes is not None:
        updates.append("reviewer_notes = ?")
        params.append(notes)

    if reason is not None:
        updates.append("selection_reason = ?")
        params.append(reason)

    if ai_quality_score is not None:
        updates.append("ai_quality_score = ?")
        params.append(ai_quality_score)

    if ai_score_reason is not None:
        updates.append("ai_score_reason = ?")
        params.append(ai_score_reason)

    if ai_episode_quality is not None:
        updates.append("ai_episode_quality = ?")
        params.append(ai_episode_quality)

    if ai_quality_tier is not None:
        updates.append("ai_quality_tier = ?")
        params.append(ai_quality_tier)

    if ai_scoring_detail is not None:
        updates.append("ai_scoring_detail = ?")
        params.append(ai_scoring_detail)

    if ai_task_type is not None:
        updates.append("ai_task_type = ?")
        params.append(ai_task_type)

    if ai_outcome_badge is not None:
        updates.append("ai_outcome_badge = ?")
        params.append(ai_outcome_badge)

    if ai_value_badges is not None:
        updates.append("ai_value_badges = ?")
        params.append(ai_value_badges)

    if ai_risk_badges is not None:
        updates.append("ai_risk_badges = ?")
        params.append(ai_risk_badges)

    if ai_display_title is not None:
        updates.append("ai_display_title = ?")
        params.append(ai_display_title)

    if not updates:
        return True

    updates.append("updated_at = ?")
    params.append(now)
    params.append(session_id)

    conn.execute(
        f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?",
        params,
    )
    conn.commit()
    return True


def query_unscored_sessions(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return sessions where ai_quality_score IS NULL.

    Returns a list of dicts with session_id, display_title, task_type,
    outcome_badge, project, and source.
    """
    params: list[Any] = []
    sql = (
        "SELECT session_id, display_title, task_type, outcome_badge, project, source "
        "FROM sessions WHERE ai_quality_score IS NULL"
    )
    if source is not None:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Full-text search across session transcripts, titles, files, and commands.

    Returns session metadata ranked by FTS5 relevance (bm25).
    Returns an empty list if FTS is not available.
    """
    if not _has_fts(conn):
        return []

    normalized_query = _normalize_fts_query(query)
    if not normalized_query:
        return []

    limit = min(max(int(limit), 1), 1000)
    offset = max(int(offset), 0)

    rows = conn.execute(
        "SELECT s.* FROM sessions s "
        "JOIN sessions_fts f ON s.session_id = f.session_id "
        "WHERE sessions_fts MATCH ? "
        "ORDER BY rank "
        "LIMIT ? OFFSET ?",
        (normalized_query, limit, offset),
    ).fetchall()
    return [dict(row) for row in rows]


def _normalize_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe literal FTS5 AND query."""
    terms = re.findall(r"\w+", str(query), flags=re.UNICODE)
    return " AND ".join(f'"{term}"' for term in terms)


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return aggregate counts grouped by status, source, and project."""
    result: dict[str, Any] = {"total": 0, "by_status": {}, "by_source": {}, "by_project": {}, "by_task_type": {}}

    # Total
    row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
    result["total"] = row["cnt"] if row else 0

    # By status
    for row in conn.execute(
        "SELECT review_status, COUNT(*) AS cnt FROM sessions GROUP BY review_status"
    ).fetchall():
        result["by_status"][row["review_status"]] = row["cnt"]

    # By source
    for row in conn.execute(
        "SELECT source, COUNT(*) AS cnt FROM sessions GROUP BY source"
    ).fetchall():
        result["by_source"][row["source"]] = row["cnt"]

    # By project
    for row in conn.execute(
        "SELECT project, COUNT(*) AS cnt FROM sessions GROUP BY project"
    ).fetchall():
        result["by_project"][row["project"]] = row["cnt"]

    # By task_type (prefer LLM classification when available)
    for row in conn.execute(
        "SELECT COALESCE(ai_task_type, task_type) AS tt, COUNT(*) AS cnt "
        "FROM sessions WHERE COALESCE(ai_task_type, task_type) IS NOT NULL "
        "GROUP BY tt ORDER BY cnt DESC"
    ).fetchall():
        result["by_task_type"][row["tt"]] = row["cnt"]

    return result


def get_dashboard_analytics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return dashboard analytics for the workbench UI."""
    result: dict[str, Any] = {}

    # Summary
    row = conn.execute(
        "SELECT COUNT(*) as total_sessions, "
        "SUM(input_tokens + output_tokens) as total_tokens, "
        "COUNT(DISTINCT project) as unique_projects, "
        "COUNT(DISTINCT source) as unique_sources "
        "FROM sessions"
    ).fetchone()
    result["summary"] = {
        "total_sessions": row["total_sessions"] or 0,
        "total_tokens": row["total_tokens"] or 0,
        "unique_projects": row["unique_projects"] or 0,
        "unique_sources": row["unique_sources"] or 0,
    }

    # Activity per day (last 30 days)
    rows = conn.execute(
        "SELECT DATE(start_time) as day, COUNT(*) as count FROM sessions "
        "WHERE start_time IS NOT NULL GROUP BY DATE(start_time) "
        "ORDER BY day DESC LIMIT 30"
    ).fetchall()
    result["activity"] = [dict(r) for r in rows]

    # Outcome badge distribution (prefer LLM classification)
    rows = conn.execute(
        "SELECT COALESCE(ai_outcome_badge, outcome_badge) as outcome_label, "
        "COUNT(*) as count FROM sessions "
        "WHERE COALESCE(ai_outcome_badge, outcome_badge) IS NOT NULL "
        "GROUP BY outcome_label"
    ).fetchall()
    result["by_outcome_label"] = [dict(r) for r in rows]

    # Value badge distribution (prefer LLM classification)
    rows = conn.execute(
        "SELECT j.value as badge, COUNT(*) as count "
        "FROM sessions, json_each(COALESCE(ai_value_badges, value_badges)) j "
        "GROUP BY j.value"
    ).fetchall()
    result["by_value_label"] = [dict(r) for r in rows]

    # Risk badge distribution (prefer LLM classification)
    rows = conn.execute(
        "SELECT j.value as badge, COUNT(*) as count "
        "FROM sessions, json_each(COALESCE(sessions.ai_risk_badges, sessions.risk_badges)) j "
        "GROUP BY j.value"
    ).fetchall()
    result["by_risk_level"] = [dict(r) for r in rows]

    # Task type (prefer LLM classification)
    rows = conn.execute(
        "SELECT COALESCE(ai_task_type, task_type) as task_type, "
        "COUNT(*) as count FROM sessions "
        "WHERE COALESCE(ai_task_type, task_type) IS NOT NULL "
        "GROUP BY task_type ORDER BY count DESC"
    ).fetchall()
    result["by_task_type"] = [dict(r) for r in rows]

    # Model
    rows = conn.execute(
        "SELECT model, COUNT(*) as count FROM sessions "
        "WHERE model IS NOT NULL GROUP BY model ORDER BY count DESC"
    ).fetchall()
    result["by_model"] = [dict(r) for r in rows]

    # Tokens by source
    rows = conn.execute(
        "SELECT source, SUM(input_tokens) as input_tokens, "
        "SUM(output_tokens) as output_tokens "
        "FROM sessions GROUP BY source"
    ).fetchall()
    result["tokens_by_source"] = [dict(r) for r in rows]

    # Quality score distribution
    rows = conn.execute(
        "SELECT ai_quality_score as score, COUNT(*) as count FROM sessions "
        "WHERE ai_quality_score IS NOT NULL GROUP BY ai_quality_score "
        "ORDER BY ai_quality_score"
    ).fetchall()
    result["by_quality_score"] = [dict(r) for r in rows]
    result["unscored_count"] = conn.execute(
        "SELECT COUNT(*) as cnt FROM sessions WHERE ai_quality_score IS NULL"
    ).fetchone()["cnt"]

    # Weekly activity (more compact than daily)
    rows = conn.execute(
        "SELECT strftime('%Y-W%W', start_time) as week, "
        "MIN(DATE(start_time)) as week_start, "
        "COUNT(*) as count FROM sessions "
        "WHERE start_time IS NOT NULL "
        "GROUP BY week ORDER BY week DESC LIMIT 12"
    ).fetchall()
    result["weekly_activity"] = [dict(r) for r in rows]

    return result


def create_bundle(
    conn: sqlite3.Connection,
    session_ids: list[str],
    attestation: str | None = None,
    note: str | None = None,
) -> str:
    """Create a bundle linking the given sessions.

    Returns the new bundle_id.
    """
    bundle_id = str(uuid.uuid4())
    now = _now_iso()

    # Verify all sessions exist
    found_ids: set[str] = set()
    old_bundle_ids: set[str] = set()
    if session_ids:
        placeholders = ", ".join("?" for _ in session_ids)
        rows = conn.execute(
            f"SELECT session_id, bundle_id FROM sessions WHERE session_id IN ({placeholders})",
            session_ids,
        ).fetchall()
        found_ids = {row["session_id"] for row in rows}
        old_bundle_ids = {row["bundle_id"] for row in rows if row["bundle_id"]}

    conn.execute(
        """INSERT INTO bundles (
            bundle_id, created_at, session_count, status,
            attestation, submission_note
        ) VALUES (?, ?, ?, 'draft', ?, ?)""",
        (bundle_id, now, len(found_ids), attestation, note),
    )

    # Link sessions to the bundle
    for sid in found_ids:
        conn.execute(
            "UPDATE sessions SET bundle_id = ?, updated_at = ? WHERE session_id = ?",
            (bundle_id, now, sid),
        )

    for old_bundle_id in old_bundle_ids:
        conn.execute(
            "UPDATE bundles SET session_count=(SELECT COUNT(*) FROM sessions WHERE bundle_id=?) WHERE bundle_id=?",
            (old_bundle_id, old_bundle_id),
        )

    conn.commit()
    return bundle_id


def get_bundles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """List all bundles ordered by creation time (newest first)."""
    rows = conn.execute(
        "SELECT * FROM bundles ORDER BY created_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_share_ready_stats(conn: sqlite3.Connection, limit: int = 10) -> dict[str, Any]:
    """Return top N approved sessions by quality score, ready to share."""
    rows = conn.execute(
        "SELECT session_id, project, model, source, display_title,"
        " ai_quality_score, user_messages, assistant_messages, tool_uses,"
        " input_tokens, outcome_badge"
        " FROM sessions"
        " WHERE review_status = 'approved'"
        " ORDER BY ai_quality_score DESC NULLS LAST, start_time DESC"
        " LIMIT ?",
        (int(limit),),
    ).fetchall()
    cols = ["session_id", "project", "model", "source", "display_title",
            "ai_quality_score", "user_messages", "assistant_messages",
            "tool_uses", "input_tokens", "outcome_badge"]
    sessions = [dict(zip(cols, r)) for r in rows]
    projects: set[str] = set()
    models: set[str] = set()
    for s in sessions:
        if s.get("project"):
            projects.add(s["project"])
        if s.get("model"):
            models.add(s["model"])
    return {
        "count": len(sessions),
        "total_approved": conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE review_status = 'approved'"
        ).fetchone()[0],
        "projects": sorted(projects),
        "models": sorted(models),
        "sessions": sessions,
    }


def get_bundle(
    conn: sqlite3.Connection,
    bundle_id: str,
) -> dict[str, Any] | None:
    """Get bundle detail with linked session metadata.

    Returns None if the bundle is not found.
    """
    row = conn.execute(
        "SELECT * FROM bundles WHERE bundle_id = ?",
        (bundle_id,),
    ).fetchone()
    if row is None:
        return None

    result = dict(row)

    # Fetch linked sessions
    session_rows = conn.execute(
        "SELECT * FROM sessions WHERE bundle_id = ? ORDER BY start_time ASC",
        (bundle_id,),
    ).fetchall()
    result["sessions"] = [dict(r) for r in session_rows]

    # Parse manifest JSON if present
    if result.get("manifest"):
        try:
            result["manifest"] = json.loads(result["manifest"])
        except (json.JSONDecodeError, ValueError):
            pass

    return result


EXPORT_FIELDS = {
    "session_id", "project", "source", "model",
    "start_time", "end_time", "duration_seconds",
    "git_branch",
    "user_messages", "assistant_messages", "tool_uses",
    "input_tokens", "output_tokens",
    "display_title", "messages",
    "outcome_badge", "value_badges", "risk_badges",
    "ai_quality_score", "task_type",
    "files_touched", "commands_run",
}


def build_bundle_export(
    conn: sqlite3.Connection,
    bundle_id: str,
    bundle: dict[str, Any],
    *,
    custom_strings: list[str] | None = None,
    user_allowlist: list[dict[str, Any]] | None = None,
    extra_usernames: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build the single canonical, redacted representation of a bundle."""
    from .anonymizer import Anonymizer
    from .secrets import redact_session

    anonymizer = Anonymizer(extra_usernames=extra_usernames or [])

    def anonymize(value: Any) -> Any:
        if isinstance(value, str):
            return anonymizer.text(value)
        if isinstance(value, list):
            return [anonymize(item) for item in value]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                safe_key = anonymizer.text(str(key))
                while safe_key in result:
                    safe_key += "_"
                result[safe_key] = anonymize(item)
            return result
        return value

    lines: list[str] = []
    redaction_types: dict[str, int] = {}
    manifest: dict[str, Any] = {
        "bundle_id": bundle_id,
        "session_count": 0,
        "sessions": [],
    }
    manifest_metadata, total_redactions, metadata_log = redact_session(
        {
            "attestation": bundle.get("attestation"),
            "submission_note": bundle.get("submission_note"),
        },
        custom_strings=custom_strings,
        user_allowlist=user_allowlist,
    )
    manifest.update(anonymize(manifest_metadata))
    for entry in metadata_log:
        redaction_type = entry.get("type", "unknown")
        redaction_types[redaction_type] = redaction_types.get(redaction_type, 0) + 1
    metadata_custom_count = total_redactions - len(metadata_log)
    if metadata_custom_count > 0:
        redaction_types["custom"] = metadata_custom_count

    for session_row in bundle.get("sessions", []):
        detail = get_session_detail(conn, session_row["session_id"])
        if not detail:
            continue
        detail, n_redacted, redaction_log = redact_session(
            detail, custom_strings=custom_strings, user_allowlist=user_allowlist,
        )
        detail = anonymize(detail)
        total_redactions += n_redacted
        for entry in redaction_log:
            redaction_type = entry.get("type", "unknown")
            redaction_types[redaction_type] = redaction_types.get(redaction_type, 0) + 1
        custom_count = n_redacted - len(redaction_log)
        if custom_count > 0:
            redaction_types["custom"] = redaction_types.get("custom", 0) + custom_count

        clean = {key: value for key, value in detail.items() if key in EXPORT_FIELDS}
        lines.append(json.dumps(clean, default=str))
        manifest["sessions"].append({
            "session_id": clean.get("session_id"),
            "project": clean.get("project"),
            "source": clean.get("source"),
            "model": clean.get("model"),
        })

    manifest["session_count"] = len(manifest["sessions"])
    manifest["redaction_summary"] = {
        "total_redactions": total_redactions,
        "by_type": redaction_types,
    }
    return lines, manifest


def export_bundle_to_disk(
    conn: sqlite3.Connection,
    bundle_id: str,
    bundle: dict[str, Any],
    *,
    output_path: str | None = None,
    custom_strings: list[str] | None = None,
    user_allowlist: list[dict[str, Any]] | None = None,
    extra_usernames: list[str] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Export a bundle's sessions to disk as JSONL + manifest.

    Returns (export_dir, manifest). Returns (None, {}) if output_path
    validation fails.
    """
    if output_path:
        export_dir = Path(output_path).resolve()
        home = Path.home().resolve()
        if not export_dir.is_relative_to(home) and not export_dir.is_relative_to(Path("/tmp").resolve()):
            return None, {}
    else:
        export_dir = CONFIG_DIR / "bundles" / bundle_id
    _ensure_private_dir(export_dir)

    sessions_file = export_dir / "sessions.jsonl"
    lines, manifest = build_bundle_export(
        conn, bundle_id, bundle,
        custom_strings=custom_strings,
        user_allowlist=user_allowlist,
        extra_usernames=extra_usernames,
    )
    manifest["export_path"] = str(export_dir)
    sessions_fd, tmp_sessions_name = tempfile.mkstemp(
        prefix=".sessions-", suffix=".tmp", dir=export_dir,
    )
    tmp_sessions_file = Path(tmp_sessions_name)

    try:
        with os.fdopen(sessions_fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_sessions_file, sessions_file)
        _ensure_private_file(sessions_file)
    except BaseException:
        tmp_sessions_file.unlink(missing_ok=True)
        raise

    manifest_file = export_dir / "manifest.json"
    fd, manifest_tmp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=export_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(manifest_tmp_name, 0o600)
        os.replace(manifest_tmp_name, manifest_file)
        _ensure_private_file(manifest_file)
    except BaseException:
        try:
            os.unlink(manifest_tmp_name)
        except OSError:
            pass
        raise

    conn.execute(
        "UPDATE bundles SET status = 'exported', manifest = ? WHERE bundle_id = ?",
        (json.dumps(manifest, default=str), bundle_id),
    )
    conn.commit()

    return export_dir, manifest


def get_policies(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all policy rules."""
    rows = conn.execute(
        "SELECT * FROM policies ORDER BY created_at ASC"
    ).fetchall()
    return [dict(row) for row in rows]


def add_policy(
    conn: sqlite3.Connection,
    policy_type: str,
    value: str,
    reason: str | None = None,
) -> str:
    """Add a policy rule. Returns the new policy_id."""
    policy_id = str(uuid.uuid4())
    now = _now_iso()

    conn.execute(
        """INSERT INTO policies (policy_id, policy_type, value, reason, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (policy_id, policy_type, value, reason, now),
    )
    conn.commit()
    return policy_id


def remove_policy(conn: sqlite3.Connection, policy_id: str) -> bool:
    """Remove a policy rule. Returns True if it existed and was removed."""
    cursor = conn.execute(
        "DELETE FROM policies WHERE policy_id = ?",
        (policy_id,),
    )
    conn.commit()
    return cursor.rowcount > 0
