"""Tests for the workbench SQLite index."""

import os

import pytest

from agentstracer.index import (
    add_policy,
    create_bundle,
    get_bundle,
    get_bundles,
    get_policies,
    get_session_detail,
    get_stats,
    open_index,
    query_sessions,
    remove_policy,
    search_fts,
    update_session,
    upsert_sessions,
    _write_blob,
)


@pytest.fixture
def index_conn(tmp_path, monkeypatch):
    """Open an index DB in a temp directory."""
    monkeypatch.setattr("agentstracer.index.INDEX_DB", tmp_path / "index.db")
    monkeypatch.setattr("agentstracer.index.BLOBS_DIR", tmp_path / "blobs")
    monkeypatch.setattr("agentstracer.index.CONFIG_DIR", tmp_path / "agentstracer_config")
    conn = open_index()
    yield conn
    conn.close()


def _make_session(session_id="sess-1", project="test-project", source="claude",
                  model="claude-sonnet-4", content="Fix the login bug"):
    return {
        "session_id": session_id,
        "project": project,
        "source": source,
        "model": model,
        "start_time": "2025-01-01T00:00:00+00:00",
        "end_time": "2025-01-01T00:10:00+00:00",
        "git_branch": "main",
        "messages": [
            {"role": "user", "content": content, "tool_uses": []},
            {"role": "assistant", "content": "I'll fix it.", "tool_uses": [
                {"tool": "bash", "input": {"command": "pytest"}, "output": "1 passed", "status": "success"},
            ]},
        ],
        "stats": {
            "user_messages": 1,
            "assistant_messages": 1,
            "tool_uses": 1,
            "input_tokens": 500,
            "output_tokens": 100,
        },
    }


class TestUpsertSessions:
    def test_insert_new_session(self, index_conn):
        sessions = [_make_session()]
        new_count = upsert_sessions(index_conn, sessions)
        assert new_count == 1

    def test_insert_multiple_sessions(self, index_conn):
        sessions = [
            _make_session("s1", content="First task"),
            _make_session("s2", content="Second task"),
        ]
        new_count = upsert_sessions(index_conn, sessions)
        assert new_count == 2

    def test_upsert_preserves_review_status(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        update_session(
            index_conn, "sess-1", status="approved", notes="reviewed",
            reason="useful", ai_quality_score=5, ai_score_reason="excellent",
        )

        # Re-index same session
        upsert_sessions(index_conn, [_make_session()])

        row = index_conn.execute(
            "SELECT review_status, reviewer_notes, selection_reason, ai_quality_score, ai_score_reason FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()
        assert row["review_status"] == "approved"
        assert row["reviewer_notes"] == "reviewed"
        assert row["selection_reason"] == "useful"
        assert row["ai_quality_score"] == 5
        assert row["ai_score_reason"] == "excellent"

    def test_changed_content_resets_review_and_unlinks_bundle(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        update_session(index_conn, "sess-1", status="approved", notes="reviewed", ai_quality_score=5)
        bundle_id = create_bundle(index_conn, ["sess-1"])

        upsert_sessions(index_conn, [_make_session(content="new unreviewed content")])

        row = index_conn.execute(
            "SELECT review_status, reviewer_notes, ai_quality_score, bundle_id FROM sessions WHERE session_id='sess-1'"
        ).fetchone()
        assert dict(row) == {
            "review_status": "new", "reviewer_notes": None,
            "ai_quality_score": None, "bundle_id": None,
        }
        assert get_bundle(index_conn, bundle_id)["session_count"] == 0

    def test_blob_filename_cannot_escape_storage(self, index_conn, tmp_path):
        path = _write_blob("../../escaped", _make_session("../../escaped"))
        assert path.resolve().is_relative_to((tmp_path / "blobs").resolve())
        assert not (tmp_path / "escaped.json").exists()

    def test_sensitive_index_files_are_private(self, index_conn, tmp_path):
        upsert_sessions(index_conn, [_make_session()])
        db_mode = os.stat(tmp_path / "index.db").st_mode & 0o777
        blob = next((tmp_path / "blobs").glob("*.json"))
        assert db_mode == 0o600
        assert os.stat(tmp_path / "blobs").st_mode & 0o777 == 0o700
        assert os.stat(blob).st_mode & 0o777 == 0o600

    def test_skips_session_without_id(self, index_conn):
        session = _make_session()
        del session["session_id"]
        assert upsert_sessions(index_conn, [session]) == 0

    def test_empty_list(self, index_conn):
        assert upsert_sessions(index_conn, []) == 0

    def test_badges_computed(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        row = index_conn.execute(
            "SELECT outcome_badge, task_type, display_title FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()
        assert row["display_title"] == "Fix the login bug"
        assert row["outcome_badge"] is not None
        assert row["task_type"] is not None


class TestQuerySessions:
    def test_query_all(self, index_conn):
        upsert_sessions(index_conn, [_make_session("s1"), _make_session("s2")])
        results = query_sessions(index_conn)
        assert len(results) == 2

    def test_filter_by_status(self, index_conn):
        upsert_sessions(index_conn, [_make_session("s1"), _make_session("s2")])
        update_session(index_conn, "s1", status="approved")

        results = query_sessions(index_conn, status="approved")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_filter_by_source(self, index_conn):
        upsert_sessions(index_conn, [
            _make_session("s1", source="claude"),
            _make_session("s2", source="codex"),
        ])
        results = query_sessions(index_conn, source="codex")
        assert len(results) == 1
        assert results[0]["source"] == "codex"

    def test_limit_and_offset(self, index_conn):
        sessions = [_make_session(f"s{i}") for i in range(10)]
        upsert_sessions(index_conn, sessions)

        results = query_sessions(index_conn, limit=3, offset=0)
        assert len(results) == 3

        results2 = query_sessions(index_conn, limit=3, offset=3)
        assert len(results2) == 3
        assert results[0]["session_id"] != results2[0]["session_id"]

    def test_malformed_fts_text_is_treated_as_literal(self, index_conn):
        upsert_sessions(index_conn, [_make_session(content='say "hello"')])
        assert query_sessions(index_conn, search_text='"') == []


class TestGetSessionDetail:
    def test_returns_messages(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        detail = get_session_detail(index_conn, "sess-1")
        assert detail is not None
        assert len(detail["messages"]) == 2
        assert detail["messages"][0]["role"] == "user"

    def test_not_found(self, index_conn):
        assert get_session_detail(index_conn, "nonexistent") is None


class TestSearchFts:
    def test_search_fts_matches_free_text_with_apostrophe(self, index_conn):
        upsert_sessions(index_conn, [_make_session(content="I'd like to examine these options carefully")])

        results = search_fts(index_conn, "I'd like to examine")

        assert len(results) == 1
        assert results[0]["session_id"] == "sess-1"

    def test_search_fts_returns_empty_for_punctuation_only(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])

        results = search_fts(index_conn, "!!! ??? '''")

        assert results == []

    def test_indexes_canonical_tool_output(self, index_conn):
        session = _make_session()
        session["messages"][1]["tool_uses"][0]["output"] = {"nested": ["unique-output-marker"]}
        upsert_sessions(index_conn, [session])
        assert search_fts(index_conn, "unique output marker")[0]["session_id"] == "sess-1"


class TestUpdateSession:
    def test_update_status(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        ok = update_session(index_conn, "sess-1", status="shortlisted")
        assert ok is True

        row = index_conn.execute(
            "SELECT review_status, reviewed_at FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()
        assert row["review_status"] == "shortlisted"
        assert row["reviewed_at"] is not None

    def test_update_notes(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        update_session(index_conn, "sess-1", notes="Good trace", reason="strong debugging")

        row = index_conn.execute(
            "SELECT reviewer_notes, selection_reason FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()
        assert row["reviewer_notes"] == "Good trace"
        assert row["selection_reason"] == "strong debugging"

    def test_not_found(self, index_conn):
        assert update_session(index_conn, "nope", status="blocked") is False


class TestStats:
    def test_stats(self, index_conn):
        upsert_sessions(index_conn, [
            _make_session("s1", source="claude"),
            _make_session("s2", source="codex"),
        ])
        stats = get_stats(index_conn)
        assert stats["total"] == 2
        assert stats["by_source"]["claude"] == 1
        assert stats["by_source"]["codex"] == 1
        assert stats["by_status"]["new"] == 2


class TestBundles:
    def test_create_and_get(self, index_conn):
        upsert_sessions(index_conn, [_make_session("s1"), _make_session("s2")])
        bundle_id = create_bundle(index_conn, ["s1", "s2"], note="Test bundle")

        bundle = get_bundle(index_conn, bundle_id)
        assert bundle is not None
        assert bundle["session_count"] == 2
        assert bundle["submission_note"] == "Test bundle"
        assert len(bundle["sessions"]) == 2

    def test_list_bundles(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        create_bundle(index_conn, ["sess-1"])
        bundles = get_bundles(index_conn)
        assert len(bundles) == 1

    def test_nonexistent_sessions(self, index_conn):
        bundle_id = create_bundle(index_conn, ["nonexistent"])
        bundle = get_bundle(index_conn, bundle_id)
        assert bundle["session_count"] == 0

    def test_rebinding_updates_old_bundle_count(self, index_conn):
        upsert_sessions(index_conn, [_make_session()])
        old_id = create_bundle(index_conn, ["sess-1"])
        new_id = create_bundle(index_conn, ["sess-1"])
        assert get_bundle(index_conn, old_id)["session_count"] == 0
        assert get_bundle(index_conn, new_id)["session_count"] == 1


class TestPolicies:
    def test_add_and_list(self, index_conn):
        pid = add_policy(index_conn, "redact_string", "my-secret", reason="API key")
        policies = get_policies(index_conn)
        assert len(policies) == 1
        assert policies[0]["policy_id"] == pid
        assert policies[0]["value"] == "my-secret"

    def test_remove(self, index_conn):
        pid = add_policy(index_conn, "exclude_project", "private-repo")
        assert remove_policy(index_conn, pid) is True
        assert len(get_policies(index_conn)) == 0

    def test_remove_nonexistent(self, index_conn):
        assert remove_policy(index_conn, "nope") is False
