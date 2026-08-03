"""Tests for agentstracer.cli — CLI commands and helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentstracer.cli import (
    _build_status_next_steps,
    _collect_review_attestations,
    _format_size,
    _format_token_count,
    _merge_config_list,
    _parse_csv_arg,
    _scan_for_text_occurrences,
    _scan_high_entropy_strings,
    _scan_pii,
    _share_pii_status,
    _share_preview,
    configure,
    export_to_jsonl,
    list_projects,
    main,
)


# --- _format_size ---


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        result = _format_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        result = _format_size(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero(self):
        assert _format_size(0) == "0 B"

    def test_exactly_1024(self):
        result = _format_size(1024)
        assert "KB" in result


# --- _format_token_count ---


class TestFormatTokenCount:
    def test_plain(self):
        assert _format_token_count(500) == "500"

    def test_thousands(self):
        result = _format_token_count(5000)
        assert result == "5K"

    def test_millions(self):
        result = _format_token_count(2_500_000)
        assert "M" in result

    def test_billions(self):
        result = _format_token_count(1_500_000_000)
        assert "B" in result

    def test_zero(self):
        assert _format_token_count(0) == "0"


# --- attestation helpers ---


class TestAttestationHelpers:
    def test_collect_review_attestations_valid(self):
        attestations, errors, manual_count = _collect_review_attestations(
            attest_asked_full_name=(
                "I asked Jane Doe for their full name and scanned the export for Jane Doe."
            ),
            attest_asked_sensitive=(
                "I asked about company, client, and internal names plus URLs; "
                "none were sensitive and no extra redactions were needed."
            ),
            attest_manual_scan=(
                "I performed a manual scan and reviewed 20 sessions across beginning, middle, and end."
            ),
            full_name="Jane Doe",
        )
        assert not errors
        assert manual_count == 20
        assert "Jane Doe" in attestations["asked_full_name"]

    def test_collect_review_attestations_invalid(self):
        _attestations, errors, manual_count = _collect_review_attestations(
            attest_asked_full_name="scanned quickly",
            attest_asked_sensitive="checked stuff",
            attest_manual_scan="manual scan of 5 sessions",
            full_name="Jane Doe",
        )
        assert errors
        assert "asked_full_name" in errors
        assert "asked_sensitive_entities" in errors
        assert "manual_scan_done" in errors
        assert manual_count == 5

    def test_collect_review_attestations_skip_full_name_valid(self):
        _attestations, errors, manual_count = _collect_review_attestations(
            attest_asked_full_name=(
                "User declined to share full name; skipped exact-name scan."
            ),
            attest_asked_sensitive=(
                "I asked about company/client/internal names and private URLs; "
                "none were sensitive and no extra redactions were needed."
            ),
            attest_manual_scan=(
                "I performed a manual scan and reviewed 20 sessions across beginning, middle, and end."
            ),
            full_name=None,
            skip_full_name_scan=True,
        )
        assert not errors
        assert manual_count == 20

    def test_collect_review_attestations_skip_full_name_invalid(self):
        _attestations, errors, _manual_count = _collect_review_attestations(
            attest_asked_full_name="Asked user and scanned it.",
            attest_asked_sensitive=(
                "I asked about company/client/internal names and private URLs; none found."
            ),
            attest_manual_scan=(
                "I performed a manual scan and reviewed 20 sessions across beginning, middle, and end."
            ),
            full_name=None,
            skip_full_name_scan=True,
        )
        assert "asked_full_name" in errors

    def test_scan_for_text_occurrences(self, tmp_path):
        f = tmp_path / "sample.jsonl"
        f.write_text('{"message":"Jane Doe says hi"}\n{"message":"nothing here"}\n')
        result = _scan_for_text_occurrences(f, "Jane Doe")
        assert result["match_count"] == 1


# --- _parse_csv_arg ---


class TestParseCsvArg:
    def test_none(self):
        assert _parse_csv_arg(None) is None

    def test_empty(self):
        assert _parse_csv_arg("") is None

    def test_single(self):
        assert _parse_csv_arg("foo") == ["foo"]

    def test_comma_separated(self):
        assert _parse_csv_arg("foo, bar, baz") == ["foo", "bar", "baz"]

    def test_strips_whitespace(self):
        assert _parse_csv_arg("  a ,  b  ") == ["a", "b"]

    def test_empty_items_filtered(self):
        assert _parse_csv_arg("a,,b,") == ["a", "b"]


# --- _merge_config_list ---


class TestMergeConfigList:
    def test_merge_new_values(self):
        config = {"items": ["a", "b"]}
        _merge_config_list(config, "items", ["c", "d"])
        assert sorted(config["items"]) == ["a", "b", "c", "d"]

    def test_deduplicate(self):
        config = {"items": ["a", "b"]}
        _merge_config_list(config, "items", ["b", "c"])
        assert sorted(config["items"]) == ["a", "b", "c"]

    def test_sorted(self):
        config = {"items": ["z"]}
        _merge_config_list(config, "items", ["a", "m"])
        assert config["items"] == ["a", "m", "z"]

    def test_missing_key(self):
        config = {}
        _merge_config_list(config, "items", ["a"])
        assert config["items"] == ["a"]


# --- export_to_jsonl ---


class TestExportToJsonl:
    def test_writes_jsonl(self, tmp_path, mock_anonymizer, monkeypatch):
        output = tmp_path / "out.jsonl"
        session_data = [{
            "session_id": "s1",
            "model": "claude-sonnet-4-20250514",
            "git_branch": "main",
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T01:00:00",
            "messages": [{"role": "user", "content": "hi"}],
            "stats": {"input_tokens": 100, "output_tokens": 50},
            "project": "test",
        }]
        monkeypatch.setattr(
            "agentstracer.cli.parse_project_sessions",
            lambda *a, **kw: session_data,
        )

        projects = [{"dir_name": "test", "display_name": "test"}]
        meta = export_to_jsonl(projects, output, mock_anonymizer)

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        assert meta["sessions"] == 1

    def test_skips_synthetic_model(self, tmp_path, mock_anonymizer, monkeypatch):
        output = tmp_path / "out.jsonl"
        session_data = [{
            "session_id": "s1",
            "model": "<synthetic>",
            "messages": [{"role": "user", "content": "hi"}],
            "stats": {},
        }]
        monkeypatch.setattr(
            "agentstracer.cli.parse_project_sessions",
            lambda *a, **kw: session_data,
        )
        projects = [{"dir_name": "test", "display_name": "test"}]
        meta = export_to_jsonl(projects, output, mock_anonymizer)
        assert meta["sessions"] == 0
        assert meta["skipped"] == 1

    def test_counts_redactions(self, tmp_path, mock_anonymizer, monkeypatch):
        output = tmp_path / "out.jsonl"
        session_data = [{
            "session_id": "s1",
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz"}],
            "stats": {"input_tokens": 10, "output_tokens": 5},
        }]
        monkeypatch.setattr(
            "agentstracer.cli.parse_project_sessions",
            lambda *a, **kw: session_data,
        )
        projects = [{"dir_name": "test", "display_name": "test"}]
        meta = export_to_jsonl(projects, output, mock_anonymizer)
        assert meta["redactions"] >= 1

    def test_skips_none_model(self, tmp_path, mock_anonymizer, monkeypatch):
        output = tmp_path / "out.jsonl"
        session_data = [{
            "session_id": "s1",
            "model": None,
            "messages": [{"role": "user", "content": "hi"}],
            "stats": {},
        }]
        monkeypatch.setattr(
            "agentstracer.cli.parse_project_sessions",
            lambda *a, **kw: session_data,
        )
        projects = [{"dir_name": "t", "display_name": "t"}]
        meta = export_to_jsonl(projects, output, mock_anonymizer)
        assert meta["sessions"] == 0
        assert meta["skipped"] == 1


# --- configure ---


class TestConfigure:
    def test_merges_exclude(self, tmp_config, monkeypatch, capsys):
        monkeypatch.setattr("agentstracer.cli.CONFIG_FILE", tmp_config)
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"excluded_projects": ["a"], "redact_strings": []})
        saved = {}
        monkeypatch.setattr("agentstracer.cli.save_config", lambda c: saved.update(c))

        configure(exclude=["b", "c"])
        assert sorted(saved["excluded_projects"]) == ["a", "b", "c"]

    def test_sets_source(self, tmp_config, monkeypatch, capsys):
        monkeypatch.setattr("agentstracer.cli.CONFIG_FILE", tmp_config)
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"source": None})
        saved = {}
        monkeypatch.setattr("agentstracer.cli.save_config", lambda c: saved.update(c))

        configure(source="codex")
        assert saved["source"] == "codex"


# --- list_projects ---


class TestListProjects:
    def test_with_projects(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "agentstracer.cli.discover_projects",
            lambda: [{"display_name": "proj1", "session_count": 5, "total_size_bytes": 1024}],
        )
        monkeypatch.setattr(
            "agentstracer.cli.load_config",
            lambda: {"excluded_projects": []},
        )
        list_projects()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "proj1"

    def test_no_projects(self, monkeypatch, capsys):
        monkeypatch.setattr("agentstracer.cli.discover_projects", lambda: [])
        list_projects()
        captured = capsys.readouterr()
        assert "No Claude Code, Codex, Gemini CLI, OpenCode, OpenClaw, Kimi Code CLI, or Custom sessions" in captured.out

    def test_source_filter_codex(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "agentstracer.cli.discover_projects",
            lambda: [
                {"display_name": "proj1", "session_count": 5, "total_size_bytes": 1024, "source": "claude"},
                {"display_name": "codex:proj2", "session_count": 3, "total_size_bytes": 512, "source": "codex"},
            ],
        )
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"excluded_projects": []})
        list_projects(source_filter="codex")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "codex:proj2"
        assert data[0]["source"] == "codex"

    def test_no_projects_for_selected_source(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "agentstracer.cli.discover_projects",
            lambda: [{"display_name": "proj1", "session_count": 5, "total_size_bytes": 1024, "source": "claude"}],
        )
        list_projects(source_filter="codex")
        captured = capsys.readouterr()
        assert "No Codex sessions found." in captured.out

    def test_main_list_uses_configured_source_when_auto(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "agentstracer.cli.discover_projects",
            lambda: [
                {"display_name": "proj1", "session_count": 5, "total_size_bytes": 1024, "source": "claude"},
                {"display_name": "codex:proj2", "session_count": 3, "total_size_bytes": 512, "source": "codex"},
            ],
        )
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"source": "codex", "excluded_projects": []})
        monkeypatch.setattr("sys.argv", ["agentstracer", "list"])
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "codex:proj2"


class TestWorkflowGateMessages:
    @staticmethod
    def _extract_json(stdout: str) -> dict:
        start = stdout.find("{")
        assert start >= 0, f"No JSON payload found in output: {stdout!r}"
        return json.loads(stdout[start:])

    def test_confirm_without_export_shows_step_process(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "missing.jsonl"
        monkeypatch.setattr(
            "sys.argv",
            ["agentstracer", "confirm", "--file", str(missing)],
        )
        with pytest.raises(SystemExit):
            main()
        payload = self._extract_json(capsys.readouterr().out)
        assert payload["error"] == "No export file found."
        assert payload["blocked_on_step"] == "Step 1/2"
        assert len(payload["process_steps"]) == 2
        assert "export --output" in payload["process_steps"][0]

    def test_confirm_missing_full_name_explains_purpose_and_skip(self, tmp_path, monkeypatch, capsys):
        export_file = tmp_path / "export.jsonl"
        export_file.write_text('{"project":"p","model":"m","messages":[]}\n')
        monkeypatch.setattr(
            "sys.argv",
            [
                "agentstracer",
                "confirm",
                "--file",
                str(export_file),
                "--attest-full-name",
                "Asked for full name and scanned export.",
                "--attest-sensitive",
                "Asked about company/client/internal names and private URLs; none found.",
                "--attest-manual-scan",
                "Manually scanned 20 sessions across beginning/middle/end and reviewed findings.",
            ],
        )
        with pytest.raises(SystemExit):
            main()
        payload = self._extract_json(capsys.readouterr().out)
        assert payload["error"] == "Missing required --full-name for verification scan."
        assert "--skip-full-name-scan" in payload["hint"]
        assert payload["blocked_on_step"] == "Step 2/2"
        assert len(payload["process_steps"]) == 2

    def test_confirm_skip_full_name_scan_succeeds(self, tmp_path, monkeypatch, capsys):
        export_file = tmp_path / "export.jsonl"
        export_file.write_text('{"project":"p","model":"m","messages":[]}\n')
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {})
        monkeypatch.setattr("agentstracer.cli.save_config", lambda _c: None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "agentstracer",
                "confirm",
                "--file",
                str(export_file),
                "--skip-full-name-scan",
                "--attest-full-name",
                "User declined to share full name; skipped exact-name scan.",
                "--attest-sensitive",
                "I asked about company/client/internal names and private URLs; none found.",
                "--attest-manual-scan",
                "I performed a manual scan and reviewed 20 sessions across beginning, middle, and end.",
            ],
        )
        main()
        payload = self._extract_json(capsys.readouterr().out)
        assert payload["stage"] == "confirmed"
        assert payload["full_name_scan"]["skipped"] is True

    def test_export_requires_project_confirmation_with_full_flow(self, monkeypatch, capsys):
        monkeypatch.setattr("agentstracer.cli._has_session_sources", lambda _src: True)
        monkeypatch.setattr(
            "agentstracer.cli.discover_projects",
            lambda: [
                {
                    "display_name": "proj1",
                    "session_count": 2,
                    "total_size_bytes": 1024,
                    "source": "claude",
                }
            ],
        )
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"source": "all"})
        monkeypatch.setattr("sys.argv", ["agentstracer", "export"])
        with pytest.raises(SystemExit):
            main()
        payload = self._extract_json(capsys.readouterr().out)
        assert payload["error"] == "Project selection is not confirmed yet."
        assert payload["blocked_on_step"] == "Step 3/5"
        assert len(payload["process_steps"]) == 5
        assert "prep && agentstracer list" in payload["process_steps"][0]
        assert payload["required_action"].startswith("Send the full project/folder list")
        assert "in a message" in payload["required_action"]
        assert isinstance(payload["projects"], list)
        assert payload["projects"][0]["name"] == "proj1"
        assert payload["projects"][0]["sessions"] == 2

    def test_export_requires_explicit_source_selection(self, monkeypatch, capsys):
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {})
        monkeypatch.setattr("sys.argv", ["agentstracer", "export"])
        with pytest.raises(SystemExit):
            main()
        payload = self._extract_json(capsys.readouterr().out)
        assert payload["error"] == "Source scope is not confirmed yet."
        assert payload["blocked_on_step"] == "Step 2/5"
        assert len(payload["process_steps"]) == 5
        assert payload["allowed_sources"] == ["all", "both", "claude", "codex", "custom", "gemini", "kimi", "openclaw", "opencode"]
        assert payload["next_command"] == "agentstracer config --source all"

    def test_configure_next_steps_require_full_folder_presentation(self):
        steps, _next = _build_status_next_steps(
            "configure",
            {"projects_confirmed": False},
        )
        assert any("agentstracer list" in step for step in steps)
        assert any("FULL project/folder list" in step for step in steps)
        assert any("in your next message" in step for step in steps)
        assert any("source scope" in step.lower() for step in steps)

    def test_review_next_steps_explain_full_name_purpose_and_skip_option(self):
        steps, _next = _build_status_next_steps(
            "review",
            {},
        )
        assert any("exact-name privacy check" in step for step in steps)
        assert any("--skip-full-name-scan" in step for step in steps)


# --- _scan_high_entropy_strings ---


class TestScanHighEntropyStrings:
    def test_detects_real_secret(self):
        # A realistic API key-like string with high entropy and mixed chars
        secret = "aB3dE6gH9jK2mN5pQ8rS1tU4wX7yZ0c"
        content = f"some config here token {secret} and more text"
        results = _scan_high_entropy_strings(content)
        assert len(results) >= 1
        assert any(r["match"] == secret for r in results)
        # Entropy should be >= 4.0
        for r in results:
            if r["match"] == secret:
                assert r["entropy"] >= 4.0

    def test_filters_uuid(self):
        content = "id=550e8400e29b41d4a716446655440000 done"
        results = _scan_high_entropy_strings(content)
        assert not any("550e8400" in r["match"] for r in results)

    def test_filters_uuid_with_hyphens(self):
        # UUID with hyphens won't match the 20+ contiguous regex, but without hyphens should be filtered
        content = "id=550e8400-e29b-41d4-a716-446655440000 done"
        results = _scan_high_entropy_strings(content)
        assert not any("550e8400" in r["match"] for r in results)

    def test_filters_hex_hash(self):
        content = f"commit=abcdef1234567890abcdef1234567890abcdef12 done"
        results = _scan_high_entropy_strings(content)
        assert not any("abcdef1234567890" in r["match"] for r in results)

    def test_filters_known_prefix_eyj(self):
        content = "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 done"
        results = _scan_high_entropy_strings(content)
        assert not any(r["match"].startswith("eyJ") for r in results)

    def test_filters_known_prefix_ghp(self):
        content = "token=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345 done"
        results = _scan_high_entropy_strings(content)
        assert not any(r["match"].startswith("ghp_") for r in results)

    def test_filters_file_extension_path(self):
        content = "import=some_long_module_name_thing.py done"
        results = _scan_high_entropy_strings(content)
        assert not any(".py" in r["match"] for r in results)

    def test_filters_path_like(self):
        content = "path=src/components/authentication/LoginForm done"
        results = _scan_high_entropy_strings(content)
        assert not any("src/components" in r["match"] for r in results)

    def test_filters_low_entropy(self):
        # Repetitive string with mixed chars but low entropy
        content = "val=aaaaaaBBBBBB111111aaaaaaBBBBBB111111 done"
        results = _scan_high_entropy_strings(content)
        assert not any("aaaaaa" in r["match"] for r in results)

    def test_filters_no_mixed_chars(self):
        # All lowercase - no mixed char types
        content = "val=abcdefghijklmnopqrstuvwxyz done"
        results = _scan_high_entropy_strings(content)
        assert not any("abcdefghijklmnop" in r["match"] for r in results)

    def test_context_snippet(self):
        secret = "aB3dE6gH9jK2mN5pQ8rS1tU4wX7yZ0c"
        prefix = "before_context "
        suffix = " after_context"
        content = prefix + secret + suffix
        results = _scan_high_entropy_strings(content)
        matched = [r for r in results if r["match"] == secret]
        assert len(matched) == 1
        assert "before_context" in matched[0]["context"]
        assert "after_context" in matched[0]["context"]

    def test_results_capped_at_max(self):
        # Generate many distinct high-entropy strings
        import string
        import random
        rng = random.Random(42)
        chars = string.ascii_letters + string.digits
        secrets = []
        for _ in range(25):
            s = "".join(rng.choices(chars, k=30))
            secrets.append(s)
        content = " ".join(f"key={s}" for s in secrets)
        results = _scan_high_entropy_strings(content, max_results=15)
        assert len(results) <= 15

    def test_empty_content(self):
        assert _scan_high_entropy_strings("") == []

    def test_sorted_by_entropy_descending(self):
        secret1 = "aB3dE6gH9jK2mN5pQ8rS1tU4wX7yZ0c"
        secret2 = "Zx9Yw8Xv7Wu6Ts5Rq4Po3Nm2Lk1Jh0G"
        content = f"a={secret1} b={secret2}"
        results = _scan_high_entropy_strings(content)
        if len(results) >= 2:
            assert results[0]["entropy"] >= results[1]["entropy"]

    def test_filters_benign_prefix_https(self):
        content = "url=https://example.com/some/long/path/here done"
        results = _scan_high_entropy_strings(content)
        assert not any(r["match"].startswith("https://") for r in results)

    def test_filters_three_dots(self):
        content = "ver=com.example.app.module.v1.2.3 done"
        results = _scan_high_entropy_strings(content)
        assert not any("com.example.app" in r["match"] for r in results)

    def test_filters_node_modules(self):
        content = "path=some_long_node_modules_path_thing done"
        results = _scan_high_entropy_strings(content)
        assert not any("node_modules" in r["match"] for r in results)


# --- _scan_pii integration with high_entropy_strings ---


class TestScanPiiHighEntropy:
    def test_includes_high_entropy_when_present(self, tmp_path):
        secret = "aB3dE6gH9jK2mN5pQ8rS1tU4wX7yZ0c"
        f = tmp_path / "export.jsonl"
        f.write_text(f'{{"message": "config token {secret} end"}}\n')
        results = _scan_pii(f)
        assert "high_entropy_strings" in results
        assert any(r["match"] == secret for r in results["high_entropy_strings"])

    def test_excludes_high_entropy_when_clean(self, tmp_path):
        f = tmp_path / "export.jsonl"
        f.write_text('{"message": "nothing suspicious here at all"}\n')
        results = _scan_pii(f)
        assert "high_entropy_strings" not in results


# --- Bundle CLI commands ---


@pytest.fixture
def bundle_index(tmp_path, monkeypatch):
    """Set up an index DB with sessions for bundle testing."""
    monkeypatch.setattr("agentstracer.index.INDEX_DB", tmp_path / "index.db")
    monkeypatch.setattr("agentstracer.index.BLOBS_DIR", tmp_path / "blobs")
    monkeypatch.setattr("agentstracer.index.CONFIG_DIR", tmp_path / "agentstracer_config")

    from agentstracer.index import open_index, upsert_sessions

    conn = open_index()
    sessions = [
        {
            "session_id": f"sess-{i}",
            "project": "test-project",
            "source": "claude",
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": f"Task {i}"},
                {"role": "assistant", "content": "Done."},
            ],
            "stats": {"user_messages": 1, "assistant_messages": 1,
                       "tool_uses": 0, "input_tokens": 100, "output_tokens": 50},
        }
        for i in range(3)
    ]
    upsert_sessions(conn, sessions)

    # Approve sessions for bundle testing
    from agentstracer.index import update_session
    for i in range(3):
        update_session(conn, f"sess-{i}", status="approved")
    conn.close()
    return tmp_path


class TestBundleCreate:
    def test_create_by_ids(self, bundle_index, capsys):
        from agentstracer.cli import _run_bundle_create
        args = MagicMock(session_ids=["sess-0", "sess-1"], status=None,
                         note="test", attestation=None, json=True)
        _run_bundle_create(args)
        output = json.loads(capsys.readouterr().out)
        assert output["session_count"] == 2
        assert output["status"] == "draft"
        assert "bundle_id" in output

    def test_create_by_status(self, bundle_index, capsys):
        from agentstracer.cli import _run_bundle_create
        args = MagicMock(session_ids=[], status="approved", note=None, attestation=None, json=True)
        _run_bundle_create(args)
        output = json.loads(capsys.readouterr().out)
        assert output["session_count"] == 3

    def test_create_no_sessions_exits(self, bundle_index):
        from agentstracer.cli import _run_bundle_create
        args = MagicMock(session_ids=[], status=None, note=None, attestation=None)
        with pytest.raises(SystemExit):
            _run_bundle_create(args)


class TestBundleList:
    def test_list_empty(self, bundle_index, capsys):
        from agentstracer.cli import _run_bundle_list
        _run_bundle_list(MagicMock(json=True))
        output = json.loads(capsys.readouterr().out)
        assert output["total"] == 0
        assert output["bundles"] == []

    def test_list_after_create(self, bundle_index, capsys):
        from agentstracer.index import create_bundle, open_index
        conn = open_index()
        create_bundle(conn, ["sess-0"], note="test")
        conn.close()

        from agentstracer.cli import _run_bundle_list
        _run_bundle_list(MagicMock(json=True))
        output = json.loads(capsys.readouterr().out)
        assert output["total"] == 1
        assert output["bundles"][0]["session_count"] == 1

    def test_list_human(self, bundle_index, capsys):
        from agentstracer.cli import _run_bundle_list
        _run_bundle_list(MagicMock(json=False))
        out = capsys.readouterr().out
        assert "No bundles" in out


class TestBundleView:
    def test_view(self, bundle_index, capsys):
        from agentstracer.index import create_bundle, open_index
        conn = open_index()
        bundle_id = create_bundle(conn, ["sess-0", "sess-1"])
        conn.close()

        from agentstracer.cli import _run_bundle_view
        args = MagicMock(bundle_id=bundle_id, json=True)
        _run_bundle_view(args)
        output = json.loads(capsys.readouterr().out)
        assert output["bundle_id"] == bundle_id
        assert len(output["sessions"]) == 2

    def test_view_prefix(self, bundle_index, capsys):
        from agentstracer.index import create_bundle, open_index
        conn = open_index()
        bundle_id = create_bundle(conn, ["sess-0"])
        conn.close()

        from agentstracer.cli import _run_bundle_view
        args = MagicMock(bundle_id=bundle_id[:8], json=True)
        _run_bundle_view(args)
        output = json.loads(capsys.readouterr().out)
        assert output["bundle_id"] == bundle_id

    def test_view_not_found(self, bundle_index):
        from agentstracer.cli import _run_bundle_view
        args = MagicMock(bundle_id="nonexistent")
        with pytest.raises(SystemExit):
            _run_bundle_view(args)


class TestBundleExport:
    def test_export(self, bundle_index, capsys):
        from agentstracer.index import create_bundle, open_index
        conn = open_index()
        bundle_id = create_bundle(conn, ["sess-0", "sess-1"])
        conn.close()

        from agentstracer.cli import _run_bundle_export
        args = MagicMock(bundle_id=bundle_id, output=None, json=True)
        _run_bundle_export(args)
        output = json.loads(capsys.readouterr().out)
        assert output["session_count"] == 2
        assert "sessions.jsonl" in output["files"]
        assert Path(output["export_path"]).exists()

    def test_export_redacts_custom_strings(self, tmp_path, monkeypatch, capsys):
        """Bundle export applies redact_strings from config."""
        monkeypatch.setattr("agentstracer.index.INDEX_DB", tmp_path / "index.db")
        monkeypatch.setattr("agentstracer.index.BLOBS_DIR", tmp_path / "blobs")
        monkeypatch.setattr("agentstracer.index.CONFIG_DIR", tmp_path / "agentstracer_config")

        from agentstracer.index import create_bundle, open_index, upsert_sessions

        conn = open_index()
        sessions = [
            {
                "session_id": "redact-test",
                "project": "test-project",
                "source": "claude",
                "model": "claude-sonnet-4",
                "messages": [
                    {"role": "user", "content": "Hello MySecretName, check sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAA"},
                    {"role": "assistant", "content": "Sure MySecretName, done."},
                ],
                "stats": {"user_messages": 1, "assistant_messages": 1,
                           "tool_uses": 0, "input_tokens": 100, "output_tokens": 50},
            },
        ]
        upsert_sessions(conn, sessions)

        from agentstracer.index import update_session
        update_session(conn, "redact-test", status="approved")
        bundle_id = create_bundle(conn, ["redact-test"])
        conn.close()

        # Configure redact_strings — patch in cli's namespace where load_config is bound
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {"redact_strings": ["MySecretName"]})

        from agentstracer.cli import _run_bundle_export
        args = MagicMock(bundle_id=bundle_id, output=None, json=True)
        _run_bundle_export(args)
        output = json.loads(capsys.readouterr().out)

        # Read the exported JSONL and verify redaction
        sessions_file = Path(output["export_path"]) / "sessions.jsonl"
        content = sessions_file.read_text()
        assert "MySecretName" not in content, "Custom redact_string was not redacted"
        assert "sk-ant-api03" not in content, "API key was not redacted"


class TestSearch:
    def test_search_json(self, bundle_index, capsys):
        from agentstracer.cli import _run_search
        args = MagicMock(query="Task", limit=20, source=None, json=True)
        _run_search(args)
        output = json.loads(capsys.readouterr().out)
        assert output["query"] == "Task"
        # FTS might not be built for this test; just check structure
        assert "results" in output

    def test_search_table(self, bundle_index, capsys):
        from agentstracer.cli import _run_search
        args = MagicMock(query="zzzznonexistent", limit=20, source=None, json=False)
        _run_search(args)
        out = capsys.readouterr().out
        assert "No results" in out


class TestShareHelpers:
    def test_share_preview_json_returns_payload(self):
        payload = _share_preview([{"session_id": "s1", "display_title": "hello", "risk_badges": "[]"}], output_json=True)
        assert payload is not None
        assert payload["total"] == 1
        assert payload["sessions"][0]["session_id"] == "s1"

    def test_share_pii_status_warns_without_export(self, monkeypatch):
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {})
        status = _share_pii_status()
        assert status["level"] == "warn"
        assert "No recent export record" in status["message"]

    def test_share_pii_status_info_with_sanitized(self, monkeypatch):
        monkeypatch.setattr("agentstracer.cli.load_config", lambda: {
            "last_export": {
                "output_file": "/tmp/a.jsonl",
                "pii_review": {"finding_count": 3},
                "pii_apply": {"output": "/tmp/a.sanitized.jsonl"},
            }
        })
        status = _share_pii_status()
        assert status["level"] == "info"
        assert "/tmp/a.sanitized.jsonl" in status["message"]


class TestShare:
    def test_share_approved(self, bundle_index, capsys, monkeypatch):
        """share --status approved creates bundle + exports + shares."""
        from agentstracer.cli import _run_share

        def mock_share_bundle(conn, bundle_id, *, force=False, custom_strings=None):
            return {"ok": True, "session_count": 3, "bundle_hash": "abc123",
                    "shared_at": "2026-01-01",
                    "redaction_summary": {"total_redactions": 2, "by_type": {"jwt": 1, "email": 1}}}

        # _run_share imports share_bundle from agentstracer.daemon at call time
        monkeypatch.setattr("agentstracer.daemon.share_bundle", mock_share_bundle)

        args = MagicMock(session_ids=[], status="approved", note="test",
                         force=False, json=False, preview=False)
        _run_share(args)
        out = capsys.readouterr().out
        assert "Shared 3 sessions" in out
        assert "uploaded successfully" in out
        assert "Privacy:" in out
        assert "2 redactions applied" in out

    def test_share_preview(self, bundle_index, capsys):
        """share --preview shows session list without uploading."""
        from agentstracer.cli import _run_share
        args = MagicMock(session_ids=[], status="approved", note=None,
                         force=False, json=False, preview=True)
        _run_share(args)
        out = capsys.readouterr().out
        assert "sessions ready to share" in out


class TestScore:
    def test_score_single_session_returns_error_on_judge_failure(self, monkeypatch):
        from agentstracer.cli import _score_single_session

        monkeypatch.setattr(
            "agentstracer.scoring.score_session",
            lambda conn, session_id, model=None, backend="auto": (_ for _ in ()).throw(RuntimeError("backend auth failed")),
        )

        result = _score_single_session(object(), "sess-1")
        assert result["session_id"] == "sess-1"
        assert "Judge failed: backend auth failed" in result["error"]

    def test_score_help_includes_default_limit_10(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["agentstracer", "score", "--help"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "Max sessions for batch mode (default: 10)" in out

    def test_share_preview_json(self, bundle_index, capsys):
        """share --preview --json outputs session list as JSON."""
        from agentstracer.cli import _run_share
        args = MagicMock(session_ids=[], status="approved", note=None,
                         force=False, json=True, preview=True)
        _run_share(args)
        output = json.loads(capsys.readouterr().out)
        assert "sessions" in output
        assert output["total"] == 3

    def test_share_no_sessions_exits(self, bundle_index):
        from agentstracer.cli import _run_share
        args = MagicMock(session_ids=[], status=None, note=None,
                         force=False, json=False, preview=False)
        with pytest.raises(SystemExit):
            _run_share(args)


class TestPiiCli:
    def test_pii_review_and_apply(self, tmp_path, monkeypatch, capsys):
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        output_file = tmp_path / "output.jsonl"
        session = {
            "session_id": "s1",
            "messages": [
                {"content": '{"sender_id":"7859110712","name":"Kai D","username":"kaidagent"}'}
            ],
        }
        input_file.write_text(json.dumps(session) + "\n")

        monkeypatch.setattr("agentstracer.cli.review_session_pii_hybrid", lambda session, ignore_llm_errors=True, **kw: [
            {
                "session_id": "s1",
                "message_index": 0,
                "field": "content",
                "entity_text": "7859110712",
                "entity_type": "user_id",
                "confidence": 0.98,
                "reason": "id",
                "replacement": "[REDACTED_USER_ID]",
                "source": "rule",
            },
            {
                "session_id": "s1",
                "message_index": 0,
                "field": "content",
                "entity_text": "Kai D",
                "entity_type": "person_name",
                "confidence": 0.95,
                "reason": "name",
                "replacement": "[REDACTED_PERSON]",
                "source": "rule",
            },
            {
                "session_id": "s1",
                "message_index": 0,
                "field": "content",
                "entity_text": "kaidagent",
                "entity_type": "username",
                "confidence": 0.98,
                "reason": "username",
                "replacement": "[REDACTED_USERNAME]",
                "source": "rule",
            },
        ])
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--json",
        ])
        main()
        review_output = json.loads(capsys.readouterr().out)
        assert review_output["provider"] == "hybrid"
        assert review_output["finding_count"] >= 3

        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-apply",
            "--file", str(input_file),
            "--findings", str(findings_file),
            "--output", str(output_file),
            "--json",
        ])
        main()
        apply_output = json.loads(capsys.readouterr().out)
        assert apply_output["replacements"] >= 3
        data = json.loads(output_file.read_text().strip())
        assert "Kai D" not in json.dumps(data)
        assert "kaidagent" not in json.dumps(data)

    def _make_ai_provider_mock(self, monkeypatch):
        monkeypatch.setattr("agentstracer.cli.review_session_pii_with_agent", lambda session, **kw: [{
            "session_id": "s1",
            "message_index": 0,
            "field": "content",
            "entity_text": "Kai D",
            "entity_type": "person_name",
            "confidence": 0.95,
            "reason": "name",
            "replacement": "[REDACTED_PERSON]",
            "source": "ai",
        }])

    def test_pii_review_ai_provider(self, tmp_path, monkeypatch, capsys):
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": [{"content": "Kai D"}]}) + "\n")
        self._make_ai_provider_mock(monkeypatch)
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "ai",
            "--json",
        ])
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["provider"] == "ai"
        assert output["finding_count"] == 1

    def test_pii_review_claude_provider_backward_compat(self, tmp_path, monkeypatch, capsys):
        """--provider claude still works as an alias for --provider ai."""
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": [{"content": "Kai D"}]}) + "\n")
        self._make_ai_provider_mock(monkeypatch)
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "claude",
            "--json",
        ])
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["provider"] == "claude"
        assert output["finding_count"] == 1

    def test_pii_review_hybrid_tolerates_llm_failure(self, tmp_path, monkeypatch, capsys):
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": [{"content": '{"name":"Kai D"}'}]}) + "\n")
        monkeypatch.setattr("agentstracer.cli.review_session_pii_hybrid", lambda session, ignore_llm_errors=True, **kw: [{
            "session_id": "s1",
            "message_index": 0,
            "field": "content",
            "entity_text": "Kai D",
            "entity_type": "person_name",
            "confidence": 0.95,
            "reason": "name",
            "replacement": "[REDACTED_PERSON]",
            "source": "rule",
        }])
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "hybrid",
            "--json",
        ])
        main()
        output = json.loads(capsys.readouterr().out)
        assert output["provider"] == "hybrid"
        assert output["finding_count"] == 1

    def test_apply_pii_findings_helper(self, tmp_path):
        from agentstracer.cli import _apply_pii_findings
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        output_file = tmp_path / "output.jsonl"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": [{"content": "Kai D"}]}) + "\n")
        findings_file.write_text(json.dumps({"findings": [{
            "session_id": "s1",
            "message_index": 0,
            "field": "content",
            "entity_text": "Kai D",
            "entity_type": "person_name",
            "confidence": 0.95,
            "replacement": "[REDACTED_PERSON]"
        }]}))
        result = _apply_pii_findings(input_file, findings_file, output_file)
        assert result["replacements"] == 1
        assert "[REDACTED_PERSON]" in output_file.read_text()

    def test_pii_review_unimplemented_provider(self, tmp_path, monkeypatch):
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": []}) + "\n")
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "codex",
        ])
        with pytest.raises(SystemExit):
            main()

    def test_pii_review_passes_backend(self, tmp_path, monkeypatch, capsys):
        """--backend flag on pii-review is threaded to the dispatcher."""
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps({"session_id": "s1", "messages": [{"content": "Alice"}]}) + "\n")
        called_backends = []

        def fake_agent_review(session, backend="auto", **kw):
            called_backends.append(backend)
            return []

        monkeypatch.setattr("agentstracer.cli.review_session_pii_with_agent", fake_agent_review)
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "ai",
            "--backend", "codex",
            "--json",
        ])
        main()
        assert called_backends == ["codex"]

    def test_pii_review_progress_output(self, tmp_path, monkeypatch, capsys):
        """Non-rules providers print per-session progress to stderr."""
        input_file = tmp_path / "input.jsonl"
        findings_file = tmp_path / "findings.json"
        input_file.write_text(
            json.dumps({"session_id": "abc123", "messages": [{"content": "x"}]}) + "\n"
            + json.dumps({"session_id": "def456", "messages": [{"content": "y"}]}) + "\n"
        )
        monkeypatch.setattr("agentstracer.cli.review_session_pii_with_agent", lambda s, **kw: [])
        monkeypatch.setattr("sys.argv", [
            "agentstracer", "pii-review",
            "--file", str(input_file),
            "--output", str(findings_file),
            "--provider", "ai",
            "--json",
        ])
        main()
        stderr = capsys.readouterr().err
        assert "[1/2]" in stderr
        assert "[2/2]" in stderr
