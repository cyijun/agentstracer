"""Tests for agentstracer.config — config persistence."""

import json

from agentstracer.config import load_config, save_config


class TestLoadConfig:
    def test_no_file_returns_defaults(self, tmp_config):
        config = load_config()
        assert config["repo"] is None
        assert config["excluded_projects"] == []
        assert config["redact_strings"] == []

    def test_valid_file_merged(self, tmp_config):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text(json.dumps({"repo": "alice/data", "custom_key": "val"}))
        config = load_config()
        assert config["repo"] == "alice/data"
        assert config["custom_key"] == "val"
        # Defaults still present
        assert "excluded_projects" in config

    def test_corrupt_json_returns_defaults(self, tmp_config, capsys):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text("not valid json {{{")
        config = load_config()
        assert config["repo"] is None
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_non_object_json_returns_defaults(self, tmp_config, capsys):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text("[]")
        config = load_config()
        assert config["repo"] is None
        assert config["redact_strings"] == []
        assert "Warning" in capsys.readouterr().err

    def test_invalid_known_field_types_are_ignored(self, tmp_config, capsys):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text(json.dumps({
            "redact_strings": "not-a-list",
            "no_secrets_redaction": 1,
            "custom_key": {"kept": True},
        }))
        config = load_config()
        assert config["redact_strings"] == []
        assert config["no_secrets_redaction"] is False
        assert config["custom_key"] == {"kept": True}
        assert "redact_strings" in capsys.readouterr().err

    def test_default_mutable_values_are_isolated_between_loads(self, tmp_config):
        first = load_config()
        first["excluded_projects"].append("private")
        second = load_config()
        assert second["excluded_projects"] == []

    def test_extra_keys_preserved(self, tmp_config):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text(json.dumps({"repo": None, "my_extra": [1, 2, 3]}))
        config = load_config()
        assert config["my_extra"] == [1, 2, 3]


class TestSaveConfig:
    def test_creates_dir_and_writes(self, tmp_config):
        save_config({"repo": "alice/data", "excluded_projects": []})
        assert tmp_config.exists()
        data = json.loads(tmp_config.read_text())
        assert data["repo"] == "alice/data"

    def test_overwrites_existing(self, tmp_config):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text(json.dumps({"repo": "old"}))
        save_config({"repo": "new"})
        data = json.loads(tmp_config.read_text())
        assert data["repo"] == "new"

    def test_uses_private_directory_and_file_permissions(self, tmp_config):
        save_config({"repo": "alice/data"})
        assert tmp_config.parent.stat().st_mode & 0o777 == 0o700
        assert tmp_config.stat().st_mode & 0o777 == 0o600

    def test_oserror_prints_warning(self, tmp_config, monkeypatch, capsys):
        # Make the directory unwritable
        monkeypatch.setattr(
            "agentstracer.config.CONFIG_DIR",
            tmp_config.parent / "nonexistent" / "deep" / "dir",
        )
        # Actually mock mkdir to raise
        def failing_mkdir(self, *a, **kw):
            raise OSError("Permission denied")

        monkeypatch.setattr(type(tmp_config.parent), "mkdir", failing_mkdir)
        assert save_config({"repo": "test"}) is False
        captured = capsys.readouterr()
        assert "Warning" in captured.err
