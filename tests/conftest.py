"""Fixtures shared by the repository test suite."""

import pytest

from agentstracer.anonymizer import Anonymizer


@pytest.fixture
def mock_anonymizer(monkeypatch):
    """A deterministic real anonymizer for tests outside parser fixtures."""
    monkeypatch.setattr(
        "agentstracer.anonymizer._detect_home_dir",
        lambda: ("/Users/testuser", "testuser"),
    )
    return Anonymizer()


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Temporary config path used by CLI configuration tests."""
    path = tmp_path / "config.json"
    monkeypatch.setattr("agentstracer.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("agentstracer.config.CONFIG_FILE", path)
    return path
