"""Persistent config for ClawTrace — stored at ~/.clawtrace/config.json"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, cast

CONFIG_DIR = Path.home() / ".clawtrace"
CONFIG_FILE = CONFIG_DIR / "config.json"


class ClawTraceConfig(TypedDict, total=False):
    """Expected shape of the config dict."""

    repo: str | None
    source: str | None  # "claude" | "codex" | "gemini" | "all"
    excluded_projects: list[str]
    redact_strings: list[str]
    redact_usernames: list[str]
    allowlist_entries: list[dict]  # [{type, text/regex/match_type, scope, reason, added}]
    no_secrets_redaction: bool  # Private use: skip API key/secrets redaction
    last_export: dict
    stage: str | None  # "auth" | "configure" | "review" | "confirmed" | "done"
    projects_confirmed: bool  # True once user has addressed folder exclusions
    review_attestations: dict
    review_verification: dict
    last_confirm: dict
    publish_attestation: str
    daemon_port: int | None
    device_id: str | None
    device_token: str | None


DEFAULT_CONFIG: ClawTraceConfig = {
    "repo": None,
    "source": None,
    "excluded_projects": [],
    "redact_strings": [],
    "allowlist_entries": [],
    "no_secrets_redaction": False,
}


def load_config() -> ClawTraceConfig:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                stored = json.load(f)
            return cast(ClawTraceConfig, {**DEFAULT_CONFIG, **stored})
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {CONFIG_FILE}: {e}", file=sys.stderr)
    return cast(ClawTraceConfig, dict(DEFAULT_CONFIG))


def save_config(config: ClawTraceConfig) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=CONFIG_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f, indent=2)
            os.replace(tmp_path, CONFIG_FILE)
        except BaseException:
            os.unlink(tmp_path)
            raise
    except OSError as e:
        print(f"Warning: could not save {CONFIG_FILE}: {e}", file=sys.stderr)
