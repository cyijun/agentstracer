"""Persistent config for AgentsTrace — stored at ~/.agentstracer/config.json"""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, cast

CONFIG_DIR = Path.home() / ".agentstracer"
CONFIG_FILE = CONFIG_DIR / "config.json"


class AgentsTraceConfig(TypedDict, total=False):
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


DEFAULT_CONFIG: AgentsTraceConfig = {
    "repo": None,
    "source": None,
    "excluded_projects": [],
    "redact_strings": [],
    "redact_usernames": [],
    "allowlist_entries": [],
    "no_secrets_redaction": False,
}


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_dict_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _valid_field(key: str, value: object) -> bool:
    nullable_strings = {"repo", "source", "stage", "device_id", "device_token"}
    string_lists = {"excluded_projects", "redact_strings", "redact_usernames"}
    dict_fields = {
        "last_export", "review_attestations", "review_verification", "last_confirm",
    }
    if key in nullable_strings:
        return value is None or isinstance(value, str)
    if key in string_lists:
        return _is_string_list(value)
    if key == "allowlist_entries":
        return _is_dict_list(value)
    if key in {"no_secrets_redaction", "projects_confirmed"}:
        return isinstance(value, bool)
    if key in dict_fields:
        return isinstance(value, dict)
    if key == "publish_attestation":
        return isinstance(value, str)
    if key == "daemon_port":
        return value is None or (isinstance(value, int) and not isinstance(value, bool))
    # Preserve forward-compatible/third-party keys without pretending to know
    # their schema.
    return True


def _default_config() -> AgentsTraceConfig:
    return cast(AgentsTraceConfig, copy.deepcopy(DEFAULT_CONFIG))


def _harden_permissions() -> None:
    """Keep privacy-sensitive config storage private on multi-user systems."""
    if CONFIG_DIR.exists():
        os.chmod(CONFIG_DIR, 0o700)
    if CONFIG_FILE.exists():
        os.chmod(CONFIG_FILE, 0o600)


def load_config() -> AgentsTraceConfig:
    if CONFIG_FILE.exists():
        try:
            _harden_permissions()
            with open(CONFIG_FILE, encoding="utf-8") as f:
                stored = json.load(f)
            if not isinstance(stored, dict):
                raise ValueError("config root must be a JSON object")
            clean: dict = {}
            invalid: list[str] = []
            for key, value in stored.items():
                if _valid_field(key, value):
                    clean[key] = value
                else:
                    invalid.append(key)
            if invalid:
                print(
                    f"Warning: ignored invalid config fields: {', '.join(sorted(invalid))}",
                    file=sys.stderr,
                )
            return cast(AgentsTraceConfig, {**_default_config(), **clean})
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"Warning: could not read {CONFIG_FILE}: {e}", file=sys.stderr)
    return _default_config()


def save_config(config: AgentsTraceConfig) -> bool:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(CONFIG_DIR, 0o700)
        fd, tmp_path = tempfile.mkstemp(dir=CONFIG_DIR, suffix=".tmp")
        try:
            os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, CONFIG_FILE)
            os.chmod(CONFIG_FILE, 0o600)
            return True
        except BaseException:
            os.unlink(tmp_path)
            raise
    except OSError as e:
        print(f"Warning: could not save {CONFIG_FILE}: {e}", file=sys.stderr)
        return False
