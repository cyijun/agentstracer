"""Anonymize PII in Claude Code log data."""

import hashlib
import hmac
import os
import re
import secrets
import threading
from pathlib import Path


_ANONYMIZATION_KEY: bytes | None = None
_KEY_LOCK = threading.Lock()


def _get_anonymization_key() -> bytes:
    """Load or create the private installation key used for stable aliases."""
    global _ANONYMIZATION_KEY
    if _ANONYMIZATION_KEY is not None:
        return _ANONYMIZATION_KEY
    with _KEY_LOCK:
        if _ANONYMIZATION_KEY is not None:
            return _ANONYMIZATION_KEY
        try:
            from .config import CONFIG_DIR

            CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            CONFIG_DIR.chmod(0o700)
            key_path = Path(CONFIG_DIR) / "anonymization.key"
            try:
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                key = key_path.read_bytes()
            else:
                key = secrets.token_bytes(32)
                with os.fdopen(fd, "wb") as key_file:
                    key_file.write(key)
                    key_file.flush()
                    os.fsync(key_file.fileno())
            if len(key) < 32:
                raise OSError("invalid anonymization key")
            key_path.chmod(0o600)
            _ANONYMIZATION_KEY = key[:32]
        except OSError:
            # Read-only environments still get non-dictionary aliases for the
            # lifetime of the process, but cannot promise cross-run stability.
            _ANONYMIZATION_KEY = secrets.token_bytes(32)
        return _ANONYMIZATION_KEY


def _hash_username(username: str) -> str:
    # A keyed 64-bit identifier keeps deterministic local joins without making
    # common usernames recoverable through a public hash dictionary.
    digest = hmac.new(
        _get_anonymization_key(),
        username.encode("utf-8", errors="replace"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return "user_" + digest


def _detect_home_dir() -> tuple[str, str]:
    home = os.path.expanduser("~")
    username = os.path.basename(home)
    return home, username


def anonymize_path(path: str, username: str, username_hash: str, home: str | None = None) -> str:
    """Strip a path to project-relative and hash the username."""
    if not path:
        return path

    if home is None:
        home = os.path.expanduser("~")
    # Canonicalize separators for matching even when Windows traces are being
    # processed on macOS/Linux (or vice versa).
    normalized = path.replace("\\", "/")
    normalized_home = home.replace("\\", "/").rstrip("/")
    prefixes = set()
    for base in (f"/Users/{username}", f"/home/{username}", normalized_home):
        base = base.rstrip("/")
        for subdir in ("Documents", "Downloads", "Desktop"):
            prefixes.add(f"{base}/{subdir}/")
        prefixes.add(f"{base}/")

    # Try longest prefixes first (subdirectory matches before bare home)
    home_patterns = sorted(prefixes, key=len, reverse=True)

    for prefix in home_patterns:
        if normalized.casefold().startswith(prefix.casefold()):
            rest = normalized[len(prefix):]
            if "/Documents/" in prefix or "/Downloads/" in prefix or "/Desktop/" in prefix:
                return rest
            return f"{username_hash}/{rest}"

    escaped = re.escape(username)
    normalized = re.sub(
        rf"(?i)(?:[A-Za-z]:)?/(?:Users|home)/{escaped}(?=/|$)",
        f"/{username_hash}",
        normalized,
    )
    return normalized


def anonymize_text(text: str, username: str, username_hash: str) -> str:
    if not text or not username:
        return text

    escaped = re.escape(username)

    # Replace Unix and Windows home paths.  Canonical slash output prevents a
    # drive letter or separator style from retaining identifying information.
    text = re.sub(
        rf"(?:[A-Za-z]:)?[\\/](?:Users|home)[\\/]{escaped}(?=[\\/]|[^a-zA-Z0-9_-]|$)",
        f"/{username_hash}",
        text,
        flags=re.IGNORECASE,
    )

    # Catch hyphen-encoded paths: -Users-peteromalley- or -Users-peteromalley/
    text = re.sub(rf"-Users-{escaped}(?=-|/|$)", f"-Users-{username_hash}", text, flags=re.IGNORECASE)
    text = re.sub(rf"-home-{escaped}(?=-|/|$)", f"-home-{username_hash}", text, flags=re.IGNORECASE)

    # Also handle underscore-to-hyphen encoding: kaid_aiagent → kaid-aiagent
    if "_" in username:
        hyphen_variant = username.replace("_", "-")
        hyphen_escaped = re.escape(hyphen_variant)
        text = re.sub(rf"-Users-{hyphen_escaped}(?=-|/|$)", f"-Users-{username_hash}", text, flags=re.IGNORECASE)
        text = re.sub(rf"-home-{hyphen_escaped}(?=-|/|$)", f"-home-{username_hash}", text, flags=re.IGNORECASE)

    # Catch temp paths like /private/tmp/claude-501/-Users-peteromalley/
    text = re.sub(rf"claude-\d+/-Users-{escaped}", f"claude-XXX/-Users-{username_hash}", text)

    # The username is known, not heuristically guessed, so even short login
    # names must be removed.  ASCII identifier boundaries avoid replacing
    # e.g. ``bob`` inside ``bobcat``.
    text = re.sub(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        username_hash,
        text,
        flags=re.IGNORECASE,
    )

    return text


class Anonymizer:
    """Stateful anonymizer that consistently hashes usernames."""

    def __init__(self, extra_usernames: list[str] | None = None):
        self.home, self.username = _detect_home_dir()
        self.username_hash = _hash_username(self.username)

        # Additional usernames to anonymize (GitHub handles, Discord names, etc.)
        self._extra: list[tuple[str, str]] = []
        seen = {self.username.casefold()}
        names: list[str] = []
        for raw_name in (extra_usernames or []):
            name = raw_name.strip()
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                names.append(name)
        # Longer names first avoids an explicit short handle consuming part of
        # another configured handle.
        for name in sorted(names, key=len, reverse=True):
            name = name.strip()
            self._extra.append((name, _hash_username(name)))

    def path(self, file_path: str) -> str:
        result = anonymize_path(file_path, self.username, self.username_hash, self.home)
        result = anonymize_text(result, self.username, self.username_hash)
        for name, hashed in self._extra:
            result = _replace_username(result, name, hashed)
        return result

    def text(self, content: str) -> str:
        result = anonymize_text(content, self.username, self.username_hash)
        for name, hashed in self._extra:
            result = _replace_username(result, name, hashed)
        return result


def _replace_username(text: str, username: str, username_hash: str) -> str:
    if not text or not username:
        return text
    escaped = re.escape(username)
    text = re.sub(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        username_hash,
        text,
        flags=re.IGNORECASE,
    )
    return text
