# Kimi Code Trace Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy `~/.kimi/sessions` parser in `agentstracer` with a parser that reads the new Kimi Code CLI session layout at `~/.kimi-code/sessions`, keeping the `kimi` source name transparent to users.

**Architecture:** Update directory constants and discovery in `agentstracer/parser.py`, rewrite `_parse_kimi_session_file` to consume the event-based `agents/main/wire.jsonl`, and add unit tests for both migrated (`ses_*`) and native (`session_*`) wire formats.

**Tech Stack:** Python 3.10+, pytest, standard library only.

## Global Constraints

- Source name remains `"kimi"`; no new source enum.
- Legacy `~/.kimi/sessions` support is intentionally removed.
- All paths and text content must pass through the existing `Anonymizer`.
- No new runtime dependencies.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `agentstracer/parser.py` | Directory discovery, wire parsing, message reconstruction |
| `tests/test_parser.py` | Unit tests for migrated and native wire formats |
| `README.md` | Update supported data location from `~/.kimi/sessions` to `~/.kimi-code/sessions` |

---

### Task 1: Update Kimi constants and remove legacy helpers

**Files:**
- Modify: `agentstracer/parser.py:43-46`
- Modify: `agentstracer/parser.py:77`

**Interfaces:**
- Produces: `KIMI_SESSIONS_DIR = Path.home() / ".kimi-code" / "sessions"`
- Removes: `KIMI_DIR`, `KIMI_CONFIG_PATH`, `_KIMI_PROJECT_INDEX`

- [ ] **Step 1: Edit constants block**

Replace lines 43-46:

```python
KIMI_DIR = Path.home() / ".kimi"
KIMI_SESSIONS_DIR = KIMI_DIR / "sessions"
KIMI_CONFIG_PATH = KIMI_DIR / "kimi.json"
UNKNOWN_KIMI_CWD = "<unknown-cwd>"
```

with:

```python
KIMI_SESSIONS_DIR = Path.home() / ".kimi-code" / "sessions"
UNKNOWN_KIMI_CWD = "<unknown-cwd>"
```

- [ ] **Step 2: Remove unused index variable**

Delete line 77:

```python
_KIMI_PROJECT_INDEX: dict[str, list[Path]] = {}
```

so the index block becomes:

```python
_CODEX_PROJECT_INDEX: dict[str, list[Path]] = {}
_GEMINI_HASH_MAP: dict[str, str] = {}
_OPENCODE_PROJECT_INDEX: dict[str, list[str]] = {}
_OPENCLAW_PROJECT_INDEX: dict[str, list[Path]] = {}
```

- [ ] **Step 3: Run parser import smoke test**

Run: `python -c "from agentstracer.parser import KIMI_SESSIONS_DIR; print(KIMI_SESSIONS_DIR)"`
Expected: `/home/<user>/.kimi-code/sessions`

- [ ] **Step 4: Commit**

```bash
git add agentstracer/parser.py
git commit -m "chore(parser): point kimi constants to ~/.kimi-code/sessions"
```

---

### Task 2: Rewrite Kimi project discovery

**Files:**
- Modify: `agentstracer/parser.py:284-357`

**Interfaces:**
- Consumes: `KIMI_SESSIONS_DIR`, `KIMI_SOURCE`
- Produces: `_discover_kimi_projects() -> list[dict]`, `_build_kimi_project_name(dir_name: str) -> str`

- [ ] **Step 1: Replace `_load_kimi_work_dirs`, `_get_kimi_project_hash`, `_discover_kimi_projects`, and `_build_kimi_project_name`**

Delete the existing functions from line 284 through 357 and replace with:

```python
def _discover_kimi_projects() -> list[dict]:
    """Discover Kimi Code projects under ~/.kimi-code/sessions.

    Layout: wd_<project_name>_<hash>/session_<uuid>/agents/main/wire.jsonl
    """
    if not KIMI_SESSIONS_DIR.exists():
        return []

    projects = []
    for project_dir in sorted(KIMI_SESSIONS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        if not project_dir.name.startswith("wd_"):
            continue

        session_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        if not session_dirs:
            continue

        total_sessions = 0
        total_size = 0
        for session_dir in session_dirs:
            wire_file = session_dir / "agents" / "main" / "wire.jsonl"
            if wire_file.exists():
                total_sessions += 1
                total_size += wire_file.stat().st_size

        if total_sessions == 0:
            continue

        projects.append(
            {
                "dir_name": project_dir.name,
                "display_name": _build_kimi_project_name(project_dir.name),
                "session_count": total_sessions,
                "total_size_bytes": total_size,
                "source": KIMI_SOURCE,
            }
        )
    return projects


def _build_kimi_project_name(dir_name: str) -> str:
    """Convert wd_<name>_<hash> into a readable project name."""
    if not dir_name.startswith("wd_"):
        return f"kimi:{dir_name}"
    parts = dir_name.split("_")
    if len(parts) >= 3:
        name = "_".join(parts[1:-1])
    else:
        name = dir_name
    return f"kimi:{name or dir_name}"
```

- [ ] **Step 2: Run discovery unit tests**

Run: `pytest tests/test_parser.py::TestDiscoverProjects -v`
Expected: existing tests still pass (they disable Kimi, so the change is neutral).

- [ ] **Step 3: Commit**

```bash
git add agentstracer/parser.py
git commit -m "feat(parser): discover kimi-code sessions under wd_<name>_<hash>"
```

---

### Task 3: Add wire parsing helpers

**Files:**
- Modify: `agentstracer/parser.py` (insert before `_parse_kimi_session_file`)

**Interfaces:**
- Produces:
  - `_extract_kimi_text_parts(content, anonymizer) -> list[str]`
  - `_extract_kimi_thinking_parts(content, anonymizer) -> list[str]`
  - `_build_kimi_tool_use(name, tool_call_id, arguments, anonymizer) -> dict`

- [ ] **Step 1: Insert helper functions**

Insert the following block immediately before `def _parse_kimi_session_file` (around line 1699):

```python
def _extract_kimi_text_parts(content: Any, anonymizer: Anonymizer) -> list[str]:
    """Extract text parts from a Kimi message content list."""
    parts: list[str] = []
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(anonymizer.text(text))
    return parts


def _extract_kimi_thinking_parts(content: Any, anonymizer: Anonymizer) -> list[str]:
    """Extract think parts from a Kimi message content list."""
    parts: list[str] = []
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "think":
            think = block.get("think", "").strip()
            if think:
                parts.append(anonymizer.text(think))
    return parts


def _build_kimi_tool_use(
    name: str | None,
    tool_call_id: str | None,
    arguments: Any,
    anonymizer: Anonymizer,
) -> dict[str, Any]:
    """Build a normalized tool_use dict from a Kimi tool call."""
    return {
        "tool": name,
        "id": tool_call_id,
        "input": _parse_tool_input(name, arguments, anonymizer),
    }
```

- [ ] **Step 2: Verify import and lint**

Run: `python -c "from agentstracer.parser import _extract_kimi_text_parts, _extract_kimi_thinking_parts, _build_kimi_tool_use; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agentstracer/parser.py
git commit -m "feat(parser): add kimi-code wire parsing helpers"
```

---

### Task 4: Rewrite `_parse_kimi_session_file`

**Files:**
- Modify: `agentstracer/parser.py:1699-1791`

**Interfaces:**
- Consumes: `_extract_kimi_text_parts`, `_extract_kimi_thinking_parts`, `_build_kimi_tool_use`, `_iter_jsonl`, `_normalize_timestamp`, `_make_stats`, `_make_session_result`, `_safe_int`
- Produces: `_parse_kimi_session_file(session_dir: Path, anonymizer: Anonymizer, include_thinking: bool = True) -> dict | None`

- [ ] **Step 1: Replace the function**

Replace `def _parse_kimi_session_file(...)` with:

```python
def _parse_kimi_session_file(
    session_dir: Path,
    anonymizer: Anonymizer,
    include_thinking: bool = True,
) -> dict | None:
    """Parse a Kimi Code session directory into structured session data."""
    messages: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "session_id": session_dir.name,
        "cwd": None,
        "git_branch": None,
        "model": None,
        "start_time": None,
        "end_time": None,
    }
    stats = _make_stats()

    state_file = session_dir / "state.json"
    try:
        state = json.loads(state_file.read_text())
        metadata["start_time"] = state.get("createdAt")
        metadata["end_time"] = state.get("updatedAt")
    except (OSError, json.JSONDecodeError):
        pass

    wire_file = session_dir / "agents" / "main" / "wire.jsonl"
    if not wire_file.exists():
        return None

    # Pre-pass: collect tool results from migrated role=tool messages.
    tool_result_map: dict[str, dict] = {}
    try:
        for entry in _iter_jsonl(wire_file):
            if entry.get("type") != "context.append_message":
                continue
            msg = entry.get("message", {})
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("toolCallId")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            text_parts = _extract_kimi_text_parts(msg.get("content"), anonymizer)
            tool_result_map[tool_call_id] = {
                "output": "\n\n".join(text_parts),
                "status": "success",
            }
    except OSError:
        pass

    current_step: dict[str, Any] | None = None

    try:
        for entry in _iter_jsonl(wire_file):
            event_type = entry.get("type")

            if event_type == "config.update":
                if metadata["model"] is None:
                    metadata["model"] = entry.get("modelAlias")
                continue

            if event_type == "usage.record":
                usage = entry.get("usage", {})
                stats["input_tokens"] += _safe_int(usage.get("inputOther"))
                stats["input_tokens"] += _safe_int(usage.get("inputCacheRead"))
                stats["output_tokens"] += _safe_int(usage.get("output"))
                continue

            if event_type == "context.append_message":
                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                timestamp = _normalize_timestamp(entry.get("time"))

                if role == "user":
                    text_parts = _extract_kimi_text_parts(msg.get("content"), anonymizer)
                    if text_parts:
                        messages.append({
                            "role": "user",
                            "content": "\n\n".join(text_parts),
                            "timestamp": timestamp,
                        })
                        stats["user_messages"] += 1
                        _update_time_bounds(metadata, timestamp)

                elif role == "assistant":
                    text_parts = _extract_kimi_text_parts(msg.get("content"), anonymizer)
                    thinking_parts = (
                        _extract_kimi_thinking_parts(msg.get("content"), anonymizer)
                        if include_thinking
                        else []
                    )
                    tool_uses = []
                    for tc in msg.get("toolCalls", []):
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        if isinstance(func, dict):
                            args_str = func.get("arguments", "")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError:
                                args = args_str
                            tool_use = _build_kimi_tool_use(
                                func.get("name"),
                                tc.get("id"),
                                args,
                                anonymizer,
                            )
                            result = tool_result_map.get(tool_use.get("id", ""))
                            if result:
                                tool_use["output"] = result["output"]
                                tool_use["status"] = result["status"]
                            tool_uses.append(tool_use)

                    assistant_msg: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        assistant_msg["content"] = "\n\n".join(text_parts)
                    if thinking_parts:
                        assistant_msg["thinking"] = "\n\n".join(thinking_parts)
                    if tool_uses:
                        assistant_msg["tool_uses"] = tool_uses
                        stats["tool_uses"] += len(tool_uses)

                    if text_parts or thinking_parts or tool_uses:
                        messages.append(assistant_msg)
                        stats["assistant_messages"] += 1
                        _update_time_bounds(metadata, timestamp)

                continue

            if event_type == "context.append_loop_event":
                event = entry.get("event", {})
                if not isinstance(event, dict):
                    continue
                sub_type = event.get("type")
                timestamp = _normalize_timestamp(entry.get("time"))

                if sub_type == "step.begin":
                    current_step = {
                        "text_parts": [],
                        "thinking_parts": [],
                        "tool_uses": [],
                        "timestamp": timestamp,
                    }

                elif sub_type == "content.part" and current_step is not None:
                    part = event.get("part", {})
                    if not isinstance(part, dict):
                        pass
                    elif part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text:
                            current_step["text_parts"].append(anonymizer.text(text))
                    elif part.get("type") == "think" and include_thinking:
                        think = part.get("think", "").strip()
                        if think:
                            current_step["thinking_parts"].append(anonymizer.text(think))

                elif sub_type == "tool.call" and current_step is not None:
                    args = event.get("args", {})
                    tool_use = _build_kimi_tool_use(
                        event.get("name"),
                        event.get("toolCallId"),
                        args,
                        anonymizer,
                    )
                    current_step["tool_uses"].append(tool_use)

                elif sub_type == "tool.result" and current_step is not None:
                    tool_call_id = event.get("toolCallId")
                    result = event.get("result", {})
                    output = result.get("output", "")
                    for tu in current_step["tool_uses"]:
                        if tu.get("id") == tool_call_id:
                            tu["output"] = anonymizer.text(str(output)) if isinstance(output, str) else output
                            tu["status"] = "success"
                            break

                elif sub_type == "step.end" and current_step is not None:
                    assistant_msg = {"role": "assistant"}
                    if current_step["text_parts"]:
                        assistant_msg["content"] = "\n\n".join(current_step["text_parts"])
                    if current_step["thinking_parts"]:
                        assistant_msg["thinking"] = "\n\n".join(current_step["thinking_parts"])
                    if current_step["tool_uses"]:
                        assistant_msg["tool_uses"] = current_step["tool_uses"]
                        stats["tool_uses"] += len(current_step["tool_uses"])

                    if (
                        current_step["text_parts"]
                        or current_step["thinking_parts"]
                        or current_step["tool_uses"]
                    ):
                        messages.append(assistant_msg)
                        stats["assistant_messages"] += 1
                        _update_time_bounds(metadata, current_step["timestamp"])
                    current_step = None

    except OSError:
        return None

    return _make_session_result(metadata, messages, stats)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agentstracer.parser import _parse_kimi_session_file; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agentstracer/parser.py
git commit -m "feat(parser): parse kimi-code wire.jsonl event stream"
```

---

### Task 5: Update `parse_project_sessions` Kimi branch

**Files:**
- Modify: `agentstracer/parser.py:453-476`

**Interfaces:**
- Consumes: `_parse_kimi_session_file`, `_build_kimi_project_name`, `KIMI_SESSIONS_DIR`, `KIMI_SOURCE`

- [ ] **Step 1: Replace the Kimi branch**

Replace:

```python
    if source == KIMI_SOURCE:
        project_hash = _get_kimi_project_hash(project_dir_name)
        project_path = KIMI_SESSIONS_DIR / project_hash
        if not project_path.exists():
            return []
        sessions = []
        for session_dir in sorted(project_path.iterdir()):
            if not session_dir.is_dir():
                continue
            context_file = session_dir / "context.jsonl"
            if not context_file.exists():
                continue
            parsed = _parse_kimi_session_file(
                context_file,
                anonymizer=anonymizer,
                include_thinking=include_thinking,
            )
            if parsed and parsed["messages"]:
                parsed["project"] = _build_kimi_project_name(project_dir_name)
                parsed["source"] = KIMI_SOURCE
                if not parsed.get("model"):
                    parsed["model"] = "kimi-k2"
                sessions.append(parsed)
        return sessions
```

with:

```python
    if source == KIMI_SOURCE:
        project_path = KIMI_SESSIONS_DIR / project_dir_name
        if not project_path.exists():
            return []
        sessions = []
        for session_dir in sorted(project_path.iterdir()):
            if not session_dir.is_dir():
                continue
            wire_file = session_dir / "agents" / "main" / "wire.jsonl"
            if not wire_file.exists():
                continue
            parsed = _parse_kimi_session_file(
                session_dir,
                anonymizer=anonymizer,
                include_thinking=include_thinking,
            )
            if parsed and parsed["messages"]:
                parsed["project"] = _build_kimi_project_name(project_dir_name)
                parsed["source"] = KIMI_SOURCE
                if not parsed.get("model"):
                    parsed["model"] = "kimi-code"
                sessions.append(parsed)
        return sessions
```

- [ ] **Step 2: Run parse_project_sessions tests**

Run: `pytest tests/test_parser.py::TestDiscoverProjects::test_parse_project_sessions -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add agentstracer/parser.py
git commit -m "feat(parser): wire kimi-code session parsing into parse_project_sessions"
```

---

### Task 6: Add test for migrated (`ses_*`) wire format

**Files:**
- Modify: `tests/test_parser.py`

**Interfaces:**
- Consumes: `KIMI_SESSIONS_DIR`, `parse_project_sessions`, `discover_projects`

- [ ] **Step 1: Add test class and fixture**

Append to `tests/test_parser.py`:

```python
class TestKimiCodeMigratedSession:
    def _disable_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agentstracer.parser.PROJECTS_DIR", tmp_path / "no-claude")
        monkeypatch.setattr("agentstracer.parser.CODEX_SESSIONS_DIR", tmp_path / "no-codex-sessions")
        monkeypatch.setattr("agentstracer.parser.CODEX_ARCHIVED_DIR", tmp_path / "no-codex-archived")
        monkeypatch.setattr("agentstracer.parser._CODEX_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.GEMINI_DIR", tmp_path / "no-gemini")
        monkeypatch.setattr("agentstracer.parser.OPENCODE_DB_PATH", tmp_path / "no-opencode.db")
        monkeypatch.setattr("agentstracer.parser._OPENCODE_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.OPENCLAW_AGENTS_DIR", tmp_path / "no-openclaw-agents")
        monkeypatch.setattr("agentstracer.parser._OPENCLAW_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.CUSTOM_DIR", tmp_path / "no-custom")

    def _make_migrated_session_dir(self, base: Path, project: str, session_id: str):
        session_dir = base / project / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "state.json").write_text(json.dumps({
            "createdAt": "2026-07-06T00:00:00+00:00",
            "updatedAt": "2026-07-06T00:01:00+00:00",
        }))
        wire_dir = session_dir / "agents" / "main"
        wire_dir.mkdir(parents=True)
        wire_lines = [
            json.dumps({"type": "metadata", "protocol_version": "1.0"}),
            json.dumps({
                "type": "context.append_message",
                "time": 1783000000000,
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                    "toolCalls": [],
                },
            }),
            json.dumps({
                "type": "context.append_message",
                "time": 1783000001000,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "think", "think": "I should greet the user."},
                        {"type": "text", "text": "Hi there!"},
                    ],
                    "toolCalls": [
                        {
                            "type": "function",
                            "id": "tool_1",
                            "function": {
                                "name": "Read",
                                "arguments": json.dumps({"path": "/tmp/file.txt"}),
                            },
                        },
                    ],
                },
            }),
            json.dumps({
                "type": "context.append_message",
                "time": 1783000002000,
                "message": {
                    "role": "tool",
                    "toolCallId": "tool_1",
                    "content": [{"type": "text", "text": "file contents"}],
                },
            }),
        ]
        (wire_dir / "wire.jsonl").write_text("\n".join(wire_lines) + "\n")
        return session_dir

    def test_discover_and_parse_migrated_session(self, tmp_path, monkeypatch, mock_anonymizer):
        self._disable_others(tmp_path, monkeypatch)
        kimi_dir = tmp_path / "kimi-code-sessions"
        monkeypatch.setattr("agentstracer.parser.KIMI_SESSIONS_DIR", kimi_dir)

        self._make_migrated_session_dir(kimi_dir, "wd_myapp_a1b2c3d4e5f6", "ses_00000000-0000-0000-0000-000000000001")

        projects = discover_projects()
        assert len(projects) == 1
        assert projects[0]["display_name"] == "kimi:myapp"
        assert projects[0]["session_count"] == 1

        sessions = parse_project_sessions("wd_myapp_a1b2c3d4e5f6", mock_anonymizer, source="kimi")
        assert len(sessions) == 1
        session = sessions[0]
        assert session["source"] == "kimi"
        assert session["project"] == "kimi:myapp"
        assert len(session["messages"]) == 2
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][0]["content"] == "Hello"
        assert session["messages"][1]["role"] == "assistant"
        assert session["messages"][1]["content"] == "Hi there!"
        assert session["messages"][1]["thinking"] == "I should greet the user."
        assert len(session["messages"][1]["tool_uses"]) == 1
        tool = session["messages"][1]["tool_uses"][0]
        assert tool["tool"] == "Read"
        assert tool["input"] == {"path": "/tmp/file.txt"}
        assert tool["output"] == "file contents"
        assert tool["status"] == "success"
        assert session["stats"]["user_messages"] == 1
        assert session["stats"]["assistant_messages"] == 1
        assert session["stats"]["tool_uses"] == 1
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_parser.py::TestKimiCodeMigratedSession -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_parser.py
git commit -m "test(parser): add migrated kimi-code session test"
```

---

### Task 7: Add test for native (`session_*`) wire format

**Files:**
- Modify: `tests/test_parser.py`

**Interfaces:**
- Consumes: `KIMI_SESSIONS_DIR`, `parse_project_sessions`, `_parse_kimi_session_file`

- [ ] **Step 1: Add test class and fixture**

Append to `tests/test_parser.py`:

```python
class TestKimiCodeNativeSession:
    def _disable_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agentstracer.parser.PROJECTS_DIR", tmp_path / "no-claude")
        monkeypatch.setattr("agentstracer.parser.CODEX_SESSIONS_DIR", tmp_path / "no-codex-sessions")
        monkeypatch.setattr("agentstracer.parser.CODEX_ARCHIVED_DIR", tmp_path / "no-codex-archived")
        monkeypatch.setattr("agentstracer.parser._CODEX_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.GEMINI_DIR", tmp_path / "no-gemini")
        monkeypatch.setattr("agentstracer.parser.OPENCODE_DB_PATH", tmp_path / "no-opencode.db")
        monkeypatch.setattr("agentstracer.parser._OPENCODE_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.OPENCLAW_AGENTS_DIR", tmp_path / "no-openclaw-agents")
        monkeypatch.setattr("agentstracer.parser._OPENCLAW_PROJECT_INDEX", {})
        monkeypatch.setattr("agentstracer.parser.CUSTOM_DIR", tmp_path / "no-custom")

    def _make_native_session_dir(self, base: Path, project: str, session_id: str):
        session_dir = base / project / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "state.json").write_text(json.dumps({
            "createdAt": "2026-07-06T00:00:00+00:00",
            "updatedAt": "2026-07-06T00:01:00+00:00",
        }))
        wire_dir = session_dir / "agents" / "main"
        wire_dir.mkdir(parents=True)
        wire_lines = [
            json.dumps({"type": "metadata", "protocol_version": "1.4"}),
            json.dumps({"type": "config.update", "modelAlias": "kimi-code/kimi-for-coding"}),
            json.dumps({
                "type": "context.append_message",
                "time": 1783000000000,
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Run a command"}],
                    "toolCalls": [],
                },
            }),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "step.begin", "step": 1}}),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "content.part", "part": {"type": "think", "think": "I will run ls."}}}),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "tool.call", "toolCallId": "tool_1", "name": "Bash", "args": {"command": "ls -la"}}}),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "tool.result", "toolCallId": "tool_1", "result": {"output": "file.txt"}}}),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "content.part", "part": {"type": "text", "text": "Done."}}}),
            json.dumps({"type": "context.append_loop_event", "time": 1783000001000, "event": {"type": "step.end", "step": 1}}),
            json.dumps({"type": "usage.record", "time": 1783000001000, "usage": {"inputOther": 100, "output": 20, "inputCacheRead": 50}}),
        ]
        (wire_dir / "wire.jsonl").write_text("\n".join(wire_lines) + "\n")
        return session_dir

    def test_native_step_reconstruction(self, tmp_path, monkeypatch, mock_anonymizer):
        self._disable_others(tmp_path, monkeypatch)
        kimi_dir = tmp_path / "kimi-code-sessions"
        monkeypatch.setattr("agentstracer.parser.KIMI_SESSIONS_DIR", kimi_dir)

        self._make_native_session_dir(kimi_dir, "wd_native_123456789abc", "session_00000000-0000-0000-0000-000000000001")

        sessions = parse_project_sessions("wd_native_123456789abc", mock_anonymizer, source="kimi")
        assert len(sessions) == 1
        session = sessions[0]
        assert session["model"] == "kimi-code/kimi-for-coding"
        assert session["project"] == "kimi:native"
        assert len(session["messages"]) == 2

        user_msg = session["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Run a command"

        assistant_msg = session["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["thinking"] == "I will run ls."
        assert assistant_msg["content"] == "Done."
        assert len(assistant_msg["tool_uses"]) == 1
        tool = assistant_msg["tool_uses"][0]
        assert tool["tool"] == "Bash"
        assert tool["input"] == {"command": "ls -la"}
        assert tool["output"] == "file.txt"
        assert tool["status"] == "success"

        assert session["stats"]["input_tokens"] == 150
        assert session["stats"]["output_tokens"] == 20
        assert session["stats"]["tool_uses"] == 1

    def test_include_thinking_false(self, tmp_path, monkeypatch, mock_anonymizer):
        self._disable_others(tmp_path, monkeypatch)
        kimi_dir = tmp_path / "kimi-code-sessions"
        monkeypatch.setattr("agentstracer.parser.KIMI_SESSIONS_DIR", kimi_dir)

        self._make_native_session_dir(kimi_dir, "wd_native_123456789abc", "session_00000000-0000-0000-0000-000000000001")

        sessions = parse_project_sessions("wd_native_123456789abc", mock_anonymizer, source="kimi", include_thinking=False)
        assert len(sessions) == 1
        assistant_msg = sessions[0]["messages"][1]
        assert "thinking" not in assistant_msg
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/test_parser.py::TestKimiCodeNativeSession -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_parser.py
git commit -m "test(parser): add native kimi-code step reconstruction test"
```

---

### Task 8: Update README data location

**Files:**
- Modify: `README.md:65`

**Interfaces:**
- Produces: updated supported-tools table

- [ ] **Step 1: Update the Kimi row**

Change:

```markdown
| Kimi CLI | `~/.kimi/sessions/` | ✅ |
```

to:

```markdown
| Kimi Code CLI | `~/.kimi-code/sessions/` | ✅ |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update kimi data location to ~/.kimi-code/sessions"
```

---

### Task 9: Run full test suite and smoke test

**Files:**
- None (verification only)

- [ ] **Step 1: Run parser tests**

Run: `pytest tests/test_parser.py -v`
Expected: all tests PASS

- [ ] **Step 2: Run CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: all tests PASS

- [ ] **Step 3: Smoke test against real `~/.kimi-code/sessions`**

Run:

```bash
python agentstracer/cli.py list --source kimi
```

Expected: lists Kimi Code projects such as `kimi:agentstracer` with non-zero session counts.

- [ ] **Step 4: Commit (if any fixes were needed)**

If no fixes were needed, no commit is necessary.

---

## Self-Review

1. **Spec coverage:**
   - Discovery rewrite → Task 2
   - Wire parsing (both formats) → Tasks 3-4
   - Tool result matching → Task 4 (tool.result handling and migrated role=tool pre-pass)
   - Model extraction → Task 4 (config.update)
   - Token stats → Task 4 (usage.record)
   - Tests for migrated + native formats → Tasks 6-7
   - README update → Task 8
   - No legacy support → Task 1 removes old constants/helpers

2. **Placeholder scan:** No TBD/TODO/fill-in-details; every step has exact code or commands.

3. **Type consistency:** `_parse_kimi_session_file` signature changes from `filepath: Path` to `session_dir: Path`; Task 5 updates the only caller. Helper signatures match usage in Task 4.
