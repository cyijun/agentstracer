# Kimi Code Trace Parser Adaptation

## Objective

Make `agentstracer` correctly parse conversation traces produced by the new Kimi Code CLI (`../kimi-code`), replacing the legacy Kimi CLI (`~/.kimi/sessions`) parser.

## Background

The original `agentstracer` parser reads legacy Kimi CLI sessions from:

```
~/.kimi/sessions/<md5_hash>/<uuid>/context.jsonl
```

The new Kimi Code CLI stores sessions at:

```
~/.kimi-code/sessions/wd_<project_name>_<hash>/session_<uuid>/
    state.json
    agents/main/wire.jsonl
```

Current `agentstracer` cannot see the new directory and cannot parse the new event-based `wire.jsonl` format, so exported Kimi traces are empty.

## Design Decisions

1. **Drop legacy Kimi CLI support.** Only scan `~/.kimi-code/sessions`.
2. **Keep the source name `kimi`.** Existing user configs (`--source kimi`, `--source all`) continue to work without migration.
3. **In-place rewrite.** Replace `_discover_kimi_projects` and `_parse_kimi_session_file` in `agentstracer/parser.py`; delete all legacy-only helpers.
4. **Support both migrated and native wire formats.** Migrated sessions (`ses_*`) use only `context.append_message`; native sessions (`session_*`) also emit `context.append_loop_event` step events.

## Implementation

### Discovery

- Change `KIMI_SESSIONS_DIR` to `Path.home() / ".kimi-code" / "sessions"`.
- Remove `KIMI_DIR`, `KIMI_CONFIG_PATH`, `_load_kimi_work_dirs`, `_get_kimi_project_hash`, and the MD5 bucket logic.
- `_discover_kimi_projects` iterates over `wd_<name>_<hash>` directories.
  - Each workdir directory contains `session_<uuid>` subdirectories.
  - Use `session_index.jsonl` to map `<name>_<hash>` to the real working directory when available.
  - Fallback: derive a readable project name from the `wd_...` directory name (strip `wd_` prefix and hash suffix).
- `parse_project_sessions` resolves `project_dir_name` to the matching workdir directory and iterates over its session subdirectories.

### Wire format parsing

For each session read:

- `state.json` for `createdAt`, `updatedAt`, `title`, `agents.main.homedir`.
- `agents/main/wire.jsonl` as a JSONL event stream.

Supported event types:

| Event | Purpose |
|-------|---------|
| `metadata` | protocol version, session creation time |
| `config.update` | model alias, system prompt |
| `turn.prompt` | user turn input (also captured via `context.append_message`) |
| `context.append_message` | user / assistant / tool messages (migration format and user messages in native format) |
| `context.append_loop_event` | native assistant step details: `step.begin`, `content.part`, `tool.call`, `tool.result`, `step.end` |
| `usage.record` | per-turn token usage |

### Message reconstruction

- **User messages**: `context.append_message` with `role=user`; extract `text` content parts.
- **Assistant messages (migrated)**: `context.append_message` with `role=assistant`; extract `text`/`think` content parts and `toolCalls`.
- **Assistant messages (native)**: aggregate all events between a `step.begin` and matching `step.end`:
  - `content.part type=think` → `thinking`
  - `content.part type=text` → `content`
  - `tool.call` → start a `tool_use` with `tool`/`input`/`id`
  - `tool.result` with matching `toolCallId` → populate `output`/`status`
- **Tool uses**: count toward `stats["tool_uses"]`, inputs anonymized via `_parse_tool_input`.
- **Model**: from `config.update.modelAlias`; fallback to `kimi-code`.
- **Timestamps**: event `time` is milliseconds since epoch; normalize to ISO 8601 via existing `_normalize_timestamp`. Fall back to `state.json` `createdAt`/`updatedAt`.
- **Stats**: reuse `_make_stats`, `_update_time_bounds`, `_make_session_result`.

### Error handling

- Skip malformed JSON lines.
- Skip sessions whose wire file has no usable messages.
- If `state.json` is missing, infer session id from directory name and timestamps from wire events.
- Preserve existing anonymizer behavior for paths, commands, and text.

## Testing Plan

Add fixtures and tests in `tests/test_parser.py`:

1. **Migrated session fixture** (`ses_*` style): only `context.append_message` events.
   - Assert user + assistant messages are extracted.
   - Assert `think` content part becomes `thinking` when enabled.
   - Assert `toolCalls` become `tool_uses`.

2. **Native session fixture** (`session_*` style): `context.append_loop_event` events.
   - Assert a full assistant step is reconstructed from `step.begin` → `content.part`/`tool.call`/`tool.result` → `step.end`.
   - Assert tool result matching by `toolCallId`.
   - Assert `usage.record` updates token stats.

3. **Discovery test**:
   - Create `wd_test_<hash>/session_<uuid>/state.json` + `agents/main/wire.jsonl` under a temp `~/.kimi-code/sessions`.
   - Assert `discover_projects()` returns a `kimi:test` project with one session.

4. **Regression test**:
   - Ensure `--source all` still includes `kimi` in the allowed sources list.

## Rollback

This change intentionally removes legacy `~/.kimi/sessions` support. If users need legacy data, they must first migrate via the Kimi Code CLI's built-in migration, which writes imported sessions into `~/.kimi-code/sessions`.
