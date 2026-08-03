"""Provider-specific adapters that preserve agent execution structure.

Unlike :mod:`agentstracer.parser`, these adapters do not flatten independent
agent streams into one artificial dialogue.  They emit a canonical tree of
agent, generation, tool, and event observations with source timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .parser import (
    CODEX_ARCHIVED_DIR,
    CODEX_SESSIONS_DIR,
    GEMINI_DIR,
    KIMI_SESSIONS_DIR,
    OPENCODE_DB_PATH,
    OPENCLAW_AGENTS_DIR,
    PROJECTS_DIR,
    _openclaw_session_identity,
)
from .trace_model import (
    AgentTrace,
    TraceLink,
    TraceObservation,
    bounds,
    canonical_json,
    file_mtime_ns,
    timestamp_ns,
)

logger = logging.getLogger(__name__)

TRACE_SOURCES = ("claude", "codex", "gemini", "kimi", "opencode", "openclaw")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with path.open(errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(entry, dict):
                    entry.setdefault("_agentstracer_line", line_no)
                    entry.setdefault("_agentstracer_file", path.name)
                    entries.append(entry)
    except OSError:
        return []
    return entries


def _content_text(content: Any, *types: str) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if not isinstance(content, list):
        return None
    wanted = set(types or ("text",))
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in wanted:
            continue
        value = block.get("text")
        if value is None and block.get("type") in ("thinking", "think"):
            value = block.get("thinking", block.get("think"))
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts) or None


def _tool_output(value: Any) -> Any:
    """Normalize the many tool-result payload shapes without losing data."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    return str(value)


def _entry_id(prefix: str, entry: dict[str, Any]) -> str:
    native = (
        entry.get("uuid") or entry.get("id") or
        (entry.get("payload") or {}).get("id")
    )
    if native:
        return f"{prefix}:{native}"
    ordinal = entry.get("ordinal")
    if isinstance(ordinal, int) and not isinstance(ordinal, bool):
        return f"{prefix}:ordinal:{ordinal}"
    return f"{prefix}:{entry.get('_agentstracer_file')}:{entry.get('_agentstracer_line')}"


def _usage(value: Any, *, claude: bool = False) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}

    def safe_int(raw: Any) -> int:
        if raw is None or isinstance(raw, bool):
            return 0
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if not raw:
                    return 0
                try:
                    return int(raw)
                except ValueError:
                    return int(float(raw))
            return int(raw)
        except (TypeError, ValueError, OverflowError):
            return 0

    if claude:
        result = {
            "input": safe_int(value.get("input_tokens")),
            "output": safe_int(value.get("output_tokens")),
            "cache_read": safe_int(value.get("cache_read_input_tokens")),
            "cache_write": safe_int(value.get("cache_creation_input_tokens")),
        }
    else:
        cache = value.get("cache") if isinstance(value.get("cache"), dict) else {}
        result = {
            "input": safe_int(value.get("input") or value.get("input_tokens")),
            "output": safe_int(value.get("output") or value.get("output_tokens")),
            "cache_read": safe_int(
                value.get("cacheRead") or value.get("cached_input_tokens") or
                cache.get("read")
            ),
            "cache_write": safe_int(
                value.get("cacheWrite") or value.get("cache_write_input_tokens") or
                cache.get("write")
            ),
            "reasoning": safe_int(value.get("reasoning_output_tokens")),
        }
    return {key: val for key, val in result.items() if val}


def _select_interval(ts: int | None, intervals: list[tuple[int, int]]) -> int:
    if not intervals or ts is None:
        return 0
    for index, (start, end) in enumerate(intervals):
        if start <= ts < end:
            return index
    if ts < intervals[0][0]:
        return 0
    return len(intervals) - 1


# ---------------------------------------------------------------------------
# Claude Code


def _claude_result_map(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if call_id:
                result[str(call_id)] = {"entry": entry, "block": block}
    return result


def _claude_message_text(entry: dict[str, Any], include_thinking: bool) -> dict[str, Any]:
    message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
    content = message.get("content")
    output: dict[str, Any] = {}
    text = _content_text(content, "text")
    if text:
        output["content"] = text
    if include_thinking:
        thinking = _content_text(content, "thinking")
        if thinking:
            output["thinking"] = thinking
    calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input"),
                })
    if calls:
        output["tool_calls"] = calls
    return output


def _claude_stream_observations(
    entries: list[dict[str, Any]],
    *,
    parent: str,
    stream_id: str,
    result_map: dict[str, dict[str, Any]],
    include_thinking: bool,
) -> tuple[list[TraceObservation], dict[str, str], str | None]:
    observations: list[TraceObservation] = []
    tool_ids: dict[str, str] = {}
    previous_ts: int | None = None
    previous_user: Any = None
    last_text: str | None = None
    for entry in sorted(
        entries,
        key=lambda item: (
            timestamp_ns(item.get("timestamp")) or 0,
            item.get("_agentstracer_line", 0),
        ),
    ):
        ts = timestamp_ns(entry.get("timestamp"))
        if ts is None:
            ts = previous_ts
        kind = entry.get("type")
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        if kind == "user":
            content = message.get("content")
            text = _content_text(content, "text")
            if text:
                previous_user = text
            if ts is not None:
                previous_ts = ts
            continue
        if kind != "assistant" or ts is None:
            continue
        output = _claude_message_text(entry, include_thinking)
        if not output:
            previous_ts = ts
            continue
        generation_id = _entry_id(f"{stream_id}:generation", entry)
        start = previous_ts if previous_ts is not None and previous_ts <= ts else ts
        model = message.get("model") if isinstance(message.get("model"), str) else None
        observations.append(TraceObservation(
            logical_id=generation_id,
            name="generate-agent-response",
            as_type="generation",
            start_ns=start,
            end_ns=ts,
            parent_logical_id=parent,
            input=previous_user,
            output=output,
            model=model,
            usage=_usage(message.get("usage"), claude=True),
            metadata={
                "source_event_id": entry.get("uuid"),
                "timestamp_quality": "end_exact_start_inferred",
                "agent_id": entry.get("agentId"),
            },
        ))
        if isinstance(output.get("content"), str):
            last_text = output["content"]
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = str(block.get("id") or _entry_id("claude-tool", entry))
                logical_id = f"{stream_id}:tool:{call_id}"
                result = result_map.get(call_id)
                result_entry = result.get("entry") if result else None
                result_block = result.get("block") if result else None
                end = timestamp_ns(result_entry.get("timestamp")) if result_entry else None
                end = end if end is not None and end >= ts else ts
                raw_output = result_block.get("content") if result_block else None
                is_error = bool(result_block and result_block.get("is_error"))
                observations.append(TraceObservation(
                    logical_id=logical_id,
                    name=str(block.get("name") or "tool-call"),
                    as_type="tool",
                    start_ns=ts,
                    end_ns=end,
                    parent_logical_id=parent,
                    input=block.get("input"),
                    output=_tool_output(raw_output),
                    level="ERROR" if is_error else "DEFAULT",
                    status_message="tool result marked as error" if is_error else None,
                    metadata={
                        "tool_call_id": call_id,
                        "requested_by_generation": generation_id,
                        "timestamp_quality": "exact" if result else "open_call",
                    },
                    links=[TraceLink(
                        generation_id, {"agentstracer.relationship": "requested_by"},
                    )],
                ))
                tool_ids[call_id] = logical_id
        previous_ts = ts
    return observations, tool_ids, last_text


def iter_claude_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    if not PROJECTS_DIR.exists():
        return
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        for root_file in sorted(project_dir.glob("*.jsonl")):
            root_entries = _jsonl(root_file)
            if not root_entries:
                continue
            session_id = next(
                (str(e["sessionId"]) for e in root_entries if e.get("sessionId")),
                root_file.stem,
            )
            sub_files = sorted((project_dir / root_file.stem / "subagents").glob("agent-*.jsonl"))
            sub_streams = [(path, _jsonl(path)) for path in sub_files]
            all_entries = [*root_entries, *(e for _, stream in sub_streams for e in stream)]
            result_map = _claude_result_map(all_entries)

            prompts: list[tuple[int, dict[str, Any], str]] = []
            for entry in root_entries:
                if entry.get("type") != "user" or entry.get("isMeta"):
                    continue
                text = _content_text((entry.get("message") or {}).get("content"), "text")
                ts = timestamp_ns(entry.get("timestamp"))
                if text and ts is not None:
                    prompts.append((ts, entry, text))
            if not prompts:
                start, _ = bounds((timestamp_ns(e.get("timestamp")) for e in all_entries), file_mtime_ns(root_file))
                prompts = [(start, root_entries[0], "[session without a root user prompt]")]
            prompts.sort(key=lambda item: item[0])
            intervals: list[tuple[int, int]] = []
            for index, (start, _, _) in enumerate(prompts):
                end = prompts[index + 1][0] if index + 1 < len(prompts) else 2**63 - 1
                intervals.append((start, end))

            root_by_turn: list[list[dict[str, Any]]] = [[] for _ in intervals]
            for entry in root_entries:
                index = _select_interval(timestamp_ns(entry.get("timestamp")), intervals)
                root_by_turn[index].append(entry)
            sub_by_turn: list[list[tuple[Path, list[dict[str, Any]]]]] = [[] for _ in intervals]
            for path, stream in sub_streams:
                first = min((timestamp_ns(e.get("timestamp")) for e in stream if timestamp_ns(e.get("timestamp")) is not None), default=None)
                sub_by_turn[_select_interval(first, intervals)].append((path, stream))

            # Root tool result metadata is the strongest available mapping from
            # a completed Agent/Task tool call to its subagent file.
            spawn_by_agent: dict[str, str] = {}
            for call_id, result in result_map.items():
                entry = result["entry"]
                tool_result = entry.get("toolUseResult")
                if isinstance(tool_result, dict):
                    agent_id = tool_result.get("agentId") or tool_result.get("agent_id")
                    if agent_id:
                        spawn_by_agent[str(agent_id)] = call_id

            for turn_index, (prompt_ts, prompt_entry, prompt_text) in enumerate(prompts):
                root_logical = "root-agent"
                observations, tool_ids, last_root_text = _claude_stream_observations(
                    root_by_turn[turn_index],
                    parent=root_logical,
                    stream_id="root",
                    result_map=result_map,
                    include_thinking=include_thinking,
                )
                subagent_observations: list[TraceObservation] = []
                for path, stream in sub_by_turn[turn_index]:
                    if not stream:
                        continue
                    agent_id = str(stream[0].get("agentId") or path.stem.removeprefix("agent-"))
                    agent_logical = f"agent:{agent_id}"
                    call_id = spawn_by_agent.get(agent_id)
                    parent = tool_ids.get(call_id, root_logical) if call_id else root_logical
                    stream_times = [timestamp_ns(e.get("timestamp")) for e in stream]
                    agent_start, agent_end = bounds(stream_times, file_mtime_ns(path))
                    child, _, child_output = _claude_stream_observations(
                        stream,
                        parent=agent_logical,
                        stream_id=agent_logical,
                        result_map=result_map,
                        include_thinking=include_thinking,
                    )
                    first_user = next(
                        (_content_text((e.get("message") or {}).get("content"), "text") for e in stream if e.get("type") == "user"),
                        None,
                    )
                    subagent_observations.append(TraceObservation(
                        logical_id=agent_logical,
                        name=f"run-{stream[0].get('slug') or 'subagent'}",
                        as_type="agent",
                        start_ns=agent_start,
                        end_ns=agent_end,
                        parent_logical_id=parent,
                        input=first_user,
                        output=child_output,
                        metadata={
                            "agent_id": agent_id,
                            "agent_type": "subagent",
                            "parent_link_quality": "exact_agent_result" if call_id else "session_fallback",
                        },
                    ))
                    subagent_observations.extend(child)
                observations.extend(subagent_observations)
                all_times = [prompt_ts]
                for obs in observations:
                    all_times.extend((obs.start_ns, obs.end_ns))
                start, end = bounds(all_times, prompt_ts)
                root = TraceObservation(
                    logical_id=root_logical,
                    name="run-claude-code-turn",
                    as_type="agent",
                    start_ns=start,
                    end_ns=end,
                    input=prompt_text,
                    output=last_root_text,
                    metadata={
                        "source_session_id": session_id,
                        "turn_index": turn_index,
                        "git_branch": prompt_entry.get("gitBranch"),
                        "cli_version": prompt_entry.get("version"),
                        "cwd": prompt_entry.get("cwd"),
                    },
                )
                yield AgentTrace(
                    logical_id=f"{session_id}:turn:{turn_index}",
                    name="claude-code-turn",
                    source="claude",
                    source_session_id=session_id,
                    project=project_dir.name,
                    observations=[root, *observations],
                    metadata={"adapter": "claude-v2", "turn_index": turn_index},
                )


# ---------------------------------------------------------------------------
# Codex


def _codex_files() -> list[Path]:
    files = list(CODEX_SESSIONS_DIR.rglob("*.jsonl")) if CODEX_SESSIONS_DIR.exists() else []
    if CODEX_ARCHIVED_DIR.exists():
        files.extend(CODEX_ARCHIVED_DIR.glob("*.jsonl"))
    return sorted(files)


def _codex_meta(entries: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    for entry in entries:
        if entry.get("type") == "session_meta" and isinstance(entry.get("payload"), dict):
            payload = dict(entry["payload"])
            payload.setdefault("id", path.stem)
            return payload
    return {"id": path.stem}


_CODEX_KNOWN_TOP_LEVEL_TYPES = frozenset({
    "session_meta",
    "inter_agent_communication",
    "inter_agent_communication_metadata",
    "compacted",
    "turn_context",
    "response_item",
    "world_state",
    "event_msg",
})
_CODEX_KNOWN_EVENT_TYPES = frozenset({
    "error",
    "warning",
    "guardian_warning",
    "realtime_conversation_started",
    "realtime_conversation_realtime",
    "realtime_conversation_closed",
    "realtime_conversation_sdp",
    "model_reroute",
    "model_verification",
    "turn_moderation_metadata",
    "safety_buffering",
    "context_compacted",
    "thread_rolled_back",
    "task_started",
    "turn_started",
    "thread_settings_applied",
    "task_complete",
    "turn_complete",
    "token_count",
    "agent_message",
    "user_message",
    "agent_reasoning",
    "agent_reasoning_raw_content",
    "agent_reasoning_section_break",
    "session_configured",
    "environment_connected",
    "environment_disconnected",
    "thread_goal_updated",
    "mcp_startup_update",
    "mcp_startup_complete",
    "mcp_tool_call_begin",
    "mcp_tool_call_end",
    "web_search_begin",
    "web_search_end",
    "image_generation_begin",
    "image_generation_end",
    "exec_command_begin",
    "exec_command_output_delta",
    "terminal_interaction",
    "exec_command_end",
    "view_image_tool_call",
    "exec_approval_request",
    "request_permissions",
    "request_user_input",
    "dynamic_tool_call_request",
    "dynamic_tool_call_response",
    "elicitation_request",
    "apply_patch_approval_request",
    "guardian_assessment",
    "deprecation_notice",
    "stream_error",
    "patch_apply_begin",
    "patch_apply_updated",
    "patch_apply_end",
    "turn_diff",
    "realtime_conversation_list_voices_response",
    "plan_update",
    "turn_aborted",
    "shutdown_complete",
    "entered_review_mode",
    "exited_review_mode",
    "raw_response_item",
    "raw_response_completed",
    "item_started",
    "item_completed",
    "hook_started",
    "hook_completed",
    "agent_message_content_delta",
    "plan_delta",
    "reasoning_content_delta",
    "reasoning_raw_content_delta",
    "collab_agent_spawn_begin",
    "collab_agent_spawn_end",
    "collab_agent_interaction_begin",
    "collab_agent_interaction_end",
    "collab_waiting_begin",
    "collab_waiting_end",
    "collab_close_begin",
    "collab_close_end",
    "collab_resume_begin",
    "collab_resume_end",
    "sub_agent_activity",
})
_CODEX_KNOWN_RESPONSE_ITEM_TYPES = frozenset({
    "additional_tools",
    "message",
    "agent_message",
    "reasoning",
    "local_shell_call",
    "function_call",
    "tool_search_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "tool_search_output",
    "web_search_call",
    "image_generation_call",
    "compaction",
    "compaction_summary",
    "compaction_trigger",
    "context_compaction",
    "other",
})
_CODEX_KNOWN_MESSAGE_ROLES = frozenset({
    "user", "developer", "system", "assistant", "tool",
})
_CODEX_KNOWN_CONTENT_TYPES = frozenset({
    "input_text", "input_image", "input_audio", "output_text",
    "reasoning_text", "text", "summary_text", "encrypted_content",
})
_CODEX_SEMANTIC_EVENT_TYPES = frozenset({
    "task_started", "turn_started", "task_complete", "turn_complete",
    "turn_aborted", "user_message", "agent_message", "agent_reasoning",
    "token_count", "context_compacted", "sub_agent_activity",
    "thread_settings_applied", "web_search_end", "patch_apply_end",
    "mcp_tool_call_end",
})
_CODEX_SEMANTIC_RESPONSE_ITEM_TYPES = frozenset({
    "message", "agent_message", "reasoning", "function_call",
    "function_call_output", "custom_tool_call", "custom_tool_call_output",
    "tool_search_call", "tool_search_output",
})


def _codex_order(entry: dict[str, Any]) -> int:
    """Use Codex's official monotonic ordinal, then the physical source line."""
    ordinal = entry.get("ordinal")
    if isinstance(ordinal, int) and not isinstance(ordinal, bool):
        return ordinal
    return int(entry.get("_agentstracer_line") or 0)


def _codex_schema_profile(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe observed Codex schema enums and surface future drift.

    The source is intentionally treated as an open schema.  Known values get
    semantic mappings, while new values are listed explicitly and emitted as
    generic observations by :func:`iter_codex_traces` instead of disappearing.
    """
    top_level: dict[str, int] = defaultdict(int)
    event_types: dict[str, int] = defaultdict(int)
    response_types: dict[str, int] = defaultdict(int)
    message_roles: dict[str, int] = defaultdict(int)
    content_types: dict[str, int] = defaultdict(int)
    signatures: set[str] = set()
    for entry in entries:
        entry_type = str(entry.get("type") or "<missing>")
        top_level[entry_type] += 1
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        payload_type = str(payload.get("type") or "<none>")
        signatures.add(
            f"{entry_type}:{payload_type}:{','.join(sorted(str(key) for key in payload))}"
        )
        if entry_type == "event_msg":
            event_types[payload_type] += 1
        if entry_type == "response_item":
            response_types[payload_type] += 1
            if payload_type == "message":
                message_roles[str(payload.get("role") or "<missing>")] += 1
            content = payload.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        content_types[str(block.get("type") or "<missing>")] += 1
    unknown_top = sorted(set(top_level) - _CODEX_KNOWN_TOP_LEVEL_TYPES)
    unknown_events = sorted(set(event_types) - _CODEX_KNOWN_EVENT_TYPES)
    unknown_responses = sorted(
        set(response_types) - _CODEX_KNOWN_RESPONSE_ITEM_TYPES
    )
    # `ResponseItem::Message.role` is an official open `String`, not a closed
    # enum. Keep unusual roles visible without classifying them as schema drift.
    nonstandard_roles = sorted(set(message_roles) - _CODEX_KNOWN_MESSAGE_ROLES)
    unknown_content = sorted(set(content_types) - _CODEX_KNOWN_CONTENT_TYPES)
    ordinals = [
        int(entry["ordinal"])
        for entry in entries
        if isinstance(entry.get("ordinal"), int)
        and not isinstance(entry.get("ordinal"), bool)
    ]
    return {
        "contract": {
            "rollout": "codex_protocol::protocol::RolloutItem",
            "events": "codex_protocol::protocol::EventMsg",
            "responses": "codex_protocol::models::ResponseItem",
            "content": "codex_protocol::models::ContentItem and reasoning content",
            "message_role": "open_string",
        },
        "record_count": len(entries),
        "schema_signature_count": len(signatures),
        "ordering": {
            "basis": "ordinal_then_source_line",
            "ordinal_count": len(ordinals),
            "missing_ordinal_count": len(entries) - len(ordinals),
            "ordinal_min": min(ordinals) if ordinals else None,
            "ordinal_max": max(ordinals) if ordinals else None,
        },
        "top_level_types": dict(sorted(top_level.items())),
        "event_types": dict(sorted(event_types.items())),
        "response_item_types": dict(sorted(response_types.items())),
        "message_roles": dict(sorted(message_roles.items())),
        "nonstandard_message_roles": nonstandard_roles,
        "content_types": dict(sorted(content_types.items())),
        "unmapped": {
            "top_level_types": unknown_top,
            "event_types": unknown_events,
            "response_item_types": unknown_responses,
            "message_roles": [],
            "content_types": unknown_content,
        },
        "coverage_complete": not any((
            unknown_top, unknown_events, unknown_responses,
            unknown_content,
        )),
    }


def _codex_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep source fields while excluding opaque internal transport blobs."""
    excluded = {
        "type", "internal_chat_message_metadata_passthrough", "encrypted_content",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _codex_turns(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        event_type = payload.get("type") if entry.get("type") == "event_msg" else None
        turn_id = payload.get("turn_id")
        if event_type in ("task_started", "turn_started") and turn_id:
            payload_start = timestamp_ns(payload.get("started_at"))
            starts[str(turn_id)] = {
                "start": payload_start or timestamp_ns(entry.get("timestamp")),
                "start_line": _codex_order(entry),
                "payload": payload,
                "start_source": "payload.started_at" if payload_start else "event.timestamp",
            }
        elif event_type in (
            "task_complete", "turn_complete", "turn_aborted",
        ) and turn_id:
            start = starts.pop(str(turn_id), {})
            payload_end = timestamp_ns(payload.get("completed_at"))
            envelope_end = timestamp_ns(entry.get("timestamp"))
            start_ns = (
                start.get("start") or timestamp_ns(payload.get("started_at"))
                or payload_end or envelope_end
            )
            inferred_end = bool(
                not payload_end
                and start.get("start_source") == "payload.started_at"
            )
            end_ns = start_ns if inferred_end else (payload_end or envelope_end or start_ns)
            turns.append({
                "id": str(turn_id),
                "start": start_ns,
                "end": max(start_ns or 0, end_ns or start_ns or 0),
                "start_line": int(
                    start.get("start_line") or _codex_order(entry)
                ),
                "end_line": _codex_order(entry),
                "output": payload.get("last_agent_message"),
                "aborted": event_type == "turn_aborted",
                "payload": payload,
                "timestamp_quality": (
                    "payload_lifecycle" if payload_end
                    else "payload_start_next_turn_end" if inferred_end
                    else "event_envelope"
                ),
                "end_inferred_from_next_turn": inferred_end,
            })
    last_ts = max((timestamp_ns(e.get("timestamp")) or 0 for e in entries), default=0)
    last_line = max((_codex_order(entry) for entry in entries), default=0)
    for turn_id, start in starts.items():
        start_ns = start.get("start") or last_ts
        turns.append({
            "id": turn_id,
            "start": start_ns,
            "end": max(start_ns, last_ts),
            "start_line": int(start.get("start_line") or 0),
            "end_line": last_line,
            "output": None,
            "aborted": False,
            "open": True,
            "payload": start.get("payload", {}),
            "timestamp_quality": "open_event_envelope",
        })
    turns.sort(key=lambda turn: (turn.get("start_line", 0), turn["start"] or 0))
    for index, turn in enumerate(turns[:-1]):
        if not turn.pop("end_inferred_from_next_turn", False):
            continue
        next_start = turns[index + 1]["start"]
        if next_start >= turn["start"]:
            turn["end"] = next_start
    if turns:
        turns[-1].pop("end_inferred_from_next_turn", None)
    return turns


def _codex_container(
    ts: int,
    turns: list[dict[str, Any]],
    default: str,
    line: int | None = None,
) -> str:
    if line is not None:
        line_matches = [
            turn for turn in turns
            if int(turn.get("start_line") or 0) <= line <= int(turn.get("end_line") or 0)
        ]
        if line_matches:
            turn = max(line_matches, key=lambda item: int(item.get("start_line") or 0))
            return str(turn.get("logical_id") or f"turn:{turn['id']}")
    time_matches = [turn for turn in turns if turn["start"] <= ts <= turn["end"]]
    if time_matches:
        turn = max(time_matches, key=lambda item: item["start"] or 0)
        return str(turn.get("logical_id") or f"turn:{turn['id']}")
    return default


def _decode_codex_call_input(payload: dict[str, Any]) -> Any:
    value = payload.get("arguments", payload.get("input"))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_CODEX_TOOL_CALL_RE = re.compile(r"\btools\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CODEX_DECLARATION_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*="
)
_CODEX_ARRAY_MAP_RE = re.compile(
    r"\b(?P<collection>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*map\s*\(\s*"
    r"(?:async\s+)?(?:\(\s*)?\[\s*"
    r"(?P<items>[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*,\s*[A-Za-z_$][A-Za-z0-9_$]*)*)?\s*\]"
    r"(?:\s*,\s*(?P<index>[A-Za-z_$][A-Za-z0-9_$]*))?"
    r"\s*(?:\)\s*)?=>"
)
_CODEX_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_CODEX_NUMBER_RE = re.compile(
    r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|"
    r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _codex_visible_javascript(source: str) -> str:
    """Blank JavaScript strings/comments while retaining source positions."""
    visible = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in ("'", '"', "`"):
                state = "string"
                quote = char
                visible[index] = " "
            elif char == "/" and following == "/":
                state = "line_comment"
                visible[index] = visible[index + 1] = " "
                index += 1
            elif char == "/" and following == "*":
                state = "block_comment"
                visible[index] = visible[index + 1] = " "
                index += 1
        elif state == "string":
            visible[index] = " "
            if char == "\\":
                if index + 1 < len(source):
                    visible[index + 1] = " "
                    index += 1
            elif char == quote:
                state = "code"
        elif state == "line_comment":
            visible[index] = " "
            if char in ("\n", "\r"):
                state = "code"
        elif state == "block_comment":
            visible[index] = " "
            if char == "*" and following == "/":
                visible[index + 1] = " "
                index += 1
                state = "code"
        index += 1
    return "".join(visible)


def _codex_closing_parenthesis(source: str, open_paren: int) -> int | None:
    """Locate a JavaScript closing parenthesis without executing the source."""
    index = open_paren + 1
    depth = 1
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in ("'", '"', "`"):
                state = "string"
                quote = char
            elif char == "/" and following == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and following == "*":
                state = "block_comment"
                index += 1
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == quote:
                state = "code"
        elif state == "line_comment":
            if char in ("\n", "\r"):
                state = "code"
        elif state == "block_comment" and char == "*" and following == "/":
            index += 1
            state = "code"
        index += 1
    return None


def _codex_parenthesized_input(source: str, open_paren: int) -> str | None:
    """Return one call's argument expression without executing JavaScript."""
    closing = _codex_closing_parenthesis(source, open_paren)
    if closing is None:
        return None
    return source[open_paren + 1:closing]


def _codex_nested_tool_call_sites(source: Any) -> list[tuple[str, str, int]]:
    """Find real ``tools.foo(...)`` calls, arguments, and source positions."""
    if not isinstance(source, str) or "tools." not in source:
        return []
    visible = _codex_visible_javascript(source)
    calls: list[tuple[str, str, int]] = []
    for match in _CODEX_TOOL_CALL_RE.finditer(visible):
        expression = _codex_parenthesized_input(source, match.end() - 1)
        if expression is not None:
            calls.append((match.group(1), expression, match.start()))
    return calls


def _codex_nested_tool_calls(source: Any) -> list[tuple[str, str]]:
    """Find real ``tools.foo(...)`` calls and their argument expressions."""
    return [(name, expression) for name, expression, _ in _codex_nested_tool_call_sites(source)]


def _codex_nested_tool_names(source: Any) -> list[str]:
    """Return concrete tool identifiers in first-call order."""
    names = [name for name, _ in _codex_nested_tool_calls(source)]

    # Keep first-call order while removing duplicate invocations.  The raw
    # identifier remains in metadata; the display form is made readable below.
    return list(dict.fromkeys(names))


class _CodexJSLiteralParser:
    """Conservative parser for the JavaScript literal subset used in tool args."""

    def __init__(
        self,
        source: str,
        constants: dict[str, Any] | None = None,
        start: int = 0,
    ):
        self.source = source
        self.constants = constants or {}
        self.index = start

    def _skip_space(self) -> None:
        while self.index < len(self.source):
            if self.source[self.index].isspace():
                self.index += 1
                continue
            if self.source.startswith("//", self.index):
                newline = self.source.find("\n", self.index + 2)
                self.index = len(self.source) if newline < 0 else newline + 1
                continue
            if self.source.startswith("/*", self.index):
                end = self.source.find("*/", self.index + 2)
                if end < 0:
                    raise ValueError("unterminated JavaScript comment")
                self.index = end + 2
                continue
            break

    def parse_value(self) -> Any:
        self._skip_space()
        if self.index >= len(self.source):
            raise ValueError("missing JavaScript value")
        char = self.source[self.index]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char in ("'", '"', "`"):
            return self._parse_string()
        number = _CODEX_NUMBER_RE.match(self.source, self.index)
        if number:
            self.index = number.end()
            token = number.group(0)
            if token.lower().startswith(("0x", "+0x", "-0x")):
                return int(token, 16)
            if token.lower().startswith(("0b", "+0b", "-0b")):
                return int(token, 2)
            if token.lower().startswith(("0o", "+0o", "-0o")):
                return int(token, 8)
            return float(token) if any(mark in token for mark in ".eE") else int(token)
        identifier = _CODEX_IDENTIFIER_RE.match(self.source, self.index)
        if not identifier:
            raise ValueError("unsupported JavaScript expression")
        self.index = identifier.end()
        name = identifier.group(0)
        if name == "true":
            return True
        if name == "false":
            return False
        if name in ("null", "undefined"):
            return None
        if name in self.constants:
            return self.constants[name]
        raise ValueError(f"unresolved JavaScript identifier {name}")

    def parse_arguments(self) -> Any:
        self._skip_space()
        if self.index >= len(self.source):
            return {}
        values = [self.parse_value()]
        self._skip_space()
        while self.index < len(self.source):
            if self.source[self.index] != ",":
                raise ValueError("non-literal JavaScript argument")
            self.index += 1
            values.append(self.parse_value())
            self._skip_space()
        return values[0] if len(values) == 1 else {"args": values}

    def _parse_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self.index += 1
        while True:
            self._skip_space()
            if self.index >= len(self.source):
                raise ValueError("unterminated JavaScript object")
            if self.source[self.index] == "}":
                self.index += 1
                return result
            if self.source[self.index] in ("'", '"', "`"):
                key = str(self._parse_string())
            else:
                key_match = _CODEX_IDENTIFIER_RE.match(self.source, self.index)
                if not key_match:
                    raise ValueError("unsupported JavaScript object key")
                key = key_match.group(0)
                self.index = key_match.end()
            self._skip_space()
            if self.index >= len(self.source) or self.source[self.index] != ":":
                if key in self.constants:
                    result[key] = self.constants[key]
                else:
                    raise ValueError("unsupported JavaScript object shorthand")
            else:
                self.index += 1
                result[key] = self.parse_value()
            self._skip_space()
            if self.index < len(self.source) and self.source[self.index] == ",":
                self.index += 1
                continue
            if self.index < len(self.source) and self.source[self.index] == "}":
                self.index += 1
                return result
            raise ValueError("invalid JavaScript object separator")

    def _parse_array(self) -> list[Any]:
        result: list[Any] = []
        self.index += 1
        while True:
            self._skip_space()
            if self.index >= len(self.source):
                raise ValueError("unterminated JavaScript array")
            if self.source[self.index] == "]":
                self.index += 1
                return result
            result.append(self.parse_value())
            self._skip_space()
            if self.index < len(self.source) and self.source[self.index] == ",":
                self.index += 1
                continue
            if self.index < len(self.source) and self.source[self.index] == "]":
                self.index += 1
                return result
            raise ValueError("invalid JavaScript array separator")

    def _parse_string(self) -> str:
        quote = self.source[self.index]
        self.index += 1
        result: list[str] = []
        escapes = {
            "b": "\b", "f": "\f", "n": "\n", "r": "\r",
            "t": "\t", "v": "\v", "0": "\0",
        }
        while self.index < len(self.source):
            char = self.source[self.index]
            self.index += 1
            if char == quote:
                return "".join(result)
            if quote == "`" and char == "$" and self.source.startswith("{", self.index):
                raise ValueError("interpolated template literal")
            if char != "\\":
                result.append(char)
                continue
            if self.index >= len(self.source):
                raise ValueError("unterminated JavaScript escape")
            escaped = self.source[self.index]
            self.index += 1
            if escaped in ("\n", "\r"):
                if escaped == "\r" and self.source.startswith("\n", self.index):
                    self.index += 1
                continue
            if escaped == "x":
                digits = self.source[self.index:self.index + 2]
                if len(digits) != 2 or not all(c in "0123456789abcdefABCDEF" for c in digits):
                    raise ValueError("invalid JavaScript hex escape")
                result.append(chr(int(digits, 16)))
                self.index += 2
            elif escaped == "u":
                if self.source.startswith("{", self.index):
                    end = self.source.find("}", self.index + 1)
                    if end < 0:
                        raise ValueError("invalid JavaScript unicode escape")
                    digits = self.source[self.index + 1:end]
                    self.index = end + 1
                else:
                    digits = self.source[self.index:self.index + 4]
                    self.index += 4
                try:
                    result.append(chr(int(digits, 16)))
                except (ValueError, OverflowError) as exc:
                    raise ValueError("invalid JavaScript unicode escape") from exc
            else:
                result.append(escapes.get(escaped, escaped))
        raise ValueError("unterminated JavaScript string")


def _codex_js_constants(source: str) -> dict[str, Any]:
    """Resolve literal ``const name = value`` declarations without eval."""
    constants: dict[str, Any] = {}
    visible = _codex_visible_javascript(source)
    for match in _CODEX_DECLARATION_RE.finditer(visible):
        parser = _CodexJSLiteralParser(source, constants, match.end())
        try:
            constants[match.group(1)] = parser.parse_value()
        except ValueError:
            continue
    return constants


def _codex_static_map_bindings(
    source: str,
    constants: dict[str, Any],
    call_position: int,
) -> tuple[list[dict[str, Any]], str] | None:
    """Expand literal ``array.map(([a, b]) => tools.foo({a, b}))`` inputs.

    Codex frequently batches tool calls through ``Promise.all``.  The tool
    arguments then contain callback-local identifiers even though every value
    is statically present in a literal array.  This resolves only that narrow,
    deterministic subset; arbitrary JavaScript is never evaluated.
    """
    visible = _codex_visible_javascript(source)
    candidates: list[tuple[int, list[dict[str, Any]], str]] = []
    for match in _CODEX_ARRAY_MAP_RE.finditer(visible):
        open_paren = visible.find("(", match.start(), match.end())
        if open_paren < 0:
            continue
        close_paren = _codex_closing_parenthesis(source, open_paren)
        if close_paren is None or not (match.end() <= call_position < close_paren):
            continue
        collection_name = match.group("collection")
        rows = constants.get(collection_name)
        if not isinstance(rows, list):
            continue
        item_names = [
            item.strip() for item in (match.group("items") or "").split(",")
            if item.strip()
        ]
        index_name = match.group("index")
        bindings: list[dict[str, Any]] = []
        valid = True
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) < len(item_names):
                valid = False
                break
            binding = {name: row[position] for position, name in enumerate(item_names)}
            if index_name:
                binding[index_name] = index
            bindings.append(binding)
        if valid and bindings:
            candidates.append((open_paren, bindings, collection_name))
    if not candidates:
        return None
    _, bindings, collection_name = max(candidates, key=lambda item: item[0])
    return bindings, collection_name


def _codex_wrap_tool_argument(tool_name: str, value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if tool_name == "apply_patch":
        return {"patch": value}
    if tool_name == "functions.exec":
        return {"script": value}
    return {"value": value}


def _codex_tool_input(
    payload: dict[str, Any],
    tool_name: str,
    metadata: dict[str, Any],
) -> Any:
    """Turn Codex tool input into JSON-compatible arguments, never eval code."""
    raw_input = payload.get("input", payload.get("arguments"))
    decoded = _decode_codex_call_input(payload)
    outer_name = str(payload.get("name") or "")
    calls = _codex_nested_tool_call_sites(raw_input) if outer_name == "exec" else []
    if calls and isinstance(raw_input, str):
        constants = _codex_js_constants(raw_input)
        structured_calls: list[dict[str, Any]] = []
        fallback_count = 0
        expanded_iterations = 0
        expanded_collections: list[str] = []
        for identifier, expression, call_position in calls:
            display_name = _codex_display_tool_name(identifier)
            static_map = _codex_static_map_bindings(
                raw_input, constants, call_position,
            )
            environments = [constants]
            if static_map is not None:
                bindings, collection_name = static_map
                environments = [{**constants, **binding} for binding in bindings]
                expanded_iterations += len(bindings)
                expanded_collections.append(collection_name)
            for environment in environments:
                try:
                    value = _CodexJSLiteralParser(
                        expression, environment,
                    ).parse_arguments()
                    arguments = _codex_wrap_tool_argument(display_name, value)
                except ValueError:
                    arguments = {"arguments_code": expression.strip()}
                    fallback_count += 1
                structured_calls.append({"tool": display_name, "arguments": arguments})
        metadata["tool_input_source_call_count"] = len(calls)
        metadata["tool_input_call_count"] = len(structured_calls)
        if expanded_iterations:
            metadata["static_map_iterations"] = expanded_iterations
            metadata["static_map_collections"] = list(dict.fromkeys(expanded_collections))
        if fallback_count == len(structured_calls):
            metadata["tool_input_quality"] = "raw_expression_fallback"
        elif fallback_count:
            metadata["tool_input_quality"] = (
                "partial_static_map" if expanded_iterations else "partial_js_literal"
            )
        else:
            metadata["tool_input_quality"] = (
                "expanded_static_map" if expanded_iterations else "extracted_js_literal"
            )
        if fallback_count:
            metadata["unparsed_tool_inputs"] = fallback_count
        if len(structured_calls) == 1:
            return structured_calls[0]["arguments"]
        return {"calls": structured_calls}
    if isinstance(decoded, str):
        metadata["tool_input_quality"] = "raw_text_wrapped"
        return _codex_wrap_tool_argument(tool_name, decoded)
    if decoded is None:
        metadata["tool_input_quality"] = "missing"
        return {}
    metadata["tool_input_quality"] = "exact_json"
    return decoded


def _codex_display_tool_name(identifier: str) -> str:
    """Render namespace-encoded tool identifiers like ``web__run``."""
    return identifier.replace("__", ".")


def _codex_tool_descriptor(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve Codex's transport-level tool to its concrete dispatched tool."""
    outer_name = str(payload.get("name") or "")
    namespace = str(payload.get("namespace") or "")
    raw_input = payload.get("input", payload.get("arguments"))
    decoded_input = _decode_codex_call_input(payload)
    nested = _codex_nested_tool_names(raw_input)

    quality = "exact"
    concrete: list[str] = []
    if outer_name == "exec" and nested:
        concrete = nested
        quality = "extracted_from_exec_wrapper"
    elif outer_name == "exec" and isinstance(decoded_input, dict) and (
        "cmd" in decoded_input or "command" in decoded_input
    ):
        # Older Codex sessions called the shell transport simply ``exec``.
        concrete = ["exec_command"]
        quality = "inferred_from_command_arguments"
    elif outer_name == "exec":
        concrete = ["functions.exec"]
        quality = "transport_only"
    elif outer_name:
        identifier = f"{namespace}.{outer_name}" if namespace else outer_name
        concrete = [identifier]
    elif payload.get("type") == "tool_search_call":
        concrete = ["tool_search"]
        quality = "exact_event_type"
    else:
        concrete = ["tool_call"]
        quality = "fallback"

    display_names = [_codex_display_tool_name(name) for name in concrete]
    if len(display_names) == 1:
        display_name = display_names[0]
    else:
        display_name = " + ".join(display_names)

    metadata: dict[str, Any] = {
        "tool_name": display_name,
        "tool_name_quality": quality,
        "transport_tool": outer_name or payload.get("type"),
    }
    if namespace:
        metadata["tool_namespace"] = namespace
    if nested:
        metadata["nested_tools"] = [
            _codex_display_tool_name(name) for name in nested
        ]
        metadata["nested_tool_identifiers"] = nested
    return display_name, metadata


def _codex_tool_failed(status: Any, output: Any) -> bool:
    if str(status or "").lower() in {"failed", "error", "aborted"}:
        return True
    if not isinstance(output, dict):
        return False
    if output.get("isError") is True or output.get("is_error") is True:
        return True
    exit_code = output.get("exit_code")
    return isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0


def _codex_tool_error_message(status: Any, output: Any) -> str:
    normalized_status = str(status or "").lower()
    if normalized_status in {"failed", "error", "aborted"}:
        return str(status)
    if isinstance(output, dict):
        exit_code = output.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
            return f"exit_code={exit_code}"
    return "tool returned an error"


CODEX_ROLLOUT_TRACE_ROOT_ENV = "CODEX_ROLLOUT_TRACE_ROOT"
CODEX_ROLLOUT_TRACE_DEFAULT_DIR = Path.home() / ".codex" / "rollout-traces"

_CODEX_ROLLOUT_RAW_EVENT_TYPES = frozenset({
    "rollout_started", "rollout_ended", "thread_started", "thread_ended",
    "codex_turn_started", "codex_turn_ended", "inference_started",
    "inference_completed", "inference_failed", "inference_cancelled",
    "tool_call_started", "mcp_tool_call_correlation_assigned",
    "tool_call_runtime_started", "tool_call_runtime_ended", "tool_call_ended",
    "code_cell_started", "code_cell_initial_response", "code_cell_ended",
    "compaction_request_started", "compaction_request_completed",
    "compaction_request_failed", "compaction_installed", "agent_result_observed",
    "protocol_event_observed", "other",
})


def _codex_rollout_trace_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get(CODEX_ROLLOUT_TRACE_ROOT_ENV)
    if configured:
        roots.append(Path(configured).expanduser())
    if CODEX_ROLLOUT_TRACE_DEFAULT_DIR.exists():
        roots.append(CODEX_ROLLOUT_TRACE_DEFAULT_DIR)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _codex_rollout_trace_bundles() -> list[Path]:
    bundles: list[Path] = []
    for root in _codex_rollout_trace_roots():
        if (root / "manifest.json").is_file():
            bundles.append(root)
        elif root.is_dir():
            bundles.extend(path.parent for path in root.rglob("manifest.json"))
    return sorted(set(bundles))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _codex_rollout_payload(
    bundle: Path,
    reference: Any,
) -> tuple[Any, dict[str, Any]]:
    """Read one official bundle payload without allowing path traversal."""
    if not isinstance(reference, dict):
        return None, {"payload_quality": "missing_reference"}
    raw_path = reference.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None, {
            "payload_quality": "invalid_reference",
            "raw_payload_id": reference.get("raw_payload_id"),
        }
    relative = Path(raw_path)
    bundle_root = bundle.resolve(strict=False)
    candidate = (bundle / relative).resolve(strict=False)
    if relative.is_absolute() or (
        candidate != bundle_root and bundle_root not in candidate.parents
    ):
        return None, {
            "payload_quality": "unsafe_reference",
            "raw_payload_id": reference.get("raw_payload_id"),
        }
    try:
        value = json.loads(candidate.read_text(errors="replace"))
    except OSError:
        return None, {
            "payload_quality": "missing_payload_file",
            "raw_payload_id": reference.get("raw_payload_id"),
        }
    except (json.JSONDecodeError, TypeError):
        return None, {
            "payload_quality": "invalid_payload_json",
            "raw_payload_id": reference.get("raw_payload_id"),
        }
    return value, {
        "payload_quality": "exact_official_payload",
        "raw_payload_id": reference.get("raw_payload_id"),
        "raw_payload_kind": reference.get("kind"),
    }


def _codex_rollout_status(status: Any) -> tuple[str, str | None]:
    normalized = str(status or "running").lower()
    if normalized in {"failed", "error"}:
        return "ERROR", normalized
    if normalized in {"cancelled", "aborted", "terminated", "running"}:
        return "WARNING", normalized
    return "DEFAULT", None


def _codex_official_item_text(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return item
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        value = block.get("text")
        if isinstance(value, str) and value:
            text_parts.append(value)
    return "\n".join(text_parts) if text_parts else content


def _codex_official_turn_trigger(
    request: Any,
    turn_id: str,
    *,
    allow_user_fallback: bool,
) -> dict[str, Any] | None:
    if not isinstance(request, dict) or not isinstance(request.get("input"), list):
        return None
    user_exact: list[dict[str, Any]] = []
    user_fallback: list[dict[str, Any]] = []
    agent_exact: list[dict[str, Any]] = []
    agent_fallback: list[dict[str, Any]] = []
    for item in request["input"]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("internal_chat_message_metadata_passthrough")
        item_turn = metadata.get("turn_id") if isinstance(metadata, dict) else None
        if item.get("type") == "message" and item.get("role") == "user":
            trigger = {
                "role": "user",
                "content": _codex_official_item_text(item),
                "source": "official_inference_request.message.user",
            }
            user_fallback.append(trigger)
            if item_turn == turn_id:
                user_exact.append(trigger)
        elif item.get("type") == "agent_message":
            trigger = {
                "role": "agent",
                "content": _codex_official_item_text(item),
                "author": item.get("author"),
                "recipient": item.get("recipient"),
                "source": "official_inference_request.agent_message",
            }
            agent_fallback.append(trigger)
            if item_turn == turn_id:
                agent_exact.append(trigger)
    if user_exact:
        return user_exact[-1]
    if agent_exact:
        return agent_exact[-1]
    if agent_fallback and not allow_user_fallback:
        return agent_fallback[-1]
    if allow_user_fallback and user_fallback:
        return user_fallback[-1]
    return None


def _codex_official_response_text(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    parts: list[str] = []
    items = response.get("output_items")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            text = _codex_official_item_text(item)
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n\n".join(parts) or None


def _codex_official_tool_name(
    kind: Any,
    invocation: Any,
) -> tuple[str, dict[str, Any]]:
    kind_dict = kind if isinstance(kind, dict) else {}
    kind_type = str(kind_dict.get("type") or kind or "other")
    invocation_dict = invocation if isinstance(invocation, dict) else {}
    raw_name = invocation_dict.get("tool_name")
    namespace = invocation_dict.get("tool_namespace")
    if isinstance(raw_name, str) and raw_name:
        name = f"{namespace}.{raw_name}" if namespace else raw_name
        quality = "exact_official_dispatch"
    elif kind_type == "mcp":
        server = kind_dict.get("server")
        tool = kind_dict.get("tool")
        name = ".".join(str(value) for value in (server, tool) if value) or "mcp"
        quality = "exact_official_kind"
    elif kind_type == "other" and kind_dict.get("name"):
        name = str(kind_dict["name"])
        quality = "exact_official_kind"
    else:
        name = kind_type
        quality = "exact_official_kind"
    return name, {
        "official_tool_kind": kind,
        "tool_namespace": namespace,
        "transport_tool": None,
        "tool_name_quality": quality,
    }


def _codex_official_tool_input(invocation: Any) -> tuple[Any, str]:
    if not isinstance(invocation, dict):
        return {}, "missing_official_invocation"
    payload = invocation.get("payload")
    if not isinstance(payload, dict):
        return {}, "missing_official_dispatch_payload"
    payload_type = payload.get("type")
    if payload_type == "function":
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                return {"arguments_text": arguments}, "unparseable_official_json_string"
            if isinstance(decoded, dict):
                return decoded, "parsed_official_json_string"
            return {"arguments": decoded}, "parsed_official_json_scalar"
        if isinstance(arguments, dict):
            return arguments, "exact_official_object"
        return {"arguments": arguments}, "exact_official_scalar"
    if payload_type == "tool_search":
        arguments = payload.get("arguments")
        return (
            arguments if isinstance(arguments, dict) else {"arguments": arguments},
            "exact_official_tool_search",
        )
    if payload_type == "custom":
        value = payload.get("input")
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return {"input": value}, "exact_official_custom_text"
            if isinstance(decoded, dict):
                return decoded, "parsed_official_custom_json"
            return {"input": decoded}, "parsed_official_custom_scalar"
        return {"input": value}, "exact_official_custom_value"
    if payload_type == "local_shell":
        return {
            key: value for key, value in payload.items()
            if key != "type" and value is not None
        }, "exact_official_local_shell"
    return {
        key: value for key, value in payload.items() if key != "type"
    }, "exact_official_unknown_payload"


def _codex_official_tool_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    result_type = result.get("type")
    if result_type == "code_mode_response":
        return result.get("value")
    if result_type == "direct_response":
        response_item = result.get("response_item")
        if isinstance(response_item, dict) and set(response_item) >= {"type", "output"}:
            return _tool_output(response_item.get("output"))
        return response_item
    if result_type == "error":
        return {"error": result.get("error")}
    return result


def _codex_spawn_task_name(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("task_name", "task", "name"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _iter_codex_rollout_trace_traces(
    include_thinking: bool = True,
) -> Iterator[AgentTrace]:
    del include_thinking  # The official payload already states what is readable.
    for bundle in _codex_rollout_trace_bundles():
        manifest = _read_json_object(bundle / "manifest.json")
        if not manifest:
            continue
        event_log_name = manifest.get("raw_event_log") or "trace.jsonl"
        event_log = bundle / str(event_log_name)
        events = _jsonl(event_log)
        if not events:
            continue
        events.sort(key=lambda entry: (
            int(entry.get("seq")) if isinstance(entry.get("seq"), int) else 2**63,
            int(entry.get("_agentstracer_line") or 0),
        ))
        rollout_id = str(manifest.get("rollout_id") or events[0].get("rollout_id") or bundle.name)
        official_trace_id = str(manifest.get("trace_id") or bundle.name)
        root_thread_id = str(manifest.get("root_thread_id") or "root")
        manifest_start = timestamp_ns(manifest.get("started_at_unix_ms"))
        event_times = [timestamp_ns(event.get("wall_time_unix_ms")) for event in events]
        trace_start, trace_end = bounds(event_times, manifest_start or file_mtime_ns(event_log))
        observations: list[TraceObservation] = []
        by_id: dict[str, TraceObservation] = {}
        threads: dict[str, TraceObservation] = {}
        thread_metadata: dict[str, dict[str, Any]] = {}
        turns: dict[str, TraceObservation] = {}
        inferences: dict[str, TraceObservation] = {}
        tools_by_id: dict[str, TraceObservation] = {}
        code_cells: dict[str, TraceObservation] = {}
        model_call_generations: dict[str, str] = {}
        seen_user_turns: set[str] = set()
        root_input: Any = None
        root_output: Any = None
        project = "codex"
        raw_type_counts: dict[str, int] = defaultdict(int)
        unknown_raw_types: set[str] = set()
        payload_issue_count = 0
        rollout_status = "running"

        def add(observation: TraceObservation) -> TraceObservation:
            observations.append(observation)
            by_id[observation.logical_id] = observation
            return observation

        def event_ns(event: dict[str, Any]) -> int:
            return int(timestamp_ns(event.get("wall_time_unix_ms")) or trace_start)

        def event_seq(event: dict[str, Any]) -> int:
            return int(event.get("seq") or 0)

        def thread_logical(thread_id: str) -> str:
            return f"thread:{thread_id}"

        def turn_logical(turn_id: str) -> str:
            return f"turn:{turn_id}"

        def parent_for(event: dict[str, Any]) -> str:
            turn_id = event.get("codex_turn_id")
            if turn_id and str(turn_id) in turns:
                return turn_logical(str(turn_id))
            thread_id = event.get("thread_id")
            if thread_id and str(thread_id) in threads:
                return thread_logical(str(thread_id))
            return thread_logical(root_thread_id)

        for event in events:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            payload_type = str(payload.get("type") or "<missing>")
            raw_type_counts[payload_type] += 1
            if payload_type not in _CODEX_ROLLOUT_RAW_EVENT_TYPES:
                unknown_raw_types.add(payload_type)
            ts = event_ns(event)
            seq = event_seq(event)
            common_metadata = {
                "official_seq": seq,
                "official_schema_version": event.get("schema_version"),
                "timestamp_quality": "exact_official_wall_time_with_seq_order",
            }

            if payload_type == "rollout_ended":
                rollout_status = str(payload.get("status") or "completed")
                continue

            if payload_type == "thread_started":
                thread_id = str(payload.get("thread_id") or event.get("thread_id") or root_thread_id)
                metadata, payload_meta = _codex_rollout_payload(
                    bundle, payload.get("metadata_payload"),
                )
                if payload_meta["payload_quality"] not in {
                    "exact_official_payload", "missing_reference",
                }:
                    payload_issue_count += 1
                metadata = metadata if isinstance(metadata, dict) else {}
                thread_metadata[thread_id] = metadata
                spawn = (
                    metadata.get("session_source", {}).get("subagent", {}).get("thread_spawn", {})
                    if isinstance(metadata.get("session_source"), dict) else {}
                )
                parent_thread_id = spawn.get("parent_thread_id") if isinstance(spawn, dict) else None
                parent = (
                    thread_logical(str(parent_thread_id))
                    if parent_thread_id else None if thread_id == root_thread_id
                    else thread_logical(root_thread_id)
                )
                observation = add(TraceObservation(
                    logical_id=thread_logical(thread_id),
                    name="run-codex-agent" if thread_id == root_thread_id else "run-codex-subagent",
                    as_type="agent",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent,
                    model=metadata.get("model"),
                    metadata={
                        **common_metadata,
                        "thread_id": thread_id,
                        "agent_path": payload.get("agent_path") or metadata.get("agent_path"),
                        "parent_thread_id": parent_thread_id,
                        "agent_nickname": metadata.get("nickname"),
                        "agent_role": metadata.get("agent_role"),
                        "task_name": metadata.get("task_name"),
                        "session_source": metadata.get("session_source"),
                        "cwd": metadata.get("cwd"),
                        "provider_name": metadata.get("provider_name"),
                        **payload_meta,
                    },
                ))
                threads[thread_id] = observation
                if thread_id == root_thread_id and metadata.get("cwd"):
                    project = str(metadata["cwd"])
                continue

            if payload_type == "thread_ended":
                thread_id = str(payload.get("thread_id") or event.get("thread_id") or "")
                observation = threads.get(thread_id)
                if observation:
                    observation.end_ns = max(observation.start_ns, ts)
                    level, message = _codex_rollout_status(payload.get("status"))
                    observation.level = level
                    observation.status_message = message
                    observation.metadata["official_end_seq"] = seq
                    observation.metadata["official_status"] = payload.get("status")
                continue

            if payload_type == "codex_turn_started":
                turn_id = str(payload.get("codex_turn_id") or event.get("codex_turn_id") or f"seq-{seq}")
                thread_id = str(payload.get("thread_id") or event.get("thread_id") or root_thread_id)
                observation = add(TraceObservation(
                    logical_id=turn_logical(turn_id),
                    name="run-agent-turn",
                    as_type="agent",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=thread_logical(thread_id),
                    metadata={
                        **common_metadata,
                        "turn_id": turn_id,
                        "thread_id": thread_id,
                        "user_input_quality": "awaiting_official_inference_request",
                    },
                ))
                turns[turn_id] = observation
                continue

            if payload_type == "codex_turn_ended":
                turn_id = str(payload.get("codex_turn_id") or event.get("codex_turn_id") or "")
                observation = turns.get(turn_id)
                if observation:
                    observation.end_ns = max(observation.start_ns, ts)
                    level, message = _codex_rollout_status(payload.get("status"))
                    observation.level = level
                    observation.status_message = message
                    observation.metadata["official_end_seq"] = seq
                    observation.metadata["official_status"] = payload.get("status")
                continue

            if payload_type == "inference_started":
                inference_id = str(payload.get("inference_call_id") or f"seq-{seq}")
                thread_id = str(payload.get("thread_id") or event.get("thread_id") or root_thread_id)
                turn_id = str(payload.get("codex_turn_id") or event.get("codex_turn_id") or "")
                request, payload_meta = _codex_rollout_payload(bundle, payload.get("request_payload"))
                if payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                observation = add(TraceObservation(
                    logical_id=f"inference:{inference_id}",
                    name="generate-codex-step",
                    as_type="generation",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=(
                        turn_logical(turn_id) if turn_id in turns else thread_logical(thread_id)
                    ),
                    input=request if request is not None else {
                        "available": False, "reason": payload_meta["payload_quality"],
                    },
                    output={"available": False, "reason": "inference_still_running"},
                    model=payload.get("model"),
                    metadata={
                        **common_metadata,
                        **payload_meta,
                        "inference_call_id": inference_id,
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "provider_name": payload.get("provider_name"),
                        "input_quality": "exact_official_inference_request",
                    },
                ))
                inferences[inference_id] = observation
                trigger = _codex_official_turn_trigger(
                    request, turn_id, allow_user_fallback=thread_id == root_thread_id,
                )
                if trigger:
                    trigger_public = {
                        key: value for key, value in trigger.items() if key != "source"
                    }
                    turn = turns.get(turn_id)
                    if turn and turn.input is None:
                        turn.input = trigger_public
                        turn.metadata["user_input_quality"] = trigger["source"]
                    thread = threads.get(thread_id)
                    if thread and thread.input is None:
                        thread.input = trigger_public
                    if trigger["role"] == "user":
                        fingerprint = canonical_json((thread_id, turn_id, trigger_public))
                        if fingerprint not in seen_user_turns:
                            seen_user_turns.add(fingerprint)
                            add(TraceObservation(
                                logical_id=f"user-turn:{thread_id}:{turn_id}:{seq}",
                                name="user-turn",
                                as_type="event",
                                start_ns=ts,
                                end_ns=ts,
                                parent_logical_id=(
                                    turn_logical(turn_id) if turn_id in turns
                                    else thread_logical(thread_id)
                                ),
                                input=trigger_public,
                                metadata={
                                    **common_metadata,
                                    "role": "user",
                                    "message_source": trigger["source"],
                                    "turn_id": turn_id,
                                },
                            ))
                        if thread_id == root_thread_id and root_input is None:
                            root_input = trigger_public
                continue

            if payload_type in {
                "inference_completed", "inference_failed", "inference_cancelled",
            }:
                inference_id = str(payload.get("inference_call_id") or "")
                observation = inferences.get(inference_id)
                reference = payload.get("response_payload") or payload.get("partial_response_payload")
                response, payload_meta = _codex_rollout_payload(bundle, reference)
                if reference is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                if observation:
                    observation.end_ns = max(observation.start_ns, ts)
                    observation.output = response if response is not None else {
                        "available": False,
                        "reason": payload.get("error") or payload.get("reason") or payload_meta["payload_quality"],
                    }
                    observation.metadata.update({
                        "official_end_seq": seq,
                        "response_id": payload.get("response_id"),
                        "upstream_request_id": payload.get("upstream_request_id"),
                        "output_quality": (
                            "exact_official_inference_response"
                            if response is not None else "explicit_missing_response"
                        ),
                    })
                    if isinstance(response, dict):
                        observation.usage = _usage(response.get("token_usage"))
                        output_items = response.get("output_items")
                        if isinstance(output_items, list):
                            for item in output_items:
                                if isinstance(item, dict) and item.get("call_id"):
                                    model_call_generations[str(item["call_id"])] = observation.logical_id
                    if payload_type != "inference_completed":
                        observation.level = "ERROR" if payload_type == "inference_failed" else "WARNING"
                        observation.status_message = str(payload.get("error") or payload.get("reason") or payload_type)
                    text_output = _codex_official_response_text(response)
                    if text_output:
                        root_output = text_output if observation.metadata.get("thread_id") == root_thread_id else root_output
                        thread = threads.get(str(observation.metadata.get("thread_id")))
                        if thread:
                            thread.output = text_output
                continue

            if payload_type == "code_cell_started":
                runtime_cell_id = str(payload.get("runtime_cell_id") or f"seq-{seq}")
                model_call_id = str(payload.get("model_visible_call_id") or "")
                observation = add(TraceObservation(
                    logical_id=f"code-cell:{runtime_cell_id}",
                    name="functions.exec",
                    as_type="chain",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent_for(event),
                    input={"script": payload.get("source_js")},
                    output={"available": False, "reason": "code_cell_still_running"},
                    links=(
                        [TraceLink(model_call_generations[model_call_id], {
                            "relationship": "model_visible_exec_call",
                        })]
                        if model_call_id in model_call_generations else []
                    ),
                    metadata={
                        **common_metadata,
                        "runtime_cell_id": runtime_cell_id,
                        "thread_id": event.get("thread_id"),
                        "turn_id": event.get("codex_turn_id"),
                        "model_visible_call_id": model_call_id or None,
                        "official_runtime_object": "code_cell",
                    },
                ))
                code_cells[runtime_cell_id] = observation
                continue

            if payload_type in {"code_cell_initial_response", "code_cell_ended"}:
                runtime_cell_id = str(payload.get("runtime_cell_id") or "")
                observation = code_cells.get(runtime_cell_id)
                response, payload_meta = _codex_rollout_payload(bundle, payload.get("response_payload"))
                if payload.get("response_payload") is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                if observation:
                    observation.output = response if response is not None else observation.output
                    observation.metadata[
                        "official_initial_response_seq" if payload_type == "code_cell_initial_response" else "official_end_seq"
                    ] = seq
                    observation.metadata["official_status"] = payload.get("status")
                    if payload_type == "code_cell_ended":
                        observation.end_ns = max(observation.start_ns, ts)
                        level, message = _codex_rollout_status(payload.get("status"))
                        observation.level = level
                        observation.status_message = message
                    elif str(payload.get("status") or "").lower() == "yielded":
                        observation.metadata["yielded_at_ns"] = ts
                continue

            if payload_type == "tool_call_started":
                tool_id = str(payload.get("tool_call_id") or f"seq-{seq}")
                invocation, payload_meta = _codex_rollout_payload(bundle, payload.get("invocation_payload"))
                if payload.get("invocation_payload") is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                tool_name, descriptor = _codex_official_tool_name(payload.get("kind"), invocation)
                tool_input, input_quality = _codex_official_tool_input(invocation)
                requester = payload.get("requester") if isinstance(payload.get("requester"), dict) else {}
                runtime_cell_id = requester.get("runtime_cell_id") if requester.get("type") == "code_cell" else None
                parent = (
                    code_cells[str(runtime_cell_id)].logical_id
                    if runtime_cell_id is not None and str(runtime_cell_id) in code_cells
                    else parent_for(event)
                )
                model_call_id = str(payload.get("model_visible_call_id") or "")
                links = []
                if model_call_id in model_call_generations:
                    links.append(TraceLink(model_call_generations[model_call_id], {
                        "relationship": "generated_tool_call",
                        "model_visible_call_id": model_call_id,
                    }))
                observation = add(TraceObservation(
                    logical_id=f"tool:{tool_id}",
                    name=tool_name,
                    as_type="tool",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent,
                    input=tool_input,
                    output={"available": False, "reason": "tool_still_running"},
                    links=links,
                    metadata={
                        **common_metadata,
                        **payload_meta,
                        **descriptor,
                        "tool_call_id": tool_id,
                        "thread_id": event.get("thread_id"),
                        "turn_id": event.get("codex_turn_id"),
                        "model_visible_call_id": model_call_id or None,
                        "code_mode_runtime_tool_id": payload.get("code_mode_runtime_tool_id"),
                        "requester": payload.get("requester"),
                        "summary": payload.get("summary"),
                        "tool_input_quality": input_quality,
                        "official_runtime_object": "tool_call",
                    },
                ))
                tools_by_id[tool_id] = observation
                continue

            if payload_type == "mcp_tool_call_correlation_assigned":
                tool = tools_by_id.get(str(payload.get("tool_call_id") or ""))
                if tool:
                    tool.metadata["mcp_call_id"] = payload.get("mcp_call_id")
                    tool.metadata["mcp_correlation_seq"] = seq
                continue

            if payload_type in {"tool_call_runtime_started", "tool_call_runtime_ended"}:
                tool = tools_by_id.get(str(payload.get("tool_call_id") or ""))
                runtime, payload_meta = _codex_rollout_payload(bundle, payload.get("runtime_payload"))
                if payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                if tool:
                    tool.metadata.setdefault("runtime_events", []).append({
                        "type": payload_type,
                        "seq": seq,
                        "wall_time_unix_ms": event.get("wall_time_unix_ms"),
                        "status": payload.get("status"),
                        "payload": runtime,
                        **payload_meta,
                    })
                continue

            if payload_type == "tool_call_ended":
                tool = tools_by_id.get(str(payload.get("tool_call_id") or ""))
                result, payload_meta = _codex_rollout_payload(bundle, payload.get("result_payload"))
                if payload.get("result_payload") is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                if tool:
                    tool.end_ns = max(tool.start_ns, ts)
                    tool.output = _codex_official_tool_output(result) if result is not None else {
                        "available": False, "reason": payload_meta["payload_quality"],
                    }
                    level, message = _codex_rollout_status(payload.get("status"))
                    tool.level = level
                    tool.status_message = message
                    tool.metadata.update({
                        "official_end_seq": seq,
                        "official_status": payload.get("status"),
                        "duration_ms": max(0, (tool.end_ns - tool.start_ns) / 1_000_000),
                        "output_quality": (
                            "exact_official_tool_result" if result is not None
                            else "explicit_missing_tool_result"
                        ),
                    })
                continue

            if payload_type == "agent_result_observed":
                carried, payload_meta = _codex_rollout_payload(bundle, payload.get("carried_payload"))
                if payload.get("carried_payload") is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                child_thread_id = str(payload.get("child_thread_id") or "")
                parent_thread_id = str(payload.get("parent_thread_id") or root_thread_id)
                links = []
                if child_thread_id in threads:
                    links.append(TraceLink(thread_logical(child_thread_id), {
                        "relationship": "agent_result_source",
                    }))
                child_turn_id = str(payload.get("child_codex_turn_id") or "")
                if child_turn_id in turns:
                    links.append(TraceLink(turn_logical(child_turn_id), {
                        "relationship": "agent_result_turn",
                    }))
                add(TraceObservation(
                    logical_id=f"agent-result:{payload.get('edge_id') or seq}",
                    name="agent-result-observed",
                    as_type="event",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=thread_logical(parent_thread_id),
                    input={
                        "child_thread_id": child_thread_id,
                        "message": payload.get("message"),
                        "carried_payload": carried,
                    },
                    links=links,
                    metadata={
                        **common_metadata,
                        **payload_meta,
                        "edge_id": payload.get("edge_id"),
                        "official_interaction_edge": "agent_result",
                    },
                ))
                continue

            if payload_type == "protocol_event_observed":
                value, payload_meta = _codex_rollout_payload(bundle, payload.get("event_payload"))
                if payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                add(TraceObservation(
                    logical_id=f"protocol-event:{seq}",
                    name=f"codex-protocol:{payload.get('event_type') or 'unknown'}",
                    as_type="event",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent_for(event),
                    input=value,
                    level="WARNING" if str(payload.get("event_type")) in {"error", "warning"} else "DEFAULT",
                    metadata={**common_metadata, **payload_meta},
                ))
                continue

            if payload_type.startswith("compaction_"):
                reference = payload.get("request_payload") or payload.get("response_payload") or payload.get("checkpoint_payload")
                value, payload_meta = _codex_rollout_payload(bundle, reference)
                if reference is not None and payload_meta["payload_quality"] != "exact_official_payload":
                    payload_issue_count += 1
                add(TraceObservation(
                    logical_id=(
                        f"compaction:{payload_type}:"
                        f"{payload.get('compaction_request_id') or payload.get('compaction_id') or seq}:"
                        f"{seq}"
                    ),
                    name=payload_type.replace("_", "-"),
                    as_type="event",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent_for(event),
                    input=value if value is not None else {
                        key: item for key, item in payload.items()
                        if key not in {"type", "request_payload", "response_payload", "checkpoint_payload"}
                    },
                    level="ERROR" if payload_type == "compaction_request_failed" else "DEFAULT",
                    status_message=payload.get("error"),
                    metadata={**common_metadata, **payload_meta},
                ))
                continue

            if payload_type == "other" or payload_type not in _CODEX_ROLLOUT_RAW_EVENT_TYPES:
                raw_payloads: list[Any] = []
                for reference in payload.get("payloads", []) if isinstance(payload.get("payloads"), list) else []:
                    value, payload_meta = _codex_rollout_payload(bundle, reference)
                    if payload_meta["payload_quality"] != "exact_official_payload":
                        payload_issue_count += 1
                    raw_payloads.append(value)
                unknown = payload_type not in _CODEX_ROLLOUT_RAW_EVENT_TYPES
                add(TraceObservation(
                    logical_id=f"raw-event:{seq}",
                    name="unmapped-codex-rollout-trace-event" if unknown else f"codex-trace:{payload.get('kind') or 'other'}",
                    as_type="event",
                    start_ns=ts,
                    end_ns=ts,
                    parent_logical_id=parent_for(event),
                    input={
                        "summary": payload.get("summary"),
                        "metadata": payload.get("metadata"),
                        "payloads": raw_payloads,
                    },
                    level="WARNING" if unknown else "DEFAULT",
                    status_message=(f"unknown official raw event type {payload_type}" if unknown else None),
                    metadata=common_metadata,
                ))

        if root_thread_id not in threads:
            threads[root_thread_id] = add(TraceObservation(
                logical_id=thread_logical(root_thread_id),
                name="run-codex-agent",
                as_type="agent",
                start_ns=trace_start,
                end_ns=trace_end,
                input=root_input,
                output=root_output,
                metadata={
                    "thread_id": root_thread_id,
                    "timestamp_quality": "official_manifest_fallback",
                    "missing_thread_started_event": True,
                },
            ))

        # Project the official spawn information into a Langfuse parent tree.
        # The original edge semantics remain explicit in metadata and links.
        spawn_tools = [
            tool for tool in tools_by_id.values()
            if str((tool.metadata.get("official_tool_kind") or {}).get("type")
                   if isinstance(tool.metadata.get("official_tool_kind"), dict)
                   else tool.metadata.get("official_tool_kind")) == "spawn_agent"
            or tool.name.endswith("spawn_agent")
        ]

        def owning_thread_id(observation: TraceObservation) -> str | None:
            cursor: TraceObservation | None = observation
            visited: set[str] = set()
            while cursor is not None and cursor.logical_id not in visited:
                visited.add(cursor.logical_id)
                thread_id = cursor.metadata.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    return thread_id
                cursor = by_id.get(str(cursor.parent_logical_id or ""))
            return None

        for thread_id, thread in threads.items():
            if thread_id == root_thread_id:
                continue
            metadata = thread_metadata.get(thread_id, {})
            spawn_meta = (
                metadata.get("session_source", {}).get("subagent", {}).get("thread_spawn", {})
                if isinstance(metadata.get("session_source"), dict) else {}
            )
            parent_thread_id = str(spawn_meta.get("parent_thread_id") or thread.metadata.get("parent_thread_id") or root_thread_id)
            task_name = str(metadata.get("task_name") or "")
            candidates = [
                tool for tool in spawn_tools
                if owning_thread_id(tool) == parent_thread_id
            ]
            matching = [
                tool for tool in candidates
                if task_name and _codex_spawn_task_name(tool.input) == task_name
            ]
            selected = matching[0] if len(matching) == 1 else candidates[0] if len(candidates) == 1 else None
            if selected:
                original_runtime_end = selected.end_ns
                selected.end_ns = max(selected.end_ns, thread.end_ns)
                selected.metadata.setdefault("official_runtime_end_ns", original_runtime_end)
                selected.metadata["presentation_end_expanded_for_subagent"] = True
                thread.parent_logical_id = selected.logical_id
                thread.links.append(TraceLink(thread_logical(parent_thread_id), {
                    "relationship": "official_parent_thread",
                }))
                thread.metadata["parent_link_quality"] = (
                    "official_spawn_metadata_and_task_name"
                    if matching else "official_spawn_metadata_and_unique_tool"
                )
                thread.metadata["relationship_projection"] = "official_interaction_edge_to_langfuse_tree"
            else:
                thread.parent_logical_id = thread_logical(parent_thread_id)
                thread.links.extend(TraceLink(tool.logical_id, {
                    "relationship": "possible_spawn_tool",
                }) for tool in candidates)
                thread.metadata["parent_link_quality"] = "official_parent_thread"

        # Official runtime objects may legally outlive their activating turn.
        # Keep exact bounds and move those objects to the owning thread while
        # retaining a causal link to the turn instead of falsifying timestamps.
        for observation in observations:
            parent = by_id.get(str(observation.parent_logical_id or ""))
            if not parent or parent.name != "run-agent-turn":
                continue
            if observation.as_type not in {"tool", "chain"}:
                continue
            if observation.start_ns >= parent.start_ns and observation.end_ns <= parent.end_ns:
                continue
            original_parent = observation.parent_logical_id
            observation.parent_logical_id = parent.parent_logical_id
            observation.links.append(TraceLink(str(original_parent), {
                "relationship": "started_by_codex_turn",
            }))
            observation.metadata["parent_projection_quality"] = "official_runtime_outlives_turn"

        root = threads[root_thread_id]
        root.start_ns = min(root.start_ns, trace_start, *(item.start_ns for item in observations))
        root.end_ns = max(root.end_ns, trace_end, *(item.end_ns for item in observations))
        root.input = root_input if root_input is not None else root.input
        root.output = root_output if root_output is not None else root.output
        level, status_message = _codex_rollout_status(rollout_status)
        root.level = level
        root.status_message = status_message
        root.metadata["official_rollout_status"] = rollout_status

        for observation in observations:
            if observation is root:
                continue
            if observation.end_ns == observation.start_ns and observation.as_type in {
                "agent", "generation", "tool", "chain",
            } and observation.metadata.get("official_end_seq") is None:
                observation.end_ns = max(observation.start_ns, trace_end)
                observation.level = "WARNING"
                observation.status_message = observation.status_message or "official lifecycle still open"
                observation.metadata["lifecycle_quality"] = "open_official_trace"

        seqs = [int(event["seq"]) for event in events if isinstance(event.get("seq"), int)]
        seq_contiguous = seqs == list(range(1, len(seqs) + 1))
        yield AgentTrace(
            logical_id=f"codex-rollout-trace:{official_trace_id}",
            name="codex-agent-session",
            source="codex",
            source_session_id=rollout_id,
            project=project,
            observations=observations,
            metadata={
                "adapter": "codex-rollout-trace-v1",
                "data_source": "official_codex_rollout_trace_bundle",
                "official_trace_id": official_trace_id,
                "official_rollout_id": rollout_id,
                "official_manifest_schema_version": manifest.get("schema_version"),
                "raw_event_count": len(events),
                "raw_event_types": dict(sorted(raw_type_counts.items())),
                "unknown_raw_event_types": sorted(unknown_raw_types),
                "seq_first": seqs[0] if seqs else None,
                "seq_last": seqs[-1] if seqs else None,
                "seq_contiguous": seq_contiguous,
                "payload_issue_count": payload_issue_count,
                "thread_count": len(threads),
                "turn_count": len(turns),
                "user_turn_count": sum(item.name == "user-turn" for item in observations),
                "source_contract": "codex-rs/rollout-trace schema v1",
            },
        )


def iter_codex_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    official_traces = list(_iter_codex_rollout_trace_traces(include_thinking))
    for trace in official_traces:
        yield trace
    official_rollout_ids = {
        str(trace.metadata.get("official_rollout_id") or trace.source_session_id)
        for trace in official_traces
    }
    records: dict[str, tuple[Path, list[dict[str, Any]], dict[str, Any]]] = {}
    for path in _codex_files():
        entries = _jsonl(path)
        if not entries:
            continue
        entries.sort(key=lambda entry: (
            _codex_order(entry), int(entry.get("_agentstracer_line") or 0),
        ))
        meta = _codex_meta(entries, path)
        records[str(meta.get("id") or path.stem)] = (path, entries, meta)
    if not records:
        return

    def root_of(thread_id: str) -> str:
        seen: set[str] = set()
        current = thread_id
        while current not in seen:
            seen.add(current)
            parent = records.get(current, (None, None, {}))[2].get("parent_thread_id")
            if not parent or str(parent) not in records:
                return current
            current = str(parent)
        return thread_id

    groups: dict[str, list[str]] = defaultdict(list)
    for thread_id in records:
        groups[root_of(thread_id)].append(thread_id)

    for root_id, thread_ids in groups.items():
        if root_id in official_rollout_ids:
            continue
        observations: list[TraceObservation] = []
        root_input: Any = None
        root_output: Any = None
        all_times: list[int | None] = []
        group_entries: list[dict[str, Any]] = []
        root_meta = records[root_id][2]
        for thread_id in sorted(thread_ids, key=lambda item: item != root_id):
            thread_observation_offset = len(observations)
            path, entries, meta = records[thread_id]
            group_entries.extend(entries)
            times = [timestamp_ns(e.get("timestamp")) for e in entries]
            start, end = bounds(times, file_mtime_ns(path))
            all_times.extend((start, end))
            agent_logical = f"thread:{thread_id}"
            parent_id = meta.get("parent_thread_id")
            parent = f"thread:{parent_id}" if parent_id and str(parent_id) in records else None
            turns = _codex_turns(entries)
            if turns:
                start = min(start, *(turn["start"] for turn in turns))
                end = max(end, *(turn["end"] for turn in turns))
                all_times.extend(
                    value for turn in turns for value in (turn["start"], turn["end"])
                )
            for turn in turns:
                turn["logical_id"] = f"{agent_logical}:turn:{turn['id']}"
            user_messages = [
                {
                    "entry": e,
                    "timestamp": timestamp_ns(e.get("timestamp")),
                    "line": _codex_order(e),
                    "content": (e.get("payload") or {}).get("message"),
                    "role": "user",
                    "source": "event_msg.user_message",
                }
                for e in entries
                if e.get("type") == "event_msg" and (e.get("payload") or {}).get("type") == "user_message"
            ]
            response_user_messages = []
            for entry in entries:
                response_payload = (
                    entry.get("payload")
                    if isinstance(entry.get("payload"), dict) else {}
                )
                if not (
                    entry.get("type") == "response_item"
                    and response_payload.get("type") == "message"
                    and response_payload.get("role") == "user"
                ):
                    continue
                response_content = _content_text(
                    response_payload.get("content"), "input_text", "text",
                )
                if response_content is None and response_payload.get("content") is not None:
                    response_content = response_payload.get("content")
                if response_content is not None:
                    response_user_messages.append({
                        "entry": entry,
                        "timestamp": timestamp_ns(entry.get("timestamp")),
                        "line": _codex_order(entry),
                        "content": response_content,
                        "role": "user",
                        "source": "response_item.message.user",
                    })
            current_agent_path = str(meta.get("agent_path") or "/root")
            inter_agent_inputs = []
            for entry in entries:
                inter_payload = (
                    entry.get("payload")
                    if isinstance(entry.get("payload"), dict) else {}
                )
                if not (
                    entry.get("type") == "response_item"
                    and inter_payload.get("type") == "agent_message"
                    and inter_payload.get("recipient") == current_agent_path
                ):
                    continue
                inter_content = _content_text(
                    inter_payload.get("content"), "input_text", "text",
                )
                if inter_content:
                    inter_agent_inputs.append({
                        "entry": entry,
                        "timestamp": timestamp_ns(entry.get("timestamp")),
                        "line": _codex_order(entry),
                        "content": inter_content,
                        "role": "agent",
                        "author": inter_payload.get("author"),
                        "recipient": inter_payload.get("recipient"),
                        "source": "response_item.agent_message.inbound",
                    })
            agent_messages = [
                (timestamp_ns(e.get("timestamp")), (e.get("payload") or {}).get("message"))
                for e in entries
                if e.get("type") == "event_msg" and (e.get("payload") or {}).get("type") == "agent_message"
            ]
            if thread_id == root_id:
                root_input = next(
                    (item["content"] for item in user_messages if item["content"]),
                    next((item["content"] for item in response_user_messages if item["content"]), None),
                )
                root_output = next((value for _, value in reversed(agent_messages) if value), None)
            agent_input = next(
                (item["content"] for item in user_messages if item["content"]),
                next(
                    (item["content"] for item in inter_agent_inputs if item["content"]),
                    next((item["content"] for item in response_user_messages if item["content"]), None),
                ),
            )
            observations.append(TraceObservation(
                logical_id=agent_logical,
                name="run-codex-agent" if thread_id == root_id else "run-codex-subagent",
                as_type="agent",
                start_ns=start,
                end_ns=end,
                parent_logical_id=parent,
                input=agent_input,
                output=next((value for _, value in reversed(agent_messages) if value), None),
                model=next(
                    ((e.get("payload") or {}).get("model") for e in entries if e.get("type") == "turn_context" and (e.get("payload") or {}).get("model")),
                    None,
                ),
                metadata={
                    "thread_id": thread_id,
                    "parent_thread_id": parent_id,
                    "forked_from_id": meta.get("forked_from_id"),
                    "agent_nickname": meta.get("agent_nickname"),
                    "agent_path": meta.get("agent_path"),
                    "cwd": meta.get("cwd"),
                    "originator": meta.get("originator"),
                },
            ))

            turn_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for message in user_messages:
                message_ts = int(message["timestamp"] or start)
                container = _codex_container(
                    message_ts, turns, agent_logical, int(message["line"]),
                )
                if container != agent_logical:
                    turn_inputs[container].append(message)
            # Subagent prompts and older imported records occasionally omit
            # event_msg.user_message.  Use role=user response items only for a
            # turn that has no authoritative event message.
            for message in response_user_messages:
                message_ts = int(message["timestamp"] or start)
                container = _codex_container(
                    message_ts, turns, agent_logical, int(message["line"]),
                )
                if container == agent_logical or turn_inputs.get(container):
                    continue
                if not any(
                    canonical_json(existing["content"])
                    == canonical_json(message["content"])
                    for existing in turn_inputs[container]
                ):
                    turn_inputs[container].append(message)
            for message in inter_agent_inputs:
                message_ts = int(message["timestamp"] or start)
                container = _codex_container(
                    message_ts, turns, agent_logical, int(message["line"]),
                )
                if container == agent_logical or turn_inputs.get(container):
                    continue
                turn_inputs[container].append(message)

            for turn in turns:
                turn_logical = str(turn["logical_id"])
                payload = turn.get("payload", {})
                messages = turn_inputs.get(turn_logical, [])
                turn_input = None
                if len(messages) == 1:
                    turn_input = {
                        "role": messages[0]["role"],
                        "content": messages[0]["content"],
                        **(
                            {"author": messages[0].get("author")}
                            if messages[0].get("author") else {}
                        ),
                    }
                elif messages:
                    turn_input = {"messages": [
                        {
                            "role": message["role"],
                            "content": message["content"],
                            **(
                                {"author": message.get("author")}
                                if message.get("author") else {}
                            ),
                        }
                        for message in messages
                    ]}
                observations.append(TraceObservation(
                    logical_id=turn_logical,
                    name="run-agent-turn",
                    as_type="agent",
                    start_ns=turn["start"],
                    end_ns=max(turn["start"], turn["end"]),
                    parent_logical_id=agent_logical,
                    input=turn_input,
                    output=turn.get("output"),
                    level="ERROR" if turn.get("aborted") else ("WARNING" if turn.get("open") else "DEFAULT"),
                    status_message="turn aborted" if turn.get("aborted") else ("turn still open" if turn.get("open") else None),
                    metadata={
                        "turn_id": turn["id"],
                        "duration_ms": payload.get("duration_ms"),
                        "time_to_first_token_ms": payload.get("time_to_first_token_ms"),
                        "user_message_count": len(messages),
                        "user_input_quality": (
                            "exact_event_messages" if any(
                                message["source"] == "event_msg.user_message"
                                for message in messages
                            ) else "inter_agent_message" if any(
                                message["source"]
                                == "response_item.agent_message.inbound"
                                for message in messages
                            ) else "response_item_fallback" if messages else "missing"
                        ),
                        "timestamp_quality": turn.get("timestamp_quality"),
                    },
                ))

            for message in [
                item for values in turn_inputs.values() for item in values
                if item["role"] == "user"
            ]:
                raw_ts = int(message["timestamp"] or start)
                parent_logical = _codex_container(
                    raw_ts, turns, agent_logical, int(message["line"]),
                )
                parent_turn = next(
                    (turn for turn in turns if turn.get("logical_id") == parent_logical),
                    None,
                )
                visible_ts = raw_ts
                timestamp_quality = "exact_event_timestamp"
                if parent_turn and not (
                    parent_turn["start"] <= raw_ts <= parent_turn["end"]
                ):
                    visible_ts = parent_turn["start"]
                    timestamp_quality = "inferred_from_turn_start"
                source_entry = message["entry"]
                observations.append(TraceObservation(
                    logical_id=_entry_id(f"{agent_logical}:user-turn", source_entry),
                    name="user-turn",
                    as_type="event",
                    start_ns=visible_ts,
                    end_ns=visible_ts,
                    parent_logical_id=parent_logical,
                    input={"role": "user", "content": message["content"]},
                    metadata={
                        "role": "user",
                        "message_source": message["source"],
                        "turn_id": (
                            parent_turn.get("id") if parent_turn else None
                        ),
                        "timestamp_quality": timestamp_quality,
                        "source_line": message["line"],
                    },
                ))

            outputs: dict[str, dict[str, Any]] = {}
            known_call_ids: set[str] = set()
            for entry in entries:
                payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                if (
                    entry.get("type") == "response_item"
                    and payload.get("call_id")
                    and payload.get("type") in (
                        "function_call", "custom_tool_call", "tool_search_call",
                    )
                ):
                    known_call_ids.add(str(payload["call_id"]))
                if entry.get("type") == "response_item" and payload.get("call_id") and payload.get("type") in (
                    "function_call_output", "custom_tool_call_output", "tool_search_output",
                ):
                    outputs[str(payload["call_id"])] = {"entry": entry, "payload": payload}

            pending_tools_by_turn: dict[str, list[TraceObservation]] = defaultdict(list)
            generation_start = start
            generation_inputs: list[dict[str, Any]] = []
            generation_input_keys: set[str] = set()
            next_generation_inputs: list[dict[str, Any]] = []
            next_generation_input_keys: set[str] = set()
            generation_parts: list[Any] = []
            generation_output_keys: set[str] = set()
            generation_tool_calls: list[dict[str, Any]] = []
            generation_reasoning_events = 0
            generation_first_output_ts: int | None = None
            generation_last_output_ts: int | None = None
            generation_first_output_line: int | None = None
            current_model: str | None = None

            def add_generation_input(
                target: list[dict[str, Any]],
                fingerprints: set[str],
                message: dict[str, Any],
            ) -> None:
                fingerprint = canonical_json(message)
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    target.append(message)

            def add_generation_part(part: Any) -> None:
                fingerprint = canonical_json(part)
                if fingerprint not in generation_output_keys:
                    generation_output_keys.add(fingerprint)
                    generation_parts.append(part)

            def mark_generation_output(output_ts: int, output_line: int) -> None:
                nonlocal generation_first_output_ts, generation_last_output_ts
                nonlocal generation_first_output_line
                generation_first_output_ts = generation_first_output_ts or output_ts
                generation_last_output_ts = output_ts
                generation_first_output_line = (
                    generation_first_output_line
                    if generation_first_output_line is not None else output_line
                )

            def flush_codex_generation(
                logical: str,
                boundary_ts: int,
                boundary_line: int,
                last_usage: dict[str, Any] | None = None,
            ) -> None:
                nonlocal generation_start
                nonlocal generation_inputs, generation_input_keys
                nonlocal next_generation_inputs, next_generation_input_keys
                nonlocal generation_parts, generation_output_keys, generation_tool_calls
                nonlocal generation_reasoning_events
                nonlocal generation_first_output_ts, generation_last_output_ts
                nonlocal generation_first_output_line

                has_output = bool(
                    generation_parts or generation_tool_calls or generation_reasoning_events
                )
                if not has_output:
                    return
                output: dict[str, Any] = {}
                if generation_parts:
                    output["content"] = generation_parts
                if generation_tool_calls:
                    output["tool_calls"] = generation_tool_calls
                if generation_reasoning_events:
                    output["reasoning"] = {
                        "event_count": generation_reasoning_events,
                        "content_available": any(
                            isinstance(part, dict) and "thinking" in part
                            for part in generation_parts
                        ),
                    }
                if generation_inputs:
                    input_value: Any = {"messages": generation_inputs}
                    input_quality = "source_messages"
                else:
                    input_value = {
                        "context": {
                            "available": False,
                            "reason": "source_log_omits_prompt_snapshot",
                        }
                    }
                    input_quality = "explicit_missing_context"

                observation_end = max(
                    generation_start,
                    generation_last_output_ts or boundary_ts,
                )
                parent = _codex_container(
                    generation_first_output_ts or boundary_ts,
                    turns,
                    agent_logical,
                    generation_first_output_line or boundary_line,
                )
                parent_turn = next(
                    (turn for turn in turns if turn.get("logical_id") == parent),
                    None,
                )
                timestamp_quality = "source_output_event_bounds"
                observation_start = min(generation_start, observation_end)
                completion_start = generation_first_output_ts
                if parent_turn and not (
                    parent_turn["start"] <= observation_end <= parent_turn["end"]
                ):
                    observation_start = parent_turn["start"]
                    observation_end = parent_turn["end"]
                    completion_start = parent_turn["start"]
                    timestamp_quality = "inferred_from_parent_turn"
                observations.append(TraceObservation(
                    logical_id=logical,
                    name="generate-codex-step",
                    as_type="generation",
                    start_ns=observation_start,
                    end_ns=observation_end,
                    parent_logical_id=parent,
                    input=input_value,
                    output=output,
                    model=current_model,
                    usage=_usage(last_usage or {}),
                    completion_start_ns=completion_start,
                    metadata={
                        "timestamp_quality": timestamp_quality,
                        "usage_timestamp_ns": boundary_ts if last_usage else None,
                        "input_quality": input_quality,
                        "reasoning_content_encrypted": bool(
                            generation_reasoning_events
                            and not output["reasoning"]["content_available"]
                        ),
                    },
                ))
                for tool in pending_tools_by_turn.pop(parent, []):
                    tool.metadata["requested_by_generation"] = logical
                    tool.links.append(TraceLink(
                        logical, {"agentstracer.relationship": "requested_by"},
                    ))

                generation_inputs = next_generation_inputs
                generation_input_keys = next_generation_input_keys
                next_generation_inputs = []
                next_generation_input_keys = set()
                generation_parts = []
                generation_output_keys = set()
                generation_tool_calls = []
                generation_reasoning_events = 0
                generation_first_output_ts = None
                generation_last_output_ts = None
                generation_first_output_line = None
                generation_start = boundary_ts

            for entry in entries:
                ts = timestamp_ns(entry.get("timestamp")) or generation_start
                line = _codex_order(entry)
                payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                if entry.get("type") == "turn_context" and isinstance(payload.get("model"), str):
                    current_model = payload["model"]
                if entry.get("type") == "response_item":
                    item_type = payload.get("type")
                    if item_type == "message":
                        role = str(payload.get("role") or "assistant")
                        if role in ("user", "developer", "system"):
                            content = _content_text(
                                payload.get("content"), "input_text", "text",
                            )
                            if content:
                                add_generation_input(
                                    generation_inputs,
                                    generation_input_keys,
                                    {"role": role, "content": content},
                                )
                        else:
                            content = _content_text(
                                payload.get("content"), "output_text", "text",
                            )
                            if content:
                                add_generation_part(content)
                                mark_generation_output(ts, line)
                    elif item_type == "reasoning":
                        generation_reasoning_events += 1
                        mark_generation_output(ts, line)
                        if include_thinking:
                            summary = _content_text(
                                payload.get("summary"), "summary_text", "text",
                            )
                            if summary:
                                add_generation_part({"thinking": summary})
                    elif item_type == "agent_message":
                        inter_agent_content = _content_text(
                            payload.get("content"), "input_text", "text",
                        )
                        encrypted_parts = sum(
                            isinstance(block, dict)
                            and block.get("type") == "encrypted_content"
                            for block in (payload.get("content") or [])
                        ) if isinstance(payload.get("content"), list) else 0
                        observations.append(TraceObservation(
                            logical_id=_entry_id(
                                f"{agent_logical}:inter-agent-message", entry,
                            ),
                            name="inter-agent-message",
                            as_type="event",
                            start_ns=ts,
                            end_ns=ts,
                            parent_logical_id=_codex_container(
                                ts, turns, agent_logical, line,
                            ),
                            input={
                                "author": payload.get("author"),
                                "recipient": payload.get("recipient"),
                                "content": inter_agent_content,
                            },
                            metadata={
                                "message_source": "response_item.agent_message",
                                "encrypted_part_count": encrypted_parts,
                                "content_available": inter_agent_content is not None,
                            },
                        ))
                    if item_type in ("function_call", "custom_tool_call", "tool_search_call"):
                        call_id = str(payload.get("call_id") or _entry_id("codex-call", entry))
                        result = outputs.get(call_id)
                        result_ts = timestamp_ns(result["entry"].get("timestamp")) if result else None
                        container = _codex_container(
                            ts, turns, agent_logical, line,
                        )
                        parent = container
                        raw_result = result["payload"].get("output") if result else None
                        status = payload.get("status")
                        tool_name, tool_metadata = _codex_tool_descriptor(payload)
                        tool_input = _codex_tool_input(
                            payload, tool_name, tool_metadata,
                        )
                        generation_tool_calls.append({
                            "id": call_id,
                            "name": tool_name,
                            "arguments": tool_input,
                        })
                        mark_generation_output(ts, line)
                        tool_output = _tool_output(raw_result)
                        failed = _codex_tool_failed(status, tool_output)
                        duration_ms = max(0, (max(ts, result_ts or ts) - ts) / 1_000_000)
                        tool_observation = TraceObservation(
                            logical_id=f"{agent_logical}:tool:{call_id}",
                            name=tool_name,
                            as_type="tool",
                            start_ns=ts,
                            end_ns=max(ts, result_ts or ts),
                            parent_logical_id=parent,
                            input=tool_input,
                            output=tool_output,
                            level="ERROR" if failed else "DEFAULT",
                            status_message=(
                                _codex_tool_error_message(status, tool_output)
                                if failed else None
                            ),
                            metadata={
                                "tool_call_id": call_id,
                                **tool_metadata,
                                "duration_ms": duration_ms,
                                "timestamp_quality": "exact" if result else "open_call",
                            },
                        )
                        observations.append(tool_observation)
                        pending_tools_by_turn[container].append(tool_observation)
                    elif item_type in (
                        "function_call_output", "custom_tool_call_output",
                        "tool_search_output",
                    ):
                        result_call_id = str(
                            payload.get("call_id") or _entry_id("codex-result", entry)
                        )
                        add_generation_input(
                            next_generation_inputs,
                            next_generation_input_keys,
                            {
                                "role": "tool",
                                "tool_call_id": result_call_id,
                                "content": _tool_output(payload.get("output")),
                            },
                        )
                if entry.get("type") == "event_msg" and payload.get("type") == "user_message":
                    if payload.get("message") is not None:
                        add_generation_input(
                            generation_inputs,
                            generation_input_keys,
                            {"role": "user", "content": payload.get("message")},
                        )
                if entry.get("type") == "event_msg" and payload.get("type") == "agent_message":
                    if isinstance(payload.get("message"), str) and payload["message"]:
                        add_generation_part(payload["message"])
                        mark_generation_output(ts, line)
                if entry.get("type") == "event_msg" and payload.get("type") == "agent_reasoning":
                    generation_reasoning_events += 1
                    mark_generation_output(ts, line)
                    if include_thinking and isinstance(payload.get("text"), str):
                        add_generation_part({"thinking": payload["text"]})
                if entry.get("type") == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
                    last_usage = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
                    flush_codex_generation(
                        _entry_id(f"{agent_logical}:generation", entry),
                        ts,
                        line,
                        last_usage,
                    )
                if entry.get("type") == "event_msg" and payload.get("type") in (
                    "context_compacted", "sub_agent_activity", "turn_aborted",
                ):
                    observations.append(TraceObservation(
                        logical_id=_entry_id(f"{agent_logical}:event", entry),
                        name=str(payload.get("type")),
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input={k: v for k, v in payload.items() if k != "type"},
                        level="ERROR" if payload.get("type") == "turn_aborted" else "DEFAULT",
                    ))
                if entry.get("type") == "event_msg" and payload.get("type") == "thread_settings_applied":
                    settings = payload.get("thread_settings")
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:thread-settings", entry,
                        ),
                        name="thread-settings-applied",
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input=settings,
                        metadata={
                            "setting_keys": (
                                sorted(str(key) for key in settings)
                                if isinstance(settings, dict) else []
                            ),
                        },
                    ))
                if entry.get("type") == "event_msg" and payload.get("type") in (
                    "web_search_end", "patch_apply_end", "mcp_tool_call_end",
                ):
                    completion_type = str(payload["type"])
                    call_id = str(payload.get("call_id") or "")
                    parent_logical = _codex_container(
                        ts, turns, agent_logical, line,
                    )
                    if call_id and call_id in known_call_ids:
                        parent_logical = f"{agent_logical}:tool:{call_id}"
                    completion_input: Any
                    completion_output: Any
                    completion_start = ts
                    if completion_type == "web_search_end":
                        completion_input = {
                            "query": payload.get("query"),
                            "action": payload.get("action"),
                        }
                        completion_output = payload.get("results")
                        completion_name = "web-search-completed"
                    elif completion_type == "patch_apply_end":
                        completion_input = {"changes": payload.get("changes")}
                        completion_output = {
                            "status": payload.get("status"),
                            "success": payload.get("success"),
                            "stdout": payload.get("stdout"),
                            "stderr": payload.get("stderr"),
                        }
                        completion_name = "patch-apply-completed"
                    else:
                        completion_input = payload.get("invocation")
                        completion_output = payload.get("result")
                        completion_name = "mcp-tool-call-completed"
                        duration = payload.get("duration")
                        if isinstance(duration, dict):
                            duration_ns = int(duration.get("secs") or 0) * 1_000_000_000
                            duration_ns += int(duration.get("nanos") or 0)
                            completion_start = max(0, ts - duration_ns)
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:tool-completion", entry,
                        ),
                        name=completion_name,
                        as_type="event",
                        start_ns=completion_start,
                        end_ns=ts,
                        parent_logical_id=parent_logical,
                        input=completion_input,
                        output=completion_output,
                        level=(
                            "ERROR" if payload.get("success") is False else "DEFAULT"
                        ),
                        metadata={
                            "source_event_type": completion_type,
                            "tool_call_id": call_id or None,
                            "matched_tool_observation": bool(
                                call_id and call_id in known_call_ids
                            ),
                            **{
                                key: value for key, value in payload.items()
                                if key in {
                                    "plugin_id", "connector_id", "app_name",
                                    "action_name", "link_id", "mcp_app_resource_uri",
                                }
                            },
                        },
                    ))
                if entry.get("type") == "world_state":
                    state = payload.get("state")
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:world-state", entry,
                        ),
                        name="world-state-update",
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input={
                            "full": payload.get("full"),
                            "state_keys": (
                                sorted(str(key) for key in state)
                                if isinstance(state, dict) else []
                            ),
                        },
                        metadata={"payload_preserved": False, "summary_only": True},
                    ))
                if entry.get("type") == "compacted":
                    history = payload.get("replacement_history")
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:compacted", entry,
                        ),
                        name="context-window-compacted",
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input={
                            "window_id": payload.get("window_id"),
                            "window_number": payload.get("window_number"),
                            "previous_window_id": payload.get("previous_window_id"),
                            "first_window_id": payload.get("first_window_id"),
                            "replacement_history_items": (
                                len(history) if isinstance(history, list) else None
                            ),
                            "summary_available": bool(payload.get("message")),
                        },
                        metadata={"payload_preserved": False, "summary_only": True},
                    ))
                if entry.get("type") == "inter_agent_communication_metadata":
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:inter-agent-metadata", entry,
                        ),
                        name="inter-agent-trigger",
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input=_codex_public_payload(payload),
                    ))

                entry_type = str(entry.get("type") or "<missing>")
                payload_type = str(payload.get("type") or "<none>")
                generic_official = (
                    entry_type == "event_msg"
                    and payload_type in _CODEX_KNOWN_EVENT_TYPES
                    and payload_type not in _CODEX_SEMANTIC_EVENT_TYPES
                ) or (
                    entry_type == "response_item"
                    and payload_type in _CODEX_KNOWN_RESPONSE_ITEM_TYPES
                    and payload_type not in _CODEX_SEMANTIC_RESPONSE_ITEM_TYPES
                ) or entry_type == "inter_agent_communication"
                if generic_official:
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:official-generic", entry,
                        ),
                        name=(
                            f"codex-event:{payload_type}"
                            if entry_type == "event_msg"
                            else f"codex-response-item:{payload_type}"
                            if entry_type == "response_item"
                            else "inter-agent-communication"
                        ),
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input=_codex_public_payload(payload),
                        level=(
                            "ERROR" if payload_type == "error"
                            else "WARNING" if payload_type in {
                                "warning", "guardian_warning", "stream_error",
                            } else "DEFAULT"
                        ),
                        metadata={
                            "mapping_quality": "official_contract_generic",
                            "top_level_type": entry_type,
                            "payload_type": payload_type,
                            "source_order": line,
                            "source_ordinal": entry.get("ordinal"),
                        },
                    ))
                unmapped = (
                    entry_type not in _CODEX_KNOWN_TOP_LEVEL_TYPES
                    or (
                        entry_type == "event_msg"
                        and payload_type not in _CODEX_KNOWN_EVENT_TYPES
                    )
                    or (
                        entry_type == "response_item"
                        and payload_type not in _CODEX_KNOWN_RESPONSE_ITEM_TYPES
                    )
                )
                if unmapped:
                    observations.append(TraceObservation(
                        logical_id=_entry_id(
                            f"{agent_logical}:unmapped-schema", entry,
                        ),
                        name="unmapped-codex-schema",
                        as_type="event",
                        start_ns=ts,
                        end_ns=ts,
                        parent_logical_id=_codex_container(
                            ts, turns, agent_logical, line,
                        ),
                        input=_codex_public_payload(payload),
                        level="WARNING",
                        status_message=(
                            f"unmapped Codex schema {entry_type}:{payload_type}"
                        ),
                        metadata={
                            "top_level_type": entry_type,
                            "payload_type": payload_type,
                            "payload_keys": sorted(str(key) for key in payload),
                        },
                    ))

            if generation_parts or generation_tool_calls or generation_reasoning_events:
                final_entry = entries[-1]
                final_ts = timestamp_ns(final_entry.get("timestamp")) or end
                flush_codex_generation(
                    _entry_id(f"{agent_logical}:generation:final", final_entry),
                    final_ts,
                    _codex_order(final_entry),
                )

            # Imported/forked history can preserve lifecycle timestamps in the
            # payload while stamping every copied record with the import time.
            # Source order still gives the correct parent turn; clamp only
            # large cross-clock-domain skews and label the repair explicitly.
            thread_observations = observations[thread_observation_offset:]
            thread_by_id = {
                observation.logical_id: observation
                for observation in thread_observations
            }
            for observation in thread_observations:
                parent_observation = thread_by_id.get(
                    str(observation.parent_logical_id or "")
                )
                if not parent_observation or parent_observation.name != "run-agent-turn":
                    continue
                before_ns = max(0, parent_observation.start_ns - observation.start_ns)
                after_ns = max(0, observation.end_ns - parent_observation.end_ns)
                skew_ns = max(before_ns, after_ns)
                if skew_ns == 0:
                    continue
                original_start = observation.start_ns
                original_end = observation.end_ns
                observation.start_ns = max(
                    parent_observation.start_ns,
                    min(observation.start_ns, parent_observation.end_ns),
                )
                observation.end_ns = max(
                    observation.start_ns,
                    min(observation.end_ns, parent_observation.end_ns),
                )
                observation.metadata["original_start_ns"] = original_start
                observation.metadata["original_end_ns"] = original_end
                observation.metadata["timestamp_quality"] = (
                    "clamped_to_parent_turn_due_source_clock_domain"
                    if skew_ns > 5_000_000_000
                    else "clamped_to_parent_turn_boundary"
                )

        trace_start, trace_end = bounds(all_times)
        # The root thread observation already exists; expand it to encompass
        # children and carry the overall trace input/output.
        root_obs = next(o for o in observations if o.logical_id == f"thread:{root_id}")
        root_obs.start_ns = min(root_obs.start_ns, trace_start)
        root_obs.end_ns = max(root_obs.end_ns, trace_end)
        root_obs.input = root_input
        root_obs.output = root_output
        schema_profile = _codex_schema_profile(group_entries)
        yield AgentTrace(
            logical_id=f"codex-thread-tree:{root_id}",
            name="codex-agent-session",
            source="codex",
            source_session_id=root_id,
            project=str(root_meta.get("cwd") or "codex"),
            observations=observations,
            metadata={
                "adapter": "codex-v3",
                "thread_count": len(thread_ids),
                "turn_count": sum(
                    observation.name == "run-agent-turn"
                    for observation in observations
                ),
                "user_turn_count": sum(
                    observation.name == "user-turn"
                    for observation in observations
                ),
                "schema_profile": schema_profile,
            },
        )


# ---------------------------------------------------------------------------
# Kimi CLI


def _kimi_wire_sessions() -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    if not KIMI_SESSIONS_DIR.exists():
        return
    for path in sorted(KIMI_SESSIONS_DIR.rglob("wire.jsonl")):
        entries = _jsonl(path)
        if entries:
            yield path, entries


def _kimi_message(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("message") if isinstance(entry.get("message"), dict) else {}


def _structured_tool_input(value: Any) -> tuple[dict[str, Any], str]:
    """Normalize string-encoded tool arguments without guessing source fragments."""
    if isinstance(value, dict):
        return value, "source_object"
    if isinstance(value, list):
        return {"arguments": value}, "source_array_wrapped"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}, "empty_source"
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError):
            return {"arguments_text": value}, "unparseable_source_text"
        if isinstance(decoded, dict):
            return decoded, "parsed_json"
        return {"arguments": decoded}, "parsed_json_scalar_wrapped"
    if value is None:
        return {}, "missing_source"
    return {"arguments": value}, "source_scalar_wrapped"


def _kimi_call_tool_input(call: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
    payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    raw = function.get("arguments")
    parts = [part for part in call.get("argument_parts", []) if isinstance(part, str)]
    if parts and (isinstance(raw, str) or raw is None):
        combined = (raw or "") + "".join(parts)
        tool_input, quality = _structured_tool_input(combined)
        if quality == "parsed_json":
            quality = "reconstructed_json_parts"
        elif quality == "unparseable_source_text":
            quality = "unparseable_source_fragments"
        return tool_input, quality, len(parts)
    tool_input, quality = _structured_tool_input(raw)
    return tool_input, quality, 0


def _kimi_generation_input(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    if messages:
        return {"messages": messages}, "source_messages"
    return {
        "context": {
            "available": False,
            "reason": "source_stream_omits_prompt_snapshot",
        }
    }, "explicit_missing_context"


def _kimi_generation_output(
    content: list[Any], tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if content:
        output["content"] = content
    if tool_calls:
        output["tool_calls"] = tool_calls
    if not output:
        output["completion"] = {
            "available": False,
            "reason": "usage_event_without_visible_output",
        }
    return output


def iter_kimi_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    for path, entries in _kimi_wire_sessions():
        session_id = path.parent.name
        root_events = [e for e in entries if _kimi_message(e).get("type") in ("TurnBegin", "TurnEnd")]
        begins = [e for e in root_events if _kimi_message(e).get("type") == "TurnBegin"]
        if not begins:
            first, _ = bounds((timestamp_ns(e.get("timestamp")) for e in entries), file_mtime_ns(path))
            begins = [{"timestamp": first, "message": {"type": "TurnBegin", "payload": {}}}]
        begin_times = [timestamp_ns(e.get("timestamp")) or 0 for e in begins]
        intervals = [
            (start, begin_times[index + 1] if index + 1 < len(begin_times) else 2**63 - 1)
            for index, start in enumerate(begin_times)
        ]
        turn_entries: list[list[dict[str, Any]]] = [[] for _ in intervals]
        for entry in entries:
            turn_entries[_select_interval(timestamp_ns(entry.get("timestamp")), intervals)].append(entry)

        for turn_index, stream in enumerate(turn_entries):
            times = [timestamp_ns(e.get("timestamp")) for e in stream]
            start, end = bounds(times, file_mtime_ns(path))
            root_logical = "root-agent"
            observations: list[TraceObservation] = []
            calls: dict[str, dict[str, Any]] = {}
            results: dict[str, dict[str, Any]] = {}
            subevents: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            pending_partial_calls: list[str] = []
            root_input: Any = None
            output_parts: list[str] = []
            for entry in stream:
                message = _kimi_message(entry)
                payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
                typ = message.get("type")
                if typ == "TurnBegin":
                    root_input = payload.get("user_input")
                elif typ == "ContentPart" and isinstance(payload.get("text"), str):
                    output_parts.append(payload["text"])
                elif typ == "ToolCall" and payload.get("id"):
                    call_id = str(payload["id"])
                    calls[call_id] = {"entry": entry, "payload": payload, "argument_parts": []}
                    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
                    _, quality = _structured_tool_input(function.get("arguments"))
                    if quality in ("empty_source", "unparseable_source_text"):
                        pending_partial_calls.append(call_id)
                elif typ == "ToolCallPart" and isinstance(payload.get("arguments_part"), str):
                    if pending_partial_calls:
                        call_id = pending_partial_calls[-1]
                        calls[call_id]["argument_parts"].append(payload["arguments_part"])
                        _, quality, _ = _kimi_call_tool_input(calls[call_id])
                        if quality == "reconstructed_json_parts":
                            pending_partial_calls.pop()
                elif typ == "ToolResult" and payload.get("tool_call_id"):
                    result_call_id = str(payload["tool_call_id"])
                    results[result_call_id] = {"entry": entry, "payload": payload}
                    pending_partial_calls = [
                        call_id for call_id in pending_partial_calls if call_id != result_call_id
                    ]
                elif typ == "SubagentEvent" and payload.get("agent_id"):
                    subevents[str(payload["agent_id"])].append((entry, payload))

            tool_logical: dict[str, str] = {}
            pending_step_tools: list[TraceObservation] = []
            step_start = start
            step_parts: list[Any] = []
            step_tool_calls: list[dict[str, Any]] = []
            step_usage: dict[str, Any] = {}
            step_input_messages: list[dict[str, Any]] = (
                [{"role": "user", "content": root_input}] if root_input is not None else []
            )
            pending_step_input_messages: list[dict[str, Any]] = []
            seen_step = False
            for entry in stream:
                message = _kimi_message(entry)
                payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
                typ = message.get("type")
                ts = timestamp_ns(entry.get("timestamp")) or step_start
                if typ == "StepBegin":
                    if seen_step and (step_parts or step_tool_calls or step_usage):
                        logical = f"root:generation:{entry.get('_agentstracer_line')}:previous"
                        generation_input, input_quality = _kimi_generation_input(step_input_messages)
                        observations.append(TraceObservation(
                            logical_id=logical, name="generate-kimi-step", as_type="generation",
                            start_ns=step_start, end_ns=ts, parent_logical_id=root_logical,
                            input=generation_input,
                            output=_kimi_generation_output(step_parts, step_tool_calls),
                            usage=_usage(step_usage),
                            metadata={
                                "input_quality": input_quality,
                                "timestamp_quality": "exact_stream_bounds",
                            },
                        ))
                        for tool in pending_step_tools:
                            tool.metadata["requested_by_generation"] = logical
                            tool.links.append(TraceLink(
                                logical, {"agentstracer.relationship": "requested_by"},
                            ))
                        pending_step_tools = []
                    if seen_step:
                        step_input_messages = pending_step_input_messages
                        pending_step_input_messages = []
                    seen_step = True
                    step_start = ts
                    step_parts = []
                    step_tool_calls = []
                    step_usage = {}
                elif typ == "ContentPart":
                    if isinstance(payload.get("text"), str):
                        step_parts.append(payload["text"])
                    elif include_thinking and isinstance(payload.get("think"), str):
                        step_parts.append({"thinking": payload["think"]})
                elif typ == "StatusUpdate" and isinstance(payload.get("token_usage"), dict):
                    step_usage = payload["token_usage"]
                elif typ == "ToolCall" and payload.get("id"):
                    call_id = str(payload["id"])
                    result = results.get(call_id)
                    result_ts = timestamp_ns(result["entry"].get("timestamp")) if result else None
                    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
                    call = calls.get(call_id, {"payload": payload})
                    tool_input, tool_input_quality, tool_input_part_count = _kimi_call_tool_input(call)
                    step_tool_calls.append({
                        "id": call_id,
                        "name": str(function.get("name") or "tool-call"),
                        "arguments": tool_input,
                    })
                    logical = f"root:tool:{call_id}"
                    tool_observation = TraceObservation(
                        logical_id=logical,
                        name=str(function.get("name") or "tool-call"),
                        as_type="tool",
                        start_ns=ts,
                        end_ns=max(ts, result_ts or ts),
                        parent_logical_id=root_logical,
                        input=tool_input,
                        output=(result["payload"].get("return_value") if result else None),
                        level=(
                            "ERROR"
                            if result
                            and isinstance(result["payload"].get("return_value"), dict)
                            and result["payload"]["return_value"].get("is_error")
                            else "DEFAULT"
                        ),
                        metadata={
                            "tool_call_id": call_id,
                            "tool_input_quality": tool_input_quality,
                            "tool_input_part_count": tool_input_part_count,
                            "timestamp_quality": "exact" if result else "open_call",
                        },
                    )
                    observations.append(tool_observation)
                    pending_step_tools.append(tool_observation)
                    tool_logical[call_id] = logical
                elif typ == "ToolResult" and payload.get("tool_call_id"):
                    pending_step_input_messages.append({
                        "role": "tool",
                        "tool_call_id": str(payload["tool_call_id"]),
                        "content": payload.get("return_value"),
                    })
            if step_parts or step_tool_calls or step_usage:
                logical = "root:generation:last"
                generation_input, input_quality = _kimi_generation_input(step_input_messages)
                observations.append(TraceObservation(
                    logical_id=logical,
                    name="generate-kimi-step",
                    as_type="generation",
                    start_ns=step_start,
                    end_ns=end,
                    parent_logical_id=root_logical,
                    input=generation_input,
                    output=_kimi_generation_output(step_parts, step_tool_calls),
                    usage=_usage(step_usage),
                    metadata={
                        "input_quality": input_quality,
                        "timestamp_quality": "exact_stream_bounds",
                    },
                ))
                for tool in pending_step_tools:
                    tool.metadata["requested_by_generation"] = logical
                    tool.links.append(TraceLink(
                        logical, {"agentstracer.relationship": "requested_by"},
                    ))

            for agent_id, agent_events in subevents.items():
                event_times = [timestamp_ns(entry.get("timestamp")) for entry, _ in agent_events]
                agent_start, agent_end = bounds(event_times, start)
                first_payload = agent_events[0][1]
                parent_call = first_payload.get("parent_tool_call_id")
                agent_logical = f"subagent:{agent_id}"
                observations.append(TraceObservation(
                    logical_id=agent_logical,
                    name=f"run-{first_payload.get('subagent_type') or 'kimi-subagent'}",
                    as_type="agent",
                    start_ns=agent_start,
                    end_ns=agent_end,
                    parent_logical_id=tool_logical.get(str(parent_call), root_logical),
                    metadata={
                        "agent_id": agent_id,
                        "subagent_type": first_payload.get("subagent_type"),
                        "parent_tool_call_id": parent_call,
                        "parent_link_quality": "exact" if str(parent_call) in tool_logical else "session_fallback",
                    },
                ))
                sub_calls: dict[str, dict[str, Any]] = {}
                sub_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
                sub_parts: list[Any] = []
                pending_sub_partial_calls: list[str] = []
                for entry, wrapper in agent_events:
                    event = wrapper.get("event") if isinstance(wrapper.get("event"), dict) else {}
                    event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                    event_type = event.get("type")
                    if event_type == "ContentPart":
                        if event_payload.get("text"):
                            sub_parts.append(event_payload["text"])
                        elif include_thinking and event_payload.get("think"):
                            sub_parts.append({"thinking": event_payload["think"]})
                    elif event_type == "ToolCall" and event_payload.get("id"):
                        sub_call_id = str(event_payload["id"])
                        sub_calls[sub_call_id] = {
                            "entry": entry,
                            "payload": event_payload,
                            "argument_parts": [],
                        }
                        function = (
                            event_payload.get("function")
                            if isinstance(event_payload.get("function"), dict) else {}
                        )
                        _, quality = _structured_tool_input(function.get("arguments"))
                        if quality in ("empty_source", "unparseable_source_text"):
                            pending_sub_partial_calls.append(sub_call_id)
                    elif event_type == "ToolCallPart" and isinstance(
                        event_payload.get("arguments_part"), str
                    ):
                        if pending_sub_partial_calls:
                            sub_call_id = pending_sub_partial_calls[-1]
                            sub_calls[sub_call_id]["argument_parts"].append(
                                event_payload["arguments_part"]
                            )
                            _, quality, _ = _kimi_call_tool_input(sub_calls[sub_call_id])
                            if quality == "reconstructed_json_parts":
                                pending_sub_partial_calls.pop()
                    elif event_type == "ToolResult" and event_payload.get("tool_call_id"):
                        result_call_id = str(event_payload["tool_call_id"])
                        sub_results[result_call_id] = (entry, event_payload)
                        pending_sub_partial_calls = [
                            call_id for call_id in pending_sub_partial_calls
                            if call_id != result_call_id
                        ]
                if sub_parts:
                    observations.append(TraceObservation(
                        logical_id=f"{agent_logical}:generation",
                        name="generate-subagent-response",
                        as_type="generation",
                        start_ns=agent_start,
                        end_ns=agent_end,
                        parent_logical_id=agent_logical,
                        input={
                            "context": {
                                "available": False,
                                "reason": "subagent_event_omits_prompt_snapshot",
                            },
                            "parent_tool_call_id": parent_call,
                        },
                        output={"content": sub_parts},
                        metadata={
                            "input_quality": "explicit_missing_context",
                            "timestamp_quality": "event_bounds",
                        },
                    ))
                for call_id, call in sub_calls.items():
                    call_entry = call["entry"]
                    call_payload = call["payload"]
                    result = sub_results.get(call_id)
                    call_ts = timestamp_ns(call_entry.get("timestamp")) or agent_start
                    result_ts = timestamp_ns(result[0].get("timestamp")) if result else None
                    function = call_payload.get("function") if isinstance(call_payload.get("function"), dict) else {}
                    tool_input, tool_input_quality, tool_input_part_count = _kimi_call_tool_input(call)
                    observations.append(TraceObservation(
                        logical_id=f"{agent_logical}:tool:{call_id}",
                        name=str(function.get("name") or "subagent-tool"),
                        as_type="tool",
                        start_ns=call_ts,
                        end_ns=max(call_ts, result_ts or call_ts),
                        parent_logical_id=agent_logical,
                        input=tool_input,
                        output=result[1].get("return_value") if result else None,
                        metadata={
                            "tool_call_id": call_id,
                            "tool_input_quality": tool_input_quality,
                            "tool_input_part_count": tool_input_part_count,
                            "timestamp_quality": "exact" if result else "open_call",
                        },
                    ))

            root = TraceObservation(
                logical_id=root_logical,
                name="run-kimi-turn",
                as_type="agent",
                start_ns=start,
                end_ns=max([end, *(o.end_ns for o in observations)]),
                input=root_input,
                output="".join(output_parts) or None,
                metadata={"turn_index": turn_index, "timestamp_quality": "wire_exact"},
            )
            yield AgentTrace(
                logical_id=f"{session_id}:turn:{turn_index}",
                name="kimi-agent-turn",
                source="kimi",
                source_session_id=session_id,
                project=path.parent.parent.name,
                observations=[root, *observations],
                metadata={"adapter": "kimi-wire-v2", "turn_index": turn_index},
            )


# ---------------------------------------------------------------------------
# OpenClaw


def _openclaw_session_files() -> list[Path]:
    if not OPENCLAW_AGENTS_DIR.exists():
        return []
    files: list[Path] = []
    for agent_dir in OPENCLAW_AGENTS_DIR.iterdir():
        session_dir = agent_dir / "sessions"
        if session_dir.is_dir():
            files.extend(
                path for path in session_dir.glob("*.jsonl*")
                if ".trajectory.jsonl" not in path.name and path.is_file()
            )
    return sorted(files)


def _openclaw_raw_observations(
    entries: list[dict[str, Any]], root: str, include_thinking: bool,
) -> tuple[list[TraceObservation], Any, Any]:
    observations: list[TraceObservation] = []
    result_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in entries:
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        if entry.get("type") == "message" and message.get("role") == "toolResult" and message.get("toolCallId"):
            result_map[str(message["toolCallId"])] = (entry, message)
    previous_ts: int | None = None
    trace_input: Any = None
    trace_output: Any = None
    pending_generation_input: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        role = message.get("role")
        ts = timestamp_ns(message.get("timestamp")) or timestamp_ns(entry.get("timestamp"))
        if ts is None:
            continue
        if role == "user":
            text = _content_text(message.get("content"), "text")
            if trace_input is None and text:
                trace_input = text
            pending_generation_input = [{
                "role": "user",
                "content": text if text is not None else message.get("content"),
            }]
            previous_ts = ts
            continue
        if role == "toolResult":
            pending_generation_input.append({
                "role": "tool",
                "tool_call_id": message.get("toolCallId"),
                "content": message.get("content"),
                "is_error": message.get("isError"),
            })
            previous_ts = ts
            continue
        if role != "assistant":
            continue
        output: dict[str, Any] = {}
        text = _content_text(message.get("content"), "text")
        if text:
            output["content"] = text
            trace_output = text
        if include_thinking:
            thinking = _content_text(message.get("content"), "thinking")
            if thinking:
                output["thinking"] = thinking
        content = message.get("content")
        tool_blocks = [
            block for block in content
            if isinstance(block, dict) and block.get("type") == "toolCall"
        ] if isinstance(content, list) else []
        tool_call_outputs: list[dict[str, Any]] = []
        normalized_tool_inputs: dict[str, tuple[dict[str, Any], str]] = {}
        for block in tool_blocks:
            call_id = str(block.get("id") or _entry_id("openclaw-tool", entry))
            tool_input, tool_input_quality = _structured_tool_input(block.get("arguments"))
            normalized_tool_inputs[call_id] = (tool_input, tool_input_quality)
            tool_call_outputs.append({
                "id": call_id,
                "name": str(block.get("name") or "tool-call"),
                "arguments": tool_input,
            })
        if tool_call_outputs:
            output["tool_calls"] = tool_call_outputs
        if not output:
            output["completion"] = {
                "available": False,
                "reason": "assistant_message_omits_visible_output",
            }
        generation_input, input_quality = (
            ({"messages": pending_generation_input}, "source_messages")
            if pending_generation_input
            else ({
                "context": {
                    "available": False,
                    "reason": "source_log_omits_prompt_snapshot",
                }
            }, "explicit_missing_context")
        )
        generation = _entry_id("openclaw:generation", entry)
        observations.append(TraceObservation(
            logical_id=generation,
            name="generate-openclaw-response",
            as_type="generation",
            start_ns=previous_ts if previous_ts is not None and previous_ts <= ts else ts,
            end_ns=ts,
            parent_logical_id=root,
            input=generation_input,
            output=output,
            model=message.get("model"),
            usage=_usage(message.get("usage")),
            level="ERROR" if message.get("errorMessage") else "DEFAULT",
            status_message=message.get("errorMessage"),
            metadata={
                "provider": message.get("provider"),
                "stop_reason": message.get("stopReason"),
                "input_quality": input_quality,
            },
        ))
        if tool_blocks:
            for block in tool_blocks:
                call_id = str(block.get("id") or _entry_id("openclaw-tool", entry))
                tool_input, tool_input_quality = normalized_tool_inputs[call_id]
                result = result_map.get(call_id)
                result_ts = (
                    timestamp_ns(result[1].get("timestamp")) or timestamp_ns(result[0].get("timestamp"))
                    if result else None
                )
                is_error = bool(result and result[1].get("isError"))
                observations.append(TraceObservation(
                    logical_id=f"openclaw:tool:{call_id}",
                    name=str(block.get("name") or "tool-call"),
                    as_type="tool",
                    start_ns=ts,
                    end_ns=max(ts, result_ts or ts),
                    parent_logical_id=root,
                    input=tool_input,
                    output=result[1].get("content") if result else None,
                    level="ERROR" if is_error else "DEFAULT",
                    status_message="tool result marked as error" if is_error else None,
                    metadata={
                        "tool_call_id": call_id,
                        "tool_input_quality": tool_input_quality,
                        "requested_by_generation": generation,
                        "timestamp_quality": "exact" if result else "open_call",
                    },
                    links=[TraceLink(
                        generation, {"agentstracer.relationship": "requested_by"},
                    )],
                ))
        pending_generation_input = []
        previous_ts = ts
    return observations, trace_input, trace_output


def iter_openclaw_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    trajectory_candidates: dict[str, list[Path]] = defaultdict(list)
    if OPENCLAW_AGENTS_DIR.exists():
        for path in sorted(OPENCLAW_AGENTS_DIR.rglob("*.trajectory.jsonl*")):
            entries = _jsonl(path)
            for trace_id in {
                str(entry["traceId"])
                for entry in entries
                if entry.get("traceId")
            }:
                trajectory_candidates[trace_id].append(path)

    # A trajectory may have active and archived copies.  Read only the best
    # copy so stale deleted events cannot overwrite newer active events.
    trajectory_paths: dict[str, Path] = {}
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for trace_id, paths in trajectory_candidates.items():
        selected = min(
            paths,
            key=lambda path: (
                1 if ".jsonl.reset." in path.name else 2 if ".jsonl.deleted." in path.name else 0,
                path.name,
            ),
        )
        trajectory_paths[trace_id] = selected
        trajectories[trace_id] = [
            entry for entry in _jsonl(selected)
            if str(entry.get("traceId")) == trace_id
        ]

    raw_records: dict[str, tuple[Path, list[dict[str, Any]], str, str]] = {}
    raw_by_filename: dict[str, str] = {}
    raw_by_native: dict[str, list[str]] = defaultdict(list)
    for path in _openclaw_session_files():
        entries = _jsonl(path)
        if entries and entries[0].get("type") == "session":
            native_id = str(entries[0].get("id") or path.name.split(".jsonl", 1)[0])
            session_id, session_status = _openclaw_session_identity(path, native_id)
            raw_records[session_id] = (path, entries, native_id, session_status)
            raw_by_filename[path.name] = session_id
            raw_by_native[native_id].append(session_id)

    used_raw: set[str] = set()
    for native_trace_id, raw_events in trajectories.items():
        # Deduplicate active/deleted copies using the source sequence number.
        unique: dict[tuple[Any, Any], dict[str, Any]] = {}
        for event in raw_events:
            unique[(event.get("seq"), event.get("type"))] = event
        events = sorted(unique.values(), key=lambda event: (event.get("seq", 0), timestamp_ns(event.get("ts")) or 0))
        if not events:
            continue
        started = next((e for e in events if e.get("type") == "session.started"), events[0])
        ended = next((e for e in reversed(events) if e.get("type") == "session.ended"), events[-1])
        start = timestamp_ns(started.get("ts")) or file_mtime_ns(trajectory_paths[native_trace_id]) or 1
        end = timestamp_ns(ended.get("ts")) or start
        start_data = started.get("data") if isinstance(started.get("data"), dict) else {}
        session_file = start_data.get("sessionFile")
        raw_id = Path(session_file).name.split(".jsonl", 1)[0] if isinstance(session_file, str) else None
        raw_key = raw_by_filename.get(Path(session_file).name) if isinstance(session_file, str) else None
        if raw_key is None and raw_id:
            candidates = raw_by_native.get(str(raw_id), [])
            if candidates:
                raw_key = min(
                    candidates,
                    key=lambda key: (
                        0 if raw_records[key][3] == "active" else 1,
                        raw_records[key][0].name,
                    ),
                )
        raw = raw_records.get(raw_key) if raw_key else None
        root_logical = "root-agent"
        observations: list[TraceObservation] = []
        trace_input: Any = None
        trace_output: Any = None
        if raw:
            used_raw.add(str(raw_key))
            children, trace_input, trace_output = _openclaw_raw_observations(raw[1], root_logical, include_thinking)
            observations.extend(children)
            if children:
                end = max(end, max(child.end_ns for child in children))
        prompt = next((e.get("data") for e in events if e.get("type") == "prompt.submitted"), None)
        completed = next((e.get("data") for e in reversed(events) if e.get("type") == "model.completed"), None)
        ended_data = ended.get("data") if isinstance(ended.get("data"), dict) else {}
        if trace_input is None and isinstance(prompt, dict):
            trace_input = prompt.get("prompt")
        if trace_output is None and isinstance(completed, dict):
            trace_output = completed.get("finalPromptText") or completed.get("assistantTexts")
        status = str(ended_data.get("status") or "")
        failed = bool(ended_data.get("terminalError") or status.lower() in ("error", "failed"))
        root = TraceObservation(
            logical_id=root_logical,
            name="run-openclaw-agent",
            as_type="agent",
            start_ns=start,
            end_ns=max(start, end),
            input=trace_input,
            output=trace_output,
            level="ERROR" if failed else "DEFAULT",
            status_message=ended_data.get("terminalError") if failed else None,
            metadata={
                "native_trace_id": native_trace_id,
                "run_id": started.get("runId"),
                "agent_id": start_data.get("agentId"),
                "trigger": start_data.get("trigger"),
                "workspace_dir": start_data.get("workspaceDir"),
                "status": status,
                "aborted": ended_data.get("aborted"),
                "timed_out": ended_data.get("timedOut"),
                "timestamp_quality": "trajectory_exact",
            },
        )
        session_id = str(raw_key or raw_id or started.get("sessionId") or native_trace_id)
        yield AgentTrace(
            logical_id=f"openclaw:{native_trace_id}",
            name="openclaw-agent-run",
            source="openclaw",
            source_session_id=session_id,
            project=str(start_data.get("workspaceDir") or "openclaw"),
            observations=[root, *observations],
            metadata={"adapter": "openclaw-trajectory-v2"},
        )

    # Older sessions may have no trajectory companion; preserve them through
    # a raw-session fallback instead of silently dropping them.
    for session_id, (path, entries, native_id, session_status) in raw_records.items():
        if session_id in used_raw:
            continue
        times = [timestamp_ns(e.get("timestamp")) for e in entries]
        start, end = bounds(times, file_mtime_ns(path))
        children, trace_input, trace_output = _openclaw_raw_observations(entries, "root-agent", include_thinking)
        if children:
            end = max(end, max(child.end_ns for child in children))
        header = entries[0]
        yield AgentTrace(
            logical_id=f"openclaw-raw:{session_id}",
            name="openclaw-agent-session",
            source="openclaw",
            source_session_id=session_id,
            project=str(header.get("cwd") or "openclaw"),
            observations=[TraceObservation(
                logical_id="root-agent", name="run-openclaw-agent", as_type="agent",
                start_ns=start, end_ns=end, input=trace_input, output=trace_output,
                metadata={
                    "timestamp_quality": "message_exact",
                    "session_status": session_status,
                    "native_session_id": native_id,
                },
            ), *children],
            metadata={"adapter": "openclaw-raw-v2"},
        )


# ---------------------------------------------------------------------------
# OpenCode and Gemini (smaller, but still preserve native parent/timing fields)


def _load_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def iter_opencode_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    if not OPENCODE_DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(f"file:{OPENCODE_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        sessions = conn.execute("SELECT * FROM session ORDER BY time_created, id").fetchall()
    except sqlite3.Error:
        return
    try:
        rows_by_id = {str(row["id"]): row for row in sessions}

        def root_of(session_id: str) -> str:
            seen: set[str] = set()
            current = session_id
            while current not in seen:
                seen.add(current)
                parent = rows_by_id[current]["parent_id"]
                if not parent or str(parent) not in rows_by_id:
                    return current
                current = str(parent)
            return session_id

        groups: dict[str, list[str]] = defaultdict(list)
        for session_id in rows_by_id:
            groups[root_of(session_id)].append(session_id)
        for root_id, session_ids in groups.items():
            observations: list[TraceObservation] = []
            for session_id in session_ids:
                row = rows_by_id[session_id]
                agent_logical = f"session:{session_id}"
                parent_id = row["parent_id"]
                start = timestamp_ns(row["time_created"]) or 1
                end = timestamp_ns(row["time_updated"]) or start
                messages = conn.execute(
                    "SELECT * FROM message WHERE session_id=? ORDER BY time_created,id", (session_id,)
                ).fetchall()
                first_user: Any = None
                last_output: Any = None
                child_observations: list[TraceObservation] = []
                previous_ts = start
                pending_generation_input: list[dict[str, Any]] = []
                for message_row in messages:
                    data = _load_json_object(message_row["data"])
                    role = data.get("role")
                    msg_start = timestamp_ns((data.get("time") or {}).get("created")) or timestamp_ns(message_row["time_created"]) or previous_ts
                    msg_end = timestamp_ns((data.get("time") or {}).get("completed")) or timestamp_ns(message_row["time_updated"]) or msg_start
                    parts = [
                        _load_json_object(part["data"])
                        for part in conn.execute("SELECT * FROM part WHERE message_id=? ORDER BY time_created,id", (message_row["id"],))
                    ]
                    text = "\n\n".join(str(p.get("text")) for p in parts if p.get("type") == "text" and p.get("text")) or None
                    if role == "user":
                        first_user = first_user or text
                        pending_generation_input = [{
                            "role": "user",
                            "content": text if text is not None else {"parts": parts},
                        }]
                        previous_ts = msg_end
                        continue
                    if role != "assistant":
                        continue
                    reasoning = "\n\n".join(str(p.get("text")) for p in parts if p.get("type") == "reasoning" and p.get("text")) or None
                    output: dict[str, Any] = {}
                    if text:
                        output["content"] = text
                        last_output = text
                    if include_thinking and reasoning:
                        output["thinking"] = reasoning
                    tool_parts = [part for part in parts if part.get("type") == "tool"]
                    normalized_tool_inputs: dict[str, tuple[dict[str, Any], str]] = {}
                    tool_call_outputs: list[dict[str, Any]] = []
                    for part in tool_parts:
                        state = part.get("state") if isinstance(part.get("state"), dict) else {}
                        call_id = str(part.get("callID") or part.get("id"))
                        tool_input, tool_input_quality = _structured_tool_input(state.get("input"))
                        normalized_tool_inputs[call_id] = (tool_input, tool_input_quality)
                        tool_call_outputs.append({
                            "id": call_id,
                            "name": str(part.get("tool") or "tool-call"),
                            "arguments": tool_input,
                        })
                    if tool_call_outputs:
                        output["tool_calls"] = tool_call_outputs
                    if not output:
                        output["completion"] = {
                            "available": False,
                            "reason": "assistant_message_omits_visible_output",
                        }
                    generation_input, input_quality = (
                        ({"messages": pending_generation_input}, "source_messages")
                        if pending_generation_input
                        else ({
                            "context": {
                                "available": False,
                                "reason": "source_log_omits_prompt_snapshot",
                            }
                        }, "explicit_missing_context")
                    )
                    generation = f"{agent_logical}:generation:{message_row['id']}"
                    model = data.get("model") if isinstance(data.get("model"), dict) else {}
                    child_observations.append(TraceObservation(
                        logical_id=generation,
                        name="generate-opencode-response",
                        as_type="generation",
                        start_ns=msg_start,
                        end_ns=max(msg_start, msg_end),
                        parent_logical_id=agent_logical,
                        input=generation_input,
                        output=output,
                        model="/".join(filter(None, (model.get("providerID"), model.get("modelID")))) or None,
                        usage=_usage(data.get("tokens")),
                        cost={"total": data.get("cost")} if isinstance(data.get("cost"), (int, float)) else {},
                        metadata={
                            "finish": data.get("finish"),
                            "agent": data.get("agent"),
                            "input_quality": input_quality,
                        },
                    ))
                    next_generation_input: list[dict[str, Any]] = []
                    for part in tool_parts:
                        state = part.get("state") if isinstance(part.get("state"), dict) else {}
                        call_id = str(part.get("callID") or part.get("id"))
                        tool_input, tool_input_quality = normalized_tool_inputs[call_id]
                        tool_start = timestamp_ns((state.get("time") or {}).get("start")) or msg_start
                        tool_end = timestamp_ns((state.get("time") or {}).get("end")) or msg_end
                        status = state.get("status")
                        child_observations.append(TraceObservation(
                            logical_id=f"{agent_logical}:tool:{part.get('callID') or part.get('id')}",
                            name=str(part.get("tool") or "tool-call"),
                            as_type="tool",
                            start_ns=tool_start,
                            end_ns=max(tool_start, tool_end),
                            parent_logical_id=agent_logical,
                            input=tool_input,
                            output=state.get("output"),
                            level="ERROR" if status in ("error", "failed") else "DEFAULT",
                            status_message=str(state.get("error")) if state.get("error") else None,
                            metadata={
                                "status": status,
                                "tool_call_id": part.get("callID"),
                                "tool_input_quality": tool_input_quality,
                                "requested_by_generation": generation,
                            },
                            links=[TraceLink(
                                generation, {"agentstracer.relationship": "requested_by"},
                            )],
                        ))
                        if state.get("output") is not None or state.get("error") is not None:
                            next_generation_input.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": state.get("output"),
                                "error": state.get("error"),
                            })
                    pending_generation_input = next_generation_input
                    previous_ts = msg_end
                observations.append(TraceObservation(
                    logical_id=agent_logical,
                    name="run-opencode-agent" if session_id == root_id else "run-opencode-subagent",
                    as_type="agent",
                    start_ns=start,
                    end_ns=max(end, *(o.end_ns for o in child_observations)) if child_observations else end,
                    parent_logical_id=f"session:{parent_id}" if parent_id and str(parent_id) in rows_by_id else None,
                    input=first_user,
                    output=last_output,
                    metadata={"native_session_id": session_id, "agent": row["agent"], "cwd": row["directory"]},
                ))
                observations.extend(child_observations)
            root_row = rows_by_id[root_id]
            yield AgentTrace(
                logical_id=f"opencode:{root_id}", name="opencode-agent-session", source="opencode",
                source_session_id=root_id, project=str(root_row["directory"]), observations=observations,
                metadata={"adapter": "opencode-v2", "session_count": len(session_ids)},
            )
    finally:
        conn.close()


def iter_gemini_traces(include_thinking: bool = True) -> Iterator[AgentTrace]:
    if not GEMINI_DIR.exists():
        return
    for path in sorted(GEMINI_DIR.glob("*/chats/session-*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        session_id = str(data.get("sessionId") or path.stem)
        start = timestamp_ns(data.get("startTime")) or file_mtime_ns(path) or 1
        end = timestamp_ns(data.get("lastUpdated")) or start
        observations: list[TraceObservation] = []
        first_input: Any = None
        last_output: Any = None
        previous_ts = start
        pending_generation_input: list[dict[str, Any]] = []
        for index, message in enumerate(data.get("messages", [])):
            if not isinstance(message, dict):
                continue
            ts = timestamp_ns(message.get("timestamp")) or previous_ts
            if message.get("type") == "user":
                user_content = _content_text(message.get("content"), "text")
                first_input = first_input or user_content
                pending_generation_input = [{
                    "role": "user",
                    "content": user_content if user_content is not None else message.get("content"),
                }]
                previous_ts = ts
                continue
            if message.get("type") != "gemini":
                continue
            output: dict[str, Any] = {}
            text = _content_text(message.get("content"), "text")
            if text:
                output["content"] = text
                last_output = text
            if include_thinking and message.get("thoughts"):
                output["thinking"] = message.get("thoughts")
            tool_calls = [call for call in message.get("toolCalls", []) if isinstance(call, dict)]
            normalized_tool_inputs: list[tuple[dict[str, Any], str]] = []
            tool_call_outputs: list[dict[str, Any]] = []
            for call_index, call in enumerate(tool_calls):
                tool_input, tool_input_quality = _structured_tool_input(call.get("args"))
                normalized_tool_inputs.append((tool_input, tool_input_quality))
                tool_call_outputs.append({
                    "id": call.get("id") or f"gemini:{index}:{call_index}",
                    "name": str(call.get("name") or "tool-call"),
                    "arguments": tool_input,
                })
            if tool_call_outputs:
                output["tool_calls"] = tool_call_outputs
            if not output:
                output["completion"] = {
                    "available": False,
                    "reason": "assistant_message_omits_visible_output",
                }
            generation_input, input_quality = (
                ({"messages": pending_generation_input}, "source_messages")
                if pending_generation_input
                else ({
                    "context": {
                        "available": False,
                        "reason": "source_log_omits_prompt_snapshot",
                    }
                }, "explicit_missing_context")
            )
            generation = f"gemini:generation:{index}"
            observations.append(TraceObservation(
                logical_id=generation, name="generate-gemini-response", as_type="generation",
                start_ns=min(previous_ts, ts), end_ns=ts, parent_logical_id="root-agent",
                input=generation_input, output=output,
                model=message.get("model"), usage=_usage(message.get("tokens")),
                metadata={
                    "input_quality": input_quality,
                    "timestamp_quality": "end_exact_start_inferred",
                },
            ))
            next_generation_input: list[dict[str, Any]] = []
            for call_index, call in enumerate(tool_calls):
                tool_input, tool_input_quality = normalized_tool_inputs[call_index]
                observations.append(TraceObservation(
                    logical_id=f"gemini:tool:{index}:{call_index}",
                    name=str(call.get("name") or "tool-call"), as_type="tool",
                    start_ns=ts, end_ns=ts, parent_logical_id="root-agent",
                    input=tool_input, output=call.get("result"),
                    level="ERROR" if call.get("status") in ("error", "failed") else "DEFAULT",
                    metadata={
                        "status": call.get("status"),
                        "tool_input_quality": tool_input_quality,
                        "requested_by_generation": generation,
                        "timestamp_quality": "message_only",
                    },
                    links=[TraceLink(
                        generation, {"agentstracer.relationship": "requested_by"},
                    )],
                ))
                if call.get("result") is not None:
                    next_generation_input.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or f"gemini:{index}:{call_index}",
                        "content": call.get("result"),
                    })
            pending_generation_input = next_generation_input
            previous_ts = ts
        if observations:
            end = max(end, max(o.end_ns for o in observations))
        yield AgentTrace(
            logical_id=f"gemini:{session_id}", name="gemini-agent-session", source="gemini",
            source_session_id=session_id, project=path.parent.parent.name,
            observations=[TraceObservation(
                logical_id="root-agent", name="run-gemini-agent", as_type="agent",
                start_ns=start, end_ns=end, input=first_input, output=last_output,
                metadata={"timestamp_quality": "session_exact"},
            ), *observations],
            metadata={"adapter": "gemini-v2"},
        )


ADAPTERS = {
    "claude": iter_claude_traces,
    "codex": iter_codex_traces,
    "gemini": iter_gemini_traces,
    "kimi": iter_kimi_traces,
    "opencode": iter_opencode_traces,
    "openclaw": iter_openclaw_traces,
}


def iter_agent_traces(
    sources: Iterable[str] | None = None,
    *,
    include_thinking: bool = True,
) -> Iterator[AgentTrace]:
    selected = tuple(sources or TRACE_SOURCES)
    unknown = sorted(set(selected) - set(ADAPTERS))
    if unknown:
        raise ValueError(f"unsupported trace sources: {', '.join(unknown)}")
    for source in selected:
        yield from ADAPTERS[source](include_thinking=include_thinking)
