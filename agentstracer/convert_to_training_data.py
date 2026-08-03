"""Convert agentstracer session exports to provider-agnostic training format.

Handles:
- User message envelope stripping (Sender metadata, System async echoes, timestamps)
- [[reply_to_current]] protocol marker removal
- Async exec->process(poll) collapsing into single tool_call/tool_result pairs
- Thinking + narration merging into reasoning
- Infrastructure parameter stripping from tool inputs
- Tool output cleaning (TUI box-drawing)

Output: one JSONL line per turn (user message -> agent loop -> reply).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# User message cleaning
# ---------------------------------------------------------------------------

# Matches the full OpenClaw envelope:
#   [optional System: ... prefix]
#   Sender (untrusted metadata):
#   ```json { ... } ```
#   [timestamp] <actual text>
_SYSTEM_PREFIX_RE = re.compile(
    r"^System:\s*\[.*?\]\s*Exec completed\s*\(.*?\)\s*::.*?(?=Sender \(untrusted metadata\)|$)",
    re.DOTALL,
)
_SENDER_BLOCK_RE = re.compile(
    r"Sender \(untrusted metadata\):\s*```json\s*\{.*?\}\s*```\s*",
    re.DOTALL,
)
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\[.*?\]\s*",
)


def extract_user_text(content: str) -> str:
    """Strip OpenClaw envelope from user message, return clean text."""
    text = content
    text = _SYSTEM_PREFIX_RE.sub("", text).strip()

    # Only strip timestamp prefix if we actually removed an OpenClaw envelope
    had_envelope = bool(_SENDER_BLOCK_RE.search(text))
    text = _SENDER_BLOCK_RE.sub("", text).strip()
    if had_envelope:
        text = _TIMESTAMP_PREFIX_RE.sub("", text).strip()

    return text


# ---------------------------------------------------------------------------
# Tool input cleaning
# ---------------------------------------------------------------------------

_EXEC_INFRA_KEYS = {"workdir", "yieldMs", "timeout", "elevated", "host", "security", "ask"}
_PROCESS_INFRA_KEYS = {"timeout", "offset", "limit"}


def clean_tool_input(tool_name: str, raw_input: dict) -> dict:
    """Remove infrastructure-only parameters from tool inputs."""
    if tool_name == "exec":
        return {k: v for k, v in raw_input.items() if k not in _EXEC_INFRA_KEYS}
    if tool_name == "process":
        return {k: v for k, v in raw_input.items() if k not in _PROCESS_INFRA_KEYS}
    return raw_input


# ---------------------------------------------------------------------------
# Tool output cleaning
# ---------------------------------------------------------------------------

_BOX_CHARS = set("─│┌┐└┘├┤┬┴┼╭╮╰╯╱╲╳═║╔╗╚╝╠╣╦╩╬◇◆●○■□▪▫▸▹►◂◃◄")
_BOX_LEADER_RE = re.compile(r"^[│┃┆┇┊┋]+\s?")
_BOX_TRAILER_RE = re.compile(r"\s?[│┃┆┇┊┋]+$")


def clean_tool_output(text: str) -> str:
    """Clean TUI box-drawing formatting from tool outputs.

    Strips lines that are purely decorative (>80% box-drawing chars) and
    removes box-drawing leaders/trailers from content lines while
    preserving indentation and blank lines.
    """
    if not text:
        return text
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are purely box-drawing decoration
        if stripped and len(stripped) > 3:
            box_count = sum(1 for c in stripped if c in _BOX_CHARS)
            if box_count / len(stripped) > 0.8:
                continue
        # Strip box-drawing leaders/trailers but keep indentation
        content = _BOX_LEADER_RE.sub("", line)
        content = _BOX_TRAILER_RE.sub("", content)
        cleaned.append(content)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Async exec->process collapsing
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"Command still running \(session (\S+),")


def _is_still_running(output: dict | None) -> str | None:
    """If tool output is 'still running', return the session ID. Else None."""
    if not output:
        return None
    text = output.get("text") or ""
    m = _SESSION_ID_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------

def group_turns(msgs: list[dict]) -> list[dict]:
    """Group messages into turns: user -> WORK* -> REPLY.

    Each turn has:
      user_msg: the user message dict
      work_msgs: list of intermediate assistant messages (WORK)
      reply_msg: the final assistant message (REPLY), or None
    """
    def finish(turn: dict | None) -> None:
        if turn is None:
            return
        # OpenClaw marks replies explicitly.  Other providers generally do
        # not, so a final content-only assistant message is the reply when it
        # is the last assistant record in the turn.
        if turn["reply_msg"] is None and turn["work_msgs"]:
            candidate = turn["work_msgs"][-1]
            if (
                str(candidate.get("content") or "").strip()
                and not candidate.get("tool_uses")
            ):
                turn["reply_msg"] = turn["work_msgs"].pop()
        turns.append(turn)

    turns: list[dict] = []
    current: dict | None = None
    for msg in msgs:
        role = msg.get("role")
        if role == "user":
            finish(current)
            current = {"user_msg": msg, "work_msgs": [], "reply_msg": None}
        elif role == "assistant" and current is not None:
            content = msg.get("content") or ""
            if "[[reply_to_current]]" in content:
                current["reply_msg"] = msg
            else:
                current["work_msgs"].append(msg)
    finish(current)
    return turns


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def _make_tc_id(namespace: str, ordinal: int, tool_name: str) -> str:
    encoded = json.dumps(
        [namespace, ordinal, tool_name], ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "tc_" + hashlib.sha256(encoded).hexdigest()[:16]


def _result_status(
    declared_status: Any,
    raw_output: dict | None,
    *,
    default: str,
) -> str:
    raw_status = raw_output.get("status") if isinstance(raw_output, dict) else None
    value = str(raw_status or declared_status or "").strip().lower()
    if value in {"error", "failed", "failure", "cancelled", "canceled", "timeout"}:
        return "error"
    if value in {"success", "succeeded", "completed", "complete", "ok"}:
        return "success"
    return default


def build_output_sequence(
    work_msgs: list[dict],
    reply_msg: dict | None,
    *,
    id_namespace: str = "training-turn",
) -> list[dict]:
    """Convert WORK messages + REPLY into a flat output sequence.

    Collapses exec->process(poll) async chains into single tool_call/tool_result.
    Merges thinking + narration into reasoning blocks.
    """
    output: list[dict] = []
    pending_async: dict[str, tuple[str, dict]] = {}
    tool_ordinal = 0

    for msg in work_msgs:
        thinking = msg.get("thinking") or ""
        content = msg.get("content") or ""
        tool_uses = msg.get("tool_uses") or []

        # Merge thinking + narration into reasoning
        reasoning_parts = []
        if thinking:
            reasoning_parts.append(thinking.strip())
        if content:
            reasoning_parts.append(content.strip())
        if reasoning_parts:
            output.append({"type": "reasoning", "text": "\n\n".join(reasoning_parts)})

        for tu in tool_uses:
            tool_name = tu.get("tool", "unknown")
            raw_input = tu.get("input", {})
            raw_output = tu.get("output")
            status = tu.get("status", "unknown")

            # Process(poll/log) resolving a pending async exec
            if tool_name == "process" and raw_input.get("sessionId"):
                session_id = raw_input["sessionId"]
                if session_id in pending_async:
                    tc_id, clean_input = pending_async[session_id]
                    result_text = ""
                    if raw_output:
                        result_text = raw_output.get("text") or ""
                    next_session_id = _is_still_running(raw_output)
                    if next_session_id:
                        if next_session_id != session_id:
                            pending_async.pop(session_id)
                            pending_async[next_session_id] = (tc_id, clean_input)
                        continue
                    pending_async.pop(session_id)
                    if result_text and f"No session found for {session_id}" not in result_text:
                        output.append({
                            "type": "tool_result", "id": tc_id,
                            "output": clean_tool_output(result_text),
                            "status": _result_status(status, raw_output, default="success"),
                        })
                    else:
                        output.append({
                            "type": "tool_result", "id": tc_id,
                            "output": None, "status": "lost",
                        })
                    continue
                # Orphaned process call (not matching a pending async) — skip it
                continue

            # Regular tool call
            tc_id = _make_tc_id(id_namespace, tool_ordinal, tool_name)
            tool_ordinal += 1
            clean_input = clean_tool_input(tool_name, raw_input)
            output.append({
                "type": "tool_call", "id": tc_id,
                "name": tool_name, "arguments": clean_input,
            })

            async_sid = _is_still_running(raw_output)
            if async_sid:
                pending_async[async_sid] = (tc_id, clean_input)
            else:
                result_text = (raw_output.get("text") or "") if raw_output else ""
                output.append({
                    "type": "tool_result", "id": tc_id,
                    "output": clean_tool_output(result_text),
                    "status": _result_status(status, raw_output, default="unknown"),
                })

    # Unresolved async execs
    for _session_id, (tc_id, _) in pending_async.items():
        output.append({"type": "tool_result", "id": tc_id, "output": None, "status": "lost"})

    # REPLY message
    if reply_msg:
        thinking = reply_msg.get("thinking") or ""
        content = reply_msg.get("content") or ""
        if thinking:
            output.append({"type": "reasoning", "text": thinking.strip()})
        reply_text = content.replace("[[reply_to_current]]", "").strip()
        if reply_text:
            output.append({"type": "message", "content": reply_text})

    return output


# ---------------------------------------------------------------------------
# Session stats extraction
# ---------------------------------------------------------------------------

def _extract_session_stats(session: dict) -> dict:
    """Extract session-level stats useful for filtering/analysis."""
    raw_stats = session.get("stats")
    nested: dict[Any, Any] = raw_stats if isinstance(raw_stats, dict) else {}

    def value(key: str) -> Any:
        return nested.get(key, session.get(key))

    return {
        "user_messages": value("user_messages"),
        "assistant_messages": value("assistant_messages"),
        "tool_uses": value("tool_uses"),
        "input_tokens": value("input_tokens"),
        "output_tokens": value("output_tokens"),
        "duration_seconds": value("duration_seconds"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_session(session: dict) -> list[dict]:
    """Convert one agentstracer session into a list of training turns."""
    msgs = session.get("messages", [])
    turns = group_turns(msgs)
    result = []

    session_id = session.get("session_id", "unknown")
    model = session.get("model", "unknown")
    source = session.get("source", "unknown")
    session_stats = _extract_session_stats(session)

    for turn_idx, turn in enumerate(turns):
        user_text = extract_user_text(turn["user_msg"].get("content") or "")
        if not user_text:
            continue
        output_seq = build_output_sequence(
            turn["work_msgs"], turn["reply_msg"],
            id_namespace=f"{session_id}:{turn_idx}",
        )
        if not output_seq:
            continue

        result.append({
            "turn_id": f"{session_id}_{turn_idx:03d}",
            "session_id": session_id,
            "turn_index": turn_idx,
            "model": model,
            "source": source,
            "input": {"role": "user", "content": user_text},
            "output": output_seq,
            "session_stats": session_stats,
            "metadata": {"timestamp": turn["user_msg"].get("timestamp")},
        })

    return result


def convert_sessions_to_training(
    sessions: Iterable[dict],
    output_path: Path,
) -> dict[str, Any]:
    """Convert sessions JSONL to training-format JSONL.

    Returns summary dict with counts.
    """
    total_turns = 0
    total_sessions = 0
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for session in sessions:
                total_sessions += 1
                turns = convert_session(session)
                for turn in turns:
                    out.write(json.dumps(turn, ensure_ascii=False) + "\n")
                    total_turns += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, output_path)
        os.chmod(output_path, 0o600)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "sessions": total_sessions,
        "turns": total_turns,
        "output": str(output_path),
    }
