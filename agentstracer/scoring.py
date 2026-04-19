"""Structured scoring pipeline for agentic traces.

Implements: Format → Judge → Store
See docs/scoring-algorithm.md for the full specification.

All scoring judgment lives in the rubric (skills/agentstracer-score/RUBRIC.md).
Python code handles formatting, calling the judge, and storing results. Zero scoring logic.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import (
    BACKEND_CHOICES,
    BACKEND_COMMANDS,
    BACKEND_COMMAND_ALIASES,
    BACKEND_ENV_MARKERS,
    SUPPORTED_BACKENDS,
    check_backend_runtime as _check_backend_runtime,
    detect_current_agent,
    format_codex_runtime_error as _format_codex_runtime_error,
    require_backend_command as _require_backend_command,
    resolve_backend,
    summarize_process_error as _summarize_process_error,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single tool-call cycle within a segment."""
    plan: str              # assistant text before tool call
    action_tool: str       # tool name
    action_input: str      # first arg / summary of input
    result_output: str     # tool output (may be truncated)
    result_status: str     # "success", "error", "failure", ""
    reflect: str           # assistant text after result (may be empty)


@dataclass
class Segment:
    """A block of agent work bounded by user messages."""
    user_message: str
    steps: list[Step]
    user_response: str | None = None   # next user message, or None
    judge_result: dict | None = None


@dataclass
class ScoringResult:
    """Final scoring output for one session."""
    segments: list[Segment]
    quality: int                 # 1-5, from judge
    reason: str                  # judge's reasoning
    display_title: str = ""              # LLM-generated concise title
    task_type: str = "unknown"           # LLM-classified task type
    outcome_label: str = "unknown"       # LLM-classified outcome
    value_labels: list[str] = field(default_factory=list)  # LLM-classified value signals
    risk_level: list[str] = field(default_factory=list)     # LLM-classified risk signals
    taste_signals: list[dict] = field(default_factory=list)
    detail_json: str = "{}"


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def get_message_text(msg: dict) -> str:
    """Extract text content from a message dict."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                return block
            if isinstance(block, dict) and block.get("text"):
                return block["text"]
    return ""


def extract_tool_uses(msg: dict) -> list[dict]:
    """Extract tool uses from a message, handling both parsed and raw formats."""
    tool_uses = msg.get("tool_uses", [])
    if tool_uses:
        return tool_uses
    content = msg.get("content")
    if isinstance(content, list):
        uses = []
        for block in content:
            if isinstance(block, dict) and block.get("tool"):
                inp = block.get("input", {})
                first_arg = ""
                if isinstance(inp, dict):
                    for v in inp.values():
                        if isinstance(v, str) and v.strip():
                            first_arg = v.strip()
                            break
                uses.append({
                    "tool": block["tool"],
                    "input": inp,
                    "output": block.get("output", ""),
                    "status": block.get("status", ""),
                    "first_arg": first_arg,
                })
        return uses
    return []


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _first_input_value(inp: dict | Any) -> str:
    """Return the first string value from a tool input dict."""
    if isinstance(inp, dict):
        for v in inp.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(inp, str):
        return inp.strip()
    return ""


# ---------------------------------------------------------------------------
# Format: parse messages into turns and build the judge prompt
# ---------------------------------------------------------------------------

def segment_session(messages: list[dict]) -> list[Segment]:
    """Split a message list into Segments bounded by user messages.

    Each user message starts a new segment. Within a segment, each tool_use
    in an assistant message becomes a Step. This is purely structural formatting.
    """
    if not messages:
        return []

    segments: list[Segment] = []
    current_user_msg = ""
    current_steps: list[Step] = []
    pending_plan = ""

    def _flush_segment() -> None:
        nonlocal current_user_msg, current_steps, pending_plan
        if current_user_msg or current_steps:
            segments.append(Segment(
                user_message=current_user_msg,
                steps=current_steps,
            ))
        current_user_msg = ""
        current_steps = []
        pending_plan = ""

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            text = get_message_text(msg)
            _flush_segment()
            if segments:
                segments[-1].user_response = text
            current_user_msg = text

        elif role == "assistant":
            text = get_message_text(msg)
            tool_uses = extract_tool_uses(msg)

            if not tool_uses:
                if current_steps:
                    current_steps[-1].reflect = text
                else:
                    pending_plan = text
            else:
                for i, tu in enumerate(tool_uses):
                    plan = text if i == 0 else ""
                    if i == 0 and pending_plan and not text:
                        plan = pending_plan
                        pending_plan = ""

                    output = tu.get("output", "")
                    if isinstance(output, dict):
                        output = json.dumps(output)[:500]
                    elif not isinstance(output, str):
                        output = str(output)[:500] if output else ""

                    current_steps.append(Step(
                        plan=plan,
                        action_tool=tu.get("tool", ""),
                        action_input=_first_input_value(tu.get("input", {})),
                        result_output=output,
                        result_status=tu.get("status", ""),
                        reflect="",
                    ))
                pending_plan = ""

    _flush_segment()

    if not segments and messages:
        segments.append(Segment(user_message="", steps=[]))

    return segments


def compute_basic_metrics(segments: list[Segment], detail: dict) -> dict:
    """Compute simple stats for the judge prompt. No scoring judgment."""
    total_steps = sum(len(s.steps) for s in segments)
    tool_failures = sum(
        1 for s in segments for step in s.steps
        if step.result_status in ("failure", "error")
    )
    files_touched = detail.get("files_touched", []) or []
    if isinstance(files_touched, str):
        try:
            files_touched = json.loads(files_touched)
        except (json.JSONDecodeError, ValueError):
            files_touched = []

    return {
        "total_steps": total_steps,
        "segments": len(segments),
        "tool_failures": tool_failures,
        "user_messages": detail.get("user_messages", 0),
        "input_tokens": detail.get("input_tokens", 0),
        "output_tokens": detail.get("output_tokens", 0),
        "duration_seconds": detail.get("duration_seconds"),
        "files_touched": len(files_touched),
        "outcome_badge": detail.get("outcome_badge"),
    }


def _extract_task_context(messages: list[dict]) -> str:
    """Extract the user's task from the first user message + refinements."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "user":
            text = get_message_text(msg)
            if text:
                parts.append(text)
            if len(parts) >= 3:
                break
    return "\n".join(parts) if parts else "(no user message)"


def _format_metrics_line(metrics: dict) -> str:
    """Format basic metrics as a compact one-liner."""
    parts = []
    parts.append(f"Steps: {metrics.get('total_steps', 0)}")
    failures = metrics.get("tool_failures", 0)
    if failures:
        parts.append(f"Tool failures: {failures}")
    in_tok = metrics.get("input_tokens", 0)
    out_tok = metrics.get("output_tokens", 0)
    if in_tok or out_tok:
        parts.append(f"Tokens: {in_tok} in / {out_tok} out")
    dur = metrics.get("duration_seconds")
    if dur and isinstance(dur, (int, float)):
        minutes = int(dur) // 60
        parts.append(f"Duration: {minutes}m" if minutes else f"Duration: {int(dur)}s")
    files = metrics.get("files_touched", 0)
    if files:
        parts.append(f"Files: {files}")
    badge = metrics.get("outcome_badge")
    if badge:
        parts.append(f"Outcome: {badge}")
    return " | ".join(parts)


def format_session_for_judge(
    segments: list[Segment],
    task_context: str,
    metrics: dict | None = None,
) -> str:
    """Format the full session for a single judge call."""
    lines: list[str] = []

    lines.append("## User's Task")
    lines.append(task_context)
    lines.append("")

    if metrics:
        lines.append("## Session Metrics")
        lines.append(_format_metrics_line(metrics))
        lines.append("")

    if len(segments) == 1:
        seg = segments[0]
        lines.append(f"## Agent Work ({len(seg.steps)} steps)")
        for i, step in enumerate(seg.steps, 1):
            plan_text = _truncate(step.plan, 200) if step.plan else ""
            if plan_text:
                lines.append(f"Step {i}: {plan_text}")
            else:
                lines.append(f"Step {i}:")
            input_text = _truncate(step.action_input, 150)
            lines.append(f" → {step.action_tool}({input_text})")
            result_text = _truncate(step.result_output, 300)
            lines.append(f" → {step.result_status}: {result_text}")
        lines.append("")

        lines.append("## User Response After Agent Work")
        if seg.user_response:
            lines.append(f'"{_truncate(seg.user_response, 500)}"')
        else:
            lines.append("No response — session ended")
        lines.append("")
    else:
        for idx, seg in enumerate(segments):
            lines.append(f"## Turn {idx + 1}: User")
            lines.append(_truncate(seg.user_message, 300))
            lines.append("")

            if seg.steps:
                lines.append(f"## Turn {idx + 1}: Agent Work ({len(seg.steps)} steps)")
                for i, step in enumerate(seg.steps, 1):
                    plan_text = _truncate(step.plan, 200) if step.plan else ""
                    if plan_text:
                        lines.append(f"Step {i}: {plan_text}")
                    else:
                        lines.append(f"Step {i}:")
                    input_text = _truncate(step.action_input, 150)
                    lines.append(f" → {step.action_tool}({input_text})")
                    result_text = _truncate(step.result_output, 300)
                    lines.append(f" → {step.result_status}: {result_text}")
                lines.append("")

            if seg.user_response:
                lines.append(f"## Turn {idx + 1}: User Response")
                lines.append(f'"{_truncate(seg.user_response, 500)}"')
                lines.append("")

        # Show final state
        last_seg = segments[-1]
        if not last_seg.user_response:
            lines.append("## Session End")
            lines.append("No final user response — session ended")
            lines.append("")

    lines.append("## Respond with JSON:")
    lines.append('{"quality": N, "reasoning": "...", "outcome": N, "intent": N, "taste": {"detected": false}}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge: call LLM with rubric
# ---------------------------------------------------------------------------

_RUBRIC_SEARCH_PATHS = [
    Path(__file__).parent.parent / "skills" / "agentstracer-score" / "RUBRIC.md",
]

_FALLBACK_RUBRIC = """\
Score this coding agent session 1-5 for quality. \
5=excellent (verified outcome, user satisfied), 4=good, 3=average, 2=low, 1=poor. \
Return JSON with quality, reasoning, outcome, intent, and taste fields."""


def load_scoring_rubric() -> str:
    """Load the scoring rubric from the skill file."""
    for path in _RUBRIC_SEARCH_PATHS:
        if path.exists():
            text = path.read_text()
            if text.startswith("---"):
                try:
                    end = text.index("---", 3)
                    text = text[end + 3:].strip()
                except ValueError:
                    pass
            return text
    return _FALLBACK_RUBRIC


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "quality": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string"},
        "display_title": {
            "type": "string",
            "description": (
                "A concise human-readable title (under 60 chars) summarizing "
                "what the session accomplished. Use imperative mood "
                "(e.g. 'Fix auth tests', 'Add pagination to /users'). "
                "For trivial sessions use a short description like "
                "'Slash command with no task'."
            ),
        },
        "outcome": {"type": "integer", "minimum": 1, "maximum": 5},
        "intent": {"type": "integer", "minimum": 1, "maximum": 5},
        "taste": {
            "type": "object",
            "properties": {
                "detected": {"type": "boolean"},
                "type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["detected"],
        },
        "task_type": {
            "type": "string",
            "description": "A short snake_case label for the primary task type",
        },
        "outcome_label": {
            "type": "string",
            "description": "A short snake_case label for the session outcome",
        },
        "value_labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Zero or more snake_case value signal labels",
        },
        "risk_level": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Zero or more snake_case sensitivity/risk signal labels",
        },
    },
    "required": ["quality", "reasoning", "display_title", "outcome", "intent",
                  "taste", "task_type", "outcome_label", "value_labels",
                  "risk_level"],
}


_SCORER_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "scorer.md"

# Backward-compat aliases — cli.py imports SCORING_BACKEND_CHOICES
SUPPORTED_SCORING_BACKENDS = SUPPORTED_BACKENDS
SCORING_BACKEND_CHOICES = BACKEND_CHOICES
SCORING_BACKEND_COMMANDS = BACKEND_COMMANDS
SCORING_BACKEND_ENV_MARKERS = BACKEND_ENV_MARKERS
SCORING_BACKEND_COMMAND_ALIASES = BACKEND_COMMAND_ALIASES


def _resolve_scoring_backend(backend: str = "auto", env: dict[str, str] | None = None) -> str:
    """Backward-compat wrapper around backends.resolve_backend."""
    return resolve_backend(backend, env)


def _write_agent_inputs(
    tmp_path: Path,
    *,
    prompt_text: str,
    session_data: dict[str, Any],
    metadata: dict[str, Any],
    rubric: str,
) -> None:
    """Write the judge inputs that backend CLIs can inspect."""
    (tmp_path / "judge_input.md").write_text(prompt_text, encoding="utf-8")
    (tmp_path / "session.json").write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "RUBRIC.md").write_text(rubric, encoding="utf-8")


def _call_judge_with_claude(
    prompt_text: str,
    session_data: dict[str, Any],
    metadata: dict[str, Any],
    rubric: str,
    model: str | None,
) -> dict:
    """Score a session through Claude Code in an isolated temp workspace."""
    _check_backend_runtime("claude")
    command = _require_backend_command("claude")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_agent_inputs(
            tmp_path,
            prompt_text=prompt_text,
            session_data=session_data,
            metadata=metadata,
            rubric=rubric,
        )

        cmd = [
            command, "-p",
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence",
        ]
        if model:
            cmd += ["--model", model]
        if _SCORER_PROMPT_FILE.exists():
            cmd += ["--system-prompt-file", str(_SCORER_PROMPT_FILE)]

        try:
            proc = subprocess.run(
                cmd,
                input=(
                    "Score the coding agent session in the current directory. "
                    "Read judge_input.md, session.json, metadata.json, and RUBRIC.md. "
                    "Write scoring.json with your assessment."
                ),
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out waiting for claude")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            raise RuntimeError(f"claude exited {proc.returncode}: {stderr}")

        scoring_path = tmp_path / "scoring.json"
        if not scoring_path.exists():
            raise RuntimeError("Claude did not produce scoring.json")

        try:
            result = json.loads(scoring_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError("scoring.json is not valid JSON")

        return _validate_judge_result(result)


def _call_judge_with_codex(
    prompt_text: str,
    session_data: dict[str, Any],
    metadata: dict[str, Any],
    rubric: str,
    model: str | None,
) -> dict:
    """Score a session through Codex exec in an isolated temp workspace."""
    _check_backend_runtime("codex")
    command = _require_backend_command("codex")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_agent_inputs(
            tmp_path,
            prompt_text=prompt_text,
            session_data=session_data,
            metadata=metadata,
            rubric=rubric,
        )
        schema_path = tmp_path / "judge_schema.json"
        output_path = tmp_path / "scoring.json"
        schema_path.write_text(json.dumps(JUDGE_SCHEMA), encoding="utf-8")

        cmd = [
            command, "exec",
            "-c", "analytics.enabled=false",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox", "read-only",
            "--color", "never",
            "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
            "-C", str(tmp_path),
        ]
        if model:
            cmd += ["--model", model]
        cmd.append(
            "Score the coding agent session in the current directory. "
            "Read judge_input.md, session.json, metadata.json, and RUBRIC.md. "
            "Return only a JSON object matching the provided schema."
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out waiting for codex")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            stdout = proc.stdout.strip() if proc.stdout else ""
            raise RuntimeError(_format_codex_runtime_error(proc.returncode, stderr, stdout))

        if not output_path.exists():
            raise RuntimeError("Codex did not produce a scoring result")

        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError("Codex result is not valid JSON")

        return _validate_judge_result(result)


def _extract_json_candidate_strings(value: Any) -> list[str]:
    """Collect string candidates that may contain a JSON judge result."""
    candidates: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            candidates.append(text)
    elif isinstance(value, dict):
        priority_keys = (
            "text", "message", "result", "reply", "output", "content",
            "assistant", "response",
        )
        for key in priority_keys:
            if key in value:
                candidates.extend(_extract_json_candidate_strings(value[key]))
        for nested in value.values():
            candidates.extend(_extract_json_candidate_strings(nested))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_extract_json_candidate_strings(item))
    return candidates


def _extract_judge_result_from_value(value: Any) -> dict[str, Any]:
    """Find and validate a judge result inside a backend response payload."""
    if isinstance(value, dict) and "quality" in value and "reasoning" in value:
        return _validate_judge_result(value)

    for candidate in _extract_json_candidate_strings(value):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "quality" in parsed and "reasoning" in parsed:
            return _validate_judge_result(parsed)

    raise RuntimeError("Backend response did not contain a valid JSON judge result")


def _call_judge_with_openclaw(
    prompt_text: str,
    session_data: dict[str, Any],
    metadata: dict[str, Any],
    rubric: str,
    model: str | None,
) -> dict:
    """Score a session through OpenClaw's headless one-turn agent CLI."""
    if model:
        raise RuntimeError("OpenClaw backend does not support --model override from agentstracer")

    _check_backend_runtime("openclaw")
    command = _require_backend_command("openclaw")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_agent_inputs(
            tmp_path,
            prompt_text=prompt_text,
            session_data=session_data,
            metadata=metadata,
            rubric=rubric,
        )

        message = (
            "Score the coding agent session using the files below.\n\n"
            f"Read these absolute paths:\n"
            f"- {tmp_path / 'judge_input.md'}\n"
            f"- {tmp_path / 'session.json'}\n"
            f"- {tmp_path / 'metadata.json'}\n"
            f"- {tmp_path / 'RUBRIC.md'}\n\n"
            "Return only a JSON object matching the scoring schema used in the rubric. "
            "Do not wrap it in markdown fences."
        )

        cmd = [
            command, "agent",
            "--message", message,
            "--local",
            "--json",
            "--timeout", "120",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=130,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out waiting for openclaw")

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            raise RuntimeError(f"openclaw exited {proc.returncode}: {stderr}")

        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError("OpenClaw produced no output")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return _extract_judge_result_from_value(stdout)

        return _extract_judge_result_from_value(payload)


SCORING_BACKEND_RUNNERS = {
    "claude": _call_judge_with_claude,
    "codex": _call_judge_with_codex,
    "openclaw": _call_judge_with_openclaw,
}


def call_judge(
    prompt_text: str,
    model: str | None = None,
    *,
    session_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    backend: str = "auto",
) -> dict:
    """Call the resolved scoring backend and return a validated judge result."""
    rubric = load_scoring_rubric()
    resolved_backend = _resolve_scoring_backend(backend)
    session_payload = session_data or {}
    metadata_payload = metadata or {}
    runner = SCORING_BACKEND_RUNNERS.get(resolved_backend)
    if runner is None:
        raise RuntimeError(f"Unsupported scoring backend: {resolved_backend}")
    return runner(
        prompt_text,
        session_payload,
        metadata_payload,
        rubric,
        model,
    )


def _validate_judge_result(result: dict) -> dict:
    """Parse judge result safely. No scoring decisions — just type safety."""
    quality = result.get("quality")
    if not isinstance(quality, int) or not (1 <= quality <= 5):
        quality = 3  # only safety net: invalid quality defaults to middle

    outcome = result.get("outcome")
    if not isinstance(outcome, int) or not (1 <= outcome <= 5):
        outcome = quality  # default to overall quality

    intent = result.get("intent")
    if not isinstance(intent, int) or not (1 <= intent <= 5):
        intent = quality

    # Classification fields — normalize to snake_case strings
    task_type = result.get("task_type", "unknown")
    if not isinstance(task_type, str) or not task_type.strip():
        task_type = "unknown"
    task_type = task_type.strip().lower().replace(" ", "_").replace("-", "_")

    outcome_label = result.get("outcome_label", "unknown")
    if not isinstance(outcome_label, str) or not outcome_label.strip():
        outcome_label = "unknown"
    outcome_label = outcome_label.strip().lower().replace(" ", "_").replace("-", "_")

    value_labels = result.get("value_labels", [])
    if not isinstance(value_labels, list):
        value_labels = []
    value_labels = [
        v.strip().lower().replace(" ", "_").replace("-", "_")
        for v in value_labels
        if isinstance(v, str) and v.strip()
    ]

    risk_level = result.get("risk_level", [])
    if not isinstance(risk_level, list):
        risk_level = []
    risk_level = [
        v.strip().lower().replace(" ", "_").replace("-", "_")
        for v in risk_level
        if isinstance(v, str) and v.strip()
    ]

    display_title = result.get("display_title", "")
    if not isinstance(display_title, str):
        display_title = ""
    display_title = display_title.strip()[:80]

    return {
        "quality": quality,
        "reasoning": str(result.get("reasoning", "")),
        "display_title": display_title,
        "outcome": outcome,
        "intent": intent,
        "taste": result.get("taste", {"detected": False}),
        "task_type": task_type,
        "outcome_label": outcome_label,
        "value_labels": value_labels,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# Top-level: score_session
# ---------------------------------------------------------------------------

def score_session(
    conn: Any,
    session_id: str,
    *,
    model: str | None = None,
    backend: str = "auto",
) -> ScoringResult:
    """Score a session: format → judge → store. No aggregation formulas."""
    from .index import get_session_detail

    detail = get_session_detail(conn, session_id)
    if not detail:
        return ScoringResult(
            segments=[], quality=1, reason="Session not found",
        )

    messages = detail.get("messages", [])

    # Format: parse into turns
    segments = segment_session(messages)
    if not segments:
        return ScoringResult(
            segments=[], quality=1, reason="No scorable content",
        )

    metrics = compute_basic_metrics(segments, detail)
    total_steps = metrics["total_steps"]

    if total_steps == 0:
        return ScoringResult(
            segments=segments, quality=1, reason="No tool usage",
        )

    # Judge: LLM scores holistically
    task_context = _extract_task_context(messages)
    prompt = format_session_for_judge(segments, task_context, metrics)

    result = call_judge(
        prompt,
        model,
        session_data=detail,
        metadata=metrics,
        backend=backend,
    )

    # Store: pass through judge result, no formulas
    taste_signals = []
    taste = result.get("taste", {})
    if taste.get("detected"):
        taste_signals.append(taste)

    detail_data = {
        "quality": result["quality"],
        "outcome": result["outcome"],
        "intent": result["intent"],
        "reasoning": result["reasoning"],
        "display_title": result["display_title"],
        "taste_signals": taste_signals,
        "metrics": metrics,
        "task_type": result["task_type"],
        "outcome_label": result["outcome_label"],
        "value_labels": result["value_labels"],
        "risk_level": result["risk_level"],
    }

    return ScoringResult(
        segments=segments,
        quality=result["quality"],
        reason=result["reasoning"],
        display_title=result["display_title"],
        task_type=result["task_type"],
        outcome_label=result["outcome_label"],
        value_labels=result["value_labels"],
        risk_level=result["risk_level"],
        taste_signals=taste_signals,
        detail_json=json.dumps(detail_data),
    )
