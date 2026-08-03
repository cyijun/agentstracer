"""Synthetic provider fixtures for complex agent trace extraction."""

import json
import sqlite3

from agentstracer import trace_adapters as adapters


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _raw_payload(bundle, ordinal, kind, value):
    path = bundle / "payloads" / f"{ordinal}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {
        "raw_payload_id": f"raw_payload:{ordinal}",
        "kind": {"type": kind},
        "path": f"payloads/{ordinal}.json",
    }


def test_codex_official_rollout_trace_preserves_dispatch_code_cell_and_subagent(
    tmp_path, monkeypatch,
):
    trace_root = tmp_path / "rollout-traces"
    bundle = trace_root / "trace-official-root-thread"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "trace_id": "official-trace-1",
        "rollout_id": "rollout-1",
        "root_thread_id": "root-thread",
        "started_at_unix_ms": 1_767_225_600_000,
        "raw_event_log": "trace.jsonl",
        "payloads_dir": "payloads",
    }))
    root_meta = _raw_payload(bundle, 1, "session_metadata", {
        "thread_id": "root-thread",
        "agent_path": "/root",
        "model": "gpt-test",
        "provider_name": "openai-test",
        "cwd": "/work/project",
        "session_source": "cli",
    })
    request = _raw_payload(bundle, 2, "inference_request", {
        "model": "gpt-test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "inspect with child"}],
            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-root"},
        }],
    })
    response = _raw_payload(bundle, 3, "inference_response", {
        "response_id": "response-1",
        "token_usage": {
            "input_tokens": 11,
            "cached_input_tokens": 2,
            "output_tokens": 5,
            "reasoning_output_tokens": 1,
        },
        "output_items": [{
            "type": "custom_tool_call",
            "call_id": "exec-visible",
            "name": "exec",
            "input": "const r = await tools.exec_command({cmd: 'pwd'}); text(r);",
        }],
    })
    exec_invocation = _raw_payload(bundle, 4, "tool_invocation", {
        "tool_name": "exec_command",
        "tool_namespace": None,
        "payload": {
            "type": "function",
            "arguments": json.dumps({
                "cmd": "pwd", "workdir": "/work/project", "yield_time_ms": 1000,
            }),
        },
    })
    exec_result = _raw_payload(bundle, 5, "tool_result", {
        "type": "code_mode_response",
        "value": {"exit_code": 0, "output": "/work/project"},
    })
    code_result = _raw_payload(bundle, 6, "tool_result", {
        "value": {"exit_code": 0, "output": "/work/project"},
    })
    spawn_invocation = _raw_payload(bundle, 7, "tool_invocation", {
        "tool_name": "spawn_agent",
        "tool_namespace": "collaboration",
        "payload": {
            "type": "function",
            "arguments": json.dumps({
                "task_name": "schema_audit", "message": "audit official schema",
            }),
        },
    })
    spawn_result = _raw_payload(bundle, 8, "tool_result", {
        "type": "direct_response",
        "response_item": {
            "type": "function_call_output",
            "call_id": "spawn-visible",
            "output": json.dumps({"agent_id": "child-thread"}),
        },
    })
    child_meta = _raw_payload(bundle, 9, "session_metadata", {
        "thread_id": "child-thread",
        "agent_path": "/root/schema_audit",
        "task_name": "schema_audit",
        "nickname": "Ada",
        "agent_role": "explorer",
        "model": "gpt-test",
        "session_source": {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": "root-thread",
                    "agent_path": "/root/schema_audit",
                    "agent_role": "explorer",
                },
            },
        },
    })
    child_request = _raw_payload(bundle, 10, "inference_request", {
        "input": [{
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/schema_audit",
            "content": [{"type": "input_text", "text": "audit official schema"}],
            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-child"},
        }],
    })
    child_response = _raw_payload(bundle, 11, "inference_response", {
        "response_id": "response-child",
        "output_items": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "schema is aligned"}],
        }],
    })
    root_final_request = _raw_payload(bundle, 12, "inference_request", {
        "input": [
            {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "inspect with child"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-root"},
            },
            {
                "type": "function_call_output", "call_id": "spawn-visible",
                "output": json.dumps({"agent_id": "child-thread"}),
            },
            {
                "type": "agent_message", "author": "/root/schema_audit",
                "recipient": "/root",
                "content": [{"type": "input_text", "text": "schema is aligned"}],
            },
        ],
    })
    root_final_response = _raw_payload(bundle, 13, "inference_response", {
        "response_id": "response-final",
        "token_usage": {"input_tokens": 19, "output_tokens": 4},
        "output_items": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "official trace verified"}],
        }],
    })

    rows = []

    def raw(seq, event_type, payload, thread_id=None, turn_id=None):
        rows.append({
            "schema_version": 1,
            "seq": seq,
            "wall_time_unix_ms": 1_767_225_600_000 + seq * 100,
            "rollout_id": "rollout-1",
            "thread_id": thread_id,
            "codex_turn_id": turn_id,
            "payload": {"type": event_type, **payload},
        })

    raw(1, "rollout_started", {
        "trace_id": "official-trace-1", "root_thread_id": "root-thread",
    })
    raw(2, "thread_started", {
        "thread_id": "root-thread", "agent_path": "/root",
        "metadata_payload": root_meta,
    }, "root-thread")
    raw(3, "codex_turn_started", {
        "codex_turn_id": "turn-root", "thread_id": "root-thread",
    }, "root-thread", "turn-root")
    raw(4, "inference_started", {
        "inference_call_id": "inference-root", "thread_id": "root-thread",
        "codex_turn_id": "turn-root", "model": "gpt-test",
        "provider_name": "openai-test", "request_payload": request,
    }, "root-thread", "turn-root")
    raw(5, "inference_completed", {
        "inference_call_id": "inference-root", "response_id": "response-1",
        "upstream_request_id": "request-1", "response_payload": response,
    }, "root-thread", "turn-root")
    raw(6, "code_cell_started", {
        "runtime_cell_id": "cell-1", "model_visible_call_id": "exec-visible",
        "source_js": "const r = await tools.exec_command({cmd: 'pwd'}); text(r);",
    }, "root-thread", "turn-root")
    raw(7, "tool_call_started", {
        "tool_call_id": "nested-exec", "model_visible_call_id": None,
        "code_mode_runtime_tool_id": "runtime-tool-1",
        "requester": {"type": "code_cell", "runtime_cell_id": "cell-1"},
        "kind": {"type": "exec_command"},
        "summary": {"type": "generic", "label": "exec_command"},
        "invocation_payload": exec_invocation,
    }, "root-thread", "turn-root")
    raw(8, "tool_call_ended", {
        "tool_call_id": "nested-exec", "status": "completed",
        "result_payload": exec_result,
    }, "root-thread", "turn-root")
    raw(9, "code_cell_ended", {
        "runtime_cell_id": "cell-1", "status": "completed",
        "response_payload": code_result,
    }, "root-thread", "turn-root")
    raw(10, "tool_call_started", {
        "tool_call_id": "spawn-runtime", "model_visible_call_id": "spawn-visible",
        "code_mode_runtime_tool_id": None, "requester": {"type": "model"},
        "kind": {"type": "spawn_agent"},
        "summary": {"type": "agent", "task_name": "schema_audit"},
        "invocation_payload": spawn_invocation,
    }, "root-thread", "turn-root")
    raw(11, "thread_started", {
        "thread_id": "child-thread", "agent_path": "/root/schema_audit",
        "metadata_payload": child_meta,
    }, "child-thread")
    raw(12, "tool_call_ended", {
        "tool_call_id": "spawn-runtime", "status": "completed",
        "result_payload": spawn_result,
    }, "root-thread", "turn-root")
    raw(13, "codex_turn_started", {
        "codex_turn_id": "turn-child", "thread_id": "child-thread",
    }, "child-thread", "turn-child")
    raw(14, "inference_started", {
        "inference_call_id": "inference-child", "thread_id": "child-thread",
        "codex_turn_id": "turn-child", "model": "gpt-test",
        "provider_name": "openai-test", "request_payload": child_request,
    }, "child-thread", "turn-child")
    raw(15, "inference_completed", {
        "inference_call_id": "inference-child", "response_id": "response-child",
        "upstream_request_id": None, "response_payload": child_response,
    }, "child-thread", "turn-child")
    raw(16, "codex_turn_ended", {
        "codex_turn_id": "turn-child", "status": "completed",
    }, "child-thread", "turn-child")
    raw(17, "thread_ended", {
        "thread_id": "child-thread", "status": "completed",
    }, "child-thread")
    raw(18, "inference_started", {
        "inference_call_id": "inference-final", "thread_id": "root-thread",
        "codex_turn_id": "turn-root", "model": "gpt-test",
        "provider_name": "openai-test", "request_payload": root_final_request,
    }, "root-thread", "turn-root")
    raw(19, "inference_completed", {
        "inference_call_id": "inference-final", "response_id": "response-final",
        "upstream_request_id": "request-final", "response_payload": root_final_response,
    }, "root-thread", "turn-root")
    raw(20, "codex_turn_ended", {
        "codex_turn_id": "turn-root", "status": "completed",
    }, "root-thread", "turn-root")
    raw(21, "thread_ended", {
        "thread_id": "root-thread", "status": "completed",
    }, "root-thread")
    raw(22, "rollout_ended", {"status": "completed"})
    _jsonl(bundle / "trace.jsonl", rows)

    sessions = tmp_path / "sessions"
    _jsonl(sessions / "legacy.jsonl", [{
        "type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
        "payload": {"id": "rollout-1"},
    }])
    monkeypatch.setenv(adapters.CODEX_ROLLOUT_TRACE_ROOT_ENV, str(trace_root))
    monkeypatch.setattr(adapters, "CODEX_ROLLOUT_TRACE_DEFAULT_DIR", tmp_path / "none")
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", tmp_path / "archived")

    traces = [trace.finalize() for trace in adapters.iter_codex_traces()]
    assert len(traces) == 1
    trace = traces[0]
    assert trace.metadata["adapter"] == "codex-rollout-trace-v1"
    assert trace.metadata["seq_contiguous"] is True
    assert trace.metadata["payload_issue_count"] == 0
    by_id = {item.logical_id: item for item in trace.observations}
    generation = by_id["inference:inference-root"]
    assert generation.input["input"][0]["role"] == "user"
    assert generation.output["output_items"][0]["name"] == "exec"
    assert generation.usage == {
        "input": 11, "output": 5, "cache_read": 2, "reasoning": 1,
    }
    user_turn = next(item for item in trace.observations if item.name == "user-turn")
    assert user_turn.input == {"role": "user", "content": "inspect with child"}
    code_cell = by_id["code-cell:cell-1"]
    assert code_cell.as_type == "chain"
    assert code_cell.name == "functions.exec"
    concrete_tool = by_id["tool:nested-exec"]
    assert concrete_tool.name == "exec_command"
    assert concrete_tool.parent_logical_id == code_cell.logical_id
    assert concrete_tool.input == {
        "cmd": "pwd", "workdir": "/work/project", "yield_time_ms": 1000,
    }
    assert concrete_tool.output == {"exit_code": 0, "output": "/work/project"}
    assert not any(item.as_type == "tool" and item.name == "functions.exec" for item in trace.observations)
    spawn_tool = by_id["tool:spawn-runtime"]
    child = by_id["thread:child-thread"]
    assert spawn_tool.name == "collaboration.spawn_agent"
    assert child.parent_logical_id == spawn_tool.logical_id
    assert child.metadata["parent_link_quality"] == "official_spawn_metadata_and_task_name"
    assert child.input == {
        "role": "agent", "content": "audit official schema",
        "author": "/root", "recipient": "/root/schema_audit",
    }
    assert by_id["turn:turn-child"].input == child.input
    assert by_id["inference:inference-child"].output["output_items"][0]["role"] == "assistant"
    assert by_id["inference:inference-final"].input["input"][2]["type"] == "agent_message"
    assert by_id["inference:inference-final"].output["output_items"][0]["content"][0]["text"] == "official trace verified"
    assert trace.root.output == "official trace verified"


def test_codex_schema_profile_uses_official_ordinal_and_open_message_role():
    entries = [{
        "ordinal": 17,
        "type": "response_item",
        "payload": {
            "type": "message", "role": "future-provider-role",
            "content": [{"type": "input_audio", "audio_url": "data:audio/test"}],
        },
    }]
    profile = adapters._codex_schema_profile(entries)
    assert profile["coverage_complete"] is True
    assert profile["contract"]["message_role"] == "open_string"
    assert profile["nonstandard_message_roles"] == ["future-provider-role"]
    assert profile["unmapped"]["message_roles"] == []
    assert profile["ordering"] == {
        "basis": "ordinal_then_source_line",
        "ordinal_count": 1,
        "missing_ordinal_count": 0,
        "ordinal_min": 17,
        "ordinal_max": 17,
    }


def test_claude_subagent_is_nested_under_spawning_tool(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    project = projects / "project"
    root_file = project / "sess.jsonl"
    _jsonl(root_file, [
        {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "sessionId": "sess",
         "message": {"content": "delegate"}, "cwd": "/work/project"},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:01Z", "uuid": "a1",
         "message": {"model": "claude-test", "content": [
             {"type": "tool_use", "id": "call-1", "name": "Agent", "input": {"prompt": "inspect"}},
         ]}},
        {"type": "user", "timestamp": "2026-01-01T00:00:04Z",
         "message": {"content": [
             {"type": "tool_result", "tool_use_id": "call-1", "content": "child done"},
         ]}, "toolUseResult": {"agentId": "child-a"}},
    ])
    _jsonl(project / "sess" / "subagents" / "agent-child-a.jsonl", [
        {"type": "user", "timestamp": "2026-01-01T00:00:02Z", "agentId": "child-a",
         "message": {"content": "inspect"}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:03Z", "agentId": "child-a",
         "message": {"model": "claude-test", "content": [{"type": "text", "text": "found it"}]}},
    ])
    monkeypatch.setattr(adapters, "PROJECTS_DIR", projects)

    trace = next(adapters.iter_claude_traces()).finalize()
    by_id = {observation.logical_id: observation for observation in trace.observations}
    assert by_id["agent:child-a"].parent_logical_id == "root:tool:call-1"
    assert by_id["agent:child-a"].metadata["parent_link_quality"] == "exact_agent_result"
    assert by_id["root:tool:call-1"].end_ns > by_id["root:tool:call-1"].start_ns
    assert by_id["root:tool:call-1"].parent_logical_id == "root-agent"
    assert by_id["root:tool:call-1"].links[0].target_logical_id.startswith("root:generation:")
    assert any(
        observation.parent_logical_id == "agent:child-a" and observation.as_type == "generation"
        for observation in trace.observations
    )


def test_codex_thread_tree_scopes_turn_ids_and_preserves_list_tool_output(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    _jsonl(sessions / "root.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "root", "cwd": "/work/project"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.100Z",
         "payload": {"type": "task_started", "turn_id": "shared-turn"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.200Z",
         "payload": {"type": "user_message", "message": "do work"}},
        {"type": "turn_context", "timestamp": "2026-01-01T00:00:00.300Z",
         "payload": {"model": "gpt-test"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:01Z",
         "payload": {"type": "message", "content": [{"type": "output_text", "text": "working"}]}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:01.100Z",
         "payload": {"type": "function_call", "call_id": "call-1", "name": "exec", "arguments": "{\"cmd\":\"pwd\"}"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:01.200Z",
         "payload": {"type": "function_call_output", "call_id": "call-1", "output": [{"line": "ok"}]}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01.300Z",
         "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 8, "output_tokens": 3}}}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:02Z",
         "payload": {"type": "task_complete", "turn_id": "shared-turn", "last_agent_message": "done"}},
    ])
    _jsonl(sessions / "child.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00.500Z",
         "payload": {"id": "child", "parent_thread_id": "root", "cwd": "/work/project"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.600Z",
         "payload": {"type": "task_started", "turn_id": "shared-turn"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01.500Z",
         "payload": {"type": "task_complete", "turn_id": "shared-turn"}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    by_id = {observation.logical_id: observation for observation in trace.observations}
    assert "thread:root:turn:shared-turn" in by_id
    assert "thread:child:turn:shared-turn" in by_id
    assert by_id["thread:child"].parent_logical_id == "thread:root"
    tool = by_id["thread:root:tool:call-1"]
    assert tool.name == "exec_command"
    assert tool.metadata["transport_tool"] == "exec"
    assert tool.metadata["tool_name_quality"] == "inferred_from_command_arguments"
    assert tool.metadata["duration_ms"] == 100
    assert tool.output == [{"line": "ok"}]
    assert by_id[tool.parent_logical_id].as_type == "agent"
    generation = by_id[tool.links[0].target_logical_id]
    assert generation.as_type == "generation"
    assert generation.model == "gpt-test"
    assert generation.input == {
        "messages": [{"role": "user", "content": "do work"}],
    }
    assert generation.output["content"] == ["working"]
    assert generation.output["tool_calls"] == [{
        "id": "call-1",
        "name": "exec_command",
        "arguments": {"cmd": "pwd"},
    }]
    assert generation.input is not None
    assert generation.output is not None


def test_codex_exec_wrapper_exposes_concrete_tools_and_ignores_string_examples():
    single = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const example = 'tools.append(value)'; "
            "const result = await tools.web__run({search_query: []}); text(result);"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(single)
    assert name == "web.run"
    assert metadata["transport_tool"] == "exec"
    assert metadata["nested_tools"] == ["web.run"]
    assert metadata["nested_tool_identifiers"] == ["web__run"]
    assert metadata["tool_name_quality"] == "extracted_from_exec_wrapper"

    multiple = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const first = await tools.exec_command({cmd: 'pwd'}); "
            "const second = await tools.write_stdin({session_id: 1});"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(multiple)
    assert name == "exec_command + write_stdin"
    assert metadata["nested_tools"] == ["exec_command", "write_stdin"]

    direct = {
        "type": "function_call",
        "namespace": "collaboration",
        "name": "spawn_agent",
        "arguments": "{}",
    }
    name, metadata = adapters._codex_tool_descriptor(direct)
    assert name == "collaboration.spawn_agent"
    assert metadata["tool_namespace"] == "collaboration"
    assert metadata["tool_name_quality"] == "exact"

    transport_only = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": "text(ALL_TOOLS.length)",
    }
    name, metadata = adapters._codex_tool_descriptor(transport_only)
    assert name == "functions.exec"
    assert metadata["tool_name_quality"] == "transport_only"


def test_codex_generation_preserves_tool_context_and_encrypted_reasoning_marker(
    tmp_path, monkeypatch,
):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    _jsonl(sessions / "root.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "root", "cwd": "/work/project"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.100Z",
         "payload": {"type": "task_started", "turn_id": "turn"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.200Z",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "inspect"}]}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.300Z",
         "payload": {"type": "reasoning", "encrypted_content": "opaque"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.400Z",
         "payload": {"type": "function_call", "call_id": "call",
                     "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.500Z",
         "payload": {"type": "function_call_output", "call_id": "call",
                     "output": "{\"exit_code\":0}"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.600Z",
         "payload": {"type": "token_count", "info": {
             "last_token_usage": {"input_tokens": 4, "output_tokens": 2},
         }}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.700Z",
         "payload": {"type": "reasoning", "encrypted_content": "opaque-2"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00.800Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "done"}]}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:00.900Z",
         "payload": {"type": "token_count", "info": {
             "last_token_usage": {"input_tokens": 5, "output_tokens": 1},
         }}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01Z",
         "payload": {"type": "task_complete", "turn_id": "turn",
                     "last_agent_message": "done"}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    generations = [item for item in trace.observations if item.as_type == "generation"]
    assert len(generations) == 2
    assert all(item.input is not None and item.output is not None for item in generations)
    assert generations[0].output["reasoning"] == {
        "event_count": 1, "content_available": False,
    }
    assert generations[0].output["tool_calls"][0]["name"] == "exec_command"
    assert generations[1].input == {"messages": [{
        "role": "tool", "tool_call_id": "call", "content": {"exit_code": 0},
    }]}
    assert generations[1].output["content"] == ["done"]


def test_codex_user_turns_use_source_order_when_import_timestamps_overlap(
    tmp_path, monkeypatch,
):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    envelope = "2026-07-28T04:11:31.595Z"
    _jsonl(sessions / "imported.jsonl", [
        {"type": "session_meta", "timestamp": envelope,
         "payload": {"id": "imported", "originator": "Codex Desktop"}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "task_started", "turn_id": "one",
                     "started_at": 1_700_000_000}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "user_message", "message": "first"}},
        {"type": "response_item", "timestamp": envelope,
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "one"}]}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "task_complete", "turn_id": "one",
                     "started_at": 1_700_000_000}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "task_started", "turn_id": "two",
                     "started_at": 1_700_000_100}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "user_message", "message": "second"}},
        {"type": "response_item", "timestamp": envelope,
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "two"}]}},
        {"type": "event_msg", "timestamp": envelope,
         "payload": {"type": "task_complete", "turn_id": "two",
                     "started_at": 1_700_000_100, "completed_at": 1_700_000_101}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    turns = {
        item.metadata["turn_id"]: item
        for item in trace.observations if item.name == "run-agent-turn"
    }
    user_turns = [
        item for item in trace.observations if item.name == "user-turn"
    ]
    assert turns["one"].input == {"role": "user", "content": "first"}
    assert turns["two"].input == {"role": "user", "content": "second"}
    assert {item.input["content"]: item.parent_logical_id for item in user_turns} == {
        "first": turns["one"].logical_id,
        "second": turns["two"].logical_id,
    }
    assert all(
        item.metadata["timestamp_quality"] == "inferred_from_turn_start"
        for item in user_turns
    )


def test_codex_schema_profile_surfaces_unknown_values_as_observations(
    tmp_path, monkeypatch,
):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    _jsonl(sessions / "future.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "future"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01Z",
         "payload": {"type": "task_started", "turn_id": "turn"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:02Z",
         "payload": {"type": "future_event", "mode": "new-enum"}},
        {"type": "future_record", "timestamp": "2026-01-01T00:00:03Z",
         "payload": {"kind": "new-shape"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:04Z",
         "payload": {"type": "task_complete", "turn_id": "turn"}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    profile = trace.metadata["schema_profile"]
    assert profile["coverage_complete"] is False
    assert profile["unmapped"]["event_types"] == ["future_event"]
    assert profile["unmapped"]["top_level_types"] == ["future_record"]
    unknown = [
        item for item in trace.observations
        if item.name == "unmapped-codex-schema"
    ]
    assert len(unknown) == 2
    assert all(item.level == "WARNING" for item in unknown)


def test_codex_event_messages_fill_generation_output_without_duplicates(
    tmp_path, monkeypatch,
):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    _jsonl(sessions / "messages.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "messages"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01Z",
         "payload": {"type": "task_started", "turn_id": "turn"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:02Z",
         "payload": {"type": "agent_reasoning", "text": "checking"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:02Z",
         "payload": {"type": "reasoning",
                     "summary": [{"type": "summary_text", "text": "checking"}]}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:03Z",
         "payload": {"type": "agent_message", "message": "answer"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:03Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "answer"}]}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:04Z",
         "payload": {"type": "token_count", "info": {}}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:05Z",
         "payload": {"type": "task_complete", "turn_id": "turn"}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    generation = next(
        item for item in trace.observations if item.as_type == "generation"
    )
    assert generation.output["content"].count({"thinking": "checking"}) == 1
    assert generation.output["content"].count("answer") == 1


def test_codex_subagent_turn_uses_inbound_agent_message_as_input(
    tmp_path, monkeypatch,
):
    sessions = tmp_path / "sessions"
    archived = tmp_path / "archived"
    _jsonl(sessions / "child.jsonl", [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "child", "agent_path": "/root/child"}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:01Z",
         "payload": {"type": "task_started", "turn_id": "turn"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:02Z",
         "payload": {"type": "agent_message", "author": "/root",
                     "recipient": "/root/child",
                     "content": [{"type": "input_text", "text": "audit schema"}]}},
        {"type": "event_msg", "timestamp": "2026-01-01T00:00:03Z",
         "payload": {"type": "task_complete", "turn_id": "turn"}},
    ])
    monkeypatch.setattr(adapters, "CODEX_SESSIONS_DIR", sessions)
    monkeypatch.setattr(adapters, "CODEX_ARCHIVED_DIR", archived)

    trace = next(adapters.iter_codex_traces()).finalize()
    turn = next(
        item for item in trace.observations if item.name == "run-agent-turn"
    )
    assert turn.input == {
        "role": "agent", "content": "audit schema", "author": "/root",
    }
    assert turn.metadata["user_input_quality"] == "inter_agent_message"
    assert not any(item.name == "user-turn" for item in trace.observations)
    message = next(
        item for item in trace.observations if item.name == "inter-agent-message"
    )
    assert message.input == {
        "author": "/root", "recipient": "/root/child",
        "content": "audit schema",
    }


def test_codex_exec_wrapper_extracts_json_compatible_tool_arguments():
    payload = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const workdir = '/tmp/project'; "
            "const result = await tools.exec_command({"
            "cmd: 'pwd', workdir, yield_time_ms: 1000, tty: false"
            "}); text(result);"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(payload)
    tool_input = adapters._codex_tool_input(payload, name, metadata)
    assert tool_input == {
        "cmd": "pwd",
        "workdir": "/tmp/project",
        "yield_time_ms": 1000,
        "tty": False,
    }
    assert metadata["tool_input_quality"] == "extracted_js_literal"
    assert metadata["tool_input_call_count"] == 1


def test_codex_apply_patch_resolves_literal_variable_and_wraps_freeform_input():
    wrapped = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const patch = \"*** Begin Patch\\n*** End Patch\"; "
            "const result = await tools.apply_patch(patch); text(result);"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(wrapped)
    assert adapters._codex_tool_input(wrapped, name, metadata) == {
        "patch": "*** Begin Patch\n*** End Patch",
    }
    assert metadata["tool_input_quality"] == "extracted_js_literal"

    direct = {
        "type": "custom_tool_call",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }
    name, metadata = adapters._codex_tool_descriptor(direct)
    assert adapters._codex_tool_input(direct, name, metadata) == {
        "patch": "*** Begin Patch\n*** End Patch",
    }
    assert metadata["tool_input_quality"] == "raw_text_wrapped"


def test_codex_multiple_tools_get_structured_calls_and_safe_fallback():
    payload = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const first = await tools.exec_command({cmd: 'pwd'}); "
            "const second = await tools.write_stdin(makeInput());"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(payload)
    assert adapters._codex_tool_input(payload, name, metadata) == {
        "calls": [
            {"tool": "exec_command", "arguments": {"cmd": "pwd"}},
            {"tool": "write_stdin", "arguments": {"arguments_code": "makeInput()"}},
        ],
    }
    assert metadata["tool_input_quality"] == "partial_js_literal"
    assert metadata["unparsed_tool_inputs"] == 1


def test_codex_exec_wrapper_expands_literal_array_map_arguments():
    payload = {
        "type": "custom_tool_call",
        "name": "exec",
        "input": (
            "const cmds = [[\"pwd\", 6000], [\"git status --short\", 10000]];\n"
            "const results = await Promise.all(cmds.map(async ([cmd, max], index) => "
            "await tools.exec_command({cmd, workdir: \"/workspace\", "
            "yield_time_ms: 1000, max_output_tokens: max, batch_index: index})));"
        ),
    }
    name, metadata = adapters._codex_tool_descriptor(payload)
    assert name == "exec_command"
    assert adapters._codex_tool_input(payload, name, metadata) == {
        "calls": [
            {"tool": "exec_command", "arguments": {
                "cmd": "pwd",
                "workdir": "/workspace",
                "yield_time_ms": 1000,
                "max_output_tokens": 6000,
                "batch_index": 0,
            }},
            {"tool": "exec_command", "arguments": {
                "cmd": "git status --short",
                "workdir": "/workspace",
                "yield_time_ms": 1000,
                "max_output_tokens": 10000,
                "batch_index": 1,
            }},
        ],
    }
    assert metadata["tool_input_quality"] == "expanded_static_map"
    assert metadata["tool_input_source_call_count"] == 1
    assert metadata["tool_input_call_count"] == 2
    assert metadata["static_map_iterations"] == 2
    assert metadata["static_map_collections"] == ["cmds"]


def test_codex_tool_error_uses_exit_code_when_transport_status_completed():
    output = {"exit_code": 2, "output": "failed"}
    assert adapters._codex_tool_failed("completed", output)
    assert adapters._codex_tool_error_message("completed", output) == "exit_code=2"


def test_kimi_subagent_uses_explicit_parent_tool_and_string_result(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    wire = sessions / "project" / "session" / "wire.jsonl"
    _jsonl(wire, [
        {"timestamp": "2026-01-01T00:00:00Z", "message": {"type": "TurnBegin", "payload": {"user_input": "research"}}},
        {"timestamp": "2026-01-01T00:00:00.100Z", "message": {"type": "StepBegin", "payload": {}}},
        {"timestamp": "2026-01-01T00:00:01Z", "message": {"type": "ToolCall", "payload": {
            "id": "spawn", "function": {"name": "Task", "arguments": {"prompt": "find"}},
        }}},
        {"timestamp": "2026-01-01T00:00:01.100Z", "message": {"type": "SubagentEvent", "payload": {
            "agent_id": "sub-1", "parent_tool_call_id": "spawn", "subagent_type": "researcher",
            "event": {"type": "ContentPart", "payload": {"text": "result"}},
        }}},
        {"timestamp": "2026-01-01T00:00:02Z", "message": {"type": "ToolResult", "payload": {
            "tool_call_id": "spawn", "return_value": "complete",
        }}},
        {"timestamp": "2026-01-01T00:00:03Z", "message": {"type": "TurnEnd", "payload": {}}},
    ])
    monkeypatch.setattr(adapters, "KIMI_SESSIONS_DIR", sessions)

    trace = next(adapters.iter_kimi_traces()).finalize()
    by_id = {observation.logical_id: observation for observation in trace.observations}
    assert by_id["subagent:sub-1"].parent_logical_id == "root:tool:spawn"
    assert by_id["subagent:sub-1"].metadata["parent_link_quality"] == "exact"
    assert by_id["root:tool:spawn"].output == "complete"
    assert by_id["root:tool:spawn"].level == "DEFAULT"


def test_kimi_tool_inputs_are_json_objects_with_quality_metadata(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    wire = sessions / "project" / "session" / "wire.jsonl"
    _jsonl(wire, [
        {"timestamp": "2026-01-01T00:00:00Z", "message": {"type": "TurnBegin", "payload": {"user_input": "run"}}},
        {"timestamp": "2026-01-01T00:00:00.100Z", "message": {"type": "StepBegin", "payload": {}}},
        {"timestamp": "2026-01-01T00:00:01Z", "message": {"type": "ToolCall", "payload": {
            "id": "parsed", "function": {"name": "Shell", "arguments": "{\"command\":\"pwd\"}"},
        }}},
        {"timestamp": "2026-01-01T00:00:02Z", "message": {"type": "ToolCall", "payload": {
            "id": "partial", "function": {"name": "ReadFile", "arguments": "{\"path\":"},
        }}},
        {"timestamp": "2026-01-01T00:00:02.100Z", "message": {"type": "ToolCall", "payload": {
            "id": "reconstructed", "function": {"name": "ReadFile", "arguments": "{\"path\":"},
        }}},
        {"timestamp": "2026-01-01T00:00:02.200Z", "message": {"type": "ToolCallPart", "payload": {
            "arguments_part": "\"README.md\"}",
        }}},
        {"timestamp": "2026-01-01T00:00:03Z", "message": {"type": "TurnEnd", "payload": {}}},
    ])
    monkeypatch.setattr(adapters, "KIMI_SESSIONS_DIR", sessions)

    trace = next(adapters.iter_kimi_traces()).finalize()
    by_id = {observation.logical_id: observation for observation in trace.observations}
    assert by_id["root:tool:parsed"].input == {"command": "pwd"}
    assert by_id["root:tool:parsed"].metadata["tool_input_quality"] == "parsed_json"
    assert by_id["root:tool:partial"].input == {"arguments_text": "{\"path\":"}
    assert by_id["root:tool:partial"].metadata["tool_input_quality"] == "unparseable_source_text"
    assert by_id["root:tool:reconstructed"].input == {"path": "README.md"}
    assert by_id["root:tool:reconstructed"].metadata["tool_input_quality"] == "reconstructed_json_parts"
    assert by_id["root:tool:reconstructed"].metadata["tool_input_part_count"] == 1


def test_kimi_generation_preserves_prompt_tool_calls_and_tool_results(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    wire = sessions / "project" / "session" / "wire.jsonl"
    _jsonl(wire, [
        {"timestamp": "2026-01-01T00:00:00Z", "message": {"type": "TurnBegin", "payload": {"user_input": "inspect"}}},
        {"timestamp": "2026-01-01T00:00:00.100Z", "message": {"type": "StepBegin", "payload": {}}},
        {"timestamp": "2026-01-01T00:00:01Z", "message": {"type": "ToolCall", "payload": {
            "id": "read", "function": {"name": "ReadFile", "arguments": "{\"path\":\"README.md\"}"},
        }}},
        {"timestamp": "2026-01-01T00:00:02Z", "message": {"type": "ToolResult", "payload": {
            "tool_call_id": "read", "return_value": {"output": "contents"},
        }}},
        {"timestamp": "2026-01-01T00:00:03Z", "message": {"type": "StepBegin", "payload": {}}},
        {"timestamp": "2026-01-01T00:00:04Z", "message": {"type": "ContentPart", "payload": {"text": "done"}}},
        {"timestamp": "2026-01-01T00:00:05Z", "message": {"type": "TurnEnd", "payload": {}}},
    ])
    monkeypatch.setattr(adapters, "KIMI_SESSIONS_DIR", sessions)

    trace = next(adapters.iter_kimi_traces()).finalize()
    generations = [observation for observation in trace.observations if observation.as_type == "generation"]
    assert len(generations) == 2
    assert generations[0].input == {"messages": [{"role": "user", "content": "inspect"}]}
    assert generations[0].output["tool_calls"][0]["arguments"] == {"path": "README.md"}
    assert generations[1].input == {"messages": [{
        "role": "tool", "tool_call_id": "read", "content": {"output": "contents"},
    }]}
    assert generations[1].output == {"content": ["done"]}


def test_openclaw_trajectory_enriches_raw_session_without_duplicate(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    session_dir = agents / "main" / "sessions"
    raw = session_dir / "sess.jsonl"
    _jsonl(raw, [
        {"type": "session", "id": "sess", "cwd": "/work/project"},
        {"type": "message", "timestamp": 1_767_225_600_000, "message": {
            "role": "user", "timestamp": 1_767_225_600_000,
            "content": [{"type": "text", "text": "run"}],
        }},
        {"type": "message", "timestamp": 1_767_225_601_000, "message": {
            "role": "assistant", "timestamp": 1_767_225_601_000, "model": "model",
            "content": [{"type": "toolCall", "id": "tool-1", "name": "exec", "arguments": {"cmd": "pwd"}}],
        }},
        {"type": "message", "timestamp": 1_767_225_602_000, "message": {
            "role": "toolResult", "timestamp": 1_767_225_602_000,
            "toolCallId": "tool-1", "content": [{"type": "text", "text": "ok"}],
        }},
    ])
    _jsonl(session_dir / "sess.trajectory.jsonl", [
        {"traceId": "native-trace", "runId": "run-1", "seq": 1,
         "type": "session.started", "ts": "2026-01-01T00:00:00Z",
         "data": {"sessionFile": str(raw), "agentId": "main", "workspaceDir": "/work/project"}},
        {"traceId": "native-trace", "runId": "run-1", "seq": 2,
         "type": "session.ended", "ts": "2026-01-01T00:00:03Z", "data": {"status": "completed"}},
    ])
    monkeypatch.setattr(adapters, "OPENCLAW_AGENTS_DIR", agents)

    traces = [trace.finalize() for trace in adapters.iter_openclaw_traces()]
    assert len(traces) == 1
    assert traces[0].metadata["adapter"] == "openclaw-trajectory-v2"
    assert {observation.as_type for observation in traces[0].observations} == {"agent", "generation", "tool"}
    assert traces[0].root.metadata["native_trace_id"] == "native-trace"
    generation = next(observation for observation in traces[0].observations if observation.as_type == "generation")
    assert generation.input == {"messages": [{"role": "user", "content": "run"}]}
    assert generation.output["tool_calls"][0]["arguments"] == {"cmd": "pwd"}


def test_openclaw_active_and_deleted_copies_keep_unique_ids_and_active_events(
    tmp_path, monkeypatch,
):
    agents = tmp_path / "agents"
    session_dir = agents / "main" / "sessions"
    active = session_dir / "sess.jsonl"
    deleted = session_dir / "sess.jsonl.deleted.2026-01-02T00-00-00Z"
    header = {"type": "session", "id": "sess", "cwd": "/work/project"}
    _jsonl(active, [
        header,
        {"type": "message", "timestamp": "2026-01-01T00:00:00Z", "message": {
            "role": "user", "content": "run active",
        }},
        {"type": "message", "timestamp": "2026-01-01T00:00:01Z", "message": {
            "role": "assistant", "content": "active answer",
        }},
    ])
    _jsonl(deleted, [
        header,
        {"type": "message", "timestamp": "2025-12-31T00:00:00Z", "message": {
            "role": "user", "content": "run deleted",
        }},
        {"type": "message", "timestamp": "2025-12-31T00:00:01Z", "message": {
            "role": "assistant", "content": "deleted answer",
        }},
    ])
    active_trajectory = session_dir / "sess.trajectory.jsonl"
    _jsonl(active_trajectory, [
        {"traceId": "native-trace", "seq": 1, "type": "session.started",
         "ts": "2026-01-01T00:00:00Z", "data": {"sessionFile": str(active)}},
        {"traceId": "native-trace", "seq": 2, "type": "session.ended",
         "ts": "2026-01-01T00:00:02Z", "data": {"status": "completed"}},
    ])
    _jsonl(session_dir / "sess.trajectory.jsonl.deleted.2026-01-02T00-00-00Z", [
        {"traceId": "native-trace", "seq": 1, "type": "session.started",
         "ts": "2025-12-31T00:00:00Z", "data": {"sessionFile": str(deleted)}},
        {"traceId": "native-trace", "seq": 2, "type": "session.ended",
         "ts": "2025-12-31T00:00:02Z", "data": {"status": "failed", "terminalError": "stale"}},
    ])
    monkeypatch.setattr(adapters, "OPENCLAW_AGENTS_DIR", agents)

    traces = [trace.finalize() for trace in adapters.iter_openclaw_traces()]

    assert len(traces) == 2
    assert len({trace.source_session_id for trace in traces}) == 2
    trajectory = next(trace for trace in traces if trace.metadata["adapter"] == "openclaw-trajectory-v2")
    archived = next(trace for trace in traces if trace.metadata["adapter"] == "openclaw-raw-v2")
    assert trajectory.source_session_id == "sess"
    assert trajectory.root.level == "DEFAULT"
    assert trajectory.root.output == "active answer"
    assert archived.source_session_id.startswith("sess::deleted::")
    assert archived.root.metadata["session_status"] == "deleted"
    assert archived.root.output == "deleted answer"


def test_usage_ignores_malformed_numeric_values():
    assert adapters._usage({
        "input": "12",
        "output": "not-a-number",
        "cacheRead": float("inf"),
        "reasoning_output_tokens": True,
    }) == {"input": 12}


def test_opencode_parent_sessions_and_exact_tool_times(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE session (
            id TEXT PRIMARY KEY, parent_id TEXT, time_created INTEGER,
            time_updated INTEGER, directory TEXT, agent TEXT
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT
        );
    """)
    conn.executemany("INSERT INTO session VALUES (?,?,?,?,?,?)", [
        ("root", None, 1_767_225_600_000, 1_767_225_604_000, "/work/project", "build"),
        ("child", "root", 1_767_225_601_000, 1_767_225_603_000, "/work/project", "explore"),
    ])
    assistant = json.dumps({
        "role": "assistant", "model": {"providerID": "p", "modelID": "m"},
        "time": {"created": 1_767_225_600_500, "completed": 1_767_225_602_500},
        "tokens": {"input": 4, "output": 2},
    })
    conn.execute("INSERT INTO message VALUES (?,?,?,?,?)", (
        "m1", "root", 1_767_225_600_500, 1_767_225_602_500, assistant,
    ))
    conn.executemany("INSERT INTO part VALUES (?,?,?,?)", [
        ("p1", "m1", 1_767_225_600_600, json.dumps({"type": "text", "text": "done"})),
        ("p2", "m1", 1_767_225_601_000, json.dumps({
            "type": "tool", "callID": "call-1", "tool": "read",
            "state": {"status": "completed", "input": {"path": "x"}, "output": "ok",
                      "time": {"start": 1_767_225_601_000, "end": 1_767_225_602_000}},
        })),
    ])
    conn.commit()
    conn.close()
    monkeypatch.setattr(adapters, "OPENCODE_DB_PATH", db)

    trace = next(adapters.iter_opencode_traces()).finalize()
    by_id = {observation.logical_id: observation for observation in trace.observations}
    assert by_id["session:child"].parent_logical_id == "session:root"
    tool = by_id["session:root:tool:call-1"]
    generation = by_id["session:root:generation:m1"]
    assert generation.input["context"]["available"] is False
    assert generation.output["tool_calls"][0]["arguments"] == {"path": "x"}
    assert tool.end_ns - tool.start_ns == 1_000_000_000
    assert tool.parent_logical_id == "session:root"
    assert tool.links[0].target_logical_id == "session:root:generation:m1"


def test_gemini_generation_and_tools(tmp_path, monkeypatch):
    gemini = tmp_path / "gemini"
    path = gemini / "project" / "chats" / "session-one.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "sessionId": "one", "startTime": "2026-01-01T00:00:00Z",
        "lastUpdated": "2026-01-01T00:00:02Z",
        "messages": [
            {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "content": "hello"},
            {"type": "gemini", "timestamp": "2026-01-01T00:00:01Z", "content": "done",
             "model": "gemini-test", "tokens": {"input": 4, "output": 2},
             "toolCalls": [{"name": "read", "args": {"path": "x"}, "result": "ok"}]},
        ],
    }))
    monkeypatch.setattr(adapters, "GEMINI_DIR", gemini)

    trace = next(adapters.iter_gemini_traces()).finalize()
    assert [observation.as_type for observation in trace.observations] == ["agent", "generation", "tool"]
    assert trace.observations[1].input == {"messages": [{"role": "user", "content": "hello"}]}
    assert trace.observations[1].output["tool_calls"][0]["arguments"] == {"path": "x"}
    assert trace.observations[2].parent_logical_id == "root-agent"
    assert trace.observations[2].links[0].target_logical_id == trace.observations[1].logical_id
