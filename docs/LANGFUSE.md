# Langfuse Trace 采集设计

## 目标与边界

原有 `parser.py` 面向数据集导出，会把多条执行流压平成 `messages`。Langfuse 采集使用独立的 canonical trace model，不改变现有 JSONL 导出格式，并保留：

- 一个 trace 内唯一 root，以及 agent、subagent、generation、tool、event 的父子树；
- 纳秒级 `start` / `end`、时间来源质量和必要的推断标记；
- 模型、token usage、cost、错误状态、tool call/result 关联；
- Langfuse 的 trace/span/session ID 使用稳定哈希；原生 session/thread/agent/turn/call ID 可作为已脱敏 metadata 保留，便于跨源排障；
- 活跃日志增长后的不可变快照。内容变化会产生新的 trace ID，避免修改 Langfuse v4 的不可变 observation。

工具 observation 与发起它的 generation 在 agent 下保持兄弟节点，符合 Langfuse 的 agent trace 展示习惯；两者的因果关系同时写入稳定 metadata 和 OTel span link。被工具启动的 subagent 则直接挂在该 tool 下。

Codex 优先读取其官方 `codex-rollout-trace` bundle。这里的 `exec` 是有独立生命周期的 code cell/chain；cell 内部的 `exec_command`、`write_stdin`、`apply_patch`、MCP、web、image、spawn/send/wait/close 等来自 Codex registry 的真实 dispatch 边界，各自拥有官方 `seq`、wall time、结构化 invocation/result、runtime events 和 requester，因此不再需要从 JavaScript 反推出实际工具。普通 `sessions/*.jsonl` 只作为未启用 bundle 的历史回退：新版日志把部分调用统一包装成 `exec`，回退 adapter 才会使用不执行代码的 JavaScript 字面量解析器恢复可证实的实际分派目标。动态表达式不会求值，而是降级为可解析的 `{ "arguments_code": "..." }`；纯编排脚本显示为 `functions.exec` 和 `{ "script": "..." }`，也不会伪造没有独立时间证据的子 span。

Codex 的每条人类输入都会生成独立的 `user-turn` event，并挂到源顺序对应的 `run-agent-turn` 下；同一 turn 内后续 steering 消息不会被吞掉。subagent 没有人类输入时，入站 `response_item.agent_message` 会作为结构化 agent input，但不会伪装成 `user-turn`。导入或 fork 产生的日志可能同时存在 lifecycle 时间和复制时 envelope 时间，adapter 会优先使用 payload lifecycle 与源行顺序确定 turn，再把越界子 observation 收敛到父 turn，并用 `timestamp_quality` 标明修复依据。

Codex schema 以官方源码契约为基准：`RolloutItem`、`EventMsg`、`ResponseItem`、content variant 和 rollout-trace raw event 是封闭集合，`ResponseItem::Message.role` 则是开放字符串。每个历史回退 trace 都记录已观察到的类型、schema signature、官方 `ordinal` 覆盖率和未映射枚举；官方已知但尚无专用 UI 语义的 variant 会生成普通 generic observation，真正超出官方契约的新 variant 才生成 WARNING observation，不会静默丢弃。

## Codex 官方 rollout trace

本实现按本机 `../codex` 源码中的 `codex-rs/rollout-trace` schema v1 对齐（审计基线 commit `6751b54cae32b23786001e2414d749a9916201e1`）。官方设计本身采用 “observe first, interpret later”：同一根会话和所有 spawned child thread 共用一个 append-only bundle，`trace.jsonl` 以连续 `seq` 提供因果顺序，`payloads/*.json` 保存精确 inference、tool、terminal、code cell 和 agent result 证据。AgentsTracer 直接消费 raw bundle，不要求先执行 `codex debug trace-reduce`。

rollout trace 是 Codex 明确的 opt-in 本地诊断功能。需要在**启动 Codex 之前**让其继承：

```bash
export CODEX_ROLLOUT_TRACE_ROOT="$HOME/.codex/rollout-traces"
```

AgentsTracer 会优先扫描该环境变量指向的目录；同时也扫描约定目录 `~/.codex/rollout-traces`，因此同步进程不必与录制 Codex 的 shell 共享环境变量。历史上没有启用此变量的 session 不可能补造 runtime dispatch 事件，仍使用 `sessions/*.jsonl` 回退适配器。bundle 包含 prompt、response、工具输入输出、终端输出与路径，敏感级别高于普通 session；上传 Langfuse 前仍统一经过递归 secrets/PII/用户名/路径脱敏。

generation 不会因为源日志只提供 usage、加密 reasoning 或省略 prompt snapshot 而显示成空白。Codex、Kimi、OpenClaw、OpenCode 和 Gemini 会把可见的 user/tool 上下文写入结构化 input，把文本、thinking/reasoning 与 tool calls 写入结构化 output；源端确实不可见的字段使用 `available: false` 和 `reason` 明示缺失，不伪造内容。Kimi 的字符串工具参数会优先解析成 JSON 对象；当 `ToolCall` 参数被拆到后续 `ToolCallPart` 时，adapter 会在对应 `ToolResult` 前按流顺序重组，并且只有严格 JSON 解析成功才标记为 `reconstructed_json_parts`。确实无法重组的源端片段保留为可解析的 `{ "arguments_text": "..." }`，并用 `tool_input_quality` 明示降级。

## 数据流

```text
本机 agent 原始数据
  -> provider adapter
  -> AgentTrace / TraceObservation
  -> 递归脱敏和结构校验
  -> OTLP/HTTP JSON（Langfuse ingestion v4）
  -> Observations API v2 回读验证
  -> 本地 SQLite 幂等账本
```

实现保持零运行时依赖，直接发送标准 OTLP/HTTP JSON 到 `/api/public/otel/v1/traces`。客户端会自动探测 Langfuse 写入模式：v4 携带 `x-langfuse-ingestion-version: 4` 并用 Observations API v2 回读；v3 省略该 header 并回退到 Observations API v1。可用 `LANGFUSE_INGESTION_VERSION=3|4` 强制指定。trace 级 session、tags、environment 和 metadata 会传播到每个 span，确保 Langfuse 可过滤聚合。

## Provider 映射与完备性

| 数据源 | trace 边界 | subagent 关联 | 时间 | generation / usage | 已知限制 |
|---|---|---|---|---|---|
| Claude Code | 每个根用户 turn | `toolUseResult.agentId -> Agent tool_use -> subagent file`，可精确关联；缺失时回退到 root | 消息与 tool result 有精确时间；generation start 由上一事件推断 | 消息内 model/usage 完整 | generation 没有独立 request-start 事件 |
| Codex CLI（官方 rollout trace） | 一个 root bundle 包含所有 spawned child thread；每条当前 turn 的人类输入另有 `user-turn` | 官方 thread metadata、dispatch requester、agent-result/interaction 证据；投影到 Langfuse 树时保留 edge/link 语义 | 每个 raw event 有 wall-time ms + 严格递增 `seq`；tool/inference/cell 生命周期精确 | 完整 inference request/response、model/provider/token usage | 仅覆盖启用 `CODEX_ROLLOUT_TRACE_ROOT` 后产生的会话；官方允许后台 runtime 超出激活 turn |
| Codex CLI（session JSONL 回退） | 根 thread 及所有子 thread 构成一棵 trace | `parent_thread_id`；入站 agent message 回填 subagent turn input | lifecycle 字段优先，官方 `ordinal` 优先于源行；缺失时带质量标记 | token_count、turn_context 和持久化 response item | rollout policy 本身会丢弃 transient begin/runtime 事件；动态 JS 仅做安全静态解析 |
| Kimi CLI | 每个 `TurnBegin` 区间 | `SubagentEvent.parent_tool_call_id` 精确；子 agent 内部工具单独记录 | `wire.jsonl` 绝大多数事件有时间 | StatusUpdate token usage + step generation | 极少数事件无时间时使用相邻 stream 边界 |
| OpenClaw | 原生 trajectory `traceId` | 当前本机格式没有通用 subagent parent 字段 | trajectory 的 `ts/seq` 精确；raw message 补充工具耗时 | raw session 有 model/usage；user/tool context 与 tool calls 写入 generation | 仅有 trajectory、找不到 raw session 时只能得到 run 级信息 |
| OpenCode | parent session tree | `session.parent_id` 精确 | session/message/tool state 均有 created/updated/start/end | assistant data 提供 model/tokens/cost；tool state 回填下一 generation | 老版本 DB schema 可能缺少部分列 |
| Gemini CLI | 每个 session | 当前格式无通用 subagent 字段 | session/message 时间；工具通常只有消息时间 | message model/tokens；tool call/result 保留在 generation 与 tool 节点 | tool call/result 常无独立起止时间；当前本机没有可验收的 Gemini 会话 |

无法从源数据得到的时间或关系不会伪装成精确值：adapter 会使用 `timestamp_quality`、`parent_link_quality`、`open_call`、`missing_parent` 等 metadata 明示降级。

## 隐私与运维

- 凭据仅从 `LANGFUSE_BASE_URL`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 读取，不写入仓库、配置或账本，也不打印到命令输出。
- 网络同步前，所有嵌套字符串统一经过用户名/路径匿名化、自定义字符串替换和 secrets/PII 规则脱敏。
- 单个 observation 的 input/output 默认最多 200,000 字符；超限时上传带原长度的预览。
- OTLP 每批最多 200 spans，临时网络错误和 429/5xx 使用退避重试。
- `localhost`、私网 IP、`.local` 和单标签内网主机默认绕过系统 HTTP 代理，避免内网凭据误发给代理；可用 `LANGFUSE_BYPASS_PROXY=false` 覆盖。
- 默认账本为 `~/.agentstracer/langfuse_sync.db`。已成功接收的 immutable snapshot 不重复发送；回读验证延迟不会把已接收 trace 标成可重试。

`--unsafe-no-redaction` 只关闭 secrets/PII 规则，用户名和路径仍匿名化。除非 Langfuse 完全受控且确有需要，不建议使用。
