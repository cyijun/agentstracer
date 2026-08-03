# AgentsTrace（Privacy-First Fork）

原版：[kaiaiagent/clawtrace](https://github.com/kaiaiagent/clawtrace)

---

## 快速开始

### 安装

```bash
pip install agentstracer
```

或从源码安装：

```bash
git clone https://github.com/cyijun/agentstracer.git
cd agentstracer
pip install -e .
```

### 基本使用

```bash
# 1. 配置导出源（claude/kimi/codex/gemini/all）
agentstracer config --source all

# 2. 导出对话记录
agentstracer export --no-push -o my_conversations.jsonl
```

### 私人使用（保留 API Keys）

```bash
# 禁用 secrets 脱敏（仅建议本地私人使用）
agentstracer config --no-secrets-redaction

# 导出（包含原始 API keys）
agentstracer export --no-push -o my_data.jsonl

# ⚠️ 警告：此文件包含明文 API keys，请勿分享！
```

### 其他命令

```bash
# 列出发现的项目
agentstracer list

# 启动本地 Web UI
agentstracer serve

# 查看配置
agentstracer config
```

### Langfuse 可观测性

Langfuse 凭据只从环境变量读取，不会写入 AgentsTrace 配置或同步账本：

```bash
export LANGFUSE_BASE_URL="https://langfuse.example.com"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."

# 检查服务端版本和凭据
agentstracer langfuse doctor

# 写入并回读一条不含本机对话的合成 trace
agentstracer langfuse smoke

# 只在本机解析、脱敏和检查，不发起网络请求
agentstracer langfuse preview --source all

# 增量同步并通过 Observations API v2 回读验证
agentstracer langfuse sync --source all --verify
```

同步会保留 agent / subagent / user-turn / generation / tool / event 树、时间边界与质量标记、模型和 token usage。Codex 会优先消费官方 `codex-rollout-trace` bundle，得到真实 inference、tool dispatch、code cell、terminal、subagent 边和严格递增 `seq`；未启用 bundle 的历史会话回退到 session JSONL。schema 覆盖以官方协议类型为基准，真正未知的类型以 WARNING observation 显式呈现。默认进行 secrets、用户名和路径脱敏，并以本地 SQLite 账本防止同一快照重复上传。启用方式、实现与数据源能力说明见 [Langfuse 采集设计](docs/LANGFUSE.md#codex-官方-rollout-trace)。

---

## 支持的 AI 工具

| 工具 | 数据位置 | 状态 |
|------|---------|------|
| Claude Code | `~/.claude/projects/` | ✅ |
| Kimi Code CLI | `~/.kimi-code/sessions/` | ✅ |
| Codex CLI | `~/.codex/sessions/` | ✅ |
| OpenCode | `~/.local/share/opencode/` | ✅ |
| OpenClaw | `~/.openclaw/` | ✅ |
| Gemini CLI | `~/.gemini/tmp/` | ✅ |

---

## 导出格式

**JSONL**（每行一个 JSON 对象）：

```jsonl
{"session_id": "abc-123", "model": "kimi-k2", "messages": [...], ...}
{"session_id": "def-456", "model": "claude-3-7", "messages": [...], ...}
```

### 主要字段

| 字段 | 说明 |
|------|------|
| `session_id` | 会话唯一标识 |
| `model` | AI 模型名称 |
| `project` | 项目名称（已脱敏） |
| `source` | 来源（claude/kimi/codex等） |
| `start_time` / `end_time` | ISO 8601 时间 |
| `messages` | 对话消息列表 |
| `stats` | 统计信息 |

### messages 结构

```json
{
  "role": "user|assistant",
  "content": "消息内容",
  "thinking": "思考过程（assistant）",
  "timestamp": "2024-01-01T12:00:00Z",
  "tool_uses": [{
    "tool": "bash",
    "input": {"command": "ls -la"},
    "output": {"text": "..."},
    "status": "success"
  }]
}
```

---

## 关于本版本

这是 [AgentsTrace](https://github.com/kaiaiagent/clawtrace) 的修改版本，默认专注于**本地隐私保护**；只有显式执行 `agentstracer langfuse sync` 才会向用户配置的 Langfuse 服务发送数据。

### 主要修改点

#### 1. 移除公共数据上传功能
- ❌ 云上传功能
- ❌ Skill 下载功能
- ❌ 浏览器自动打开
- ✅ 可选、显式的私有 Langfuse OTLP 同步

#### 2. 可选禁用 Secrets 脱敏
- 新增 `--no-secrets-redaction` 配置
- 私人使用时保留 API keys
- 路径/用户名脱敏始终启用

#### 3. 安全审查
- ✅ 无 `eval()` / `exec()` / `compile()`
- ✅ 无动态代码执行
- ✅ 无反序列化风险
- ✅ 路径遍历已防护

### 与原版对比

| 功能 | 原版 | 本版本 |
|------|------|--------|
| 公共云上传 | ✅ | ❌ 已移除 |
| 私有 Langfuse | ❌ | ✅ 显式启用 |
| Skill 下载 | ✅ | ❌ 已移除 |
| 可选禁用脱敏 | ❌ | ✅ 支持 |
| 适用场景 | 分享数据集 | 私人本地分析 |

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

Copyright (c) 2024 kaiaiagent (Original Author)  
Copyright (c) 2024 cyijun (Modified Version)
