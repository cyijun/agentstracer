# ClawTrace (Local-Only Fork)

这是一个 [ClawTrace](https://github.com/kaiaiagent/clawtrace) 的修改版本，专注于本地隐私保护和纯离线使用。

## 原版项目

- **作者**: kaiaiagent
- **仓库**: https://github.com/kaiaiagent/clawtrace
- **PyPI**: https://pypi.org/project/clawtrace/

## 本版本的修改点

### 🔒 1. 移除所有网络功能

**目的**: 确保数据绝对不会上传到外部服务器，纯本地使用。

**修改内容**:
- ❌ 移除了云上传功能 (`share_bundle()`)
- ❌ 移除了 Skill 下载功能 (`update_skill()` 不再从网络下载)
- ❌ 移除了浏览器自动打开
- ❌ 移除了 urllib 网络请求相关代码

**影响文件**:
- `clawtrace/daemon.py`: 删除了 `_ensure_device_token()`, `_build_multipart_body()`, 云上传相关代码
- `clawtrace/cli.py`: 删除了 `urllib` 导入，移除了网络下载 skill 的逻辑

### 🔑 2. 可选禁用 Secrets 脱敏

**目的**: 私人使用时保留 API keys 等敏感信息，便于后续分析或复现。

**新增配置**:
```bash
# 禁用 secrets 脱敏（保留 API keys）
clawtrace config --no-secrets-redaction

# 恢复默认脱敏
clawtrace config --no-secrets-redaction=false
```

**脱敏层级**:

| 层级 | 内容 | 是否可禁用 |
|-----|------|----------|
| Secrets 脱敏 | API keys, tokens, JWT, 邮箱等 | ✅ `--no-secrets-redaction` |
| 路径/用户名脱敏 | `/Users/alice` → `/user_hash` | ❌ 始终启用 |

**修改文件**:
- `clawtrace/config.py`: 添加 `no_secrets_redaction` 配置项
- `clawtrace/cli.py`: 修改 `export_to_jsonl()` 支持条件脱敏
- `clawtrace/cli.py`: 添加 `--no-secrets-redaction` 命令行参数

### 🛡️ 安全审查

对代码进行了全面的安全审查，确认无以下危险模式:
- ✅ 无 `eval()` / `exec()` / `compile()` 动态代码执行
- ✅ 无 `__import__()` 动态导入
- ✅ 无 `base64` 解码执行
- ✅ 无 `pickle` / `marshal` 反序列化
- ✅ 无 `ctypes` / `cffi` 外部库调用
- ✅ 路径遍历已正确防护 (`is_relative_to` 检查)

### 📝 其他修改

- 更新了 `LICENSE` 文件，保留原作者版权并添加修改者信息
- 创建了本文档说明所有修改点

## 安装方法

```bash
# 从源码安装
git clone https://github.com/cyijun/clawtrace-local.git
cd clawtrace-local
pip install -e .

# 或使用 pip 直接安装（如果发布了）
pip install clawtrace-local
```

## 使用方法

### 基本导出

```bash
# 配置源（claude, kimi, codex, all）
clawtrace config --source all

# 导出到本地文件
clawtrace export --no-push -o my_conversations.jsonl
```

### 私人使用（保留 API keys）

```bash
# 禁用 secrets 脱敏
clawtrace config --no-secrets-redaction

# 导出（包含原始 API keys）
clawtrace export --no-push -o my_data.jsonl

# ⚠️ 警告：此文件包含明文 API keys，请妥善保管，不要分享！
```

### 扫描会话

```bash
# 列出发现的项目
clawtrace list

# 启动本地 Web UI
clawtrace serve
```

## 导出格式

导出文件为 **JSONL 格式**（JSON Lines），每行一个独立的 JSON 对象:

```jsonl
{"session_id": "abc-123", "model": "kimi-k2", "messages": [...], "stats": {...}}
{"session_id": "def-456", "model": "claude-3-7", "messages": [...], "stats": {...}}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话唯一标识 |
| `model` | string | AI 模型名称 |
| `project` | string | 项目名称（已脱敏） |
| `source` | string | 来源（claude/kimi/codex等） |
| `start_time` | string | ISO 8601 格式时间 |
| `end_time` | string | ISO 8601 格式时间 |
| `messages` | array | 对话消息列表 |
| `stats` | object | 统计信息 |

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

## 支持的 AI 工具

| 工具 | 来源 | 状态 |
|------|------|------|
| Claude Code | `~/.claude/projects/` | ✅ 支持 |
| Kimi CLI | `~/.kimi/sessions/` | ✅ 支持 |
| Codex CLI | `~/.codex/sessions/` | ✅ 支持 |
| OpenCode | `~/.local/share/opencode/` | ✅ 支持 |
| OpenClaw | `~/.openclaw/` | ✅ 支持 |
| Gemini CLI | `~/.gemini/tmp/` | ✅ 支持 |

## 安全提醒

⚠️ **使用 `--no-secrets-redaction` 时请注意：**

1. 导出的文件将包含明文 API keys 和 tokens
2. 请勿将此类文件分享给他人
3. 请勿上传到公共仓库或云存储
4. 建议仅在本地私有环境使用

## 与原版的主要区别

| 功能 | 原版 | 本版本 |
|------|------|--------|
| 云上传 | ✅ 支持 HF | ❌ 已移除 |
| Skill 下载 | ✅ 支持 | ❌ 已移除 |
| 网络请求 | ✅ 有 | ❌ 无 |
| 可选禁用脱敏 | ❌ 无 | ✅ 支持 |
| 适用场景 | 分享数据集 | 私人本地分析 |

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**免责声明**: 本修改版本仅供学习和私人使用。使用本工具导出和处理 AI 对话数据时，请遵守相关服务条款和隐私政策。
