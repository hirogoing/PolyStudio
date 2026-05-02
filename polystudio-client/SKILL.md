---
name: polystudio-client
description: 通过 PolyStudio 平台的 AI Agent 进行多模态内容创作。覆盖场景包括：AI 图片生成、AI 视频生成、AI 音频生成、3D 模型生成、多模态内容编辑、画布（Canvas）项目管理。当用户提到 PolyStudio、需要调用 PolyStudio 生成图片/视频/音频/3D 模型、或需要与 PolyStudio 画布对话时应触发。关键判断：只要需要通过外部 Agent 驱动 PolyStudio 完成任何 AI 创作任务，都必须触发此技能。
---

# PolyStudio Client

PolyStudio 是一个多模态 AI 内容创作平台（FastAPI + LangGraph ReAct Agent + React + Excalidraw）。你通过 `curl` 向 PolyStudio 发送指令、消费 SSE 流式响应、追踪工具调用结果、获取生成的媒体文件。

## 环境配置

```bash
export POLYSTUDIO_HOST="localhost"   # 必需，PolyStudio 主机地址
export POLYSTUDIO_PORT="8000"        # 可选，默认 8000
```

## 核心概念

| 概念 | 说明 |
|------|------|
| **Canvas（画布）** | 创作项目，包含对话历史和 Excalidraw 画布数据 |
| **SSE 流** | `POST /api/chat` 返回 `text/event-stream`，实时推送 Agent 思考过程和结果 |
| **tool_call / tool_result** | Agent 调用工具（如生成图片）时发出的事件，`tool_result.content` 包含媒体 URL |
| **canvas_id** | 项目 ID，传入后对话历史自动保存到该项目，支持多轮对话 |

---

## 发送消息（核心接口）

```
POST /api/chat
Content-Type: application/json

{
  "message":   "用户消息",
  "canvas_id": "canvas-xxx"   // 可选，不传则新建项目
}
```

用 curl 调用并实时接收 SSE 流：

```bash
curl -N --no-buffer \
  -X POST "http://${POLYSTUDIO_HOST:-localhost}:${POLYSTUDIO_PORT:-8000}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "生成一张赛博朋克城市夜景图片"}'
```

> **`-N` / `--no-buffer`** 是关键：禁用 curl 输出缓冲，SSE 帧实时打印。

---

## 典型场景

### 场景 1：单次创作请求（最常见）

```bash
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "生成一张赛博朋克城市夜景图片"}'
```

流输出示例：

```
data: {"type":"delta","content":"好的，我来为你生成"}
data: {"type":"tool_call","tool_use_id":"t1","name":"generate_image","input":{"prompt":"..."}}
data: {"type":"tool_result","tool_call_id":"t1","content":"{\"image_url\":\"/storage/images/xxx.png\"}"}
data: {"type":"delta","content":"图片已生成！"}
data: [DONE]
```

收到 `[DONE]` 即表示完成，从 `tool_result` 中提取媒体 URL。

访问生成的图片：
```
GET http://localhost:8000/storage/images/<filename>
```

---

### 场景 2：多轮对话（同一项目中追加）

`canvas_id` **不会**出现在 SSE 响应中，有两种方式获取它：

**方式 A（推荐）：开始前自己生成 canvas_id，首轮一起传入**

后端收到一个不存在的 `canvas_id` 时，会自动以该 ID 新建项目。

```bash
# 生成一个唯一 ID
CANVAS_ID="canvas-$(date +%s)000"

# 第一轮 - 传入自生成的 canvas_id，后端新建项目
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"画一只橙色的猫，卡通风格\", \"canvas_id\": \"$CANVAS_ID\"}"

# 第二轮 - 同一 canvas_id，上下文自动延续
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"给猫加上一顶魔法帽\", \"canvas_id\": \"$CANVAS_ID\"}"
```

**方式 B：首轮不传 canvas_id，结束后查询**

```bash
# 第一轮 - 不传 canvas_id，后端自动创建项目
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "画一只橙色的猫，卡通风格"}'

# 流结束（[DONE]）后，查询最新项目拿到 canvas_id
CANVAS_ID=$(curl -s http://localhost:8000/api/canvases | jq -r '.[0].id')

# 第二轮 - 传入 canvas_id 继续对话
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"给猫加上一顶魔法帽\", \"canvas_id\": \"$CANVAS_ID\"}"
```

---

### 场景 3：后台运行并捕获结果

```bash
# 将 SSE 流保存到文件，同时实时输出
curl -N --no-buffer \
  -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "生成一段轻松欢快的背景音乐，30秒"}' \
  | tee /tmp/sse_output.txt

# 从结果中提取媒体 URL
grep '"tool_result"' /tmp/sse_output.txt | grep -o '"[^"]*storage[^"]*"'
```

---

## SSE 事件类型

`POST /api/chat` 返回 `text/event-stream`，每帧格式：

```
data: <json>\n\n
...
data: [DONE]\n\n
```

| 事件类型 | 结构 | 说明 |
|---------|------|------|
| `delta` | `{"type":"delta","content":"文字片段"}` | 流式文字，累积即为完整回复 |
| `tool_call` | `{"type":"tool_call","tool_use_id":"xxx","name":"generate_image","input":{...}}` | Agent 正在调用工具 |
| `tool_result` | `{"type":"tool_result","tool_call_id":"xxx","content":"{\"image_url\":\"/storage/images/...\"}"}` | 工具执行结果，含媒体 URL |
| `skill_matched` | `{"type":"skill_matched","skill":"image-generator"}` | 命中了内置技能，仅供参考 |
| `error` | `{"type":"error","content":"错误信息"}` | 发生错误 |
| `[DONE]` | 字面量字符串，不是 JSON | **流结束信号** |

**完成判断：** 收到 `data: [DONE]` 即表示本次请求全部完成。

---

## 读懂工具结果

`tool_result.content` 是 JSON 字符串，媒体文件 URL 的路径规律：

| 类型 | URL 路径前缀 | 扩展名 |
|------|------------|--------|
| 图片 | `/storage/images/` | `.jpg` `.png` `.webp` `.gif` |
| 视频 | `/storage/videos/` | `.mp4` `.mov` `.webm` |
| 音频 | `/storage/audios/` | `.mp3` `.wav` `.m4a` `.ogg` |
| 3D 模型 | `/storage/models/` | `.glb` `.gltf` `.obj` |

访问方式（文件下载）：

```bash
curl -O "http://localhost:8000/storage/images/<filename>"
```

---

## 其他 API

### 画布管理

```bash
# 列出所有项目
curl http://localhost:8000/api/canvases

# 删除项目
curl -X DELETE http://localhost:8000/api/canvases/{canvas_id}
```

### 文件上传

```bash
# 上传图片（用于图生图等场景）
curl -X POST http://localhost:8000/api/upload-image \
  -F "file=@/path/to/image.jpg"
# → {"url": "/storage/images/upload_xxx.jpg", "filename": "upload_xxx.jpg"}

# 上传音频
curl -X POST http://localhost:8000/api/upload-audio -F "file=@audio.mp3"

# 上传视频
curl -X POST http://localhost:8000/api/upload-video -F "file=@video.mp4"
```

### WebSocket 实时订阅

当需要从外部监听某个项目的对话进展时（非 SSE 主客户端），可订阅 WebSocket：

```
ws://localhost:8000/ws/{canvas_id}
```

WS 额外事件：
- `{"type":"user_message","content":"..."}` — 有新消息发入该画布时推送
- `{"type":"done"}` — 流结束（等同于 SSE 的 `[DONE]`）

---

## 你的角色

PolyStudio 后端有完整的 LangGraph ReAct Agent 负责实际创作，你负责的是**指令传达和结果获取**。

**要做的三件事：**
1. **传话** — 把用户的原始需求原封不动发给 PolyStudio Agent
2. **等待** — 消费 SSE 流直到 `[DONE]`，监控 `tool_result` 中的媒体 URL
3. **取件** — 提取媒体 URL，通知用户结果，按需下载文件

**不要做的事：**
- 不替用户改写创作描述（原话发出去，后端 Agent 比你更懂 prompt 工程）
- 不自行拆分任务（一次 send，后端自己拆解）
- 不在消息中添加自己编的技术参数（"8K 超写实 cinematic lighting"之类的）

**正确示例：**
```
用户说：「帮我生成一张水彩风格的樱花街道」
→ 直接发：「帮我生成一张水彩风格的樱花街道」
→ 等待 SSE 结束（data: [DONE]）
→ 从 tool_result 提取图片 URL → 展示给用户
```

---

## 错误处理

| HTTP 状态码 | 含义 | 处理 |
|------------|------|------|
| 400 | 参数错误 | 检查请求 JSON 格式 |
| 500 | 服务器错误 | 检查 PolyStudio 后端是否正常运行 |
| 连接拒绝 | 服务未启动 | 确认 `POLYSTUDIO_HOST:PORT` 可访问 |

SSE 中的 `error` 事件：取 `content` 字段展示给用户。
