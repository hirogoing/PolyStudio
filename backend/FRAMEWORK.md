# PolyStudio 后端框架详细文档

## 📋 目录结构

```
backend/
├── app/                          # 应用主目录
│   ├── __init__.py               # 包初始化文件
│   ├── main.py                   # FastAPI 应用入口
│   ├── routers/                  # 路由层
│   │   ├── __init__.py
│   │   └── chat.py               # 聊天相关路由
│   ├── services/                 # 服务层
│   │   ├── __init__.py
│   │   ├── agent_service.py      # Agent 服务（LangGraph Agent 管理）
│   │   ├── stream_processor.py   # 流式处理器（SSE 流式输出）
│   │   └── history_service.py    # 历史记录服务（对话和画布管理）
│   └── tools/                    # 工具层（LangChain Tools）
│       ├── __init__.py
│       ├── image_generation.py   # SiliconFlow 图片生成/编辑工具
│       ├── volcano_image_generation.py  # 火山引擎图片生成/编辑工具
│       └── model_3d_generation.py       # 腾讯云混元生3D工具
├── storage/                      # 存储目录
│   ├── images/                   # 图片存储
│   ├── models/                   # 3D模型存储
│   └── chat_history.json         # 对话历史JSON文件
├── scripts/                      # 脚本目录
│   └── normalize_storage_images.py  # 图片标准化脚本
├── requirements.txt              # Python依赖
├── env.example                   # 环境变量示例
├── start.sh                      # 启动脚本
└── FRAMEWORK.md                  # 本文档
```

---

## 🏗️ 架构层次

### 1. **入口层 (Entry Layer)**

#### `app/main.py` - FastAPI 应用入口

**职责：**
- 初始化 FastAPI 应用
- 配置 CORS 中间件
- 注册路由
- 配置静态文件服务（用于访问 storage 目录）
- 创建必要的存储目录

**关键功能：**
```python
# 主要组件
- FastAPI 应用实例
- CORS 中间件配置
- 静态文件挂载 (/storage -> storage/)
- 路由注册 (/api/*)
```

**启动方式：**
- 直接运行：`python -m app.main`
- 使用 uvicorn：`uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- 使用启动脚本：`./start.sh`

---

### 2. **路由层 (Router Layer)**

#### `app/routers/chat.py` - 聊天路由

**职责：**
- 处理 HTTP 请求
- 参数验证（使用 Pydantic）
- 调用服务层处理业务逻辑
- 返回响应（包括流式响应）

**API 端点：**

| 方法 | 路径 | 功能 | 说明 |
|------|------|------|------|
| `GET` | `/api/canvases` | 获取所有画布历史 | 返回所有保存的画布项目 |
| `POST` | `/api/canvases` | 保存/更新画布 | 保存 Excalidraw 画布数据 |
| `DELETE` | `/api/canvases/{canvas_id}` | 删除画布 | 根据ID删除画布 |
| `POST` | `/api/upload-image` | 上传图片 | 上传图片到 storage/images |
| `POST` | `/api/chat` | 聊天请求 | 流式响应（SSE格式） |

**关键实现：**
- `ChatRequest` 模型：验证请求参数
- `StreamingResponse`：返回 SSE 流式响应
- 文件上传处理：保存到本地并返回相对路径

---

### 3. **服务层 (Service Layer)**

#### 3.1 `app/services/agent_service.py` - Agent 服务

**职责：**
- 创建和管理 LangGraph Agent
- 配置 LLM 模型（ChatOpenAI）
- 注册工具（Tools）
- 处理聊天流式响应

**核心功能：**

1. **Agent 创建 (`create_agent`)**
   - 初始化 ChatOpenAI 模型（支持 SiliconFlow）
   - 配置流式输出 (`streaming=True`)
   - 禁用并行工具调用 (`parallel_tool_calls=False`)
   - 注册工具列表：
     - `generate_volcano_image_tool` - 火山引擎图片生成
     - `edit_volcano_image_tool` - 火山引擎图片编辑
     - `generate_3d_model_tool` - 3D模型生成
   - 创建 LangGraph ReAct Agent
   - 配置详细的 Agent Prompt（包含工具调用规则、沟通规范等）

2. **流式处理 (`process_chat_stream`)**
   - 接收消息历史
   - 创建 Agent 实例
   - 创建 StreamProcessor 处理流式输出
   - 返回 SSE 格式的事件流

**环境变量依赖：**
- `OPENAI_API_KEY` - API密钥（必需）
- `OPENAI_BASE_URL` - API基础URL（默认：SiliconFlow）
- `MODEL_NAME` - 模型名称（默认：DeepSeek-V3.1-Terminus）

---

#### 3.2 `app/services/stream_processor.py` - 流式处理器

**职责：**
- 处理 LangGraph 的流式输出
- 转换为 SSE (Server-Sent Events) 格式
- 解析不同类型的消息（文本、工具调用、工具结果）

**核心功能：**

1. **流式处理 (`process_stream`)**
   - 转换消息格式（OpenAI格式 -> LangChain格式）
   - 使用 `agent.astream()` 获取流式输出
   - 处理每个 chunk，转换为 SSE 事件

2. **Chunk 处理 (`_handle_chunk`)**
   - 支持多种 chunk 格式：
     - Tuple 格式：`(chunk_type, chunk_data)`
     - 列表格式：`[message1, message2, ...]`
     - 直接消息对象

3. **消息处理 (`_handle_message_chunk`)**
   - **AIMessageChunk**：处理文本内容和工具调用
     - 文本内容：发送 `delta` 事件（增量文本）
     - 工具调用：发送 `tool_call` 事件
   - **ToolMessage**：处理工具执行结果
     - 发送 `tool_result` 事件

**事件类型：**

| 事件类型 | 说明 | 数据结构 |
|---------|------|---------|
| `delta` | 文本增量 | `{"type": "delta", "content": "..."}` |
| `tool_call` | 工具调用 | `{"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}` |
| `tool_result` | 工具结果 | `{"type": "tool_result", "tool_call_id": "...", "content": "..."}` |
| `error` | 错误信息 | `{"type": "error", "error": "..."}` |
| `[DONE]` | 完成标记 | `"data: [DONE]\n\n"` |

**关键特性：**
- 累积工具调用参数（流式输出中参数可能分多个chunk）
- 文本缓冲区用于日志打印
- 递归限制配置（默认200，可通过环境变量调整）

---

#### 3.3 `app/services/history_service.py` - 历史记录服务

**职责：**
- 管理对话历史（画布数据）
- 持久化到 JSON 文件
- 提供 CRUD 操作

**核心功能：**

1. **数据模型 (`Canvas`)**
   ```python
   {
       "id": str,              # 画布ID
       "name": str,            # 画布名称
       "createdAt": float,     # 创建时间戳
       "images": List[Dict],   # 旧版图片列表（兼容）
       "data": Dict,          # Excalidraw 画布数据
       "messages": List[Dict] # 对话消息列表
   }
   ```

2. **主要方法：**
   - `get_canvases()` - 获取所有画布
   - `save_canvas(canvas_data)` - 保存/更新画布
   - `delete_canvas(canvas_id)` - 删除画布

3. **数据持久化：**
   - 存储位置：`storage/chat_history.json`
   - 格式：JSON 数组
   - 错误处理：自动备份损坏文件并重置

---

### 4. **工具层 (Tools Layer)**

工具层实现了 LangChain Tools，供 Agent 调用。所有工具都遵循 LangChain Tool 规范。

#### 4.1 `app/tools/volcano_image_generation.py` - 火山引擎图片工具

**工具列表：**

1. **`generate_volcano_image_tool`** - 图片生成
   - **输入参数：**
     - `prompt` (str) - 图像生成提示词（中英文）
     - `size` (str) - 图片尺寸（默认 "1:1"）
       - 支持宽高比：`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`, `21:9`
       - 支持自定义：`2048x2048`
       - 支持API格式：`2K`, `4K`
     - `num_images` (int) - 生成数量（默认1）
   - **输出：** JSON字符串，包含：
     - `image_url` / `image_urls` - 本地文件路径
     - `original_url` / `original_urls` - 原始API返回的URL
     - `local_path` / `local_paths` - 本地路径
     - `prompt` - 提示词
     - `provider` - 提供商（"volcano"）
     - `message` - 成功消息

2. **`edit_volcano_image_tool`** - 图片编辑
   - **输入参数：**
     - `prompt` (str) - 编辑提示词（中英文）
     - `image_url` (str) - 源图片URL或本地路径
     - `size` (str) - 输出尺寸（默认 "1:1"）
   - **输出：** 同生成工具

**核心功能：**

1. **图片处理：**
   - `prepare_image_input()` - 准备图片输入（本地文件转Base64）
   - `download_and_save_image()` - 下载并保存图片
   - sRGB 归一化（使用 PIL/ImageCms）
   - 自动格式转换（不透明->JPEG，透明->PNG）

2. **尺寸解析：**
   - `parse_size()` - 解析尺寸参数，支持多种格式

**环境变量：**
- `VOLCANO_API_KEY` - 火山引擎API密钥
- `VOLCANO_BASE_URL` - API基础URL
- `VOLCANO_IMAGE_MODEL` - 生成模型（默认：doubao-seedream-4-5-251128）
- `VOLCANO_EDIT_MODEL` - 编辑模型（默认：同生成模型）

---

#### 4.2 `app/tools/model_3d_generation.py` - 3D模型生成工具

**工具：`generate_3d_model_tool`**

**输入参数：**
- `prompt` (Optional[str]) - 文本提示词（文生3D模式）
- `image_url` (Optional[str]) - 源图片URL或本地路径（图生3D模式）
- `format` (Literal["obj", "glb"]) - 输出格式（默认 "obj"）

**输出：** JSON字符串，包含：
- `model_url` - 模型文件路径（如 `/storage/models/{folder}/model.obj`）
- `local_path` - 本地路径
- `format` - 实际格式（"obj" 或 "glb"）
- `source_image` - 源图片URL
- `preview_url` - 预览图路径
- `job_id` - 任务ID
- `mtl_url` - MTL文件路径（OBJ格式）
- `texture_url` - 纹理文件路径（OBJ格式）
- `message` - 成功消息

**核心功能：**

1. **任务提交：**
   - `submit_3d_generation_task()` - 图生3D模式（可带提示词）
   - `submit_3d_generation_task_with_prompt()` - 文生3D模式

2. **任务查询：**
   - `query_3d_generation_task()` - 轮询任务状态直到完成
   - 支持状态：RUN, PENDING, SUCCESS, FAILED 等
   - 默认最大等待时间：300秒

3. **文件处理：**
   - `download_3d_model()` - 下载模型文件
   - `extract_obj_zip()` - 解压OBJ格式ZIP文件
   - 自动修复OBJ/MTL文件中的路径引用

4. **文件结构（OBJ格式）：**
   ```
   {timestamp}_{uuid}/
   ├── model.obj      # 3D模型文件
   ├── model.mtl      # 材质文件
   ├── texture.png    # 纹理文件
   └── preview.png    # 预览图
   ```

**环境变量：**
- `TENCENT_AI3D_API_KEY` - 腾讯云混元生3D API密钥
- `TENCENT_AI3D_BASE_URL` - API基础URL

---

#### 4.3 `app/tools/image_generation.py` - SiliconFlow图片工具

**工具列表：**

1. **`generate_image_tool`** - 图片生成
   - 使用 SiliconFlow API
   - 输入：`prompt` (str) - 英文提示词
   - 输出：JSON字符串（格式同火山引擎工具）

2. **`edit_image_tool`** - 图片编辑
   - 使用 SiliconFlow API
   - 输入：
     - `prompt` (str) - 英文提示词
     - `image_url` (str) - 源图片URL或本地路径
   - 输出：JSON字符串

**注意：** 当前 Agent 配置中未启用此工具，使用的是火山引擎工具。

**环境变量：**
- `OPENAI_API_KEY` - API密钥
- `OPENAI_BASE_URL` - API基础URL（默认：SiliconFlow）
- `IMAGE_MODEL_NAME` - 生成模型（默认：Qwen/Qwen-Image）
- `EDIT_IMAGE_MODEL_NAME` - 编辑模型（默认：Qwen/Qwen-Image-Edit-2509）

---

## 🔄 数据流

### 聊天请求流程

```
1. 前端发送 POST /api/chat
   ↓
2. chat.py 接收请求，验证参数
   ↓
3. agent_service.py 创建 Agent 和 StreamProcessor
   ↓
4. stream_processor.py 处理流式输出
   ├─ 转换消息格式
   ├─ 调用 agent.astream()
   └─ 处理每个 chunk
      ├─ AIMessageChunk → delta 事件
      ├─ ToolCall → tool_call 事件
      └─ ToolMessage → tool_result 事件
   ↓
5. 工具执行（tools/）
   ├─ 调用外部API（火山引擎/腾讯云）
   ├─ 下载文件
   ├─ 保存到 storage/
   └─ 返回结果JSON
   ↓
6. 流式响应返回前端（SSE格式）
```

### 文件存储结构

```
storage/
├── images/                    # 图片存储
│   ├── volcano_*.jpg         # 火山引擎生成的图片
│   ├── upload_*.jpg          # 用户上传的图片
│   └── ...
├── models/                    # 3D模型存储
│   └── {timestamp}_{uuid}/    # 每个模型一个文件夹
│       ├── model.obj          # OBJ格式模型
│       ├── model.mtl          # 材质文件
│       ├── texture.png        # 纹理文件
│       └── preview.png        # 预览图
└── chat_history.json          # 对话历史
```

---

## 🔧 配置说明

### 环境变量（.env）

```bash
# LLM 配置
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=deepseek-ai/DeepSeek-V3.1-Terminus

# 图片生成（SiliconFlow，当前未使用）
IMAGE_MODEL_NAME=Qwen/Qwen-Image
EDIT_IMAGE_MODEL_NAME=Qwen/Qwen-Image-Edit-2509

# 基础URL（用于图片编辑）
BASE_URL=http://localhost:8000

# LangGraph 配置
RECURSION_LIMIT=200

# 火山引擎配置
VOLCANO_API_KEY=YOUR_VOLCANO_API_KEY
VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCANO_IMAGE_MODEL=doubao-seedream-4-5-251128
VOLCANO_EDIT_MODEL=doubao-seedream-4-5-251128

# 腾讯云混元生3D配置
TENCENT_AI3D_API_KEY=YOUR_TENCENT_AI3D_API_KEY
TENCENT_AI3D_BASE_URL=https://api.ai3d.cloud.tencent.com
```

---

## 📦 依赖包

主要依赖（`requirements.txt`）：

- **Web框架：**
  - `fastapi==0.104.1` - Web框架
  - `uvicorn[standard]==0.24.0` - ASGI服务器

- **LangChain生态：**
  - `langchain==1.0.0` - LangChain核心
  - `langchain-core==1.0.0` - 核心组件
  - `langchain-openai==1.0.0` - OpenAI兼容接口
  - `langgraph>=1.0.0,<1.1.0` - 图状态机
  - `langgraph-prebuilt>=1.0.0,<1.1.0` - 预构建Agent

- **工具库：**
  - `python-dotenv==1.0.0` - 环境变量管理
  - `pydantic>=2.7.4,<3.0.0` - 数据验证
  - `requests>=2.31.0` - HTTP请求
  - `httpx==0.25.2` - 异步HTTP客户端
  - `Pillow==10.4.0` - 图像处理
  - `aiofiles==23.2.1` - 异步文件操作

---

## 🚀 启动流程

1. **环境准备：**
   ```bash
   cd backend
   cp env.example .env
   # 编辑 .env 填写API密钥
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动服务：**
   ```bash
   # 方式1：使用启动脚本
   ./start.sh
   
   # 方式2：直接运行
   python -m app.main
   
   # 方式3：使用uvicorn
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **验证：**
   - 访问 `http://localhost:8000/` - 返回API信息
   - 访问 `http://localhost:8000/health` - 健康检查
   - 访问 `http://localhost:8000/docs` - API文档（Swagger UI）

---

## 🎯 核心特性

### 1. **流式输出**
- 使用 SSE (Server-Sent Events) 实现真正的流式输出
- 支持增量文本、工具调用、工具结果的实时推送
- 前端可以实时显示AI响应和工具执行状态

### 2. **多模态支持**
- **图片生成/编辑：** 火山引擎 Seedream 4.5
- **3D模型生成：** 腾讯云混元生3D（支持文生3D和图生3D）
- **图片上传：** 支持用户上传图片

### 3. **智能工具调用**
- Agent 自动选择合适的工具
- 支持从对话历史中查找图片URL
- 一次只调用一个工具，避免并行冲突

### 4. **文件管理**
- 自动下载并保存生成的文件
- 本地路径存储，避免URL过期
- 支持sRGB归一化，确保颜色一致性

### 5. **历史记录**
- JSON文件持久化
- 支持画布（Excalidraw）数据保存
- 自动错误恢复和备份

---

## 🔍 调试和日志

### 日志级别
- `INFO` - 一般信息（工具调用、文件保存等）
- `DEBUG` - 详细调试信息（chunk处理、事件发送等）
- `WARNING` - 警告信息（文件格式、API响应等）
- `ERROR` - 错误信息（异常、失败等）

### 关键日志点
- Agent创建和工具注册
- 流式处理开始/完成
- 工具调用和结果
- 文件下载和保存
- API请求和响应

---

## 📝 扩展指南

### 添加新工具

1. 在 `app/tools/` 创建新工具文件
2. 使用 `@tool` 装饰器定义工具
3. 在 `agent_service.py` 中注册工具：
   ```python
   tools = [
       # ... 现有工具
       your_new_tool,
   ]
   ```

### 添加新路由

1. 在 `app/routers/` 创建新路由文件
2. 在 `main.py` 中注册路由：
   ```python
   from app.routers import your_router
   app.include_router(your_router.router, prefix="/api", tags=["your_tag"])
   ```

### 修改Agent Prompt

编辑 `agent_service.py` 中的 `create_agent()` 函数的 `prompt` 参数。

---

## ⚠️ 注意事项

1. **API密钥安全：**
   - 不要将 `.env` 文件提交到版本控制
   - 生产环境使用环境变量或密钥管理服务

2. **文件存储：**
   - `storage/` 目录需要足够的磁盘空间
   - 定期清理旧文件（可编写清理脚本）

3. **并发处理：**
   - 当前实现为单进程，如需高并发需部署多个实例
   - 考虑使用 Redis 等共享存储管理会话状态

4. **错误处理：**
   - 所有工具调用都有异常处理
   - 流式输出中的错误会通过 `error` 事件返回

5. **性能优化：**
   - 图片下载和保存是同步的，可考虑异步优化
   - 3D模型生成是长时间任务，考虑使用任务队列

---

## 📚 相关文档

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [火山引擎 API 文档](https://www.volcengine.com/docs/82379)
- [腾讯云混元生3D API 文档](https://cloud.tencent.com/document/product/1729)

---

**最后更新：** 2025-01-02

