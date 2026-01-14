# PolyStudio

一个「对话式生图/改图 + 可视化画板」项目：FastAPI 后端通过 LangGraph 编排工具调用，React 前端用 Excalidraw 做画板承载与项目管理，支持 SSE 流式输出、图片自动落到画板、项目链接与历史记录。

> 仓库总览与其他子项目导航见根目录：`README.md`

### 功能概览

- **对话式生成/编辑图片**：支持 `generate_volcano_image` / `edit_volcano_image`（火山引擎图片生成）
- **3D 模型生成**：支持基于文本或图片生成 3D 模型（OBJ/GLB 格式），使用腾讯云混元生3D API
  - **文生3D**：通过文本描述生成 3D 模型
  - **图生3D**：基于图片生成 3D 模型
  - **混合模式**：同时使用文本和图片生成 3D 模型
- **SSE 流式输出**：逐步展示工具调用与结果
- **Excalidraw 画板**：生成的图片自动插入画布，支持缩放/对齐/框选
- **3D 模型查看器**：前端集成 3D 模型预览功能，支持 OBJ/GLB 格式
- **项目管理**：项目列表、重命名、复制链接、删除
- **全局深色主题**：前端与画板默认深色
- **本地持久化**：
  - 图片保存到 `backend/storage/images/`
  - 3D 模型保存到 `backend/storage/models/`（包含 OBJ、MTL、纹理文件）
  - 项目与聊天记录保存到 `backend/storage/chat_history.json`

### 目录结构

```
PolyStudio/
├── backend/                 # FastAPI 后端
│   ├── app/                 # 业务代码
│   │   ├── tools/           # 工具模块（图片生成、3D模型生成）
│   │   ├── services/        # 服务模块（Agent服务、流式处理）
│   │   └── utils/           # 工具函数（日志配置等）
│   ├── requirements.txt     # Python 依赖（以此为准）
│   ├── start.sh             # 推荐的后端启动脚本（确保用正确的 Python 环境）
│   ├── scripts/             # 维护脚本（如存量图片归一化）
│   ├── storage/              # 运行数据
│   │   ├── images/          # 生成的图片
│   │   ├── models/          # 生成的3D模型（OBJ、MTL、纹理文件）
│   │   └── chat_history.json # 聊天历史记录
│   └── logs/                # 日志文件（按日期和大小自动轮转）
├── frontend/                # React + Vite 前端
│   ├── src/components/      # ChatInterface / ExcalidrawCanvas / Model3DViewer / HomePage
│   └── vite.config.ts       # /api、/storage 代理
└── README.md                # 项目说明文档
```

### 快速开始

#### 环境要求

- **Python**：3.9+
- **Node.js**：18+

#### 后端启动（FastAPI）

0) （推荐）创建并进入 conda 环境：

```bash
conda create -n agentImage python=3.11 -y
conda activate agentImage
```

> 注意：`backend/start.sh` 默认会 `conda activate agentImage`。如果你用别的环境名，请同步修改该脚本里的环境名。

1) 安装依赖：

```bash
cd backend
pip install -r requirements.txt
```

2) 配置环境变量（必需）：在 `backend/.env` 写入（详见 `env.example`）

推荐方式：

```bash
cd backend
cp env.example .env
# 然后编辑 .env 文件，填入必要的 API Key
```

**必需的环境变量：**
- `OPENAI_API_KEY`：LLM API 密钥（用于对话模型）
- `VOLCANO_API_KEY`：火山引擎 API 密钥（用于图片生成）
- `TENCENT_AI3D_API_KEY`：腾讯云混元生3D API 密钥（用于 3D 模型生成）

**可选的环境变量：**
- `MOCK_MODE`：设置为 `true` 启用 Mock 模式（调试用，不调用真实 API）
- `LOG_LEVEL`：日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

3) 启动后端（推荐）：

```bash
cd backend
chmod +x start.sh
./start.sh
```

或直接：

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端默认地址：`http://localhost:8000`

#### 前端启动（Vite）

```bash
cd frontend
npm install
npm run dev
```

前端默认地址：`http://localhost:3000`

`frontend/vite.config.ts` 已配置代理：
- `/api` -> `http://localhost:8000`
- `/storage` -> `http://localhost:8000`

### API（SSE）

#### POST `/api/chat`

请求体：

```json
{
  "message": "生成一张赛博朋克城市夜景",
  "messages": [],
  "session_id": "optional-session-id"
}
```

响应（SSE）示例：

```
data: {"type":"delta","content":"..."}
data: {"type":"tool_call","id":"...","name":"generate_volcano_image","arguments":{...}}
data: {"type":"tool_result","tool_call_id":"...","content":"{...image_url...}"}
data: {"type":"tool_call","id":"...","name":"generate_3d_model","arguments":{...}}
data: {"type":"tool_result","tool_call_id":"...","content":"{...model_path...}"}
data: [DONE]
```

**支持的工具：**
- `generate_volcano_image`：生成图片（火山引擎）
- `edit_volcano_image`：编辑图片（火山引擎）
- `generate_3d_model`：生成 3D 模型（腾讯云混元生3D）
  - 参数：`prompt`（文本描述，可选）、`image_url`（图片URL，可选）、`format`（输出格式：obj/glb）

### 技术栈/服务说明（以代码为准）

- **LLM（对话模型）**：通过 `OPENAI_BASE_URL` 指向 OpenAI 兼容接口（默认 SiliconFlow），模型名由 `MODEL_NAME` 指定
- **图像生成/编辑**：后端 `generate_volcano_image` / `edit_volcano_image` 工具调用火山引擎 Seedream API（并将结果下载保存到本地 `/storage/images`）
- **3D 模型生成**：后端 `generate_3d_model` 工具调用腾讯云混元生3D API，支持文生3D、图生3D和混合模式，生成的模型保存到本地 `/storage/models/`
- **颜色一致性**：后端保存图片时会尝试做 sRGB 归一化（依赖 `Pillow`，已在 `requirements.txt` 固定）
- **日志系统**：统一的日志配置，支持输出到控制台和文件（`backend/logs/`），可按日期和大小自动轮转
- **Mock 模式**：支持启用 Mock 模式用于调试，返回固定的图片和 3D 模型数据，无需调用真实 API

### 常见问题（简版）

- **建议不要手动混装 langchain 版本**：以 `backend/requirements.txt` 为准安装，避免出现导入错误/版本不兼容。

