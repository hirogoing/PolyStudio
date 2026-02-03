# 🎨 PolyStudio

[Python 3.9+](https://www.python.org/downloads/) | [Node.js 18+](https://nodejs.org/) | [License: MIT](https://opensource.org/licenses/MIT)

PolyStudio 是一个对话式多模态内容生成平台，将语言模型与专业工具（如图片生成、视频生成、3D 模型生成）相结合，通过自然语言对话即可生成和管理多媒体内容。项目采用 FastAPI + LangGraph 构建智能 Agent 编排系统，提供无限画板承载与项目管理，支持 SSE 流式输出、自动内容插入、项目链接分享等功能。

PolyStudio 支持灵活的 API 接入，你可以轻松替换为任意图片、视频、3D 生成服务，打造属于自己的多模态创作工具。

### 功能概览

- **虚拟人生成**：支持图片 + 音频生成口型同步的虚拟主播视频，基于 ComfyUI 工作流，支持 OpenCV 或 LLM 两种人脸检测方式进行质量验证 🆕
- **音频上传与管理**：前端支持音频文件上传（MP3/WAV/M4A 等格式），在对话中可视化展示，保存到本地 🆕
- **长视频工作流**：角色一致性分镜生成 → 生图 → 图生视频 → 拼接的完整流程，兼容 moviepy 2.x 🆕
- **路径自动处理**：视频工具支持本地/公网/localhost 路径自动处理（含 base64 转换与下载） 🆕
- **统一 Mock 模式**：图片/视频/3D/虚拟人工具统一 Mock 模式，便于离线或调试 🆕
- **SSE 流式输出优化**：实时推送 delta、工具调用与结果 🆕
- **对话式生成/编辑图片**：支持图片生成和编辑，可替换为任意图片生成 API
- **视频生成**：支持基于文本、图片或首尾帧生成视频，支持图片 URL 和本地路径输入（本地路径如 `/storage/images/xxx.jpg` 会自动转换为 base64），可替换为任意视频生成 API
- **3D 模型生成**：支持基于文本或图片生成 3D 模型（OBJ/GLB 格式），可替换为任意 3D 生成 API
- **无限画板**：生成的图片和视频自动插入画布，支持缩放/对齐/框选/编辑等，视频双击可播放
- **3D 模型查看器**：前端集成 3D 模型预览功能，支持 OBJ/GLB 格式，双击可预览
- **项目管理**：项目列表、重命名、复制链接、删除
- **全局主题切换**：前端与画板支持深色和浅色模式，默认深色
- **本地持久化**：图片保存到 `backend/storage/images/`，视频保存到 `backend/storage/videos/`，音频保存到 `backend/storage/audios/`，3D 模型保存到 `backend/storage/models/`（包含 OBJ、MTL、纹理文件），项目与聊天记录保存到 `backend/storage/chat_history.json`


### 界面展示

#### 首页
![首页](assets/首页.png)

#### 编辑页
![编辑页](assets/编辑页.png)

![编辑页2](assets/编辑页2.png)

![编辑页-视频](assets/编辑页-视频.png)

![编辑页虚拟人](assets/编辑页虚拟人.png)

#### 虚拟人视频展示

<table>
  <tr>
    <td width="33%">
      <video src="assets/虚拟人效果1.mp4" controls width="100%"></video>
      <p align="center">虚拟主播示例1</p>
    </td>
    <td width="33%">
      <video src="assets/虚拟人效果2.mp4" controls width="100%"></video>
      <p align="center">虚拟主播示例2</p>
    </td>
    <td width="33%">
      <video src="assets/虚拟人效果3.mp4" controls width="100%"></video>
      <p align="center">虚拟主播示例3</p>
    </td>
  </tr>
</table>


### 目录结构

```
PolyStudio/
├── backend/                 # FastAPI 后端
│   ├── app/                 # 业务代码
│   │   ├── tools/           # 工具模块（图片生成、视频生成、3D模型生成、虚拟人生成）
│   │   ├── services/        # 服务模块（Agent服务、流式处理）
│   │   └── utils/           # 工具函数（日志配置等）
│   ├── requirements.txt     # Python 依赖（以此为准）
│   ├── start.sh             # 推荐的后端启动脚本（确保用正确的 Python 环境）
│   ├── scripts/             # 维护脚本（如存量图片归一化）
│   ├── storage/              # 运行数据
│   │   ├── images/          # 生成的图片
│   │   ├── videos/          # 生成的视频
│   │   ├── audios/          # 上传的音频
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
- `OPENAI_API_KEY`：LLM API 密钥（用于对话模型，当 `LLM_PROVIDER=siliconflow` 时使用）
- 图片/视频生成 API 密钥（根据你使用的 API 提供商配置，如 `VOLCANO_API_KEY` 等）
- 3D 模型生成 API 密钥（根据你使用的 API 提供商配置，如 `TENCENT_AI3D_API_KEY` 等）

**虚拟人生成配置（可选）：**
- `COMFYUI_SERVER_ADDRESS`：ComfyUI 服务器地址（用于虚拟人生成）
- `COMFYUI_WORKFLOW_PATH`：ComfyUI 工作流文件路径（JSON 格式）
- `FACE_DETECTION_METHOD`：人脸检测方式，可选值：`opencv`（默认）、`llm`

**其他可选环境变量：**
- `LLM_PROVIDER`：LLM 提供商，可选值：`volcano`（默认）、`siliconflow`
- `MOCK_MODE`：设置为 `true` 启用 Mock 模式（调试用，不调用真实 API）
  - 启用 Mock 模式时，必须同时配置 `MOCK_IMAGE_PATH`、`MOCK_VIDEO_PATH`、`MOCK_MODEL_PATH` 和 `MOCK_VIRTUAL_ANCHOR_PATH`
- `LOG_LEVEL`：日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL，默认：INFO）

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

### 技术栈/服务说明（以代码为准）

- **LLM（对话模型）**：支持多种提供商，可通过 `LLM_PROVIDER` 配置切换，可扩展为任意 OpenAI 兼容接口
- **图像生成/编辑**：后端工具调用图片生成 API（结果下载保存到本地 `/storage/images`），可替换为任意图片生成服务
- **视频生成**：后端工具调用视频生成 API，支持文本、图片或首尾帧生成模式，生成的视频保存到本地 `/storage/videos`，支持图片 URL 和本地路径输入（本地路径和 localhost URL 会自动转换为 base64，公网 URL 直接使用），支持自定义视频参数（时长、宽高比等）
- **3D 模型生成**：后端工具调用 3D 生成 API，支持文本、图片或混合模式，生成的模型保存到本地 `/storage/models/`，可替换为任意 3D 生成服务
- **虚拟人生成**：基于 ComfyUI 集成，支持图片 + 音频生成口型同步的虚拟主播视频
  - 支持两种人脸检测方式：OpenCV（快速）或 LLM（精准）
  - 自动上传图片和音频到 ComfyUI 服务器
  - 轮询任务状态，生成完成后自动下载并保存到本地
  - Mock 模式支持，便于调试
- **音频处理**：支持前端上传音频文件（MP3/WAV/M4A/AAC/OGG/FLAC/WMA），保存到 `backend/storage/audios/`，在对话中可视化展示
- **颜色一致性**：后端保存图片时会尝试做 sRGB 归一化（依赖 `Pillow`，已在 `requirements.txt` 固定）
- **日志系统**：统一的日志配置，支持输出到控制台和文件（`backend/logs/`），可按日期和大小自动轮转
- **Mock 模式**：支持启用 Mock 模式用于调试，返回固定的图片、视频、3D 模型和虚拟人数据，无需调用真实 API
  - 启用方式：在 `.env` 中设置 `MOCK_MODE=true`
  - 必须配置：`MOCK_IMAGE_PATH`、`MOCK_VIDEO_PATH`、`MOCK_MODEL_PATH`、`MOCK_VIRTUAL_ANCHOR_PATH`（分别指向 `/storage/` 下的实际文件路径）

### 常见问题（简版）

- **建议不要手动混装 langchain 版本**：以 `backend/requirements.txt` 为准安装，避免出现导入错误/版本不兼容。

