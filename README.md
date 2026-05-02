# 🎨 PolyStudio

[Python 3.9+](https://www.python.org/downloads/) | [Node.js 18+](https://nodejs.org/) | [License: MIT](https://opensource.org/licenses/MIT)

PolyStudio 是一个对话式多模态内容生成平台，将语言模型与专业工具（如图片生成、视频生成、3D 模型生成、多模态内容理解）相结合，通过自然语言对话即可生成、分析和管理多媒体内容。项目采用 FastAPI + LangGraph 构建智能 Agent 编排系统，提供无限画板承载与项目管理，支持 SSE 流式输出、自动内容插入、项目链接分享等功能。
PolyStudio 支持灵活的 API 接入，你可以轻松替换为任意图片、视频、3D 生成服务，打造属于自己的多模态创作工具。同时支持通过 SKILL.md 编写自定义 Skill，扩展 Agent 的专项领域能力。支持与 [OpenClaw](https://openclaw.ai) 联动——将 `polystudio-client/SKILL.md` 配置为 Skill 后，外部 Agent 可直接通过自然语言驱动 PolyStudio 完成创作，生成的内容自动同步到画布。

### 功能概览
- **外部 Agent 接入（polystudio-client）**：通过 `curl` / HTTP 驱动 PolyStudio，支持 SSE 流式消费、媒体 URL 提取、多轮对话，即使用户不在页面也能将生成内容同步到画布 🆕
- **画布后台媒体同步**：外部 Agent 在用户离开画布页面期间生成的图片/视频/3D 模型，回到页面后自动补充插入画布，不再丢失 🆕
- **内置 Skills 扩展**：新增 `paper-writing`（论文写作）、`video-creator`（长视频生成）、`virtual-anchor`（虚拟人合成）、`podcast-creator`（播客制作）四个开箱即用的 Custom Skill 🆕
- **工作空间身份记忆**：通过 AGENTS.md / TOOLS.md / IDENTITY.md / USER.md / SOUL.md / MEMORY.md 为 Agent 注入跨对话长期记忆，设置页可视化编辑
- **设置页面**：全新可视化配置中心，支持工具/Skill/MCP/环境变量/工作空间的可视化配置
- **Skill 扩展系统**：通过编写 SKILL.md 文件扩展 Agent 的专项领域能力，支持 Progressive Loading 按需加载
- **多模态内容理解**：基于 Qwen3-Omni-Flash，支持图片/音频/视频智能分析，文字+语音双模态输出，支持「理解 → 创作」联动
- **视频上传与管理**：前端支持视频文件上传（MP4/MOV/AVI/MKV/WEBM），在对话中可视化展示，保存到本地
- **智能播客生成**：从脚本创作到混音输出的完整音频内容生成工作流
- **虚拟人生成**：支持图片 + 音频生成口型同步的虚拟主播视频，基于 ComfyUI 工作流，支持 OpenCV 或 LLM 两种人脸检测方式进行质量验证
- **音频上传与管理**：前端支持音频文件上传（MP3/WAV/M4A 等格式），在对话中可视化展示，保存到本地
- **长视频工作流**：角色一致性分镜生成 → 生图 → 图生视频 → 拼接的完整流程，兼容 moviepy 2.x
- **路径自动处理**：视频工具支持本地/公网/localhost 路径自动处理（含 base64 转换与下载）
- **统一 Mock 模式**：图片/视频/3D/虚拟人工具统一 Mock 模式，便于离线或调试
- **WebSocket 实时广播**：`/ws/{canvas_id}` 端点，外部客户端发送带 `canvas_id` 的请求时，SSE 事件同步广播给所有订阅该画布的 WebSocket 连接
- **SSE 流式输出**：实时推送 delta、工具调用与结果
- **对话式生成/编辑图片**：支持图片生成和编辑，可替换为任意图片生成 API
- **视频生成**：支持基于文本、图片或首尾帧生成视频，支持图片 URL 和本地路径输入（本地路径如 `/storage/images/xxx.jpg` 会自动转换为 base64），可替换为任意视频生成 API
- **3D 模型生成**：支持基于文本或图片生成 3D 模型（OBJ/GLB 格式），可替换为任意 3D 生成 API
- **无限画板**：生成的图片和视频自动插入画布，支持缩放/对齐/框选/编辑等，视频双击可播放
- **3D 模型查看器**：前端集成 3D 模型预览功能，支持 OBJ/GLB 格式，双击可预览
- **项目管理**：项目列表、重命名、复制链接、删除
- **全局主题切换**：前端与画板支持深色和浅色模式，默认深色
- **本地持久化**：图片保存到 `backend/storage/images/`，视频保存到 `backend/storage/videos/`，音频保存到 `backend/storage/audios/`，播客保存到 `backend/storage/podcasts/`，BGM 保存到 `backend/storage/bgm/`，3D 模型保存到 `backend/storage/models/`（包含 OBJ、MTL、纹理文件），项目与聊天记录保存到 `backend/storage/chat_history.json`


### 界面展示
![SKills](assets/skills演示.gif)

#### 首页
![首页](assets/首页.png)

#### 编辑页
![编辑页](assets/编辑页.png)

![编辑页2](assets/编辑页2.png)

![编辑页-视频](assets/编辑页-视频.png)

![编辑页虚拟人](assets/编辑页虚拟人.png)

![编辑页-TTS](assets/编辑页-TTS.png)

#### 配置页
![配置页](assets/配置页.png)

![Skills使用](assets/skills的使用.png)

#### OpenClaw 联动

外部 Agent（OpenClaw）通过自然语言驱动 PolyStudio，生成内容实时同步到画布：

![OpenClaw联动1](assets/openclaw_skill1.png)

![OpenClaw联动2](assets/openclaw_skill2.png)

#### 虚拟人效果展示

<table>
  <tr>
    <td width="33%" align="center">
      <a href="assets/虚拟人效果1.mp4">
        <img src="assets/虚拟人预览1.jpg" width="100%" />
      </a>
      <p>点击查看虚拟主播示例1</p>
    </td>
    <td width="33%" align="center">
      <a href="assets/虚拟人效果2.mp4">
        <img src="assets/虚拟人预览2.jpg" width="100%" />
      </a>
      <p>点击查看虚拟主播示例2</p>
    </td>
    <td width="33%" align="center">
      <a href="assets/虚拟人效果3.mp4">
        <img src="assets/虚拟人预览3.jpg" width="100%" />
      </a>
      <p>点击查看虚拟主播示例3</p>
    </td>
  </tr>
</table>

> 💡 提示：点击上方图片可下载/查看对应的虚拟人视频演示


### 目录结构

```
PolyStudio/
├── polystudio-client/       # 外部 Agent 接入指南 🆕
│   └── SKILL.md             # curl 驱动 PolyStudio 的完整文档（SSE 消费、多轮对话、媒体提取）
├── backend/                 # FastAPI 后端
│   ├── app/                 # 业务代码
│   │   ├── tools/           # 工具模块
│   │   │   ├── qwen_tts.py          # Qwen-TTS 语音合成（声音设计、声音复刻）
│   │   │   ├── qwen_omni_understanding.py # Qwen3-Omni 多模态理解（图片/音频/视频）
│   │   │   ├── audio_mixing.py     # 音频混音工具（拼接、BGM、混音）
│   │   │   ├── volcano_image_generation.py  # 图片生成/编辑
│   │   │   ├── volcano_video_generation.py  # 视频生成
│   │   │   ├── model_3d_generation.py       # 3D模型生成
│   │   │   ├── virtual_anchor_generation.py # 虚拟人生成
│   │   │   ├── video_concatenation.py       # 视频拼接
│   │   │   ├── skill_tools.py               # Skill 工具（read/list/init/write/delete）
│   │   │   └── workspace_tools.py           # 工作空间记忆写入工具（write_memory）
│   │   ├── routers/         # API 路由模块
│   │   │   ├── chat.py             # 对话/SSE 接口
│   │   │   └── settings.py         # 设置页 API（工具/Skill/MCP/环境变量/工作空间）
│   │   ├── llm/             # LLM 多提供商封装
│   │   │   ├── base.py             # 基类接口
│   │   │   ├── factory.py          # 工厂方法（按 LLM_PROVIDER 切换）
│   │   │   ├── volcano.py          # 火山引擎
│   │   │   └── siliconflow.py      # SiliconFlow
│   │   ├── services/        # 服务模块
│   │   │   ├── agent_service.py    # Agent服务
│   │   │   ├── prompt.py           # Agent提示词（模块化）
│   │   │   ├── skill_service.py    # Skill 扫描/启用状态管理
│   │   │   ├── workspace_service.py # 工作空间身份记忆管理
│   │   │   ├── connection_manager.py # WebSocket 连接管理
│   │   │   └── stream_processor.py # 流式处理（含 skill_matched 事件）
│   │   └── utils/           # 工具函数（日志配置等）
│   ├── skills/              # Skill 扩展目录
│   │   ├── public/          # 内置 Skills（随项目分发，默认始终启用）
│   │   │   └── skill-creator/      # 快速创建新 Skill 的引导 Skill
│   │   └── custom/          # 用户自定义 Skills
│   │       ├── xiaohongshu-copywriter/ # 小红书文案创作
│   │       ├── paper-writing/      # 论文写作全流程 🆕
│   │       ├── video-creator/      # 长视频生成（分镜→生图→视频→拼接）🆕
│   │       ├── virtual-anchor/     # 虚拟人视频合成 🆕
│   │       └── podcast-creator/    # 播客制作（脚本→音色→合成→混音）🆕
│   ├── workspace/           # 工作空间身份记忆文件目录
│   │   ├── AGENTS.md        # 行为规则（工作流约定、禁止操作、快捷指令）
│   │   ├── TOOLS.md         # 工具配置（服务地址、模型偏好等本地参数）
│   │   ├── IDENTITY.md      # Agent 身份（名字、气质、签名等）
│   │   ├── USER.md          # 用户画像（风格偏好、常用角色等）
│   │   ├── SOUL.md          # Agent 人格（沟通风格、创作偏好）
│   │   └── MEMORY.md        # 创作记忆（Agent 自动维护的长期记忆）
│   ├── requirements.txt     # Python 依赖（以此为准）
│   ├── start.sh             # 推荐的后端启动脚本（确保用正确的 Python 环境）
│   ├── scripts/             # 维护脚本（如存量图片归一化）
│   ├── storage/              # 运行数据
│   │   ├── images/          # 生成的图片
│   │   ├── videos/          # 生成的视频
│   │   ├── audios/          # 上传的音频 + TTS生成的音频
│   │   ├── podcasts/        # 混音后的播客成品
│   │   ├── bgm/             # 背景音乐库（文件名作为场景描述）
│   │   ├── models/          # 生成的3D模型（OBJ、MTL、纹理文件）
│   │   ├── settings.json    # 工具/Skill 启用状态、MCP 服务器配置
│   │   └── chat_history.json # 聊天历史记录
│   └── logs/                # 日志文件（按日期和大小自动轮转）
├── frontend/                # React + Vite 前端
│   ├── src/components/      # ChatInterface / ExcalidrawCanvas / Model3DViewer / HomePage / SettingsPage
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

> **注意**：播客生成功能需要安装 `pydub`（已包含在 requirements.txt 中）。如果单独安装：`pip install pydub`

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

**TTS语音合成 & 多模态理解配置：**
- `DASHSCOPE_API_KEY`：阿里云百炼 API 密钥（用于 Qwen-TTS 语音合成 + Qwen3-Omni 多模态理解）
- `DASHSCOPE_BASE_URL`：API 地址，国内版：`https://dashscope.aliyuncs.com`

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

#### BGM 配置（播客生成可选）

如果需要使用播客生成功能的 BGM 混音，需要在 `backend/storage/bgm/` 目录添加背景音乐文件：

```bash
cd backend/storage/bgm/
# 添加 MP3/WAV 文件，文件名作为场景描述，例如：
# 欢快的开场音乐.mp3
# 科技感电子音乐.mp3
# 轻松聊天背景.mp3
# 深沉专业讨论.mp3
```

Agent 会根据播客主题智能匹配最合适的 BGM。

### 技术栈/服务说明（以代码为准）

- **外部 Agent 接入（polystudio-client）**：任意外部 Agent 均可通过 HTTP + curl 驱动 PolyStudio，无需安装额外依赖 🆕
  - `POST /api/chat` 接受 `message` + 可选 `canvas_id`，返回 `text/event-stream` SSE 流
  - 外部 Agent 消费 SSE 直到 `data: [DONE]`，从 `tool_result` 事件中提取媒体 URL
  - 支持自生成 `canvas_id` 实现多轮对话（无需首轮创建再查询）
  - 结合 WebSocket `/ws/{canvas_id}` 可在不占用 SSE 连接的情况下订阅画布实时事件
  - 完整接入文档见 `polystudio-client/SKILL.md`
- **画布后台媒体同步**：外部 Agent 在用户离开画布页面期间生成的媒体不再丢失 🆕
  - 前端维护待插入队列（`pendingMediaRef`），画布未挂载时将图片/视频/3D 模型请求入队
  - 用户返回画布页面后，`ExcalidrawCanvas` 挂载完成触发 `onReady`，自动批量插入队列内容
  - `addVideo` 增加 4 秒超时与 fallback 缩略图，解决后台标签页 `video.onseeked` 永不触发的问题
- **内置 Custom Skills**：新增四个开箱即用的 Custom Skill，放置在 `backend/skills/custom/` 🆕
  - `paper-writing`：论文写作全流程（选题 → 文献综述 → 结构规划 → 排版）
  - `video-creator`：长视频生成（角色一致性分镜 → 生图 → 图生视频 → 拼接）
  - `virtual-anchor`：虚拟人视频合成（角色生成 → 人脸检测 → 口型同步）
  - `podcast-creator`：播客制作（脚本 → 音色确认 → 批量合成 → BGM 混音）

- **Skill 扩展系统**：基于 SKILL.md 的 Progressive Loading 技能框架
  - 扫描 `skills/public/` 和 `skills/custom/` 目录，解析 YAML frontmatter 获取 name/description
  - `public/` 下的 Skill 默认始终启用（无需手动安装）；`custom/` 下的 Skill 通过设置页开关控制
  - Agent 仅在 system prompt 中注入元数据（名称 + 描述 + 文件路径），需要时通过 `read_skill_file` 工具按需读取全文，节省 context window
  - 新增 `init_skill` / `write_skill_file` / `delete_skill_file` 工具，Agent 可在对话中直接创建自定义 Skill
  - 命中 Skill 时后端发送 `skill_matched` SSE 事件，前端以技能标识卡片呈现，相关工具调用步骤自动归组
  - 内置 `skill-creator`（引导创建新 Skill）和 `xiaohongshu-copywriter`（小红书文案）两个参考 Skill
  - 支持用户将自定义 SKILL.md 放入 `skills/custom/<skill-id>/` 目录，无需重启即可在设置页启用
- **工作空间身份记忆**：基于 `backend/workspace/` 目录的跨对话长期记忆系统
  - 6 个 Markdown 文件（AGENTS.md / TOOLS.md / IDENTITY.md / USER.md / SOUL.md / MEMORY.md）在每次对话时注入 System Prompt
  - 首次运行自动生成默认模板；仅注入有实质内容的文件（纯模板跳过），节省 token
  - MEMORY.md 由 Agent 通过 `write_memory` 工具自动写入（角色资产、成功 Prompt 模板、用户偏好）
  - 设置页新增「工作空间」标签，支持可视化编辑所有文件（含 Markdown 预览）
  - 后端 `/api/settings/workspace` 路由提供 GET / GET by id / PUT 接口
- **设置页面**：前端 `SettingsPage` + 后端 `/api/settings/*` 路由，支持工具/Skill/MCP/环境变量/工作空间的可视化配置
  - 工具启用状态、MCP 服务器配置持久化到 `storage/settings.json`
  - 环境变量直接读写 `backend/.env` 文件，敏感字段（API Key 等）默认密码框隐藏
  - MCP 配置支持表单视图和 JSON 视图双向同步编辑
  - 新增「工作空间」标签，支持可视化编辑 workspace/ 下的身份记忆文件（含 Markdown 预览）
- **LLM 多提供商封装**：抽象 `base.py` 接口，通过 `LLM_PROVIDER` 环境变量切换火山引擎/SiliconFlow，可扩展为任意 OpenAI 兼容接口
- **智能播客生成**：完整的音频内容生成工作流
  - **语音合成**：基于阿里云百炼 Qwen-TTS，支持声音设计（文本描述音色）和声音复刻（参考音频克隆）
  - **音色确认**：生成音色样本供用户试听，确认满意后再批量合成
  - **音频处理**：基于 pydub，支持音频拼接、BGM 匹配、专业混音
  - **BGM 效果**：支持 BGM 开场效果（原声播放3秒后平滑过渡到5%背景音量）
  - **智能匹配**：根据播客主题和场景描述，从 BGM 库中智能匹配合适的背景音乐
- **图像生成/编辑**：后端工具调用图片生成 API（结果下载保存到本地 `/storage/images`），可替换为任意图片生成服务
- **视频生成**：后端工具调用视频生成 API，支持文本、图片或首尾帧生成模式，生成的视频保存到本地 `/storage/videos`，支持图片 URL 和本地路径输入（本地路径和 localhost URL 会自动转换为 base64，公网 URL 直接使用），支持自定义视频参数（时长、宽高比等）
- **3D 模型生成**：后端工具调用 3D 生成 API，支持文本、图片或混合模式，生成的模型保存到本地 `/storage/models/`，可替换为任意 3D 生成服务
- **虚拟人生成**：基于 ComfyUI 集成，支持图片 + 音频生成口型同步的虚拟主播视频
  - 支持两种人脸检测方式：OpenCV（快速）或 LLM（精准）
  - 自动上传图片和音频到 ComfyUI 服务器
  - 轮询任务状态，生成完成后自动下载并保存到本地
  - Mock 模式支持，便于调试
- **音频处理**：支持前端上传音频文件（MP3/WAV/M4A/AAC/OGG/FLAC/WMA），保存到 `backend/storage/audios/`，在对话中可视化展示
- **视频上传**：支持前端上传视频文件（MP4/MOV/AVI/MKV/WEBM），保存到 `backend/storage/videos/`，在对话中可视化展示
- **多模态理解**：基于 Qwen3-Omni-Flash，支持图片/音频/视频智能分析，文字+语音双模态输出
- **颜色一致性**：后端保存图片时会尝试做 sRGB 归一化（依赖 `Pillow`，已在 `requirements.txt` 固定）
- **日志系统**：统一的日志配置，支持输出到控制台和文件（`backend/logs/`），可按日期和大小自动轮转
- **Mock 模式**：支持启用 Mock 模式用于调试，返回固定的图片、视频、3D 模型和虚拟人数据，无需调用真实 API
  - 启用方式：在 `.env` 中设置 `MOCK_MODE=true`
  - 必须配置：`MOCK_IMAGE_PATH`、`MOCK_VIDEO_PATH`、`MOCK_MODEL_PATH`、`MOCK_VIRTUAL_ANCHOR_PATH`（分别指向 `/storage/` 下的实际文件路径）

### 使用示例

#### 外部 Agent 驱动 PolyStudio

可在 [OpenClaw](https://openclaw.ai) 中直接配置 `polystudio-client/SKILL.md` 作为 Skill，让外部 Agent 通过自然语言驱动 PolyStudio 完成创作。完整接入文档见 `polystudio-client/SKILL.md`。

---

#### 使用工作空间身份记忆

1. 打开设置页（右上角齿轮图标）→ **工作空间** 标签
2. 在 `USER.md` 填写你的风格偏好和常用角色
3. 在 `SOUL.md` 设定 Agent 的沟通风格和创作偏好
4. Agent 每次对话自动读取这些文件，`MEMORY.md` 由 Agent 自动更新

```
用户：帮我生成一张《原神》风格的角色立绘（结果令人满意）

Agent 工作流：
1. 调用图片生成工具，产出符合期望的角色立绘
2. 用户回复"很好，就这个风格"
3. Agent 调用 write_memory 工具，将角色参数写入 MEMORY.md「角色资产」章节
4. 下次对话时自动读取 MEMORY.md，保持角色一致性
```

#### 使用 Skill 扩展 Agent 能力

1. 将 SKILL.md 放入 `backend/skills/custom/<skill-id>/` 目录
2. 打开设置页（右上角齿轮图标）→ **Skills** 标签 → 找到你的 Skill → 启用开关
3. 在对话框中发起相关请求，Agent 自动识别并加载对应 Skill，前端以技能标识卡片展示

```
用户：帮我写一篇关于"冬季护肤"的小红书笔记

Agent 工作流（xiaohongshu-copywriter Skill 激活后）：
1. 识别到用户意图匹配 xiaohongshu-copywriter Skill
2. 调用 read_skill_file 加载 SKILL.md 获取创作规范
3. 按 Skill 定义的工作流生成：标题（含 emoji）+ 正文 + 话题标签
4. 输出符合小红书社区调性的完整帖子
```

#### 在对话中创建新 Skill

```
用户：帮我创建一个"产品经理需求文档"Skill

Agent 工作流：
1. 确认 Skill 名称和核心能力
2. 调用 init_skill 初始化 skills/custom/prd-writer/ 目录结构
3. 调用 write_skill_file 写入定制化的 SKILL.md 内容
4. 新 Skill 立即可在设置页找到并启用
```

#### 生成播客

```
用户：生成一个3分钟的AI主题播客，主持人和嘉宾对话

Agent工作流：
1. 生成对话脚本（主持人和嘉宾的完整对话内容）
2. 音色设计：
   - 为主持人生成音色样本（"亲切的女声"）
   - 为嘉宾生成音色样本（"专业的男声"）
   - 询问："请试听音色样本，是否满意？"
3. 用户确认后，批量合成所有对话（使用相同音色保持一致性）
4. 智能匹配BGM（"科技感电子音乐"）
5. 拼接所有音频片段
6. 混音处理（BGM开场3秒 → 平滑过渡到5%背景音量）
7. 输出完整播客音频
```

### 常见问题（简版）

- **外部 Agent 发送的图片/视频没有出现在画布上**：通常是因为发送时用户不在对应画布页面。现在前端有待插入队列，用户返回页面后会自动补充插入，无需重新发送。如果刷新页面后还是没有，检查 `canvas_id` 是否与 URL 中的一致。
- **如何用 curl 发送创作请求**：见 `polystudio-client/SKILL.md`，核心是 `curl -N --no-buffer -X POST .../api/chat -H "Content-Type: application/json" -d '{"message":"..."}'`，`-N` 参数禁用缓冲，SSE 才能实时输出。
- **多轮对话时怎么保持 canvas_id**：推荐在发请求前自己生成 ID（`CANVAS_ID="canvas-$(date +%s)000"`），首轮就带上，后端收到未知 ID 会自动新建项目，后续轮次复用同一 ID 即可。
- **建议不要手动混装 langchain 版本**：以 `backend/requirements.txt` 为准安装，避免出现导入错误/版本不兼容。
- **如何创建自定义 Skill**：在 `backend/skills/custom/` 下新建目录（如 `my-skill/`），在其中创建 `SKILL.md` 文件，文件需以 YAML frontmatter 开头（包含 `name` 和 `description` 字段），然后在设置页启用即可。可参考内置的 `skill-creator` Skill 了解最佳实践。也可以直接对 Agent 说"帮我创建一个 Skill"，让 Agent 通过 `init_skill` 工具自动完成初始化。
- **public/ 下的 Skill 无法在设置页关闭**：这是设计行为，`public/` 下的 Skill 始终启用（代表内置能力），如需停用请将文件移至 `custom/` 目录。
- **Skill 修改后不生效**：Skill 元数据（name/description）每次请求时都会重新扫描，无需重启；如果修改了 SKILL.md 内容，Agent 下次调用 `read_skill_file` 时会读取最新内容。
- **工作空间文件在哪**：`backend/workspace/` 目录，首次启动自动创建默认模板。可在设置页「工作空间」标签直接编辑，也可以用文本编辑器直接修改文件。
- **MEMORY.md 什么时候会被自动更新**：当用户明确表达满意（如"很好"、"就这个"）时，Agent 会调用 `write_memory` 工具将当前成果写入 MEMORY.md，不会频繁写入。
- **设置页的工具/MCP 配置存在哪**：保存在 `backend/storage/settings.json`，删除此文件会重置为默认配置。


