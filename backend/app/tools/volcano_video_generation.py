"""
火山引擎视频生成工具 - 使用 Seedance API 生成视频
"""
import json
import logging
import os
import requests
import uuid
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 优先加载 backend/.env（避免直接运行工具脚本时环境未加载）
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 从环境变量获取配置
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "").strip()
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
# 默认使用 Seedance 1.5 Pro（支持文生视频、图生视频-首帧、首尾帧）
VOLCANO_VIDEO_MODEL = os.getenv("VOLCANO_VIDEO_MODEL", "doubao-seedance-1-5-pro").strip()

# 视频存储目录
STORAGE_DIR = BASE_DIR / "storage"
VIDEOS_DIR = STORAGE_DIR / "videos"

# 确保视频存储目录存在
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Mock 模式配置（需要在 VIDEOS_DIR 定义之后）
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Mock 视频路径（启用 MOCK_MODE 时必须配置）
MOCK_VIDEO_PATH = os.getenv("MOCK_VIDEO_PATH", "").strip()
if MOCK_MODE and not MOCK_VIDEO_PATH:
    raise RuntimeError(
        "MOCK_MODE=true 时，必须配置 MOCK_VIDEO_PATH。"
        "请在 backend/.env 中设置 MOCK_VIDEO_PATH=/storage/videos/your_video.mp4"
    )


def download_and_save_video(video_url: str, prompt: str = "") -> str:
    """
    下载视频并保存到本地
    
    Args:
        video_url: 视频URL
        prompt: 提示词（用于生成文件名）
    
    Returns:
        本地文件路径（相对路径）
    """
    try:
        logger.info(f"📥 开始下载视频: {video_url}")
        
        # 下载视频
        response = requests.get(video_url, timeout=300, stream=True)
        response.raise_for_status()
        
        # 从URL获取文件扩展名，如果没有则默认为mp4
        parsed_url = urlparse(video_url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1] or ".mp4"
        if not ext.startswith("."):
            ext = ".mp4"
        
        # 生成唯一文件名：时间戳_随机ID_提示词前20字符
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        # 清理提示词，只保留字母数字和空格，用于文件名
        safe_prompt = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in prompt[:30])
        safe_prompt = safe_prompt.replace(" ", "_")
        filename = f"volcano_{timestamp}_{unique_id}_{safe_prompt}{ext}" if safe_prompt else f"volcano_{timestamp}_{unique_id}{ext}"
        
        file_path = VIDEOS_DIR / filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # 返回HTTP访问路径（以/storage/开头，前端可以直接使用）
        http_path = f"/storage/videos/{filename}"
        logger.info(f"✅ 视频已保存到本地: {file_path}")
        logger.info(f"   可通过HTTP访问: {http_path}")
        return http_path
        
    except Exception as e:
        logger.error(f"❌ 下载视频失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果下载失败，返回原始URL
        return video_url


def prepare_image_input(image_url: str) -> str:
    """
    准备图片输入，处理本地文件（转Base64）或公网URL
    
    Args:
        image_url: 本地路径（如 /storage/images/xxx.jpg）或 localhost URL（如 http://localhost:8000/storage/images/xxx.jpg）或公网URL
    
    Returns:
        Base64编码字符串（本地文件）或URL字符串（公网URL）
    
    Raises:
        FileNotFoundError: 本地文件不存在
    """
    # 检查是否是本地路径
    if image_url.startswith("/storage/"):
        # 本地文件，读取并转换为Base64
        file_path = BASE_DIR / image_url.lstrip("/")
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {file_path}")
        
        logger.info(f"📁 读取本地文件: {file_path}")
        
        # 读取文件
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        # 获取文件扩展名，确定图片格式
        ext = file_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            image_format = "jpeg"
        elif ext == ".png":
            image_format = "png"
        elif ext == ".webp":
            image_format = "webp"
        elif ext == ".bmp":
            image_format = "bmp"
        elif ext in [".tiff", ".tif"]:
            image_format = "tiff"
        elif ext == ".gif":
            image_format = "gif"
        else:
            # 默认使用jpeg
            image_format = "jpeg"
            logger.warning(f"未知图片格式 {ext}，使用 jpeg")
        
        # 转换为Base64
        base64_data = base64.b64encode(image_data).decode("utf-8")
        base64_string = f"data:image/{image_format};base64,{base64_data}"
        
        logger.info(f"✅ 已转换为Base64格式: {image_format}, 大小={len(image_data)} bytes")
        return base64_string
    
    # 检查是否是localhost URL（如 http://localhost:8000/storage/images/xxx.jpg）
    parsed = urlparse(image_url)
    if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"] or (parsed.hostname and "localhost" in parsed.hostname):
        # localhost URL，读取本地文件
        if parsed.path.startswith("/storage/"):
            file_path = BASE_DIR / parsed.path.lstrip("/")
            if not file_path.exists():
                raise FileNotFoundError(f"本地文件不存在: {file_path}")
            
            logger.info(f"📁 从localhost URL读取本地文件: {file_path}")
            
            # 读取文件并转换为Base64
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            ext = file_path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                image_format = "jpeg"
            elif ext == ".png":
                image_format = "png"
            elif ext == ".webp":
                image_format = "webp"
            elif ext == ".bmp":
                image_format = "bmp"
            elif ext in [".tiff", ".tif"]:
                image_format = "tiff"
            elif ext == ".gif":
                image_format = "gif"
            else:
                image_format = "jpeg"
            
            base64_data = base64.b64encode(image_data).decode("utf-8")
            base64_string = f"data:image/{image_format};base64,{base64_data}"
            
            logger.info(f"✅ 已转换为Base64格式: {image_format}, 大小={len(image_data)} bytes")
            return base64_string
    
    # 公网URL，直接返回
    logger.info(f"🌐 使用公网URL: {image_url}")
    return image_url


def extract_base64_from_data_url(data_url: str) -> str:
    """
    从 data:image/xxx;base64, 格式中提取纯 base64 字符串
    
    Args:
        data_url: data:image/xxx;base64,base64_string 格式的字符串
    
    Returns:
        纯 base64 字符串（不带前缀）
    """
    if data_url.startswith("data:image/"):
        parts = data_url.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return data_url


def truncate_base64_for_logging(data: dict, max_length: int = 100) -> dict:
    """
    截断字典中的base64字符串用于日志打印
    
    Args:
        data: 可能包含base64字符串的字典
        max_length: base64字符串的最大显示长度
    
    Returns:
        截断后的字典副本
    """
    import copy
    
    def truncate_value(value):
        if isinstance(value, str):
            # 检查是否是base64字符串（data:image/...;base64, 或纯base64）
            if value.startswith("data:image/") and "base64," in value:
                # 截断data URL格式的base64
                parts = value.split(",", 1)
                if len(parts) == 2 and len(parts[1]) > max_length:
                    return f"{parts[0]},{parts[1][:max_length]}... (truncated, total length: {len(parts[1])})"
            elif len(value) > max_length and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in value[:100]):
                # 可能是纯base64字符串
                return f"{value[:max_length]}... (truncated, total length: {len(value)})"
        elif isinstance(value, dict):
            return {k: truncate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [truncate_value(item) for item in value]
        return value
    
    return truncate_value(copy.deepcopy(data))


def submit_video_generation_task(
    prompt: str, 
    duration: Optional[int] = None, 
    ratio: str = "16:9",
    image_url: Optional[str] = None,
    start_image_url: Optional[str] = None,
    end_image_url: Optional[str] = None,
    mode: str = "text"  # "text", "image", "start_end"
) -> str:
    """
    提交视频生成任务到火山引擎API
    
    Args:
        prompt: 视频生成的提示词
        duration: 视频时长（秒），可选，默认 5 秒
        ratio: 视频宽高比，默认 "16:9"
        image_url: 图片URL或本地路径（图生视频-首帧模式，如 /storage/images/xxx.jpg）
        start_image_url: 首帧图片URL或本地路径（首尾帧模式）
        end_image_url: 尾帧图片URL或本地路径（首尾帧模式）
        mode: 生成模式，"text"（文生视频）、"image"（图生视频-首帧）、"start_end"（首尾帧）
    
    Returns:
        TaskId（任务ID）
    """
    if not VOLCANO_API_KEY:
        raise ValueError("未配置 VOLCANO_API_KEY（请在 backend/.env 设置）")
    
    # 火山引擎视频生成端点
    url = f"{VOLCANO_BASE_URL.rstrip('/')}/contents/generations/tasks"
    
    # 所有模式都使用 Seedance 1.5 Pro
    model = VOLCANO_VIDEO_MODEL
    
    # 构建 content 数组
    content = []
    
    # 添加文本提示
    if prompt:
        content.append({
            "type": "text",
            "text": prompt
        })
    
    # 根据模式添加图片
    if mode == "image":
        # 图生视频-首帧模式
        if not image_url:
            raise ValueError("图生视频模式需要提供 image_url")
        
        # 准备图片输入（本地路径转base64，公网URL直接使用）
        image_input = prepare_image_input(image_url)
        
        # image_url 需要是对象格式，包含 url 字段
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_input
            }
        })
    
    elif mode == "start_end":
        # 首尾帧模式
        if not start_image_url or not end_image_url:
            raise ValueError("首尾帧模式需要提供 start_image_url 和 end_image_url")
        
        # 准备首帧图片
        start_image_input = prepare_image_input(start_image_url)
        content.append({
            "type": "image_url",
            "role": "first_frame",
            "image_url": {
                "url": start_image_input
            }
        })
        
        # 准备尾帧图片
        end_image_input = prepare_image_input(end_image_url)
        content.append({
            "type": "image_url",
            "role": "last_frame",
            "image_url": {
                "url": end_image_input
            }
        })
    
    # 构建请求体
    payload = {
        "model": model,
        "content": content,
        "ratio": ratio,
        "watermark": False
    }
    
    # 如果指定了时长，添加到请求中（4-12秒，默认5秒）
    if duration:
        payload["duration"] = duration
    
    # Seedance 1.5 Pro 支持的参数
    payload["resolution"] = "720p"  # 1.5 Pro 最高 720p
    payload["audio"] = True  # 支持音频生成
    
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"🚀 提交视频生成任务: {url}")
    logger.info(f"   提示词: {prompt}")
    # 截断base64字符串用于日志打印
    payload_for_log = truncate_base64_for_logging(payload)
    logger.info(f"   请求参数: {json.dumps(payload_for_log, ensure_ascii=False, indent=2)}")
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    
    if response.status_code != 200:
        error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    data = response.json()
    # 截断base64字符串用于日志打印
    data_for_log = truncate_base64_for_logging(data)
    logger.info(f"📥 API响应: {json.dumps(data_for_log, ensure_ascii=False)}")
    
    # 提取任务ID（根据API文档，在 id 字段）
    task_id = data.get("id")
    
    if not task_id:
        raise Exception(f"API响应中未找到任务ID: {json.dumps(data, ensure_ascii=False)}")
    
    logger.info(f"✅ 任务已提交，TaskId: {task_id}")
    return str(task_id)


def query_video_generation_task(task_id: str, max_wait_time: int = 600) -> dict:
    """
    查询视频生成任务状态，轮询直到完成
    
    Args:
        task_id: 任务ID
        max_wait_time: 最大等待时间（秒），默认10分钟
    
    Returns:
        任务结果字典，包含视频URL等信息
    """
    if not VOLCANO_API_KEY:
        raise ValueError("未配置 VOLCANO_API_KEY")
    
    # 查询任务状态的端点（正确的端点）
    url = f"{VOLCANO_BASE_URL.rstrip('/')}/contents/generations/tasks/{task_id}"
    
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    poll_interval = 5  # 每5秒查询一次
    
    logger.info(f"🔄 开始轮询任务状态: TaskId={task_id}")
    
    while True:
        # 检查是否超时
        if time.time() - start_time > max_wait_time:
            raise TimeoutError(f"任务超时: 超过{max_wait_time}秒未完成")
        
        # 查询任务状态
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"查询任务失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        data = response.json()
        
        # 根据API文档，状态在 status 字段
        status = data.get("status")
        
        logger.info(f"📊 任务状态: {status}")
        
        # 检查错误
        if status and status.lower() in ["failed", "error"]:
            error_msg = data.get("error") or data.get("error_message") or "任务失败"
            raise Exception(f"任务失败: {error_msg}")
        
        # 任务完成状态（根据API文档，状态为 "succeeded"）
        if status and status.lower() in ["succeeded", "success", "completed", "done", "finished"]:
            # 提取视频URL（根据API文档，在 content.video_url 字段）
            content = data.get("content")
            if content and isinstance(content, dict):
                video_url = content.get("video_url")
            else:
                # 兼容其他可能的格式
                video_url = (
                    data.get("video_url") or
                    data.get("content", {}).get("video_url") if isinstance(data.get("content"), dict) else None
                )
            
            if video_url:
                # 截断base64字符串用于日志打印
                data_for_log = truncate_base64_for_logging(data)
                logger.info(f"✅ 任务完成: {json.dumps(data_for_log, ensure_ascii=False)}")
                return {
                    "status": status,
                    "video_url": video_url,
                    "task_id": task_id,
                    "raw_response": data
                }
            else:
                # 状态是完成但没有视频URL，继续等待
                logger.warning(f"⚠️ 任务状态为 {status} 但 video_url 为空，继续等待...")
                time.sleep(poll_interval)
                continue
        
        # 任务进行中状态（pending, processing, running 等）
        if status and status.lower() in ["pending", "processing", "running", "queued", "in_progress", "in_queue"]:
            logger.info(f"⏳ 任务进行中（状态: {status}），{poll_interval}秒后继续查询...")
            time.sleep(poll_interval)
            continue
        
        # 未知状态，默认认为进行中
        logger.warning(f"⚠️ 未知任务状态: {status}，按进行中处理，{poll_interval}秒后继续查询...")
        time.sleep(poll_interval)


class GenerateVolcanoVideoInput(BaseModel):
    """火山引擎视频生成输入参数"""
    prompt: str = Field(description="视频生成的提示词，详细描述想要生成的视频内容，支持中英文")
    duration: Optional[int] = Field(default=None, description="视频时长（秒），可选，4-12秒，默认5秒")
    ratio: str = Field(default="16:9", description="视频宽高比，支持 16:9, 9:16, 1:1, 4:3, 3:4, 21:9 等，默认 16:9")
    mode: str = Field(default="text", description="生成模式：'text'（文生视频）、'image'（图生视频-首帧）、'start_end'（首尾帧）")
    image_url: Optional[str] = Field(default=None, description="图片URL或本地路径（图生视频-首帧模式使用，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）")
    start_image_url: Optional[str] = Field(default=None, description="首帧图片URL或本地路径（首尾帧模式使用，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）")
    end_image_url: Optional[str] = Field(default=None, description="尾帧图片URL或本地路径（首尾帧模式使用，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）")


@tool("generate_volcano_video", args_schema=GenerateVolcanoVideoInput)
def generate_volcano_video_tool(
    prompt: str, 
    duration: Optional[int] = None, 
    ratio: str = "16:9",
    mode: str = "text",
    image_url: Optional[str] = None,
    start_image_url: Optional[str] = None,
    end_image_url: Optional[str] = None
) -> str:
    """
    火山引擎 AI 视频生成服务，使用 Seedance API 生成视频。
    支持三种模式：
    1. 文生视频（mode='text'）：基于文本描述生成视频，使用 Seedance 1.5 Pro
    2. 图生视频-首帧（mode='image'）：基于首帧图片生成视频，使用 Seedance 1.5 Pro
    3. 首尾帧（mode='start_end'）：基于首帧和尾帧图片生成过渡视频，使用 Seedance 1.0 Lite I2V
    
    Args:
        prompt: 视频生成的提示词（支持中英文）
        duration: 视频时长（秒），可选，4-12秒，默认5秒
        ratio: 视频宽高比，支持 16:9, 9:16, 1:1, 4:3, 3:4, 21:9 等，默认 16:9
        mode: 生成模式，"text"（文生视频）、"image"（图生视频-首帧）、"start_end"（首尾帧）
        image_url: 图片URL或本地路径（图生视频-首帧模式，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）
        start_image_url: 首帧图片URL或本地路径（首尾帧模式，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）
        end_image_url: 尾帧图片URL或本地路径（首尾帧模式，如 /storage/images/xxx.jpg，本地路径会自动转换为base64）
    
    Returns:
        生成的视频URL的JSON字符串或错误信息
    """
    # Mock 模式：直接返回固定的视频路径
    if MOCK_MODE:
        logger.info(f"🎭 [MOCK模式] 生成视频: prompt={prompt}, mode={mode}, duration={duration}, ratio={ratio}")
        result = {
            'video_url': MOCK_VIDEO_PATH,
            'original_url': MOCK_VIDEO_PATH,
            'local_path': MOCK_VIDEO_PATH,
            'prompt': prompt,
            'mode': mode,
            'duration': duration,
            'ratio': ratio,
            'provider': 'volcano',
            'mock': True,
            'message': '[MOCK] 视频已生成并保存到本地'
        }
        return json.dumps(result, ensure_ascii=False)
    
    try:
        if not VOLCANO_API_KEY:
            return "Error generating video: 未配置 VOLCANO_API_KEY（请在 backend/.env 设置，可参考 env.example）"
        
        logger.info(f"🎬 开始使用火山引擎生成视频: mode={mode}, prompt={prompt}, duration={duration}, ratio={ratio}")

        # 1. 提交视频生成任务
        task_id = submit_video_generation_task(
            prompt=prompt,
            duration=duration,
            ratio=ratio,
            image_url=image_url,
            start_image_url=start_image_url,
            end_image_url=end_image_url,
            mode=mode
        )
        
        # 2. 轮询查询任务状态
        task_result = query_video_generation_task(task_id, max_wait_time=600)
        
        # 3. 获取视频URL
        video_url = task_result.get("video_url")
        if not video_url:
            error_msg = f"任务完成但未找到视频URL。响应: {json.dumps(task_result, ensure_ascii=False)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({
                "error": error_msg
            }, ensure_ascii=False)
        
        # 4. 下载并保存视频
        local_path = download_and_save_video(video_url, prompt)
        
        # 返回结果
        result = {
            'video_url': local_path,
            'original_url': video_url,
            'local_path': local_path,
            'prompt': prompt,
            'mode': mode,
            'duration': duration,
            'ratio': ratio,
            'task_id': task_id,
            'provider': 'volcano',
            'message': '视频已生成并保存到本地'
        }
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 火山引擎视频生成成功: 已保存到本地 {local_path}")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ 火山引擎视频生成失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({
            "error": f"生成视频时出错: {str(e)}"
        }, ensure_ascii=False)


if __name__ == "__main__":
    """测试工具"""
    from dotenv import load_dotenv
    from pathlib import Path
    
    # 加载 .env 文件（从 backend 目录）
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ 已加载环境变量: {env_path}")
    else:
        print(f"⚠️  未找到 .env 文件: {env_path}")
        print("   请确保已配置环境变量或创建 .env 文件")
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试1: 文生视频（text模式）
    # print("\n" + "="*60)
    # print("测试1: 文生视频（text模式）")
    # print("="*60)
    # result1 = generate_volcano_video_tool.invoke({
    #     "prompt": "一只可爱的小猫在花园里玩耍，阳光明媚，画面温馨",
    #     "duration": 5,
    #     "ratio": "16:9",
    #     "mode": "text"
    # })
    # print("生成结果:", result1)
    
    # 测试2: 图生视频（image模式）
    # print("\n" + "="*60)
    # print("测试2: 图生视频（image模式）")
    # print("="*60)
    # # 使用storage目录下的示例图片
    # test_image_path = "/storage/images/volcano_20260117_222533_f4152f5b_冬雪马站在北极苔原上背景是绚烂的绿色极光和星空地面是结冰.jpg"
    # result2 = generate_volcano_video_tool.invoke({
    #     "prompt": "在阳光下熠熠生辉，穿梭其中",
    #     "duration": 5,
    #     "ratio": "16:9",
    #     "mode": "image",
    #     "image_url": test_image_path
    # })
    # print("生成结果:", result2)
    
    # 测试3: 首尾帧视频（start_end模式）
    print("\n" + "="*60)
    print("测试3: 首尾帧视频（start_end模式）")
    print("="*60)
    # 使用两张不同的图片作为首尾帧
    start_image = "/storage/images/volcano_20260117_222510_401d1401_冬雪马站在夜晚的都市街道上背景是霓虹闪烁的高楼大厦和飘雪的.jpg"
    end_image = "/storage/images/volcano_20260117_222533_f4152f5b_冬雪马站在北极苔原上背景是绚烂的绿色极光和星空地面是结冰.jpg"
    result3 = generate_volcano_video_tool.invoke({
        "prompt": "从城市走向北极",
        "duration": 5,
        "ratio": "16:9",
        "mode": "start_end",
        "start_image_url": start_image,
        "end_image_url": end_image
    })
    print("生成结果:", result3)
    
    # print("\n" + "="*60)
    # print("所有测试完成！")
    # print("="*60)
