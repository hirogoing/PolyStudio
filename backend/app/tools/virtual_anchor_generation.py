"""
人脸检测工具 - 支持OpenCV和大模型两种方法进行人脸检测
虚拟人生成工具 - 支持基于ComfyUI的虚拟人视频生成（图片+音频生成口型同步视频）
"""
import json
import logging
import os
import base64
import requests
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def sanitize_error_message(error_msg: str) -> str:
    """
    截断错误信息，避免在日志中打印过长内容
    
    Args:
        error_msg: 原始错误信息
    
    Returns:
        截断后的错误信息
    """
    if not isinstance(error_msg, str):
        error_msg = str(error_msg)
    
    # 简单截断，超过1000字符就截断
    if len(error_msg) > 1000:
        return error_msg[:1000] + "...(已截断)"
    return error_msg

# 优先加载 backend/.env
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 注意：人脸检测工具函数使用延迟导入，避免直接运行脚本时的路径问题

# 人脸检测相关配置
FACE_DETECTION_METHOD = os.getenv("FACE_DETECTION_METHOD", "opencv").strip()  # opencv 或 llm

# 大模型人脸检测配置（使用火山引擎，从 .env 读取）
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "").strip()
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
VOLCANO_MODEL_NAME = os.getenv("VOLCANO_MODEL_NAME", "doubao-seed-1-6-251015").strip()

# 存储目录
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
AUDIOS_DIR = STORAGE_DIR / "audios"
VIDEOS_DIR = STORAGE_DIR / "videos"

# 确保存储目录存在
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# ComfyUI 配置
COMFYUI_SERVER_ADDRESS = os.getenv("COMFYUI_SERVER_ADDRESS", "").strip()
COMFYUI_WORKFLOW_PATH = os.getenv("COMFYUI_WORKFLOW_PATH", "").strip()

# Mock 模式配置
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Mock 视频路径（启用 MOCK_MODE 时必须配置）
MOCK_VIDEO_PATH = os.getenv("MOCK_VIDEO_PATH", "").strip()
if MOCK_MODE and not MOCK_VIDEO_PATH:
    logger.warning(
        "MOCK_MODE=true 时，建议配置 MOCK_VIDEO_PATH。"
        "请在 backend/.env 中设置 MOCK_VIDEO_PATH=/storage/videos/your_video.mp4"
    )


def prepare_image_path(image_url: str) -> Path:
    """
    准备图片路径，支持本地文件和URL
    
    Args:
        image_url: 本地路径（如 /storage/images/xxx.jpg）或 URL
    
    Returns:
        Path: 本地文件路径
    """
    # 检查是否是本地路径
    if image_url.startswith("/storage/") or image_url.startswith("storage/"):
        file_path = BASE_DIR / image_url.lstrip("/")
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {file_path}")
        return file_path
    
    # 如果是URL，需要先下载（这里暂时不支持，后续可以扩展）
    raise ValueError(f"暂不支持URL图片，请使用本地路径: {image_url}")


def prepare_image_base64(image_path: Path) -> str:
    """
    将图片转换为base64编码（用于大模型输入）
    
    根据火山引擎官方示例，使用 data URI 格式：data:image/png;base64,{base64_image}
    
    Args:
        image_path: 图片路径
    
    Returns:
        str: base64编码的图片字符串（data URI 格式：data:image/格式;base64,base64数据）
    """
    # 读取图片文件
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # 根据文件扩展名确定图片格式（MIME类型）
    ext = image_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".gif":
        mime_type = "image/gif"
    else:
        # 默认使用 png（与官方示例一致）
        mime_type = "image/png"
        logger.warning(f"⚠️ 未知图片格式 {ext}，使用 image/png")
    
    # 转换为base64
    base64_data = base64.b64encode(image_data).decode("utf-8")
    
    # 返回完整的 data URI 格式（与官方示例一致）
    data_uri = f"data:{mime_type};base64,{base64_data}"
    
    logger.info(f"📷 图片已转换为Base64: 格式={mime_type}, 大小={len(image_data)} bytes, base64长度={len(base64_data)}")
    
    return data_uri


def detect_face_with_llm(image_path: Path) -> Dict[str, Any]:
    """
    使用大模型进行人脸检测（火山引擎 doubao-seed-1-6-251015）
    
    Args:
        image_path: 图片路径
    
    Returns:
        人脸检测结果字典
    """
    if not VOLCANO_API_KEY:
        raise ValueError("未配置 VOLCANO_API_KEY（请在 backend/.env 设置）")
    
    # 将图片转换为base64
    base64_image = prepare_image_base64(image_path)
    
    # 构建请求（使用火山引擎官方格式：/responses 端点）
    api_url = f"{VOLCANO_BASE_URL.rstrip('/')}/responses"
    logger.info(f"🌐 API地址: {api_url}")
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建多模态输入（使用火山引擎官方格式）
    # 根据官方文档，格式应该是：
    # - type: "input_image"
    # - type: "text" 关闭思考模式
    payload = {
        "model": VOLCANO_MODEL_NAME,
         "thinking":{
            "type":"disabled"
        },
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": base64_image  # data URI 格式：data:image/png;base64,{base64_image}
                    },
                    {
                        "type": "input_text",
                        "text": "请分析这张图片中是否包含人脸。如果包含人脸，请告诉我：1. 检测到几张人脸 2. 人脸是否清晰 3. 人脸在图片中的位置（大致描述）4. 是否适合用于虚拟主播生成。请用JSON格式返回，包含字段：has_face(bool), face_count(int), is_clear(bool), position(str), suitable_for_virtual_anchor(bool), message(str)"
                    }
                ]
            }
        ]
    }
    
    logger.info(f"🤖 使用大模型检测人脸: model={VOLCANO_MODEL_NAME}")
    
    # 打印请求参数（截断长字符串）
    payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(payload_str) > 500:
        payload_str = payload_str[:500] + "...(已截断)"
    logger.info(f"📤 请求参数: {payload_str}")
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        # 如果请求失败，记录详细错误信息
        if response.status_code != 200:
            error_text = response.text[:500] if len(response.text) > 500 else response.text
            logger.error(f"❌ API请求失败: status={response.status_code}")
            logger.error(f"   错误响应: {error_text}")
            try:
                error_data = response.json()
                error_str = json.dumps(error_data, ensure_ascii=False, indent=2)
                if len(error_str) > 500:
                    error_str = error_str[:500] + "...(已截断)"
                logger.error(f"   错误详情: {error_str}")
            except:
                pass
            response.raise_for_status()
        
        data = response.json()
        # 打印响应（截断长字符串）
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        if len(data_str) > 500:
            data_str = data_str[:500] + "...(已截断)"
        logger.info(f"📥 大模型响应: {data_str}")
        
        # 解析响应（根据实际API返回格式）
        # 火山引擎响应格式：{ "output": [{ "type": "reasoning", "summary": [{ "type": "summary_text", "text": "..." }] }] }
        content = ""
        
        # 尝试1: 包含 output 字段（火山引擎标准格式）
        if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
            output_list = data["output"]
            logger.info(f"📋 解析 output 数组，共 {len(output_list)} 个元素")
            # 遍历 output 数组，查找文本内容
            for idx, output_item in enumerate(output_list):
                if isinstance(output_item, dict):
                    output_type = output_item.get("type", "unknown")
                    logger.info(f"   元素 {idx}: type={output_type}")
                    
                    # 检查是否有 summary 字段（reasoning 类型）
                    if "summary" in output_item and isinstance(output_item["summary"], list):
                        logger.info(f"   找到 summary 数组，共 {len(output_item['summary'])} 个元素")
                        for summary_idx, summary_item in enumerate(output_item["summary"]):
                            if isinstance(summary_item, dict):
                                summary_type = summary_item.get("type", "unknown")
                                if summary_type == "summary_text":
                                    text = summary_item.get("text", "")
                                    if text:
                                        logger.info(f"   提取 summary_text: {text[:100]}...")
                                        content += text + "\n"
                    # 检查是否有直接的 text 或 content 字段
                    if "text" in output_item:
                        text = output_item["text"]
                        if isinstance(text, str) and text:
                            logger.info(f"   提取 text 字段: {text[:100]}...")
                            content += text + "\n"
                    elif "content" in output_item:
                        content_value = output_item["content"]
                        # content 可能是字符串或列表
                        if isinstance(content_value, str) and content_value:
                            logger.info(f"   提取 content 字段(字符串): {content_value[:100]}...")
                            content += content_value + "\n"
                        elif isinstance(content_value, list):
                            # content 是列表，遍历提取文本
                            logger.info(f"   提取 content 字段(列表)，共 {len(content_value)} 个元素")
                            for content_item in content_value:
                                if isinstance(content_item, dict):
                                    # 检查是否有 text 字段
                                    if "text" in content_item:
                                        text = content_item["text"]
                                        if isinstance(text, str) and text:
                                            logger.info(f"   提取 content 中的 text: {text[:100]}...")
                                            content += text + "\n"
                                    # 检查是否有其他文本字段
                                    elif "content" in content_item:
                                        text = content_item["content"]
                                        if isinstance(text, str) and text:
                                            logger.info(f"   提取 content 中的 content: {text[:100]}...")
                                            content += text + "\n"
            content = content.strip()
            logger.info(f"✅ 提取的内容长度: {len(content)} 字符")
        # 尝试2: 直接是字符串
        elif isinstance(data, str):
            content = data
        # 尝试3: OpenAI 兼容格式（choices）
        elif "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
        # 尝试4: 直接包含 content 或 text 字段
        elif "content" in data:
            content = data["content"] if isinstance(data["content"], str) else ""
        elif "text" in data:
            content = data["text"] if isinstance(data["text"], str) else ""
        
        if not content:
            # 如果所有尝试都失败，记录完整响应并抛出错误
            logger.error(f"❌ 无法从响应中提取内容")
            data_str = json.dumps(data, ensure_ascii=False, indent=2)
            if len(data_str) > 500:
                data_str = data_str[:500] + "...(已截断)"
            logger.error(f"   完整响应: {data_str}")
            raise ValueError(f"无法从响应中提取内容。请检查响应格式。完整响应已记录在日志中。")
        
        if content:
            
            # 尝试从content中提取JSON（大模型可能返回文本+JSON）
            # 这里简化处理，实际可能需要更复杂的解析
            try:
                # 尝试直接解析为JSON
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # 如果都失败，返回默认结果
                    logger.warning(f"⚠️ 无法解析大模型返回的JSON，使用默认结果")
                    result = {
                        "has_face": False,
                        "face_count": 0,
                        "is_clear": False,
                        "position": "unknown",
                        "suitable_for_virtual_anchor": False,
                        "message": "无法解析大模型返回结果"
                    }
            
            # 转换为统一格式
            return {
                "has_face": result.get("has_face", False),
                "face_count": result.get("face_count", 0),
                "face_boxes": [],  # 大模型不提供精确坐标
                "confidence": 1.0 if result.get("suitable_for_virtual_anchor", False) else 0.5,
                "largest_face": {
                    "box": None,
                    "confidence": 1.0 if result.get("is_clear", False) else 0.5,
                    "position": result.get("position", "unknown")
                } if result.get("has_face", False) else None,
                "llm_result": result,
                "method": "llm"
            }
        else:
            raise ValueError("大模型响应格式异常")
            
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        logger.error(f"❌ 大模型人脸检测失败: {error_msg}")
        logger.error(f"   API地址: {api_url}")
        raise


def detect_face(image_url: str, method: Optional[str] = None) -> Dict[str, Any]:
    """
    检测人脸（支持OpenCV和大模型两种方法）
    
    Args:
        image_url: 图片URL或本地路径
        method: 检测方法（"opencv" 或 "llm"），如果为None则从环境变量读取
    
    Returns:
        人脸检测结果字典
    """
    # 确定使用的检测方法
    if method is None:
        method = FACE_DETECTION_METHOD
    
    # 准备图片路径
    image_path = prepare_image_path(image_url)
    
    logger.info(f"🔍 开始检测人脸: method={method}, image={image_path}")
    
    # 延迟导入人脸检测工具函数（避免直接运行脚本时的路径问题）
    from app.utils.face_detection import detect_face_opencv, validate_face_quality
    
    # 根据方法选择检测函数
    if method == "llm":
        face_info = detect_face_with_llm(image_path)
    else:  # 默认使用opencv
        face_info = detect_face_opencv(image_path)
        face_info["method"] = "opencv"
    
    # 验证人脸质量
    is_valid, error_msg = validate_face_quality(face_info, image_path)
    face_info["is_valid"] = is_valid
    face_info["validation_message"] = error_msg
    
    return face_info


class DetectFaceInput(BaseModel):
    """人脸检测输入参数"""
    image_url: str = Field(description="图片URL或本地路径（如 /storage/images/xxx.jpg）")
    method: Optional[str] = Field(default=None, description="检测方法：opencv（轻量级）或 llm（大模型），默认从环境变量读取")


@tool("detect_face", args_schema=DetectFaceInput)
def detect_face_tool(image_url: str, method: Optional[str] = None) -> str:
    """
    人脸检测服务：检测图片中是否包含人脸，并验证人脸质量。
    
    支持两种检测方法：
    1. opencv（轻量级）：使用OpenCV Haar Cascade，速度快，无需API调用
    2. llm（大模型）：使用多模态模型，准确度高，支持更复杂的分析
    
    Args:
        image_url: 图片URL或本地路径（如 /storage/images/xxx.jpg）
        method: 检测方法（"opencv" 或 "llm"），如果为None则从环境变量 FACE_DETECTION_METHOD 读取
    
    Returns:
        人脸检测结果的JSON字符串，包含：
        - has_face: 是否检测到人脸
        - face_count: 检测到的人脸数量
        - is_valid: 人脸质量是否合格
        - validation_message: 验证信息
        - face_boxes: 人脸边界框列表
        - largest_face: 最大人脸的信息
    """
    try:
        logger.info(f"🔍 开始人脸检测: image_url={image_url}, method={method}")
        
        # 执行人脸检测
        face_info = detect_face(image_url, method)
        
        # 构建返回结果
        result = {
            "has_face": face_info["has_face"],
            "face_count": face_info["face_count"],
            "is_valid": face_info["is_valid"],
            "validation_message": face_info["validation_message"],
            "method": face_info.get("method", "unknown"),
            "face_boxes": face_info.get("face_boxes", []),
            "largest_face": face_info.get("largest_face"),
        }
        
        # 如果使用大模型，添加额外信息
        if "llm_result" in face_info:
            result["llm_analysis"] = face_info["llm_result"]
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 人脸检测完成: has_face={face_info['has_face']}, is_valid={face_info['is_valid']}")
        return result_json
        
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        logger.error(f"❌ 人脸检测失败: {error_msg}")
        import traceback
        # 清理traceback中的base64
        tb_str = traceback.format_exc()
        tb_sanitized = sanitize_error_message(tb_str)
        logger.error(tb_sanitized)
        return json.dumps({
            "error": f"人脸检测失败: {error_msg}",
            "has_face": False,
            "is_valid": False
        }, ensure_ascii=False)


class ComfyUIClient:
    """ComfyUI API 客户端"""
    
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.base_url = f"https://{server_address}" if not server_address.startswith("http") else server_address
    
    def queue_prompt(self, prompt: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
        """提交工作流到队列"""
        if client_id is None:
            client_id = str(uuid.uuid4())
        
        data = {
            "prompt": prompt,
            "client_id": client_id
        }
        
        response = requests.post(f"{self.base_url}/prompt", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """获取任务历史"""
        response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """下载生成的图片/视频"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        response = requests.get(f"{self.base_url}/view", params=params, timeout=300)
        response.raise_for_status()
        return response.content
    
    def upload_image(self, image_path: Path, subfolder: str = "") -> str:
        """上传图片"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {}
            if subfolder:
                data['subfolder'] = subfolder
            response = requests.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=60)
        
        response.raise_for_status()
        result = response.json()
        return result['name']
    
    def upload_audio(self, audio_path: Path, subfolder: str = "") -> str:
        """上传音频"""
        with open(audio_path, 'rb') as f:
            files = {'image': f}  # ComfyUI 使用 'image' 字段名上传任意文件
            data = {}
            if subfolder:
                data['subfolder'] = subfolder
            response = requests.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=60)
        
        response.raise_for_status()
        result = response.json()
        return result['name']


def prepare_audio_path(audio_url: str) -> Path:
    """
    准备音频路径，支持本地文件
    
    Args:
        audio_url: 本地路径（如 /storage/audios/xxx.mp3）
    
    Returns:
        Path: 本地文件路径
    """
    if audio_url.startswith("/storage/") or audio_url.startswith("storage/"):
        file_path = BASE_DIR / audio_url.lstrip("/")
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {file_path}")
        return file_path
    
    raise ValueError(f"暂不支持URL音频，请使用本地路径: {audio_url}")


class GenerateVirtualAnchorInput(BaseModel):
    """虚拟主播生成输入参数"""
    image_url: str = Field(description="肖像图片URL或本地路径（如 /storage/images/xxx.jpg）")
    audio_url: str = Field(description="音频文件URL或本地路径（如 /storage/audios/xxx.mp3）")
    workflow_path: Optional[str] = Field(default=None, description="工作流JSON文件路径，默认从环境变量读取")
    prompt_text: Optional[str] = Field(default=None, description="提示词文本")
    negative_prompt: Optional[str] = Field(default=None, description="负面提示词")
    seed: Optional[int] = Field(default=None, description="随机种子")
    num_frames: int = Field(default=1450, description="视频帧数")
    fps: int = Field(default=25, description="视频帧率")
    poll_interval: int = Field(default=10, description="轮询间隔（秒）")
    wait_for_completion: bool = Field(default=True, description="是否等待任务完成")


@tool("generate_virtual_anchor", args_schema=GenerateVirtualAnchorInput)
def generate_virtual_anchor_tool(
    image_url: str,
    audio_url: str,
    workflow_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: int = 1450,
    fps: int = 25,
    poll_interval: int = 10,
    wait_for_completion: bool = True
) -> str:
    """
    虚拟主播生成服务：根据图片和音频生成口型同步的虚拟人视频（基于ComfyUI）。
    
    使用ComfyUI InfiniteTalk工作流，包含以下步骤：
    1. 上传图片和音频到ComfyUI服务器
    2. 加载并配置工作流
    3. 提交任务到队列
    4. 轮询任务状态直到完成
    5. 下载生成的视频并保存到本地
    
    Args:
        image_url: 肖像图片URL或本地路径（如 /storage/images/xxx.jpg）
        audio_url: 音频文件URL或本地路径（如 /storage/audios/xxx.mp3）
        workflow_path: 工作流JSON文件路径，默认从环境变量 COMFYUI_WORKFLOW_PATH 读取
        prompt_text: 提示词文本（可选）
        negative_prompt: 负面提示词（可选）
        seed: 随机种子（可选）
        num_frames: 视频帧数（默认1450）
        fps: 视频帧率（默认25）
        poll_interval: 轮询间隔秒数（默认10）
        wait_for_completion: 是否等待任务完成（默认True）
    
    Returns:
        生成的视频文件路径的JSON字符串或错误信息
    """
    try:
        # Mock 模式：直接返回固定的视频路径
        if MOCK_MODE:
            logger.info(f"🎭 [MOCK模式] 生成虚拟人视频: image={image_url}, audio={audio_url}")
            result = {
                "success": True,
                "video_path": MOCK_VIDEO_PATH or "/storage/videos/mock_virtual_anchor.mp4",
                "video_filename": os.path.basename(MOCK_VIDEO_PATH) if MOCK_VIDEO_PATH else "mock_virtual_anchor.mp4",
                "provider": "comfyui",
                "mock": True,
                "message": "[MOCK] 虚拟人视频已生成并保存到本地"
            }
            logger.info(f"✅ [MOCK模式] 返回结果: {result['video_path']}")
            return json.dumps(result, ensure_ascii=False)
        
        # 检查配置
        if not COMFYUI_SERVER_ADDRESS:
            return json.dumps({
                "error": "未配置 COMFYUI_SERVER_ADDRESS",
                "message": "请在 backend/.env 中设置 COMFYUI_SERVER_ADDRESS"
            }, ensure_ascii=False)
        
        # 确定工作流路径
        if workflow_path is None:
            workflow_path = COMFYUI_WORKFLOW_PATH
        
        if not workflow_path:
            return json.dumps({
                "error": "未指定工作流路径",
                "message": "请提供 workflow_path 参数或在 backend/.env 中设置 COMFYUI_WORKFLOW_PATH"
            }, ensure_ascii=False)
        
        # 处理路径：支持 /storage/ 开头的相对路径和绝对路径
        if workflow_path.startswith("/storage/") or workflow_path.startswith("storage/"):
            workflow_path_obj = BASE_DIR / workflow_path.lstrip("/")
        else:
            workflow_path_obj = Path(workflow_path)
            # 如果是相对路径，则相对于 BASE_DIR
            if not workflow_path_obj.is_absolute():
                workflow_path_obj = BASE_DIR / workflow_path
        
        if not workflow_path_obj.exists():
            return json.dumps({
                "error": f"工作流文件不存在: {workflow_path}",
                "message": f"请检查工作流文件路径（已解析为: {workflow_path_obj}）"
            }, ensure_ascii=False)
        
        logger.info(f"🎬 开始生成虚拟人视频: image={image_url}, audio={audio_url}")
        
        # 步骤1：准备图片和音频路径
        image_path = prepare_image_path(image_url)
        audio_path = prepare_audio_path(audio_url)
        
        logger.info(f"✅ 图片路径: {image_path}")
        logger.info(f"✅ 音频路径: {audio_path}")
        
        # 步骤2：创建ComfyUI客户端
        client = ComfyUIClient(COMFYUI_SERVER_ADDRESS)
        logger.info(f"🌐 ComfyUI服务器: {client.base_url}")
        
        # 步骤3：上传图片和音频
        logger.info(f"📤 上传图片...")
        uploaded_image = client.upload_image(image_path)
        logger.info(f"✅ 图片已上传: {uploaded_image}")
        
        logger.info(f"📤 上传音频...")
        uploaded_audio = client.upload_audio(audio_path)
        logger.info(f"✅ 音频已上传: {uploaded_audio}")
        
        # 步骤4：加载工作流
        logger.info(f"📋 加载工作流: {workflow_path}")
        with open(workflow_path_obj, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # 步骤5：配置工作流参数
        # 注意：以下节点ID基于参考代码中的InfiniteTalk工作流
        # 实际使用时可能需要根据具体工作流JSON文件调整节点ID
        # 可以通过查看工作流JSON文件来确定正确的节点ID
        try:
            workflow["133"]["inputs"]["image"] = uploaded_image
            workflow["125"]["inputs"]["audio"] = uploaded_audio
            
            if prompt_text:
                workflow["135"]["inputs"]["positive_prompt"] = prompt_text
            
            if negative_prompt:
                workflow["135"]["inputs"]["negative_prompt"] = negative_prompt
            
            if seed is not None:
                workflow["128"]["inputs"]["seed"] = seed
            
            workflow["194"]["inputs"]["num_frames"] = num_frames
            workflow["194"]["inputs"]["fps"] = fps
            workflow["131"]["inputs"]["frame_rate"] = fps
        except KeyError as e:
            logger.warning(f"⚠️ 工作流节点配置失败: {e}")
            logger.warning(f"   请检查工作流JSON文件中的节点ID是否正确")
            logger.warning(f"   参考节点ID: 133(image), 125(audio), 135(prompt), 128(seed), 194(frames), 131(fps)")
            # 继续执行，让用户自己检查工作流配置
        
        logger.info(f"✅ 工作流已配置: num_frames={num_frames}, fps={fps}")
        
        # 步骤6：提交任务
        logger.info(f"📤 提交任务到队列...")
        result = client.queue_prompt(workflow)
        prompt_id = result.get("prompt_id")
        
        if not prompt_id:
            return json.dumps({
                "error": "任务提交失败",
                "message": "无法获取 prompt_id",
                "response": result
            }, ensure_ascii=False)
        
        logger.info(f"✅ 任务已提交: prompt_id={prompt_id}")
        
        # 步骤7：等待任务完成（如果需要）
        if wait_for_completion:
            logger.info(f"⏳ 等待任务完成（轮询间隔: {poll_interval}秒）...")
            max_wait_time = 3600  # 最大等待1小时
            start_time = time.time()
            
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    return json.dumps({
                        "error": "任务超时",
                        "message": f"任务执行超过 {max_wait_time} 秒",
                        "prompt_id": prompt_id
                    }, ensure_ascii=False)
                
                try:
                    history = client.get_history(prompt_id)
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if outputs:
                            logger.info(f"✅ 任务完成: prompt_id={prompt_id}")
                            break
                except Exception as e:
                    logger.warning(f"⚠️ 获取历史失败: {e}")
                
                time.sleep(poll_interval)
            
            # 步骤8：下载生成的视频
            logger.info(f"📥 下载生成的视频...")
            video_filename = None
            video_subfolder = ""
            
            for node_id, node_output in outputs.items():
                if 'gifs' in node_output:
                    for video_info in node_output['gifs']:
                        video_filename = video_info['filename']
                        video_subfolder = video_info.get('subfolder', '')
                        break
                elif 'images' in node_output:
                    # 有些工作流可能输出为images
                    for img_info in node_output['images']:
                        video_filename = img_info['filename']
                        video_subfolder = img_info.get('subfolder', '')
                        break
            
            if not video_filename:
                return json.dumps({
                    "error": "无法找到生成的视频",
                    "message": "工作流输出中未找到视频文件",
                    "outputs": outputs
                }, ensure_ascii=False)
            
            logger.info(f"📥 下载视频: {video_filename} (subfolder: {video_subfolder})")
            video_data = client.get_image(video_filename, subfolder=video_subfolder, folder_type='output')
            
            # 步骤9：保存视频到本地
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            video_filename_local = f"virtual_anchor_{timestamp}_{unique_id}.mp4"
            video_path = VIDEOS_DIR / video_filename_local
            
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            logger.info(f"✅ 视频已保存: {video_path}")
            
            # 构建返回结果
            result = {
                "success": True,
                "prompt_id": prompt_id,
                "video_path": f"/storage/videos/{video_filename_local}",
                "video_url": f"/storage/videos/{video_filename_local}",  # 添加 video_url 字段，供前端使用
                "video_filename": video_filename_local,
                "provider": "comfyui"
            }
            
            logger.info(f"🎉 虚拟人视频生成完成")
            return json.dumps(result, ensure_ascii=False)
        else:
            # 不等待完成，直接返回prompt_id
            return json.dumps({
                "success": True,
                "prompt_id": prompt_id,
                "message": "任务已提交，请稍后查询结果",
                "provider": "comfyui"
            }, ensure_ascii=False)
        
    except FileNotFoundError as e:
        error_msg = sanitize_error_message(str(e))
        logger.error(f"❌ 文件不存在: {error_msg}")
        return json.dumps({
            "error": f"文件不存在: {error_msg}",
            "success": False
        }, ensure_ascii=False)
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        logger.error(f"❌ 虚拟人视频生成失败: {error_msg}")
        import traceback
        tb_str = traceback.format_exc()
        tb_sanitized = sanitize_error_message(tb_str)
        logger.error(tb_sanitized)
        return json.dumps({
            "error": f"虚拟人视频生成失败: {error_msg}",
            "success": False
        }, ensure_ascii=False)


if __name__ == "__main__":
    """测试工具"""
    import sys
    from pathlib import Path
    
    # 添加 backend 目录到 Python 路径，以便能够导入 app 模块
    # 这必须在调用任何使用延迟导入的函数之前执行
    backend_dir = Path(__file__).parent.parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试人脸检测（由于使用了延迟导入，现在可以正常工作了）
    # result = detect_face_tool.invoke({
    #     "image_url": "/storage/images/volcano_20260121_172941_2e425df2_特写主角面部眼神中闪过一丝迷茫随即恢复坚定展现内心的抉.jpg",
    #     "method": "opencv"
    # })
    # print("检测结果:", result)

    # # 测试人脸检测（由于使用了延迟导入，现在可以正常工作了）
    # result = detect_face_tool.invoke({
    #     "image_url": "/storage/images/volcano_20260121_172941_2e425df2_特写主角面部眼神中闪过一丝迷茫随即恢复坚定展现内心的抉.jpg",
    #     "method": "llm"
    # })
    # print("检测结果llm:", result)

    result = generate_virtual_anchor_tool.invoke({
        "image_url": "/storage/images/虚拟人测试图.png",
        "audio_url": "/storage/audios/poly_studio_intro.wav",
        "workflow_path": "/storage/workflow/infinitetalk_workflow.json"
    })
    print("虚拟人视频生成结果:", result)
