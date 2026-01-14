"""
3D模型生成工具 - 腾讯云混元生3D API实现：接收图片，返回3D模型（OBJ/GLB格式）
"""
import json
import logging
import os
import uuid
import time
import base64
import zipfile
import shutil
import requests
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 优先加载 backend/.env
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 腾讯云混元生3D API配置
TENCENT_AI3D_API_KEY = os.getenv("TENCENT_AI3D_API_KEY", "").strip()
TENCENT_AI3D_BASE_URL = os.getenv("TENCENT_AI3D_BASE_URL", "https://api.ai3d.cloud.tencent.com").strip()
TENCENT_AI3D_SUBMIT_URL = f"{TENCENT_AI3D_BASE_URL}/v1/ai3d/submit"
TENCENT_AI3D_QUERY_URL = f"{TENCENT_AI3D_BASE_URL}/v1/ai3d/query"

# 3D模型存储目录
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
IMAGES_DIR = STORAGE_DIR / "images"

# 确保模型存储目录存在
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Mock 模式配置（需要在 MODELS_DIR 定义之后）
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Mock 3D模型路径（启用 MOCK_MODE 时必须配置）
MOCK_MODEL_PATH = os.getenv("MOCK_MODEL_PATH", "").strip()
if MOCK_MODE and not MOCK_MODEL_PATH:
    raise RuntimeError(
        "MOCK_MODE=true 时，必须配置 MOCK_MODEL_PATH。"
        "请在 backend/.env 中设置 MOCK_MODEL_PATH=/storage/models/your_model_dir"
    )

# 尝试导入PIL用于生成预览图
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("⚠️ PIL未安装，无法生成3D模型预览图")


def prepare_image_input(image_url: str) -> Tuple[str, str, bool]:
    """
    准备图片输入，支持本地文件和URL
    如果是本地文件，转换为base64；如果是URL，直接返回
    
    Args:
        image_url: 本地路径（如 /storage/images/xxx.jpg）或 URL
    
    Returns:
        (base64_string, url_string, is_base64) 元组
        - base64_string: 纯base64字符串（不带data:前缀），如果是URL则为空字符串
        - url_string: URL字符串，如果是本地文件则为空字符串
        - is_base64: 是否为base64格式
    """
    # 检查是否是本地路径
    if image_url.startswith("/storage/") or image_url.startswith("storage/"):
        # 本地文件，读取并转换为Base64
        file_path = BASE_DIR / image_url.lstrip("/")
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {file_path}")
        
        logger.info(f"📁 读取本地文件: {file_path}")
        
        # 读取文件
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        # 转换为Base64（纯base64字符串，不带data:前缀）
        base64_data = base64.b64encode(image_data).decode("utf-8")
        
        logger.info(f"✅ 已转换为Base64格式, 大小={len(image_data)} bytes")
        return (base64_data, "", True)
    
    # 如果是URL，直接返回
    logger.info(f"📥 使用图片URL: {image_url}")
    return ("", image_url, False)


def submit_3d_generation_task(image_url: str, prompt: Optional[str] = None) -> str:
    """
    提交3D模型生成任务到腾讯云混元生3D API（图生3D模式，可带提示词）
    
    Args:
        image_url: 图片URL或本地路径
        prompt: 可选的文本提示词
    
    Returns:
        JobId（任务ID）
    """
    if not TENCENT_AI3D_API_KEY:
        raise ValueError("未配置 TENCENT_AI3D_API_KEY（请在 backend/.env 设置）")
    
    # 准备图片输入
    base64_str, url_str, is_base64 = prepare_image_input(image_url)
    
    # 构建请求体，符合腾讯云API格式
    payload = {
        "Prompt": prompt if prompt else None,
        "ImageBase64": base64_str if is_base64 else None,
        "ImageUrl": url_str if not is_base64 else None,
        "MultiViewImages": None,
        "EnablePBR": None,
        "FaceCount": None,
        "GenerateType": None,
        "PolygonType": None
    }
    
    headers = {
        "Authorization": TENCENT_AI3D_API_KEY,
        "Content-Type": "application/json"
    }
    
    logger.info(f"🚀 提交3D生成任务到腾讯云（图生3D）: {TENCENT_AI3D_SUBMIT_URL}")
    logger.info(f"   图片: {image_url[:50]}...")
    if prompt:
        logger.info(f"   提示词: {prompt}")
    
    logger.info(f"📥 3D API请求参数: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(TENCENT_AI3D_SUBMIT_URL, json=payload, headers=headers, timeout=60)
    
    if response.status_code != 200:
        error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    data = response.json()
    logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
    
    # 处理Response包装，提取JobId
    response_data = data.get("Response") or data
    job_id = (
        response_data.get("JobId") or 
        response_data.get("job_id") or 
        response_data.get("jobId") or
        data.get("JobId") or 
        data.get("job_id") or 
        data.get("jobId")
    )
    if not job_id:
        raise Exception(f"API响应中未找到JobId: {json.dumps(data, ensure_ascii=False)}")
    
    logger.info(f"✅ 任务已提交，JobId: {job_id}")
    return str(job_id)


def submit_3d_generation_task_with_prompt(prompt: str) -> str:
    """
    提交3D模型生成任务到腾讯云混元生3D API（文生3D模式）
    
    Args:
        prompt: 文本提示词，描述要生成的3D模型
    
    Returns:
        JobId（任务ID）
    """
    if not TENCENT_AI3D_API_KEY:
        raise ValueError("未配置 TENCENT_AI3D_API_KEY（请在 backend/.env 设置）")
    
    # 构建请求体，符合腾讯云API格式（仅使用Prompt）
    payload = {
        "Prompt": prompt,
        "ImageBase64": None,
        "ImageUrl": None,
        "MultiViewImages": None,
        "EnablePBR": None,
        "FaceCount": None,
        "GenerateType": None,
        "PolygonType": None
    }
    
    headers = {
        "Authorization": TENCENT_AI3D_API_KEY,
        "Content-Type": "application/json"
    }
    
    logger.info(f"🚀 提交3D生成任务到腾讯云（文生3D）: {TENCENT_AI3D_SUBMIT_URL}")
    logger.info(f"   提示词: {prompt}")
    
    response = requests.post(TENCENT_AI3D_SUBMIT_URL, json=payload, headers=headers, timeout=60)
    
    if response.status_code != 200:
        error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    data = response.json()
    logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
    
    # 处理Response包装，提取JobId
    response_data = data.get("Response") or data
    job_id = (
        response_data.get("JobId") or 
        response_data.get("job_id") or 
        response_data.get("jobId") or
        data.get("JobId") or 
        data.get("job_id") or 
        data.get("jobId")
    )
    if not job_id:
        raise Exception(f"API响应中未找到JobId: {json.dumps(data, ensure_ascii=False)}")
    
    logger.info(f"✅ 任务已提交，JobId: {job_id}")
    return str(job_id)


def submit_3d_generation_task_with_prompt(prompt: str) -> str:
    """
    提交3D模型生成任务到腾讯云混元生3D API（文生3D模式）
    
    Args:
        prompt: 文本提示词，描述要生成的3D模型
    
    Returns:
        JobId（任务ID）
    """
    if not TENCENT_AI3D_API_KEY:
        raise ValueError("未配置 TENCENT_AI3D_API_KEY（请在 backend/.env 设置）")
    
    # 构建请求体，符合腾讯云API格式（仅使用Prompt）
    payload = {
        "Prompt": prompt,
        "ImageBase64": None,
        "ImageUrl": None,
        "MultiViewImages": None,
        "EnablePBR": None,
        "FaceCount": None,
        "GenerateType": None,
        "PolygonType": None
    }
    
    headers = {
        "Authorization": TENCENT_AI3D_API_KEY,
        "Content-Type": "application/json"
    }
    
    logger.info(f"🚀 提交3D生成任务到腾讯云（文生3D）: {TENCENT_AI3D_SUBMIT_URL}")
    logger.info(f"   提示词: {prompt}")
    
    logger.info(f"📥 3D API请求参数: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(TENCENT_AI3D_SUBMIT_URL, json=payload, headers=headers, timeout=60)
    
    if response.status_code != 200:
        error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    data = response.json()
    logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
    
    # 处理Response包装，提取JobId
    response_data = data.get("Response") or data
    job_id = (
        response_data.get("JobId") or 
        response_data.get("job_id") or 
        response_data.get("jobId") or
        data.get("JobId") or 
        data.get("job_id") or 
        data.get("jobId")
    )
    if not job_id:
        raise Exception(f"API响应中未找到JobId: {json.dumps(data, ensure_ascii=False)}")
    
    logger.info(f"✅ 任务已提交，JobId: {job_id}")
    return str(job_id)


def query_3d_generation_task(job_id: str, max_wait_time: int = 300) -> dict:
    """
    查询3D模型生成任务状态，轮询直到完成
    
    Args:
        job_id: 任务ID
        max_wait_time: 最大等待时间（秒），默认5分钟
    
    Returns:
        任务结果字典，包含模型URL等信息
    """
    if not TENCENT_AI3D_API_KEY:
        raise ValueError("未配置 TENCENT_AI3D_API_KEY")
    
    headers = {
        "Authorization": TENCENT_AI3D_API_KEY,
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    poll_interval = 3  # 每3秒查询一次
    
    logger.info(f"🔄 开始轮询任务状态: JobId={job_id}")
    
    while True:
        # 检查是否超时
        if time.time() - start_time > max_wait_time:
            raise TimeoutError(f"任务超时: 超过{max_wait_time}秒未完成")
        
        # 查询任务状态
        payload = {"JobId": job_id}
        response = requests.post(TENCENT_AI3D_QUERY_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"查询任务失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        data = response.json()
        
        # 处理Response包装
        response_data = data.get("Response") or data
        status = (
            response_data.get("Status") or 
            response_data.get("status") or 
            response_data.get("State") or 
            response_data.get("state")
        )
        
        logger.info(f"📊 任务状态: {status}")
        
        # 检查错误
        error_code = response_data.get("ErrorCode") or response_data.get("error_code")
        error_message = response_data.get("ErrorMessage") or response_data.get("error_message")
        if error_code and error_code != "":
            raise Exception(f"API返回错误: {error_message or error_code}")
        
        # 任务完成状态（必须明确是完成状态，且有结果文件）
        if status and status.upper() in ["SUCCESS", "COMPLETED", "DONE"]:
            # 检查是否有结果文件
            result_files = response_data.get("ResultFile3Ds") or response_data.get("result_file_3ds") or []
            if result_files:
                logger.info(f"✅ 任务完成: {json.dumps(response_data, ensure_ascii=False)}")
                return response_data
            else:
                # 状态是DONE但没有结果文件，可能是还在处理中，继续等待
                logger.warning(f"⚠️ 任务状态为 {status} 但 ResultFile3Ds 为空，继续等待...")
                time.sleep(poll_interval)
                continue
        
        # 任务失败状态
        if status and status.upper() in ["FAILED", "ERROR", "CANCELLED", "CANCELED"]:
            error_msg = error_message or response_data.get("Error") or "任务失败"
            raise Exception(f"任务失败: {error_msg}")
        
        # 任务进行中状态（RUN, PENDING, PROCESSING, QUEUED 等）
        if status and status.upper() in ["RUN", "RUNNING", "PENDING", "PROCESSING", "QUEUED", "IN_PROGRESS", "IN_QUEUE"]:
            logger.info(f"⏳ 任务进行中（状态: {status}），{poll_interval}秒后继续查询...")
            time.sleep(poll_interval)
            continue
        
        # 未知状态，默认认为进行中
        logger.warning(f"⚠️ 未知任务状态: {status}，按进行中处理，{poll_interval}秒后继续查询...")
        time.sleep(poll_interval)


def download_3d_model(model_url: str, output_path: Path) -> None:
    """
    下载3D模型文件
    
    Args:
        model_url: 模型文件URL
        output_path: 保存路径
    """
    logger.info(f"📥 开始下载3D模型: {model_url}")
    
    response = requests.get(model_url, timeout=120)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    logger.info(f"✅ 3D模型已下载: {output_path}, 大小={len(response.content)} bytes")


def extract_obj_zip(zip_path: Path, extract_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    解压OBJ格式的ZIP文件，提取.obj、.mtl和纹理文件
    
    Args:
        zip_path: ZIP文件路径
        extract_dir: 解压目标目录
    
    Returns:
        (obj_file, mtl_file, texture_file) 元组，如果文件不存在则为None
    """
    logger.info(f"📦 开始解压ZIP文件: {zip_path}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    obj_file = None
    mtl_file = None
    texture_file = None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 列出所有文件
            file_list = zip_ref.namelist()
            logger.info(f"📋 ZIP文件内容: {file_list}")
            
            # 解压所有文件
            zip_ref.extractall(extract_dir)
            
            # 查找.obj、.mtl和纹理文件（.png, .jpg等）
            for file_name in file_list:
                file_path = extract_dir / file_name
                
                # 跳过目录
                if file_path.is_dir():
                    continue
                
                file_lower = file_name.lower()
                
                if file_lower.endswith('.obj'):
                    obj_file = file_path
                    logger.info(f"✅ 找到OBJ文件: {file_path}")
                elif file_lower.endswith('.mtl'):
                    mtl_file = file_path
                    logger.info(f"✅ 找到MTL文件: {file_path}")
                elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    if texture_file is None:  # 只取第一个纹理文件
                        texture_file = file_path
                        logger.info(f"✅ 找到纹理文件: {file_path}")
            
            # 如果文件在子目录中，需要移动或重命名
            # 为了简化，我们保持原有结构，但确保能找到主文件
            
    except zipfile.BadZipFile:
        logger.error(f"❌ ZIP文件损坏: {zip_path}")
        raise Exception(f"ZIP文件损坏: {zip_path}")
    except Exception as e:
        logger.error(f"❌ 解压ZIP文件失败: {e}")
        raise
    
    if not obj_file:
        logger.warning(f"⚠️ ZIP文件中未找到.obj文件")
    
    logger.info(f"✅ ZIP文件解压完成: obj={obj_file}, mtl={mtl_file}, texture={texture_file}")
    return (obj_file, mtl_file, texture_file)


class Generate3DModelInput(BaseModel):
    """3D模型生成输入参数"""
    prompt: Optional[str] = Field(default=None, description="文本提示词，描述要生成的3D模型（文生3D模式）")
    image_url: Optional[str] = Field(default=None, description="源图片URL或本地路径（如/storage/images/文件名），用于图生3D模式。如果提供了prompt，可以同时提供图片作为参考")
    format: Literal["obj", "glb"] = Field(default="obj", description="输出格式：obj 或 glb，默认为obj")


@tool("generate_3d_model", args_schema=Generate3DModelInput)
def generate_3d_model_tool(prompt: Optional[str] = None, image_url: Optional[str] = None, format: str = "obj") -> str:
    """
    3D模型生成服务（腾讯云混元生3D API），支持文生3D和图生3D两种模式。
    
    使用腾讯云混元生3D专业版API：
    - 文生3D模式：仅使用文本提示词生成3D模型（prompt参数）
    - 图生3D模式：基于图片生成3D模型（image_url参数）
    - 混合模式：同时提供prompt和image_url，以文本提示词为主，图片作为参考
        
    Args:
        prompt: 文本提示词，描述要生成的3D模型（文生3D模式，可选）
        image_url: 源图片URL或本地路径（图生3D模式，可选）
        format: 输出格式，支持 "obj" 或 "glb"，默认为 "obj"（注意：实际格式取决于API返回）
    
    Returns:
        生成的3D模型文件路径的JSON字符串或错误信息
    
    注意：prompt 和 image_url 至少需要提供一个。
    """
    # Mock 模式：直接返回固定的模型路径
    if MOCK_MODE:
        logger.info(f"🎭 [MOCK模式] 生成3D模型: prompt={prompt}, image_url={image_url}, format={format}")
        result = {
            "model_path": f"{MOCK_MODEL_PATH}/model.obj",
            "model_url": f"{MOCK_MODEL_PATH}/model.obj",
            "preview_url": f"{MOCK_MODEL_PATH}/preview.png",
            "format": format,
            "prompt": prompt,
            "image_url": image_url,
            "mock": True,
            "message": "[MOCK] 3D模型已生成并保存到本地"
        }
        return json.dumps(result, ensure_ascii=False)
    
    try:
        # 验证至少提供了一个输入
        if not prompt and not image_url:
            return json.dumps({
                "error": "必须提供 prompt（文本提示词）或 image_url（图片URL）中的至少一个"
            }, ensure_ascii=False)
        
        logger.info(f"🎨 开始生成3D模型: prompt={prompt}, image_url={image_url}, format={format}")
        
        # 检查API Key配置
        if not TENCENT_AI3D_API_KEY:
            error_msg = "未配置 TENCENT_AI3D_API_KEY（请在 backend/.env 设置，可参考 env.example）"
            logger.error(f"❌ {error_msg}")
            return json.dumps({
                "error": error_msg
            }, ensure_ascii=False)
        
        # 验证格式
        if format not in ["obj", "glb"]:
            return json.dumps({
                "error": f"不支持的格式: {format}，仅支持 'obj' 或 'glb'"
            }, ensure_ascii=False)
        
        # 1. 提交生成任务
        if image_url:
            # 图生3D模式（可带提示词）
            job_id = submit_3d_generation_task(image_url, prompt)
        else:
            # 文生3D模式（仅使用提示词）
            job_id = submit_3d_generation_task_with_prompt(prompt)

        # 2. 轮询查询任务状态
        task_result = query_3d_generation_task(job_id, max_wait_time=300)
        
        # 3. 从ResultFile3Ds数组中获取对应格式的模型URL
        result_files = task_result.get("ResultFile3Ds") or task_result.get("result_file_3ds") or []
        
        if not result_files:
            error_msg = f"API响应中未找到ResultFile3Ds。响应: {json.dumps(task_result, ensure_ascii=False)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({
                "error": error_msg
            }, ensure_ascii=False)
        
        # 根据请求的格式查找对应的模型文件
        target_type = format.upper()  # OBJ 或 GLB
        model_file = None
        for file_info in result_files:
            file_type = file_info.get("Type") or file_info.get("type")
            if file_type and file_type.upper() == target_type:
                model_file = file_info
                break
        
        # 如果没找到对应格式，使用第一个可用的
        if not model_file:
            logger.warning(f"⚠️ 未找到 {target_type} 格式，使用第一个可用格式: {result_files[0]}")
            model_file = result_files[0]
        
        model_url = model_file.get("Url") or model_file.get("url")
        preview_image_url = model_file.get("PreviewImageUrl") or model_file.get("preview_image_url")
        
        if not model_url:
            error_msg = f"模型文件中未找到URL。文件信息: {json.dumps(model_file, ensure_ascii=False)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({
                "error": error_msg
            }, ensure_ascii=False)
        
        # 确定实际格式
        file_type = model_file.get("Type") or model_file.get("type") or target_type
        actual_format = file_type.lower()
        
        # 4. 创建模型文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        model_folder_name = f"{timestamp}_{unique_id}"
        model_folder = MODELS_DIR / model_folder_name
        model_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建模型文件夹: {model_folder}")
        
        # 从URL推断文件扩展名（可能是.zip或.glb）
        url_lower = model_url.lower()
        if url_lower.endswith('.glb'):
            ext = ".glb"
            is_zip = False
        elif url_lower.endswith('.zip'):
            # OBJ格式通常打包为zip
            ext = ".zip"
            is_zip = True
        elif url_lower.endswith('.obj'):
            ext = ".obj"
            is_zip = False
        else:
            # 根据格式确定扩展名
            is_zip = (actual_format == "obj")
            ext = ".zip" if is_zip else ".glb"
        
        # 下载文件到临时位置
        temp_filename = f"temp_model{ext}"
        temp_file_path = model_folder / temp_filename
        
        download_3d_model(model_url, temp_file_path)
        
        if not temp_file_path.exists():
            return json.dumps({
                "error": "3D模型文件下载失败"
            }, ensure_ascii=False)
        
        # 如果是OBJ格式的ZIP，需要解压
        final_obj_file = None
        final_mtl_file = None
        final_texture_file = None
        
        if is_zip and actual_format == "obj":
            # 创建临时解压目录（在模型文件夹内）
            extract_dir = model_folder / "temp_extract"
            
            # 解压ZIP文件
            obj_file, mtl_file, texture_file = extract_obj_zip(temp_file_path, extract_dir)
            
            if not obj_file:
                return json.dumps({
                    "error": "ZIP文件中未找到.obj文件"
                }, ensure_ascii=False)
            
            # 移动文件到模型文件夹，使用统一的命名
            final_obj_file = model_folder / "model.obj"
            shutil.copy2(obj_file, final_obj_file)
            logger.info(f"✅ OBJ文件已保存: {final_obj_file}")
            
            # 处理MTL文件
            if mtl_file:
                final_mtl_file = model_folder / "model.mtl"
                shutil.copy2(mtl_file, final_mtl_file)
                logger.info(f"✅ MTL文件已保存: {final_mtl_file}")
            
            # 处理纹理文件
            if texture_file:
                texture_ext = texture_file.suffix
                final_texture_file = model_folder / f"texture{texture_ext}"
                shutil.copy2(texture_file, final_texture_file)
                logger.info(f"✅ 纹理文件已保存: {final_texture_file}")
            
            # 删除临时ZIP文件和解压目录
            try:
                temp_file_path.unlink()
                shutil.rmtree(extract_dir)
                logger.info(f"✅ 已清理临时文件")
            except Exception as e:
                logger.warning(f"⚠️ 清理临时文件失败: {e}")
            
            # 使用OBJ文件作为最终文件
            file_path = final_obj_file
            model_filename = "model.obj"
        else:
            # GLB格式或其他格式，重命名为model.glb
            final_model_file = model_folder / "model.glb"
            shutil.move(temp_file_path, final_model_file)
            file_path = final_model_file
            model_filename = "model.glb"
        
        # 返回HTTP访问路径（相对于模型文件夹）
        http_path = f"/storage/models/{model_folder_name}/{model_filename}"
        
        # 5. 处理预览图（API已提供预览图URL，保存到模型文件夹）
        preview_url = None
        if preview_image_url:
            # 下载API提供的预览图到模型文件夹
            try:
                # 根据预览图URL的扩展名确定文件格式
                preview_ext = ".png"  # 默认为PNG
                if preview_image_url.lower().endswith('.jpg') or preview_image_url.lower().endswith('.jpeg'):
                    preview_ext = ".jpg"
                elif preview_image_url.lower().endswith('.png'):
                    preview_ext = ".png"
                
                preview_filename = f"preview{preview_ext}"
                preview_dest = model_folder / preview_filename
                
                preview_response = requests.get(preview_image_url, timeout=30)
                preview_response.raise_for_status()
                
                with open(preview_dest, 'wb') as f:
                    f.write(preview_response.content)
                
                preview_url = f"/storage/models/{model_folder_name}/{preview_filename}"
                logger.info(f"✅ 预览图已下载: {preview_url}")
            except Exception as e:
                logger.error(f"❌ 下载API预览图失败: {e}")
                # 如果下载失败，使用模型URL作为fallback
                preview_url = http_path
                logger.warning(f"⚠️ 预览图下载失败，使用模型URL作为fallback: {preview_url}")
        else:
            # 如果API没有提供预览图URL（理论上不应该发生），使用模型URL
            logger.warning("⚠️ API未提供预览图URL，使用模型URL")
            preview_url = http_path
        
        # 构建结果对象
        result = {
            'model_url': http_path,
            'local_path': http_path,
            'format': actual_format,
            'source_image': image_url,
            'preview_url': preview_url or http_path,
            'job_id': job_id,
            'message': f'3D模型已生成并保存到本地。模型URL: {http_path}。格式: {actual_format.upper()}。'
        }
        
        # 如果是OBJ格式，添加MTL和纹理文件路径（相对于模型文件夹）
        if actual_format == "obj":
            if final_mtl_file:
                result['mtl_url'] = f"/storage/models/{model_folder_name}/model.mtl"
            if final_texture_file:
                texture_filename = final_texture_file.name
                result['texture_url'] = f"/storage/models/{model_folder_name}/{texture_filename}"
            
            # 修复OBJ文件中的MTL引用路径
            if final_obj_file and final_mtl_file:
                try:
                    # 读取OBJ文件内容
                    with open(final_obj_file, 'r', encoding='utf-8') as f:
                        obj_content = f.read()
                    
                    # 替换MTL引用为正确的文件名（在同一文件夹中，只需文件名）
                    mtl_filename = "model.mtl"
                    # 查找并替换 mtllib 行
                    lines = obj_content.split('\n')
                    new_lines = []
                    for line in lines:
                        if line.strip().startswith('mtllib'):
                            # 替换为新的MTL文件名
                            new_lines.append(f'mtllib {mtl_filename}')
                        else:
                            new_lines.append(line)
                    
                    # 写回文件
                    with open(final_obj_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_lines))
                    
                    logger.info(f"✅ 已修复OBJ文件中的MTL引用: {mtl_filename}")
                except Exception as e:
                    logger.warning(f"⚠️ 修复OBJ文件MTL引用失败: {e}")
            
            # 修复MTL文件中的纹理路径
            if final_mtl_file and final_texture_file:
                try:
                    # 读取MTL文件内容
                    with open(final_mtl_file, 'r', encoding='utf-8') as f:
                        mtl_content = f.read()
                    
                    # 替换纹理路径为正确的文件名（在同一文件夹中，只需文件名）
                    texture_filename = final_texture_file.name
                    # 查找并替换 map_Kd 行（漫反射纹理）
                    lines = mtl_content.split('\n')
                    new_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('map_Kd') or stripped.startswith('map_Ka'):
                            # 提取原有的纹理文件名，替换为新的（只使用文件名，因为文件在同一文件夹）
                            parts = stripped.split()
                            if len(parts) >= 2:
                                new_lines.append(f'{parts[0]} {texture_filename}')
                            else:
                                new_lines.append(line)
                        else:
                            new_lines.append(line)
                    
                    # 写回文件
                    with open(final_mtl_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_lines))
                    
                    logger.info(f"✅ 已修复MTL文件中的纹理路径: {texture_filename}")
                except Exception as e:
                    logger.warning(f"⚠️ 修复MTL文件纹理路径失败: {e}")
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 3D模型生成成功: 已保存到本地 {http_path}, 格式={actual_format}")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ 3D模型生成失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({
            "error": f"生成3D模型时出错: {str(e)}"
        }, ensure_ascii=False)


if __name__ == "__main__":
    """测试工具"""
    from dotenv import load_dotenv
    from pathlib import Path
    
    # 加载 .env 文件
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ 已加载环境变量: {env_path}")
    else:
        print(f"⚠️  未找到 .env 文件: {env_path}")
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试生成3D模型
    # print("\n测试 generate_3d_model 工具...")
    # result = generate_3d_model_tool.invoke({
    #     "image_url": "/storage/images/volcano_20260102_160559_130842a2_卡通盲盒人物Q版造型圆润可爱大大的眼睛色彩鲜艳穿着.jpg",
    #     "format": "obj"
    # })
    # print("生成结果:", result)

    result = query_3d_generation_task("1398683551163359232")
    print("查询结果:", json.dumps(result, ensure_ascii=False))

