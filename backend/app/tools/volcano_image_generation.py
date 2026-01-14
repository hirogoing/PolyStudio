"""
火山引擎图片生成工具 - 使用 Seedream 4.0-4.5 API 生成图像
"""
import json
import logging
import os
import requests
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlparse
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 可选：用于将下载图片统一转换到 sRGB，减少 <img> 与 canvas 渲染差异
try:
    from PIL import Image, ImageCms  # type: ignore
    from io import BytesIO
except Exception:  # pragma: no cover
    Image = None
    ImageCms = None
    BytesIO = None  # type: ignore
    logger.warning("⚠️ 未安装 Pillow：将无法进行 sRGB 归一化，<img> 与 Excalidraw(canvas) 可能出现颜色差异。请安装 requirements.txt 后重启后端。")

# 优先加载 backend/.env（避免直接运行工具脚本时环境未加载）
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 从环境变量获取配置
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "").strip()
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
VOLCANO_IMAGE_MODEL = os.getenv("VOLCANO_IMAGE_MODEL", "seedream-4.5").strip()
# 若编辑模型不同，可单独配置；缺省复用生成模型
VOLCANO_EDIT_MODEL = os.getenv("VOLCANO_EDIT_MODEL", VOLCANO_IMAGE_MODEL).strip()
# Mock 模式配置
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Mock 图片路径（启用 MOCK_MODE 时必须配置）
MOCK_IMAGE_PATH = os.getenv("MOCK_IMAGE_PATH", "").strip()
if MOCK_MODE and not MOCK_IMAGE_PATH:
    raise RuntimeError(
        "MOCK_MODE=true 时，必须配置 MOCK_IMAGE_PATH。"
        "请在 backend/.env 中设置 MOCK_IMAGE_PATH=/storage/images/your_image.png"
    )

# 图片存储目录
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"

# 确保图片存储目录存在
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 宽高比到像素值的映射
ASPECT_RATIO_MAP = {
    "1:1": (2048, 2048),
    "4:3": (2304, 1728),
    "3:4": (1728, 2304),
    "16:9": (2560, 1440),
    "9:16": (1440, 2560),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "21:9": (3024, 1296),
}


def parse_size(size: str) -> str:
    """
    解析尺寸参数，支持宽高比枚举、自定义格式或API格式
    
    Args:
        size: 宽高比字符串（如 "16:9", "4:3"）、自定义格式（如 "1024x1024"）或API格式（如 "2K"）
    
    Returns:
        返回API可接受的尺寸字符串格式（如 "2K" 或 "2048x2048"）
    """
    # 先检查是否是API格式（如 "2K", "4K" 等）
    if size.upper() in ["2K", "4K", "1K"]:
        return size.upper()
    
    # 检查是否是宽高比枚举
    if size in ASPECT_RATIO_MAP:
        width, height = ASPECT_RATIO_MAP[size]
        return f"{width}x{height}"
    
    # 尝试解析自定义格式 "widthxheight"
    if "x" in size or "X" in size:
        parts = size.replace("X", "x").split("x")
        if len(parts) == 2:
            try:
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                return f"{width}x{height}"
            except ValueError:
                pass
    
    # 默认返回 1:1
    logger.warning(f"无法解析尺寸参数: {size}，使用默认 1:1 (2048x2048)")
    width, height = ASPECT_RATIO_MAP["1:1"]
    return f"{width}x{height}"


def prepare_image_input(image_url: str) -> Union[str, list]:
    """
    准备图片输入，只处理本地文件（转Base64），不支持公网URL（会过期）
    
    Args:
        image_url: 本地路径（如 /storage/images/xxx.jpg）或 localhost URL（如 http://localhost:8000/storage/images/xxx.jpg）
    
    Returns:
        Base64编码字符串
    
    Raises:
        FileNotFoundError: 本地文件不存在
        ValueError: 不支持公网URL（会过期）
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
    
    # 公网URL不支持（会过期），提示错误
    raise ValueError(
        f"不支持公网URL（会过期）: {image_url[:50]}...\n"
        f"请使用本地路径（如 /storage/images/xxx.jpg）或 localhost URL（如 http://localhost:8000/storage/images/xxx.jpg）"
    )


def download_and_save_image(image_url: str, prompt: str = "") -> str:
    """
    下载图片并保存到本地
    
    Args:
        image_url: 图片URL
        prompt: 提示词（用于生成文件名）
    
    Returns:
        本地文件路径（相对路径）
    """
    try:
        logger.info(f"📥 开始下载图片: {image_url}")
        
        # 下载图片
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        # 从URL获取文件扩展名，如果没有则默认为png
        parsed_url = urlparse(image_url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1] or ".png"
        if not ext.startswith("."):
            ext = ".png"
        
        # 生成唯一文件名：时间戳_随机ID_提示词前20字符
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        # 清理提示词，只保留字母数字和空格，用于文件名
        safe_prompt = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in prompt[:30])
        safe_prompt = safe_prompt.replace(" ", "_")
        filename = f"volcano_{timestamp}_{unique_id}_{safe_prompt}{ext}" if safe_prompt else f"volcano_{timestamp}_{unique_id}{ext}"
        
        file_path = IMAGES_DIR / filename

        # 保存文件（优先进行 sRGB 归一化，避免 Excalidraw(canvas) 与聊天(<img>) 观感不一致）
        saved = False
        if Image is not None and BytesIO is not None:
            try:
                im = Image.open(BytesIO(response.content))
                im.load()

                # 统一转换到 sRGB，并移除 ICC profile
                if ImageCms is not None:
                    icc = getattr(im, "info", {}).get("icc_profile")
                    if icc:
                        try:
                            src_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                            dst_profile = ImageCms.createProfile("sRGB")
                            output_mode = "RGBA" if (
                                im.mode in ("RGBA", "LA") or ("transparency" in getattr(im, "info", {}))
                            ) else "RGB"
                            im = ImageCms.profileToProfile(im, src_profile, dst_profile, outputMode=output_mode)
                        except Exception:
                            # ICC 转换失败：退化为普通模式转换（不抛）
                            pass

                # 彻底去掉 ICC（避免浏览器两条渲染链路按不同 profile 解释）
                try:
                    if getattr(im, "info", None) and "icc_profile" in im.info:
                        im.info.pop("icc_profile", None)
                except Exception:
                    pass

                # 关键策略：
                # - 若图片不透明：统一存为 JPEG（去掉 PNG 的 gAMA/sRGB/cHRM 等色彩块差异，减少 <img> vs canvas 偏色）
                # - 若图片含透明：存为 PNG（保留 alpha）
                has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in getattr(im, "info", {}))
                is_transparent = False
                if has_alpha:
                    try:
                        alpha = im.getchannel("A")
                        lo, hi = alpha.getextrema()
                        is_transparent = lo < 255
                    except Exception:
                        is_transparent = True

                if not is_transparent:
                    # Opaque -> JPEG
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    filename = os.path.splitext(filename)[0] + ".jpg"
                    file_path = IMAGES_DIR / filename
                    im.save(file_path, format="JPEG", quality=95, optimize=True, progressive=True)
                else:
                    # Transparent -> PNG
                    filename = os.path.splitext(filename)[0] + ".png"
                    file_path = IMAGES_DIR / filename
                    if im.mode not in ("RGBA", "RGB"):
                        im = im.convert("RGBA")
                    im.save(file_path, format="PNG", optimize=True)

                saved = True
                logger.info("🎛️ 已进行 sRGB 归一化并保存（移除 ICC profile）")
            except Exception as e:
                logger.warning(f"⚠️ sRGB 归一化失败，回退为原始字节保存: {e}")

        if not saved:
            with open(file_path, "wb") as f:
                f.write(response.content)
        
        # 返回HTTP访问路径（以/storage/开头，前端可以直接使用）
        http_path = f"/storage/images/{filename}"
        logger.info(f"✅ 图片已保存到本地: {file_path}")
        logger.info(f"   可通过HTTP访问: {http_path}")
        return http_path
        
    except Exception as e:
        logger.error(f"❌ 下载图片失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果下载失败，返回原始URL
        return image_url


class GenerateVolcanoImageInput(BaseModel):
    """火山引擎图像生成输入参数"""
    prompt: str = Field(description="图像生成的提示词，详细描述想要生成的图像内容，支持中英文")
    size: str = Field(default="1:1", description="图片尺寸，支持宽高比枚举（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 21:9）或自定义格式（如 2048x2048），默认 1:1")


@tool("generate_volcano_image", args_schema=GenerateVolcanoImageInput)
def generate_volcano_image_tool(prompt: str, size: str = "1:1") -> str:
    """
    火山引擎 AI 绘画（图片生成）服务，使用 Seedream 4.0-4.5 API 生成图像。
    输入文本描述，返回基于文本信息绘制的图片 URL。
    
    Args:
        prompt: 图像生成的提示词（支持中英文）
        size: 图片尺寸，支持宽高比枚举（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 21:9）或自定义格式（如 2048x2048），默认 1:1
    
    Returns:
        生成的图像URL的JSON字符串或错误信息
    """
    # Mock 模式：直接返回固定的图片路径
    if MOCK_MODE:
        logger.info(f"🎭 [MOCK模式] 生成图像: prompt={prompt}, size={size}")
        result = {
            'image_url': MOCK_IMAGE_PATH,
            'original_url': MOCK_IMAGE_PATH,
            'local_path': MOCK_IMAGE_PATH,
            'prompt': prompt,
            'provider': 'volcano',
            'mock': True,
            'message': '[MOCK] 图片已生成并保存到本地'
        }
        return json.dumps(result, ensure_ascii=False)
    
    try:
        if not VOLCANO_API_KEY:
            return "Error generating image: 未配置 VOLCANO_API_KEY（请在 backend/.env 设置，可参考 env.example）"
        
        # 解析尺寸参数
        size_value = parse_size(size)
        logger.info(f"🎨 开始使用火山引擎生成图像: prompt={prompt}, size={size} -> {size_value}")

        # 火山引擎 API 端点
        url = f"{VOLCANO_BASE_URL.rstrip('/')}/images/generations"
        
        # 构建请求体（固定生成1张图片）
        payload = {
            "model": VOLCANO_IMAGE_MODEL,
            "prompt": prompt,
            "size": size_value,
            "n": 1,
            "response_format": "url",  # 返回图片URL
            "stream": False,
            "watermark": True
        }
        
        headers = {
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🚀 调用火山引擎生成 API")
        logger.info(f"   URL: {url}")
        logger.info(f"   请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error generating image: {error_msg}"
            
        data = response.json()
        logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
        
        # 解析返回结果
        # 火山引擎可能返回的格式: {"data": [{"url": "..."}]} 或 {"images": [{"url": "..."}]}
        image_url = None
        
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            image_url = data["data"][0].get("url")
        elif "images" in data and isinstance(data["images"], list) and len(data["images"]) > 0:
            image_url = data["images"][0].get("url")
        elif "url" in data:
            image_url = data["url"]
        
        if not image_url:
            return f"Error: No image URL in response. Response: {json.dumps(data)}"
        
        # 下载并保存图片
        local_path = download_and_save_image(image_url, prompt)
        
        # 返回结果
        result = {
            'image_url': local_path,
            'original_url': image_url,
            'local_path': local_path,
            'prompt': prompt,
            'provider': 'volcano',
            'message': '图片已生成并保存到本地'
        }
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 火山引擎图像生成成功: 已保存到本地 {local_path}")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ 火山引擎图像生成失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating image: {str(e)}"


class EditVolcanoImageInput(BaseModel):
    """火山引擎图像编辑输入参数"""
    prompt: str = Field(description="图像编辑的提示词，详细描述想要达到的效果，支持中英文")
    image_url: str = Field(description="需要编辑的源图片URL或本地路径（/storage/images/...）")
    size: str = Field(default="1:1", description="输出图片尺寸，支持宽高比枚举（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 21:9）或自定义格式（如 2048x2048），默认 1:1")


@tool("edit_volcano_image", args_schema=EditVolcanoImageInput)
def edit_volcano_image_tool(prompt: str, image_url: str, size: str = "1:1") -> str:
    """
    火山引擎图片编辑服务（Seedream 4.0-4.5 API），基于已有图片和提示词生成新的图片。

    Args:
        prompt: 编辑提示词（支持中英文）
        image_url: 原图URL或本地路径（如 /storage/images/xxx.png）
        size: 输出图片尺寸，支持宽高比枚举（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 21:9）或自定义格式（如 2048x2048），默认 1:1

    Returns:
        生成的图像URL的JSON字符串或错误信息
    """
    # Mock 模式：直接返回固定的图片路径
    if MOCK_MODE:
        logger.info(f"🎭 [MOCK模式] 编辑图像: prompt={prompt}, image_url={image_url}, size={size}")
        result = {
            'image_url': MOCK_IMAGE_PATH,
            'original_url': MOCK_IMAGE_PATH,
            'local_path': MOCK_IMAGE_PATH,
            'prompt': prompt,
            'source_image': image_url,
            'provider': 'volcano',
            'mock': True,
            'message': '[MOCK] 图片已编辑并保存到本地'
        }
        return json.dumps(result, ensure_ascii=False)
    
    try:
        if not VOLCANO_API_KEY:
            return "Error editing image: 未配置 VOLCANO_API_KEY（请在 backend/.env 设置，可参考 env.example）"

        # 解析尺寸参数
        size_value = parse_size(size)
        logger.info(f"🖌️ 开始使用火山引擎编辑图像: prompt={prompt}, image_url={image_url}, size={size} -> {size_value}")

        # 准备图片输入（支持本地文件转Base64或公网URL）
        image_input = prepare_image_input(image_url)

        # 火山引擎图片编辑端点（使用 generations 接口，支持 image 参数）
        url = f"{VOLCANO_BASE_URL.rstrip('/')}/images/generations"

        payload = {
            "model": VOLCANO_EDIT_MODEL,
            "prompt": prompt,
            "image": image_input,  # 可以是URL字符串、Base64字符串或数组
            "size": size_value,
            "response_format": "url",
            "stream": False,
            "watermark": True
        }

        headers = {
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json"
        }

        # 打印请求参数（隐藏Base64数据和公网URL）
        payload_for_log = payload.copy()
        if isinstance(payload_for_log.get("image"), str):
            if payload_for_log["image"].startswith("data:image"):
                payload_for_log["image"] = "data:image/...;base64,<Base64数据已隐藏>"
            elif payload_for_log["image"].startswith("http"):
                payload_for_log["image"] = "<公网URL已隐藏>"
        
        logger.info(f"🚀 调用火山引擎编辑 API: model={payload['model']}, url={url}")
        logger.info(f"   请求参数: {json.dumps(payload_for_log, ensure_ascii=False, indent=2)}， 原始URL: {image_url}")

        response = requests.post(url, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error editing image: {error_msg}"

        data = response.json()
        logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")

        image_urls = []
        if "data" in data and isinstance(data["data"], list):
            image_urls = [item.get("url") for item in data["data"] if item.get("url")]
        elif "images" in data and isinstance(data["images"], list):
            image_urls = [item.get("url") for item in data["images"] if item.get("url")]
        elif "url" in data:
            image_urls = [data["url"]]

        if not image_urls:
            return f"Error: No image URL in response. Response: {json.dumps(data)}"

        new_image_url = image_urls[0]
        local_path = download_and_save_image(new_image_url, prompt)

        result = {
            "image_url": local_path,
            "original_url": new_image_url,
            "local_path": local_path,
            "prompt": prompt,
            "source_image": image_url,
            "provider": "volcano",
            "message": "图片已编辑并保存到本地",
        }

        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 火山引擎图像编辑成功: 已保存到本地 {local_path}")
        return result_json

    except Exception as e:
        logger.error(f"❌ 火山引擎图像编辑失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error editing image: {str(e)}"


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
    
    # 测试生成图片
    # print("\n测试 generate_volcano_image 工具...")
    # result = generate_volcano_image_tool.invoke({
    #     "prompt": "美丽的日落",
    #     "size": "16:9",
    #     "num_images": 1
    # })
    # print("生成结果:", result)
    
    # 测试编辑图片（需要先有生成的图片URL）
    # print("\n测试 edit_volcano_image 工具...")
    result = edit_volcano_image_tool.invoke({
        "prompt": "让它变成日出",
        "image_url": "/storage/images/volcano_20251223_231147_3c85e4db_美丽的日落.jpg",
        "size": "4:3"
    })
    print("编辑结果:", result)

