"""
图像生成工具 - 使用 SiliconFlow API 生成图像
"""
import json
import logging
import os
import requests
import uuid
from datetime import datetime
from pathlib import Path
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

# 从环境变量获取配置，与 agent_service.py 保持一致
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").strip()

# 图像生成/编辑模型可通过环境变量覆盖
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "Qwen/Qwen-Image").strip()
EDIT_IMAGE_MODEL_NAME = os.getenv("EDIT_IMAGE_MODEL_NAME", "Qwen/Qwen-Image-Edit-2509").strip()

# 图片存储目录
# BASE_DIR 已在上方定义
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"

# 确保图片存储目录存在
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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
        filename = f"{timestamp}_{unique_id}_{safe_prompt}{ext}" if safe_prompt else f"{timestamp}_{unique_id}{ext}"
        
        file_path = IMAGES_DIR / filename

        # 保存文件（优先进行 sRGB 归一化，避免 Excalidraw(canvas) 与聊天(<img>) 观感不一致）
        saved = False
        if Image is not None and BytesIO is not None:
            try:
                im = Image.open(BytesIO(response.content))
                im.load()

                # 统一转换到 sRGB，并移除 ICC profile
                # 说明：<img> 通常会做 ICC/广色域到显示器色域的转换，但 2D canvas 往往工作在 sRGB，
                # 导致同图在聊天与画板“观感不一致”。我们在落盘前把像素值归一化到 sRGB 并去掉 ICC。
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

class GenerateImageInput(BaseModel):
    """图像生成输入参数"""
    prompt: str = Field(description="图像生成的提示词，详细描述想要生成的图像内容，必须是英文")

@tool("generate_image", args_schema=GenerateImageInput)
def generate_image_tool(prompt: str) -> str:
    """
    AI 绘画（图片生成）服务，输入文本描述，返回基于文本信息绘制的图片 URL。
    
    Args:
        prompt: 图像生成的提示词
    
    Returns:
        生成的图像URL的JSON字符串或错误信息
    """
    try:
        if not OPENAI_API_KEY:
            return "Error generating image: 未配置 OPENAI_API_KEY（请在 backend/.env 设置，可参考 env.example）"
        logger.info(f"🎨 开始生成图像: prompt={prompt}")

        url = f"{OPENAI_BASE_URL.rstrip('/')}/images/generations"
        
        payload = {
            "model": IMAGE_MODEL_NAME,  # 使用指定的模型（可通过环境变量覆盖）
            "prompt": prompt,
            "image_size": "1024x1024" # 添加默认尺寸，避免API报错
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🚀 调用 SiliconFlow API: model={payload['model']}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error generating image: {error_msg}"
            
        data = response.json()
        logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
        
        # 解析返回结果
        # 预期格式: {"images": [{"url": "..."}]}
        if "images" in data and len(data["images"]) > 0:
            image_url = data["images"][0].get("url")
            if image_url:
                # 下载并保存图片到本地
                local_path = download_and_save_image(image_url, prompt)
                
                # 返回结果：image_url 使用本地路径，确保历史记录中保存的是本地路径（不会过期）
                # original_url 保留原始URL用于调试或备份
                # 注意：image_url字段是用于后续edit_image工具的主要标识符
                result = {
                    'image_url': local_path,  # 主要使用本地路径，前端直接使用这个，也是edit_image工具需要的URL
                    'original_url': image_url,  # 保留原始URL作为备份
                    'local_path': local_path,  # 明确标识本地路径
                    'prompt': prompt,
                    'message': f'图片已生成并保存到本地。图片URL: {local_path}。如需编辑此图片，请使用此URL。'
                }
                
                result_json = json.dumps(result, ensure_ascii=False)
                logger.info(f"✅ 图像生成成功: 已保存到本地 {local_path}, 原始URL={image_url}")
                return result_json
        
        return f"Error: No image URL in response. Response: {json.dumps(data)}"
        
    except Exception as e:
        logger.error(f"❌ 图像生成失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating image: {str(e)}"

class EditImageInput(BaseModel):
    """图像编辑输入参数"""
    prompt: str = Field(description="图像编辑的提示词，详细描述想要达到的效果，必须是英文")
    image_url: str = Field(description="需要编辑的源图片URL或本地路径。可以从对话历史中查找之前生成的图片URL（在generate_image工具的结果中查找image_url字段），或者使用本地路径（如/storage/images/文件名）。如果是本地路径，会自动转换为完整URL。")

@tool("edit_image", args_schema=EditImageInput)
def edit_image_tool(prompt: str, image_url: str) -> str:
    """
    AI 图像编辑服务，基于已有图片和文本提示词修改图片。
    可以用来修改图片风格、内容等。
    
    重要提示：
    - 当用户要求修改、编辑或变换之前生成的图片时，应该从对话历史中查找之前generate_image工具返回的结果
    - 在工具返回的JSON结果中查找"image_url"字段，这就是需要编辑的图片URL
    - 如果用户没有明确提供图片URL，必须从对话历史中查找最近生成的图片URL
    
    Args:
        prompt: 图像编辑的提示词
        image_url: 原图URL或本地路径（如/storage/images/文件名）。如果是本地路径，会自动转换为完整URL。
    
    Returns:
        生成的图像URL的JSON字符串或错误信息
    """
    try:
        if not OPENAI_API_KEY:
            return "Error editing image: 未配置 OPENAI_API_KEY（请在 backend/.env 设置，可参考 env.example）"
        logger.info(f"🎨 开始编辑图像: prompt={prompt}, image_url={image_url}")
        
        # 处理本地路径：如果是本地路径（以/storage/开头），转换为完整URL
        actual_image_url = image_url
        if image_url.startswith("/storage/"):
            # 本地路径，需要转换为完整URL
            # 默认使用localhost:8000，实际部署时需要从环境变量获取
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            actual_image_url = f"{base_url}{image_url}"
            logger.info(f"🔄 本地路径转换为完整URL: {image_url} -> {actual_image_url}")
        
        url = f"{OPENAI_BASE_URL.rstrip('/')}/images/generations"
        
        payload = {
            "model": EDIT_IMAGE_MODEL_NAME,
            "prompt": prompt,
            "image": actual_image_url,  # 使用转换后的URL
            # 编辑模型可能不支持 image_size，先不传或根据文档确认
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🚀 调用 SiliconFlow API: model={payload['model']}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error editing image: {error_msg}"
            
        data = response.json()
        logger.info(f"📥 API响应: {json.dumps(data, ensure_ascii=False)}")
        
        # 解析返回结果
        if "images" in data and len(data["images"]) > 0:
            new_image_url = data["images"][0].get("url")
            if new_image_url:
                # 下载并保存图片到本地
                local_path = download_and_save_image(new_image_url, prompt)
                
                # 返回结果：image_url 使用本地路径，确保历史记录中保存的是本地路径（不会过期）
                # original_url 保留原始URL用于调试或备份
                # 注意：image_url字段是用于后续edit_image工具的主要标识符
                result = {
                    'image_url': local_path,  # 主要使用本地路径，前端直接使用这个，也是edit_image工具需要的URL
                    'original_url': new_image_url,  # 保留原始URL作为备份
                    'local_path': local_path,  # 明确标识本地路径
                    'prompt': prompt,
                    'source_image': image_url,  # 记录源图片URL
                    'message': '图片已编辑并保存到本地'
                }
                
                result_json = json.dumps(result, ensure_ascii=False)
                logger.info(f"✅ 图像编辑成功: 已保存到本地 {local_path}, 原始URL={new_image_url}")
                return result_json
        
        return f"Error: No image URL in response. Response: {json.dumps(data)}"
        
    except Exception as e:
        logger.error(f"❌ 图像编辑失败: {str(e)}")
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
    print("\n测试 generate_image 工具...")
    result = generate_image_tool.invoke({"prompt": "a beautiful sunset over the ocean"})
    print("生成结果:", result)
    
    # 测试编辑图片（需要先有生成的图片URL）
    # print("\n测试 edit_image 工具...")
    # result = edit_image_tool.invoke({
    #     "prompt": "make it more colorful",
    #     "image_url": "/storage/images/xxx.jpg"
    # })
    # print("编辑结果:", result)
