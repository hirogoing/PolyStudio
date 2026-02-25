"""
Qwen-TTS工具 - 阿里云百炼千问语音合成服务
包含声音设计和声音复刻功能（一步式生成音频）
https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2975034
"""
import json
import logging
import os
import requests
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 优先加载 backend/.env
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 从环境变量获取配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com").strip()

# 音频存储目录
STORAGE_DIR = BASE_DIR / "storage"
AUDIOS_DIR = STORAGE_DIR / "audios"
AUDIOS_DIR.mkdir(parents=True, exist_ok=True)

# 语言代码映射
LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian"
}


def save_audio_from_base64(base64_audio: str, text: str = "", prefix: str = "qwen_tts") -> str:
    """
    从Base64数据保存音频到本地
    
    Args:
        base64_audio: Base64编码的音频数据
        text: 合成的文本（用于生成文件名）
        prefix: 文件名前缀
    
    Returns:
        本地文件路径（相对路径）
    """
    try:
        # 解码Base64
        audio_data = base64.b64decode(base64_audio)
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_text = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text[:30])
        safe_text = safe_text.replace(" ", "_")
        
        filename = f"{prefix}_{timestamp}_{unique_id}"
        if safe_text:
            filename += f"_{safe_text}"
        filename += ".wav"  # 默认wav格式
        
        file_path = AUDIOS_DIR / filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        # 返回HTTP访问路径
        http_path = f"/storage/audios/{filename}"
        logger.info(f"✅ 音频已保存到本地: {file_path}")
        return http_path
        
    except Exception as e:
        logger.error(f"❌ 保存音频失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def prepare_audio_input(audio_path: str) -> str:
    """
    准备音频输入，将本地文件转换为Base64 Data URI
    
    Args:
        audio_path: 本地路径（如 /storage/audios/xxx.mp3）
    
    Returns:
        Base64 Data URI 字符串
    """
    # 检查是否是本地路径
    if audio_path.startswith("/storage/"):
        file_path = BASE_DIR / audio_path.lstrip("/")
        if not file_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {file_path}")
        
        logger.info(f"📁 读取本地音频文件: {file_path}")
        
        # 读取文件
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        # 获取MIME类型
        ext = file_path.suffix.lower()
        mime_type = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4"
        }.get(ext, "audio/mpeg")
        
        # 转换为Base64
        base64_data = base64.b64encode(audio_data).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{base64_data}"
        
        logger.info(f"✅ 已转换为Base64 Data URI: {mime_type}")
        return data_uri
    
    # 如果是URL，直接返回
    return audio_path


class VoiceDesignInput(BaseModel):
    """声音设计输入参数"""
    voice_description: str = Field(description="音色描述，详细描述音色特征，如'沉稳的中年男性，音色低沉浑厚，富有磁性，语速平稳'")
    text: str = Field(description="要合成的文本内容，支持中英文等多语言，最长1024字符")
    language: str = Field(default="zh", description="语言代码，zh/en/de/it/pt/es/ja/ko/fr/ru，默认zh")


@tool("qwen_voice_design", args_schema=VoiceDesignInput)
def qwen_voice_design_tool(
    voice_description: str,
    text: str,
    language: str = "zh"
) -> str:
    """
    Qwen-TTS声音设计工具 - 通过文本描述音色特征，生成定制化语音
    
    输入音色描述（如"活泼的年轻女声，语速较快"）和要合成的文本，直接生成音频。
    
    **用途**：生成音色样本供用户试听确认
    
    **工作流**：
    1. 使用此工具生成音色样本（传入测试文本，如"大家好，欢迎来到节目"）
    2. 展示音频样本，等待用户确认是否满意
    3. 用户确认后，使用返回的 audio_url 作为 reference_audio，调用 qwen_voice_cloning 批量合成所有片段
    
    Args:
        voice_description: 音色描述文本（描述性别、年龄、音调、语速、情感等）
        text: 要合成的测试文本（建议使用脚本的第一句话）
        language: 语言代码（zh/en/de/it/pt/es/ja/ko/fr/ru）
    
    Returns:
        包含audio_url等信息的JSON字符串，audio_url可作为后续qwen_voice_cloning的reference_audio
    """
    try:
        if not DASHSCOPE_API_KEY:
            return "Error: 未配置 DASHSCOPE_API_KEY（请在 backend/.env 设置）"
        
        logger.info(f"🎨 开始声音设计: voice_description={voice_description[:50]}..., text={text[:50]}...")
        
        # Step 1: 创建音色（使用text作为preview_text）
        url = f"{DASHSCOPE_BASE_URL.rstrip('/')}/api/v1/services/audio/tts/customization"
        
        payload = {
            "model": "qwen-voice-design",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vd-2026-01-26",  # 使用非实时模型
                "voice_prompt": voice_description,
                "preview_text": text,  # 使用用户要合成的文本作为预览文本
                "preferred_name": f"vd{uuid.uuid4().hex[:8]}",
                "language": language
            },
            "parameters": {
                "sample_rate": 24000,
                "response_format": "wav"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🚀 调用声音设计API")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API调用失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        data = response.json()
        logger.info(f"📥 API响应成功")
        
        # 解析返回结果 - 直接使用preview_audio
        if "output" in data and "preview_audio" in data["output"]:
            preview_audio = data["output"]["preview_audio"]
            base64_audio = preview_audio.get("data", "")
            voice_name = data["output"].get("voice", "")
            
            # 保存音频
            local_path = save_audio_from_base64(base64_audio, text, "voice_design")
            
            result = {
                'audio_url': local_path,
                'local_path': local_path,
                'text': text,
                'voice_description': voice_description,
                'voice_name': voice_name,
                'language': language,
                'provider': 'qwen-tts-voice-design',
                'request_id': data.get("request_id", ""),
                'message': '声音设计完成，音频已生成'
            }
            
            result_json = json.dumps(result, ensure_ascii=False)
            logger.info(f"✅ 声音设计成功: 已保存到本地 {local_path}")
            return result_json
        else:
            return f"Error: 响应中未找到音频数据. Response: {json.dumps(data)}"
        
    except Exception as e:
        logger.error(f"❌ 声音设计失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


class VoiceCloningInput(BaseModel):
    """声音复刻输入参数"""
    reference_audio: str = Field(description="参考音频文件路径，用于复刻音色，支持WAV/MP3/M4A格式，推荐10-20秒")
    text: str = Field(description="要合成的文本内容，支持中英文等多语言")
    language: str = Field(default="zh", description="语言代码，zh/en/de/it/pt/es/ja/ko/fr/ru，默认zh")


@tool("qwen_voice_cloning", args_schema=VoiceCloningInput)
def qwen_voice_cloning_tool(
    reference_audio: str,
    text: str,
    language: str = "zh"
) -> str:
    """
    Qwen-TTS声音复刻工具 - 基于参考音频复刻音色并合成语音
    
    输入一段参考音频和要合成的文本，生成具有相同音色的语音。
    
    **两种使用场景**：
    
    场景1 - 生成音色样本（用于试听确认）：
    - 如果用户上传了参考音频，使用此工具生成测试样本
    - 示例：qwen_voice_cloning(reference_audio="/storage/audios/user_upload.mp3", text="测试文本")
    
    场景2 - 批量合成（保持音色一致）：
    - 用户确认音色样本后，使用相同的 reference_audio 批量合成所有片段
    - reference_audio 来源：
      * 用户上传的音频
      * qwen_voice_design 生成的音频样本（从返回结果的 audio_url 获取）
    - 示例：qwen_voice_cloning(reference_audio=样本audio_url, text="每段对话")
    
    **重要**：同一角色的所有对话都使用相同的 reference_audio，确保音色一致
    
    Args:
        reference_audio: 参考音频文件路径（用户上传或voice_design生成的audio_url）
        text: 要合成的文本内容
        language: 语言代码（zh/en/de/it/pt/es/ja/ko/fr/ru）
    
    Returns:
        包含audio_url等信息的JSON字符串
    """
    try:
        if not DASHSCOPE_API_KEY:
            return "Error: 未配置 DASHSCOPE_API_KEY（请在 backend/.env 设置）"
        
        logger.info(f"🎤 开始声音复刻: reference_audio={reference_audio}, text={text[:50]}...")
        
        # 准备音频输入
        audio_data = prepare_audio_input(reference_audio)
        
        # Step 1: 创建音色
        url = f"{DASHSCOPE_BASE_URL.rstrip('/')}/api/v1/services/audio/tts/customization"
        
        payload = {
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vc-2026-01-22",  # 使用声音复刻模型
                "preferred_name": f"vc{uuid.uuid4().hex[:8]}",
                "audio": {
                    "data": audio_data
                },
                "language": language
            }
        }
        
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🚀 调用声音复刻API - 创建音色")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"创建音色失败: status={response.status_code}, body={response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        data = response.json()
        voice_name = data.get("output", {}).get("voice", "")
        
        if not voice_name:
            return f"Error: 未获取到音色名称. Response: {json.dumps(data)}"
        
        logger.info(f"✅ 音色创建成功: {voice_name}")
        
        # Step 2: 使用音色进行语音合成
        synth_url = f"{DASHSCOPE_BASE_URL.rstrip('/')}/api/v1/services/aigc/multimodal-generation/generation"
        
        synth_payload = {
            "model": "qwen3-tts-vc-2026-01-22",
            "input": {
                "text": text,
                "voice": voice_name,
                "language_type": LANGUAGE_MAP.get(language.lower(), "Auto")
            }
        }
        
        logger.info(f"🚀 调用语音合成API")
        synth_response = requests.post(synth_url, json=synth_payload, headers=headers, timeout=120)
        
        if synth_response.status_code != 200:
            error_msg = f"语音合成失败: status={synth_response.status_code}, body={synth_response.text}"
            logger.error(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        synth_data = synth_response.json()
        
        # 解析音频URL
        if "output" in synth_data and "audio" in synth_data["output"] and "url" in synth_data["output"]["audio"]:
            audio_url = synth_data["output"]["audio"]["url"]
            
            # 下载并保存音频
            logger.info(f"📥 开始下载音频: {audio_url}")
            audio_response = requests.get(audio_url, timeout=60)
            audio_response.raise_for_status()
            
            # 保存音频
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text[:30])
            safe_text = safe_text.replace(" ", "_")
            
            ext = ".wav" if ".wav" in audio_url else ".mp3"
            filename = f"voice_cloning_{timestamp}_{unique_id}"
            if safe_text:
                filename += f"_{safe_text}"
            filename += ext
            
            file_path = AUDIOS_DIR / filename
            with open(file_path, "wb") as f:
                f.write(audio_response.content)
            
            local_path = f"/storage/audios/{filename}"
            
            result = {
                'audio_url': local_path,
                'local_path': local_path,
                'original_url': audio_url,
                'text': text,
                'reference_audio': reference_audio,
                'voice_name': voice_name,
                'language': language,
                'provider': 'qwen-tts-voice-cloning',
                'request_id': synth_data.get("request_id", ""),
                'message': '声音复刻完成，音频已生成'
            }
            
            result_json = json.dumps(result, ensure_ascii=False)
            logger.info(f"✅ 声音复刻成功: 已保存到本地 {local_path}")
            return result_json
        else:
            return f"Error: 响应中未找到音频URL. Response: {json.dumps(synth_data)}"
        
    except Exception as e:
        logger.error(f"❌ 声音复刻失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


if __name__ == "__main__":
    """测试工具"""
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ 已加载环境变量: {env_path}")
    else:
        print(f"⚠️  未找到 .env 文件: {env_path}")
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试声音设计
    print("\n测试声音设计工具...")
    # result = qwen_voice_design_tool.invoke({
    #     "voice_description": "活泼的年轻女性，语速较快，带有明显的上扬语调，",
    #     "text": "大家好，欢迎来到今天的节目！",
    #     "language": "zh"
    # })
    # print("结果:", result)

    reference_audio = "/storage/audios/voice_design_20260212_143228_a996e778_大家好欢迎来到今天的节目.wav"
    
    "/storage/audios/voice_cloning_20260212_143535_cc4b88df_这是使用复刻音色合成的语音.wav"
    # 测试声音复刻
    print("\n测试声音复刻工具...")
    result = qwen_voice_cloning_tool.invoke({
        "reference_audio":reference_audio ,
        "text": "这是使用复刻音色合成的语音。",
        "language": "zh"
    })
    print("结果:", result)
