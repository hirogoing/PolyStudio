"""
音频混音工具 - 播客后期制作
包含音频拼接、背景音乐混音、音量调整等功能
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 优先加载 backend/.env
BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 音频存储目录
STORAGE_DIR = BASE_DIR / "storage"
AUDIOS_DIR = STORAGE_DIR / "audios"
BGM_DIR = STORAGE_DIR / "bgm"  # 背景音乐目录
PODCASTS_DIR = STORAGE_DIR / "podcasts"  # 播客输出目录

# 确保目录存在
AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
BGM_DIR.mkdir(parents=True, exist_ok=True)
PODCASTS_DIR.mkdir(parents=True, exist_ok=True)

# 检查pydub是否可用
try:
    from pydub import AudioSegment
    from pydub.effects import normalize as normalize_audio
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ 未安装 pydub，音频混音功能将不可用。请安装：pip install pydub")


class ConcatenateAudioInput(BaseModel):
    """音频拼接输入参数"""
    audio_files: List[str] = Field(description="要拼接的音频文件路径列表，按顺序拼接，如['/storage/audios/1.wav', '/storage/audios/2.wav']")
    crossfade_duration: int = Field(default=200, description="交叉淡入淡出时长（毫秒），默认200ms，设为0则无过渡")
    silence_duration: int = Field(default=1200, description="音频片段之间的静音时长（毫秒），默认1200ms，播客对话推荐1000-1500ms")


@tool("concatenate_audio", args_schema=ConcatenateAudioInput)
def concatenate_audio_tool(
    audio_files: List[str],
    crossfade_duration: int = 200,
    silence_duration: int = 1200
) -> str:
    """
    音频拼接工具 - 将多个音频文件按顺序拼接成一个文件
    
    适用于将多个角色的语音片段拼接成完整对话。
    支持交叉淡入淡出和静音间隔，使过渡更自然。
    
    Args:
        audio_files: 音频文件路径列表（按拼接顺序）
        crossfade_duration: 交叉淡入淡出时长（毫秒），0表示无过渡
        silence_duration: 片段间静音时长（毫秒）
    
    Returns:
        拼接后的音频文件路径JSON
    """
    try:
        if not PYDUB_AVAILABLE:
            return "Error: 未安装 pydub 库，请运行: pip install pydub"
        
        if len(audio_files) < 2:
            return "Error: 至少需要2个音频文件才能拼接"
        
        logger.info(f"🔗 开始拼接 {len(audio_files)} 个音频文件")
        
        # 加载所有音频
        audio_segments = []
        for audio_path in audio_files:
            # 转换为绝对路径
            if audio_path.startswith("/storage/"):
                file_path = BASE_DIR / audio_path.lstrip("/")
            else:
                file_path = Path(audio_path)
            
            if not file_path.exists():
                return f"Error: 文件不存在: {audio_path}"
            
            logger.info(f"📁 加载音频: {file_path.name}")
            audio = AudioSegment.from_file(str(file_path))
            audio_segments.append(audio)
        
        # 拼接音频
        combined = audio_segments[0]
        
        for i, audio in enumerate(audio_segments[1:], 1):
            if crossfade_duration > 0:
                # 使用交叉淡入淡出
                combined = combined.append(audio, crossfade=crossfade_duration)
                logger.info(f"✅ 拼接片段 {i}（交叉淡入淡出 {crossfade_duration}ms）")
            else:
                # 添加静音间隔后直接拼接
                if silence_duration > 0:
                    silence = AudioSegment.silent(duration=silence_duration)
                    combined = combined + silence + audio
                    logger.info(f"✅ 拼接片段 {i}（静音间隔 {silence_duration}ms）")
                else:
                    combined = combined + audio
                    logger.info(f"✅ 拼接片段 {i}（无间隔）")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"concatenated_{timestamp}_{unique_id}.wav"
        output_path = AUDIOS_DIR / filename
        
        # 导出
        combined.export(str(output_path), format="wav")
        
        http_path = f"/storage/audios/{filename}"
        
        result = {
            'audio_url': http_path,
            'local_path': http_path,
            'duration_seconds': len(combined) / 1000.0,
            'file_count': len(audio_files),
            'crossfade_duration': crossfade_duration,
            'silence_duration': silence_duration,
            'message': f'成功拼接 {len(audio_files)} 个音频文件'
        }
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 音频拼接完成: {http_path}, 总时长: {len(combined)/1000:.2f}秒")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ 音频拼接失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


class SelectBGMInput(BaseModel):
    """智能选择BGM输入参数"""
    scene_description: str = Field(description="场景描述，如'欢快的开场'、'深沉的讨论'、'轻松的聊天'等，用于匹配合适的背景音乐")
    duration_seconds: Optional[float] = Field(default=None, description="期望的BGM时长（秒），如果指定则会循环或裁剪BGM以匹配时长")


@tool("select_background_music", args_schema=SelectBGMInput)
def select_bgm_tool(
    scene_description: str,
    duration_seconds: Optional[float] = None
) -> str:
    """
    智能BGM选择工具 - 从内置BGM库中选择最合适的背景音乐
    
    根据场景描述（如'欢快的开场'、'深沉的讨论'），智能匹配合适的背景音乐。
    BGM文件名即为场景描述，系统会自动进行语义匹配。
    
    Args:
        scene_description: 场景描述文本
        duration_seconds: 期望的BGM时长（秒），None表示使用原始时长
    
    Returns:
        选中的BGM文件路径JSON
    """
    try:
        if not PYDUB_AVAILABLE:
            return "Error: 未安装 pydub 库，请运行: pip install pydub"
        
        logger.info(f"🎵 开始选择BGM: scene={scene_description}")
        
        # 获取所有BGM文件
        bgm_files = list(BGM_DIR.glob("*.mp3")) + list(BGM_DIR.glob("*.wav"))
        
        if not bgm_files:
            return f"Error: BGM目录为空，请在 {BGM_DIR} 添加背景音乐文件"
        
        logger.info(f"📁 找到 {len(bgm_files)} 个BGM文件")
        
        # 简单的关键词匹配（可以后续升级为语义相似度匹配）
        scene_keywords = scene_description.lower()
        best_match = None
        best_score = 0
        
        for bgm_file in bgm_files:
            # 文件名（不含扩展名）作为描述
            bgm_desc = bgm_file.stem.lower()
            
            # 计算匹配度（简单的关键词匹配）
            score = sum(1 for word in scene_keywords.split() if word in bgm_desc)
            
            if score > best_score:
                best_score = score
                best_match = bgm_file
        
        # 如果没有关键词匹配，随机选择第一个
        if best_match is None:
            best_match = bgm_files[0]
            logger.info(f"⚠️ 未找到匹配的BGM，使用默认: {best_match.name}")
        else:
            logger.info(f"✅ 匹配到BGM: {best_match.name} (匹配度: {best_score})")
        
        # 加载BGM
        bgm = AudioSegment.from_file(str(best_match))
        original_duration = len(bgm) / 1000.0
        
        # 如果指定了时长，调整BGM
        if duration_seconds is not None and duration_seconds > 0:
            target_duration_ms = int(duration_seconds * 1000)
            current_duration_ms = len(bgm)
            
            if current_duration_ms < target_duration_ms:
                # BGM太短，循环播放
                loops_needed = (target_duration_ms // current_duration_ms) + 1
                bgm = bgm * loops_needed
                bgm = bgm[:target_duration_ms]
                logger.info(f"🔄 BGM循环播放，从 {original_duration:.2f}s 延长至 {duration_seconds:.2f}s")
            elif current_duration_ms > target_duration_ms:
                # BGM太长，裁剪
                bgm = bgm[:target_duration_ms]
                logger.info(f"✂️ BGM裁剪，从 {original_duration:.2f}s 缩短至 {duration_seconds:.2f}s")
            
            # 添加淡出效果
            bgm = bgm.fade_out(duration=2000)
        
        # 保存处理后的BGM（如果有调整）
        if duration_seconds is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"bgm_{timestamp}_{unique_id}.mp3"
            output_path = AUDIOS_DIR / filename
            bgm.export(str(output_path), format="mp3")
            http_path = f"/storage/audios/{filename}"
        else:
            http_path = f"/storage/bgm/{best_match.name}"
        
        result = {
            'bgm_path': http_path,
            'bgm_name': best_match.stem,
            'original_duration': original_duration,
            'adjusted_duration': len(bgm) / 1000.0,
            'match_score': best_score,
            'scene_description': scene_description,
            'message': f'已选择BGM: {best_match.stem}'
        }
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ BGM选择完成: {best_match.name}")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ BGM选择失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


class MixAudioWithBGMInput(BaseModel):
    """音频与BGM混音输入参数"""
    voice_audio: str = Field(description="主音频文件路径（人声/对话）")
    bgm_audio: str = Field(description="背景音乐文件路径")
    bgm_volume: float = Field(default=-26, description="BGM背景音量调整（dB），负数表示降低音量，推荐-24到-28之间（5%左右）")
    intro_duration: float = Field(default=3.0, description="BGM开场时长（秒），原声播放后再减弱，默认3秒")
    normalize: bool = Field(default=True, description="是否进行音量归一化，推荐开启")


@tool("mix_audio_with_bgm", args_schema=MixAudioWithBGMInput)
def mix_audio_with_bgm_tool(
    voice_audio: str,
    bgm_audio: str,
    bgm_volume: float = -26,
    intro_duration: float = 3.0,
    normalize: bool = True
) -> str:
    """
    音频BGM混音工具 - 将人声/对话与背景音乐混合
    
    将主音频（人声对话）与背景音乐混合，支持专业的BGM开场效果：
    - BGM先以原声音量播放开场（intro_duration秒）
    - 然后淡入过渡到背景音量（约5%音量），作为对话背景音
    - 自动调整BGM音量，确保人声清晰
    
    Args:
        voice_audio: 主音频文件路径（人声）
        bgm_audio: 背景音乐文件路径
        bgm_volume: BGM背景音量调整（dB），推荐-24到-28（约5%音量）
        intro_duration: BGM开场时长（秒），原声播放后再减弱
        normalize: 是否音量归一化
    
    Returns:
        混音后的音频文件路径JSON
    """
    try:
        if not PYDUB_AVAILABLE:
            return "Error: 未安装 pydub 库，请运行: pip install pydub"
        
        logger.info(f"🎧 开始混音: voice={voice_audio}, bgm={bgm_audio}")
        
        # 加载主音频
        if voice_audio.startswith("/storage/"):
            voice_path = BASE_DIR / voice_audio.lstrip("/")
        else:
            voice_path = Path(voice_audio)
        
        if not voice_path.exists():
            return f"Error: 主音频文件不存在: {voice_audio}"
        
        voice = AudioSegment.from_file(str(voice_path))
        voice_duration = len(voice) / 1000.0
        logger.info(f"📁 加载主音频: {voice_path.name}, 时长: {voice_duration:.2f}s")
        
        # 加载BGM
        if bgm_audio.startswith("/storage/"):
            bgm_path = BASE_DIR / bgm_audio.lstrip("/")
        else:
            bgm_path = Path(bgm_audio)
        
        if not bgm_path.exists():
            return f"Error: BGM文件不存在: {bgm_audio}"
        
        bgm = AudioSegment.from_file(str(bgm_path))
        bgm_duration = len(bgm) / 1000.0
        logger.info(f"📁 加载BGM: {bgm_path.name}, 时长: {bgm_duration:.2f}s")
        
        # 计算总时长：BGM开场 + 主音频
        intro_duration_ms = int(intro_duration * 1000)
        total_duration_ms = intro_duration_ms + len(voice)
        
        # 调整BGM时长以匹配总时长
        if len(bgm) < total_duration_ms:
            # BGM太短，循环播放
            loops_needed = (total_duration_ms // len(bgm)) + 1
            bgm = bgm * loops_needed
            bgm = bgm[:total_duration_ms]
            logger.info(f"🔄 BGM循环播放以匹配总时长")
        elif len(bgm) > total_duration_ms:
            # BGM太长，裁剪
            bgm = bgm[:total_duration_ms]
            logger.info(f"✂️ BGM裁剪以匹配总时长")
        
        # BGM音量处理：
        # - 前 intro_duration 秒：原音量
        # - intro_duration 到 intro_duration + fade_duration：从原音量过渡到背景音量
        # - 之后：背景音量
        
        fade_duration = 2000  # 2秒过渡时间
        
        # 分离BGM的三个部分
        bgm_intro = bgm[:intro_duration_ms]  # 开场部分（原音量）
        
        if intro_duration_ms + fade_duration < len(bgm):
            bgm_fade = bgm[intro_duration_ms:intro_duration_ms + fade_duration]  # 过渡部分
            bgm_background = bgm[intro_duration_ms + fade_duration:]  # 背景部分
            
            # 过渡部分：从原音量渐变到背景音量
            # 创建音量渐变效果
            fade_steps = 20  # 分20步渐变
            step_duration = fade_duration // fade_steps
            bgm_fade_processed = AudioSegment.empty()
            
            for i in range(fade_steps):
                # 计算当前步骤的音量调整
                progress = i / fade_steps  # 0 到 1
                current_volume_adjustment = bgm_volume * progress  # 从0逐渐到bgm_volume
                
                # 提取当前步骤的音频片段
                start = i * step_duration
                end = min((i + 1) * step_duration, len(bgm_fade))
                segment = bgm_fade[start:end]
                
                # 调整音量
                segment = segment + current_volume_adjustment
                bgm_fade_processed += segment
            
            # 背景部分：固定背景音量
            bgm_background_processed = bgm_background + bgm_volume
            
            # 拼接所有部分
            bgm_processed = bgm_intro + bgm_fade_processed + bgm_background_processed
            logger.info(f"🎬 BGM效果：前{intro_duration:.1f}s原音量 → {fade_duration/1000:.1f}s过渡 → 背景音量{bgm_volume}dB")
        else:
            # intro_duration太长，直接处理
            bgm_processed = bgm[:intro_duration_ms] + (bgm[intro_duration_ms:] + bgm_volume)
            logger.info(f"⚠️ BGM时长较短，简化处理")
        
        # 在BGM结尾添加淡出效果
        bgm_processed = bgm_processed.fade_out(duration=3000)
        
        # 人声处理：在开场后开始
        # 创建静音片段用于开场
        silence_intro = AudioSegment.silent(duration=intro_duration_ms)
        voice_with_intro = silence_intro + voice
        
        # 确保人声和BGM长度一致
        if len(voice_with_intro) > len(bgm_processed):
            voice_with_intro = voice_with_intro[:len(bgm_processed)]
        elif len(voice_with_intro) < len(bgm_processed):
            # 人声部分较短，BGM裁剪
            bgm_processed = bgm_processed[:len(voice_with_intro)]
        
        # 混音：叠加人声
        mixed = bgm_processed.overlay(voice_with_intro)
        logger.info(f"🎵 混音完成（包含{intro_duration:.1f}秒BGM开场）")
        
        # 音量归一化
        if normalize:
            mixed = normalize_audio(mixed)
            logger.info(f"📊 音量归一化完成")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"podcast_{timestamp}_{unique_id}.mp3"
        output_path = PODCASTS_DIR / filename
        
        # 导出（MP3格式，适合播客）
        mixed.export(str(output_path), format="mp3", bitrate="192k")
        
        http_path = f"/storage/podcasts/{filename}"
        
        result = {
            'audio_url': http_path,
            'local_path': http_path,
            'duration_seconds': len(mixed) / 1000.0,
            'voice_audio': voice_audio,
            'bgm_audio': bgm_audio,
            'bgm_volume': bgm_volume,
            'normalized': normalize,
            'message': '混音完成，播客已生成'
        }
        
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(f"✅ 混音完成: {http_path}, 时长: {len(mixed)/1000:.2f}s")
        return result_json
        
    except Exception as e:
        logger.error(f"❌ 混音失败: {str(e)}")
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
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试音频拼接
    # print("\n测试音频拼接...")
    # result = concatenate_audio_tool.invoke({
    #     "audio_files": [
    #         "/storage/audios/audio1.wav",
    #         "/storage/audios/audio2.wav"
    #     ],
    #     "crossfade_duration": 200,
    #     "silence_duration": 500
    # })
    # print("结果:", result)
    
    # 测试BGM选择
    # print("\n测试BGM选择...")
    # result = select_bgm_tool.invoke({
    #     "scene_description": "欢快的开场",
    #     "duration_seconds": 60
    # })
    # print("结果:", result)
    
    # 测试混音
    # print("\n测试混音...")
    # result = mix_audio_with_bgm_tool.invoke({
    #     "voice_audio": "/storage/audios/voice.wav",
    #     "bgm_audio": "/storage/bgm/happy.mp3",
    #     "bgm_volume": -20
    # })
    # print("结果:", result)
