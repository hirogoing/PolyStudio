"""
Agent服务 - 处理LangGraph Agent的流式输出
"""
import json
import os
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from langgraph.prebuilt import create_react_agent
from app.services.stream_processor import StreamProcessor
from app.services.prompt import get_full_prompt
from app.tools.volcano_image_generation import generate_volcano_image_tool, edit_volcano_image_tool
from app.tools.model_3d_generation import generate_3d_model_tool
from app.tools.volcano_video_generation import generate_volcano_video_tool
from app.tools.video_concatenation import concatenate_videos_tool
from app.tools.virtual_anchor_generation import detect_face_tool, generate_virtual_anchor_tool
from app.tools.qwen_tts import qwen_voice_design_tool, qwen_voice_cloning_tool
from app.tools.audio_mixing import concatenate_audio_tool, select_bgm_tool, mix_audio_with_bgm_tool
from app.llm.factory import create_llm

# 使用统一的日志配置
logger = logging.getLogger(__name__)


def create_agent():
    """创建LangGraph Agent"""
    # 使用 LLM 工厂创建模型实例（默认使用火山引擎）
    model = create_llm()

    # 创建工具列表
    tools = [
        # generate_image_tool,
        # edit_image_tool,
        generate_volcano_image_tool,
        edit_volcano_image_tool,
        generate_3d_model_tool,
        generate_volcano_video_tool,
        concatenate_videos_tool,
        detect_face_tool,
        generate_virtual_anchor_tool,
        # Qwen-TTS工具
        qwen_voice_design_tool,
        qwen_voice_cloning_tool,
        # 音频混音工具
        concatenate_audio_tool,
        select_bgm_tool,
        mix_audio_with_bgm_tool,
    ]
    logger.info(f"🛠️  注册工具: {[tool.name for tool in tools]}")

    # 动态生成工具列表描述
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    tools_list_text = "\n".join(tool_descriptions)

    # 使用prompt模块生成完整提示词
    full_prompt = get_full_prompt(tools_list_text)

    # 创建Agent
    agent = create_react_agent(
        name="polystudio_multimodal_agent",
        model=model,
        tools=tools,
        prompt=full_prompt
    )
    
    logger.info("✅ Agent创建成功")
    return agent


async def process_chat_stream(
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    处理聊天流式响应
    
    Args:
        messages: 消息历史
        session_id: 会话ID
    
    Yields:
        SSE格式的事件流
    """
    try:
        logger.info(f"💬 收到聊天请求: session_id={session_id}, messages_count={len(messages)}")
        
        # 创建Agent（在 langgraph 1.0.0 中，create_react_agent 返回的对象已经是编译后的）
        agent = create_agent()

        # 创建流处理器
        processor = StreamProcessor(session_id)

        # 处理流式响应
        async for event in processor.process_stream(agent, messages):
            try:
                yield event
            except (GeneratorExit, StopAsyncIteration, ConnectionError, BrokenPipeError, OSError) as e:
                # 客户端断开连接
                logger.info(f"⚠️ 客户端断开连接: {type(e).__name__}: {str(e)}")
                raise  # 重新抛出，让上层处理
            except Exception as e:
                # 其他异常，记录并继续
                logger.warning(f"⚠️ 发送事件时出错: {type(e).__name__}: {str(e)}")
                raise

    except (GeneratorExit, StopAsyncIteration, ConnectionError, BrokenPipeError, OSError) as e:
        # 客户端断开连接，这是正常情况，不需要记录为错误
        logger.info(f"ℹ️ 客户端断开连接，停止流式响应: {type(e).__name__}")
        # 不发送错误事件，因为客户端已经断开
        return
    except Exception as e:
        import traceback
        logger.error(f"❌ 处理聊天流时出错: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            # 尝试发送错误事件（如果客户端还在）
            error_event = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except:
            # 如果发送失败（客户端已断开），忽略
            pass

