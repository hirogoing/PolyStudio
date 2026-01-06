"""
流式处理器 - 处理LangGraph的流式输出并转换为SSE格式
"""
import json
import logging
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
from langchain_core.messages import (
    AIMessageChunk, 
    ToolMessage, 
    HumanMessage,
    AIMessage,
    convert_to_openai_messages,
    ToolCall
)

# 配置日志
logger = logging.getLogger(__name__)


class StreamProcessor:
    """流式处理器 - 负责处理智能体的流式输出"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.current_content = ""
        self.current_tool_calls = []
        self.text_buffer = "" # 用于累积日志打印的文本缓冲区
        self.tool_call_args: Dict[str, Dict[str, Any]] = {}  # 用于累积工具调用参数
        # LangGraph 默认 recursion_limit=25，生成多张图会很容易超过这个步数导致报错
        self.recursion_limit = int(os.getenv("RECURSION_LIMIT", "200"))

    async def process_stream(
        self,
        agent: Any,
        messages: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        处理整个流式响应
        
        Args:
            agent: 编译后的LangGraph Agent
            messages: 消息列表
        
        Yields:
            SSE格式的事件字符串
        """
        try:
            logger.info(f"🚀 开始处理流式响应，消息数量: {len(messages)}")
            
            # 转换消息格式为LangChain格式
            langchain_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    langchain_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_messages.append(AIMessage(content=msg.get("content", "")))
            
            logger.info(f"📨 转换后的消息数量: {len(langchain_messages)}")
            
            # 开始流式处理 - 使用 messages 模式确保逐字符流式输出
            # 关键：每个 chunk 立即 yield，不等待，类似 OpenAI 的流式输出
            chunk_count = 0
            async for chunk in agent.astream(
                {"messages": langchain_messages},
                {"recursion_limit": self.recursion_limit},
                stream_mode=["messages"]  # 使用列表格式，确保流式输出
            ):
                chunk_count += 1
                logger.debug(f"📦 收到第 {chunk_count} 个 chunk: {type(chunk)}")
                # 立即处理并发送，不等待 - 确保真正的流式输出
                event_count = 0
                async for event in self._handle_chunk(chunk):
                    event_count += 1
                    logger.debug(f"📤 发送第 {event_count} 个事件 (chunk {chunk_count}): {event[:100] if len(event) > 100 else event}")
                    # 立即 yield，确保流式输出，不缓冲
                    yield event
                logger.debug(f"✅ Chunk {chunk_count} 处理完成，发送了 {event_count} 个事件")

            # 发送完成事件
            # 打印剩余的文本缓冲区
            if self.text_buffer:
                logger.info(f"🤖 AI回答(完): {self.text_buffer}")
                self.text_buffer = ""
            
            logger.info("✅ 流式处理完成")
            yield "data: [DONE]\n\n"

        except Exception as e:
            import traceback
            logger.error(f"❌ 流式处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            error_event = {
                "type": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    async def _handle_chunk(self, chunk: Any) -> AsyncGenerator[str, None]:
        """处理单个chunk"""
        try:
            logger.debug(f"🔍 处理 chunk: type={type(chunk)}, value={str(chunk)[:200]}")
            
            # langgraph 1.0.0 的流式输出格式可能是多种形式
            # 1. tuple 格式: (chunk_type, chunk_data)
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type = chunk[0]
                chunk_data = chunk[1]
                logger.debug(f"  📦 Tuple chunk: type={chunk_type}, data_type={type(chunk_data)}")
                
                if chunk_type == "values":
                    # 处理完整状态更新
                    async for event in self._handle_values_chunk(chunk_data):
                        yield event
                else:
                    # 处理消息流
                    if isinstance(chunk_data, list) and len(chunk_data) > 0:
                        logger.debug(f"  📋 消息列表，长度: {len(chunk_data)}")
                        for message in chunk_data:
                            async for event in self._handle_message_chunk(message):
                                yield event
                    elif hasattr(chunk_data, '__iter__') and not isinstance(chunk_data, str):
                        # 可迭代对象
                        logger.debug(f"  🔄 可迭代对象")
                        for message in chunk_data:
                            async for event in self._handle_message_chunk(message):
                                yield event
                    else:
                        # 单个消息对象
                        logger.debug(f"  📨 单个消息对象")
                        async for event in self._handle_message_chunk(chunk_data):
                            yield event
            # 2. 列表格式: [message1, message2, ...]
            elif isinstance(chunk, list) and len(chunk) > 0:
                logger.debug(f"  📋 直接列表格式，长度: {len(chunk)}")
                for message in chunk:
                    async for event in self._handle_message_chunk(message):
                        yield event
            # 3. 直接是消息对象
            else:
                logger.debug(f"  📨 直接消息对象")
                async for event in self._handle_message_chunk(chunk):
                    yield event
        except Exception as e:
            import traceback
            logger.error(f"❌ 处理 chunk 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            error_event = {
                "type": "error",
                "error": f"处理chunk时出错: {str(e)}"
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    async def _handle_values_chunk(self, chunk_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """处理values类型的chunk - 包含完整消息状态"""
        all_messages = chunk_data.get("messages", [])
        
        if all_messages:
            # 转换为OpenAI格式
            oai_messages = convert_to_openai_messages(all_messages)
            
            # 发送完整消息更新
            event = {
                "type": "messages",
                "messages": oai_messages
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async def _handle_message_chunk(self, message_chunk: Any) -> AsyncGenerator[str, None]:
        """处理消息类型的chunk"""
        try:
            # 处理工具消息
            if isinstance(message_chunk, ToolMessage):
                logger.info(f"🔧 工具调用结果: tool_call_id={message_chunk.tool_call_id}")
                logger.info(f"   内容: {str(message_chunk.content)[:200]}")
                # 清理已完成的工具调用参数
                if message_chunk.tool_call_id in self.tool_call_args:
                    del self.tool_call_args[message_chunk.tool_call_id]
                event = {
                    "type": "tool_result",
                    "tool_call_id": message_chunk.tool_call_id,
                    "content": message_chunk.content
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                return

            # 处理AI消息
            if isinstance(message_chunk, AIMessageChunk):
                logger.debug(f"  🤖 AIMessageChunk: content={str(message_chunk.content)[:50] if message_chunk.content else None}")
                # 处理文本内容 - 立即发送每个 chunk，类似 OpenAI 流式输出
                # 关键：langgraph 的 AIMessageChunk 已经是增量内容，直接发送
                content = message_chunk.content
                
                # 如果 content 存在，立即发送（每个 chunk 都是增量）
                if content is not None and content != "":
                    content_str = str(content) if not isinstance(content, str) else content
                    
                    # 直接发送这个 chunk 的内容（langgraph 已经处理了增量）
                    # 类似 OpenAI: chunk.choices[0].delta.content
                    if content_str:
                        logger.debug(f"📝 发送文本 delta ({len(content_str)} 字符): {content_str[:100]}")
                        
                        # 累积到缓冲区用于日志打印
                        self.text_buffer += content_str
                        # 如果遇到换行符或标点符号，且长度足够，则打印
                        if "\n" in self.text_buffer or (len(self.text_buffer) > 50 and any(p in self.text_buffer for p in "。！？.!?")):
                            # 移除换行符，保持日志整洁
                            log_content = self.text_buffer.replace("\n", " ")
                            if log_content.strip():
                                logger.info(f"🤖 AI回答: {log_content}")
                            self.text_buffer = ""
                            
                        event = {
                            "type": "delta",
                            "content": content_str
                        }
                        # 立即 yield，不等待
                        event_str = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        logger.debug(f"📤 发送事件字符串: {event_str[:100]}")
                        yield event_str
                else:
                    logger.debug(f"  ⚠️  AIMessageChunk 没有内容")

                # 处理工具调用
                if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                    for tool_call in message_chunk.tool_calls:
                        # 处理不同的工具调用格式
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("args", {})
                        else:
                            # ToolCall 对象
                            tool_call_id = getattr(tool_call, "id", None)
                            tool_name = getattr(tool_call, "name", None)
                            tool_args = getattr(tool_call, "args", {})
                        
                        # 关键修复：严格检查 name 是否存在且非空
                        # 在流式输出中，某些 chunk 可能包含 name 为空或 None 的 tool_call
                        if not tool_name or not tool_call_id:
                            logger.debug(f"⚠️  跳过无效的工具调用 (name或id为空): name={tool_name}, id={tool_call_id}")
                            continue

                        # 累积工具调用参数（流式输出中参数可能分多个chunk）
                        if tool_call_id not in self.tool_call_args:
                            self.tool_call_args[tool_call_id] = {}
                        
                        # 合并参数（后续chunk可能包含更多参数）
                        if tool_args:
                            self.tool_call_args[tool_call_id].update(tool_args)
                        
                        # 使用累积的参数
                        final_args = self.tool_call_args[tool_call_id]

                        logger.info(f"🛠️  工具调用: name={tool_name}, id={tool_call_id}")
                        logger.info(f"   参数: {final_args}")
                        
                        event = {
                            "type": "tool_call",
                            "id": tool_call_id,
                            "name": tool_name,
                            "arguments": final_args
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                # 处理工具调用参数流（如果存在）
                if hasattr(message_chunk, "tool_call_chunks") and message_chunk.tool_call_chunks:
                    for tool_call_chunk in message_chunk.tool_call_chunks:
                        logger.debug(f"📝 工具调用参数流: {tool_call_chunk}")
                        # 将参数流发送给前端
                        # tool_call_chunk 通常包含: {index: 0, id: '...', args: '...'}
                        
                        # 处理可能的字典或对象
                        chunk_dict = tool_call_chunk
                        if not isinstance(chunk_dict, dict):
                            # 尝试转为字典
                            if hasattr(tool_call_chunk, "dict"):
                                chunk_dict = tool_call_chunk.dict()
                            else:
                                chunk_dict = {"args": str(tool_call_chunk)} # fallback

                        # 提取 args
                        args_chunk = chunk_dict.get("args")
                        index = chunk_dict.get("index", 0)
                        tc_id = chunk_dict.get("id")
                        
                        if args_chunk:
                            event = {
                                "type": "tool_call_chunk",
                                "index": index,
                                "id": tc_id,
                                "args": args_chunk
                            }
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"❌ 处理消息chunk时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            error_event = {
                "type": "error",
                "error": f"处理消息chunk时出错: {str(e)}"
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
