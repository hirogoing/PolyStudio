"""
聊天路由 - 处理对话请求
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from app.services.agent_service import process_chat_stream
from app.services.history_service import history_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = None


@router.get("/canvases")
async def get_canvases():
    """获取所有画布历史"""
    return history_service.get_canvases()


@router.post("/canvases")
async def save_canvas(request: Request):
    """保存或更新画布（项目）

    注意：前端会携带 Excalidraw 的 data(elements/appState/files)，
    用 Pydantic 模型解析容易因 extra 字段处理/热重载不同步而丢字段。
    这里直接存原始 JSON，避免 data 被过滤导致刷新后画布空白。
    """
    payload = await request.json()
    return history_service.save_canvas(payload)


@router.delete("/canvases/{canvas_id}")
async def delete_canvas(canvas_id: str):
    """删除画布"""
    history_service.delete_canvas(canvas_id)
    return {"success": True}


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片到 storage/images 目录
    
    Returns:
        上传后的图片URL（相对路径，如 /storage/images/xxx.jpg）
    """
    try:
        # 确保上传目录存在
        BASE_DIR = Path(__file__).parent.parent.parent
        IMAGES_DIR = BASE_DIR / "storage" / "images"
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 验证文件类型
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # 获取文件扩展名
        original_filename = file.filename or "image"
        ext = os.path.splitext(original_filename)[1] or ".jpg"
        if not ext.startswith("."):
            ext = ".jpg"
        
        filename = f"upload_{timestamp}_{unique_id}{ext}"
        file_path = IMAGES_DIR / filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 返回相对路径
        image_url = f"/storage/images/{filename}"
        return {"url": image_url, "filename": filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求，返回流式响应
    支持OpenAI格式的流式输出
    """
    try:
        # 构建消息历史
        messages = request.messages.copy() if request.messages else []
        messages.append({
            "role": "user",
            "content": request.message
        })

        # 返回流式响应 - 确保立即发送，不缓冲
        return StreamingResponse(
            process_chat_stream(messages, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

