"""
PolyStudio后端主程序
使用FastAPI + LangGraph实现
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.routers import chat
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PolyStudio API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保存储目录存在
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 配置静态文件服务 - 用于访问保存的图片
# 这样前端可以通过 /storage/images/文件名 访问图片
if STORAGE_DIR.exists():
    app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# 注册路由
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    return {"message": "PolyStudio API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os
    import sys
    # 确保使用当前 Python 解释器
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=["*.py"],
        log_level="info"
    )

