import json
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

HISTORY_FILE = "storage/chat_history.json"

class Canvas(BaseModel):
    id: str
    name: str
    createdAt: float
    # Legacy: old DOM-drag canvas images
    images: Optional[List[Dict[str, Any]]] = []
    # New: Excalidraw canvas data (elements/appState/files)
    data: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]]

class HistoryService:
    def __init__(self):
        self.file_path = HISTORY_FILE
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            self._save_data([])
        else:
            # 检查文件是否为空或格式错误，如果是则重置
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning("历史记录文件为空，重置为空列表")
                        self._save_data([])
                    else:
                        # 尝试解析，如果失败会在 _load_data 中处理
                        json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"初始化时发现历史记录文件格式错误: {e}，重置为空列表")
                self._save_data([])

    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(self.file_path):
                return []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 处理空文件或格式错误
                if not content:
                    logger.warning(f"历史记录文件为空，返回空列表")
                    return []
                return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"历史记录文件格式错误: {e}，尝试修复...")
            # 如果文件损坏，备份并重置
            try:
                backup_path = self.file_path + '.backup'
                if os.path.exists(self.file_path):
                    import shutil
                    shutil.copy2(self.file_path, backup_path)
                    logger.info(f"已备份损坏文件到: {backup_path}")
                # 重置为空列表
                self._save_data([])
                return []
            except Exception as backup_error:
                logger.error(f"备份文件失败: {backup_error}")
                return []
        except Exception as e:
            logger.error(f"加载历史记录失败: {e}")
            return []

    def _save_data(self, data: List[Dict[str, Any]]):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")

    def get_canvases(self) -> List[Dict[str, Any]]:
        return self._load_data()

    def save_canvas(self, canvas_data: Dict[str, Any]):
        canvases = self._load_data()
        # 查找是否存在
        index = -1
        for i, c in enumerate(canvases):
            if c.get('id') == canvas_data.get('id'):
                index = i
                break
        
        if index >= 0:
            canvases[index] = canvas_data
        else:
            canvases.insert(0, canvas_data) # 新的在前面
            
        self._save_data(canvases)
        return canvas_data

    def delete_canvas(self, canvas_id: str):
        canvases = self._load_data()
        canvases = [c for c in canvases if c.get('id') != canvas_id]
        self._save_data(canvases)
        return True

history_service = HistoryService()






