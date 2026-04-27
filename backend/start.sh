#!/bin/bash
# 启动后端服务

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentImage

# 进入后端目录
cd "$(dirname "$0")"

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，使用默认配置"
fi

# 获取当前环境的 Python 路径
PYTHON_EXEC=$(which python)
echo "使用 Python: $PYTHON_EXEC"

# 使用 python -m uvicorn 确保使用正确的 Python 环境
# 并且设置环境变量确保子进程也使用正确的 Python
export PYTHON_EXECUTABLE="$PYTHON_EXEC"
$PYTHON_EXEC -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
