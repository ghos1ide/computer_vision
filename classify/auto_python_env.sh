#!/bin/bash
# auto_python_env.sh
# 用法: bash auto_python_env.sh 3.10 myenv

PY_VERSION=$1
VENV_NAME=$2

# 检查 pyenv 是否安装
if ! command -v pyenv &> /dev/null; then
    echo "未检测到 pyenv，正在安装 pyenv..."
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# 安装指定 Python 版本
if ! pyenv versions | grep -q "$PY_VERSION"; then
    echo "正在安装 Python $PY_VERSION ..."
    pyenv install $PY_VERSION
fi

# 设置全局/本地 Python 版本
pyenv local $PY_VERSION

# 检查虚拟环境
if [ ! -d ".venv_$VENV_NAME" ]; then
    echo "创建虚拟环境 .venv_$VENV_NAME ..."
    python -m venv .venv_$VENV_NAME
fi

# 激活虚拟环境
source .venv_$VENV_NAME/bin/activate
echo "已切换到 Python $PY_VERSION，虚拟环境 .venv_$VENV_NAME 已激活"
python --version
which python