#!/bin/bash

# 更新系统包列表

sudo apt update

# 安装 spacemit-ollama-toolkit

sudo apt install -y spacemit-ollama-toolkit

# 列出已安装的 Ollama 模型

ollama list

# 显示 spacemit-ollama-toolkit 软件包信息

sudo apt show spacemit-ollama-toolkit
