#!/bin/bash

# 使用 sudo 确保所有操作拥有权限
sudo bash -c "

# 安装必要的 Python 包
pip install -i https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple llm_asr --break-system-packages

# 安装系统依赖
apt install -y libopenblas-dev

# 创建目标目录
mkdir -p /usr/local/share/voice_test
cd /usr/local/share/voice_test

# 检查并下载音频文件
if [ ! -f zh.mp3 ]; then
    echo 'Downloading zh.mp3...'
    wget -O zh.mp3 https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/example/zh.mp3
else
    echo 'zh.mp3 already exists, skipping download.'
fi

# 修改音频文件权限
chmod 644 /usr/local/share/voice_test/zh.mp3
chown $(whoami):$(whoami) /usr/local/share/voice_test/zh.mp3

# 检查并下载模型文件
if [ ! -f sensevoice.tar.gz ]; then
    echo 'Downloading sensevoice.tar.gz...'
    wget -O sensevoice.tar.gz https://archive.spacemit.com/spacemit-ai/ModelZoo/asr/sensevoice.tar.gz
else
    echo 'sensevoice.tar.gz already exists, skipping download.'
fi

# 解压模型文件
if [ ! -d sensevoice ]; then
    echo 'Extracting sensevoice.tar.gz...'
    tar zxvf sensevoice.tar.gz
else
    echo 'sensevoice directory already exists, skipping extraction.'
fi
"

