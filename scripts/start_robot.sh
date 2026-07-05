#!/bin/bash

# 等待桌面、USB设备、音频设备初始化完成
sleep 10

# 日志文件
mkdir -p /home/bianbu/Emotion_robot/logs
LOG=/home/bianbu/Emotion_robot/logs/robot.log

echo "==============================" >> "$LOG"
echo "$(date) Emotion Robot starting..." >> "$LOG"

# 基础环境变量
export HOME=/home/bianbu
export USER=bianbu
export LOGNAME=bianbu
export PYTHONUNBUFFERED=1
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 关键：让 ASR 自启时能找到 libortextensions.so.0
export LD_LIBRARY_PATH=/home/bianbu/Emotion_robot/lib:${LD_LIBRARY_PATH}

# Python 和系统命令路径
export PATH=/home/bianbu/Emotion_robot/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin

echo "$(date) LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$LOG"

# 激活虚拟环境
if [ -f /home/bianbu/Emotion_robot/venv/bin/activate ]; then
    source /home/bianbu/Emotion_robot/venv/bin/activate
    echo "$(date) venv activated." >> "$LOG"
else
    echo "$(date) WARNING: venv activate not found." >> "$LOG"
fi

# 等待 USB 麦克风/声卡
sleep 2

echo "$(date) Setting audio volume..." >> "$LOG"

# 自动调节 USB 声卡音量，-c 2 对应你的 USB Mixer 声卡
amixer -c 2 sset 'Mic' 90% unmute >> "$LOG" 2>&1 || true
amixer -c 2 sset 'PCM' 90% unmute >> "$LOG" 2>&1 || true
amixer -c 2 sset 'Auto Gain Control' on >> "$LOG" 2>&1 || true

echo "$(date) Audio volume setting finished." >> "$LOG"

# 进入项目目录
cd /home/bianbu/Emotion_robot/src || {
    echo "$(date) cd /home/bianbu/Emotion_robot/src failed" >> "$LOG"
    exit 1
}

echo "$(date) PWD=$(pwd)" >> "$LOG"
echo "$(date) python=$(which python)" >> "$LOG"
echo "$(date) python version=$(python --version 2>&1)" >> "$LOG"

# 防止 main.py 重复启动
if pgrep -f "/home/bianbu/Emotion_robot/src/main.py" > /dev/null; then
    echo "$(date) main.py already running" >> "$LOG"
    exit 0
fi

# 启动主程序
python /home/bianbu/Emotion_robot/src/main.py >> "$LOG" 2>&1