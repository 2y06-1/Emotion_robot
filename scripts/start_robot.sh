#!/bin/bash

sleep 8

mkdir -p /home/bianbu/Emotion_robot/logs

echo "==============================" >> /home/bianbu/Emotion_robot/logs/robot.log
echo "$(date) Emotion Robot starting..." >> /home/bianbu/Emotion_robot/logs/robot.log

# 等待 USB 麦克风/声卡初始化
sleep 2

# 自动调节 USB 声卡音量
# -c 2 对应你现在的 USB Mixer 声卡
echo "$(date) Setting audio volume..." >> /home/bianbu/Emotion_robot/logs/robot.log

amixer -c 2 sset 'Mic' 90% unmute >> /home/bianbu/Emotion_robot/logs/robot.log 2>&1 || true
amixer -c 2 sset 'PCM' 90% unmute >> /home/bianbu/Emotion_robot/logs/robot.log 2>&1 || true
amixer -c 2 sset 'Auto Gain Control' on >> /home/bianbu/Emotion_robot/logs/robot.log 2>&1 || true

echo "$(date) Audio volume setting finished." >> /home/bianbu/Emotion_robot/logs/robot.log

cd /home/bianbu/Emotion_robot/src || exit 1

export PYTHONUNBUFFERED=1

if pgrep -f "/home/bianbu/Emotion_robot/src/main.py" > /dev/null; then
    echo "$(date) main.py already running" >> /home/bianbu/Emotion_robot/logs/robot.log
    exit 0
fi

if [ -x /home/bianbu/Emotion_robot/venv/bin/python ]; then
    /home/bianbu/Emotion_robot/venv/bin/python /home/bianbu/Emotion_robot/src/main.py >> /home/bianbu/Emotion_robot/logs/robot.log 2>&1
else
    /usr/bin/python3 /home/bianbu/Emotion_robot/src/main.py >> /home/bianbu/Emotion_robot/logs/robot.log 2>&1
fi