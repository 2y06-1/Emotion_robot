#!/bin/bash

# 设置 pip 镜像源
pip config set global.extra-index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

# 安装 spacemit-ort, onnxruntime, numpy, opencv-python
pip install --break-system-packages spacemit-ort
pip install onnxruntime
pip install numpy
pip install opencv-python

# 展示包版本
pip list | grep -E "onnxruntime|spacemit|numpy|opencv"

# 测试 spacemit 加速功能
python3 - << EOF
import onnxruntime as ort
import spacemit_ort
print(f'ONNX Runtime: {ort.__version__}')
print(f'可用执行提供程序: {ort.get_available_providers()}')
EOF