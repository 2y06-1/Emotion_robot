项目结构：

Emotion_robot/
├── config/ # 配置文件
├── csrc/ # 外设编写
│ ├── drives/ # 各种外设驱动
│ └── include/ # 头文件
├── CMakeLists.txt # CMake 构建脚本
├── lib/ # 编译文件
├── model/ # 模型文件
│ ├── asr/ # 语音识别模型
│ ├── llm/ # 大语言模型
│ └── vision/ # 视觉模型
├── scripts/ # 安装脚本
├── src/ # Python 源代码
│ ├── asr/ 
│ ├── bert-base-multilingual-uncased/ 
│ ├── bert-base-uncased/ 
│ ├── llm/ 
│ ├── vision/ 
│ ├── llm_or_asr.py 
│ └── main.py 
├── venv/ # Python 虚拟环境
├── wav/ # 音频文件
│ ├── init.wav
│ └── zh.mp3
├── .gitignore
├── README.md 
└── requirement.txt
└── requirements.txt #python 安装依赖