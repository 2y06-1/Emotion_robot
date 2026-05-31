import sys
from pathlib import Path


class Voice_Transform:
    """
    ASR 语音转文字模块。

    接口原则：
    - 本文件不读取 config.json。
    - 本文件不写死 ASR 工程路径。
    - project_root 由 main.py 从 config.py 读取后传入。
    """

    def __init__(self, project_root):
        self.project_root = Path(project_root)

        if str(self.project_root) not in sys.path:
            sys.path.append(str(self.project_root))

        self.model = None
        self.load_model()

    def load_model(self):
        try:
            from asr import AsrModel

            self.model = AsrModel()
            print("load success")
        except Exception as e:
            self.model = None
            print(f"load failed: {e}")

    def speech_to_text(self, wav_path):
        if self.model is None:
            raise RuntimeError("ASR 模型未加载成功")

        return self.model(wav_path)
