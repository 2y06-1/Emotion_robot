import os
import sys
import subprocess

NLTK_DATA_PATH = "/home/bianbu/asr-llm-tts/asr-llm-tts/nltk_data"
os.environ["NLTK_DATA"] = NLTK_DATA_PATH

TAGGER_ZIP = os.path.join(NLTK_DATA_PATH, "taggers", "averaged_perceptron_tagger.zip")
if not os.path.exists(TAGGER_ZIP):
    sys.exit(1)

import nltk
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

sys.path.append("/home/bianbu/asr-llm-tts/asr-llm-tts/src")
from spacemit_tts import TTSModel

class Text_Tranform:
    def __init__(self):
        print("加载 TTS 模型")
        self.model = TTSModel()
        print("TTS 模型加载成功。")

    def text_to_speech(self, text: str):
        if not text.strip():
            return

        try:
            print(" 正在生成语音")
            audio_path = self.model.ort_predict(text)

            if not audio_path or not os.path.exists(audio_path):
                return

            print(" 正在播放到 USB 声卡.")
            subprocess.run(
                ["aplay", "-D", "plughw:2,0", audio_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            print("播放完成")

        except subprocess.CalledProcessError as e:
            print(f" 播放失败: {e.stderr.decode()}")
        except Exception as e:
            print(f" 异常: {e}")

def main():
    tts = Text_Tranform()
    while True:
        text = input("请输入文本: ")
        if text.strip().lower() == 'q':
            print(" 已退出。")
            break
        tts.text_to_speech(text)

if __name__ == "__main__":
    main()