import os
import time
import subprocess
import sherpa_onnx
import tempfile
import soundfile as sf
import numpy as np

class Text_Tranform:
    def __init__(self):
        print("加载 TTS 模型 (优化版)...")
        total_start = time.time()

        self.model_dir = "/home/bianbu/Emotion_robot/model/asr"
        # 使用优化后的模型
        self.model_path = f"{self.model_dir}/model_optimized.onnx"
        self.tokens_path = f"{self.model_dir}/tokens.txt"
        self.lexicon_path = f"{self.model_dir}/lexicon.txt"

        self.aplay_device = os.getenv("APLAY_DEVICE", "plughw:0,0")
        self._check_files()

        # 配置TTS模型 - 使用CPU并启用所有优化
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=self.model_path,
                    tokens=self.tokens_path,
                    lexicon=self.lexicon_path,
                ),
                provider="cpu",
                num_threads=4,  # 尝试4线程（可能更适合你的硬件）
                debug=False,
            ),
            # 正确设置规则文件参数（使用空字符串表示不使用）
            rule_fsts="",  # 使用空字符串而不是None
            rule_fars="",  # 添加这个参数
            max_num_sentences=1,  # 限制最大句子数
            silence_scale=0.2  # 添加静音比例参数
        )

        # 加载模型
        model_load_start = time.time()
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        model_load_end = time.time()
        print(f"[计时] 模型加载: {model_load_end - model_load_start:.3f}秒")

        total_end = time.time()
        print(f"总初始化时间: {total_end - total_start:.3f}秒")

    def _check_files(self):
        for path in [self.model_path, self.tokens_path, self.lexicon_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺失文件: {path}")

    def text_to_speech(self, text: str):
        text = text.strip()
        if not text:
            return

        try:
            print(f"生成语音: '{text[:20]}...'")
            start = time.time()
            audio = self.tts.generate(text)
            end = time.time()
            if len(audio.samples) == 0:
                print("生成失败")
                return
            print(f"语音生成完成，耗时: {end - start:.3f} 秒")
            # 播放
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            try:
                sf.write(temp_path, audio.samples, audio.sample_rate)
                subprocess.run(
                    ["aplay", "-D", self.aplay_device, temp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            print("播放完成")

        except Exception as e:
            print(f"错误: {e}")

def main():
    tts = Text_Tranform()

    while True:
        text = input("请输入文本: ")

        if text.strip().lower() == "q":
            print("已退出。")
            break

        tts.text_to_speech(text)

if __name__ == "__main__":
    main()