import os
import re
import time
import tempfile
import subprocess

import sherpa_onnx
import soundfile as sf


class Text_Tranform:
    def __init__(self):
        print("加载 TTS 模型...", flush=True)
        total_start = time.time()

        # 你当前测试成功的新模型目录
        self.model_dir = os.getenv(
            "TTS_MODEL_DIR",
            "/home/bianbu/Emotion_robot/model/asr"
        )

        self.model_path = os.path.join(self.model_dir, "model.onnx")
        self.tokens_path = os.path.join(self.model_dir, "tokens.txt")
        self.lexicon_path = os.path.join(self.model_dir, "lexicon.txt")

        self.rule_fsts = ",".join([
            os.path.join(self.model_dir, "phone.fst"),
            os.path.join(self.model_dir, "date.fst"),
            os.path.join(self.model_dir, "number.fst"),
        ])

        # 稳定演示版：CPU + 4 线程
        self.provider = os.getenv("TTS_PROVIDER", "cpu")
        self.num_threads = int(os.getenv("TTS_NUM_THREADS", "4"))

        # aishell3 是多说话人模型，sid=66 是你目前测试用的音色
        self.sid = int(os.getenv("TTS_SID", "66"))
        self.speed = float(os.getenv("TTS_SPEED", "1.0"))
        self.silence_scale = float(os.getenv("TTS_SILENCE_SCALE", "0.2"))

        # 音频设备。如果播放失败，可以运行前改成 APLAY_DEVICE=default
        self.aplay_device = os.getenv("APLAY_DEVICE", "plughw:0,0")

        # 限制 TTS 文本长度，避免大模型回复太长导致等待时间过久
        self.max_chars = int(os.getenv("TTS_MAX_CHARS", "100"))

        self._check_files()

        print(f"[TTS] model   = {self.model_path}", flush=True)
        print(f"[TTS] tokens  = {self.tokens_path}", flush=True)
        print(f"[TTS] lexicon = {self.lexicon_path}", flush=True)
        print(f"[TTS] rules   = {self.rule_fsts}", flush=True)
        print(f"[TTS] provider= {self.provider}", flush=True)
        print(f"[TTS] threads = {self.num_threads}", flush=True)
        print(f"[TTS] sid     = {self.sid}", flush=True)
        print(f"[TTS] device  = {self.aplay_device}", flush=True)

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=self.model_path,
                    tokens=self.tokens_path,
                    lexicon=self.lexicon_path,
                ),
                provider=self.provider,
                num_threads=self.num_threads,
                debug=False,
            ),
            rule_fsts=self.rule_fsts,
            rule_fars="",
            max_num_sentences=1,
        )

        load_start = time.time()
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        load_end = time.time()

        print(f"[计时] 模型加载: {load_end - load_start:.3f} 秒", flush=True)

        self.gen_config = sherpa_onnx.GenerationConfig()
        self.gen_config.sid = self.sid
        self.gen_config.speed = self.speed
        self.gen_config.silence_scale = self.silence_scale

        # 预热一次，避免第一次真正说话时额外变慢
        if os.getenv("TTS_WARMUP", "1") == "1":
            try:
                print("[TTS] warmup...", flush=True)
                self.tts.generate("你好", self.gen_config)
                print("[TTS] warmup done", flush=True)
            except Exception as e:
                print(f"[TTS] warmup 失败: {e}", flush=True)

        total_end = time.time()
        print(f"[计时] TTS 初始化总耗时: {total_end - total_start:.3f} 秒", flush=True)

    def _check_files(self):
        required_files = [
            self.model_path,
            self.tokens_path,
            self.lexicon_path,
            os.path.join(self.model_dir, "phone.fst"),
            os.path.join(self.model_dir, "date.fst"),
            os.path.join(self.model_dir, "number.fst"),
        ]

        for path in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺失文件: {path}")

    def _clean_text(self, text: str) -> str:
        text = text.strip()

        text = re.sub(r"", "", text, flags=re.S)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[#>*`_\[\]{}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 兜底限制长度（可适当放宽，比如 200，或完全禁用）
        if self.max_chars and len(text) > self.max_chars:
            text = text[:self.max_chars]

        return text

    def _generate_audio(self, text: str):
        start = time.time()
        audio = self.tts.generate(text, self.gen_config)
        end = time.time()

        if len(audio.samples) == 0:
            raise RuntimeError("TTS 生成失败：audio.samples 为空")

        gen_time = end - start
        audio_duration = len(audio.samples) / audio.sample_rate
        rtf = gen_time / audio_duration if audio_duration > 0 else 999.0

        print(
            f"[TTS] 生成耗时: {gen_time:.3f} 秒, "
            f"音频时长: {audio_duration:.3f} 秒, "
            f"RTF: {rtf:.3f}, "
            f"采样率: {audio.sample_rate}",
            flush=True,
        )

        return audio

    def _play_audio(self, audio):
        tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
            dir=tmp_dir
        ) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio.samples, audio.sample_rate, subtype="PCM_16")

            play_start = time.time()

            # 优先使用指定设备播放
            try:
                subprocess.run(
                    ["aplay", "-q", "-D", self.aplay_device, temp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                # 如果 plughw:0,0 播放失败，自动回退 default
                subprocess.run(
                    ["aplay", "-q", "-D", "default", temp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

            play_end = time.time()
            print(f"[TTS] 播放耗时: {play_end - play_start:.3f} 秒", flush=True)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def text_to_speech(self, text: str):
        text = self._clean_text(text)

        if not text:
            print("[TTS] 空文本，跳过播放", flush=True)
            return False

        try:
            print(f"[TTS] 输入文本: {text}", flush=True)
            audio = self._generate_audio(text)
            self._play_audio(audio)
            print("[TTS] 播放完成", flush=True)
            return True

        except Exception as e:
            print(f"[TTS] 错误: {e}", flush=True)
            return False


def main():
    tts = Text_Tranform()

    while True:
        try:
            text = input("请输入文本: ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n已退出。")
            break

        if text.strip().lower() in ["q", "quit", "exit"]:
            print("已退出。")
            break

        tts.text_to_speech(text)


if __name__ == "__main__":
    main()