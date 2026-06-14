import os
import re
import time
import tempfile
import subprocess
from pathlib import Path

import sherpa_onnx
import soundfile as sf


class Text_Tranform:
    """
    TTS 文本转语音模块。
    """

    def __init__(
        self,
        model_dir,
        provider,
        num_threads,
        sid,
        speed,
        silence_scale,
        aplay_device,
        max_chars,
        warmup,
        max_num_sentences,
    ):
        print("加载 TTS 模型...", flush=True)
        total_start = time.time()

        self.model_dir = Path(model_dir)
        self.provider = provider
        self.num_threads = int(num_threads)
        self.sid = int(sid)
        self.speed = float(speed)
        self.silence_scale = float(silence_scale)
        self.aplay_device = str(aplay_device)
        self.max_chars = int(max_chars)
        self.warmup = bool(warmup)
        self.max_num_sentences = int(max_num_sentences)

        self.model_path = self.model_dir / "model.onnx"
        self.tokens_path = self.model_dir / "tokens.txt"
        self.lexicon_path = self.model_dir / "lexicon.txt"

        self.phone_fst_path = self.model_dir / "phone.fst"
        self.date_fst_path = self.model_dir / "date.fst"
        self.number_fst_path = self.model_dir / "number.fst"

        self.rule_fsts = ",".join([
            str(self.phone_fst_path),
            str(self.date_fst_path),
            str(self.number_fst_path),
        ])

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
                    model=str(self.model_path),
                    tokens=str(self.tokens_path),
                    lexicon=str(self.lexicon_path),
                ),
                provider=self.provider,
                num_threads=self.num_threads,
                debug=False,
            ),
            rule_fsts=self.rule_fsts,
            rule_fars="",
            max_num_sentences=self.max_num_sentences,
        )

        load_start = time.time()
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        load_end = time.time()

        print(f"[计时] 模型加载: {load_end - load_start:.3f} 秒", flush=True)

        self.gen_config = sherpa_onnx.GenerationConfig()
        self.gen_config.sid = self.sid
        self.gen_config.speed = self.speed
        self.gen_config.silence_scale = self.silence_scale

        if self.warmup:
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
            self.phone_fst_path,
            self.date_fst_path,
            self.number_fst_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"缺失文件: {path}")

    def _clean_text(self, text: str) -> str:
        text = str(text).strip()

        # 逗号转句号，增加明显停顿。
        text = text.replace("，", "。").replace(",", "。")

        # 清理常见 Markdown / HTML 符号。
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[#>*`_\[\]{}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.max_chars > 0 and len(text) > self.max_chars:
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
            dir=tmp_dir,
        ) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio.samples, audio.sample_rate, subtype="PCM_16")

            play_start = time.time()

            try:
                subprocess.run(
                    ["aplay", "-q", "-D", self.aplay_device, temp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
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
