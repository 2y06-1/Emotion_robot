# -*- coding: utf-8 -*-

import os
import re
import subprocess
import tempfile
import time
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
        print(
            "加载 TTS 模型...",
            flush=True,
        )

        total_start = time.time()

        self.model_dir = Path(model_dir)
        self.provider = str(provider)
        self.num_threads = int(num_threads)
        self.sid = int(sid)
        self.speed = float(speed)
        self.silence_scale = float(
            silence_scale
        )
        self.aplay_device = str(
            aplay_device
        )
        self.max_chars = int(max_chars)
        self.warmup = bool(warmup)
        self.max_num_sentences = int(
            max_num_sentences
        )

        self.model_path = (
            self.model_dir / "model.onnx"
        )
        self.tokens_path = (
            self.model_dir / "tokens.txt"
        )
        self.lexicon_path = (
            self.model_dir / "lexicon.txt"
        )

        self.phone_fst_path = (
            self.model_dir / "phone.fst"
        )
        self.date_fst_path = (
            self.model_dir / "date.fst"
        )
        self.number_fst_path = (
            self.model_dir / "number.fst"
        )

        self.rule_fsts = ",".join(
            [
                str(self.phone_fst_path),
                str(self.date_fst_path),
                str(self.number_fst_path),
            ]
        )

        self._check_files()

        print(
            f"[TTS] model   = {self.model_path}",
            flush=True,
        )
        print(
            f"[TTS] tokens  = {self.tokens_path}",
            flush=True,
        )
        print(
            f"[TTS] lexicon = {self.lexicon_path}",
            flush=True,
        )
        print(
            f"[TTS] rules   = {self.rule_fsts}",
            flush=True,
        )
        print(
            f"[TTS] provider= {self.provider}",
            flush=True,
        )
        print(
            f"[TTS] threads = {self.num_threads}",
            flush=True,
        )
        print(
            f"[TTS] sid     = {self.sid}",
            flush=True,
        )
        print(
            f"[TTS] device  = {self.aplay_device}",
            flush=True,
        )

        tts_config = (
            sherpa_onnx.OfflineTtsConfig(
                model=(
                    sherpa_onnx
                    .OfflineTtsModelConfig(
                        vits=(
                            sherpa_onnx
                            .OfflineTtsVitsModelConfig(
                                model=str(
                                    self.model_path
                                ),
                                tokens=str(
                                    self.tokens_path
                                ),
                                lexicon=str(
                                    self.lexicon_path
                                ),
                            )
                        ),
                        provider=self.provider,
                        num_threads=(
                            self.num_threads
                        ),
                        debug=False,
                    )
                ),
                rule_fsts=self.rule_fsts,
                rule_fars="",
                max_num_sentences=(
                    self.max_num_sentences
                ),
            )
        )

        load_start = time.time()

        self.tts = sherpa_onnx.OfflineTts(
            tts_config
        )

        load_end = time.time()

        print(
            "[计时] 模型加载: "
            f"{load_end - load_start:.3f} 秒",
            flush=True,
        )

        self.gen_config = (
            sherpa_onnx.GenerationConfig()
        )
        self.gen_config.sid = self.sid
        self.gen_config.speed = self.speed
        self.gen_config.silence_scale = (
            self.silence_scale
        )

        if self.warmup:
            try:
                print(
                    "[TTS] warmup...",
                    flush=True,
                )

                self.tts.generate(
                    "你好",
                    self.gen_config,
                )

                print(
                    "[TTS] warmup done",
                    flush=True,
                )

            except Exception as exc:
                print(
                    "[TTS] warmup 失败: "
                    f"{exc}",
                    flush=True,
                )

        total_end = time.time()

        print(
            "[计时] TTS 初始化总耗时: "
            f"{total_end - total_start:.3f} 秒",
            flush=True,
        )

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
                raise FileNotFoundError(
                    f"缺失文件: {path}"
                )

    def _clean_text(
        self,
        text: str,
    ) -> str:
        text = str(text or "").strip()

        # 逗号转句号，增加明显停顿。
        text = text.replace(
            "，",
            "。",
        ).replace(
            ",",
            "。",
        )

        # 清理常见 Markdown 和 HTML 符号。
        text = re.sub(
            r"<.*?>",
            "",
            text,
        )
        text = re.sub(
            r"[#>*`_\[\]{}]",
            "",
            text,
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if (
            self.max_chars > 0
            and len(text) > self.max_chars
        ):
            text = text[
                : self.max_chars
            ]

        return text

    def _generate_audio(
        self,
        text: str,
    ):
        start = time.time()

        audio = self.tts.generate(
            text,
            self.gen_config,
        )

        end = time.time()

        if len(audio.samples) == 0:
            raise RuntimeError(
                "TTS 生成失败："
                "audio.samples 为空"
            )

        gen_time = end - start

        audio_duration = (
            len(audio.samples)
            / audio.sample_rate
        )

        if audio_duration > 0:
            rtf = (
                gen_time
                / audio_duration
            )
        else:
            rtf = 999.0

        print(
            f"[TTS] 生成耗时: "
            f"{gen_time:.3f} 秒, "
            f"音频时长: "
            f"{audio_duration:.3f} 秒, "
            f"RTF: {rtf:.3f}, "
            f"采样率: "
            f"{audio.sample_rate}",
            flush=True,
        )

        return audio

    def _run_aplay(
        self,
        temp_path,
        device,
        on_playback_started=None,
    ):
        """
        启动一个 aplay 进程并等待播放完成。

        回调在 aplay 子进程创建成功后执行。
        这是软件侧能够稳定获取的播放开始时刻。
        """
        command = [
            "aplay",
            "-q",
            "-D",
            str(device),
            str(temp_path),
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Popen 成功返回，表示 aplay 已经启动。
        if on_playback_started is not None:
            on_playback_started()

        _stdout, stderr = (
            process.communicate()
        )

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                stderr=stderr,
            )

    def _play_audio(
        self,
        audio,
        on_playback_started=None,
    ):
        """
        将音频写入临时 WAV 并调用 aplay 播放。

        on_playback_started:
            在软件侧启动 aplay 时回调，用于通知
            主进程计算 TTS 等待时间和端到端延迟。
        """
        tmp_dir = (
            "/dev/shm"
            if os.path.isdir("/dev/shm")
            else None
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
            dir=tmp_dir,
        ) as file_obj:
            temp_path = file_obj.name

        playback_notified = False

        def notify_playback_started():
            nonlocal playback_notified

            if playback_notified:
                return

            playback_notified = True

            if on_playback_started is None:
                return

            try:
                on_playback_started()

            except Exception as exc:
                # 性能事件失败不能影响播音。
                print(
                    "[TTS] 播放开始回调失败: "
                    f"{exc}",
                    flush=True,
                )

        try:
            sf.write(
                temp_path,
                audio.samples,
                audio.sample_rate,
                subtype="PCM_16",
            )

            play_start = time.time()

            try:
                self._run_aplay(
                    temp_path=temp_path,
                    device=(
                        self.aplay_device
                    ),
                    on_playback_started=(
                        notify_playback_started
                    ),
                )

            except subprocess.CalledProcessError as exc:
                error_text = str(
                    exc.stderr or exc
                ).strip()

                print(
                    "[TTS] 指定声卡播放失败，"
                    "尝试 default 设备: "
                    f"{error_text}",
                    flush=True,
                )

                self._run_aplay(
                    temp_path=temp_path,
                    device="default",
                    on_playback_started=(
                        notify_playback_started
                    ),
                )

            play_end = time.time()

            print(
                "[TTS] 播放耗时: "
                f"{play_end - play_start:.3f} 秒",
                flush=True,
            )

        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def text_to_speech(
        self,
        text: str,
        on_playback_started=None,
    ):
        """
        将文本转换为语音并播放。

        返回：
            True：成功生成并播放。
            False：文本为空、生成失败或播放失败。
        """
        text = self._clean_text(text)

        if not text:
            print(
                "[TTS] 空文本，跳过播放",
                flush=True,
            )
            return False

        try:
            print(
                f"[TTS] 输入文本: {text}",
                flush=True,
            )

            audio = self._generate_audio(
                text
            )

            self._play_audio(
                audio,
                on_playback_started=(
                    on_playback_started
                ),
            )

            print(
                "[TTS] 播放完成",
                flush=True,
            )

            return True

        except Exception as exc:
            print(
                f"[TTS] 错误: {exc}",
                flush=True,
            )

            return False