import os
import time
import wave
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd


class Voice_Collect:
    """
    录音采集模块。

    接口原则：
    - 本文件不读取 config.json。
    - 本文件不写死录音目录、设备号、阈值、通道数等配置。
    - 所有配置由 main.py 从 config.py 读取后传入。
    """

    def __init__(
        self,
        voice_path,
        device_id,
        max_keep_files,
        voice_threshold,
        min_voice_sec,
        channels,
        chunk_size,
        dtype,
    ):
        self.voice_path = Path(voice_path)
        self.device_id = int(device_id)
        self.max_keep_files = int(max_keep_files)
        self.voice_threshold = int(voice_threshold)
        self.min_voice_sec = int(min_voice_sec)
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.dtype = str(dtype)

        self._stop_recording = threading.Event()
        self._recording_lock = threading.Lock()
        self._is_recording = False

    def clean_files(self):
        files = sorted(self.voice_path.glob("*.wav"), key=lambda x: x.stat().st_mtime)

        if len(files) <= self.max_keep_files:
            return

        for f in files[:-self.max_keep_files]:
            try:
                os.remove(f)
                print(f"删除旧文件: {f.name}")
            except Exception as e:
                print("删除失败:", e)

    def stop_recording(self):
        self._stop_recording.set()
        print("收到停止录音请求")

    def is_recording(self):
        return self._is_recording

    def _has_voice_in_data(self, audio_array, fs):
        try:
            chunk_size = fs  # 每 1 秒为一个检测块
            voice_sec = 0

            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                if len(chunk) == 0:
                    continue

                if np.max(np.abs(chunk)) > self.voice_threshold:
                    voice_sec += 1

                if voice_sec >= self.min_voice_sec:
                    print(f"检测到人声（{voice_sec}秒）")
                    return True

            max_volume = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
            print(f"未检测到人声（最大音量: {max_volume}）")
            return False

        except Exception as e:
            print(f"检查人声错误: {e}")
            return False

    def record_audio(self, max_duration):
        with self._recording_lock:
            self._is_recording = True
            self._stop_recording.clear()

            self.voice_path.mkdir(parents=True, exist_ok=True)
            wav_path = self.voice_path / f"audio_{int(time.time())}.wav"

            try:
                device_info = sd.query_devices(self.device_id)
                fs = int(device_info["default_samplerate"])
            except Exception as e:
                print(f"获取录音设备失败: {e}")
                self._is_recording = False
                return None

            audio_data = []
            print("--开始录音（等待停止信号）--")

            try:
                with sd.InputStream(
                    samplerate=fs,
                    channels=self.channels,
                    device=self.device_id,
                    dtype=self.dtype,
                ) as stream:
                    start_time = time.time()

                    while True:
                        if self._stop_recording.is_set():
                            print("收到停止信号，结束录音")
                            break

                        if time.time() - start_time > max_duration:
                            print("达到最大录音时间（安全限制），结束录音")
                            break

                        chunk, overflowed = stream.read(self.chunk_size)
                        audio_data.append(chunk)

            except Exception as e:
                print(f"录音错误: {e}")
                self._is_recording = False
                return None

            if len(audio_data) == 0:
                print("没有录到任何音频数据")
                self._is_recording = False
                return None

            audio_data_bytes = b"".join([c.tobytes() for c in audio_data])
            audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)

            if not self._has_voice_in_data(audio_array, fs):
                print("录音中未检测到有效人声，丢弃此段录音")
                self._is_recording = False
                return None

            try:
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(audio_data_bytes)
            except Exception as e:
                print(f"保存录音失败: {e}")
                self._is_recording = False
                return None

            print(f"已保存: {wav_path}")
            self.clean_files()
            self._is_recording = False
            return str(wav_path)
