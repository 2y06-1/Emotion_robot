import wave
import sounddevice as sd
from pathlib import Path
import time
import os
import tempfile
import numpy as np
import threading

voice_path = Path(tempfile.gettempdir()) / "emotion_robot_wav"
device_id = 2

class Voice_Collect:
    def __init__(self, voice_path, device_id):
        self.device_id = device_id
        self.voice_path = Path(voice_path)
        self._stop_recording = threading.Event()  # 停止事件
        self._recording_lock = threading.Lock()
        self._is_recording = False

    def clean_files(self, max_files):
        files = sorted(self.voice_path.glob("*.wav"), key=lambda x: x.stat().st_mtime)
        if len(files) > max_files:
            for f in files[:-max_files]:
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

    def _has_voice_in_data(self, audio_array, fs, threshold=600, min_voice_sec=2):
        try:
            chunk_size = fs  # 每1秒为一个检测块
            voice_sec = 0
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i+chunk_size]
                if np.max(np.abs(chunk)) > threshold:
                    voice_sec += 1
                if voice_sec >= min_voice_sec:
                    print(f"检测到人声（{voice_sec}秒）")
                    return True
            print(f"未检测到人声（最大音量: {np.max(np.abs(audio_array))}）")
            return False
        except Exception as e:
            print(f"检查人声错误: {e}")
            return False

    def record_audio(self, max_duration=60):
        with self._recording_lock:
            self._is_recording = True
            self._stop_recording.clear()
            
            save_path = self.voice_path
            save_path.mkdir(parents=True, exist_ok=True)
            filename = f"audio_{int(time.time())}.wav"
            wav_path = save_path / filename

            device_info = sd.query_devices(self.device_id)
            fs = int(device_info['default_samplerate'])
            channels = 1
            chunk_size = 1024
            audio_data = []

            print("--开始录音（等待停止信号）--")
            try:
                with sd.InputStream(samplerate=fs, channels=channels, device=self.device_id, dtype='int16') as stream:
                    start_time = time.time()
                    while True:
                        # 检查是否收到停止信号（按钮点击触发）
                        if self._stop_recording.is_set():
                            print("收到停止信号，结束录音")
                            break
                            
                        # 安全限制
                        if time.time() - start_time > max_duration:
                            print("达到最大录音时间（安全限制），结束录音")
                            break
                            
                        chunk, overflowed = stream.read(chunk_size)
                        audio_data.append(chunk)
                        
            except Exception as e:
                print(f"录音错误: {e}")
                self._is_recording = False
                return None

            # 检查是否录到了有效音频数据
            if len(audio_data) == 0:
                print("没有录到任何音频数据")
                self._is_recording = False
                return None

            # 合并音频数据
            audio_data_bytes = b''.join([c.tobytes() for c in audio_data])
            
            # 检查是否有人声（直接在内存中检测）
            audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
            if not self._has_voice_in_data(audio_array, fs):
                print("录音中未检测到有效人声，丢弃此段录音")
                self._is_recording = False
                return None

            # 保存音频文件
            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(audio_data_bytes)

            print(f"已保存: {wav_path}")
            self.clean_files(max_files=4)
            self._is_recording = False
            return str(wav_path)