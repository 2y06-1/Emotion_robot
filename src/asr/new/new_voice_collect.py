import wave
import sounddevice as sd
from pathlib import Path
import time
import os
import tempfile
import numpy as np

voice_path = Path(tempfile.gettempdir()) / "emotion_robot_wav"
device_id = 2

class Voice_Collect:
    def __init__(self, voice_path, device_id):
        self.device_id = device_id
        self.voice_path = Path(voice_path)

    def clean_files(self, max_files):
        files = sorted(self.voice_path.glob("*.wav"), key=lambda x: x.stat().st_mtime)
        if len(files) > max_files:
            for f in files[:-max_files]:
                try:
                    os.remove(f)
                    print(f"删除旧文件: {f.name}")
                except Exception as e:
                    print("删除失败:", e)

    def record_audio(self, max_duration=20, silence_threshold=500, silence_duration=3.0):
        save_path = self.voice_path
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"audio_{int(time.time())}.wav"
        wav_path = save_path / filename

        device_info = sd.query_devices(self.device_id)
        fs = int(device_info['default_samplerate'])
        channels = 1
        chunk_size = 1024
        audio_data = []

        silent_chunks = 0
        max_silent_chunks = int((silence_duration * fs) / chunk_size)

        print("--开始录音--")
        with sd.InputStream(samplerate=fs, channels=channels, device=self.device_id, dtype='int16') as stream:
            start_time = time.time()
            while True:
                chunk, overflowed = stream.read(chunk_size)
                audio_data.append(chunk)
                max_val = chunk.max()
                if max_val < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks >= max_silent_chunks:
                    print("检测到静音，结束录音")
                    break

                if time.time() - start_time > max_duration:
                    print("达到最大录音时间，结束录音")
                    break

        audio_data = b''.join([c.tobytes() for c in audio_data])
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data)

        print(f"已保存: {wav_path}")
        self.clean_files(max_files=4)
        return str(wav_path)

def has_voice(wav_file, threshold=500, min_voice_sec=2):
    with wave.open(str(wav_file), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        fs = wf.getframerate()
        chunk_size = fs  
        voice_sec = 0
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if np.max(np.abs(chunk)) > threshold:
                voice_sec += 1
            if voice_sec >= min_voice_sec:
                return True
    return False

def main():
    vc = Voice_Collect(voice_path, device_id)
    while True:
        wav_file = vc.record_audio()
        if not wav_file:
            continue
        if not has_voice(wav_file, threshold=500, min_voice_sec=2):
            print("录音全程静音，跳过")
            print(wav_file)
            continue

if __name__ == "__main__":
    main()