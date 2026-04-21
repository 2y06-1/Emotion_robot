import wave
import sounddevice as sd
from pathlib import Path
import time
import os

class Voice_Collect:
    def __init__(self, voice_path):
        self.voice_path = voice_path

    def clean_files(self,max_files):

        files = sorted(Path(self.voice_path).glob("*.wav"),key=lambda x: x.stat().st_mtime)

        if len(files) > max_files:
            for f in files[:-max_files]:
                try:
                    os.remove(f)
                    print(f"删除旧文件: {f.name}")
                except Exception as e:
                    print("删除失败:", e)


    def record_audio(self):

        save_path = Path(self.voice_path)
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"audio_{int(time.time())}.wav"
        wav_path = save_path / filename

        fs = 16000
        voice_time = 3

        print("--begin--")

        audio_data = sd.rec(
            int(voice_time * fs),
            samplerate=fs,
            channels=1,
            dtype='int16'
        )

        sd.wait()

        print("--end--")

        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())

        print(f"已保存: {wav_path}")

        self.clean_files(max_files=4)

        return str(wav_path)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    voice_path = current_dir.parent.parent/ "voice"

    vc=Voice_Collect(voice_path)
    while True:
        wav_file = vc.record_audio()
        if not wav_file:
            break