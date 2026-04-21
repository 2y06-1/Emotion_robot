import os
import shutil
import llm_asr

class Voice_Transform:
    def __init__(self):
        self.model_path = "/usr/local/share/voice_test/sensevoice"

        print("加载 SenseVoice 模型...")
        self.model = llm_asr.llm_asr(self.model_path)
        print("加载完成")

    def speech_to_text(self, wav_path):
        temp_path = "/tmp/temp.wav"

        shutil.copy(wav_path, temp_path)

        text = self.model.audio_to_text(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return text