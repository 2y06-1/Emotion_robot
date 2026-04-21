import os
import shutil
import llm_asr

sensevoice_path = "/usr/local/share/voice_test/sensevoice"
sensevoiceModel = llm_asr.llm_asr(sensevoice_path)

original_filepath = "/usr/local/share/voice_test/zh.mp3"
temp_filepath = "/tmp/temp_audio.mp3"

shutil.copy(original_filepath, temp_filepath)

text = sensevoiceModel.audio_to_text(temp_filepath)
print(text)

if os.path.exists(temp_filepath):
    os.remove(temp_filepath)