import sys
from pathlib import Path
import tempfile 

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

voice_path = Path(tempfile.gettempdir()) / "emotion_robot_wav"
device_id=2

current_dir = Path(__file__).resolve().parent
base_url = "http://localhost:11434"
model_name = "my_test:latest"
txt_path = current_dir.parent/"model"/"llm"/"chat_history.txt"

from src.asr.txt_tranform import Text_Tranform
from src.asr.voice_collect import Voice_Collect
from src.asr.voice_tranform import Voice_Transform
from src.llm.llm  import Ollama_chat 

vc=Voice_Collect(voice_path,device_id)
print("vc_init succees")
sensevoice_model=Voice_Transform()
print("asr succees")
bot = Ollama_chat(base_url, model_name, txt_path)
print("llm succees")
tts = Text_Tranform()
print("tts success")
def main():

    while True:
        wav_file = vc.record_audio()
        text = sensevoice_model.speech_to_text(wav_file)
        print(text)
        print("begin")
        full_reply=bot.chat_ollama(text) 
        print("end")
        print(full_reply)
        tts.text_to_speech(full_reply)
        if not wav_file:
            break

if __name__ == "__main__":
    main()