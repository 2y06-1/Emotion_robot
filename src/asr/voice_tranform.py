import llm_asr
from pathlib import Path
import shutil

current_dir = Path(__file__).resolve().parent
sensevoice_model_path = current_dir.parent.parent/ "models"/"asr"/"sensevoice"
wav_path=current_dir.parent.parent/"wav"

class Voice_Transform:
    def __init__(self,model_path,wav_path):
        self.model_path = model_path
        self.wav_path = wav_path
        self.load_model()    
    
    def load_model(self):
        try:
            self.model = llm_asr.llm_asr(self.model_path)
            print("load success")
        except Exception as e:
            self.model=None
            print("load failed")
    
    def speech_to_text(self):
        temp_filepath=self.wav_path
        shutil.copyfile(self.wav_path/"zh.mp3", temp_filepath/"new.mp3")
        text = self.model.audio_to_text(temp_filepath/"new.mp3")
        
        return text

if __name__ == "__main__":
    sensevoice_model=Voice_Transform(sensevoice_model_path,wav_path)
    text=sensevoice_model.speech_to_text()
    while True:
        pass