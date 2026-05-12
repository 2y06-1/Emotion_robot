import sys
from pathlib import Path

project_root = Path("/home/bianbu/asr-llm-tts/asr-llm-tts/src")
sys.path.append(str(project_root))

from asr import AsrModel

class Voice_Transform:
    def __init__(self):
        self.load_model()    
    
    def load_model(self):
        try:
            self.model = AsrModel()
            print("load success")
        except Exception as e:
            self.model=None
            print("load failed")
    
    def speech_to_text(self,wav_path):
        temp_filepath=wav_path
        text = self.model(temp_filepath)
        
        return text

def main():
    # 测试用
    sensevoice_model=Voice_Transform()
    text=sensevoice_model.speech_to_text("/home/bianbu/Emotion_robot/wav/audio_1778415089.wav")
    print(text)

if __name__ == "__main__":
    main()
