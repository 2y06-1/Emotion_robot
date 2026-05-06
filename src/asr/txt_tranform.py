import subprocess
from pathlib import Path

def tts_to_file(text, output_path="output.wav"):
    if not text.strip():
        return
    
    subprocess.run([
        "espeak-ng",
        "-v", "cmn-latn-pinyin",   
        "-s", "140",     
        text
    ])

if __name__ == "__main__":
    while True:
        txt = input("输入文本: ")
        if txt == "q":
            break
        tts_to_file(txt)