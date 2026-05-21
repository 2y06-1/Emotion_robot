import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.asr.new_txt_tranform import Text_Tranform


print("初始化 TTS...", flush=True)

tts = Text_Tranform()

print("TTS 初始化完成", flush=True)


while True:
    try:
        text = sys.stdin.readline()

        if not text:
            break

        text = text.strip()

        if not text:
            continue

        tts.text_to_speech(text)

        # 通知主程序：播放完成
        print("TTS_DONE", flush=True)

    except KeyboardInterrupt:
        break

    except Exception as e:
        print(f"TTS worker error: {e}", flush=True)