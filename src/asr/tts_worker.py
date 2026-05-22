import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent    # src/
sys.path.insert(0, str(BASE_DIR / "asr"))              # new_txt_tranform 在这里
sys.path.insert(0, str(BASE_DIR / "llm"))

from new_txt_tranform import Text_Tranform

def main():
    print("[TTS Worker] 正在加载模型...", flush=True)
    tts = Text_Tranform()
    print("[TTS Worker] 模型就绪，等待输入...", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(f"[TTS Worker] 收到: {line}", flush=True)
        tts.text_to_speech(line)
        print("TTS_DONE", flush=True)        # 主进程等待这个标记

if __name__ == "__main__":
    main()