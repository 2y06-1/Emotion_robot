import sys
from pathlib import Path
import tempfile
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

voice_path = Path(tempfile.gettempdir()) / "emotion_robot_wav"
device_id = 2

current_dir = Path(__file__).resolve().parent

base_url = "http://localhost:11434"
model_name = "my_test:latest"

txt_path = current_dir.parent / "model" / "llm" / "chat_history.txt"

from src.asr.new_voice_collect import Voice_Collect, has_voice
from src.asr.voice_tranform import Voice_Transform
from src.llm.llm import Ollama_chat


# -------------------------
# 中文有效性检测
# -------------------------
def is_valid_chinese(text, threshold=0.3):
    chinese_count = sum('\u4e00' <= c <= '\u9fff' for c in text)

    if len(text) == 0:
        return False

    return chinese_count / len(text) >= threshold


# -------------------------
# 初始化 ASR / LLM
# -------------------------
vc = Voice_Collect(voice_path, device_id)
print("vc_init success")

sensevoice_model = Voice_Transform()
print("asr success")

bot = Ollama_chat(base_url, model_name, txt_path)
print("llm success")


# -------------------------
# 启动常驻 TTS 子进程
# -------------------------
TTS_PROCESS = subprocess.Popen(
    [
        "/home/bianbu/Emotion_robot/venv_tts/bin/python",
        "/home/bianbu/Emotion_robot/src/asr/tts_worker.py",
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

print("tts process success")


# -------------------------
# 播放启动音
# -------------------------
init_wav_path = "/home/bianbu/Emotion_robot/wav/init.wav"

subprocess.run(
    ["aplay", "-D", "plughw:0,0", init_wav_path],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)


# -------------------------
# TTS 播放
# -------------------------
def speak(text):
    if not text.strip():
        return

    if TTS_PROCESS.stdin:
        TTS_PROCESS.stdin.write(text + "\n")
        TTS_PROCESS.stdin.flush()

    # 等待 TTS 播放完成
    while True:
        line = TTS_PROCESS.stdout.readline()

        if not line:
            break

        line = line.strip()

        print(line)

        if line == "TTS_DONE":
            break


# -------------------------
# 主循环
# -------------------------
def main():
    try:
        while True:

            # -------------------------
            # 录音
            # -------------------------
            wav_file = vc.record_audio()

            if not wav_file:
                continue

            # -------------------------
            # 静音检测
            # -------------------------
            if not has_voice(
                wav_file,
                threshold=500,
                min_voice_sec=2
            ):
                print("录音全程静音，跳过")
                print(wav_file)
                continue

            # -------------------------
            # ASR
            # -------------------------
            text = sensevoice_model.speech_to_text(wav_file)

            text = text.strip()

            print(f"识别结果: {text}")

            # -------------------------
            # 无效文本过滤
            # -------------------------
            invalid_texts = {
                "",
                " ",
                "嗯",
                "啊",
                "哦",
                "呃",
                "额",
                "哈",
                "测试",
                "字幕",
                "空",
                "谢谢观看",
            }

            # 文本太短
            if len(text) < 2:
                print("文本过短，跳过")
                continue

            # 无效词
            if text in invalid_texts:
                print("无效文本，跳过")
                continue

            # 中文占比太低
            if not is_valid_chinese(text):
                print("中文占比太低，跳过")
                continue

            # 全是重复字符
            if len(set(text)) <= 1:
                print("疑似噪声文本，跳过")
                continue

            # -------------------------
            # LLM
            # -------------------------
            print("begin")

            full_reply = bot.chat_ollama(text)

            print("end")

            full_reply = full_reply.strip()

            print(full_reply)

            # LLM 空回复
            if not full_reply:
                print("LLM 空回复，跳过")
                continue

            # -------------------------
            # TTS
            # -------------------------
            speak(full_reply)

    finally:
        print("关闭 TTS 子进程")

        if TTS_PROCESS:
            TTS_PROCESS.terminate()


if __name__ == "__main__":
    main()