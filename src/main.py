import sys
import os
import time
import threading
import random
import subprocess
from pathlib import Path
import tempfile
import re

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QObject, pyqtSignal

# ===== 路径配置 =====
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "src" / "asr"))
sys.path.append(str(BASE_DIR / "src" / "llm"))
sys.path.append(str(BASE_DIR / "src" / "ui"))

from ui import MainWindow
from new_voice_collect import Voice_Collect
from voice_tranform import Voice_Transform
from llm import Ollama_chat

# ========== 全局配置 ==========
VOICE_PATH = Path(tempfile.gettempdir()) / "emotion_robot_wav"
DEVICE_ID = 2
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "my_test:latest"
CHAT_HISTORY = BASE_DIR / "model" / "llm" / "chat_history.txt"
INIT_WAV = BASE_DIR / "wav" / "init.wav"
EMOTION_WAV = BASE_DIR / "wav"

# TTS 子进程配置（根据你的环境修改）
TTS_VENV_PYTHON = "/home/bianbu/Emotion_robot/venv_tts/bin/python"
TTS_WORKER_PATH = "/home/bianbu/Emotion_robot/src/asr/tts_worker.py"

class EmotionRobot(QObject):
    # ===== UI 信号 =====
    ui_clear_chat = pyqtSignal()
    ui_append_user = pyqtSignal(str)
    ui_append_ai = pyqtSignal(str)
    ui_append_system = pyqtSignal(str)
    ui_append_emotion = pyqtSignal(str, str)
    ui_set_state_emotion_detecting = pyqtSignal()
    ui_set_state_chatting = pyqtSignal()
    ui_set_state_error = pyqtSignal(str)
    ui_set_emotion = pyqtSignal(str, str)

    # TTS 播放状态信号（主线程安全）
    tts_start = pyqtSignal(str)   # 通知播放
    tts_stop = pyqtSignal()       # 播放完成

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.ui = MainWindow()

        # 核心模块
        self.voice_collector = Voice_Collect(VOICE_PATH, DEVICE_ID)
        self.asr = Voice_Transform()
        self.llm_bot = Ollama_chat(OLLAMA_URL, MODEL_NAME, CHAT_HISTORY)

        # 状态标志
        self.current_mode = "emotion"
        self.is_recording = False
        self.is_playing_tts = False      # 新增：TTS 播放中标志，防止录音
        self.active_emotion = None
        self.pending_strong_emotion = None

        # 连接 UI 信号
        self.ui_clear_chat.connect(self._on_ui_clear_chat)
        self.ui_append_user.connect(self.ui.append_user_message)
        self.ui_append_ai.connect(self.ui.append_ai_message)
        self.ui_append_system.connect(self.ui.append_system_message)
        self.ui_append_emotion.connect(self.ui.append_emotion_message)
        self.ui_set_state_emotion_detecting.connect(self.ui.set_state_emotion_detecting)
        self.ui_set_state_chatting.connect(self.ui.set_state_chatting)
        self.ui_set_state_error.connect(self.ui.set_state_error)
        self.ui_set_emotion.connect(self.ui.set_emotion)

        # 连接按钮
        self.ui.record_button_clicked.connect(self.on_record_button)
        self.ui.exit_chat_clicked.connect(self.on_exit_chat)
        self.ui.exit_program_clicked.connect(self.on_exit_program)

        # 后台情绪模拟定时器
        self.emotion_timer = QTimer()
        self.emotion_timer.timeout.connect(self._simulate_emotion_detection)
        self.emotion_timer.start(5000)

        # 启动 TTS 子进程（独立线程，不阻塞主界面）
        self._start_tts_process()

        # 连接 TTS 控制信号
        self.tts_start.connect(self._on_tts_start)
        self.tts_stop.connect(self._on_tts_stop)

        # 界面启动
        self.ui.showFullScreen()
        self._play_init_sound()

    # ---------- TTS 子进程管理 ----------
    def _start_tts_process(self):
        try:
            self.tts_proc = subprocess.Popen(
                [TTS_VENV_PYTHON, TTS_WORKER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # 读取启动信息（模型加载完成标志）
            startup = self.tts_proc.stdout.readline()
            print(f"[TTS] {startup.strip()}", flush=True)
            print("[Main] TTS 子进程已启动", flush=True)
        except Exception as e:
            print(f"[Main] 无法启动 TTS 子进程: {e}", flush=True)
            self.tts_proc = None

    def _cleanup_tts(self):
        if self.tts_proc:
            self.tts_proc.terminate()
            try:
                self.tts_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.tts_proc.kill()
            self.tts_proc = None

    # ---------- 语音播放（在线程中运行） ----------
    def _play_tts_text(self, text):
        """阻塞式播放，在独立线程中调用"""
        if not text.strip() or self.tts_proc is None or self.tts_proc.stdin is None:
            return
        try:
            self.tts_proc.stdin.write(text.strip() + "\n")
            self.tts_proc.stdin.flush()
            while True:
                line = self.tts_proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                print(line, flush=True)
                if line == "TTS_DONE":
                    break
        except Exception as e:
            print(f"[TTS] 播放异常: {e}", flush=True)

    # ---------- 播放控制信号槽 ----------
    def _on_tts_start(self, text):
        """启动播放线程"""
        self.is_playing_tts = True
        self.ui.record_button.setEnabled(False)   # 播放时禁用录音按钮
        def worker():
            self._play_tts_text(text)
            self.tts_stop.emit()
        threading.Thread(target=worker, daemon=True).start()

    def _on_tts_stop(self):
        self.is_playing_tts = False
        self.ui.record_button.setEnabled(True)
        # 根据当前模式刷新 UI 状态，确保按钮文本/提示正确
        if self.current_mode == "emotion":
            self.ui_set_state_emotion_detecting.emit()
        else:
            self.ui_set_state_chatting.emit()
    # ---------- UI 槽函数 ----------
    def _on_ui_clear_chat(self):
        self.ui.chat_box.clear()

    # ---------- 按钮事件 ----------
    def on_record_button(self):
        if self.is_playing_tts:                # 播放中不允许录音
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.is_recording = True
        self.ui.set_state_listening()
        self.rec_thread = threading.Thread(target=self._record_thread, daemon=True)
        self.rec_thread.start()

    def _stop_recording(self):
        self.voice_collector.stop_recording()

    def on_exit_chat(self):
        if self.is_recording:
            self.voice_collector.stop_recording()
            self.is_recording = False
        self.llm_bot.history_clear()
        self.current_mode = "emotion"
        self.active_emotion = None
        self.pending_strong_emotion = None
        self.ui_clear_chat.emit()
        self.ui_set_state_emotion_detecting.emit()
        self.ui_set_emotion.emit("--", "等待检测人物情绪")

    def on_exit_program(self):
        reply = QMessageBox.question(
            self.ui, "确认退出", "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.is_recording:
                self.voice_collector.stop_recording()
            self._cleanup_tts()
            self.app.quit()

    # ---------- 录音线程 ----------
    def _record_thread(self):
        try:
            audio_path = self.voice_collector.record_audio(max_duration=60)
            if not audio_path:
                self.ui_append_system.emit("未检测到有效声音")
                self._after_record_reset()
                return

            self._process_audio(audio_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            self.ui_append_system.emit(f"录音错误: {e}")
            self.ui_set_state_error.emit(str(e))
        finally:
            self.is_recording = False

    def _after_record_reset(self):
        if self.current_mode == "emotion":
            self.ui_set_state_emotion_detecting.emit()
        else:
            self.ui_set_state_chatting.emit()

    # ---------- 音频处理分派 ----------
    def _process_audio(self, audio_path):
        user_text = self.asr.speech_to_text(audio_path).strip()
        print(f"[用户说] {user_text}")
        if len(user_text) < 2 or not self._is_valid_chinese(user_text):
            self.ui_append_system.emit("未识别到有效文本，请重试")
            self._after_record_reset()
            return

        if self.current_mode == "emotion":
            if self.pending_strong_emotion:
                emotion = self.pending_strong_emotion
                self.pending_strong_emotion = None
                self._enter_chat_with_emotion(user_text, emotion)
            else:
                self._start_normal_chat(user_text)
        else:
            self._handle_chat_message(user_text)

    # ---------- 后台情绪模拟 ----------
    def _simulate_emotion_detection(self):
        if self.current_mode != "emotion" or self.is_recording:
            return
        if random.random() < 0.1:
            emotion = random.choice(["Happy", "Sad", "Angry", "Surprise", "Fear"])
            if self.pending_strong_emotion == emotion:
                return
            self.pending_strong_emotion = emotion
            ask_text = f"检测到您似乎{emotion}，愿意和我聊聊吗？"
            print(ask_text)
            self.ui_append_system.emit(ask_text)
            self._play_emotion_wav(emotion)   # 播放预置情绪语音

    # ---------- 进入聊天模式 ----------
    def _enter_chat_with_emotion(self, user_text, emotion):
        self.current_mode = "chat"
        self.active_emotion = emotion
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)
        enhanced = f"[用户当前情绪：{emotion}] {user_text}"
        self._generate_ai_reply(enhanced)

    def _start_normal_chat(self, user_text):
        self.current_mode = "chat"
        self.active_emotion = None
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)
        self._generate_ai_reply(user_text)

    def _handle_chat_message(self, user_text):
        self.ui_append_user.emit(user_text)
        if self.active_emotion:
            message = f"[用户当前情绪：{self.active_emotion}] {user_text}"
        else:
            message = user_text
        self._generate_ai_reply(message)

    # ---------- LLM 生成（独立线程） ----------
    def _generate_ai_reply(self, message):
        def worker():
            try:
                reply = self.llm_bot.chat_ollama(message).strip()
                if not reply:
                    reply = "我好像没听清，可以再说一遍吗？"
            except Exception as e:
                reply = "抱歉，我现在脑子有点乱，请稍后再试。"
                print(f"[LLM 错误] {e}")
            print(f"[AI 回复] {reply}")
            self.ui_append_ai.emit(reply)
            # 播放 AI 回复的语音
            self.tts_start.emit(reply)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- 播放预置情绪 WAV ----------
    def _play_emotion_wav(self, emotion):
        """播放固定的情绪提示音（短音频）"""
        wav_name = emotion.lower() + ".wav"
        wav_path = EMOTION_WAV / wav_name
        if wav_path.exists():
            subprocess.run(
                ["aplay", "-D", "plughw:0,0", str(wav_path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

    # ---------- 辅助方法 ----------
    def _is_valid_chinese(self, text, threshold=0.3):
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return len(text) > 0 and chinese_count / len(text) >= threshold

    def _play_init_sound(self):
        if INIT_WAV.exists():
            subprocess.run(["aplay", "-D", "plughw:0,0", str(INIT_WAV)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    robot = EmotionRobot()
    robot.run()