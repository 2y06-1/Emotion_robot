import sys
import os
import time
import threading
import random
import subprocess
from pathlib import Path
import tempfile
import match

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QObject, pyqtSignal

# ===== 路径配置（按你的项目结构调整） =====
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "src" / "asr"))
sys.path.append(str(BASE_DIR / "src" / "llm"))

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

class EmotionRobot(QObject):
    # ===== UI 信号（主线程安全） =====
    ui_clear_chat = pyqtSignal()
    ui_append_user = pyqtSignal(str)
    ui_append_ai = pyqtSignal(str)
    ui_append_system = pyqtSignal(str)
    ui_append_emotion = pyqtSignal(str, str)          # 目前保留，但不主动使用
    ui_set_state_emotion_detecting = pyqtSignal()     # 切换到情绪检测界面
    ui_set_state_chatting = pyqtSignal()              # 切换到聊天界面
    ui_set_state_error = pyqtSignal(str)
    ui_set_emotion = pyqtSignal(str, str)             # 更新主界面的情绪显示

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.ui = MainWindow()

        # 核心模块
        self.voice_collector = Voice_Collect(VOICE_PATH, DEVICE_ID)
        self.asr = Voice_Transform()
        self.llm_bot = Ollama_chat(OLLAMA_URL, MODEL_NAME, CHAT_HISTORY)

        # 状态标志
        self.current_mode = "emotion"      # "emotion" 或 "chat"
        self.is_recording = False
        self.active_emotion = None         # 进入聊天后持续使用的情绪标签
        self.pending_strong_emotion = None # 后台检测到的、待使用的强烈情绪

        # 连接 UI 信号到主线程槽函数
        self.ui_clear_chat.connect(self._on_ui_clear_chat)
        self.ui_append_user.connect(self.ui.append_user_message)
        self.ui_append_ai.connect(self.ui.append_ai_message)
        self.ui_append_system.connect(self.ui.append_system_message)
        self.ui_append_emotion.connect(self.ui.append_emotion_message)
        self.ui_set_state_emotion_detecting.connect(self.ui.set_state_emotion_detecting)
        self.ui_set_state_chatting.connect(self.ui.set_state_chatting)
        self.ui_set_state_error.connect(self.ui.set_state_error)
        self.ui_set_emotion.connect(self.ui.set_emotion)

        # 连接按钮信号
        self.ui.record_button_clicked.connect(self.on_record_button)
        self.ui.exit_chat_clicked.connect(self.on_exit_chat)
        self.ui.exit_program_clicked.connect(self.on_exit_program)

        # 后台情绪监测定时器（模拟，每 5 秒执行一次）
        self.emotion_timer = QTimer()
        self.emotion_timer.timeout.connect(self._simulate_emotion_detection)
        self.emotion_timer.start(5000)

        # 启动界面
        self.ui.showFullScreen()
        self._play_init_sound()

    # ---------- UI 槽函数 ----------
    def _on_ui_clear_chat(self):
        self.ui.chat_box.clear()

    # ---------- 按钮事件 ----------
    def on_record_button(self):
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
        """退出聊天，重置所有状态，回到情绪检测"""
        if self.is_recording:
            self.voice_collector.stop_recording()
            self.is_recording = False
        self.llm_bot.history_clear()
        self.current_mode = "emotion"
        self.active_emotion = None
        self.pending_strong_emotion = None      # 清空待处理情绪
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
        """录音结束后的通用状态恢复（无有效声音时用）"""
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
            # 使用后台检测的 pending 情绪，不再重复检测
            if self.pending_strong_emotion:
                emotion = self.pending_strong_emotion
                self.pending_strong_emotion = None          # 消费掉
                self._enter_chat_with_emotion(user_text, emotion)
            else:
                self._start_normal_chat(user_text)
        else:
            self._handle_chat_message(user_text)

    # ---------- 后台情绪检测模拟 ----------
    def _simulate_emotion_detection(self):
        """
        定时检测情绪（实际项目中替换为摄像头/声纹等实时检测）
        仅当处于情绪检测模式且没有录音时，才会设置 pending_strong_emotion
        """
        if self.current_mode != "emotion" or self.is_recording:
            return

        # 模拟：10% 概率检测到强烈情绪，其余为 Neutral
        if random.random() < 0.1:
            # 随机一个强烈情绪
            emotion = random.choice(["Happy", "Sad", "Angry", "Surprise", "Fear"])
            # 如果与当前 pending 相同则忽略
            if self.pending_strong_emotion == emotion:
                return

            self.pending_strong_emotion = emotion
            ask_text = f"检测到您似乎{emotion}，愿意和我聊聊吗？"
            print(ask_text)
            self.ui_append_system.emit(ask_text)
            self._play_tts(emotion)
        # 若检测到 Neutral 则不做任何事

    # ---------- 进入聊天模式（带情绪） ----------
    def _enter_chat_with_emotion(self, user_text, emotion):
        """进入带情绪标签的聊天模式（不显示额外提示）"""
        self.current_mode = "chat"
        self.active_emotion = emotion
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)

        enhanced = f"[用户当前情绪：{emotion}] {user_text}"
        self._generate_ai_reply(enhanced)

    # ---------- 进入普通聊天模式 ----------
    def _start_normal_chat(self, user_text):
        """进入普通聊天模式（无情绪标签）"""
        self.current_mode = "chat"
        self.active_emotion = None
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)
        self._generate_ai_reply(user_text)

    # ---------- 聊天模式中的后续对话 ----------
    def _handle_chat_message(self, user_text):
        """聊天模式下的录音处理，根据 active_emotion 决定是否添加标签"""
        self.ui_append_user.emit(user_text)
        if self.active_emotion:
            message = f"[用户当前情绪：{self.active_emotion}] {user_text}"
        else:
            message = user_text
        self._generate_ai_reply(message)

    # ---------- 调用 LLM（子线程） ----------
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
        threading.Thread(target=worker, daemon=True).start()

    # ---------- TTS 播放（占位） ----------
    def _play_tts(self, text):
        match text:
            case "Happy":
                wav = EMOTION_WAV / "happy.wav"
                subprocess.run(
                ["aplay", "-D", "plughw:0,0", wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                )
            case "Sad":
                wav = EMOTION_WAV / "sad.wav"
                subprocess.run(
                ["aplay", "-D", "plughw:0,0", wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                )
            case "Angry":
                wav = EMOTION_WAV / "angry.wav"
                subprocess.run(
                ["aplay", "-D", "plughw:0,0", wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                )
            case "Surprise":
                wav = EMOTION_WAV / "surprise.wav"
                subprocess.run(
                ["aplay", "-D", "plughw:0,0", wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
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