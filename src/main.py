import sys
import os
import time
import threading
import subprocess
import signal
from pathlib import Path
import tempfile
import cv2

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal

# ===== 路径配置 =====
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "src" / "asr"))
sys.path.append(str(BASE_DIR / "src" / "llm"))
sys.path.append(str(BASE_DIR / "src" / "ui"))
sys.path.append(str(BASE_DIR / "src" / "vision" / "version3"))  # 如果视觉模块在 vision 目录，否则根据实际调整

from ui import MainWindow
from new_voice_collect import Voice_Collect
from voice_tranform import Voice_Transform
from llm import Ollama_chat

# 视觉模块
from face_detect import Face_Detect
from emotion_detect import EmotionClassifier

# ========== 全局配置 ==========
VOICE_PATH = Path(tempfile.gettempdir()) / "emotion_robot_wav"
DEVICE_ID = 2
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "my_test:latest"
CHAT_HISTORY = BASE_DIR / "model" / "llm" / "chat_history.txt"
INIT_WAV = BASE_DIR / "wav" / "init.wav"
EMOTION_WAV = BASE_DIR / "wav"

# TTS 子进程配置
TTS_VENV_PYTHON = "/home/bianbu/Emotion_robot/venv_tts/bin/python"
TTS_WORKER_PATH = "/home/bianbu/Emotion_robot/src/asr/tts_worker.py"

# 视觉模型路径
# 新版 face_detect.py 默认人脸模型也是 best.onnx；新版 emotion_detect.py 默认是 emotion_best_uint8_static.onnx。
# 这里显式配置，方便 main.py 统一管理。
FACE_MODEL_PATH = BASE_DIR / "model" / "vision" / "best.onnx"
EMOTION_MODEL_PATH = BASE_DIR / "model" / "vision" / "emotion_best_uint8_static.onnx"
CAMERA_DEVICE = "/dev/video20"   # 如果失败会回退到 0

# 新版视觉模块参数
FACE_PROVIDER = "auto"            # auto: 优先 SpaceMITExecutionProvider，没有则 CPU
FACE_THREADS = 4
EMOTION_THREADS = 4
FACE_IMG_SIZE = 224
FACE_CONF = 0.5
FACE_IOU = 0.4
FACE_PAD = 20
FACE_EXTRA_RATIO = 0.25
FACE_DETECT_EVERY = 3             # 每 3 帧做人脸检测，其余帧复用上次框，降低 CPU 占用
MIRROR_CAMERA = True              # 机器人交互一般使用镜像画面；不需要可改 False
STRONG_EMOTION_CONF = 0.70
VISION_IDLE_SLEEP = 0.03

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
    ui_set_emotion = pyqtSignal(str, str, bool)   # 支持 strong 标记
    ui_show_robot = pyqtSignal()                  # 切回第一个 UI
    ui_show_chat = pyqtSignal()                   # 切到第二个 UI，预留给主流程主动切换
    ui_set_user_face = pyqtSignal(object, str, float)  # 第三个 UI：人脸画面、情绪、置信度

    tts_start = pyqtSignal(str, int)
    tts_stop = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.ui = MainWindow()

        # 核心模块
        self.voice_collector = Voice_Collect(VOICE_PATH, DEVICE_ID)
        self.asr = Voice_Transform()
        self.llm_bot = Ollama_chat(OLLAMA_URL, MODEL_NAME, CHAT_HISTORY)

        # 状态
        self.current_mode = "emotion"
        self.is_recording = False
        self.is_playing_tts = False
        self.active_emotion = None
        self.pending_strong_emotion = None

        # 取消控制：点击“结束聊天”后，录音/ASR/LLM/TTS 的旧任务全部失效
        self.task_lock = threading.Lock()
        self.task_id = 0
        self.cancel_event = threading.Event()

        # 视觉模块：接入新版 face_detect.py / emotion_detect.py
        self.face_detector = None
        self.emotion_cls = None
        self.cap = None
        try:
            self.face_detector = Face_Detect(
                str(FACE_MODEL_PATH),
                provider=FACE_PROVIDER,
                threads=FACE_THREADS,
            )
            self.emotion_cls = EmotionClassifier(
                str(EMOTION_MODEL_PATH),
                img_size=64,
                top_k=1,
                threads=EMOTION_THREADS,
            )
            self.cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print("[视觉] 摄像头打开失败，尝试备用设备(0)")
                self.cap = cv2.VideoCapture(0)
        except Exception as e:
            print(f"[视觉] 初始化失败: {e}")
            self.cap = None

        self.vision_running = False
        self.vision_pause = threading.Event()
        self.vision_thread = None
        self.last_strong_emotion_time = 0.0
        self.vision_frame_id = 0
        self.last_face_boxes = []

        # 连接信号
        self.ui_clear_chat.connect(self._on_ui_clear_chat)
        self.ui_append_user.connect(self.ui.append_user_message)
        self.ui_append_ai.connect(self.ui.append_ai_message)
        self.ui_append_system.connect(self.ui.append_system_message)
        self.ui_append_emotion.connect(self.ui.append_emotion_message)
        self.ui_set_state_emotion_detecting.connect(self.ui.set_state_emotion_detecting)
        self.ui_set_state_chatting.connect(self.ui.set_state_chatting)
        self.ui_set_state_error.connect(self.ui.set_state_error)
        self.ui_set_emotion.connect(self.ui.set_emotion)
        self.ui_show_robot.connect(self.ui.show_robot_ui)
        self.ui_show_chat.connect(self.ui.show_chat_ui)
        self.ui_set_user_face.connect(self.ui.update_user_face)

        self.ui.record_button_clicked.connect(self.on_record_button)
        self.ui.page_changed.connect(self.on_page_changed)
        self.ui.exit_chat_clicked.connect(self.on_exit_chat)
        self.ui.exit_program_clicked.connect(self.on_exit_program)

        # 启动视觉
        self._start_vision()

        # TTS 子进程
        self._start_tts_process()

        self.tts_start.connect(self._on_tts_start)
        self.tts_stop.connect(self._on_tts_stop)

        self.ui.showFullScreen()
        self._play_init_sound()

    # ---------- 视觉线程 ----------
    def _select_main_face(self, faces):
        """从新版 Face_Detect.crop() 返回的人脸列表中选择最大/最可靠的一张。"""
        if not faces:
            return None
        return max(
            faces,
            key=lambda item: ((item[2] - item[0]) * (item[3] - item[1]), item[4])
        )

    def _draw_face_boxes(self, display_frame, boxes):
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (80, 230, 255), 2)
            cv2.putText(
                display_frame,
                f"{conf:.2f}",
                (max(0, x1), max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 230, 255),
                2,
            )

    def _vision_loop(self):
        """视觉线程：用新版 face_detect 检脸，用新版 emotion_detect 对裁剪后的人脸分类。"""
        while self.vision_running:
            if self.vision_pause.is_set():
                time.sleep(0.1)
                continue

            if (
                self.cap is None
                or not self.cap.isOpened()
                or self.face_detector is None
                or self.emotion_cls is None
            ):
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            self.vision_frame_id += 1

            try:
                # 新版人脸检测模型相对耗时，所以不是每帧都 detect，复用上一轮 boxes。
                if self.vision_frame_id % FACE_DETECT_EVERY == 0 or not self.last_face_boxes:
                    self.last_face_boxes = self.face_detector.detect_face(
                        frame,
                        img_size=FACE_IMG_SIZE,
                        conf_threshold=FACE_CONF,
                        iou_threshold=FACE_IOU,
                    )

                boxes = self.last_face_boxes or []
                if boxes:
                    self._draw_face_boxes(display_frame, boxes)
                    faces = self.face_detector.crop(
                        frame,
                        boxes,
                        pad=FACE_PAD,
                        extra_ratio=FACE_EXTRA_RATIO,
                    )
                    main_face = self._select_main_face(faces)
                    if main_face is not None:
                        _, _, _, _, _, face_img = main_face
                        emotion, prob = self.emotion_cls.predict(face_img)
                        strong = (emotion != "neutral" and prob >= STRONG_EMOTION_CONF)

                        self.ui_set_emotion.emit(emotion, f"置信度 {prob:.2f}", strong)
                        self.ui_set_user_face.emit(display_frame, emotion, prob)

                        # 只有待机/第一个 UI 下，强情绪才触发主动问候；聊天页和第三页只更新显示。
                        if strong and self.current_mode == "emotion" and self.ui.current_page == "robot":
                            now = time.time()
                            if now - self.last_strong_emotion_time > 5.0:
                                self.last_strong_emotion_time = now
                                self.pending_strong_emotion = emotion.capitalize()
                                cn_emotion = {
                                    "angry": "生气",
                                    "happy": "开心",
                                    "sad": "难过",
                                    "surprise": "惊讶",
                                }.get(emotion, emotion)
                                ask_text = f"检测到您似乎{cn_emotion}，愿意和我聊聊吗？"
                                print(ask_text)
                                self.ui_append_system.emit(ask_text)
                                self._play_emotion_wav(emotion)
                    else:
                        self.ui_set_emotion.emit("no_face", "", False)
                        self.ui_set_user_face.emit(display_frame, "no_face", 0.0)
                else:
                    self.ui_set_emotion.emit("no_face", "", False)
                    self.ui_set_user_face.emit(display_frame, "no_face", 0.0)
            except Exception as e:
                # 视觉线程不能因为单帧异常退出，否则 UI 后续不再更新。
                print(f"[视觉] 单帧处理失败: {e}")
                self.last_face_boxes = []
                self.ui_set_emotion.emit("no_face", "", False)
                self.ui_set_user_face.emit(display_frame, "no_face", 0.0)

            time.sleep(VISION_IDLE_SLEEP)

    def _start_vision(self):
        if self.vision_running:
            return
        self.vision_running = True
        self.vision_pause.clear()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        print("[视觉] 线程已启动")

    def _pause_vision(self):
        self.vision_pause.set()
        print("[视觉] 已暂停")

    def _resume_vision(self):
        self.vision_pause.clear()
        print("[视觉] 已恢复")

    def _stop_vision(self):
        self.vision_running = False
        self.vision_pause.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[视觉] 线程已停止")

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
                preexec_fn=os.setsid,   # 单独进程组，方便结束聊天时连同 aplay 一起终止
            )
            startup = self.tts_proc.stdout.readline()
            print(f"[TTS] {startup.strip()}", flush=True)
            print("[Main] TTS 子进程已启动", flush=True)
        except Exception as e:
            print(f"[Main] 无法启动 TTS 子进程: {e}", flush=True)
            self.tts_proc = None

    def _cleanup_tts(self):
        """终止 TTS worker。
        worker 内部会调用 aplay 播放音频，所以这里杀整个进程组，避免退出聊天后还在播报。
        """
        if self.tts_proc:
            try:
                os.killpg(os.getpgid(self.tts_proc.pid), signal.SIGTERM)
                self.tts_proc.wait(timeout=1.5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.tts_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.tts_proc = None

    def _restart_tts_process(self):
        # 只有“正在播报时被取消”才会走这里。
        # 普通结束聊天不会调用它，因此 TTS 模型不会反复初始化。
        self._cleanup_tts()

    def _new_task_id(self):
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.clear()
            return self.task_id

    def _cancel_all_running_tasks(self):
        """点击结束聊天时调用：让录音/ASR/LLM/TTS 的旧流程全部失效。"""
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.set()

        if self.is_recording:
            self.voice_collector.stop_recording()
            self.is_recording = False

        # 不要每次退出聊天都杀 TTS worker，否则下次播报会重新加载 TTS 模型。
        # 只有当前确实正在播报时，才终止 TTS 进程组来强制停声。
        was_playing_tts = self.is_playing_tts
        self.is_playing_tts = False
        if was_playing_tts:
            self._cleanup_tts()

    def _is_task_cancelled(self, task_id):
        return self.cancel_event.is_set() or task_id != self.task_id

    def _play_tts_text(self, text, task_id):
        if self._is_task_cancelled(task_id):
            return
        if not text.strip():
            return
        if self.tts_proc is None or self.tts_proc.stdin is None:
            self._start_tts_process()
        if self._is_task_cancelled(task_id) or self.tts_proc is None or self.tts_proc.stdin is None:
            return
        try:
            self.tts_proc.stdin.write(text.strip() + "\n")
            self.tts_proc.stdin.flush()
            while True:
                if self._is_task_cancelled(task_id):
                    self._restart_tts_process()
                    return
                line = self.tts_proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                print(line, flush=True)
                if line == "TTS_DONE":
                    break
        except Exception as e:
            if not self._is_task_cancelled(task_id):
                print(f"[TTS] 播放异常: {e}", flush=True)

    def _on_tts_start(self, text, task_id):
        if self._is_task_cancelled(task_id):
            return
        self.is_playing_tts = True
        self.ui.record_button.setEnabled(False)
        def worker():
            self._play_tts_text(text, task_id)
            self.tts_stop.emit(task_id)
        threading.Thread(target=worker, daemon=True).start()

    def _on_tts_stop(self, task_id):
        if self._is_task_cancelled(task_id):
            return
        self.is_playing_tts = False
        self.ui.record_button.setEnabled(True)
        if self.current_mode == "emotion":
            self.ui_set_state_emotion_detecting.emit()
        else:
            self.ui_set_state_chatting.emit()

    def _on_ui_clear_chat(self):
        self.ui.clear_chat()

    def on_page_changed(self, page):
        """UI 页面切换时控制视觉资源。
        - 第一个 UI：需要情绪检测驱动机器人眼睛，所以恢复视觉。
        - 第二个 UI：正常聊天时暂停视觉，释放资源。
        - 第三个 UI：需要显示人脸窗口和用户表情，所以临时恢复视觉。
        """
        if page == "robot":
            self._resume_vision()
        elif page == "face":
            self._resume_vision()
        elif page == "chat":
            # 未正式进入聊天时可以继续待机检测；已经在聊天中则暂停，节省资源。
            if self.current_mode == "chat" and not self.is_recording:
                self._pause_vision()

    # ---------- 按钮事件 ----------
    def on_record_button(self):
        if self.is_playing_tts:
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._pause_vision()          # 暂停视觉，释放 CPU
        task_id = self._new_task_id()
        self.is_recording = True
        self.ui.set_state_listening()
        self.rec_thread = threading.Thread(target=self._record_thread, args=(task_id,), daemon=True)
        self.rec_thread.start()

    def _stop_recording(self):
        self.voice_collector.stop_recording()

    def on_exit_chat(self):
        self._cancel_all_running_tasks()
        self.llm_bot.history_clear()
        self.current_mode = "emotion"
        self.active_emotion = None
        self.pending_strong_emotion = None
        self.ui_clear_chat.emit()
        self.ui_set_state_emotion_detecting.emit()
        self.ui_set_emotion.emit("no_face", "", False)
        self.ui_show_robot.emit()      # 第二个 UI 点击结束聊天后切回第一个 UI
        self._resume_vision()          # 恢复视觉

    def on_exit_program(self):
        reply = QMessageBox.question(
            self.ui, "确认退出", "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.is_recording:
                self.voice_collector.stop_recording()
            self._stop_vision()
            self._cleanup_tts()
            self.app.quit()

    # ---------- 录音线程 ----------
    def _record_thread(self, task_id):
        try:
            audio_path = self.voice_collector.record_audio(max_duration=60)
            if self._is_task_cancelled(task_id):
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                return
            if not audio_path:
                self.ui_append_system.emit("未检测到有效声音")
                self._after_record_reset(task_id)
                return
            self._process_audio(audio_path, task_id)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            if not self._is_task_cancelled(task_id):
                self.ui_append_system.emit(f"录音错误: {e}")
                self.ui_set_state_error.emit(str(e))
        finally:
            if not self._is_task_cancelled(task_id):
                self.is_recording = False

    def _after_record_reset(self, task_id=None):
        if task_id is not None and self._is_task_cancelled(task_id):
            return
        if self.current_mode == "emotion":
            # 未进入正式聊天时恢复情绪检测；界面仍保持当前页，由 UI 状态决定按钮显示。
            self.ui_set_state_emotion_detecting.emit()
            self._resume_vision()
        else:
            self.ui_set_state_chatting.emit()

    # ---------- 音频处理分派 ----------
    def _process_audio(self, audio_path, task_id):
        if self._is_task_cancelled(task_id):
            return
        user_text = self.asr.speech_to_text(audio_path).strip()
        if self._is_task_cancelled(task_id):
            return
        print(f"[用户说] {user_text}")
        if len(user_text) < 2 or not self._is_valid_chinese(user_text):
            self.ui_append_system.emit("未识别到有效文本，请重试")
            self._after_record_reset(task_id)
            return

        if self.current_mode == "emotion":
            if self.pending_strong_emotion:
                emotion = self.pending_strong_emotion
                self.pending_strong_emotion = None
                self._enter_chat_with_emotion(user_text, emotion, task_id)
            else:
                self._start_normal_chat(user_text, task_id)
        else:
            self._handle_chat_message(user_text, task_id)

    # ---------- 进入聊天模式 ----------
    def _enter_chat_with_emotion(self, user_text, emotion, task_id):
        self.current_mode = "chat"
        self.active_emotion = emotion
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)
        enhanced = f"[用户当前情绪：{emotion}] {user_text}"
        self._generate_ai_reply(enhanced, task_id)

    def _start_normal_chat(self, user_text, task_id):
        self.current_mode = "chat"
        self.active_emotion = None
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        self.ui_set_state_chatting.emit()
        self.ui_append_user.emit(user_text)
        self._generate_ai_reply(user_text, task_id)

    def _handle_chat_message(self, user_text, task_id):
        self.ui_append_user.emit(user_text)
        if self.active_emotion:
            message = f"[用户当前情绪：{self.active_emotion}] {user_text}"
        else:
            message = user_text
        self._generate_ai_reply(message, task_id)

    # ---------- LLM 生成 ----------
    def _generate_ai_reply(self, message, task_id):
        def worker():
            if self._is_task_cancelled(task_id):
                return
            try:
                reply = self.llm_bot.chat_ollama(message).strip()
                if self._is_task_cancelled(task_id):
                    print("[LLM] 本轮聊天已结束，丢弃旧回复", flush=True)
                    return
                if not reply:
                    reply = "我好像没听清，可以再说一遍吗？"
            except Exception as e:
                if self._is_task_cancelled(task_id):
                    return
                reply = "抱歉，我现在脑子有点乱，请稍后再试。"
                print(f"[LLM 错误] {e}")
            print(f"[AI 回复] {reply}")
            if self._is_task_cancelled(task_id):
                return
            self.ui_append_ai.emit(reply)
            self.tts_start.emit(reply, task_id)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- 播放预置情绪 WAV ----------
    def _play_emotion_wav(self, emotion):
        wav_name = emotion.lower() + ".wav"
        wav_path = EMOTION_WAV / wav_name
        if wav_path.exists():
            subprocess.run(
                ["aplay", "-D", "plughw:0,0", str(wav_path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        else:
            print(f"[情绪语音] 文件不存在: {wav_path}")

    # ---------- 辅助 ----------
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