import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

# 当前目录结构：
# Emotion_robot/
# ├── config/config.py
# └── src/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from config import cfg

# 先把模块目录加入 sys.path，再导入各功能模块。
for path in cfg.PYTHON_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ui import MainWindow
from new_voice_collect import Voice_Collect
from voice_tranform import Voice_Transform
from llm import Ollama_chat
from face_detect import Face_Detect
from emotion_detect import EmotionClassifier


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
    ui_set_emotion = pyqtSignal(str, str, bool)
    ui_show_robot = pyqtSignal()
    ui_show_chat = pyqtSignal()
    ui_set_user_face = pyqtSignal(object, str, float)

    tts_start = pyqtSignal(str, int)
    tts_stop = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.app = QApplication(sys.argv)
        self.ui = MainWindow()

        # ===== 核心模块：所有配置从 cfg 来，功能模块本身不读取 json =====
        self.voice_collector = Voice_Collect(
            voice_path=cfg.VOICE_PATH,
            device_id=cfg.DEVICE_ID,
            max_keep_files=cfg.MAX_KEEP_FILES,
            voice_threshold=cfg.VOICE_THRESHOLD,
            min_voice_sec=cfg.MIN_VOICE_SEC,
            channels=cfg.CHANNELS,
            chunk_size=cfg.CHUNK_SIZE,
            dtype=cfg.DTYPE,
        )

        self.asr = Voice_Transform(
            project_root=cfg.ASR_PROJECT_ROOT,
        )

        self.llm_bot = Ollama_chat(
            base_url=cfg.OLLAMA_URL,
            model_name=cfg.MODEL_NAME,
            txt_path=cfg.CHAT_HISTORY,
            stream=cfg.LLM_STREAM,
            timeout=cfg.LLM_TIMEOUT,
        )

        # ===== 状态 =====
        self.current_mode = "emotion"
        self.is_recording = False
        self.is_playing_tts = False
        self.active_emotion = None
        self.pending_strong_emotion = None

        self.task_lock = threading.Lock()
        self.task_id = 0
        self.cancel_event = threading.Event()

        # ===== 视觉模块 =====
        self.face_detector = None
        self.emotion_cls = None
        self.cap = None
        self._init_vision_modules()

        self.vision_running = False
        self.vision_pause = threading.Event()
        self.vision_thread = None
        self.last_strong_emotion_time = 0.0
        self.vision_frame_id = 0
        self.last_face_boxes = []

        # 表情识别平滑状态
        self.emotion_window = deque(maxlen=cfg.EMOTION_SMOOTH_WINDOW)
        self.last_emotion_infer_time = 0.0
        self.last_emotion = "no_face"
        self.last_emotion_prob = 0.0

        # ===== 连接 UI 信号 =====
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

        # 启动视觉线程和 TTS 子进程
        self._start_vision()
        self._start_tts_process()

        self.tts_start.connect(self._on_tts_start)
        self.tts_stop.connect(self._on_tts_stop)

        if cfg.UI_FULLSCREEN:
            self.ui.showFullScreen()
        else:
            self.ui.show()

        self._play_init_sound()

    # ---------- 初始化 ----------
    def _init_vision_modules(self):
        try:
            self.face_detector = Face_Detect(
                model_path=cfg.FACE_MODEL_PATH,
                provider=cfg.FACE_PROVIDER,
                threads=cfg.FACE_THREADS,
            )

            self.emotion_cls = EmotionClassifier(
                model_path=cfg.EMOTION_MODEL_PATH,
                img_size=cfg.EMOTION_IMG_SIZE,
                top_k=cfg.EMOTION_TOP_K,
                threads=cfg.EMOTION_THREADS,
                provider=cfg.EMOTION_PROVIDER,
                class_names=cfg.EMOTION_CLASS_NAMES,
                mean=cfg.EMOTION_MEAN,
                std=cfg.EMOTION_STD,
            )

            self.cap = cv2.VideoCapture(cfg.CAMERA_DEVICE, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"[视觉] 摄像头打开失败: {cfg.CAMERA_DEVICE}，尝试备用设备: {cfg.CAMERA_FALLBACK}", flush=True)
                self.cap = cv2.VideoCapture(cfg.CAMERA_FALLBACK)

            if not self.cap.isOpened():
                print("[视觉] 备用摄像头也打开失败", flush=True)
                self.cap = None

        except Exception as e:
            print(f"[视觉] 初始化失败: {e}", flush=True)
            self.face_detector = None
            self.emotion_cls = None
            self.cap = None

    # ---------- 视觉线程 ----------
    def _select_main_face(self, faces):
        if not faces:
            return None
        return max(
            faces,
            key=lambda item: ((item[2] - item[0]) * (item[3] - item[1]), item[4]),
        )

    def _reset_emotion_smooth(self):
        self.emotion_window.clear()
        self.last_emotion = "no_face"
        self.last_emotion_prob = 0.0

    def _smooth_emotion(self, emotion, prob):
        if prob >= cfg.EMOTION_MIN_ACCEPT_CONF:
            self.emotion_window.append(emotion)

        if len(self.emotion_window) >= cfg.EMOTION_MIN_VOTE_FRAMES:
            smooth_emotion = Counter(self.emotion_window).most_common(1)[0][0]
        else:
            smooth_emotion = emotion

        self.last_emotion = smooth_emotion
        self.last_emotion_prob = prob
        return smooth_emotion, prob

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

            if cfg.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            self.vision_frame_id += 1

            try:
                if self.vision_frame_id % cfg.FACE_DETECT_EVERY == 0 or not self.last_face_boxes:
                    self.last_face_boxes = self.face_detector.detect_face(
                        frame,
                        img_size=cfg.FACE_IMG_SIZE,
                        conf_threshold=cfg.FACE_CONF,
                        iou_threshold=cfg.FACE_IOU,
                    )

                boxes = self.last_face_boxes or []
                if boxes:
                    self._draw_face_boxes(display_frame, boxes)
                    faces = self.face_detector.crop(
                        frame,
                        boxes,
                        pad=cfg.FACE_PAD,
                        extra_ratio=cfg.FACE_EXTRA_RATIO,
                    )
                    main_face = self._select_main_face(faces)
                    if main_face is not None:
                        _, _, _, _, _, face_img = main_face

                        now = time.time()
                        if now - self.last_emotion_infer_time >= cfg.EMOTION_INFER_INTERVAL:
                            self.last_emotion_infer_time = now
                            emotion, prob = self.emotion_cls.predict(face_img)
                            emotion, prob = self._smooth_emotion(emotion, prob)
                        else:
                            emotion = self.last_emotion
                            prob = self.last_emotion_prob

                        strong = emotion != "neutral" and prob >= cfg.STRONG_EMOTION_CONF
                        self.ui_set_emotion.emit(emotion, f"置信度 {prob:.2f}", strong)
                        self.ui_set_user_face.emit(display_frame, emotion, prob)

                        if strong and self.current_mode == "emotion" and self.ui.current_page == "robot":
                            now = time.time()
                            if now - self.last_strong_emotion_time > cfg.STRONG_EMOTION_COOLDOWN:
                                self.last_strong_emotion_time = now
                                self.pending_strong_emotion = emotion.capitalize()
                                cn_emotion = {
                                    "angry": "生气",
                                    "happy": "开心",
                                    "neutral": "平静",
                                    "sad": "难过",
                                    "surprise": "惊讶",
                                    "fear": "害怕",
                                    "disgust": "厌恶",
                                }.get(emotion, emotion)
                                ask_text = f"检测到您似乎{cn_emotion}，愿意和我聊聊吗？"
                                print(ask_text, flush=True)
                                self.ui_append_system.emit(ask_text)
                                self._play_emotion_wav(emotion)
                    else:
                        self._reset_emotion_smooth()
                        self.ui_set_emotion.emit("no_face", "", False)
                        self.ui_set_user_face.emit(display_frame, "no_face", 0.0)
                else:
                    self._reset_emotion_smooth()
                    self.ui_set_emotion.emit("no_face", "", False)
                    self.ui_set_user_face.emit(display_frame, "no_face", 0.0)

            except Exception as e:
                print(f"[视觉] 单帧处理失败: {e}", flush=True)
                self.last_face_boxes = []
                self._reset_emotion_smooth()
                self.ui_set_emotion.emit("no_face", "", False)
                self.ui_set_user_face.emit(display_frame, "no_face", 0.0)

            time.sleep(cfg.VISION_IDLE_SLEEP)

    def _start_vision(self):
        if self.vision_running:
            return
        self.vision_running = True
        self.vision_pause.clear()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        print("[视觉] 线程已启动", flush=True)

    def _pause_vision(self):
        self.vision_pause.set()
        print("[视觉] 已暂停", flush=True)

    def _resume_vision(self):
        self.vision_pause.clear()
        print("[视觉] 已恢复", flush=True)

    def _stop_vision(self):
        self.vision_running = False
        self.vision_pause.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[视觉] 线程已停止", flush=True)

    # ---------- TTS 子进程管理 ----------
    def _start_tts_process(self):
        try:
            self.tts_proc = subprocess.Popen(
                [
                    str(cfg.TTS_VENV_PYTHON),
                    str(cfg.TTS_WORKER_PATH),
                    "--model-dir", str(cfg.TTS_MODEL_DIR),
                    "--provider", cfg.TTS_PROVIDER,
                    "--threads", str(cfg.TTS_THREADS),
                    "--sid", str(cfg.TTS_SID),
                    "--speed", str(cfg.TTS_SPEED),
                    "--silence-scale", str(cfg.TTS_SILENCE_SCALE),
                    "--aplay-device", cfg.APLAY_DEVICE,
                    "--max-chars", str(cfg.TTS_MAX_CHARS),
                    "--warmup", "1" if cfg.TTS_WARMUP else "0",
                    "--max-num-sentences", str(cfg.TTS_MAX_NUM_SENTENCES),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
            startup = self.tts_proc.stdout.readline()
            print(f"[TTS] {startup.strip()}", flush=True)
            print("[Main] TTS 子进程已启动", flush=True)
        except Exception as e:
            print(f"[Main] 无法启动 TTS 子进程: {e}", flush=True)
            self.tts_proc = None

    def _cleanup_tts(self):
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
        self._cleanup_tts()

    def _new_task_id(self):
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.clear()
            return self.task_id

    def _cancel_all_running_tasks(self):
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.set()

        if self.is_recording:
            self.voice_collector.stop_recording()
            self.is_recording = False

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

    # ---------- UI 页面/按钮 ----------
    def _on_ui_clear_chat(self):
        self.ui.clear_chat()

    def on_page_changed(self, page):
        if page == "robot":
            self._resume_vision()
        elif page == "face":
            self._resume_vision()
        elif page == "chat":
            if self.current_mode == "chat" and not self.is_recording:
                self._pause_vision()

    def on_record_button(self):
        if self.is_playing_tts:
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._pause_vision()
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
        self.ui_show_robot.emit()
        self._resume_vision()

    def on_exit_program(self):
        reply = QMessageBox.question(
            self.ui,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
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
            audio_path = self.voice_collector.record_audio(max_duration=cfg.RECORD_MAX_DURATION)
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
            self.ui_set_state_emotion_detecting.emit()
            self._resume_vision()
        else:
            self.ui_set_state_chatting.emit()

    # ---------- 音频处理 ----------
    def _process_audio(self, audio_path, task_id):
        if self._is_task_cancelled(task_id):
            return

        user_text = self.asr.speech_to_text(audio_path).strip()
        if self._is_task_cancelled(task_id):
            return

        print(f"[用户说] {user_text}", flush=True)
        if len(user_text) < cfg.MIN_TEXT_LEN or not self._is_valid_chinese(user_text):
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

    # ---------- LLM ----------
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
                print(f"[LLM 错误] {e}", flush=True)

            print(f"[AI 回复] {reply}", flush=True)
            if self._is_task_cancelled(task_id):
                return
            self.ui_append_ai.emit(reply)
            self.tts_start.emit(reply, task_id)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- 播放预置 WAV ----------
    def _play_emotion_wav(self, emotion):
        wav_name = emotion.lower() + ".wav"
        wav_path = cfg.EMOTION_WAV_DIR / wav_name
        if wav_path.exists():
            subprocess.run(
                ["aplay", "-D", cfg.APLAY_DEVICE, str(wav_path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            print(f"[情绪语音] 文件不存在: {wav_path}", flush=True)

    def _play_init_sound(self):
        if cfg.INIT_WAV.exists():
            subprocess.run(
                ["aplay", "-D", cfg.APLAY_DEVICE, str(cfg.INIT_WAV)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

    # ---------- 辅助 ----------
    def _is_valid_chinese(self, text):
        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return len(text) > 0 and chinese_count / len(text) >= cfg.VALID_CHINESE_RATIO

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    robot = EmotionRobot()
    robot.run()
