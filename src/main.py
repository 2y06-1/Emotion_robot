import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from PyQt5.QtCore import QObject, pyqtSignal,QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# main.py 位于 src 目录，server 位于项目根目录。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from config import cfg

# 将配置中的模块路径加入 Python 搜索路径。
for module_path in cfg.PYTHON_PATHS:
    module_path_str = str(module_path)

    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

from ui import MainWindow
from monitor.performance_monitor import PerformanceMonitor

from new_voice_collect import Voice_Collect
from voice_tranform import Voice_Transform
from llm import Ollama_chat
from emotion_prompt import (
    SPECIAL_EMOTIONS,
    build_robot_system_prompt,
    normalize_emotion_key,
)
from face_detect import Face_Detect
from emotion_detect import EmotionClassifier
from server.board_ws import run_ws_server
from server.robot_state import robot_state


def cfg_get(name, default=None):
    """兼容旧配置：配置中不存在字段时返回默认值。"""
    return getattr(cfg, name, default)


class EmotionRobot(QObject):
    # =========================================================
    # UI 信号
    # =========================================================
    ui_clear_chat = pyqtSignal()
    ui_append_user = pyqtSignal(str)
    ui_append_ai = pyqtSignal(str)
    ui_append_system = pyqtSignal(str)
    ui_append_emotion = pyqtSignal(str, str)

    ui_set_state_emotion_detecting = pyqtSignal()
    ui_set_state_chatting = pyqtSignal()
    ui_set_state_thinking = pyqtSignal()
    ui_set_state_speaking = pyqtSignal()
    ui_set_state_error = pyqtSignal(str)

    ui_set_emotion = pyqtSignal(str, str, bool)
    ui_show_robot = pyqtSignal()
    ui_show_chat = pyqtSignal()
    ui_set_user_face = pyqtSignal(object, str, float)

    ui_lock_chat_emotion = pyqtSignal(str)
    ui_unlock_chat_emotion = pyqtSignal()

    tts_start = pyqtSignal(str, int)
    tts_stop = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.app = QApplication(sys.argv)
        self.ui = MainWindow()

        self.performance_monitor = PerformanceMonitor(vision_provider="CPU",)
        # =====================================================
        # 核心功能模块
        # =====================================================
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

        # =====================================================
        # 对话与任务状态
        # =====================================================
        self.current_mode = "emotion"
        self.is_recording = False
        self.is_playing_tts = False

        self.active_emotion = None
        self.active_emotion_confidence = 0

        self.pending_strong_emotion = None
        self.pending_strong_confidence = 0

        self.task_lock = threading.Lock()
        self.task_id = 0
        self.cancel_event = threading.Event()
        self.rec_thread = None

        # =====================================================
        # TTS 子进程并发状态
        # =====================================================
        # 进程生命周期锁：只保护 tts_proc、启动状态和代数。
        self.tts_proc = None
        self.tts_process_lock = threading.RLock()
        self.tts_start_cond = threading.Condition(
            self.tts_process_lock
        )
        self.tts_starting = False
        self.tts_generation = 0

        # 通信锁：保证同一时间只有一个线程读写 worker。
        self.tts_io_lock = threading.Lock()

        # =====================================================
        # 视觉模块
        # =====================================================
        self.face_detector = None
        self.emotion_cls = None
        self.cap = None

        self.vision_running = False
        self.vision_pause = threading.Event()
        self.vision_thread = None

        self.last_strong_emotion_time = 0.0
        self.vision_frame_id = 0
        self.last_face_boxes = []

        # 表情平滑状态。
        self.emotion_window = deque(
            maxlen=cfg.EMOTION_SMOOTH_WINDOW
        )
        self.last_emotion_infer_time = 0.0
        self.last_emotion = "no_face"
        self.last_emotion_prob = 0.0

        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0
        self.last_stable_switch_time = 0.0

        self.last_debug_save_time = 0.0
        self._last_print_emotion = None
        self.no_face_count = 0

        # =====================================================
        # 情绪状态并发保护
        # =====================================================
        # UI 线程可能在视觉线程推理期间退出聊天。
        # 使用 RLock 和 epoch，防止旧帧重新写回旧情绪。
        self.emotion_state_lock = threading.RLock()
        self.emotion_state_epoch = 0

        # 情绪关怀提示。
        self.last_emotion_prompt_time = 0.0
        self.emotion_wav_lock = threading.Lock()
        self.emotion_wav_proc = None

        # 特殊情绪锁定。
        self.special_emotions = set(SPECIAL_EMOTIONS)
        self.locked_special_emotion = None
        self.locked_special_prob = 0.0
        self.locked_special_until = 0.0
        self.last_unlocked_special_emotion = None

        # 初始化模型和摄像头。
        self._init_vision_modules()

        # =====================================================
        # 连接 UI 信号
        # =====================================================
        self.ui_clear_chat.connect(self._on_ui_clear_chat)
        self.ui_append_user.connect(
            self.ui.append_user_message
        )
        self.ui_append_ai.connect(
            self.ui.append_ai_message
        )
        self.ui_append_system.connect(
            self.ui.append_system_message
        )
        self.ui_append_emotion.connect(
            self.ui.append_emotion_message
        )

        self.ui_set_state_emotion_detecting.connect(
            self.ui.set_state_emotion_detecting
        )
        self.ui_set_state_chatting.connect(
            self.ui.set_state_chatting
        )
        self.ui_set_state_thinking.connect(
            self.ui.set_state_thinking
        )
        self.ui_set_state_speaking.connect(
            self.ui.set_state_speaking
        )
        self.ui_set_state_error.connect(
            self.ui.set_state_error
        )

        self.ui_set_emotion.connect(
            self.ui.set_emotion
        )
        self.ui_show_robot.connect(
            self.ui.show_robot_ui
        )
        self.ui_show_chat.connect(
            self.ui.show_chat_ui
        )
        self.ui_set_user_face.connect(
            self.ui.update_user_face
        )
        self.ui_lock_chat_emotion.connect(
            self.ui.lock_chat_emotion
        )
        self.ui_unlock_chat_emotion.connect(
            self.ui.unlock_chat_emotion
        )
        self.ui.record_button_clicked.connect(
            self.on_record_button
        )
        self.ui.page_changed.connect(
            self.on_page_changed
        )
        self.ui.exit_chat_clicked.connect(
            self.on_exit_chat
        )
        self.ui.exit_program_clicked.connect(
            self.on_exit_program
        )
        self.performance_timer = QTimer(self)
        self.performance_timer.setInterval(1000)
        self.performance_timer.timeout.connect(
            self._refresh_performance_ui
        )
        self.performance_timer.start()
        # 必须先连接信号，再启动后台服务。
        self.tts_start.connect(self._on_tts_start)
        self.tts_stop.connect(self._on_tts_stop)

        # 启动服务。
        self._start_ws_server()
        self._start_vision()

        # TTS 模型加载放到后台线程，不阻塞 PyQt UI。
        self._start_tts_process_async()

        if cfg.UI_FULLSCREEN:
            self.ui.showFullScreen()
        else:
            self.ui.show()

        self._play_init_sound()
    # =========================================================
    # 性能监控
    # =========================================================
    def _refresh_performance_ui(self):
        """每秒采集系统数据并刷新第4个性能页面。"""
        try:
            self.performance_monitor.sample_system()
            snapshot = self.performance_monitor.snapshot()
            self.ui.update_performance(snapshot)

        except Exception as exc:
            # 性能页面失败不能影响主程序正常运行。
            print(
                f"[性能监控] 刷新失败: {exc}",
                flush=True,
            )
    # =========================================================
    # WebSocket
    # =========================================================
    def _start_ws_server(self):
        """启动板端 WebSocket 服务。"""
        thread = threading.Thread(
            target=run_ws_server,
            kwargs={
                "host": cfg.WS_HOST,
                "port": cfg.WS_PORT,
                "push_interval": cfg.WS_PUSH_INTERVAL,
            },
            daemon=True,
        )
        thread.start()

        print(
            f"[WS] 板端 WebSocket 已启动: "
            f"ws://{cfg.WS_HOST}:{cfg.WS_PORT}",
            flush=True,
        )

    # =========================================================
    # 摄像头与视觉模块初始化
    # =========================================================
    @staticmethod
    def _try_set_cap(cap, prop, value, name):
        del name

        if cap is None or value is None:
            return

        try:
            cap.set(prop, value)
        except Exception:
            pass

    def _apply_camera_settings(self, cap):
        if cap is None:
            return

        fourcc = str(
            cfg_get("CAMERA_FOURCC", "MJPG")
        )

        if fourcc:
            try:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(
                        *fourcc[:4]
                    ),
                )
            except Exception:
                pass

        self._try_set_cap(
            cap,
            cv2.CAP_PROP_FRAME_WIDTH,
            cfg_get("CAMERA_WIDTH", 640),
            "WIDTH",
        )
        self._try_set_cap(
            cap,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cfg_get("CAMERA_HEIGHT", 480),
            "HEIGHT",
        )
        self._try_set_cap(
            cap,
            cv2.CAP_PROP_FPS,
            cfg_get("CAMERA_FPS", 30),
            "FPS",
        )
        self._try_set_cap(
            cap,
            cv2.CAP_PROP_BUFFERSIZE,
            1,
            "BUFFERSIZE",
        )

        auto_exposure = bool(
            cfg_get(
                "CAMERA_AUTO_EXPOSURE",
                True,
            )
        )

        self._try_set_cap(
            cap,
            cv2.CAP_PROP_AUTO_EXPOSURE,
            0.75 if auto_exposure else 0.25,
            "AUTO_EXPOSURE",
        )

        if not auto_exposure:
            self._try_set_cap(
                cap,
                cv2.CAP_PROP_EXPOSURE,
                cfg_get(
                    "CAMERA_EXPOSURE",
                    -4,
                ),
                "EXPOSURE",
            )

        self._try_set_cap(
            cap,
            cv2.CAP_PROP_GAIN,
            cfg_get("CAMERA_GAIN", 80),
            "GAIN",
        )
        self._try_set_cap(
            cap,
            cv2.CAP_PROP_BRIGHTNESS,
            cfg_get(
                "CAMERA_BRIGHTNESS",
                150,
            ),
            "BRIGHTNESS",
        )
        self._try_set_cap(
            cap,
            cv2.CAP_PROP_CONTRAST,
            cfg_get(
                "CAMERA_CONTRAST",
                128,
            ),
            "CONTRAST",
        )

        warmup_frames = int(
            cfg_get(
                "CAMERA_WARMUP_FRAMES",
                10,
            )
        )

        for _ in range(
            max(0, warmup_frames)
        ):
            cap.read()

    def _open_camera_by_device(self, device):
        """先按设备路径打开，再尝试设备索引。"""
        try:
            cap = cv2.VideoCapture(
                device,
                cv2.CAP_V4L2,
            )
        except Exception:
            cap = cv2.VideoCapture(device)

        if (
            cap is not None
            and cap.isOpened()
        ):
            self._apply_camera_settings(cap)
            return cap

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        if isinstance(device, str):
            match = re.search(
                r"/dev/video(\d+)$",
                device,
            )

            if match:
                index = int(match.group(1))

                try:
                    cap = cv2.VideoCapture(
                        index,
                        cv2.CAP_V4L2,
                    )
                except Exception:
                    cap = cv2.VideoCapture(
                        index
                    )

                if cap.isOpened():
                    self._apply_camera_settings(
                        cap
                    )
                    return cap

                try:
                    cap.release()
                except Exception:
                    pass

        return None

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
                class_names=(
                    cfg.EMOTION_CLASS_NAMES
                ),
                mean=cfg.EMOTION_MEAN,
                std=cfg.EMOTION_STD,
                auto_enhance=cfg_get(
                    "EMOTION_AUTO_ENHANCE",
                    True,
                ),
                enhance_dark_threshold=cfg_get(
                    "EMOTION_ENHANCE_DARK_THRESHOLD",
                    70,
                ),
                gamma=cfg_get(
                    "EMOTION_GAMMA",
                    1.6,
                ),
            )

            self.cap = (
                self._open_camera_by_device(
                    cfg.CAMERA_DEVICE
                )
            )

            if self.cap is None:
                print(
                    "[视觉] 主摄像头打开失败: "
                    f"{cfg.CAMERA_DEVICE}，"
                    "尝试备用设备: "
                    f"{cfg.CAMERA_FALLBACK}",
                    flush=True,
                )

                self.cap = (
                    self._open_camera_by_device(
                        cfg.CAMERA_FALLBACK
                    )
                )

            if (
                self.cap is None
                or not self.cap.isOpened()
            ):
                print(
                    "[视觉] 摄像头初始化失败",
                    flush=True,
                )
                self.cap = None

        except Exception as exc:
            print(
                f"[视觉] 初始化失败: {exc}",
                flush=True,
            )
            self.face_detector = None
            self.emotion_cls = None
            self.cap = None

    # =========================================================
    # 情绪状态重置
    # =========================================================
    def _reset_special_emotion_lock(
        self,
        clear_relock_history=True,
    ):
        """清空特殊情绪锁定。"""
        self.locked_special_emotion = None
        self.locked_special_prob = 0.0
        self.locked_special_until = 0.0

        if clear_relock_history:
            self.last_unlocked_special_emotion = (
                None
            )

    def _reset_emotion_smooth(self):
        """
        清空视觉情绪平滑状态。

        该函数不会增加 epoch；普通无人脸重置可以直接调用。
        """
        self.emotion_window.clear()

        self.last_emotion = "no_face"
        self.last_emotion_prob = 0.0

        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0

        self.no_face_count = 0
        self.last_face_boxes = []

        # 下一次恢复视觉后立即重新检测、重新推理。
        self.vision_frame_id = 0
        self.last_emotion_infer_time = 0.0
        self.last_strong_emotion_time = 0.0

        self.last_stable_switch_time = (
            time.time()
        )
        self._last_print_emotion = None

        self._reset_special_emotion_lock(
            clear_relock_history=True,
        )

    def _reset_all_emotion_state(self):
        """
        完全清空一次聊天会话留下的情绪状态。

        epoch 增加后，视觉线程中尚未提交的旧帧会被丢弃。
        """
        with self.emotion_state_lock:
            self.emotion_state_epoch += 1

            self.active_emotion = None
            self.active_emotion_confidence = 0

            self.pending_strong_emotion = None
            self.pending_strong_confidence = 0

            self._reset_emotion_smooth()

            # 防止退出聊天后立即重复上一轮关怀播报。
            self.last_emotion_prompt_time = (
                time.time()
            )

            print(
                "[情绪状态] 已清空情绪上下文、"
                "平滑窗口、旧人脸框和特殊锁",
                flush=True,
            )

    # =========================================================
    # 情绪平滑与特殊锁
    # =========================================================
    def _smooth_emotion(
        self,
        raw_emotion,
        prob,
    ):
        """投票、置信度过滤和滞回切换。"""
        raw_emotion = str(
            raw_emotion or "neutral"
        ).lower()
        prob = float(prob or 0.0)
        now = time.time()

        emotion_accept_conf = float(
            cfg_get(
                "EMOTION_MIN_ACCEPT_CONF",
                0.42,
            )
        )
        neutral_accept_conf = float(
            cfg_get(
                "NEUTRAL_MIN_ACCEPT_CONF",
                0.76,
            )
        )

        emotion_min_vote_frames = int(
            cfg_get(
                "EMOTION_MIN_VOTE_FRAMES",
                5,
            )
        )
        neutral_min_vote_frames = int(
            cfg_get(
                "NEUTRAL_MIN_VOTE_FRAMES",
                7,
            )
        )

        emotion_switch_frames = int(
            cfg_get(
                "EMOTION_SWITCH_FRAMES",
                3,
            )
        )
        neutral_switch_frames = int(
            cfg_get(
                "NEUTRAL_SWITCH_FRAMES",
                5,
            )
        )

        emotion_min_vote_ratio = float(
            cfg_get(
                "EMOTION_MIN_VOTE_RATIO",
                0.45,
            )
        )
        neutral_min_vote_ratio = float(
            cfg_get(
                "NEUTRAL_MIN_VOTE_RATIO",
                0.58,
            )
        )

        min_hold_seconds = float(
            cfg_get(
                "EMOTION_MIN_HOLD_SECONDS",
                0.9,
            )
        )
        strong_switch_conf = float(
            cfg_get(
                "EMOTION_STRONG_SWITCH_CONF",
                0.78,
            )
        )

        accept_conf = (
            neutral_accept_conf
            if raw_emotion == "neutral"
            else emotion_accept_conf
        )

        if prob >= accept_conf:
            self.emotion_window.append(
                (
                    raw_emotion,
                    prob,
                    now,
                )
            )

        if not self.emotion_window:
            return (
                self.last_emotion,
                self.last_emotion_prob,
            )

        window_length = len(
            self.emotion_window
        )
        counter = Counter(
            item[0]
            for item in self.emotion_window
        )

        prob_sum = {}
        latest_time = {}

        for (
            emotion,
            item_prob,
            item_time,
        ) in self.emotion_window:
            prob_sum[emotion] = (
                prob_sum.get(emotion, 0.0)
                + item_prob
            )
            latest_time[emotion] = item_time

        candidates = []

        for emotion, count in counter.items():
            average_prob = (
                prob_sum[emotion]
                / max(1, count)
            )
            vote_ratio = (
                count
                / max(1, window_length)
            )

            required_votes = (
                neutral_min_vote_frames
                if emotion == "neutral"
                else emotion_min_vote_frames
            )
            required_ratio = (
                neutral_min_vote_ratio
                if emotion == "neutral"
                else emotion_min_vote_ratio
            )

            if (
                count >= required_votes
                and vote_ratio
                >= required_ratio
            ):
                candidates.append(
                    (
                        emotion,
                        count,
                        vote_ratio,
                        average_prob,
                        latest_time.get(
                            emotion,
                            0.0,
                        ),
                    )
                )

        if not candidates:
            return (
                self.last_emotion,
                self.last_emotion_prob,
            )

        candidates.sort(
            key=lambda item: (
                item[1],
                item[2],
                item[3],
                item[4],
            ),
            reverse=True,
        )

        (
            target_emotion,
            _vote_count,
            _vote_ratio,
            average_prob,
            _latest,
        ) = candidates[0]

        if (
            target_emotion
            == self.last_emotion
        ):
            self.smooth_candidate_emotion = (
                None
            )
            self.smooth_candidate_count = 0
            self.last_emotion_prob = (
                average_prob
            )

            return (
                self.last_emotion,
                self.last_emotion_prob,
            )

        if (
            target_emotion
            == self.smooth_candidate_emotion
        ):
            self.smooth_candidate_count += 1
        else:
            self.smooth_candidate_emotion = (
                target_emotion
            )
            self.smooth_candidate_count = 1

        required_switch_frames = (
            neutral_switch_frames
            if target_emotion == "neutral"
            else emotion_switch_frames
        )

        if (
            target_emotion != "neutral"
            and average_prob
            >= strong_switch_conf
        ):
            required_switch_frames = max(
                2,
                min(
                    required_switch_frames,
                    2,
                ),
            )

        if (
            self.last_emotion != "no_face"
            and (
                now
                - self.last_stable_switch_time
            )
            < min_hold_seconds
        ):
            return (
                self.last_emotion,
                self.last_emotion_prob,
            )

        if (
            self.smooth_candidate_count
            < required_switch_frames
        ):
            return (
                self.last_emotion,
                self.last_emotion_prob,
            )

        self.last_emotion = target_emotion
        self.last_emotion_prob = average_prob
        self.last_stable_switch_time = now

        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0

        return (
            self.last_emotion,
            self.last_emotion_prob,
        )

    def _apply_special_emotion_lock(
        self,
        emotion,
        prob,
        allow_start=True,
    ):
        """
        锁定 angry、happy、sad、surprise 一段时间，
        防止情绪标签快速来回跳动。
        """
        emotion = str(
            emotion or "neutral"
        ).lower()
        prob = float(prob or 0.0)
        now = time.time()

        lock_seconds = float(
            cfg_get(
                "SPECIAL_EMOTION_LOCK_SECONDS",
                6.0,
            )
        )
        lock_min_conf = float(
            cfg_get(
                "SPECIAL_EMOTION_LOCK_MIN_CONF",
                cfg_get(
                    "STRONG_EMOTION_CONF",
                    0.58,
                ),
            )
        )
        relock_requires_exit = bool(
            cfg_get(
                "SPECIAL_EMOTION_RELOCK_REQUIRES_EXIT",
                True,
            )
        )

        if emotion == "no_face":
            if self.locked_special_emotion:
                print(
                    "[情绪锁定] 检测到 no_face，"
                    "清除锁定 "
                    f"{self.locked_special_emotion}",
                    flush=True,
                )

            self._reset_special_emotion_lock(
                clear_relock_history=True,
            )
            return emotion, prob

        # 锁定期内保持原特殊情绪。
        if (
            self.locked_special_emotion
            and now
            < self.locked_special_until
        ):
            locked_emotion = (
                self.locked_special_emotion
            )

            if emotion == locked_emotion:
                self.locked_special_prob = max(
                    self.locked_special_prob,
                    prob,
                )

            self.last_emotion = (
                locked_emotion
            )
            self.last_emotion_prob = (
                self.locked_special_prob
            )

            return (
                locked_emotion,
                self.locked_special_prob,
            )

        # 锁定自然结束。
        if (
            self.locked_special_emotion
            and now
            >= self.locked_special_until
        ):
            old_emotion = (
                self.locked_special_emotion
            )

            print(
                f"[情绪锁定] {old_emotion} "
                "锁定结束，允许检测下一个情绪",
                flush=True,
            )

            self.last_unlocked_special_emotion = (
                old_emotion
            )
            self._reset_special_emotion_lock(
                clear_relock_history=False,
            )

            return emotion, prob

        if emotion == "neutral":
            if (
                relock_requires_exit
                and self.last_unlocked_special_emotion
            ):
                self.last_unlocked_special_emotion = (
                    None
                )

            return emotion, prob

        if emotion not in self.special_emotions:
            if (
                relock_requires_exit
                and self.last_unlocked_special_emotion
            ):
                self.last_unlocked_special_emotion = (
                    None
                )

            return emotion, prob

        if not allow_start:
            return emotion, prob

        if prob < lock_min_conf:
            return emotion, prob

        if (
            relock_requires_exit
            and emotion
            == self.last_unlocked_special_emotion
        ):
            return emotion, prob

        self.locked_special_emotion = emotion
        self.locked_special_prob = prob
        self.locked_special_until = (
            now + lock_seconds
        )
        self.last_unlocked_special_emotion = (
            None
        )

        self.last_emotion = emotion
        self.last_emotion_prob = prob

        print(
            f"[情绪锁定] 锁定 {emotion} "
            f"{lock_seconds:.1f}s，"
            f"prob={prob:.2f}",
            flush=True,
        )

        return emotion, prob

    # =========================================================
    # 情绪事件、无人脸与视觉显示
    # =========================================================
    def _handle_emotion_count_event(
        self,
        event,
    ):
        if not event:
            return

        if self.current_mode != "emotion":
            return

        if (
            self.is_recording
            or self.is_playing_tts
        ):
            return

        now = time.time()
        prompt_min_interval = float(
            cfg_get(
                "EMOTION_PROMPT_MIN_INTERVAL",
                2.0,
            )
        )

        if (
            now
            - self.last_emotion_prompt_time
            < prompt_min_interval
        ):
            return

        emotion = str(
            event.get("emotion", "")
        ).lower()

        if emotion in (
            "",
            "neutral",
            "no_face",
        ):
            return

        self.last_emotion_prompt_time = now
        self.pending_strong_emotion = emotion
        self.pending_strong_confidence = (
            event.get("confidence", 0)
        )

        chinese_emotion = (
            event.get("emotion_cn")
            or {
                "angry": "生气",
                "happy": "开心",
                "neutral": "平静",
                "sad": "难过",
                "surprise": "惊讶",
                "fear": "害怕",
                "disgust": "厌恶",
            }.get(emotion, emotion)
        )

        ask_text = (
            f"检测到您似乎{chinese_emotion}，"
            "愿意和我聊聊吗？"
        )

        print(
            f"[情绪事件] {emotion} +1，"
            f"当前页面={self.ui.current_page}，"
            "触发提示和语音",
            flush=True,
        )

        self.ui_append_system.emit(ask_text)
        self._play_emotion_wav_async(
            emotion
        )

    def _handle_no_face(
        self,
        display_frame,
        frame_epoch,
    ):
        """
        处理无人脸状态。

        frame_epoch 不属于当前会话时立即丢弃，
        防止退出聊天前的旧帧覆盖清零结果。
        """
        with self.emotion_state_lock:
            if (
                frame_epoch
                != self.emotion_state_epoch
            ):
                return False

            self.no_face_count += 1

            reset_frames = int(
                cfg_get(
                    "NO_FACE_RESET_FRAMES",
                    5,
                )
            )

            if (
                self.no_face_count
                >= reset_frames
            ):
                self._reset_emotion_smooth()

                self.ui_set_emotion.emit(
                    "no_face",
                    "",
                    False,
                )
                self.ui_set_user_face.emit(
                    display_frame,
                    "no_face",
                    0.0,
                )

                robot_state.update_emotion(
                    emotion="no_face",
                    confidence=0,
                    face_detected=False,
                )
            else:
                self.ui_set_emotion.emit(
                    self.last_emotion,
                    (
                        "置信度 "
                        f"{self.last_emotion_prob:.2f}"
                    ),
                    False,
                )
                self.ui_set_user_face.emit(
                    display_frame,
                    self.last_emotion,
                    self.last_emotion_prob,
                )

                robot_state.update_emotion(
                    emotion=self.last_emotion,
                    confidence=(
                        self.last_emotion_prob
                    ),
                    face_detected=False,
                )

        return True

    @staticmethod
    def _draw_face_boxes(
        display_frame,
        boxes,
    ):
        for (
            x1,
            y1,
            x2,
            y2,
            conf,
        ) in boxes:
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (80, 230, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"{conf:.2f}",
                (
                    max(0, x1),
                    max(20, y1 - 6),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 230, 255),
                2,
            )

    @staticmethod
    def _select_main_face(faces):
        if not faces:
            return None

        return max(
            faces,
            key=lambda item: (
                (
                    item[2]
                    - item[0]
                )
                * (
                    item[3]
                    - item[1]
                ),
                item[4],
            ),
        )

    def _vision_loop(self):
        while self.vision_running:
            if self.vision_pause.is_set():
                time.sleep(0.1)
                continue

            with self.emotion_state_lock:
                frame_epoch = (
                    self.emotion_state_epoch
                )

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
                if (
                    self.vision_frame_id
                    % cfg.FACE_DETECT_EVERY
                    == 0
                    or not self.last_face_boxes
                ):
                    detected_boxes = (
                        self.face_detector.detect_face(
                            frame,
                            img_size=(
                                cfg.FACE_IMG_SIZE
                            ),
                            conf_threshold=(
                                cfg.FACE_CONF
                            ),
                            iou_threshold=(
                                cfg.FACE_IOU
                            ),
                        )
                    )

                    with self.emotion_state_lock:
                        if (
                            frame_epoch
                            != self.emotion_state_epoch
                        ):
                            continue

                        self.last_face_boxes = (
                            detected_boxes or []
                        )

                with self.emotion_state_lock:
                    if (
                        frame_epoch
                        != self.emotion_state_epoch
                    ):
                        continue

                    boxes = list(
                        self.last_face_boxes
                        or []
                    )

                if not boxes:
                    self._handle_no_face(
                        display_frame,
                        frame_epoch,
                    )

                    # 本帧已经完成取帧、检测和无人脸状态更新。
                    self.performance_monitor.record_vision_frame()

                    time.sleep(
                        cfg.VISION_IDLE_SLEEP
                    )
                    continue

                with self.emotion_state_lock:
                    if (
                        frame_epoch
                        != self.emotion_state_epoch
                    ):
                        continue

                    self.no_face_count = 0

                self._draw_face_boxes(
                    display_frame,
                    boxes,
                )

                faces = (
                    self.face_detector.crop(
                        frame,
                        boxes,
                        pad=cfg.FACE_PAD,
                        extra_ratio=(
                            cfg.FACE_EXTRA_RATIO
                        ),
                    )
                )

                main_face = (
                    self._select_main_face(
                        faces
                    )
                )

                if main_face is None:
                    self._handle_no_face(
                        display_frame,
                        frame_epoch,
                    )

                    self.performance_monitor.record_vision_frame()

                    time.sleep(
                        cfg.VISION_IDLE_SLEEP
                    )
                    continue

                (
                    _x1,
                    _y1,
                    _x2,
                    _y2,
                    _conf,
                    face_image,
                ) = main_face

                now = time.time()

                if (
                    now
                    - self.last_emotion_infer_time
                    >= cfg.EMOTION_INFER_INTERVAL
                ):
                    # 模型推理放在锁外，避免长时间阻塞 UI。
                    (
                        raw_emotion,
                        raw_prob,
                    ) = self.emotion_cls.predict(
                        face_image
                    )

                    with self.emotion_state_lock:
                        if (
                            frame_epoch
                            != self.emotion_state_epoch
                        ):
                            continue

                        self.last_emotion_infer_time = (
                            now
                        )

                        (
                            emotion,
                            prob,
                        ) = self._smooth_emotion(
                            raw_emotion,
                            raw_prob,
                        )

                        (
                            emotion,
                            prob,
                        ) = (
                            self._apply_special_emotion_lock(
                                emotion,
                                prob,
                                allow_start=True,
                            )
                        )

                        if (
                            emotion
                            != self._last_print_emotion
                        ):
                            print(
                                "[当前表情] "
                                f"{emotion} "
                                f"({prob:.2f})",
                                flush=True,
                            )
                            self._last_print_emotion = (
                                emotion
                            )

                        ui_tip = (
                            f"置信度 {prob:.2f}"
                        )
                else:
                    with self.emotion_state_lock:
                        if (
                            frame_epoch
                            != self.emotion_state_epoch
                        ):
                            continue

                        emotion = (
                            self.last_emotion
                        )
                        prob = (
                            self.last_emotion_prob
                        )

                        (
                            emotion,
                            prob,
                        ) = (
                            self._apply_special_emotion_lock(
                                emotion,
                                prob,
                                allow_start=False,
                            )
                        )

                        ui_tip = (
                            f"置信度 {prob:.2f}"
                        )

                with self.emotion_state_lock:
                    if (
                        frame_epoch
                        != self.emotion_state_epoch
                    ):
                        continue

                    strong = (
                        emotion != "neutral"
                        and emotion != "no_face"
                        and prob
                        >= cfg.STRONG_EMOTION_CONF
                    )

                    self.ui_set_emotion.emit(
                        emotion,
                        ui_tip,
                        strong,
                    )
                    self.ui_set_user_face.emit(
                        display_frame,
                        emotion,
                        prob,
                    )

                    emotion_event = (
                        robot_state.update_emotion(
                            emotion=emotion,
                            confidence=prob,
                            face_detected=True,
                        )
                    )

                self._handle_emotion_count_event(
                    emotion_event
                )
                self.performance_monitor.record_vision_frame()
            except Exception as exc:
                print(
                    "[视觉] 单帧处理失败: "
                    f"{exc}",
                    flush=True,
                )

                with self.emotion_state_lock:
                    if (
                        frame_epoch
                        == self.emotion_state_epoch
                    ):
                        self._reset_emotion_smooth()

                        self.ui_set_emotion.emit(
                            "no_face",
                            "",
                            False,
                        )
                        self.ui_set_user_face.emit(
                            display_frame,
                            "no_face",
                            0.0,
                        )

                        robot_state.update_emotion(
                            emotion="no_face",
                            confidence=0,
                            face_detected=False,
                        )

            time.sleep(
                cfg.VISION_IDLE_SLEEP
            )

    def _start_vision(self):
        if self.vision_running:
            return

        self.vision_running = True
        self.vision_pause.clear()

        self.vision_thread = (
            threading.Thread(
                target=self._vision_loop,
                daemon=True,
            )
        )
        self.vision_thread.start()

    def _pause_vision(self):
        self.vision_pause.set()

    def _resume_vision(self):
        self.vision_pause.clear()

    def _stop_vision(self):
        self.vision_running = False
        self.vision_pause.set()

        if (
            self.vision_thread
            and self.vision_thread.is_alive()
        ):
            self.vision_thread.join(
                timeout=2
            )

        if (
            self.cap
            and self.cap.isOpened()
        ):
            self.cap.release()

    # =========================================================
    # TTS 子进程
    # =========================================================
    def _start_tts_process_async(self):
        """
        在后台线程确保 TTS 已启动。

        该函数本身立即返回，因此不会阻塞 PyQt UI 主线程。
        """
        threading.Thread(
            target=self._start_tts_process,
            name="tts-loader",
            daemon=True,
        ).start()

    def _start_tts_process(self):
        """
        确保 TTS worker 已启动，并返回可用 Popen 对象。

        多个后台线程同时调用时，仅允许一个线程创建进程；
        其他线程等待创建结束后复用同一个进程。
        """
        deadline = (
            time.monotonic() + 60.0
        )

        with self.tts_start_cond:
            while self.tts_starting:
                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:
                    print(
                        "[TTS] 等待模型加载超时",
                        flush=True,
                    )
                    return None

                self.tts_start_cond.wait(
                    timeout=min(
                        0.5,
                        remaining,
                    )
                )

            proc = self.tts_proc

            if (
                proc is not None
                and proc.poll() is None
                and proc.stdin is not None
                and proc.stdout is not None
            ):
                return proc

            self.tts_proc = None
            self.tts_starting = True
            start_generation = (
                self.tts_generation
            )

        proc = None

        try:
            proc = subprocess.Popen(
                [
                    str(
                        cfg.TTS_VENV_PYTHON
                    ),
                    str(
                        cfg.TTS_WORKER_PATH
                    ),
                    "--model-dir",
                    str(
                        cfg.TTS_MODEL_DIR
                    ),
                    "--provider",
                    str(
                        cfg.TTS_PROVIDER
                    ),
                    "--threads",
                    str(
                        cfg.TTS_THREADS
                    ),
                    "--sid",
                    str(cfg.TTS_SID),
                    "--speed",
                    str(cfg.TTS_SPEED),
                    "--silence-scale",
                    str(
                        cfg.TTS_SILENCE_SCALE
                    ),
                    "--aplay-device",
                    str(
                        cfg.APLAY_DEVICE
                    ),
                    "--max-chars",
                    str(
                        cfg.TTS_MAX_CHARS
                    ),
                    "--warmup",
                    (
                        "1"
                        if cfg.TTS_WARMUP
                        else "0"
                    ),
                    "--max-num-sentences",
                    str(
                        cfg.TTS_MAX_NUM_SENTENCES
                    ),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )

            # Popen 返回后立即登记。
            # cleanup 可以在模型加载期间终止该进程。
            with self.tts_process_lock:
                if (
                    start_generation
                    != self.tts_generation
                ):
                    startup_cancelled = True
                else:
                    self.tts_proc = proc
                    self.performance_monitor.set_tts_pid(proc.pid)
                    startup_cancelled = False

            if startup_cancelled:
                self._terminate_tts_process(
                    proc
                )
                return None

            if proc.stdout is None:
                raise RuntimeError(
                    "TTS 子进程 stdout 不可用"
                )

            # 只在后台线程等待 worker 的启动消息。
            while True:
                startup_message = (
                    proc.stdout.readline()
                )

                if not startup_message:
                    raise RuntimeError(
                        "TTS 子进程未返回就绪信息"
                    )

                startup_text = (
                    startup_message.strip()
                )

                if startup_text:
                    print(
                        f"[TTS] {startup_text}",
                        flush=True,
                    )

                if startup_text == "TTS_READY":
                    break

                if proc.poll() is not None:
                    raise RuntimeError(
                        "TTS 子进程加载期间退出，"
                        f"returncode={proc.returncode}"
                    )

            if proc.poll() is not None:
                raise RuntimeError(
                    "TTS 子进程启动后已退出，"
                    f"returncode={proc.returncode}"
                )

            with self.tts_process_lock:
                if (
                    self.tts_proc is not proc
                    or start_generation
                    != self.tts_generation
                ):
                    return None

            return proc

        except Exception as exc:
            with self.tts_process_lock:
                invalidated = (
                    start_generation
                    != self.tts_generation
                )

            # cleanup 主动终止加载时，不打印误导性错误。
            if not invalidated:
                print(
                    "[Main] 无法启动 "
                    f"TTS 子进程: {exc}",
                    flush=True,
                )

            self._terminate_tts_process(
                proc
            )

            with self.tts_process_lock:
                if self.tts_proc is proc:
                    self.tts_proc = None
                    self.performance_monitor.set_tts_pid(None)

            return None

        finally:
            with self.tts_start_cond:
                self.tts_starting = False
                self.tts_start_cond.notify_all()

    @staticmethod
    def _terminate_tts_process(proc):
        """
        终止指定 TTS 进程。

        不直接读取或修改 self.tts_proc，防止并发覆盖。
        """
        if proc is None:
            return

        try:
            if proc.poll() is None:
                os.killpg(
                    os.getpgid(proc.pid),
                    signal.SIGTERM,
                )

                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    os.killpg(
                        os.getpgid(proc.pid),
                        signal.SIGKILL,
                    )
                    proc.wait(timeout=0.5)

        except ProcessLookupError:
            pass

        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=0.5)
            except Exception:
                pass

        finally:
            for stream in (
                proc.stdin,
                proc.stdout,
            ):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    def _cleanup_tts(self):
        """
        安全摘除并终止当前 TTS 进程。

        先把共享指针设为 None，再在锁外等待进程退出。
        """
        with self.tts_start_cond:
            self.tts_generation += 1

            proc = self.tts_proc
            self.tts_proc = None
            self.performance_monitor.set_tts_pid(None)
            self.tts_start_cond.notify_all()

        self._terminate_tts_process(
            proc
        )

    def _restart_tts_process(self):
        """
        终止旧进程，并在后台重新预加载 TTS。

        即使旧进程仍处于加载阶段，新线程也会等待旧启动流程
        退出，然后再创建新进程。
        """
        self._cleanup_tts()
        self._start_tts_process_async()

    # =========================================================
    # 任务取消和 TTS 播放
    # =========================================================
    def _new_task_id(self):
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.clear()

            return self.task_id

    def _cancel_all_running_tasks(self):
        self.performance_monitor.cancel_interaction()
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.set()

        # 不只是忽略旧结果，还要关闭正在读取的 Ollama HTTP 响应，
        # 真正停止模型继续生成。
        try:
            self.llm_bot.cancel_active_request()
        except Exception as exc:
            print(
                f"[LLM] 取消当前请求失败: {exc}",
                flush=True,
            )

        if self.is_recording:
            self.voice_collector.stop_recording()
            self.is_recording = False

        was_playing_tts = (
            self.is_playing_tts
        )
        self.is_playing_tts = False

        if was_playing_tts:
            # 中断播报后在后台重建，下一次不用冷启动。
            self._restart_tts_process()

    def _is_task_cancelled(
        self,
        task_id,
    ):
        return (
            self.cancel_event.is_set()
            or task_id != self.task_id
        )

    def _play_tts_text(
        self,
        text,
        task_id,
    ):
        if self._is_task_cancelled(
            task_id
        ):
            return

        text = str(text or "").strip()

        if not text:
            return

        # 一次完整播报独占 worker 的 stdin/stdout。
        with self.tts_io_lock:
            if self._is_task_cancelled(
                task_id
            ):
                return

            # 本函数运行于后台播放线程。
            # 冷启动等待不会阻塞 UI 主线程。
            proc = self._start_tts_process()

            if (
                self._is_task_cancelled(
                    task_id
                )
                or proc is None
                or proc.poll() is not None
                or proc.stdin is None
                or proc.stdout is None
            ):
                return

            try:
                proc.stdin.write(
                    text + "\n"
                )
                proc.stdin.flush()
                playback_started_received = False
                tts_failed = False
                while True:
                    if (
                        self._is_task_cancelled(
                            task_id
                        )
                    ):
                        self._restart_tts_process()
                        return

                    # 使用局部 proc。
                    # cleanup 清空 self.tts_proc 后也不会
                    # 产生 NoneType.stdout 并发异常。
                    line = (proc.stdout.readline())

                    if not line:
                        # Worker 意外退出，清理本轮未完成计时。
                        self.performance_monitor.cancel_interaction()
                        break

                    line_text = line.strip()

                    if line_text == "TTS_PLAYBACK_STARTED":
                        if not playback_started_received:
                            playback_started_received = True

                            tts_wait_ms, end_to_end_ms = (
                                self.performance_monitor
                                .mark_tts_playback_started()
                            )

                            print(
                                "[性能监控] TTS 播放开始，"
                                f"TTS等待="
                                f"{(tts_wait_ms or 0.0) / 1000.0:.3f}s，"
                                f"端到端="
                                f"{(end_to_end_ms or 0.0) / 1000.0:.3f}s",
                                flush=True,
                            )

                        continue

                    if line_text == "TTS_FAILED":
                        tts_failed = True

                        print(
                            "[TTS Worker] 本轮语音生成或播放失败",
                            flush=True,
                        )

                        continue

                    if line_text == "TTS_DONE":
                        if (
                            tts_failed
                            or not playback_started_received
                        ):
                            # 本轮没有真正进入播放阶段，
                            # 不能写入一条虚假的完整交互数据。
                            self.performance_monitor.cancel_interaction()

                        break

                    if line_text:
                        print(
                            f"[TTS Worker] {line_text}",
                            flush=True,
                        )

            except (
                BrokenPipeError,
                OSError,
                ValueError,
            ) as exc:
                if not self._is_task_cancelled(
                    task_id
                ):
                    print(
                        "[TTS] 播放异常: "
                        f"{exc}",
                        flush=True,
                    )

            except Exception as exc:
                if not self._is_task_cancelled(
                    task_id
                ):
                    print(
                        "[TTS] 未知播放异常: "
                        f"{exc}",
                        flush=True,
                    )

    def _on_tts_start(
        self,
        text,
        task_id,
    ):
        if self._is_task_cancelled(
            task_id
        ):
            return

        self.is_playing_tts = True
        self.ui_set_state_speaking.emit()
        self.ui.record_button.setEnabled(
            False
        )

        def worker():
            self._play_tts_text(
                text,
                task_id,
            )
            self.tts_stop.emit(
                task_id
            )

        threading.Thread(
            target=worker,
            name=f"tts-play-{task_id}",
            daemon=True,
        ).start()

    def _on_tts_stop(
    self,
    task_id,):
        if self._is_task_cancelled(
            task_id
        ):
            return

        self.is_playing_tts = False
        self.ui.record_button.setEnabled(
            True
        )

        if self.current_mode == "emotion":
            # 当前没有进入连续聊天，恢复情绪检测。
            self.ui_set_state_emotion_detecting.emit()
            self._resume_vision()
        else:
            # 已进入连续聊天：
            # 不解除情绪锁，不恢复视觉识别。
            self.ui_set_state_chatting.emit()
            self._pause_vision()

    # =========================================================
    # UI 页面与按钮
    # =========================================================
    def _on_ui_clear_chat(self):
        self.ui.clear_chat()

    def on_page_changed(
        self,
        page,
    ):
        # 只要还处于连续聊天模式，
        # 无论切换到聊天页、性能页还是人脸页，
        # 都不重新启动视觉情绪识别。
        if self.current_mode == "chat":
            self._pause_vision()
            return

        # 当前没有进行连续聊天，可以正常检测情绪。
        if page in (
            "robot",
            "chat",
            "face",
            "performance",
        ):
            self._resume_vision()

    def on_record_button(self):
        if self.is_playing_tts:
            return

        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        # 确定这一轮真正会使用的情绪。
        lock_emotion = "no_face"

        with self.emotion_state_lock:
            if self.current_mode == "emotion":
                # 情绪触发后的第一次聊天，
                # 使用等待处理的明显情绪。
                candidate = normalize_emotion_key(
                    self.pending_strong_emotion
                )

                if candidate in SPECIAL_EMOTIONS:
                    lock_emotion = candidate

            elif self.current_mode == "chat":
                # 已进入连续聊天后，
                # 使用当前对话保持的情绪。
                candidate = normalize_emotion_key(
                    self.active_emotion
                )

                if candidate in SPECIAL_EMOTIONS:
                    lock_emotion = candidate

        # 从用户真正开始录音时锁定显示。
        # 后续视觉结果只更新缓存，不覆盖该标签。
        self.ui_lock_chat_emotion.emit(
            lock_emotion
        )

        # 锁定之后再暂停视觉，避免最后一帧覆盖显示。
        self._pause_vision()

        task_id = self._new_task_id()
        self.is_recording = True

        self.ui.set_state_listening()

        self.rec_thread = threading.Thread(
            target=self._record_thread,
            args=(task_id,),
            daemon=True,
        )
        self.rec_thread.start()

    def _stop_recording(self):
        self.voice_collector.stop_recording()

    def on_exit_chat(self):
        """
        退出聊天并彻底清除上一轮情绪。

        顺序：
        暂停视觉 -> 取消任务 -> 增加 epoch 并清空状态
        -> UI/小程序清零 -> 恢复视觉。
        """
        print(
            "[聊天] 正在退出聊天并重置情绪",
            flush=True,
        )

        self._pause_vision()
        self._cancel_all_running_tasks()
        self.llm_bot.history_clear()

        self.current_mode = "emotion"
        self.is_recording = False
        self.is_playing_tts = False

        self._reset_all_emotion_state()
        self.ui_unlock_chat_emotion.emit()

        
        self.ui_clear_chat.emit()
        self.ui_set_state_emotion_detecting.emit()
        self.ui_set_emotion.emit(
            "no_face",
            "",
            False,
        )
        self.ui_show_robot.emit()

        robot_state.update_emotion(
            emotion="no_face",
            confidence=0,
            face_detected=False,
        )

        print(
            "[聊天] 已退出聊天，情绪标签已清零，"
            "等待重新检测",
            flush=True,
        )

        self._resume_vision()

    def on_exit_program(self):
        reply = QMessageBox.question(
            self.ui,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes
            | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # 程序退出也要先让 LLM 线程失效并关闭 HTTP 响应。
        with self.task_lock:
            self.task_id += 1
            self.cancel_event.set()

        try:
            self.llm_bot.cancel_active_request()
        except Exception as exc:
            print(
                f"[LLM] 退出时取消请求失败: {exc}",
                flush=True,
            )

        if self.is_recording:
            self.voice_collector.stop_recording()

        self._stop_vision()
        self._cleanup_tts()

        with self.emotion_wav_lock:
            if (
                self.emotion_wav_proc
                is not None
                and self.emotion_wav_proc.poll()
                is None
            ):
                try:
                    self.emotion_wav_proc.terminate()
                except Exception:
                    pass
        
        if hasattr(self,"performance_timer",):
            self.performance_timer.stop()
        self.app.quit()

    # =========================================================
    # 录音与 ASR
    # =========================================================
    def _record_thread(
        self,
        task_id,
    ):
        audio_path = None

        try:
            audio_path = (
                self.voice_collector.record_audio(
                    max_duration=(
                        cfg.RECORD_MAX_DURATION
                    )
                )
            )

            if self._is_task_cancelled(
                task_id
            ):
                return

            if not audio_path:
                self.performance_monitor.cancel_interaction()
                self.ui_append_system.emit(
                    "未检测到有效声音"
                )
                self._after_record_reset(
                    task_id
                )
                return

            self.performance_monitor.mark_recording_finished()

            self._process_audio(
                audio_path,
                task_id,
            )

        except Exception as exc:
            self.performance_monitor.cancel_interaction()
            if not self._is_task_cancelled(
                task_id
            ):
                self.ui_append_system.emit(
                    f"录音错误: {exc}"
                )
                self.ui_set_state_error.emit(
                    str(exc)
                )

        finally:
            if (
                audio_path
                and os.path.exists(
                    audio_path
                )
            ):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

            if not self._is_task_cancelled(
                task_id
            ):
                self.is_recording = False

    def _after_record_reset(
        self,
        task_id=None,
    ):
        if (
            task_id is not None
            and self._is_task_cancelled(
                task_id
            )
        ):
            return

        self.is_recording = False

        if self.current_mode == "emotion":
            self.ui_set_state_emotion_detecting.emit()
            self._resume_vision()
        else:
            self.ui_set_state_chatting.emit()

    def _process_audio(
        self,
        audio_path,
        task_id,
    ):
        if self._is_task_cancelled(
            task_id
        ):
            return

        asr_start = time.perf_counter()
        user_text = (
            self.asr.speech_to_text(
                audio_path
            ).strip()
        )
        asr_ms = (time.perf_counter()- asr_start) * 1000.0
        self.performance_monitor.set_asr_latency(asr_ms)

        if self._is_task_cancelled(
            task_id
        ):
            return

        print(
            f"[用户说] {user_text}",
            flush=True,
        )

        robot_state.add_chat(
            role="user",
            content=user_text,
            emotion=(
                robot_state.current_emotion
            ),
        )

        if (
            len(user_text)
            < cfg.MIN_TEXT_LEN
            or not self._is_valid_chinese(
                user_text
            )
        ):
            self.performance_monitor.cancel_interaction()
            self.ui_append_system.emit(
                "未识别到有效文本，请重试"
            )
            self._after_record_reset(
                task_id
            )
            return

        if self.current_mode == "emotion":
            if self.pending_strong_emotion:
                emotion = (
                    self.pending_strong_emotion
                )
                confidence = (
                    self.pending_strong_confidence
                )

                self.pending_strong_emotion = (
                    None
                )
                self.pending_strong_confidence = (
                    0
                )

                self._enter_chat_with_emotion(
                    user_text,
                    emotion,
                    confidence,
                    task_id,
                )
            else:
                self._start_normal_chat(
                    user_text,
                    task_id,
                )
        else:
            self._handle_chat_message(
                user_text,
                task_id,
            )

    def _enter_chat_with_emotion(
        self,
        user_text,
        emotion,
        emotion_confidence,
        task_id,
    ):
        self.current_mode = "chat"

        emotion_key = (
            normalize_emotion_key(
                emotion
            )
        )

        if (
            emotion_key
            in SPECIAL_EMOTIONS
        ):
            self.active_emotion = (
                emotion_key
            )
            self.active_emotion_confidence = (
                emotion_confidence
            )
        else:
            self.active_emotion = None
            self.active_emotion_confidence = (
                0
            )
        self.ui_lock_chat_emotion.emit(self.active_emotion or "no_face")
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        self.ui_set_state_thinking.emit()
        self.ui_append_user.emit(
            user_text
        )

        self._generate_ai_reply(
            user_text,
            task_id,
            emotion_override=(
                self.active_emotion
            ),
        )

    def _start_normal_chat(
        self,
        user_text,
        task_id,
    ):
        self.current_mode = "chat"
        self.active_emotion = None
        self.active_emotion_confidence = 0
        self.ui_lock_chat_emotion.emit("no_face")

        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        self.ui_set_state_thinking.emit()
        self.ui_append_user.emit(
            user_text
        )

        self._generate_ai_reply(
            user_text,
            task_id,
        )

    def _handle_chat_message(
        self,
        user_text,
        task_id,
    ):
        self.ui_lock_chat_emotion.emit(self.active_emotion or "no_face")
        self.ui_append_user.emit(
            user_text
        )
        self.ui_set_state_thinking.emit()

        self._generate_ai_reply(
            user_text,
            task_id,
            emotion_override=(
                self.active_emotion
            ),
        )

    # =========================================================
    # LLM
    # =========================================================
    def _build_robot_system_prompt(
        self,
        emotion_override=None,
    ):
        status = robot_state.get_status()

        system_prompt, info = (
            build_robot_system_prompt(
                emotion=status.get(
                    "emotion",
                    "neutral",
                ),
                confidence=status.get(
                    "confidence",
                    0,
                ),
                face_detected=status.get(
                    "face_detected",
                    False,
                ),
                active_emotion=(
                    emotion_override
                ),
                active_confidence=(
                    self.active_emotion_confidence
                    if emotion_override
                    else None
                ),
                min_confidence=50,
            )
        )

        print(
            "[LLM提示] "
            f"mode={info.get('mode')}, "
            f"emotion={info.get('emotion')}, "
            f"confidence="
            f"{info.get('confidence')}, "
            "face_detected="
            f"{info.get('face_detected')}, "
            f"source={info.get('source')}",
            flush=True,
        )

        return system_prompt

    def _fallback_reply(
        self,
        user_text,
        emotion_override=None,
    ):
        del user_text

        emotion = normalize_emotion_key(
            emotion_override
            or robot_state.current_emotion
        )

        replies = {
            "happy": "这份开心真好，我也替你高兴。",
            "angry": "受了这样的委屈，生气也很正常。",
            "sad": "今天确实很难熬，我会陪着你。",
            "surprise": "这确实让人惊讶，我懂你的感受。",
            "neutral": "我听见你的感受了，会陪着你。",
            "no_face": "最近确实辛苦了，我会陪着你。",
        }

        return replies.get(
            emotion,
            replies["neutral"],
        )

    def _generate_ai_reply(
        self,
        user_text,
        task_id,
        emotion_override=None,
    ):
        def worker():
            if self._is_task_cancelled(
                task_id
            ):
                return

            try:
                system_prompt = (
                    self._build_robot_system_prompt(
                        emotion_override=(
                            emotion_override
                        ),
                    )
                )

                llm_start = time.perf_counter()

                try:
                    reply = (
                        self.llm_bot.chat_ollama(
                            user_message=(
                                user_text
                            ),
                            system_prompt=(
                                system_prompt
                            ),
                        ).strip()
                    )

                finally:
                    # 用户取消的请求不计入最近一次成功交互。
                    if not self._is_task_cancelled(
                        task_id
                    ):
                        llm_ms = (
                            time.perf_counter()
                            - llm_start
                        ) * 1000.0

                        self.performance_monitor.set_llm_latency(
                            llm_ms
                        )

                    if self._is_task_cancelled(
                        task_id
                    ):
                        return

                    if not reply:
                        reply = (
                            self._fallback_reply(
                                user_text=(
                                    user_text
                                ),
                                emotion_override=(
                                    emotion_override
                                ),
                            )
                        )

            except Exception as exc:
                if self._is_task_cancelled(
                    task_id
                ):
                    return

                print(
                    "[LLM 错误] "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                reply = (
                    self._fallback_reply(
                        user_text=user_text,
                        emotion_override=(
                            emotion_override
                        ),
                    )
                )

            if self._is_task_cancelled(
                task_id
            ):
                return

            robot_state.add_chat(
                role="robot",
                content=reply,
                emotion=(
                    robot_state.current_emotion
                ),
            )

            self.ui_append_ai.emit(reply)
            self.performance_monitor.mark_tts_submitted()
            self.tts_start.emit(
                reply,
                task_id,
            )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    # =========================================================
    # 预置情绪提示音与初始化音
    # =========================================================
    def _play_emotion_wav_async(
        self,
        emotion,
    ):
        threading.Thread(
            target=self._play_emotion_wav,
            args=(emotion,),
            daemon=True,
        ).start()

    def _play_emotion_wav(
        self,
        emotion,
    ):
        wav_name = (
            str(emotion).lower()
            + ".wav"
        )
        wav_path = (
            Path(cfg.EMOTION_WAV_DIR)
            / wav_name
        )

        if not wav_path.exists():
            print(
                "[情绪语音] 文件不存在: "
                f"{wav_path}",
                flush=True,
            )
            return

        with self.emotion_wav_lock:
            try:
                if (
                    self.emotion_wav_proc
                    is not None
                    and self.emotion_wav_proc.poll()
                    is None
                ):
                    return

                self.emotion_wav_proc = (
                    subprocess.Popen(
                        [
                            "aplay",
                            "-D",
                            str(
                                cfg.APLAY_DEVICE
                            ),
                            str(wav_path),
                        ],
                        stdout=(
                            subprocess.DEVNULL
                        ),
                        stderr=(
                            subprocess.DEVNULL
                        ),
                    )
                )

            except Exception as exc:
                print(
                    "[情绪语音] 播放失败: "
                    f"{exc}",
                    flush=True,
                )

    def _play_init_sound(self):
        init_wav = Path(cfg.INIT_WAV)

        if not init_wav.exists():
            return

        try:
            subprocess.run(
                [
                    "aplay",
                    "-D",
                    str(cfg.APLAY_DEVICE),
                    str(init_wav),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            print(
                "[初始化语音] 播放失败: "
                f"{exc}",
                flush=True,
            )

    # =========================================================
    # 工具与程序入口
    # =========================================================
    @staticmethod
    def _is_valid_chinese(text):
        text = str(text or "")

        if not text:
            return False

        chinese_count = sum(
            1
            for char in text
            if "\u4e00"
            <= char
            <= "\u9fff"
        )

        return (
            chinese_count
            / len(text)
            >= cfg.VALID_CHINESE_RATIO
        )

    def run(self):
        sys.exit(
            self.app.exec_()
        )


if __name__ == "__main__":
    robot = EmotionRobot()
    robot.run()