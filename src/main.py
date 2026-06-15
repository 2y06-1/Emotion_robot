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
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# main.py 在 src 目录中，server 目录在项目根目录中。
# 所以这里同时加入 PROJECT_ROOT 和 CONFIG_DIR，保证能导入 server 与 config。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from config import cfg

for path in cfg.PYTHON_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ui import MainWindow
from new_voice_collect import Voice_Collect
from voice_tranform import Voice_Transform
from llm import Ollama_chat
from emotion_prompt import build_robot_system_prompt, normalize_emotion_key, SPECIAL_EMOTIONS
from face_detect import Face_Detect
from emotion_detect import EmotionClassifier

# 给微信小程序提供 WebSocket 实时通信
from server.board_ws import run_ws_server
from server.robot_state import robot_state


def cfg_get(name, default=None):
    """兼容旧版 config.py：config.json 里新加字段没有映射时，使用默认值。"""
    return getattr(cfg, name, default)


class EmotionRobot(QObject):
    # ===== UI 信号 =====
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
        self.active_emotion_confidence = 0
        self.pending_strong_emotion = None
        self.pending_strong_confidence = 0

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
        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0
        self.last_debug_save_time = 0.0
        # 终端只在表情变化时打印一次，避免刷屏。
        self._last_print_emotion = None
        self.no_face_count = 0
        self.last_stable_switch_time = 0.0

        # 情绪关怀提示：只跟“统计事件”绑定，不再跟每一帧识别绑定。
        # 这样小程序统计 +1 时，板端 UI/语音才提示一次，避免随机帧抖动触发播报。
        self.last_emotion_prompt_time = 0.0
        self.emotion_wav_lock = threading.Lock()
        self.emotion_wav_proc = None

        # 特殊情绪锁定：检测到 angry / happy / sad / surprise 后，
        # 锁定一段时间，不允许马上跳到另一个特殊情绪。
        # 例如 surprise 刚触发后，6 秒内不会立刻跳到 happy。
        self.special_emotions = set(SPECIAL_EMOTIONS)
        self.locked_special_emotion = None
        self.locked_special_prob = 0.0
        self.locked_special_until = 0.0
        # 记录刚刚解锁的特殊情绪，避免锁定结束后同一个情绪马上反复重新锁定。
        # 例如 happy 锁定结束后，如果平滑结果仍然是 happy，不会马上再次锁 happy；
        # 只有先离开 happy（变成 neutral/no_face 或另一个特殊情绪）后，才允许再次锁 happy。
        self.last_unlocked_special_emotion = None

        # ===== 连接 UI 信号 =====
        self.ui_clear_chat.connect(self._on_ui_clear_chat)
        self.ui_append_user.connect(self.ui.append_user_message)
        self.ui_append_ai.connect(self.ui.append_ai_message)
        self.ui_append_system.connect(self.ui.append_system_message)
        self.ui_append_emotion.connect(self.ui.append_emotion_message)
        self.ui_set_state_emotion_detecting.connect(self.ui.set_state_emotion_detecting)
        self.ui_set_state_chatting.connect(self.ui.set_state_chatting)
        self.ui_set_state_thinking.connect(self.ui.set_state_thinking)
        self.ui_set_state_speaking.connect(self.ui.set_state_speaking)
        self.ui_set_state_error.connect(self.ui.set_state_error)
        self.ui_set_emotion.connect(self.ui.set_emotion)
        self.ui_show_robot.connect(self.ui.show_robot_ui)
        self.ui_show_chat.connect(self.ui.show_chat_ui)
        self.ui_set_user_face.connect(self.ui.update_user_face)

        self.ui.record_button_clicked.connect(self.on_record_button)
        self.ui.page_changed.connect(self.on_page_changed)
        self.ui.exit_chat_clicked.connect(self.on_exit_chat)
        self.ui.exit_program_clicked.connect(self.on_exit_program)

        # 启动 WebSocket 服务、视觉线程和 TTS 子进程
        self._start_ws_server()
        self._start_vision()
        self._start_tts_process()

        self.tts_start.connect(self._on_tts_start)
        self.tts_stop.connect(self._on_tts_stop)

        if cfg.UI_FULLSCREEN:
            self.ui.showFullScreen()
        else:
            self.ui.show()

        self._play_init_sound()

    # ---------- WebSocket 接口 ----------
    def _start_ws_server(self):
        """只启动 WebSocket 服务。

        小程序端的首页状态、聊天记录、情绪统计、关怀建议全部走 WebSocket。
        这样不会再出现 HTTP 一套数据、WS 一套数据、mock 又一套数据的问题。
        """
        t = threading.Thread(
            target=run_ws_server,
            kwargs={
                "host": cfg.WS_HOST,
                "port": cfg.WS_PORT,
                "push_interval": cfg.WS_PUSH_INTERVAL,
            },
            daemon=True,
        )
        t.start()
        print(f"[WS] 板端 WebSocket 已启动: ws://{cfg.WS_HOST}:{cfg.WS_PORT}", flush=True)

    # ---------- 初始化 ----------
    def _try_set_cap(self, cap, prop, value, name):
        """设置摄像头属性。有些摄像头不支持，失败不影响程序继续运行。"""
        if value is None:
            return
        try:
            cap.set(prop, value)
        except Exception:
            pass

    def _apply_camera_settings(self, cap):
        """解决摄像头画面暗、延迟高：设置分辨率、MJPG、FPS、曝光、增益、亮度。"""
        if cap is None:
            return

        fourcc = str(cfg_get("CAMERA_FOURCC", "MJPG"))
        if fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
            except Exception:
                pass

        self._try_set_cap(cap, cv2.CAP_PROP_FRAME_WIDTH, cfg_get("CAMERA_WIDTH", 640), "WIDTH")
        self._try_set_cap(cap, cv2.CAP_PROP_FRAME_HEIGHT, cfg_get("CAMERA_HEIGHT", 480), "HEIGHT")
        self._try_set_cap(cap, cv2.CAP_PROP_FPS, cfg_get("CAMERA_FPS", 30), "FPS")
        self._try_set_cap(cap, cv2.CAP_PROP_BUFFERSIZE, 1, "BUFFERSIZE")

        # Linux V4L2 下常见：0.25=手动曝光，0.75=自动曝光。
        auto_exp = bool(cfg_get("CAMERA_AUTO_EXPOSURE", True))
        self._try_set_cap(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exp else 0.25, "AUTO_EXPOSURE")
        if not auto_exp:
            self._try_set_cap(cap, cv2.CAP_PROP_EXPOSURE, cfg_get("CAMERA_EXPOSURE", -4), "EXPOSURE")

        self._try_set_cap(cap, cv2.CAP_PROP_GAIN, cfg_get("CAMERA_GAIN", 80), "GAIN")
        self._try_set_cap(cap, cv2.CAP_PROP_BRIGHTNESS, cfg_get("CAMERA_BRIGHTNESS", 150), "BRIGHTNESS")
        self._try_set_cap(cap, cv2.CAP_PROP_CONTRAST, cfg_get("CAMERA_CONTRAST", 128), "CONTRAST")

        # 丢弃前几帧，让自动曝光稳定下来。
        warmup = int(cfg_get("CAMERA_WARMUP_FRAMES", 10))
        for _ in range(max(0, warmup)):
            cap.read()


    def _open_camera_by_device(self, device):
        """更稳的摄像头打开方式：/dev/video20 失败时会再尝试索引 20。"""
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            self._apply_camera_settings(cap)
            return cap

        try:
            cap.release()
        except Exception:
            pass

        if isinstance(device, str):
            match = re.search(r"/dev/video(\d+)$", device)
            if match:
                index = int(match.group(1))
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if cap.isOpened():
                    self._apply_camera_settings(cap)
                    return cap
                try:
                    cap.release()
                except Exception:
                    pass

        return None

    def _save_debug_images(self, display_frame, face_img, face_for_model, emotion, prob, topk_text):
        """保存调试图片：原始画面、人脸裁剪、人脸增强后图像。"""
        if not cfg_get("EMOTION_SAVE_DEBUG", True):
            return

        now = time.time()
        if now - self.last_debug_save_time < float(cfg_get("EMOTION_DEBUG_INTERVAL", 1.0)):
            return
        self.last_debug_save_time = now

        try:
            debug_dir = Path(str(cfg_get("EMOTION_DEBUG_DIR", "/tmp/emotion_debug")))
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((now - int(now)) * 1000)
            prefix = debug_dir / f"{ts}_{ms:03d}_{emotion}_{prob:.2f}"

            cv2.imwrite(str(prefix) + "_frame.jpg", display_frame)
            cv2.imwrite(str(prefix) + "_face_raw.jpg", face_img)
            cv2.imwrite(str(prefix) + "_face_model.jpg", face_for_model)

            info_path = str(prefix) + "_info.txt"
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(topk_text + "\n")
                f.write(f"raw_face_gray_mean={self.emotion_cls.gray_mean(face_img):.2f}\n")
                f.write(f"model_face_gray_mean={self.emotion_cls.gray_mean(face_for_model):.2f}\n")
            print(f"[emotion debug] 已保存: {prefix}_*.jpg", flush=True)
        except Exception as e:
            print(f"[emotion debug] 保存失败: {e}", flush=True)

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
                auto_enhance=cfg_get("EMOTION_AUTO_ENHANCE", True),
                enhance_dark_threshold=cfg_get("EMOTION_ENHANCE_DARK_THRESHOLD", 70),
                gamma=cfg_get("EMOTION_GAMMA", 1.6),
            )

            self.cap = self._open_camera_by_device(cfg.CAMERA_DEVICE)
            if self.cap is None:
                print(f"[视觉] 摄像头打开失败: {cfg.CAMERA_DEVICE}，尝试备用设备: {cfg.CAMERA_FALLBACK}", flush=True)
                self.cap = self._open_camera_by_device(cfg.CAMERA_FALLBACK)

            if self.cap is None or not self.cap.isOpened():
                print("[视觉] 备用摄像头也打开失败", flush=True)
                self.cap = None
            else:
                # 摄像头参数已经在 _apply_camera_settings() 中设置。
                pass

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
        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0
        self.last_stable_switch_time = time.time()
        self._reset_special_emotion_lock()

    def _reset_special_emotion_lock(self, clear_relock_history=True):
        """清空特殊情绪锁定状态。

        clear_relock_history=True 时，连“刚刚解锁过的情绪”也一起清空。
        no_face、退出聊天、重置状态时应该清空；
        但锁定自然结束时不能清空，否则同一个情绪会马上再次锁定。
        """
        self.locked_special_emotion = None
        self.locked_special_prob = 0.0
        self.locked_special_until = 0.0
        if clear_relock_history:
            self.last_unlocked_special_emotion = None

    def _smooth_emotion(self, raw_emotion, prob):
        """二级平滑：先过滤低置信度，再用投票 + 滞回切换。

        这版专门解决你现在终端里 angry / happy / neutral / sad 来回跳的问题：
        - 低置信度结果不进入窗口；
        - neutral 门槛更高，防止模型一抖就回平静；
        - 最近 N 次结果必须达到票数和占比，才可能成为候选；
        - 候选还要连续保持几轮，才真正切换 UI、App、统计状态。
        """
        raw_emotion = str(raw_emotion or "neutral").lower()
        prob = float(prob or 0.0)
        now = time.time()

        emotion_accept_conf = float(cfg_get("EMOTION_MIN_ACCEPT_CONF", 0.42))
        neutral_accept_conf = float(cfg_get("NEUTRAL_MIN_ACCEPT_CONF", 0.76))
        emotion_min_vote_frames = int(cfg_get("EMOTION_MIN_VOTE_FRAMES", 5))
        neutral_min_vote_frames = int(cfg_get("NEUTRAL_MIN_VOTE_FRAMES", 7))
        emotion_switch_frames = int(cfg_get("EMOTION_SWITCH_FRAMES", 3))
        neutral_switch_frames = int(cfg_get("NEUTRAL_SWITCH_FRAMES", 5))
        emotion_min_vote_ratio = float(cfg_get("EMOTION_MIN_VOTE_RATIO", 0.45))
        neutral_min_vote_ratio = float(cfg_get("NEUTRAL_MIN_VOTE_RATIO", 0.58))
        min_hold_seconds = float(cfg_get("EMOTION_MIN_HOLD_SECONDS", 0.9))
        strong_switch_conf = float(cfg_get("EMOTION_STRONG_SWITCH_CONF", 0.78))

        accept_conf = neutral_accept_conf if raw_emotion == "neutral" else emotion_accept_conf
        if prob >= accept_conf:
            self.emotion_window.append((raw_emotion, prob, now))

        if not self.emotion_window:
            return self.last_emotion, self.last_emotion_prob

        window_len = len(self.emotion_window)
        counter = Counter(item[0] for item in self.emotion_window)
        prob_sum = {}
        latest_time = {}
        for emotion, p, t in self.emotion_window:
            prob_sum[emotion] = prob_sum.get(emotion, 0.0) + p
            latest_time[emotion] = t

        candidates = []
        for emotion, count in counter.items():
            avg_prob = prob_sum[emotion] / max(1, count)
            vote_ratio = count / max(1, window_len)
            need_votes = neutral_min_vote_frames if emotion == "neutral" else emotion_min_vote_frames
            need_ratio = neutral_min_vote_ratio if emotion == "neutral" else emotion_min_vote_ratio

            if count >= need_votes and vote_ratio >= need_ratio:
                # 排序优先级：票数、占比、平均置信度、出现时间。
                candidates.append((emotion, count, vote_ratio, avg_prob, latest_time.get(emotion, 0.0)))

        if not candidates:
            # 没有足够证据时，保持上一次稳定状态，不把抖动传给 UI 和 App。
            return self.last_emotion, self.last_emotion_prob

        candidates.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        target_emotion, vote_count, vote_ratio, avg_prob, _ = candidates[0]

        if target_emotion == self.last_emotion:
            self.smooth_candidate_emotion = None
            self.smooth_candidate_count = 0
            self.last_emotion_prob = avg_prob
            return self.last_emotion, self.last_emotion_prob

        # 情绪切换的滞回：目标候选必须连续成为第一候选几次，且当前稳定情绪至少保持一小段时间。
        if target_emotion == self.smooth_candidate_emotion:
            self.smooth_candidate_count += 1
        else:
            self.smooth_candidate_emotion = target_emotion
            self.smooth_candidate_count = 1

        need_switch = neutral_switch_frames if target_emotion == "neutral" else emotion_switch_frames

        # 非平静情绪非常明显时可以略快，但仍然至少要连续 2 次候选。
        if target_emotion != "neutral" and avg_prob >= strong_switch_conf:
            need_switch = max(2, min(need_switch, 2))

        if self.last_emotion != "no_face" and now - self.last_stable_switch_time < min_hold_seconds:
            return self.last_emotion, self.last_emotion_prob

        if self.smooth_candidate_count < need_switch:
            return self.last_emotion, self.last_emotion_prob

        self.last_emotion = target_emotion
        self.last_emotion_prob = avg_prob
        self.last_stable_switch_time = now
        self.smooth_candidate_emotion = None
        self.smooth_candidate_count = 0
        return self.last_emotion, self.last_emotion_prob

    def _apply_special_emotion_lock(self, emotion, prob, allow_start=True):
        """特殊情绪锁定机制。\n\n        目标：\n        1. 检测到 angry / happy / sad / surprise 后，短时间锁定；\n        2. 锁定期间不允许马上跳到另一个特殊情绪；\n        3. 锁定自然结束后，不允许“同一个情绪”立刻再次锁定；\n        4. 只有离开该情绪后，才允许它下一次重新锁定。\n\n        这样可以避免终端一直出现：\n        锁定 happy -> happy 锁定结束 -> 又锁定 happy -> 又结束 -> 又锁定 happy\n        """
        emotion = str(emotion or "neutral").lower()
        prob = float(prob or 0.0)
        now = time.time()

        lock_seconds = float(cfg_get("SPECIAL_EMOTION_LOCK_SECONDS", 6.0))
        lock_min_conf = float(cfg_get("SPECIAL_EMOTION_LOCK_MIN_CONF", cfg_get("STRONG_EMOTION_CONF", 0.58)))
        relock_requires_exit = bool(cfg_get("SPECIAL_EMOTION_RELOCK_REQUIRES_EXIT", True))

        # no_face：直接清空锁定和重锁历史。
        # 没有人脸时，不应该继续保持刚才的特殊情绪。
        if emotion == "no_face":
            if self.locked_special_emotion:
                print(
                    f"[情绪锁定] 检测到 no_face，清除锁定 {self.locked_special_emotion}",
                    flush=True,
                )
            self._reset_special_emotion_lock(clear_relock_history=True)
            return emotion, prob

        # 1. 锁定期内：无论平滑层又输出了什么，都继续保持锁定情绪。
        if self.locked_special_emotion and now < self.locked_special_until:
            locked = self.locked_special_emotion

            # 同一个情绪继续出现时，只更新更高置信度，不重新计时。
            if emotion == locked:
                self.locked_special_prob = max(self.locked_special_prob, prob)

            self.last_emotion = locked
            self.last_emotion_prob = self.locked_special_prob
            return locked, self.locked_special_prob

        # 2. 锁定期自然结束：记录刚刚解锁的情绪，但不要立刻重锁同一个情绪。
        if self.locked_special_emotion and now >= self.locked_special_until:
            old = self.locked_special_emotion
            print(
                f"[情绪锁定] {old} 锁定结束，允许检测下一个情绪",
                flush=True,
            )
            self.last_unlocked_special_emotion = old
            self._reset_special_emotion_lock(clear_relock_history=False)

            # 这里直接返回当前平滑结果，避免在同一轮马上重新锁定 old。
            return emotion, prob

        # 3. neutral：说明已经离开特殊情绪，可以允许以后同一个特殊情绪重新锁定。
        if emotion == "neutral":
            if relock_requires_exit and self.last_unlocked_special_emotion:
                self.last_unlocked_special_emotion = None
            return emotion, prob

        # 4. 非特殊情绪不触发锁定。
        if emotion not in self.special_emotions:
            if relock_requires_exit and self.last_unlocked_special_emotion:
                self.last_unlocked_special_emotion = None
            return emotion, prob

        # 5. 本轮没有新推理时，只维持已有锁，不开启新锁。
        if not allow_start:
            return emotion, prob

        # 6. 置信度不够，不触发锁定。
        if prob < lock_min_conf:
            return emotion, prob

        # 7. 防止同一个情绪刚解锁就马上重新锁。
        # 例如 happy 锁定结束后，如果模型仍然稳定输出 happy，就正常显示 happy，
        # 但不再打印“锁定 happy”，也不会继续阻止其他情绪切换。
        if relock_requires_exit and emotion == self.last_unlocked_special_emotion:
            return emotion, prob

        # 8. 如果是新的特殊情绪，开始锁定。
        self.locked_special_emotion = emotion
        self.locked_special_prob = prob
        self.locked_special_until = now + lock_seconds
        self.last_unlocked_special_emotion = None

        self.last_emotion = emotion
        self.last_emotion_prob = prob

        print(
            f"[情绪锁定] 锁定 {emotion} {lock_seconds:.1f}s，prob={prob:.2f}",
            flush=True,
        )

        return emotion, prob

    def _handle_emotion_count_event(self, event):
        """
        情绪统计事件触发后的关怀提示和预置语音。

        注意：
        这里不能再限制 current_page == "robot"。
        因为第 2 个 UI 是 chat，第 3 个 UI 是 face，
        但这两个页面也应该能同步情绪提示和语音播报。
        """

        if not event:
            return

        # 只有在“情绪检测模式”下才主动关怀。
        # 如果已经进入正式聊天模式，就不要频繁打断用户。
        if self.current_mode != "emotion":
            return

        # 用户正在录音或者机器人正在说话时，不叠加情绪提示音。
        if self.is_recording or self.is_playing_tts:
            return

        now = time.time()
        prompt_min_interval = float(cfg_get("EMOTION_PROMPT_MIN_INTERVAL", 2.0))
        if now - self.last_emotion_prompt_time < prompt_min_interval:
            return

        emotion = str(event.get("emotion", "")).lower()
        if emotion in ("", "neutral", "no_face"):
            return

        self.last_emotion_prompt_time = now
        self.pending_strong_emotion = emotion
        self.pending_strong_confidence = event.get("confidence", 0)

        cn_emotion = event.get("emotion_cn") or {
            "angry": "生气",
            "happy": "开心",
            "neutral": "平静",
            "sad": "难过",
            "surprise": "惊讶",
            "fear": "害怕",
            "disgust": "厌恶",
        }.get(emotion, emotion)

        ask_text = f"检测到您似乎{cn_emotion}，愿意和我聊聊吗？"

        print(
            f"[情绪事件] {emotion} +1，当前页面={self.ui.current_page}，触发提示和语音",
            flush=True
        )

        # 不管当前在第 1 / 第 2 / 第 3 个 UI，都把提示写入聊天区。
        # 如果当前不在聊天页，切到聊天页后也能看到。
        self.ui_append_system.emit(ask_text)

        # 不管当前在哪个页面，只要形成有效情绪事件，就播报。
        self._play_emotion_wav_async(emotion)

    def _handle_no_face(self, display_frame):
        self.no_face_count += 1
        no_face_reset_frames = cfg_get("NO_FACE_RESET_FRAMES", 5)
        if self.no_face_count >= no_face_reset_frames:
            self._reset_emotion_smooth()
            self.ui_set_emotion.emit("no_face", "", False)
            self.ui_set_user_face.emit(display_frame, "no_face", 0.0)

            # 同步给小程序：首页显示未检测到人脸
            robot_state.update_emotion(
                emotion="no_face",
                confidence=0,
                face_detected=False
            )
        else:
            # 短暂丢脸时不马上清空表情，减少 UI 闪烁。
            self.ui_set_emotion.emit(self.last_emotion, f"置信度 {self.last_emotion_prob:.2f}", False)
            self.ui_set_user_face.emit(display_frame, self.last_emotion, self.last_emotion_prob)

            # 小程序也保持上一次状态，但标记为暂时未检测到人脸
            robot_state.update_emotion(
                emotion=self.last_emotion,
                confidence=self.last_emotion_prob,
                face_detected=False
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
                if not boxes:
                    self._handle_no_face(display_frame)
                    time.sleep(cfg.VISION_IDLE_SLEEP)
                    continue

                self.no_face_count = 0
                self._draw_face_boxes(display_frame, boxes)
                faces = self.face_detector.crop(
                    frame,
                    boxes,
                    pad=cfg.FACE_PAD,
                    extra_ratio=cfg.FACE_EXTRA_RATIO,
                )
                main_face = self._select_main_face(faces)
                if main_face is None:
                    self._handle_no_face(display_frame)
                    time.sleep(cfg.VISION_IDLE_SLEEP)
                    continue

                _, _, _, _, _, face_img = main_face

                now = time.time()
                if now - self.last_emotion_infer_time >= cfg.EMOTION_INFER_INTERVAL:
                    self.last_emotion_infer_time = now

                    raw_emotion, prob = self.emotion_cls.predict(face_img)

                    emotion, prob = self._smooth_emotion(raw_emotion, prob)

                    # 特殊情绪锁定：避免刚检测到 surprise，又马上跳到 happy。
                    emotion, prob = self._apply_special_emotion_lock(
                        emotion,
                        prob,
                        allow_start=True,
                    )

                    if emotion != self._last_print_emotion:
                        print(f"[当前表情] {emotion} ({prob:.2f})", flush=True)
                        self._last_print_emotion = emotion

                    ui_tip = f"置信度 {prob:.2f}"
                else:
                    emotion = self.last_emotion
                    prob = self.last_emotion_prob

                    # 没有新推理时，只维持已有锁定，不重新开启锁定。
                    emotion, prob = self._apply_special_emotion_lock(
                        emotion,
                        prob,
                        allow_start=False,
                    )

                    ui_tip = f"置信度 {prob:.2f}"

                strong = emotion != "neutral" and emotion != "no_face" and prob >= cfg.STRONG_EMOTION_CONF
                self.ui_set_emotion.emit(emotion, ui_tip, strong)
                self.ui_set_user_face.emit(display_frame, emotion, prob)

                # 同步当前稳定表情给微信小程序。
                # 如果这次形成了新的“情绪统计事件”，就同步触发板端 UI 提示和预置语音。
                emotion_event = robot_state.update_emotion(
                    emotion=emotion,
                    confidence=prob,
                    face_detected=True
                )
                self._handle_emotion_count_event(emotion_event)

            except Exception as e:
                print(f"[视觉] 单帧处理失败: {e}", flush=True)
                self.last_face_boxes = []
                self._reset_emotion_smooth()
                self.ui_set_emotion.emit("no_face", "", False)
                self.ui_set_user_face.emit(display_frame, "no_face", 0.0)

                # 单帧异常时同步给小程序，避免首页一直停在旧状态
                robot_state.update_emotion(
                    emotion="no_face",
                    confidence=0,
                    face_detected=False
                )

            time.sleep(cfg.VISION_IDLE_SLEEP)

    def _start_vision(self):
        if self.vision_running:
            return
        self.vision_running = True
        self.vision_pause.clear()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _pause_vision(self):
        self.vision_pause.set()

    def _resume_vision(self):
        self.vision_pause.clear()

    def _stop_vision(self):
        self.vision_running = False
        self.vision_pause.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)
        if self.cap and self.cap.isOpened():
            self.cap.release()

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
                if line == "TTS_DONE":
                    break
        except Exception as e:
            if not self._is_task_cancelled(task_id):
                print(f"[TTS] 播放异常: {e}", flush=True)

    def _on_tts_start(self, text, task_id):
        if self._is_task_cancelled(task_id):
            return

        self.is_playing_tts = True
        # AI 文本已经生成，接下来进入语音播报阶段。
        # 这样第一次对话不会在 ASR 文本刚出现时就变回“开始说话”。
        self.ui_set_state_speaking.emit()
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
        self.active_emotion_confidence = 0
        self.pending_strong_emotion = None
        self.pending_strong_confidence = 0
        self._reset_special_emotion_lock()
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

        # 同步用户聊天内容给小程序聊天页
        robot_state.add_chat(
            role="user",
            content=user_text,
            emotion=robot_state.current_emotion
        )

        if len(user_text) < cfg.MIN_TEXT_LEN or not self._is_valid_chinese(user_text):
            self.ui_append_system.emit("未识别到有效文本，请重试")
            self._after_record_reset(task_id)
            return

        if self.current_mode == "emotion":
            if self.pending_strong_emotion:
                emotion = self.pending_strong_emotion
                emotion_confidence = self.pending_strong_confidence
                self.pending_strong_emotion = None
                self.pending_strong_confidence = 0
                self._enter_chat_with_emotion(user_text, emotion, emotion_confidence, task_id)
            else:
                self._start_normal_chat(user_text, task_id)
        else:
            self._handle_chat_message(user_text, task_id)

    def _enter_chat_with_emotion(self, user_text, emotion, emotion_confidence, task_id):
        self.current_mode = "chat"

        # 只有 angry / happy / sad / surprise 作为特殊情绪回复触发条件。
        # 这里保存的是“触发聊天的那次情绪事件”，避免录音时视觉暂停或暂时 no_face 导致置信度变成 0。
        emotion_key = normalize_emotion_key(emotion)
        if emotion_key in SPECIAL_EMOTIONS:
            self.active_emotion = emotion_key
            self.active_emotion_confidence = emotion_confidence
        else:
            self.active_emotion = None
            self.active_emotion_confidence = 0

        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        # 第一次进入聊天页后，AI 还没生成回复，应该保持“思考中”。
        self.ui_set_state_thinking.emit()
        self.ui_append_user.emit(user_text)
        self._generate_ai_reply(user_text, task_id, emotion_override=self.active_emotion)

    def _start_normal_chat(self, user_text, task_id):
        self.current_mode = "chat"
        self.active_emotion = None
        self.active_emotion_confidence = 0
        self.llm_bot.history_clear()
        self.ui_clear_chat.emit()
        self.ui_show_chat.emit()
        # 第一次普通聊天：用户文本显示后，AI 正在生成回复。
        self.ui_set_state_thinking.emit()
        self.ui_append_user.emit(user_text)
        self._generate_ai_reply(user_text, task_id)

    def _handle_chat_message(self, user_text, task_id):
        self.ui_append_user.emit(user_text)
        # 后续对话也统一显示“思考中”，直到 AI 回复生成并开始播报。
        self.ui_set_state_thinking.emit()
        self._generate_ai_reply(user_text, task_id, emotion_override=self.active_emotion)

    # ---------- LLM ----------
    def _build_robot_system_prompt(self, emotion_override=None):
        """构造 system prompt。

        关键变化：
        - 用户原话不再拼进大段 Prompt；
        - 用户原话会作为 user message 单独发给 Ollama；
        - 情绪要求只放在 system message，避免模型把提示词原样复述出来。
        """
        status = robot_state.get_status()

        system_prompt, info = build_robot_system_prompt(
            emotion=status.get("emotion", "neutral"),
            confidence=status.get("confidence", 0),
            face_detected=status.get("face_detected", False),
            active_emotion=emotion_override,
            active_confidence=self.active_emotion_confidence if emotion_override else None,
            min_confidence=50,
        )

        print(
            f"[LLM提示] mode={info.get('mode')}, emotion={info.get('emotion')}, "
            f"confidence={info.get('confidence')}, face_detected={info.get('face_detected')}, "
            f"source={info.get('source')}",
            flush=True,
        )

        return system_prompt

    def _fallback_reply(self, user_text, emotion_override=None):
        """当本地模型复述提示词或输出为空时的兜底回复。"""
        text = str(user_text or "").strip()
        emotion = normalize_emotion_key(emotion_override or robot_state.current_emotion)
        cn = {
            "happy": "开心",
            "angry": "生气或烦躁",
            "sad": "难过",
            "surprise": "惊喜或惊讶",
        }.get(emotion, "平静")

        if any(k in text for k in ["你是谁", "你叫什么", "你是干什么"]):
            return "我是您的智能情感陪伴机器人，可以听您说话，也可以结合您的表情状态给出更贴近心情的回应。"

        if any(k in text for k in ["心情", "情绪", "表情", "状态"]):
            if emotion in SPECIAL_EMOTIONS:
                return f"我感觉您现在更接近{cn}的状态。您可以慢慢和我说，我会认真听。"
            return "我现在没有检测到特别明显的情绪波动，整体看起来比较平静。"

        return "我在这里，可以慢慢和我说。"

    def _generate_ai_reply(self, user_text, task_id, emotion_override=None):
        def worker():
            if self._is_task_cancelled(task_id):
                return
            try:
                system_prompt = self._build_robot_system_prompt(
                    emotion_override=emotion_override,
                )

                reply = self.llm_bot.chat_ollama(
                    user_text,
                    system_prompt=system_prompt,
                ).strip()

                if self._is_task_cancelled(task_id):
                    return
                if not reply:
                    reply = self._fallback_reply(user_text, emotion_override)
            except Exception as e:
                if self._is_task_cancelled(task_id):
                    return
                reply = self._fallback_reply(user_text, emotion_override)
                print(f"[LLM 错误] {e}", flush=True)

            if self._is_task_cancelled(task_id):
                return

            # 同步机器人回复给小程序聊天页
            robot_state.add_chat(
                role="robot",
                content=reply,
                emotion=robot_state.current_emotion
            )

            self.ui_append_ai.emit(reply)
            self.tts_start.emit(reply, task_id)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- 播放预置 WAV ----------
    def _play_emotion_wav_async(self, emotion):
        """异步播放情绪提示音，避免 aplay 阻塞视觉识别线程。"""
        threading.Thread(target=self._play_emotion_wav, args=(emotion,), daemon=True).start()

    def _play_emotion_wav(self, emotion):
        wav_name = emotion.lower() + ".wav"
        wav_path = cfg.EMOTION_WAV_DIR / wav_name
        if not wav_path.exists():
            print(f"[情绪语音] 文件不存在: {wav_path}", flush=True)
            return

        with self.emotion_wav_lock:
            try:
                # 上一个提示音还没播完就不叠加，避免声音混在一起。
                if self.emotion_wav_proc is not None and self.emotion_wav_proc.poll() is None:
                    return

                self.emotion_wav_proc = subprocess.Popen(
                    ["aplay", "-D", cfg.APLAY_DEVICE, str(wav_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[情绪语音] 播放失败: {e}", flush=True)

    def _play_init_sound(self):
        if cfg.INIT_WAV.exists():
            subprocess.run(
                ["aplay", "-D", cfg.APLAY_DEVICE, str(cfg.INIT_WAV)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

    def _is_valid_chinese(self, text):
        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return len(text) > 0 and chinese_count / len(text) >= cfg.VALID_CHINESE_RATIO

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    robot = EmotionRobot()
    robot.run()
