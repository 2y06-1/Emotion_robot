import json
from pathlib import Path


class AppConfig:
    """
    配置读取层。

    规则：
    - config.json 只存配置数据。
    - config.py 只负责读取 json，并把配置整理成 main.py 好调用的属性。
    - 功能模块不读取 config.json，也不写死真实路径。
    """

    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.PROJECT_ROOT = self._resolve_project_root()
        self._load_project()
        self._load_audio()
        self._load_asr()
        self._load_llm()
        self._load_tts()
        self._load_vision()
        self._load_chat()
        self._load_ui()
        self._load_server()

    @staticmethod
    def _default_config_path():
        return Path(__file__).resolve().parent / "config.json"

    def _resolve_project_root(self):
        value = self.data.get("project", {}).get("root")
        if value:
            return Path(value).expanduser().resolve()
        return self.config_path.resolve().parent.parent

    def _path(self, value):
        p = Path(str(value)).expanduser()
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p

    def _load_project(self):
        project = self.data.get("project", {})
        self.PYTHON_PATHS = [
            self._path(p)
            for p in project.get("python_paths", [])
        ]

    def _load_audio(self):
        audio = self.data["audio"]
        self.VOICE_PATH = self._path(audio["voice_path"])
        self.DEVICE_ID = int(audio["device_id"])
        self.MAX_KEEP_FILES = int(audio["max_keep_files"])
        self.VOICE_THRESHOLD = int(audio["voice_threshold"])
        self.MIN_VOICE_SEC = int(audio["min_voice_sec"])
        self.CHANNELS = int(audio["channels"])
        self.CHUNK_SIZE = int(audio["chunk_size"])
        self.DTYPE = str(audio["dtype"])
        self.RECORD_MAX_DURATION = float(audio["record_max_duration"])

        self.APLAY_DEVICE = str(audio["aplay_device"])
        self.INIT_WAV = self._path(audio["init_wav"])
        self.EMOTION_WAV_DIR = self._path(audio["emotion_wav_dir"])

    def _load_asr(self):
        asr = self.data["asr"]
        self.ASR_PROJECT_ROOT = self._path(asr["project_root"])

    def _load_llm(self):
        llm = self.data["llm"]
        self.OLLAMA_URL = str(llm["base_url"])
        self.MODEL_NAME = str(llm["model_name"])
        self.CHAT_HISTORY = self._path(llm["chat_history"])
        self.LLM_STREAM = bool(llm["stream"])
        timeout = llm.get("timeout")
        self.LLM_TIMEOUT = None if timeout is None else float(timeout)

    def _load_tts(self):
        tts = self.data["tts"]
        self.TTS_VENV_PYTHON = self._path(tts["venv_python"])
        self.TTS_WORKER_PATH = self._path(tts["worker_path"])
        self.TTS_MODEL_DIR = self._path(tts["model_dir"])
        self.TTS_PROVIDER = str(tts["provider"])
        self.TTS_THREADS = int(tts["threads"])
        self.TTS_SID = int(tts["sid"])
        self.TTS_SPEED = float(tts["speed"])
        self.TTS_SILENCE_SCALE = float(tts["silence_scale"])
        self.TTS_MAX_CHARS = int(tts["max_chars"])
        self.TTS_WARMUP = bool(tts["warmup"])
        self.TTS_MAX_NUM_SENTENCES = int(tts["max_num_sentences"])

    def _load_vision(self):
        vision = self.data["vision"]
        self.CAMERA_DEVICE = vision["camera_device"]
        self.CAMERA_FALLBACK = vision.get("camera_fallback")
        self.MIRROR_CAMERA = bool(vision.get("mirror_camera", True))
        self.VISION_IDLE_SLEEP = float(vision.get("idle_sleep", 0.02))

        # 摄像头画面参数：用于解决板子画面偏暗、延迟高的问题
        self.CAMERA_WIDTH = int(vision.get("camera_width", 640))
        self.CAMERA_HEIGHT = int(vision.get("camera_height", 480))
        self.CAMERA_FPS = int(vision.get("camera_fps", 30))
        self.CAMERA_FOURCC = str(vision.get("camera_fourcc", "MJPG"))
        self.CAMERA_WARMUP_FRAMES = int(vision.get("camera_warmup_frames", 10))
        self.CAMERA_AUTO_EXPOSURE = bool(vision.get("camera_auto_exposure", True))
        self.CAMERA_EXPOSURE = vision.get("camera_exposure", -4)
        self.CAMERA_GAIN = vision.get("camera_gain", 80)
        self.CAMERA_BRIGHTNESS = vision.get("camera_brightness", 150)
        self.CAMERA_CONTRAST = vision.get("camera_contrast", 128)

        self.FACE_MODEL_PATH = self._path(vision["face_model_path"])
        self.FACE_PROVIDER = str(vision.get("face_provider", "auto"))
        self.FACE_THREADS = int(vision.get("face_threads", 4))
        self.FACE_IMG_SIZE = int(vision.get("face_img_size", 224))
        self.FACE_CONF = float(vision.get("face_conf", 0.25))
        self.FACE_IOU = float(vision.get("face_iou", 0.45))
        self.FACE_PAD = int(vision.get("face_pad", 10))
        self.FACE_EXTRA_RATIO = float(vision.get("face_extra_ratio", 0.18))
        self.FACE_DETECT_EVERY = int(vision.get("face_detect_every", 1))

        self.EMOTION_MODEL_PATH = self._path(vision["emotion_model_path"])
        self.EMOTION_PROVIDER = str(vision.get("emotion_provider", "cpu"))
        self.EMOTION_THREADS = int(vision.get("emotion_threads", 4))
        self.EMOTION_IMG_SIZE = int(vision.get("emotion_img_size", 96))
        self.EMOTION_TOP_K = int(vision.get("emotion_top_k", 5))
        self.EMOTION_CLASS_NAMES = list(vision["emotion_class_names"])
        self.EMOTION_MEAN = list(vision.get("emotion_mean", [0.485, 0.456, 0.406]))
        self.EMOTION_STD = list(vision.get("emotion_std", [0.229, 0.224, 0.225]))

        self.EMOTION_INFER_INTERVAL = float(vision.get("emotion_infer_interval", 0.12))
        self.EMOTION_SMOOTH_WINDOW = int(vision.get("emotion_smooth_window", 5))
        self.EMOTION_MIN_ACCEPT_CONF = float(vision.get("emotion_min_accept_conf", 0.20))
        self.NEUTRAL_MIN_ACCEPT_CONF = float(vision.get("neutral_min_accept_conf", 0.70))
        self.EMOTION_MIN_VOTE_FRAMES = int(vision.get("emotion_min_vote_frames", 5))
        self.NEUTRAL_MIN_VOTE_FRAMES = int(vision.get("neutral_min_vote_frames", 7))
        self.EMOTION_MIN_VOTE_RATIO = float(vision.get("emotion_min_vote_ratio", 0.45))
        self.NEUTRAL_MIN_VOTE_RATIO = float(vision.get("neutral_min_vote_ratio", 0.58))
        self.EMOTION_SWITCH_FRAMES = int(vision.get("emotion_switch_frames", 3))
        self.NEUTRAL_SWITCH_FRAMES = int(vision.get("neutral_switch_frames", 5))
        self.EMOTION_MIN_HOLD_SECONDS = float(vision.get("emotion_min_hold_seconds", 0.9))
        self.EMOTION_STRONG_SWITCH_CONF = float(vision.get("emotion_strong_switch_conf", 0.78))
        self.STRONG_EMOTION_CONF = float(vision.get("strong_emotion_conf", 0.58))
        self.SPECIAL_EMOTION_LOCK_SECONDS = float(vision.get("special_emotion_lock_seconds", 6.0))
        self.SPECIAL_EMOTION_LOCK_MIN_CONF = float(vision.get("special_emotion_lock_min_conf", self.STRONG_EMOTION_CONF))
        self.SPECIAL_EMOTION_RELOCK_REQUIRES_EXIT = bool(vision.get("special_emotion_relock_requires_exit", True))
        self.STRONG_EMOTION_COOLDOWN = float(vision.get("strong_emotion_cooldown", 8.0))
        self.EMOTION_PROMPT_MIN_INTERVAL = float(vision.get("emotion_prompt_min_interval", 2.0))
        self.NO_FACE_RESET_FRAMES = int(vision.get("no_face_reset_frames", 8))
        self.EMOTION_SAVE_DEBUG = bool(vision.get("emotion_save_debug", False))
        self.EMOTION_DEBUG_INTERVAL = float(vision.get("emotion_debug_interval", 1.0))
        self.EMOTION_DEBUG_DIR = str(vision.get("emotion_debug_dir", "/tmp/emotion_debug"))
        self.EMOTION_AUTO_ENHANCE = bool(vision.get("emotion_auto_enhance", True))
        self.EMOTION_ENHANCE_DARK_THRESHOLD = float(vision.get("emotion_enhance_dark_threshold", 70))
        self.EMOTION_GAMMA = float(vision.get("emotion_gamma", 1.6))

    def _load_chat(self):
        chat = self.data["chat"]
        self.MIN_TEXT_LEN = int(chat["min_text_len"])
        self.VALID_CHINESE_RATIO = float(chat["valid_chinese_ratio"])

    def _load_ui(self):
        ui = self.data.get("ui", {})
        self.UI_FULLSCREEN = bool(ui.get("fullscreen", True))

    def _load_server(self):
        server = self.data.get("server", {})
        self.WS_HOST = str(server.get("ws_host", "0.0.0.0"))
        self.WS_PORT = int(server.get("ws_port", 8765))
        self.WS_PUSH_INTERVAL = float(server.get("ws_push_interval", 0.5))


cfg = AppConfig()
