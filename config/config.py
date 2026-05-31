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

    @staticmethod
    def _default_config_path():
        # 当前目录结构：
        # Emotion_robot/config/config.py
        # Emotion_robot/config/config.json
        return Path(__file__).resolve().parent / "config.json"

    def _resolve_project_root(self):
        value = self.data.get("project", {}).get("root")
        if value:
            return Path(value).expanduser().resolve()
        # config.json 在 Emotion_robot/config/ 下，项目根目录是它的上一级。
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
        self.MIRROR_CAMERA = bool(vision["mirror_camera"])
        self.VISION_IDLE_SLEEP = float(vision["idle_sleep"])

        self.FACE_MODEL_PATH = self._path(vision["face_model_path"])
        self.FACE_PROVIDER = str(vision["face_provider"])
        self.FACE_THREADS = int(vision["face_threads"])
        self.FACE_IMG_SIZE = int(vision["face_img_size"])
        self.FACE_CONF = float(vision["face_conf"])
        self.FACE_IOU = float(vision["face_iou"])
        self.FACE_PAD = int(vision["face_pad"])
        self.FACE_EXTRA_RATIO = float(vision["face_extra_ratio"])
        self.FACE_DETECT_EVERY = int(vision["face_detect_every"])

        self.EMOTION_MODEL_PATH = self._path(vision["emotion_model_path"])
        self.EMOTION_PROVIDER = str(vision["emotion_provider"])
        self.EMOTION_THREADS = int(vision["emotion_threads"])
        self.EMOTION_IMG_SIZE = int(vision["emotion_img_size"])
        self.EMOTION_TOP_K = int(vision["emotion_top_k"])
        self.EMOTION_CLASS_NAMES = list(vision["emotion_class_names"])
        self.EMOTION_MEAN = list(vision["emotion_mean"])
        self.EMOTION_STD = list(vision["emotion_std"])
        self.EMOTION_INFER_INTERVAL = float(vision["emotion_infer_interval"])
        self.EMOTION_SMOOTH_WINDOW = int(vision["emotion_smooth_window"])
        self.EMOTION_MIN_ACCEPT_CONF = float(vision["emotion_min_accept_conf"])
        self.EMOTION_MIN_VOTE_FRAMES = int(vision["emotion_min_vote_frames"])
        self.STRONG_EMOTION_CONF = float(vision["strong_emotion_conf"])
        self.STRONG_EMOTION_COOLDOWN = float(vision["strong_emotion_cooldown"])

    def _load_chat(self):
        chat = self.data["chat"]
        self.MIN_TEXT_LEN = int(chat["min_text_len"])
        self.VALID_CHINESE_RATIO = float(chat["valid_chinese_ratio"])

    def _load_ui(self):
        ui = self.data.get("ui", {})
        self.UI_FULLSCREEN = bool(ui.get("fullscreen", True))


cfg = AppConfig()
