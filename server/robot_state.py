# server/robot_state.py
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import time


EMOTION_CN = {
    "happy": "开心",
    "angry": "生气",
    "neutral": "平静",
    "sad": "难过",
    "surprise": "惊讶",
    "fear": "害怕",
    "disgust": "厌恶",
    "no_face": "未检测到人脸",
}

EMOTION_EMOJI = {
    "happy": "😊",
    "angry": "😠",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😮",
    "fear": "😨",
    "disgust": "🤢",
    "no_face": "👤",
}

EMOTION_COLOR = {
    "happy": "#45c474",
    "neutral": "#4f7cff",
    "sad": "#5b6ee1",
    "angry": "#ff4d4f",
    "surprise": "#ffb020",
    "fear": "#8a8f99",
    "disgust": "#7ac943",
    "no_face": "#999999",
}


class RobotState:
    def __init__(self):
        self.lock = Lock()

        # 当前状态，给首页 /api/status 使用
        self.current_emotion = "no_face"
        self.confidence = 0
        self.face_detected = False
        self.last_text = "等待情绪识别结果..."
        self.update_time = self.now_time()

        # 情绪统计，给情绪页 /api/stats 使用
        self.emotion_counts = defaultdict(int)
        self.total_count = 0
        self.last_stats_time = 0.0
        self.stats_interval = 2.0  # 每 2 秒最多统计一次，避免摄像头每帧都累计导致数据过大

        # 聊天记录，给聊天页 /api/chat 使用
        self.chat_history = deque(maxlen=100)

    def now_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def _normalize_emotion(self, emotion):
        if not emotion:
            return "no_face"

        emotion = str(emotion).strip()

        # 兼容你程序里偶尔出现的首字母大写，比如 Happy / Sad
        emotion = emotion.lower()

        if emotion not in EMOTION_CN:
            emotion = "neutral"

        return emotion

    def _normalize_confidence(self, confidence):
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0

        # 兼容 0.92 和 92 两种格式
        if confidence <= 1:
            confidence = int(confidence * 100)
        else:
            confidence = int(confidence)

        if confidence < 0:
            confidence = 0
        if confidence > 100:
            confidence = 100

        return confidence

    def update_emotion(self, emotion, confidence=0, face_detected=True):
        """
        emotion: happy / angry / neutral / sad / surprise / fear / disgust / no_face
        confidence: 可以是 0.92，也可以是 92
        face_detected: 是否检测到人脸
        """
        emotion = self._normalize_emotion(emotion)
        confidence = self._normalize_confidence(confidence)
        face_detected = bool(face_detected) and emotion != "no_face"

        now = time.time()

        with self.lock:
            self.current_emotion = emotion
            self.confidence = confidence
            self.face_detected = face_detected
            self.update_time = self.now_time()

            if emotion == "no_face":
                self.last_text = "当前未检测到人脸。"
            else:
                cn = EMOTION_CN.get(emotion, "平静")
                self.last_text = f"当前检测到用户情绪为：{cn}，置信度 {confidence}%。"

            # no_face 不计入情绪统计；其它情绪按时间间隔采样
            if emotion != "no_face" and now - self.last_stats_time >= self.stats_interval:
                self.emotion_counts[emotion] += 1
                self.total_count += 1
                self.last_stats_time = now

    def add_chat(self, role, content, emotion=None):
        if not content:
            return

        role = role or "user"
        emotion = self._normalize_emotion(emotion or self.current_emotion)

        with self.lock:
            self.chat_history.append({
                "role": role,
                "content": str(content),
                "emotion": emotion,
                "emotion_cn": EMOTION_CN.get(emotion, emotion),
                "time": self.now_time(),
            })

    def get_status(self):
        with self.lock:
            emotion = self.current_emotion
            return {
                "emotion": emotion,
                "emotion_cn": EMOTION_CN.get(emotion, "平静"),
                "emoji": EMOTION_EMOJI.get(emotion, "😐"),
                "confidence": self.confidence,
                "face_detected": self.face_detected,
                "last_text": self.last_text,
                "update_time": self.update_time,
            }

    def get_chat(self):
        with self.lock:
            return list(self.chat_history)

    def get_stats(self):
        with self.lock:
            total = self.total_count

            if total == 0:
                return {
                    "total": 0,
                    "main_emotion": "暂无数据",
                    "trend": "当前还没有足够的情绪数据。",
                    "items": [],
                }

            sorted_items = sorted(
                self.emotion_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            main_emotion = sorted_items[0][0]

            items = []
            for emotion, count in sorted_items:
                percent = int(count * 100 / total)
                items.append({
                    "name": EMOTION_CN.get(emotion, emotion),
                    "key": emotion,
                    "count": count,
                    "percent": percent,
                    "color": EMOTION_COLOR.get(emotion, "#4f7cff"),
                })

            trend = self._make_trend_text(main_emotion)

            return {
                "total": total,
                "main_emotion": EMOTION_CN.get(main_emotion, main_emotion),
                "trend": trend,
                "items": items,
            }

    def _make_trend_text(self, main_emotion):
        if main_emotion == "happy":
            return "最近开心情绪较多，整体状态较好。"
        if main_emotion == "neutral":
            return "最近整体情绪较平稳，可以继续保持正常交流。"
        if main_emotion == "sad":
            return "最近难过情绪出现较多，建议适当陪伴和安抚。"
        if main_emotion == "angry":
            return "最近生气情绪偏多，建议减少刺激并保持安静环境。"
        if main_emotion == "fear":
            return "最近害怕情绪较明显，建议给予安全感和陪伴。"
        if main_emotion == "surprise":
            return "最近惊讶情绪较明显，可以继续观察用户状态。"
        if main_emotion == "disgust":
            return "最近厌恶情绪较明显，建议检查当前环境或交流内容是否让用户不适。"
        return "最近情绪有一定波动，建议继续观察。"

    def get_alerts(self):
        with self.lock:
            status = self.current_emotion
            confidence = self.confidence

        if status == "sad":
            return [
                {
                    "level": "warning",
                    "tag": "需要关注",
                    "title": "检测到难过情绪",
                    "content": "建议使用温和语气陪伴用户，可以播放舒缓音乐或提醒用户休息。",
                }
            ]

        if status == "angry":
            return [
                {
                    "level": "danger",
                    "tag": "情绪波动",
                    "title": "检测到生气情绪",
                    "content": "建议降低机器人音量，减少主动打扰，等待用户情绪稳定。",
                }
            ]

        if status == "fear":
            return [
                {
                    "level": "warning",
                    "tag": "需要安抚",
                    "title": "检测到害怕情绪",
                    "content": "建议机器人使用安抚性话术，并保持陪伴状态。",
                }
            ]

        if status == "disgust":
            return [
                {
                    "level": "warning",
                    "tag": "需要调整",
                    "title": "检测到厌恶情绪",
                    "content": "建议检查当前声音、画面、环境或对话内容，减少可能引起用户反感的刺激。",
                }
            ]

        if status == "no_face":
            return [
                {
                    "level": "normal",
                    "tag": "等待检测",
                    "title": "暂未检测到人脸",
                    "content": "请确认摄像头朝向、距离和光照环境是否正常。",
                }
            ]

        return [
            {
                "level": "normal",
                "tag": "状态良好",
                "title": "当前情绪状态较稳定",
                "content": f"当前情绪识别置信度为 {confidence}%，可以保持正常交流。",
            }
        ]

    def reset_stats(self):
        with self.lock:
            self.emotion_counts.clear()
            self.total_count = 0
            self.last_stats_time = 0.0

    def clear_chat(self):
        with self.lock:
            self.chat_history.clear()


robot_state = RobotState()
