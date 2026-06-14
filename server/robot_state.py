# server/robot_state.py
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from threading import Lock
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
    """板端统一状态中心。

    注意：
    - 当前情绪状态会实时更新，用于首页展示。
    - 情绪统计只统计明显的非 neutral 情绪。
    - App 端只展示这里的统计结果，不再自己二次计数。
    """

    def __init__(self):
        self.lock = Lock()

        # 当前状态，给 WebSocket status 使用
        self.current_emotion = "no_face"
        self.confidence = 0
        self.face_detected = False
        self.last_text = "等待情绪识别结果..."
        self.update_time = self.now_time()

        # 情绪统计，给 WebSocket stats 使用
        self.emotion_counts = defaultdict(int)
        self.total_count = 0

        # 统计策略：neutral/no_face 不计入；非平静情绪稳定一段时间后才算 1 次。
        self.stats_ignore = {"no_face", "neutral"}
        self.stats_min_confidence = 35          # 百分比，低置信度不记
        self.stats_min_stable_frames = 8        # 至少收到 8 次同一情绪状态
        self.stats_min_stable_seconds = 1.2     # 同一情绪至少稳定 1.2 秒
        self.stats_same_emotion_cooldown = 6.0  # 同一种情绪短时间内不重复刷次数

        self.stats_candidate_emotion = None
        self.stats_candidate_frames = 0
        self.stats_candidate_start_time = 0.0
        self.stats_active_emotion = None
        self.stats_last_count_time_by_emotion = defaultdict(float)

        # 聊天记录，给 WebSocket chat 使用
        self.chat_history = deque(maxlen=100)

    def now_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def _normalize_emotion(self, emotion):
        if not emotion:
            return "no_face"

        emotion = str(emotion).strip().lower()
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

        return max(0, min(100, confidence))

    def _reset_stats_candidate(self):
        self.stats_candidate_emotion = None
        self.stats_candidate_frames = 0
        self.stats_candidate_start_time = 0.0
        self.stats_active_emotion = None

    def _try_count_emotion(self, emotion, confidence, now):
        """只把稳定的非平静情绪记为一次情绪事件。

        返回值：
        - None：这次没有形成新的统计事件。
        - dict：这次形成了新的统计事件，main.py 可以据此播报/提示。

        规则：
        1. neutral 不统计，因为平静本来就会最多。
        2. no_face 不统计。
        3. 低置信度不统计。
        4. 同一种非平静情绪连续稳定一段时间后才 +1。
        5. 同一次连续情绪只记一次，不会每帧狂加。
        """
        if emotion in self.stats_ignore or confidence < self.stats_min_confidence:
            self._reset_stats_candidate()
            return None

        if emotion == self.stats_candidate_emotion:
            self.stats_candidate_frames += 1
        else:
            self.stats_candidate_emotion = emotion
            self.stats_candidate_frames = 1
            self.stats_candidate_start_time = now
            self.stats_active_emotion = None

        stable_seconds = now - self.stats_candidate_start_time
        if self.stats_candidate_frames < self.stats_min_stable_frames:
            return None
        if stable_seconds < self.stats_min_stable_seconds:
            return None

        # 当前连续情绪已经记过，就不重复刷。
        if self.stats_active_emotion == emotion:
            return None

        last_count_time = self.stats_last_count_time_by_emotion[emotion]
        if now - last_count_time < self.stats_same_emotion_cooldown:
            return None

        self.emotion_counts[emotion] += 1
        self.total_count += 1
        self.stats_active_emotion = emotion
        self.stats_last_count_time_by_emotion[emotion] = now

        return {
            "emotion": emotion,
            "emotion_cn": EMOTION_CN.get(emotion, emotion),
            "confidence": confidence,
            "count": self.emotion_counts[emotion],
            "total": self.total_count,
            "time": self.now_time(),
        }

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

            event = self._try_count_emotion(emotion, confidence, now)
            return event

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
                    "trend": "当前还没有足够的非平静情绪数据。",
                    "items": [],
                }

            sorted_items = sorted(
                self.emotion_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            main_emotion = sorted_items[0][0]
            items = []
            for emotion, count in sorted_items:
                percent = int(round(count * 100 / total))
                items.append({
                    "name": EMOTION_CN.get(emotion, emotion),
                    "key": emotion,
                    "count": count,
                    "percent": percent,
                    "color": EMOTION_COLOR.get(emotion, "#4f7cff"),
                })

            return {
                "total": total,
                "main_emotion": EMOTION_CN.get(main_emotion, main_emotion),
                "trend": self._make_trend_text(main_emotion),
                "items": items,
            }

    def _make_trend_text(self, main_emotion):
        if main_emotion == "happy":
            return "最近开心情绪出现较多，整体互动状态较积极。"
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
        return "最近出现了非平静情绪波动，建议继续观察。"

    def get_alerts(self):
        with self.lock:
            status = self.current_emotion
            confidence = self.confidence

        if status == "sad":
            return [{
                "level": "warning",
                "tag": "需要关注",
                "title": "检测到难过情绪",
                "content": "建议使用温和语气陪伴用户，可以播放舒缓音乐或提醒用户休息。",
            }]

        if status == "angry":
            return [{
                "level": "danger",
                "tag": "情绪波动",
                "title": "检测到生气情绪",
                "content": "建议降低机器人音量，减少主动打扰，等待用户情绪稳定。",
            }]

        if status == "fear":
            return [{
                "level": "warning",
                "tag": "需要安抚",
                "title": "检测到害怕情绪",
                "content": "建议机器人使用安抚性话术，并保持陪伴状态。",
            }]

        if status == "disgust":
            return [{
                "level": "warning",
                "tag": "需要调整",
                "title": "检测到厌恶情绪",
                "content": "建议检查当前声音、画面、环境或对话内容，减少可能引起用户反感的刺激。",
            }]

        if status == "no_face":
            return [{
                "level": "normal",
                "tag": "等待检测",
                "title": "暂未检测到人脸",
                "content": "请确认摄像头朝向、距离和光照环境是否正常。",
            }]

        return [{
            "level": "normal",
            "tag": "状态良好",
            "title": "当前情绪状态较稳定",
            "content": f"当前情绪识别置信度为 {confidence}%，可以保持正常交流。",
        }]

    def reset_stats(self):
        with self.lock:
            self.emotion_counts.clear()
            self.total_count = 0
            self.stats_last_count_time_by_emotion.clear()
            self._reset_stats_candidate()

    def clear_chat(self):
        with self.lock:
            self.chat_history.clear()


robot_state = RobotState()
