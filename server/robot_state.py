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

# 第 3 页统计图、第 4 页建议都按这些时间段汇总。
# 这里按“稳定情绪事件发生时刻”归类，不统计 neutral/no_face。
TIME_PERIODS = [
    {"key": "night", "name": "凌晨", "range": "00:00-06:00", "start": 0, "end": 6},
    {"key": "morning", "name": "上午", "range": "06:00-12:00", "start": 6, "end": 12},
    {"key": "noon", "name": "中午", "range": "12:00-14:00", "start": 12, "end": 14},
    {"key": "afternoon", "name": "下午", "range": "14:00-18:00", "start": 14, "end": 18},
    {"key": "evening", "name": "晚上", "range": "18:00-24:00", "start": 18, "end": 24},
]


class RobotState:
    """
    板端统一状态中心。
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
        self.emotion_events = deque(maxlen=500)

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

    def now_date(self):
        return datetime.now().strftime("%Y-%m-%d")

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

    def _get_period_info(self, dt=None):
        dt = dt or datetime.now()
        hour = dt.hour

        for item in TIME_PERIODS:
            if item["start"] <= hour < item["end"]:
                return item

        return TIME_PERIODS[-1]

    def _reset_stats_candidate(self):
        self.stats_candidate_emotion = None
        self.stats_candidate_frames = 0
        self.stats_candidate_start_time = 0.0
        self.stats_active_emotion = None

    def _record_emotion_event(self, emotion, confidence, now_ts):
        """记录一次已经通过稳定判断的情绪事件。"""
        dt = datetime.now()
        period = self._get_period_info(dt)

        self.emotion_events.append({
            "emotion": emotion,
            "emotion_cn": EMOTION_CN.get(emotion, emotion),
            "confidence": confidence,
            "timestamp": now_ts,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "period_key": period["key"],
            "period_name": period["name"],
            "period_range": period["range"],
            "color": EMOTION_COLOR.get(emotion, "#4f7cff"),
        })

    def _try_count_emotion(self, emotion, confidence, now):
        """
        只把稳定的非平静情绪记为一次情绪事件。
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
        self._record_emotion_event(emotion, confidence, now)

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
        emotion: happy / angry / neutral / sad / surprise / no_face
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

    def _make_items_from_counts(self, counts, total):
        if total <= 0:
            return []

        sorted_items = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

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

        return items

    def _build_period_stats_locked(self):
        period_map = {}
        for period in TIME_PERIODS:
            period_map[period["key"]] = {
                "key": period["key"],
                "name": period["name"],
                "range": period["range"],
                "total": 0,
                "counts": defaultdict(int),
            }

        for event in self.emotion_events:
            period_key = event.get("period_key")
            emotion = event.get("emotion")
            if period_key not in period_map:
                continue
            if emotion in self.stats_ignore:
                continue

            period_map[period_key]["total"] += 1
            period_map[period_key]["counts"][emotion] += 1

        result = []
        for period in TIME_PERIODS:
            item = period_map[period["key"]]
            total = item["total"]
            emotion_items = self._make_items_from_counts(item["counts"], total)
            main_emotion_key = emotion_items[0]["key"] if emotion_items else ""
            main_emotion = emotion_items[0]["name"] if emotion_items else "暂无"

            result.append({
                "key": item["key"],
                "name": item["name"],
                "range": item["range"],
                "total": total,
                "main_emotion": main_emotion,
                "mainEmotion": main_emotion,
                "main_emotion_key": main_emotion_key,
                "mainEmotionKey": main_emotion_key,
                "items": emotion_items,
                "segments": emotion_items,
            })

        return result

    def get_stats(self):
        with self.lock:
            total = self.total_count
            period_stats = self._build_period_stats_locked()

            if total == 0:
                return {
                    "total": 0,
                    "main_emotion": "暂无数据",
                    "mainEmotion": "暂无数据",
                    "trend": "当前还没有足够的非平静情绪数据。",
                    "items": [],
                    "period_stats": period_stats,
                    "periodStats": period_stats,
                }

            sorted_items = sorted(
                self.emotion_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            main_emotion = sorted_items[0][0]
            items = self._make_items_from_counts(self.emotion_counts, total)

            return {
                "total": total,
                "main_emotion": EMOTION_CN.get(main_emotion, main_emotion),
                "mainEmotion": EMOTION_CN.get(main_emotion, main_emotion),
                "trend": self._make_trend_text(main_emotion),
                "items": items,
                "period_stats": period_stats,
                "periodStats": period_stats,
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

    def _make_period_suggestion(self, period_name, period_range, main_emotion, total):
        """
        按时间段生成面向用户本人的建议。
        """
        emotion_cn = EMOTION_CN.get(main_emotion, main_emotion)
        prefix = f"{period_name}（{period_range}）共记录 {total} 次明显情绪，"

        if main_emotion == "happy":
            return {
                "level": "normal",
                "tag": "积极时段",
                "title": f"{period_name}开心情绪比较多",
                "content": f"{prefix}其中开心情绪比较多。您可以把开心的事情分享给家人或朋友，也可以和我聊聊，把这份好心情记录下来。",
            }

        if main_emotion == "sad":
            return {
                "level": "warning",
                "tag": "需要陪伴",
                "title": f"{period_name}难过情绪比较多",
                "content": f"{prefix}其中难过情绪比较多。您可以先放慢节奏，听一会儿舒缓的音乐，或者把心里的事慢慢说出来，我会陪您听。",
            }

        if main_emotion == "angry":
            return {
                "level": "danger",
                "tag": "情绪波动",
                "title": f"{period_name}生气情绪比较多",
                "content": f"{prefix}其中生气情绪比较多。您可以先暂停争论或复杂任务，做几次深呼吸，等情绪缓和后再处理事情，也可以把让您不舒服的原因说给我听。",
            }

        if main_emotion == "surprise":
            return {
                "level": "tip",
                "tag": "值得记录",
                "title": f"{period_name}惊讶情绪比较多",
                "content": f"{prefix}其中惊讶情绪比较多。您可以回想一下这个时间段发生了什么新鲜或意外的事情，把它记录下来，或者和我聊聊当时的感受。",
            }

        if main_emotion == "fear":
            return {
                "level": "warning",
                "tag": "需要安心",
                "title": f"{period_name}害怕情绪比较多",
                "content": f"{prefix}其中害怕情绪比较多。您可以先待在更熟悉、更安静的环境里，打开灯或联系信任的人，也可以告诉我让您不安的事情。",
            }

        if main_emotion == "disgust":
            return {
                "level": "warning",
                "tag": "需要调整",
                "title": f"{period_name}厌恶情绪比较多",
                "content": f"{prefix}其中厌恶情绪比较多。您可以先远离让自己不舒服的内容或环境，换一个更轻松的活动，也可以和我说说哪里让您感到反感。",
            }

        return {
            "level": "normal",
            "tag": "建议观察",
            "title": f"{period_name}{emotion_cn}情绪比较多",
            "content": f"{prefix}主要为{emotion_cn}情绪。您可以留意这个时间段通常发生了什么，也可以把当时的感受记录下来，方便之后更了解自己的情绪变化。",
        }

    def get_alerts(self):
        """第 4 页：按时间段给出建议，不再按当前表情直接给建议。"""
        with self.lock:
            period_stats = self._build_period_stats_locked()

        alerts = []
        for period in period_stats:
            total = int(period.get("total", 0))
            if total <= 0:
                continue

            main_key = period.get("main_emotion_key") or period.get("mainEmotionKey")
            if not main_key:
                continue

            suggestion = self._make_period_suggestion(
                period_name=period.get("name", "当前时间段"),
                period_range=period.get("range", ""),
                main_emotion=main_key,
                total=total,
            )

            suggestion.update({
                "periodName": period.get("name", ""),
                "period_name": period.get("name", ""),
                "periodRange": period.get("range", ""),
                "period_range": period.get("range", ""),
                "total": total,
                "mainEmotion": EMOTION_CN.get(main_key, main_key),
                "main_emotion": EMOTION_CN.get(main_key, main_key),
                "mainEmotionKey": main_key,
                "main_emotion_key": main_key,
                "color": EMOTION_COLOR.get(main_key, "#4f7cff"),
            })
            alerts.append(suggestion)

        if alerts:
            return alerts

        return [{
            "level": "normal",
            "tag": "数据不足",
            "title": "暂无分时段关怀建议",
            "content": "当前还没有形成稳定的非平静情绪统计。等待板端识别到开心、生气、难过、惊讶等明显情绪后，这里会按时间段给出建议。",
            "periodName": "全时段",
            "period_name": "全时段",
            "periodRange": "00:00-24:00",
            "period_range": "00:00-24:00",
            "total": 0,
            "mainEmotion": "暂无",
            "main_emotion": "暂无",
            "mainEmotionKey": "unknown",
            "main_emotion_key": "unknown",
            "color": "#8A8F99",
        }]

    def reset_stats(self):
        with self.lock:
            self.emotion_counts.clear()
            self.total_count = 0
            self.emotion_events.clear()
            self.stats_last_count_time_by_emotion.clear()
            self._reset_stats_candidate()

    def clear_chat(self):
        with self.lock:
            self.chat_history.clear()


robot_state = RobotState()
